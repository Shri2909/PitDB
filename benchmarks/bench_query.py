"""End-to-end query benchmarks for the full-scan and pushdown-query paths.

The benchmark deliberately times complete SQL execution, including parsing,
chunk decompression, and exact row filtering. The full-scan baseline and the
pushdown-indexed executor receive the same SQL and their exact outputs are
compared before any speedup is reported.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np
import pandas as pd

from benchmarks.reporting.validation import assert_equivalent_and_report
from benchmarks.stats import (
    confidence_interval_95,
    intervals_overlap,
    trimmed_mean_and_std,
)
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor, QueryResult
from src.store.chunk_store import ChunkStore

# Drops the slowest 15% of latency trials before computing statistics -- see
# trimmed_mean_and_std's docstring and benchmarks/bench_bitemporal.py's own
# use of the same constant. Without this, a single slow trial (a GC pause, an
# OS scheduling hiccup) can blow out a reported confidence interval far wider
# than every other measurement's -- observed directly in this file's own
# selectivity sweep, where the 1% point's untrimmed std exceeded its mean.
LATENCY_TRIM_FRACTION = 0.15

try:
    from benchmarks.config import (
        BENCHMARK_QUERIES,
        NUM_TRIALS,
        SELECTIVITY_SWEEP_TARGETS,
        WARMUP_RUNS,
    )
except ModuleNotFoundError:
    # Keep this module independently usable until config.py is introduced.
    NUM_TRIALS = 15
    WARMUP_RUNS = 3
    BENCHMARK_QUERIES = {
        "Q1": ("SELECT * FROM data WHERE symbol = 'AAPL' AND date = '2024-03-01'"),
        "Q2": (
            "SELECT * FROM data WHERE symbol = 'AAPL' "
            "AND date >= '2024-03-01' AND date < '2024-04-01'"
        ),
        "Q3": (
            "SELECT * FROM data WHERE symbol IN ('AAPL', 'MSFT') "
            "AND date >= '2024-03-01' AND date < '2024-03-08'"
        ),
        "Q4": (
            "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 200 "
            "AND date >= '2024-01-01'"
        ),
        "Q5": "SELECT * FROM data WHERE date >= '2024-01-01'",
    }
    SELECTIVITY_SWEEP_TARGETS = [1.0, 0.5, 0.1, 0.01, 0.001]


class _QueryEngine(Protocol):
    def execute(self, sql: str) -> QueryResult: ...


def _validate_benchmark_configuration() -> None:
    if not isinstance(NUM_TRIALS, int) or NUM_TRIALS <= 0:
        raise ValueError("NUM_TRIALS must be a positive integer")
    if not isinstance(WARMUP_RUNS, int) or WARMUP_RUNS < 0:
        raise ValueError("WARMUP_RUNS must be a non-negative integer")
    if not isinstance(BENCHMARK_QUERIES, dict) or not BENCHMARK_QUERIES:
        raise ValueError("BENCHMARK_QUERIES must be a non-empty dictionary")


def _time_query(engine: _QueryEngine, sql: str) -> tuple[np.ndarray, QueryResult]:
    """Warm an engine, then measure complete query wall-clock latency."""

    for _ in range(WARMUP_RUNS):
        engine.execute(sql)

    times = np.empty(NUM_TRIALS, dtype=np.float64)
    result: QueryResult | None = None
    for trial in range(NUM_TRIALS):
        started = time.perf_counter()
        result = engine.execute(sql)
        times[trial] = time.perf_counter() - started

    if result is None:  # Protected by configuration validation.
        raise AssertionError("query timing completed without a result")
    return times, result


def _assert_equivalent(
    full_scan_result: QueryResult,
    pushdown_result: QueryResult,
    *,
    query_name: str,
) -> None:
    """Require the same ordered rows, values, columns, dtypes, and duplicates."""

    if full_scan_result.row_count != pushdown_result.row_count:
        raise AssertionError(
            f"{query_name}: full scan returned {full_scan_result.row_count} "
            f"rows but pushdown query returned {pushdown_result.row_count}"
        )
    try:
        pd.testing.assert_frame_equal(
            full_scan_result.data.reset_index(drop=True),
            pushdown_result.data.reset_index(drop=True),
            check_exact=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{query_name}: pushdown-query results differ from the full scan"
        ) from exc


def _latency_metrics(times: np.ndarray, result: QueryResult) -> dict[str, float | int]:
    mean, std, kept_n = trimmed_mean_and_std(times, LATENCY_TRIM_FRACTION)
    ci_low, ci_high = confidence_interval_95(mean, std, kept_n)
    return {
        "mean_latency_seconds": mean,
        "std_latency_seconds": std,
        "ci95_low_seconds": ci_low,
        "ci95_high_seconds": ci_high,
        "row_count": result.row_count,
    }


def annotate_pairwise_significance(
    fixed_query_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flag which consecutive fixed queries' pushdown-query latencies are
    noise-plausible.

    Two means whose 95% CIs overlap are not distinguishable from noise at
    that confidence level -- reporting only the means (as the original
    summary did) invites reading a real effect into differences like Q1 vs Q2
    (1.200ms vs 1.174ms in the pre-fix baseline) that may just be OS
    scheduling jitter or GC pauses.
    """

    names = list(fixed_query_results.keys())
    comparisons: list[dict[str, Any]] = []
    for first, second in zip(names, names[1:]):
        a = fixed_query_results[first]["pushdown_query"]
        b = fixed_query_results[second]["pushdown_query"]
        overlap = intervals_overlap(
            a["ci95_low_seconds"],
            a["ci95_high_seconds"],
            b["ci95_low_seconds"],
            b["ci95_high_seconds"],
        )
        comparisons.append(
            {
                "query_a": first,
                "query_b": second,
                "a_mean_ms": a["mean_latency_seconds"] * 1000.0,
                "b_mean_ms": b["mean_latency_seconds"] * 1000.0,
                "ci95_overlap": overlap,
                "distinguishable_from_noise": not overlap,
            }
        )
    return comparisons


def _speedup(full_scan_times: np.ndarray, pushdown_times: np.ndarray) -> float:
    # Trimmed, matching _latency_metrics -- otherwise the reported speedup
    # would silently disagree with the mean_latency_seconds figures it's
    # supposedly derived from.
    full_scan_mean, _, _ = trimmed_mean_and_std(full_scan_times, LATENCY_TRIM_FRACTION)
    pushdown_mean, _, _ = trimmed_mean_and_std(pushdown_times, LATENCY_TRIM_FRACTION)
    return full_scan_mean / pushdown_mean if pushdown_mean > 0.0 else math.inf


def _benchmark_pair(
    full_scan_engine: FullScanQueryEngine,
    pushdown_engine: QueryExecutor,
    sql: str,
    *,
    query_name: str,
    quiet: bool = False,
) -> tuple[dict[str, float | int], dict[str, float | int], float, QueryResult]:
    """Time both paths under identical controls and prove equivalence.

    ``quiet=True`` suppresses only the correctness PASS-path print (see
    ``assert_equivalent_and_report``) for callers that will re-render the
    outcome themselves elsewhere -- the assertion and its exception
    behavior on failure are unaffected either way.
    """

    full_scan_times, full_scan_result = _time_query(full_scan_engine, sql)
    pushdown_times, pushdown_result = _time_query(pushdown_engine, sql)
    # Printed PASS/FAIL block around the same _assert_equivalent call this
    # project has always made before computing any speedup -- see
    # benchmarks/reporting/validation.py. A failure still raises and aborts
    # the run exactly as before; only the pass path newly becomes visible.
    assert_equivalent_and_report(
        lambda: _assert_equivalent(
            full_scan_result, pushdown_result, query_name=query_name
        ),
        name=query_name,
        full_scan_row_count=full_scan_result.row_count,
        pushdown_row_count=pushdown_result.row_count,
        quiet=quiet,
    )

    full_scan_metrics = _latency_metrics(full_scan_times, full_scan_result)
    pushdown_metrics = _latency_metrics(pushdown_times, pushdown_result)
    pushdown_metrics["chunks_skipped_ratio"] = (
        pushdown_result.pushdown.chunks_skipped_ratio
    )
    # Exact integer counts already computed on pushdown_result.pushdown --
    # exposed here (not derived by rounding chunks_skipped_ratio) so callers
    # needing "chunks scanned"/"chunks pruned" as whole numbers, not just a
    # percentage, don't have to reconstruct them lossily from the ratio.
    pushdown_metrics["candidate_chunks"] = pushdown_result.pushdown.candidate_count
    pushdown_metrics["total_chunks"] = pushdown_result.pushdown.total_chunks
    return (
        full_scan_metrics,
        pushdown_metrics,
        _speedup(full_scan_times, pushdown_times),
        pushdown_result,
    )


def run_query_benchmarks(
    store: ChunkStore, queries: dict[str, str] | None = None, *, quiet: bool = False
) -> dict[str, dict[str, Any]]:
    """Benchmark a fixed query workload using full-scan and pushdown paths.

    Defaults to the Q1-Q5 Dataset A workload; pass ``queries`` to run the
    same full-scan/pushdown/speedup/equivalence machinery against a
    different store, such as Dataset B's intraday queries.

    ``quiet=True`` suppresses the correctness PASS-path print for every
    query in this call (see ``_benchmark_pair``) -- for a caller that will
    render each query's correctness outcome itself, embedded in a per-query
    result block, rather than as a standalone line at computation time.
    """

    if not isinstance(store, ChunkStore):
        raise TypeError("store must be a ChunkStore")
    _validate_benchmark_configuration()
    if queries is None:
        queries = BENCHMARK_QUERIES

    full_scan_engine = FullScanQueryEngine(store)
    pushdown_engine = QueryExecutor(store)
    results: dict[str, dict[str, Any]] = {}

    for query_name, sql in queries.items():
        if not isinstance(query_name, str) or not isinstance(sql, str):
            raise ValueError("benchmark query names and SQL values must be strings")
        full_scan_metrics, pushdown_metrics, speedup, pushdown_result = _benchmark_pair(
            full_scan_engine,
            pushdown_engine,
            sql,
            query_name=query_name,
            quiet=quiet,
        )
        # Selectivity is measured from the actual pushdown-query result,
        # never inferred from the query text or its expected workload class.
        selectivity = (
            pushdown_result.row_count / store.total_rows if store.total_rows else 0.0
        )
        results[query_name] = {
            "query": sql,
            "full_scan": full_scan_metrics,
            "pushdown_query": pushdown_metrics,
            "speedup": speedup,
            "selectivity": selectivity,
            "num_trials": NUM_TRIALS,
            "warmup_runs": WARMUP_RUNS,
        }

    return results


def _validate_sweep_targets() -> list[float]:
    targets: list[float] = []
    for target in SELECTIVITY_SWEEP_TARGETS:
        if isinstance(target, bool):
            raise ValueError("selectivity targets must be numeric fractions")
        try:
            numeric = float(target)
        except (TypeError, ValueError) as exc:
            raise ValueError("selectivity targets must be numeric fractions") from exc
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError("selectivity targets must lie between 0 and 1")
        targets.append(numeric)
    return targets


def _sweep_query(store: ChunkStore, target: float) -> str:
    """Construct an approximate trailing-time-range query for one target.

    Time density need not be uniform, so interpolation only proposes a
    threshold.  The benchmark records the achieved row selectivity afterward.
    """

    labels = store.labels
    if not labels:
        return "SELECT * FROM data"

    time_start = min(label.time_start for label in labels)
    time_end = max(label.time_end for label in labels)
    if target <= 0.0:
        threshold = time_end + timedelta(microseconds=1)
    elif target >= 1.0 or time_start == time_end:
        threshold = time_start
    else:
        threshold = time_start + (time_end - time_start) * (1.0 - target)

    if not isinstance(threshold, datetime):
        # pandas.Timestamp is a datetime subclass, but keep failures from a
        # custom label type descriptive rather than emitting malformed SQL.
        raise TypeError("chunk label timestamps must be datetime instances")
    literal = threshold.isoformat(sep=" ")
    return f"SELECT * FROM data WHERE timestamp >= '{literal}'"


def run_selectivity_sweep(store: ChunkStore, *, quiet: bool = False) -> dict[str, Any]:
    """Benchmark approximate sweep queries and report achieved selectivity.

    ``quiet=True`` suppresses the correctness PASS-path print for every
    sweep point, matching ``run_query_benchmarks``'s own ``quiet`` parameter.
    """

    if not isinstance(store, ChunkStore):
        raise TypeError("store must be a ChunkStore")
    _validate_benchmark_configuration()
    targets = _validate_sweep_targets()

    full_scan_engine = FullScanQueryEngine(store)
    pushdown_engine = QueryExecutor(store)
    points: list[dict[str, Any]] = []

    for target in targets:
        sql = _sweep_query(store, target)

        # This untimed probe is intentional: target values guide query
        # construction, while charts must use measured result selectivity.
        probe = pushdown_engine.execute(sql)
        actual_selectivity = (
            probe.row_count / store.total_rows if store.total_rows else 0.0
        )

        point_name = f"selectivity target {target:g}"
        full_scan_metrics, pushdown_metrics, speedup, timed_result = _benchmark_pair(
            full_scan_engine,
            pushdown_engine,
            sql,
            query_name=point_name,
            quiet=quiet,
        )
        if timed_result.row_count != probe.row_count:
            raise AssertionError(
                f"{point_name}: query result changed between probe and timing"
            )

        points.append(
            {
                "target_selectivity": target,
                "actual_selectivity": actual_selectivity,
                "query": sql,
                "full_scan": full_scan_metrics,
                "pushdown_query": pushdown_metrics,
                "speedup": speedup,
                "chunks_skipped_ratio": (timed_result.pushdown.chunks_skipped_ratio),
            }
        )

    return {
        # Charts should read this ordered point collection and use
        # actual_selectivity, not target_selectivity, for their x-axis.
        "points": points,
        "num_trials": NUM_TRIALS,
        "warmup_runs": WARMUP_RUNS,
    }


def build_intraday_queries(store: ChunkStore) -> dict[str, str]:
    """Construct representative queries for Dataset B.

    Dataset B's date range is a trailing window relative to whenever the
    data was fetched (see data/download_data.py), unlike Dataset A's fixed
    2023-2024 history -- fixed date literals like Q1-Q5 use would silently
    stop matching anything once enough time passes. Thresholds here are
    derived from the store's own labels instead, so the queries stay valid
    regardless of when the merged CSV was generated.
    """

    labels = [label for label in store.labels if label.symbol == "AAPL"]
    if not labels:
        raise ValueError("Dataset B store has no AAPL chunks to query")

    labels.sort(key=lambda label: label.time_start)
    first_chunk = labels[0]
    all_labels = store.labels
    overall_start = min(label.time_start for label in all_labels)
    overall_end = max(label.time_end for label in all_labels)
    price_midpoint = (
        min(label.min_price for label in labels)
        + max(label.max_price for label in labels)
    ) / 2.0

    narrow_start = first_chunk.time_start.isoformat(sep=" ")
    narrow_end = first_chunk.time_end.isoformat(sep=" ")

    return {
        "narrow_symbol_hour": (
            f"SELECT * FROM data WHERE symbol = 'AAPL' "
            f"AND date >= '{narrow_start}' AND date <= '{narrow_end}'"
        ),
        "price_threshold": (
            f"SELECT * FROM data WHERE symbol = 'AAPL' AND price > {price_midpoint:.4f}"
        ),
        "broad_full_range": (
            f"SELECT * FROM data WHERE date >= '{overall_start.isoformat(sep=' ')}' "
            f"AND date <= '{overall_end.isoformat(sep=' ')}'"
        ),
    }


def run_price_pushdown_demonstration(store: ChunkStore) -> dict[str, Any]:
    """Isolate the price label's incremental pruning effect.

    A query filtering on symbol='AAPL' AND price>150 AND date>=2024-01-01 was
    found to add zero incremental pruning beyond its date range: AAPL's
    actual 2024 close price stayed in [163.36, 257.38] the whole year, so
    every 2024 chunk trivially satisfies price>150 and price-label pushdown
    is never really exercised, even though it is present in the SQL text.
    Checked directly against this store's own AAPL 2024 chunk labels, a
    threshold of 200 splits AAPL's 12 monthly 2024 chunks into 5 chunks whose
    max price never reaches 200 (prunable) and 7 that do (candidates) -- a
    verified, real split of this store's actual data, not an assumed one.
    """

    pushdown_engine = QueryExecutor(store)
    date_only_sql = "SELECT * FROM data WHERE symbol = 'AAPL' AND date >= '2024-01-01'"
    date_and_price_sql = (
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 200 "
        "AND date >= '2024-01-01'"
    )

    date_only_result = pushdown_engine.execute(date_only_sql)
    date_and_price_result = pushdown_engine.execute(date_and_price_sql)

    date_only_candidates = date_only_result.pushdown.candidate_count
    date_and_price_candidates = date_and_price_result.pushdown.candidate_count
    return {
        "date_only": {
            "query": date_only_sql,
            "candidate_chunks": date_only_candidates,
            "total_chunks": date_only_result.pushdown.total_chunks,
            "chunks_skipped_ratio": date_only_result.pushdown.chunks_skipped_ratio,
            "row_count": date_only_result.row_count,
        },
        "date_and_price": {
            "query": date_and_price_sql,
            "candidate_chunks": date_and_price_candidates,
            "total_chunks": date_and_price_result.pushdown.total_chunks,
            "chunks_skipped_ratio": (
                date_and_price_result.pushdown.chunks_skipped_ratio
            ),
            "row_count": date_and_price_result.row_count,
        },
        "price_clause_incremental_chunks_pruned": (
            date_only_candidates - date_and_price_candidates
        ),
        "price_clause_pruned_any_chunks": (
            date_and_price_candidates < date_only_candidates
        ),
    }


__all__ = [
    "build_intraday_queries",
    "run_price_pushdown_demonstration",
    "run_query_benchmarks",
    "run_selectivity_sweep",
]
