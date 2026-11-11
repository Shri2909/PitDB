"""End-to-end query benchmarks for the full-scan and pushdown-query paths."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.config import (
    BENCHMARK_QUERIES,
    NUM_TRIALS,
    SELECTIVITY_SWEEP_TARGETS,
    WARMUP_RUNS,
)
from benchmarks.reporting.validation import assert_equivalent_and_report
from benchmarks.stats import (
    intervals_overlap,
    latency_summary,
    time_interleaved,
)
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor, QueryResult
from src.store.chunk_store import ChunkStore


def _validate_benchmark_configuration() -> None:
    if not isinstance(NUM_TRIALS, int) or NUM_TRIALS <= 0:
        raise ValueError("NUM_TRIALS must be a positive integer")
    if not isinstance(WARMUP_RUNS, int) or WARMUP_RUNS < 0:
        raise ValueError("WARMUP_RUNS must be a non-negative integer")
    if not isinstance(BENCHMARK_QUERIES, dict) or not BENCHMARK_QUERIES:
        raise ValueError("BENCHMARK_QUERIES must be a non-empty dictionary")


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


def _latency_metrics(times: np.ndarray, result: QueryResult) -> dict[str, Any]:
    return {**latency_summary(times), "row_count": result.row_count}


def annotate_pairwise_significance(
    fixed_query_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Describe adjacent-query CI overlap (descriptive, not a significance test)."""

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
                "inference": "none; interval overlap is descriptive",
            }
        )
    return comparisons


def _speedup(full_scan_times: np.ndarray, pushdown_times: np.ndarray) -> float:
    full_scan_mean = float(np.mean(full_scan_times))
    pushdown_mean = float(np.mean(pushdown_times))
    return full_scan_mean / pushdown_mean if pushdown_mean > 0.0 else math.inf


def _benchmark_pair(
    full_scan_engine: FullScanQueryEngine,
    pushdown_engine: QueryExecutor,
    sql: str,
    *,
    query_name: str,
    quiet: bool = False,
    num_trials: int | None = None,
    warmup: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], float, QueryResult]:
    """Time both paths under identical controls and prove equivalence."""

    vectorized_engine = QueryExecutor(pushdown_engine.store, enable_pruning=False)
    samples, results, acquisition_order = time_interleaved(
        {
            "full_scan": lambda: full_scan_engine.execute(sql),
            "vectorized_full_scan": lambda: vectorized_engine.execute(sql),
            "pushdown_query": lambda: pushdown_engine.execute(sql),
        },
        NUM_TRIALS if num_trials is None else num_trials,
        WARMUP_RUNS if warmup is None else warmup,
    )
    full_scan_times, full_scan_result = samples["full_scan"], results["full_scan"]
    pushdown_times, pushdown_result = (
        samples["pushdown_query"],
        results["pushdown_query"],
    )
    _assert_equivalent(
        full_scan_result,
        results["vectorized_full_scan"],
        query_name=f"{query_name} vectorized full scan",
    )
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
    pushdown_metrics["vectorized_full_scan"] = _latency_metrics(
        samples["vectorized_full_scan"], results["vectorized_full_scan"]
    )
    pushdown_metrics["vectorized_full_scan_speedup"] = _speedup(
        samples["vectorized_full_scan"], pushdown_times
    )
    pushdown_metrics["acquisition_order"] = acquisition_order
    pushdown_metrics["requested_trials"] = (
        NUM_TRIALS if num_trials is None else num_trials
    )
    pushdown_metrics["timing_method"] = (
        "interleaved paired rounds; trials rounded up to whole engine-"
        "permutation cycles so position/precedence are balanced; gc "
        "collected then disabled around every timed call; warm single "
        "process"
    )
    pushdown_metrics["chunks_skipped_ratio"] = (
        pushdown_result.pushdown.chunks_skipped_ratio
    )
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
    """Benchmark a fixed workload (default Q1-Q5) on full-scan and pushdown paths.

    ``quiet=True`` suppresses the per-query correctness PASS print.
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
        # Measured from the actual result, never inferred from the query text.
        selectivity = (
            pushdown_result.row_count / store.total_rows if store.total_rows else 0.0
        )
        results[query_name] = {
            "query": sql,
            "full_scan": full_scan_metrics,
            "pushdown_query": pushdown_metrics,
            "speedup": speedup,
            "selectivity": selectivity,
            "num_trials": pushdown_metrics["sample_count"],
            "requested_trials": NUM_TRIALS,
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
    """Build an approximate trailing-time-range query for one target."""

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
        raise TypeError("chunk label timestamps must be datetime instances")
    literal = threshold.isoformat(sep=" ")
    return f"SELECT * FROM data WHERE timestamp >= '{literal}'"


def run_selectivity_sweep(store: ChunkStore, *, quiet: bool = False) -> dict[str, Any]:
    """Benchmark approximate sweep queries and report achieved selectivity."""
    if not isinstance(store, ChunkStore):
        raise TypeError("store must be a ChunkStore")
    _validate_benchmark_configuration()
    targets = _validate_sweep_targets()

    full_scan_engine = FullScanQueryEngine(store)
    pushdown_engine = QueryExecutor(store)
    points: list[dict[str, Any]] = []

    for target in targets:
        sql = _sweep_query(store, target)

        # Untimed probe: charts need measured selectivity, not the target.
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
                "row_count": timed_result.row_count,
                "chunks_skipped_ratio": (timed_result.pushdown.chunks_skipped_ratio),
            }
        )

    return {
        "points": points,
        "num_trials": points[0]["full_scan"]["sample_count"] if points else 0,
        "requested_trials": NUM_TRIALS,
        "warmup_runs": WARMUP_RUNS,
    }


def build_intraday_queries(store: ChunkStore) -> dict[str, str]:
    """Build Dataset B queries from the store's own labels (its window floats)."""

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
    """Isolate the price label's incremental pruning effect."""
    pushdown_engine = QueryExecutor(store)
    date_only_sql = "SELECT * FROM data WHERE symbol = 'AAPL' AND date >= '2024-01-01'"
    date_and_price_sql = (
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 200 "
        "AND date >= '2024-01-01'"
    )

    date_only_result = pushdown_engine.execute(date_only_sql)
    date_and_price_result = pushdown_engine.execute(date_and_price_sql)

    aapl_2024 = [
        label
        for label in store.labels
        if label.symbol == "AAPL" and label.time_start.year == 2024
    ]
    aapl_2024_range = (
        [
            min(label.min_price for label in aapl_2024),
            max(label.max_price for label in aapl_2024),
        ]
        if aapl_2024
        else None
    )

    date_only_candidates = date_only_result.pushdown.candidate_count
    date_and_price_candidates = date_and_price_result.pushdown.candidate_count
    return {
        "aapl_2024_close_range": aapl_2024_range,
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
