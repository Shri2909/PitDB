"""Bitemporal correction-volume sweep.

Measures how the cost of transaction-time versioning grows as the fraction
of corrected rows increases relative to base data: ingest_correction
throughput, compact_corrections() throughput, isolated AS OF
pushdown-pruning cost, and end-to-end AS OF query latency (pushdown vs.
full-scan) -- each measured **both before and after** compact_corrections(),
so the actual effect of compaction is always visible directly in this
benchmark's own output, never requiring an ad hoc script to see it.

Mirrors benchmarks/bench_scale.py's conventions -- ddof=1 sample std,
stats.py's confidence_interval_95 normal approximation, and the same
full-scan/pushdown correctness gate (_assert_equivalent) every other
benchmark in this suite uses before reporting any speedup number. Also
mirrors bench_query.py's annotate_pairwise_significance: every pre/post
comparison is flagged as statistically distinguishable or not via
stats.py's intervals_overlap, rather than letting point estimates alone
imply an effect that might just be noise.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_query import _assert_equivalent
from benchmarks.bench_scale import _time_calls
from benchmarks.config import NUM_TRIALS as DEFAULT_NUM_TRIALS
from benchmarks.report import format_ms, format_pct, format_x, render_table
from benchmarks.stats import (
    confidence_interval_95,
    intervals_overlap,
    trimmed_mean_and_std,
)
from benchmarks.synthetic_data import generate_synthetic_ohlcv
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.query.predicate import PredicateNode
from src.query.pushdown import evaluate_pushdown
from src.query.sql_parser import parse_where_clause
from src.store.chunk_boundary import MonthlyBoundary
from src.store.chunk_store import ChunkStore

NUM_SYMBOLS = 20
ROWS_PER_SYMBOL = 501  # matches bench_scale.py's Dataset-A-shaped row count
CORRECTION_RATES = [0.0, 0.01, 0.05, 0.20, 0.50]
QUERY_NUM_TRIALS = DEFAULT_NUM_TRIALS  # matches benchmarks/config.py's convention
QUERY_WARMUP = 2
# Drops the slowest 15% of query-latency trials before reporting mean/std/CI
# -- see trimmed_mean_and_std's docstring. Decompression-heavy queries in
# this benchmark are far more exposed to one-off OS/GC hiccups (multi-hundred
# ms per trial) than the microsecond-scale isolated pushdown-eval loop below,
# which is left on a plain mean.
LATENCY_TRIM_FRACTION = 0.15
PUSHDOWN_ISOLATED_TRIALS = 50


def apply_corrections(
    df: pd.DataFrame, correction_rate: float, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic fraction of rows and build correction events.

    Every selected (symbol, timestamp) row is corrected to a bumped close
    price, with a fixed transaction-time lag (one day after the dataset's
    last observed timestamp) -- every correction is uniformly "learned"
    well after every base row.
    """

    if not 0.0 <= correction_rate <= 1.0:
        raise ValueError("correction_rate must be within [0.0, 1.0]")

    num_corrections = int(round(len(df) * correction_rate))
    if num_corrections == 0:
        return []

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(df), size=num_corrections, replace=False)
    transaction_time = df["timestamp"].max() + pd.Timedelta(days=1)

    corrections: list[dict[str, Any]] = []
    for position in selected:
        row = df.iloc[position]
        values = np.array(
            [row.open, row.high, row.low, row.close * 1.05, row.volume],
            dtype=np.float64,
        )
        corrections.append(
            {
                "timestamp": row.timestamp,
                "symbol": row.symbol,
                "values": values,
                "transaction_time": transaction_time,
            }
        )
    return corrections


def measure_scenario(
    *,
    store: ChunkStore,
    predicate: PredicateNode,
    as_of_value: Any,
    sql: str,
    query_name: str,
) -> dict[str, Any]:
    """Measure isolated pushdown-eval cost and end-to-end query latency for
    one (AS OF value, SQL) scenario against the store's *current* state.

    Called twice per scenario (before and after compact_corrections()) so
    the caller can compare the two dicts this returns directly.
    """

    pushdown_times = np.empty(PUSHDOWN_ISOLATED_TRIALS, dtype=np.float64)
    for trial in range(PUSHDOWN_ISOLATED_TRIALS):
        t0 = time.perf_counter()
        evaluate_pushdown(predicate, store.labels, as_of=as_of_value, as_of_index=store)
        pushdown_times[trial] = time.perf_counter() - t0
    pushdown_result = evaluate_pushdown(
        predicate, store.labels, as_of=as_of_value, as_of_index=store
    )

    pushdown_engine = QueryExecutor(store)
    full_scan_engine = FullScanQueryEngine(store)

    pushdown_query_times, pushdown_query_result = _time_calls(
        lambda: pushdown_engine.execute(sql), QUERY_NUM_TRIALS, QUERY_WARMUP
    )
    full_scan_query_times, full_scan_query_result = _time_calls(
        lambda: full_scan_engine.execute(sql), QUERY_NUM_TRIALS, QUERY_WARMUP
    )
    # Same correctness gate every other benchmark in this suite uses: full
    # row/value/dtype equality between the naive and pushdown paths, now
    # with corrections and an AS OF cutoff in play, before any latency is
    # reported.
    _assert_equivalent(
        full_scan_query_result, pushdown_query_result, query_name=query_name
    )

    # Trimmed, not plain, mean: latency outliers here are one-directional
    # (a GC pause or OS scheduling hiccup only ever makes a trial slower),
    # and a single such trial can otherwise blow out the reported CI far
    # wider than every other measurement's -- see LATENCY_TRIM_FRACTION.
    pushdown_mean, pushdown_std, pushdown_n = trimmed_mean_and_std(
        pushdown_query_times, LATENCY_TRIM_FRACTION
    )
    ci_low, ci_high = confidence_interval_95(pushdown_mean, pushdown_std, pushdown_n)
    full_scan_mean, _, _ = trimmed_mean_and_std(
        full_scan_query_times, LATENCY_TRIM_FRACTION
    )

    return {
        "total_chunks": store.total_chunks,
        "pushdown_eval_ns_per_label": (
            float(np.mean(pushdown_times)) * 1e9 / store.total_chunks
        ),
        "chunks_skipped_ratio": pushdown_result.chunks_skipped_ratio,
        "candidate_chunks": pushdown_result.candidate_count,
        "query_pushdown_mean_ms": pushdown_mean * 1000.0,
        "query_pushdown_ci95_low_ms": ci_low * 1000.0,
        "query_pushdown_ci95_high_ms": ci_high * 1000.0,
        "query_full_scan_mean_ms": full_scan_mean * 1000.0,
        "query_speedup": full_scan_mean / pushdown_mean if pushdown_mean else 0.0,
        "row_count": pushdown_query_result.row_count,
    }


def _print_scenario(name: str, stats: dict[str, Any]) -> None:
    print(
        f"  [{name}] chunks={stats['total_chunks']:,} "
        f"skipped={format_pct(stats['chunks_skipped_ratio'])} "
        f"candidates={stats['candidate_chunks']}/{stats['total_chunks']} | "
        f"pushdown={format_ms(stats['query_pushdown_mean_ms'])} "
        f"full_scan={format_ms(stats['query_full_scan_mean_ms'])} "
        f"speedup={format_x(stats['query_speedup'])}"
    )


def _compare(
    label: str, before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Flag whether a before/after pushdown-latency delta is real or noise,
    the same way bench_query.py's annotate_pairwise_significance does."""

    overlap = intervals_overlap(
        before["query_pushdown_ci95_low_ms"],
        before["query_pushdown_ci95_high_ms"],
        after["query_pushdown_ci95_low_ms"],
        after["query_pushdown_ci95_high_ms"],
    )
    comparison = {
        "comparison": label,
        "before_mean_ms": before["query_pushdown_mean_ms"],
        "after_mean_ms": after["query_pushdown_mean_ms"],
        "ci95_overlap": overlap,
        "distinguishable_from_noise": not overlap,
    }
    verdict = (
        "DISTINGUISHABLE (a real difference)"
        if comparison["distinguishable_from_noise"]
        else "NOT distinguishable (likely measurement noise)"
    )
    print(
        f"  significance [{label}]: {format_ms(comparison['before_mean_ms'])} -> "
        f"{format_ms(comparison['after_mean_ms'])}: {verdict}"
    )
    return comparison


def run_correction_rate(*, label: str, correction_rate: float) -> dict[str, Any]:
    print(f"\n=== correction rate {label}: {correction_rate:.1%} of rows corrected ===")

    df = generate_synthetic_ohlcv(
        num_symbols=NUM_SYMBOLS, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )
    store = ChunkStore(chunk_boundary=MonthlyBoundary())
    store.ingest_dataframe(df)
    base_chunks = store.total_chunks
    base_rows = store.total_rows

    corrections = apply_corrections(df, correction_rate, seed=2)

    t0 = time.perf_counter()
    for correction in corrections:
        store.ingest_correction(
            correction["timestamp"],
            correction["symbol"],
            correction["values"],
            transaction_time=correction["transaction_time"],
        )
    correction_seconds = time.perf_counter() - t0
    correction_rows_per_second = (
        len(corrections) / correction_seconds
        if corrections and correction_seconds > 0
        else 0.0
    )
    chunks_before_compaction = store.total_chunks
    print(
        f"ingested {base_rows:,} base rows into {base_chunks:,} chunks; "
        f"{len(corrections):,} corrections into "
        f"{chunks_before_compaction - base_chunks:,} additional chunks "
        f"in {correction_seconds:.3f}s"
        + (f" ({correction_rows_per_second:,.0f} corrections/s)" if corrections else "")
    )

    where_clause = "timestamp >= '2023-01-01'"
    predicate = parse_where_clause(f"SELECT * FROM data WHERE {where_clause}")
    current_sql = f"SELECT * FROM data WHERE {where_clause}"
    as_of_value = df["timestamp"].max() if corrections else None
    as_of_sql = (
        f"SELECT * FROM data TIMESTAMP AS OF '{as_of_value.isoformat()}' "
        f"WHERE {where_clause}"
        if corrections
        else None
    )

    # Pre-compaction measurement, taken before compact_corrections() is ever
    # called, so the effect of compaction below is a real before/after
    # comparison -- not two already-compacted runs compared against each
    # other, which is not the same thing.
    current_pre = measure_scenario(
        store=store,
        predicate=predicate,
        as_of_value=None,
        sql=current_sql,
        query_name=f"bitemporal-{label}-current-pre",
    )
    _print_scenario("current (pre-compaction)", current_pre)

    as_of_pre = None
    if corrections:
        as_of_pre = measure_scenario(
            store=store,
            predicate=predicate,
            as_of_value=as_of_value,
            sql=as_of_sql,
            query_name=f"bitemporal-{label}-as_of-pre",
        )
        _print_scenario("as_of (pre-compaction)", as_of_pre)

    # Reflects the recommended production workflow: ingest_correction stays
    # cheap and never blocks (measured above), consolidation happens later
    # via this explicit, idempotent maintenance call. Timed directly, not
    # inferred -- this used to be an untimed, unprinted step.
    t0 = time.perf_counter()
    compacted = store.compact_corrections()
    compaction_seconds = time.perf_counter() - t0
    if compacted:
        print(
            f"compact_corrections(): {chunks_before_compaction:,} -> "
            f"{store.total_chunks:,} chunks ({compacted:,} fewer) "
            f"in {compaction_seconds * 1000:.3f}ms"
        )

    current_post = measure_scenario(
        store=store,
        predicate=predicate,
        as_of_value=None,
        sql=current_sql,
        query_name=f"bitemporal-{label}-current-post",
    )
    _print_scenario("current (post-compaction)", current_post)
    current_comparison = _compare(
        "current pre-vs-post compaction", current_pre, current_post
    )

    as_of_post = None
    as_of_comparison = None
    if corrections:
        as_of_post = measure_scenario(
            store=store,
            predicate=predicate,
            as_of_value=as_of_value,
            sql=as_of_sql,
            query_name=f"bitemporal-{label}-as_of-post",
        )
        _print_scenario("as_of (post-compaction)", as_of_post)
        as_of_comparison = _compare(
            "as_of pre-vs-post compaction", as_of_pre, as_of_post
        )

    entry = {
        "label": label,
        "correction_rate": correction_rate,
        "base_rows": base_rows,
        "base_chunks": base_chunks,
        "num_corrections": len(corrections),
        "correction_chunks": chunks_before_compaction - base_chunks,
        "correction_seconds": correction_seconds,
        "correction_rows_per_second": correction_rows_per_second,
        "compacted_chunks": compacted,
        "compaction_seconds": compaction_seconds,
        "total_chunks": store.total_chunks,
        "scenarios": {
            "current_pre_compaction": current_pre,
            "current_post_compaction": current_post,
            **({"as_of_pre_compaction": as_of_pre} if as_of_pre else {}),
            **({"as_of_post_compaction": as_of_post} if as_of_post else {}),
        },
        "compaction_significance": {
            "current": current_comparison,
            **({"as_of": as_of_comparison} if as_of_comparison else {}),
        },
    }
    return entry


def _scenario_speedup_cell(scenarios: dict[str, Any], key: str) -> str:
    """Format one scenario's speedup for the summary table, or "--" if that
    scenario wasn't measured at this point (0% correction rate has no AS OF
    scenario -- there's nothing to look up as of a time before any
    correction existed)."""

    scenario = scenarios.get(key)
    return format_x(scenario["query_speedup"]) if scenario else "--"


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    print("\nBitemporal correction-volume sweep -- speedup summary")
    headers = [
        "Corrected",
        "Current, pre-cleanup",
        "Current, post-cleanup",
        "AS OF, pre-cleanup",
        "AS OF, post-cleanup",
    ]
    rows = [
        [
            entry["label"],
            _scenario_speedup_cell(entry["scenarios"], "current_pre_compaction"),
            _scenario_speedup_cell(entry["scenarios"], "current_post_compaction"),
            _scenario_speedup_cell(entry["scenarios"], "as_of_pre_compaction"),
            _scenario_speedup_cell(entry["scenarios"], "as_of_post_compaction"),
        ]
        for entry in results
    ]
    print(render_table(headers, rows))
    print(
        "(pre/post-cleanup = before/after compact_corrections(); AS OF is only\n"
        " measured once corrections exist, so 0% shows -- in both AS OF columns.\n"
        " 'Current' = an ordinary query, no AS OF cutoff; 'AS OF' = a point-in-time\n"
        " query predating every correction in this sweep.)"
    )


def main() -> int:
    results = [
        run_correction_rate(label=f"{rate:.0%}", correction_rate=rate)
        for rate in CORRECTION_RATES
    ]

    _print_summary_table(results)

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "bitemporal_sweep.json"
    payload = {"points": results}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
