"""Scale-sweep benchmark using synthetic data at 10x/100x/1000x the size of
Dataset A.

Dataset A tops out at ~5,000 rows / 240 chunks, so no measurement exists at
production-representative scale. This module generates synthetic OHLCV data
with the same shape (random-walk price columns, lognormal volume) at
increasing symbol counts -- more tracked tickers over the same date range,
so both row count and chunk count grow together the way they would in a
real deployment -- and reports how ingestion throughput, pushdown-evaluation
cost, query latency, and chunks-skipped-ratio behave as scale increases.

The full-scan baseline decompresses every chunk on every call, so its cost
scales linearly with total chunk count. At the largest scales that makes a
full NUM_TRIALS run prohibitively slow; trial counts are reduced (and
reported) at those scales rather than silently skipping the measurement.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_query import _assert_equivalent
from benchmarks.report import format_ms, format_pct, format_x, render_table
from benchmarks.stats import confidence_interval_95, trimmed_mean_and_std
from benchmarks.synthetic_data import generate_synthetic_ohlcv
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.query.pushdown import evaluate_pushdown
from src.query.sql_parser import parse_where_clause
from src.store.chunk_boundary import MonthlyBoundary
from src.store.chunk_store import ChunkStore

ROWS_PER_SYMBOL = 501  # matches Dataset A's ~2-year daily row count per symbol
# Matches bench_query.py's and bench_bitemporal.py's own use of the same
# constant -- see trimmed_mean_and_std's docstring. Applied only to the
# decompression-heavy query-latency measurements below, not the microsecond-
# scale isolated evaluate_pushdown() loop, which stays on a plain mean (same
# split bench_bitemporal.py already makes).
LATENCY_TRIM_FRACTION = 0.15


def _time_calls(fn, num_trials: int, warmup: int) -> tuple[np.ndarray, Any]:
    for _ in range(warmup):
        fn()
    times = np.empty(num_trials, dtype=np.float64)
    result = None
    for trial in range(num_trials):
        t0 = time.perf_counter()
        result = fn()
        times[trial] = time.perf_counter() - t0
    return times, result


def run_scale_point(
    *,
    label: str,
    num_symbols: int,
    num_trials: int,
    warmup: int,
    run_full_scan: bool = True,
) -> dict[str, Any]:
    print(
        f"\n=== scale point {label}: {num_symbols} symbols, "
        f"~{num_symbols * ROWS_PER_SYMBOL:,} rows "
        f"(query trials={num_trials}, warmup={warmup}, "
        f"full_scan={run_full_scan}) ==="
    )

    # Collect the previous scale point's garbage before measuring this one's
    # memory footprint, so consecutive points in the same process don't
    # inflate each other's RSS reading.
    gc.collect()

    df = generate_synthetic_ohlcv(
        num_symbols=num_symbols, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )

    t0 = time.perf_counter()
    store = ChunkStore(chunk_boundary=MonthlyBoundary())
    store.ingest_dataframe(df)
    ingest_seconds = time.perf_counter() - t0
    rss_after_ingest_bytes = psutil.Process().memory_info().rss
    print(
        f"ingested {store.total_rows:,} rows into {store.total_chunks:,} "
        f"chunks in {ingest_seconds:.2f}s "
        f"({store.total_rows / ingest_seconds:,.0f} rows/s); "
        f"RSS after ingest = {rss_after_ingest_bytes / (1024 * 1024):,.1f} MB"
    )

    # Isolate pushdown label-evaluation cost from full query execution
    # (decompression, filtering). Timed directly, not inferred.
    narrow_predicate = parse_where_clause(
        f"SELECT * FROM data WHERE symbol = 'SYN{0:06d}'"
    )
    broad_predicate = parse_where_clause(
        "SELECT * FROM data WHERE timestamp >= '2023-01-01'"
    )
    pushdown_trials = 50
    narrow_pushdown_times = np.empty(pushdown_trials, dtype=np.float64)
    for trial in range(pushdown_trials):
        t0 = time.perf_counter()
        evaluate_pushdown(narrow_predicate, store.labels)
        narrow_pushdown_times[trial] = time.perf_counter() - t0
    broad_pushdown_times = np.empty(pushdown_trials, dtype=np.float64)
    for trial in range(pushdown_trials):
        t0 = time.perf_counter()
        evaluate_pushdown(broad_predicate, store.labels)
        broad_pushdown_times[trial] = time.perf_counter() - t0
    pushdown_eval = {
        "narrow_mean_ms": float(np.mean(narrow_pushdown_times)) * 1000.0,
        "broad_mean_ms": float(np.mean(broad_pushdown_times)) * 1000.0,
        "ns_per_label_narrow": (
            float(np.mean(narrow_pushdown_times)) * 1e9 / store.total_chunks
        ),
        "ns_per_label_broad": (
            float(np.mean(broad_pushdown_times)) * 1e9 / store.total_chunks
        ),
    }
    print(
        f"pushdown evaluate_pushdown() isolated: narrow="
        f"{pushdown_eval['narrow_mean_ms']:.4f}ms broad="
        f"{pushdown_eval['broad_mean_ms']:.4f}ms over {store.total_chunks:,} labels "
        f"({pushdown_eval['ns_per_label_broad']:.1f} ns/label)"
    )

    one_symbol = f"SYN{0:06d}"
    narrow_sql = f"SELECT * FROM data WHERE symbol = '{one_symbol}'"
    broad_sql = "SELECT * FROM data WHERE timestamp >= '2023-01-01'"

    pushdown_engine = QueryExecutor(store)
    query_results: dict[str, Any] = {}
    for name, sql in (
        ("narrow_symbol_eq", narrow_sql),
        ("broad_full_range", broad_sql),
    ):
        pushdown_times, pushdown_result = _time_calls(
            lambda sql=sql: pushdown_engine.execute(sql), num_trials, warmup
        )
        pushdown_mean, pushdown_std, pushdown_kept_n = trimmed_mean_and_std(
            pushdown_times, LATENCY_TRIM_FRACTION
        )
        pushdown_ci_low, pushdown_ci_high = confidence_interval_95(
            pushdown_mean, pushdown_std, pushdown_kept_n
        )
        entry: dict[str, Any] = {
            "sql": sql,
            "pushdown_mean_ms": pushdown_mean * 1000.0,
            "pushdown_std_ms": pushdown_std * 1000.0,
            "pushdown_ci95_low_ms": pushdown_ci_low * 1000.0,
            "pushdown_ci95_high_ms": pushdown_ci_high * 1000.0,
            "row_count": pushdown_result.row_count,
            "chunks_skipped_ratio": pushdown_result.pushdown.chunks_skipped_ratio,
            "candidate_chunks": pushdown_result.pushdown.candidate_count,
            "total_chunks": store.total_chunks,
        }
        if run_full_scan:
            full_scan_engine = FullScanQueryEngine(store)
            full_scan_times, full_scan_result = _time_calls(
                lambda sql=sql: full_scan_engine.execute(sql), num_trials, warmup
            )
            # Same correctness gate bench_query.py uses for every other query
            # in this project: full row/value/dtype equality, not just a
            # count match, so a bug that returns the right count but wrong
            # rows can't pass silently at scale.
            _assert_equivalent(full_scan_result, pushdown_result, query_name=name)
            full_scan_mean, full_scan_std, _ = trimmed_mean_and_std(
                full_scan_times, LATENCY_TRIM_FRACTION
            )
            entry["full_scan_mean_ms"] = full_scan_mean * 1000.0
            entry["full_scan_std_ms"] = full_scan_std * 1000.0
            entry["speedup"] = entry["full_scan_mean_ms"] / entry["pushdown_mean_ms"]
        query_results[name] = entry
        ci_low, ci_high = entry["pushdown_ci95_low_ms"], entry["pushdown_ci95_high_ms"]
        print(
            f"  {name}: pushdown={format_ms(entry['pushdown_mean_ms'])} "
            f"[{ci_low:.3f},{ci_high:.3f}] "
            + (
                f"full_scan={format_ms(entry['full_scan_mean_ms'])} "
                f"speedup={format_x(entry['speedup'])} "
                if run_full_scan
                else "(full scan skipped) "
            )
            + f"skipped={format_pct(entry['chunks_skipped_ratio'])} "
            f"candidates={entry['candidate_chunks']}/{entry['total_chunks']}"
        )

    return {
        "label": label,
        "num_symbols": num_symbols,
        "total_rows": store.total_rows,
        "total_chunks": store.total_chunks,
        "ingest_seconds": ingest_seconds,
        "ingest_rows_per_second": store.total_rows / ingest_seconds,
        "rss_after_ingest_bytes": rss_after_ingest_bytes,
        "pushdown_eval": pushdown_eval,
        "queries": query_results,
        "query_num_trials": num_trials,
        "query_warmup": warmup,
    }


def main() -> int:
    scale_points = [
        {
            "label": "1x",
            "num_symbols": 10,
            "num_trials": 15,
            "warmup": 3,
            "run_full_scan": True,
        },
        {
            "label": "10x",
            "num_symbols": 100,
            "num_trials": 15,
            "warmup": 3,
            "run_full_scan": True,
        },
        {
            "label": "100x",
            "num_symbols": 1000,
            "num_trials": 10,
            "warmup": 2,
            "run_full_scan": True,
        },
        {
            "label": "1000x",
            "num_symbols": 10000,
            "num_trials": 3,
            "warmup": 0,
            "run_full_scan": True,
        },
    ]

    results = [run_scale_point(**point) for point in scale_points]

    print("\nScale sweep -- summary")
    headers = [
        "Scale",
        "Symbols",
        "Rows",
        "Ingest rows/s",
        "Narrow speedup",
        "Broad speedup",
        "ns/label",
    ]
    rows = []
    for point in results:
        narrow = point["queries"]["narrow_symbol_eq"]
        broad = point["queries"]["broad_full_range"]
        rows.append(
            [
                point["label"],
                f"{point['num_symbols']:,}",
                f"{point['total_rows']:,}",
                f"{point['ingest_rows_per_second']:,.0f}",
                format_x(narrow["speedup"]),
                format_x(broad["speedup"]),
                f"{point['pushdown_eval']['ns_per_label_broad']:.1f}",
            ]
        )
    print(render_table(headers, rows))

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "scale_sweep.json"
    # Wrapped in an object, not a bare array: charts/generate_all.py's loader
    # requires every *.json file in benchmarks/results/ to parse to a JSON
    # object.
    payload = {"points": results}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
