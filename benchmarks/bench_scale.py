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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_query import _assert_equivalent
from benchmarks.measurement import measure_query_paths
from benchmarks.report import format_ms, format_pct, format_x, render_table
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit, hardware_snapshot
from benchmarks.reporting.provenance import (
    create_manifest,
    dataframe_fingerprint,
    write_result,
)
from benchmarks.stats import latency_summary, time_interleaved
from benchmarks.synthetic_data import generate_synthetic_ohlcv
from src.query.executor import QueryExecutor
from src.query.pushdown import evaluate_pushdown
from src.query.sql_parser import parse_where_clause
from src.store.chunk_boundary import MonthlyBoundary
from src.store.chunk_store import ChunkStore

ROWS_PER_SYMBOL = 501  # matches Dataset A's ~2-year daily row count per symbol


def run_scale_point(
    *,
    label: str,
    num_symbols: int,
    num_trials: int,
    warmup: int,
    run_full_scan: bool = True,
) -> dict[str, Any]:
    print()
    print(
        console.section_header(
            f"Scale point {label}: {num_symbols:,} symbols, "
            f"~{num_symbols * ROWS_PER_SYMBOL:,} rows",
            [
                console.phase(
                    "MEASURE",
                    f"query trials={num_trials}, warmup={warmup}, "
                    f"full_scan={run_full_scan}",
                ),
            ],
        )
    )

    # Collect the previous scale point's garbage before measuring this one's
    # memory footprint, so consecutive points in the same process don't
    # inflate each other's RSS reading.
    gc.collect()

    df = generate_synthetic_ohlcv(
        num_symbols=num_symbols, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )

    fixture = dataframe_fingerprint(df)
    t0 = time.perf_counter()
    store = ChunkStore(chunk_boundary=MonthlyBoundary())
    store.ingest_dataframe(df)
    ingest_seconds = time.perf_counter() - t0
    rss_after_ingest_bytes = psutil.Process().memory_info().rss
    print(
        console.phase(
            "BUILD",
            f"Ingested {store.total_rows:,} rows into {store.total_chunks:,} "
            f"chunks in {ingest_seconds:.2f}s "
            f"({store.total_rows / ingest_seconds:,.0f} rows/s); "
            f"RSS after ingest: {rss_after_ingest_bytes / (1024 * 1024):,.1f} MiB",
        )
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
        "narrow_raw_samples_seconds": narrow_pushdown_times.tolist(),
        "broad_raw_samples_seconds": broad_pushdown_times.tolist(),
        "num_trials": pushdown_trials,
        "acquisition_order": ["narrow"] * pushdown_trials + ["broad"] * pushdown_trials,
        "method": (
            "evaluate_pushdown(predicate, store.labels); includes label "
            "materialization, traversal and full candidate/skipped ID "
            "outputs; no AS OF index"
        ),
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
        console.phase(
            "MEASURE",
            f"Isolated evaluate_pushdown(): narrow="
            f"{pushdown_eval['narrow_mean_ms']:.4f}ms broad="
            f"{pushdown_eval['broad_mean_ms']:.4f}ms over "
            f"{store.total_chunks:,} labels "
            f"({pushdown_eval['ns_per_label_broad']:.1f} ns/label)",
        )
    )

    one_symbol = f"SYN{0:06d}"
    narrow_sql = f"SELECT * FROM data WHERE symbol = '{one_symbol}'"
    broad_sql = "SELECT * FROM data WHERE timestamp >= '2023-01-01'"

    query_results: dict[str, Any] = {}
    for name, sql in (
        ("narrow_symbol_eq", narrow_sql),
        ("broad_full_range", broad_sql),
    ):
        if run_full_scan:
            timing_metrics, pushdown_result = measure_query_paths(
                store,
                sql,
                num_trials=num_trials,
                warmup=warmup,
                query_name=f"{label} / {name}",
                quiet=False,
            )
            push = timing_metrics["pushdown_query"]
        else:
            pushdown_engine = QueryExecutor(store)
            vectorized_engine = QueryExecutor(store, enable_pruning=False)
            samples, results, order = time_interleaved(
                {
                    "pushdown_query": lambda: pushdown_engine.execute(sql),
                    "vectorized_full_scan": lambda: vectorized_engine.execute(sql),
                },
                num_trials,
                warmup,
            )
            pushdown_result = results["pushdown_query"]
            _assert_equivalent(
                results["vectorized_full_scan"], pushdown_result, query_name=name
            )
            push = latency_summary(samples["pushdown_query"])
            vector = latency_summary(samples["vectorized_full_scan"])
            timing_metrics = {
                "pushdown_query": push,
                "vectorized_full_scan": vector,
                "vectorized_full_scan_speedup": vector["mean_latency_seconds"]
                / push["mean_latency_seconds"],
                "acquisition_order": order,
                "timing_method": (
                    "interleaved paired rounds; alternating engine order; "
                    "warm single process"
                ),
                "full_scan_status": "not measured: explicitly disabled",
            }
        entry: dict[str, Any] = {
            "sql": sql,
            "pushdown_mean_ms": push["mean_latency_seconds"] * 1000,
            "pushdown_std_ms": push["std_latency_seconds"] * 1000,
            "pushdown_ci95_low_ms": push["ci95_low_seconds"] * 1000,
            "pushdown_ci95_high_ms": push["ci95_high_seconds"] * 1000,
            "row_count": pushdown_result.row_count,
            "chunks_skipped_ratio": pushdown_result.pushdown.chunks_skipped_ratio,
            "candidate_chunks": pushdown_result.pushdown.candidate_count,
            "total_chunks": store.total_chunks,
            **timing_metrics,
        }
        if run_full_scan:
            entry["full_scan_mean_ms"] = timing_metrics["query_full_scan_mean_ms"]
            entry["full_scan_std_ms"] = (
                timing_metrics["full_scan"]["std_latency_seconds"] * 1000
            )
            entry["speedup"] = timing_metrics["query_speedup"]
        query_results[name] = entry
        ci_low, ci_high = entry["pushdown_ci95_low_ms"], entry["pushdown_ci95_high_ms"]
        print(
            console.phase(
                "RESULT",
                f"{name}: pushdown={format_ms(entry['pushdown_mean_ms'])} "
                f"approx. normal interval (raw mean)=[{ci_low:.3f},{ci_high:.3f}]ms "
                + (
                    f"full_scan={format_ms(entry['full_scan_mean_ms'])} "
                    f"speedup={format_x(entry['speedup'])} "
                    if run_full_scan
                    else "(full scan skipped) "
                )
                + "vectorized_control_speedup="
                f"{format_x(entry['vectorized_full_scan_speedup'])} "
                + f"pruning_rate={format_pct(entry['chunks_skipped_ratio'])} "
                f"candidates={entry['candidate_chunks']}/{entry['total_chunks']}",
            )
        )

    return {
        "fixture": fixture,
        "synthetic_seed": 1,
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
    print(
        console.benchmark_header(
            index=3,
            total=5,
            name="Scale Sweep",
            question=(
                "Does zone-map pushdown's advantage over a naive full scan "
                "grow, shrink, or stay flat as row/chunk count scales?"
            ),
            config_lines=[
                "Dataset  : synthetic (same statistical shape as Dataset A), "
                "symbol count scaled 10 -> 10,000",
                "Chunking : MonthlyBoundary (same default as Dataset A)",
                "Queries  : narrow_symbol_eq (single symbol), "
                "broad_full_range (full date range)",
            ],
        )
    )

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

    manifest = create_manifest(
        parameters={
            "benchmark": "scale_sweep",
            "scale_points": scale_points,
            "rows_per_symbol": ROWS_PER_SYMBOL,
            "synthetic_seed": 1,
        }
    )
    results = [run_scale_point(**point) for point in scale_points]

    print()
    print(console.phase("RESULT", "Scale sweep -- summary (all scale points)"))
    headers = [
        "Scale",
        "Symbols",
        "Rows",
        "Chunks",
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
                f"{point['total_chunks']:,}",
                f"{point['ingest_rows_per_second']:,.0f}",
                format_x(narrow["speedup"]),
                format_x(broad["speedup"]),
                f"{point['pushdown_eval']['ns_per_label_broad']:.1f}",
            ]
        )
    print(render_table(headers, rows))

    first_narrow_speedup = results[0]["queries"]["narrow_symbol_eq"]["speedup"]
    last_narrow_speedup = results[-1]["queries"]["narrow_symbol_eq"]["speedup"]
    first_broad_speedup = results[0]["queries"]["broad_full_range"]["speedup"]
    last_broad_speedup = results[-1]["queries"]["broad_full_range"]["speedup"]
    print()
    print(
        console.interpretation(
            observation=(
                f"Narrow-query speedup goes from {format_x(first_narrow_speedup)} "
                f"at {results[0]['label']} to {format_x(last_narrow_speedup)} at "
                f"{results[-1]['label']}; broad-query speedup goes from "
                f"{format_x(first_broad_speedup)} to {format_x(last_broad_speedup)} "
                "over the same range."
            ),
            evidence=(
                "ns/label (isolated pushdown-evaluation cost) and the per-point "
                "pruning rate printed above are the measured basis for this trend."
            ),
            limitation=(
                f"Tested up to {results[-1]['num_symbols']:,} synthetic symbols / "
                f"{results[-1]['total_rows']:,} rows on one machine; not validated "
                "beyond this range or under concurrent query load."
            ),
        )
    )

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "scale_sweep.json"
    # Wrapped in an object, not a bare array: charts/generate_all.py's loader
    # requires every *.json file in benchmarks/results/ to parse to a JSON
    # object. "points" is byte-identical to what this file has always
    # written; "generated_at_utc"/"git_commit"/"hardware" are new, additive
    # provenance keys only -- see BENCHMARK_OUTPUT_AUDIT.md Section 9, item 1.
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hardware": hardware_snapshot(),
        "points": results,
    }
    write_result(output_path, payload, manifest)
    print()
    print(console.phase("EXPORT", str(output_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
