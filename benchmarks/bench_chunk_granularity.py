"""Chunk-granularity tradeoff sweep for combined date-range + price-band queries.

Graph 1/2's Q5 crossover (broad, 50%-selectivity query, ~2.9x speedup) isn't
a bug -- once a query genuinely needs half the data, no zone map avoids
reading half of it. What *is* addressable: for a combined predicate (date
range AND price band), a coarser chunk means a coarser price zone map too,
so tuning ChunkBoundary granularity trades ingestion/chunk-count overhead
for pruning precision. This sweeps a few granularities against one fixed
combined-predicate query and reports the actual measured curve -- evidence
for an informed choice, not a change to MonthlyBoundary's status as the
default.

Mirrors benchmarks/bench_scale.py's and bench_query.py's conventions: ddof=1
sample std via trimmed_mean_and_std, stats.py's confidence_interval_95, and
the same full-scan/pushdown correctness gate (_assert_equivalent) every other
benchmark in this suite uses before reporting any latency number.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_query import _assert_equivalent
from benchmarks.bench_scale import _time_calls
from benchmarks.report import format_ms, format_pct, format_x, render_table
from benchmarks.stats import confidence_interval_95, trimmed_mean_and_std
from benchmarks.synthetic_data import generate_synthetic_ohlcv
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.store.chunk_boundary import ChunkBoundary, DurationBoundary, MonthlyBoundary
from src.store.chunk_store import ChunkStore

NUM_SYMBOLS = 10
ROWS_PER_SYMBOL = 501  # matches bench_scale.py's Dataset-A-shaped row count
QUERY_SYMBOL = "SYN000003"
QUERY_NUM_TRIALS = 15
QUERY_WARMUP = 3
# Matches bench_query.py's/bench_bitemporal.py's own use of the same
# constant -- see trimmed_mean_and_std's docstring.
LATENCY_TRIM_FRACTION = 0.15

GRANULARITIES: list[tuple[str, ChunkBoundary]] = [
    ("weekly", DurationBoundary(timedelta(days=7))),
    ("monthly (default)", MonthlyBoundary()),
    ("quarterly", DurationBoundary(timedelta(days=90))),
]


def run_granularity_point(label: str, boundary: ChunkBoundary) -> dict[str, Any]:
    print(f"\n=== granularity {label} ===")

    df = generate_synthetic_ohlcv(
        num_symbols=NUM_SYMBOLS, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )

    t0 = time.perf_counter()
    store = ChunkStore(chunk_boundary=boundary)
    store.ingest_dataframe(df)
    ingest_seconds = time.perf_counter() - t0

    # A price threshold that genuinely splits the query symbol's chunks:
    # the median close price across its own history, not an arbitrary
    # constant that might trivially match everything or nothing.
    symbol_closes = df.loc[df["symbol"] == QUERY_SYMBOL, "close"]
    price_threshold = float(symbol_closes.median())

    sql = (
        f"SELECT * FROM data WHERE symbol = '{QUERY_SYMBOL}' "
        f"AND close > {price_threshold:.4f} "
        "AND timestamp >= '2023-06-01' AND timestamp < '2023-09-01'"
    )

    pushdown_engine = QueryExecutor(store)
    full_scan_engine = FullScanQueryEngine(store)

    pushdown_times, pushdown_result = _time_calls(
        lambda: pushdown_engine.execute(sql), QUERY_NUM_TRIALS, QUERY_WARMUP
    )
    full_scan_times, full_scan_result = _time_calls(
        lambda: full_scan_engine.execute(sql), QUERY_NUM_TRIALS, QUERY_WARMUP
    )
    _assert_equivalent(
        full_scan_result, pushdown_result, query_name=f"granularity-{label}"
    )

    pushdown_mean, pushdown_std, pushdown_kept_n = trimmed_mean_and_std(
        pushdown_times, LATENCY_TRIM_FRACTION
    )
    pushdown_ci_low, pushdown_ci_high = confidence_interval_95(
        pushdown_mean, pushdown_std, pushdown_kept_n
    )
    full_scan_mean, _, _ = trimmed_mean_and_std(full_scan_times, LATENCY_TRIM_FRACTION)

    entry = {
        "label": label,
        "total_chunks": store.total_chunks,
        "total_rows": store.total_rows,
        "avg_rows_per_chunk": store.total_rows / store.total_chunks,
        "ingest_seconds": ingest_seconds,
        "price_threshold": price_threshold,
        "chunks_skipped_ratio": pushdown_result.pushdown.chunks_skipped_ratio,
        "candidate_chunks": pushdown_result.pushdown.candidate_count,
        "query_pushdown_mean_ms": pushdown_mean * 1000.0,
        "query_pushdown_ci95_low_ms": pushdown_ci_low * 1000.0,
        "query_pushdown_ci95_high_ms": pushdown_ci_high * 1000.0,
        "query_full_scan_mean_ms": full_scan_mean * 1000.0,
        "query_speedup": full_scan_mean / pushdown_mean if pushdown_mean else 0.0,
        "row_count": pushdown_result.row_count,
    }
    print(
        f"  {entry['total_chunks']:,} chunks "
        f"({entry['avg_rows_per_chunk']:.1f} rows/chunk avg), "
        f"ingest={entry['ingest_seconds']:.3f}s | "
        f"skipped={format_pct(entry['chunks_skipped_ratio'])} "
        f"candidates={entry['candidate_chunks']}/{entry['total_chunks']} | "
        f"pushdown={format_ms(entry['query_pushdown_mean_ms'])} "
        f"full_scan={format_ms(entry['query_full_scan_mean_ms'])} "
        f"speedup={format_x(entry['query_speedup'])}"
    )
    return entry


def main() -> int:
    results = [
        run_granularity_point(label, boundary) for label, boundary in GRANULARITIES
    ]

    print("\nChunk-granularity tradeoff -- summary")
    headers = ["Granularity", "Chunks", "Chunks skipped", "Speedup"]
    rows = [
        [
            entry["label"],
            f"{entry['total_chunks']:,}",
            format_pct(entry["chunks_skipped_ratio"]),
            format_x(entry["query_speedup"]),
        ]
        for entry in results
    ]
    print(render_table(headers, rows))

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "chunk_granularity_sweep.json"
    payload = {"points": results}
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
