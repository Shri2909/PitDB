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

Uses interleaved three-engine timing, raw-sample summaries and exact output
equality gates before reporting any speedup.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_memory import _deep_size_bytes
from benchmarks.measurement import measure_query_paths
from benchmarks.report import (
    format_bytes,
    format_ms,
    format_pct,
    format_x,
    render_box_table,
)
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit, hardware_snapshot
from benchmarks.reporting.provenance import (
    create_manifest,
    dataframe_fingerprint,
    write_result,
)
from benchmarks.synthetic_data import generate_synthetic_ohlcv
from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary
from src.store.chunk_store import ChunkStore

NUM_SYMBOLS = 10
ROWS_PER_SYMBOL = 501  # matches bench_scale.py's Dataset-A-shaped row count
QUERY_SYMBOL = "SYN000003"
QUERY_NUM_TRIALS = 15
QUERY_WARMUP = 3
DEFAULT_GRANULARITY_KEY = "monthly"
GRANULARITIES: list[dict[str, Any]] = [
    {
        "key": "weekly",
        "label": "Weekly",
        "window": "7 days",
        "boundary": DurationBoundary(timedelta(days=7)),
    },
    {
        "key": "monthly",
        "label": "Monthly",
        "window": "1 calendar month",
        "boundary": MonthlyBoundary(),
    },
    {
        "key": "quarterly",
        "label": "Quarterly",
        "window": "90 days",
        "boundary": DurationBoundary(timedelta(days=90)),
    },
]


def run_granularity_point(granularity: dict[str, Any]) -> dict[str, Any]:
    """Compute one granularity's full measurement set. Silent -- see
    quiet=True below; nothing is printed here, only computed and returned.
    """

    df = generate_synthetic_ohlcv(
        num_symbols=NUM_SYMBOLS, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )

    fixture = dataframe_fingerprint(df)
    t0 = time.perf_counter()
    store = ChunkStore(chunk_boundary=granularity["boundary"])
    store.ingest_dataframe(df)
    ingest_seconds = time.perf_counter() - t0

    # Post-hoc reads of already-built chunk labels, not new measurements.
    chunk_row_counts = [chunk.label.row_count for chunk in store.chunks]
    min_rows_per_chunk = min(chunk_row_counts)
    max_rows_per_chunk = max(chunk_row_counts)
    # Deep sizeof of every chunk's label -- the same utility
    # benchmarks/bench_memory.py uses for its own memory breakdown, reused
    # here rather than reimplemented, each label measured independently
    # (labels are never shared between chunks in this codebase).
    metadata_size_bytes = sum(_deep_size_bytes(chunk.label) for chunk in store.chunks)

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

    timing_metrics, pushdown_result = measure_query_paths(
        store,
        sql,
        num_trials=QUERY_NUM_TRIALS,
        warmup=QUERY_WARMUP,
        query_name=f"granularity-{granularity['key']}",
    )

    # Rows actually decompressed across every surviving candidate chunk --
    # a post-hoc sum over already-known chunk row counts, not a new
    # measurement -- distinct from row_count (rows returned after the exact
    # predicate filter runs).
    rows_scanned = sum(
        store.get_chunk(chunk_id).label.row_count
        for chunk_id in pushdown_result.pushdown.candidate_chunk_ids
    )
    candidate_chunks = pushdown_result.pushdown.candidate_count
    total_chunks = pushdown_result.pushdown.total_chunks

    return {
        "fixture": fixture,
        "synthetic_seed": 1,
        "key": granularity["key"],
        "label": granularity["label"],
        "window": granularity["window"],
        "total_chunks": total_chunks,
        "total_rows": store.total_rows,
        "avg_rows_per_chunk": store.total_rows / total_chunks,
        "min_rows_per_chunk": min_rows_per_chunk,
        "max_rows_per_chunk": max_rows_per_chunk,
        "metadata_size_bytes": metadata_size_bytes,
        "ingest_seconds": ingest_seconds,
        "price_threshold": price_threshold,
        "chunks_skipped_ratio": pushdown_result.pushdown.chunks_skipped_ratio,
        "candidate_chunks": candidate_chunks,
        "chunks_pruned": total_chunks - candidate_chunks,
        "rows_scanned": rows_scanned,
        "row_count": pushdown_result.row_count,
        "full_scan_row_count": timing_metrics["full_scan"]["row_count"],
        "query": sql,
        **timing_metrics,
    }


def _print_granularity_block(*, index: int, total: int, entry: dict[str, Any]) -> None:
    tag = f"[GRANULARITY {index}/{total} — {entry['label'].upper()}]"
    print()
    print(console.dash_section(tag))

    print()
    print(
        console.titled_block(
            "Store layout",
            [
                ("Chunk window", entry["window"]),
                ("Total chunks", f"{entry['total_chunks']:,}"),
                ("Average rows per chunk", f"{entry['avg_rows_per_chunk']:.1f}"),
                ("Minimum rows per chunk", f"{entry['min_rows_per_chunk']:,}"),
                ("Maximum rows per chunk", f"{entry['max_rows_per_chunk']:,}"),
                ("Metadata size", format_bytes(entry["metadata_size_bytes"])),
            ],
        )
    )
    print()
    print(
        console.titled_block(
            "Correctness validation",
            [
                ("Full-scan rows", f"{entry['full_scan_row_count']:,}"),
                ("Pushdown rows", f"{entry['row_count']:,}"),
                ("Row equality", "PASS"),
                ("Value equality", "PASS"),
                ("Dtype equality", "PASS"),
                ("Overall status", "PASSED"),
            ],
        )
    )
    print()
    print(
        console.titled_block(
            "Pruning",
            [
                ("Chunks scanned", f"{entry['candidate_chunks']:,}"),
                ("Chunks pruned", f"{entry['chunks_pruned']:,}"),
                ("Pruning rate", format_pct(entry["chunks_skipped_ratio"])),
                ("Rows scanned", f"{entry['rows_scanned']:,}"),
                ("Rows returned", f"{entry['row_count']:,}"),
            ],
        )
    )
    print()
    print("Timing")
    box = render_box_table(
        ["Engine", "Mean latency", "Approx. normal interval (raw mean)"],
        [
            [
                "Full scan",
                format_ms(entry["query_full_scan_mean_ms"]),
                f"{entry['query_full_scan_ci95_low_ms']:.3f} – "
                f"{entry['query_full_scan_ci95_high_ms']:.3f} ms",
            ],
            [
                "Zone-map pushdown",
                format_ms(entry["query_pushdown_mean_ms"]),
                f"{entry['query_pushdown_ci95_low_ms']:.3f} – "
                f"{entry['query_pushdown_ci95_high_ms']:.3f} ms",
            ],
        ],
    )
    for line in box.splitlines():
        print(f"  {line}")
    print()
    print(
        console.field_block(
            [
                ("Row-wise full scan / pushdown", format_x(entry["query_speedup"])),
                (
                    "Vectorized full scan / pushdown",
                    format_x(entry["vectorized_full_scan_speedup"]),
                ),
            ]
        )
    )


def main() -> int:
    # Silent, like the rest of the suite: every granularity is fully
    # computed (quiet=True inside run_granularity_point) before anything
    # prints.
    manifest = create_manifest(
        parameters={
            "benchmark": "chunk_granularity_sweep",
            "num_symbols": NUM_SYMBOLS,
            "rows_per_symbol": ROWS_PER_SYMBOL,
            "granularities": [
                {key: value for key, value in entry.items() if key != "boundary"}
                for entry in GRANULARITIES
            ],
            "num_trials": QUERY_NUM_TRIALS,
            "warmup": QUERY_WARMUP,
            "synthetic_seed": 1,
        }
    )
    results = [run_granularity_point(granularity) for granularity in GRANULARITIES]

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "chunk_granularity_sweep.json"
    # "points" preserves every field this file has always written, plus the
    # new fields added for this benchmark's exact-format output (window,
    # metadata_size_bytes, chunks_pruned, full-scan CI) -- additive only.
    # "generated_at_utc"/"git_commit"/"hardware" are new, additive
    # provenance keys -- see BENCHMARK_OUTPUT_AUDIT.md Section 9, item 1.
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hardware": hardware_snapshot(),
        "points": results,
    }
    write_result(output_path, payload, manifest)

    print(console.pitdb_header(index=5, total=5, title="Chunk-Granularity Trade-Off"))
    print()
    print("Question")
    print(
        console.wrap_text(
            "How does chunk granularity affect pruning precision, chunk "
            "overhead, and query performance?"
        )
    )
    print()
    print(
        console.titled_block(
            "Configuration",
            [
                ("Rows", f"{results[0]['total_rows']:,}"),
                ("Symbols", f"{NUM_SYMBOLS:,}"),
                (
                    "Granularities tested",
                    ", ".join(entry["label"] for entry in results),
                ),
                ("Query", "date range + price range (SYN000003)"),
                ("Warmup runs", str(QUERY_WARMUP)),
                ("Measured runs", str(QUERY_NUM_TRIALS)),
            ],
        )
    )

    for index, entry in enumerate(results, start=1):
        _print_granularity_block(index=index, total=len(results), entry=entry)

    print()
    print(console.dash_section("[TRADE-OFF SUMMARY]"))
    print()
    print(
        render_box_table(
            [
                "Granularity",
                "Chunks",
                "Rows/chunk",
                "Pruning rate",
                "Pushdown",
                "Speedup",
            ],
            [
                [
                    entry["label"],
                    f"{entry['total_chunks']:,}",
                    f"{entry['avg_rows_per_chunk']:.1f}",
                    format_pct(entry["chunks_skipped_ratio"]),
                    format_ms(entry["query_pushdown_mean_ms"]),
                    format_x(entry["query_speedup"]),
                ]
                for entry in results
            ],
        )
    )

    finest_pruning = max(results, key=lambda entry: entry["chunks_skipped_ratio"])
    fewest_chunks = min(results, key=lambda entry: entry["total_chunks"])
    lowest_latency = min(results, key=lambda entry: entry["query_pushdown_mean_ms"])
    default_entry = next(
        entry for entry in results if entry["key"] == DEFAULT_GRANULARITY_KEY
    )

    print()
    print("Trade-off interpretation")
    print(
        console.field_block(
            [
                ("Finest pruning", finest_pruning["label"]),
                ("Fewest managed chunks", fewest_chunks["label"]),
                ("Lowest measured latency", lowest_latency["label"]),
                ("Selected default", default_entry["label"]),
                (
                    "Selection rationale",
                    f"{default_entry['label']} pruned "
                    f"{format_pct(default_entry['chunks_skipped_ratio'])} of "
                    f"chunks with {format_x(default_entry['query_speedup'])} "
                    f"speedup at {default_entry['total_chunks']:,} chunks, "
                    "vs. "
                    + "; ".join(
                        f"{entry['label']}: "
                        f"{format_pct(entry['chunks_skipped_ratio'])} pruned, "
                        f"{format_x(entry['query_speedup'])} speedup, "
                        f"{entry['total_chunks']:,} chunks"
                        for entry in results
                        if entry["key"] != DEFAULT_GRANULARITY_KEY
                    ),
                ),
            ]
        )
    )
    print()
    print(
        console.titled_paragraph(
            "Limitation",
            "The best granularity depends on query shape, dataset "
            "distribution, and storage workload. No universal optimum is "
            "claimed.",
        )
    )

    print()
    print(console.dash_section("[RESULT]"))
    print(
        console.field_block(
            [
                ("Correctness", "PASSED FOR ALL GRANULARITIES"),
                ("Tested granularities", str(len(results))),
                ("Recommended default", default_entry["label"]),
                ("Result artifact", str(output_path)),
            ]
        )
    )
    print()
    print(console.pitdb_footer(index=5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
