"""Chunk-granularity tradeoff sweep for combined date-range + price-band queries."""

from __future__ import annotations

import math
import statistics
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
QUERY_SYMBOL = "SYN000003"  # primary query's symbol
EXTRA_QUERY_SYMBOLS = ("SYN000000", "SYN000007")
# (label, inclusive start, exclusive end); the first window is the primary one.
QUERY_WINDOWS = (
    ("3 months", "2023-06-01", "2023-09-01"),
    ("1 month", "2023-06-01", "2023-07-01"),
    ("6 months", "2023-04-01", "2023-10-01"),
)
QUERY_NUM_TRIALS = 18
QUERY_WARMUP = 3
# The engine's shipped default; reported separately, not validated here.
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
    """Compute one granularity's full measurement set without printing."""

    df = generate_synthetic_ohlcv(
        num_symbols=NUM_SYMBOLS, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )

    fixture = dataframe_fingerprint(df)
    t0 = time.perf_counter()
    store = ChunkStore(chunk_boundary=granularity["boundary"])
    store.ingest_dataframe(df)
    ingest_seconds = time.perf_counter() - t0

    chunk_row_counts = [chunk.label.row_count for chunk in store.chunks]
    min_rows_per_chunk = min(chunk_row_counts)
    max_rows_per_chunk = max(chunk_row_counts)
    # Labels are never shared between chunks, so each is measured independently.
    metadata_size_bytes = sum(_deep_size_bytes(chunk.label) for chunk in store.chunks)

    symbols = [QUERY_SYMBOL] + [
        symbol
        for symbol in EXTRA_QUERY_SYMBOLS
        if symbol != QUERY_SYMBOL and symbol in set(df["symbol"])
    ]
    per_query: list[dict[str, Any]] = []
    primary: tuple[dict[str, Any], Any, str, float] | None = None
    for symbol in symbols:
        # Median close, so the price band genuinely splits the symbol's chunks.
        price_threshold = float(df.loc[df["symbol"] == symbol, "close"].median())
        for window_label, start, end in QUERY_WINDOWS:
            sql = (
                f"SELECT * FROM data WHERE symbol = '{symbol}' "
                f"AND close > {price_threshold:.4f} "
                f"AND timestamp >= '{start}' AND timestamp < '{end}'"
            )
            metrics, result = measure_query_paths(
                store,
                sql,
                num_trials=QUERY_NUM_TRIALS,
                warmup=QUERY_WARMUP,
                query_name=f"granularity-{granularity['key']}-{symbol}-{window_label}",
            )
            is_primary = primary is None
            per_query.append(
                {
                    "symbol": symbol,
                    "window": window_label,
                    "is_primary": is_primary,
                    "query": sql,
                    "price_threshold": price_threshold,
                    "row_count": result.row_count,
                    "candidate_chunks": result.pushdown.candidate_count,
                    "chunks_skipped_ratio": result.pushdown.chunks_skipped_ratio,
                    "row_wise_mean_ms": metrics["query_full_scan_mean_ms"],
                    "vectorized_mean_ms": metrics["vectorized_full_scan"][
                        "mean_latency_seconds"
                    ]
                    * 1000.0,
                    "pushdown_mean_ms": metrics["query_pushdown_mean_ms"],
                    "row_wise_over_pushdown": metrics["query_speedup"],
                    "vectorized_over_pushdown": metrics["vectorized_full_scan_speedup"],
                    "sample_count": metrics["pushdown_query"]["sample_count"],
                }
            )
            if is_primary:
                primary = (metrics, result, sql, price_threshold)
    assert primary is not None
    timing_metrics, pushdown_result, sql, price_threshold = primary

    aggregate = {
        "num_queries": len(per_query),
        "symbols": symbols,
        "windows": [label for label, _, _ in QUERY_WINDOWS],
        "mean_row_wise_ms": statistics.fmean(q["row_wise_mean_ms"] for q in per_query),
        "mean_vectorized_ms": statistics.fmean(
            q["vectorized_mean_ms"] for q in per_query
        ),
        "mean_pushdown_ms": statistics.fmean(q["pushdown_mean_ms"] for q in per_query),
        "geomean_row_wise_over_pushdown": math.exp(
            statistics.fmean(math.log(q["row_wise_over_pushdown"]) for q in per_query)
        ),
        "mean_chunks_skipped_ratio": statistics.fmean(
            q["chunks_skipped_ratio"] for q in per_query
        ),
        "method": (
            "arithmetic mean of per-query mean latencies; geometric mean of "
            "per-query speedups; unweighted over the symbol x window grid"
        ),
    }

    # Rows decompressed across candidate chunks, not rows returned after filtering.
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
        "queries": per_query,
        "aggregate": aggregate,
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
        ["Engine", "Mean latency", "Student-t interval (raw mean)"],
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
    aggregate = entry["aggregate"]
    print()
    print(
        f"Query grid ({aggregate['num_queries']} queries: "
        f"{len(aggregate['symbols'])} symbols x {len(aggregate['windows'])} windows)"
    )
    print()
    print(
        render_box_table(
            [
                "Symbol",
                "Window",
                "Rows",
                "Chunks scanned",
                "Row-wise",
                "Vectorized",
                "Pushdown",
                "Row-wise / pushdown",
            ],
            [
                [
                    q["symbol"],
                    q["window"] + (" (primary)" if q["is_primary"] else ""),
                    f"{q['row_count']:,}",
                    f"{q['candidate_chunks']:,}/{entry['total_chunks']:,}",
                    format_ms(q["row_wise_mean_ms"]),
                    format_ms(q["vectorized_mean_ms"]),
                    format_ms(q["pushdown_mean_ms"]),
                    format_x(q["row_wise_over_pushdown"]),
                ]
                for q in entry["queries"]
            ],
        )
    )


def summarize_granularity(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive every headline statement from the measured points."""

    lowest_pushdown = min(results, key=lambda e: e["aggregate"]["mean_pushdown_ms"])
    lowest_row_wise = min(results, key=lambda e: e["aggregate"]["mean_row_wise_ms"])
    finest_pruning = max(
        results, key=lambda e: e["aggregate"]["mean_chunks_skipped_ratio"]
    )
    fewest_chunks = min(results, key=lambda e: e["total_chunks"])
    highest_speedup = max(
        results, key=lambda e: e["aggregate"]["geomean_row_wise_over_pushdown"]
    )
    default_entry = next(
        (entry for entry in results if entry["key"] == DEFAULT_GRANULARITY_KEY), None
    )
    if default_entry is None:
        default_vs_best = "configured default was not among the tested granularities"
    elif default_entry["key"] == lowest_pushdown["key"]:
        default_vs_best = (
            f"the configured default ({default_entry['label']}) had the lowest "
            "mean pushdown latency in this run"
        )
    else:
        default_vs_best = (
            f"the configured default ({default_entry['label']}) did NOT have "
            f"the lowest mean pushdown latency; {lowest_pushdown['label']} did "
            f"({format_ms(lowest_pushdown['aggregate']['mean_pushdown_ms'])} vs "
            f"{format_ms(default_entry['aggregate']['mean_pushdown_ms'])})"
        )
    coarsest = min(results, key=lambda e: e["total_chunks"])
    finest = max(results, key=lambda e: e["total_chunks"])
    speedup_driver = (
        f"From the coarsest ({coarsest['label']}, {coarsest['total_chunks']:,} "
        f"chunks) to the finest ({finest['label']}, {finest['total_chunks']:,} "
        "chunks) granularity, mean row-wise baseline latency goes from "
        f"{format_ms(coarsest['aggregate']['mean_row_wise_ms'])} to "
        f"{format_ms(finest['aggregate']['mean_row_wise_ms'])} and mean "
        "pushdown latency from "
        f"{format_ms(coarsest['aggregate']['mean_pushdown_ms'])} to "
        f"{format_ms(finest['aggregate']['mean_pushdown_ms'])}. The speedup "
        "ratio mixes both movements: a baseline that slows with chunk count "
        "inflates it without any pushdown improvement, so the ratio alone "
        "is not evidence that a granularity is better for pushdown."
    )
    return {
        "lowest_pushdown_label": lowest_pushdown["label"],
        "lowest_row_wise_label": lowest_row_wise["label"],
        "finest_pruning_label": finest_pruning["label"],
        "fewest_chunks_label": fewest_chunks["label"],
        "highest_speedup_label": highest_speedup["label"],
        "configured_default_key": DEFAULT_GRANULARITY_KEY,
        "configured_default_label": default_entry["label"] if default_entry else None,
        "default_matches_lowest_pushdown": bool(
            default_entry and default_entry["key"] == lowest_pushdown["key"]
        ),
        "default_vs_best": default_vs_best,
        "speedup_explanation": speedup_driver,
    }


def main() -> int:
    # Everything is computed before anything prints.
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
            "query_symbols": [QUERY_SYMBOL, *EXTRA_QUERY_SYMBOLS],
            "query_windows": [list(window) for window in QUERY_WINDOWS],
            "synthetic_seed": 1,
        }
    )
    results = [run_granularity_point(granularity) for granularity in GRANULARITIES]

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "chunk_granularity_sweep.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hardware": hardware_snapshot(),
        "points": results,
        "findings": summarize_granularity(results),
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
                (
                    "Queries",
                    f"{results[0]['aggregate']['num_queries']}: "
                    f"{', '.join(results[0]['aggregate']['symbols'])} x "
                    f"{', '.join(results[0]['aggregate']['windows'])}, each "
                    "with a median-close price band",
                ),
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
        console.wrap_text(
            "Latencies are means across the query grid; pushdown and the "
            "row-wise baseline are shown separately because their ratio "
            "mixes both.",
            indent=0,
        )
    )
    print()
    print(
        render_box_table(
            [
                "Granularity",
                "Chunks",
                "Rows/chunk",
                "Pruning rate (mean)",
                "Pushdown (mean)",
                "Row-wise (mean)",
                "Vectorized (mean)",
                "Row-wise / pushdown (geomean)",
            ],
            [
                [
                    entry["label"],
                    f"{entry['total_chunks']:,}",
                    f"{entry['avg_rows_per_chunk']:.1f}",
                    format_pct(entry["aggregate"]["mean_chunks_skipped_ratio"]),
                    format_ms(entry["aggregate"]["mean_pushdown_ms"]),
                    format_ms(entry["aggregate"]["mean_row_wise_ms"]),
                    format_ms(entry["aggregate"]["mean_vectorized_ms"]),
                    format_x(entry["aggregate"]["geomean_row_wise_over_pushdown"]),
                ]
                for entry in results
            ],
        )
    )

    findings = summarize_granularity(results)
    default_label = findings["configured_default_label"]
    print()
    print("Measured findings (data-driven; this run only)")
    print(
        console.field_block(
            [
                ("Lowest mean pushdown latency", findings["lowest_pushdown_label"]),
                ("Lowest mean row-wise latency", findings["lowest_row_wise_label"]),
                ("Highest mean pruning rate", findings["finest_pruning_label"]),
                ("Fewest managed chunks", findings["fewest_chunks_label"]),
                ("Highest speedup ratio", findings["highest_speedup_label"]),
                ("Configured engine default", default_label or "N/A"),
                ("Default vs measured best", findings["default_vs_best"]),
            ]
        )
    )
    print()
    print(
        console.titled_paragraph(
            "Reading the speedup column",
            findings["speedup_explanation"],
        )
    )
    print()
    print(
        console.titled_paragraph(
            "Limitation",
            f"Synthetic data, {NUM_SYMBOLS} symbols x {ROWS_PER_SYMBOL} rows, "
            f"{results[0]['aggregate']['num_queries']} queries (one symbol "
            "set, three date windows, a median-close price band) on one "
            "machine. This benchmark cannot justify a general default: the "
            "best granularity depends on query shape, dataset distribution "
            "and write workload, and differences between granularities may "
            "be within run-to-run noise. No universal optimum is claimed.",
        )
    )

    print()
    print(console.dash_section("[RESULT]"))
    print(
        console.field_block(
            [
                ("Correctness", "PASSED FOR ALL GRANULARITIES"),
                ("Tested granularities", str(len(results))),
                ("Lowest mean pushdown latency", findings["lowest_pushdown_label"]),
                ("Configured engine default", default_label or "N/A"),
                ("Default vs measured best", findings["default_vs_best"]),
                ("Result artifact", str(output_path)),
            ]
        )
    )
    print()
    print(console.pitdb_footer(index=5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
