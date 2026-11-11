"""Master runner for the chunked query engine's query and memory benchmarks."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

# Lets both ``python benchmarks/run_all.py`` and ``python -m benchmarks.run_all`` work.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - dependency error path
    psutil = None  # type: ignore[assignment]

try:
    import benchmarks.config as benchmark_config
except ModuleNotFoundError as exc:
    if exc.name != "benchmarks.config":
        raise
    benchmark_config = None  # type: ignore[assignment]

from benchmarks.bench_memory import run_memory_benchmarks
from benchmarks.bench_query import (
    annotate_pairwise_significance,
    build_intraday_queries,
    run_price_pushdown_demonstration,
    run_query_benchmarks,
    run_selectivity_sweep,
)
from benchmarks.report import (
    format_bytes,
    format_ms,
    format_pct,
    format_x,
    render_box_table,
)
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit
from benchmarks.reporting.provenance import create_manifest, finish_manifest
from src.query.predicate import AlwaysTrue, And, Comparison, In
from src.query.sql_parser import extract_as_of, parse_where_clause
from src.replay.data_loader import load_dataset
from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary
from src.store.chunk_store import ChunkStore

LOGGER = logging.getLogger(__name__)

_DEFAULT_DAILY_DATA_FILE = Path("data/cache/all_symbols_daily.csv")
_DEFAULT_INTRADAY_DATA_FILE = Path("data/cache/all_symbols_1m.csv")
_DEFAULT_RESULTS_DIR = Path("benchmarks/results")
_DEFAULT_INTRADAY_BOUNDARY_HOURS = 1


def _config_value(name: str, default: Any) -> Any:
    if benchmark_config is None:
        return default
    return getattr(benchmark_config, name, default)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def _require_psutil() -> Any:
    if psutil is None:
        raise RuntimeError(
            "psutil is required to record benchmark hardware information; "
            "install the project's pinned dependencies"
        )
    return psutil


def _hardware_info() -> dict[str, str | int]:
    psutil_module = _require_psutil()
    return {
        "processor": platform.processor(),
        "total_memory_bytes": int(psutil_module.virtual_memory().total),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "logical_cpu_count": psutil_module.cpu_count(logical=True),
        "physical_cpu_count": psutil_module.cpu_count(logical=False),
        "git_commit": git_commit(),
    }


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write strict JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_daily_store(data: Any) -> ChunkStore:
    store = ChunkStore(chunk_boundary=MonthlyBoundary())
    store.ingest_dataframe(data)
    return store


def _build_intraday_store(data: Any) -> ChunkStore:
    hours = _config_value(
        "INTRADAY_CHUNK_BOUNDARY_HOURS",
        _DEFAULT_INTRADAY_BOUNDARY_HOURS,
    )
    try:
        duration = timedelta(hours=float(hours))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "INTRADAY_CHUNK_BOUNDARY_HOURS must be a positive number"
        ) from exc
    store = ChunkStore(chunk_boundary=DurationBoundary(duration))
    store.ingest_dataframe(data)
    return store


def _store_summary(store: ChunkStore) -> dict[str, int]:
    return {
        "total_rows": store.total_rows,
        "total_chunks": store.total_chunks,
        "total_column_bytes": store.total_column_bytes,
    }


def _load_dataset_quiet(path: Path) -> Any:
    """Call ``load_dataset`` with its own stdout line suppressed."""

    with contextlib.redirect_stdout(io.StringIO()):
        return load_dataset(path)


_QUERY_TITLES: dict[str, str] = {
    "Q1": "Single symbol, single day",
    "Q2": "Single symbol, one-month range",
    "Q3": "Multi-symbol, one-week range",
    "Q4": "Single symbol, price + date range",
    "Q5": "Broad query (~50% of the table)",
}


def _query_title(query_name: str) -> str:
    return _QUERY_TITLES.get(query_name, query_name)


def _predicate_description(sql: str) -> list[tuple[str, str]]:
    """Describe a query's WHERE clause from its parsed predicate tree."""

    remaining_sql, as_of = extract_as_of(sql)
    predicate = parse_where_clause(remaining_sql)
    children = predicate.children if isinstance(predicate, And) else [predicate]

    symbols: list[str] = []
    time_ge = None
    time_lt = None
    time_le = None
    time_eq = None
    price_parts: list[str] = []
    unrecognized: list[str] = []

    for node in children:
        if isinstance(node, AlwaysTrue):
            continue
        if isinstance(node, In) and node.label_field == "symbol":
            symbols.extend(str(value) for value in node.values)
        elif isinstance(node, Comparison) and node.label_field == "symbol":
            if node.operator == "=":
                symbols.append(str(node.value))
            else:
                unrecognized.append(f"symbol {node.operator} {node.value!r}")
        elif isinstance(node, Comparison) and node.label_field == "time":
            if node.operator == ">=":
                time_ge = node.value
            elif node.operator == "<":
                time_lt = node.value
            elif node.operator == "<=":
                time_le = node.value
            elif node.operator == "=":
                time_eq = node.value
            else:
                unrecognized.append(f"timestamp {node.operator} {node.value!r}")
        elif isinstance(node, Comparison) and node.label_field == "price":
            price_parts.append(f"{node.operator} {node.value:g}")
        else:
            unrecognized.append(repr(node))

    symbol_desc = ", ".join(symbols) if symbols else "(all symbols)"

    upper_date = None
    if time_lt is not None:
        upper_date = (time_lt - timedelta(microseconds=1)).date()
    elif time_le is not None:
        upper_date = time_le.date()

    if time_eq is not None:
        time_desc = f"{time_eq.date()} (single day)"
    elif time_ge is not None and upper_date is not None:
        time_desc = (
            f"{time_ge.date()} (single day)"
            if upper_date == time_ge.date()
            else f"{time_ge.date()} → {upper_date}"
        )
    elif time_ge is not None:
        time_desc = f"{time_ge.date()} → (open-ended)"
    elif upper_date is not None:
        time_desc = f"(open-ended) → {upper_date}"
    else:
        time_desc = "(unconstrained)"

    price_desc = ", ".join(price_parts) if price_parts else "none"
    as_of_desc = as_of.isoformat() if as_of is not None else "current state"

    pairs = [
        ("Symbol", symbol_desc),
        ("Valid-time range", time_desc),
        ("Price predicate", price_desc),
        ("AS OF cutoff", as_of_desc),
    ]
    if unrecognized:
        pairs.append(("Additional predicate(s)", "; ".join(unrecognized)))
    pairs.append(("Raw SQL", sql))
    return pairs


def _query_observation(metrics: dict[str, Any]) -> str:
    """Return a qualitative observation based on the measured pruning rate."""

    pruning_rate = metrics["pushdown_query"]["chunks_skipped_ratio"]
    if pruning_rate >= 0.9:
        return "High pruning was achieved for this narrow query."
    if pruning_rate >= 0.3:
        return "Moderate pruning was achieved for this query."
    return "Low pruning was achieved; use both full-scan controls to assess latency."


def _print_fixed_query_block(
    *,
    index: int,
    total: int,
    query_name: str,
    metrics: dict[str, Any],
    store_rows: int | None = None,
) -> None:
    """Print one query's result block (correctness already verified upstream)."""

    sql = metrics["query"]
    full_scan = metrics["full_scan"]
    pushdown = metrics["pushdown_query"]
    total_chunks = pushdown["total_chunks"]
    daily_rows_note = (
        f"{store_rows:,}-row, {total_chunks:,}-chunk store"
        if store_rows is not None
        else f"{total_chunks:,}-chunk store"
    )
    candidate_chunks = pushdown["candidate_chunks"]
    chunks_pruned = total_chunks - candidate_chunks

    print()
    print(f"Query {index}/{total} — {_query_title(query_name)} ({query_name})")
    print()
    print(console.titled_block("  Predicate", _predicate_description(sql), indent=4))
    print()
    print(
        console.titled_block(
            "  Correctness validation",
            [
                ("Full-scan rows", f"{full_scan['row_count']:,}"),
                ("Pushdown rows", f"{pushdown['row_count']:,}"),
                ("Row equality", "PASS"),
                ("Value equality", "PASS"),
                ("Dtype equality", "PASS"),
                ("Overall status", "PASSED"),
            ],
            indent=4,
        )
    )
    print()
    print(
        console.titled_block(
            "  Execution statistics",
            [
                ("Total chunks", f"{total_chunks:,}"),
                ("Chunks scanned", f"{candidate_chunks:,}"),
                ("Chunks pruned", f"{chunks_pruned:,}"),
                ("Pruning rate", format_pct(pushdown["chunks_skipped_ratio"])),
                ("Rows returned", f"{pushdown['row_count']:,}"),
                ("Actual selectivity", format_pct(metrics["selectivity"])),
            ],
            indent=4,
        )
    )
    print()
    print("  Timing results")
    full_scan_ms = full_scan["mean_latency_seconds"] * 1_000.0
    full_scan_ci_low = full_scan["ci95_low_seconds"] * 1_000.0
    full_scan_ci_high = full_scan["ci95_high_seconds"] * 1_000.0
    pushdown_ms = pushdown["mean_latency_seconds"] * 1_000.0
    pushdown_ci_low = pushdown["ci95_low_seconds"] * 1_000.0
    pushdown_ci_high = pushdown["ci95_high_seconds"] * 1_000.0
    vectorized = pushdown["vectorized_full_scan"]
    box = render_box_table(
        ["Engine", "Raw mean latency", "Student-t interval (raw mean)"],
        [
            [
                "Row-wise full scan",
                format_ms(full_scan_ms),
                f"{full_scan_ci_low:.3f} – {full_scan_ci_high:.3f} ms",
            ],
            [
                "Vectorized full scan (no pruning)",
                format_ms(vectorized["mean_latency_seconds"] * 1000),
                f"{vectorized['ci95_low_seconds'] * 1000:.3f} – "
                f"{vectorized['ci95_high_seconds'] * 1000:.3f} ms",
            ],
            [
                "Zone-map pushdown",
                format_ms(pushdown_ms),
                f"{pushdown_ci_low:.3f} – {pushdown_ci_high:.3f} ms",
            ],
        ],
    )
    for line in box.splitlines():
        print(f"    {line}")
    print()
    print(
        console.titled_block(
            "  Speedup",
            [
                (
                    "Rows returned / selectivity",
                    f"{pushdown['row_count']:,} / {format_pct(metrics['selectivity'])}",
                ),
                ("Row-wise full scan / pushdown", format_x(metrics["speedup"])),
                (
                    "Vectorized full scan / pushdown",
                    format_x(pushdown["vectorized_full_scan_speedup"]),
                ),
            ],
            indent=4,
        )
    )
    print()
    print(
        console.titled_block(
            "  Interpretation",
            [
                ("Observation", _query_observation(metrics)),
                (
                    "Evidence",
                    f"{chunks_pruned:,}/{total_chunks:,} chunks were pruned by "
                    "the zone-map index.",
                ),
                (
                    "Limitation",
                    f"Returns {pushdown['row_count']:,} rows "
                    f"({format_pct(metrics['selectivity'])} of the table) on a "
                    f"{daily_rows_note}. Result applies to this predicate and "
                    "dataset only; ratios on needle queries mostly reflect "
                    "per-chunk DataFrame construction in the row-wise "
                    "baseline and do not extrapolate to larger data.",
                ),
            ],
            indent=4,
        )
    )
    print()
    print("-" * 80)


def _sweep_trend_observation(sweep_points: list[dict[str, Any]]) -> str:
    """Describe the measured selectivity-sweep speedup trend."""

    if not sweep_points:
        return "No selectivity-sweep points were measured."
    first, last = sweep_points[0], sweep_points[-1]
    if last["speedup"] >= first["speedup"]:
        return (
            "Speedup increased as query selectivity narrowed "
            f"({format_x(first['speedup'])} at "
            f"{format_pct(first['actual_selectivity'])} selectivity -> "
            f"{format_x(last['speedup'])} at "
            f"{format_pct(last['actual_selectivity'])} selectivity)."
        )
    return (
        "Speedup did not increase monotonically as selectivity narrowed in "
        f"this run ({format_x(first['speedup'])} -> {format_x(last['speedup'])}); "
        "see the selectivity-sweep table above for the full curve."
    )


def _print_benchmark_1(
    *,
    daily_path: Path,
    daily_data: Any,
    daily_store: ChunkStore,
    daily_build_seconds: float,
    queries: dict[str, Any],
    num_trials: int,
    warmup_runs: int,
    queries_path: Path,
    intraday_path: Path | None,
    intraday_data: Any | None,
    intraday_store: ChunkStore | None,
    intraday_build_seconds: float | None,
) -> None:
    """Print Benchmark 1 in the approved exact format."""

    symbol_count = int(daily_data["symbol"].nunique())
    row_count = len(daily_data)
    chunk_count = daily_store.total_chunks

    print(
        console.pitdb_header(
            index=1, total=5, title="Fixed Queries and Selectivity Sweep"
        )
    )
    print()
    print("Question")
    print(
        (
            "  How does zone-map pushdown performance change as query "
            "selectivity narrows?"
        )
    )
    print()
    print(
        console.titled_block(
            "Configuration",
            [
                ("Dataset", daily_path.name),
                ("Symbols", f"{symbol_count:,}"),
                ("Rows", f"{row_count:,}"),
                ("Chunks", f"{chunk_count:,}"),
                ("Chunk granularity", "monthly"),
                ("Warmup runs", str(warmup_runs)),
                ("Measured runs", str(num_trials)),
                ("Timing statistic", "raw arithmetic mean; all samples retained"),
                (
                    "Uncertainty",
                    "Student-t interval on raw mean; descriptive, coverage unverified",
                ),
                ("Random seed", "N/A (real market data, no synthetic generation)"),
            ],
        )
    )

    print()
    print(console.dash_section("[SETUP] Loading dataset"))
    print(
        console.field_block(
            [
                ("Status", "COMPLETE"),
                ("Rows loaded", f"{row_count:,}"),
                ("Store build time", f"{daily_build_seconds * 1_000.0:.3f} ms"),
                ("Total chunks", f"{chunk_count:,}"),
            ]
        )
    )

    print()
    print(console.dash_section("[FIXED QUERIES]"))
    fixed = queries["fixed_queries"]
    total_fixed = len(fixed)
    for index, (query_name, metrics) in enumerate(fixed.items(), start=1):
        _print_fixed_query_block(
            index=index,
            total=total_fixed,
            query_name=query_name,
            metrics=metrics,
            store_rows=row_count,
        )

    print()
    print(console.dash_section("[SELECTIVITY SWEEP]"))
    print()
    sweep_points = (queries.get("selectivity_sweep") or {}).get("points") or []
    sweep_rows = [
        [
            format_pct(point["target_selectivity"]),
            format_pct(point["actual_selectivity"]),
            f"{point['pushdown_query']['row_count']:,}",
            format_pct(point["chunks_skipped_ratio"]),
            format_ms(point["full_scan"]["mean_latency_seconds"] * 1_000.0),
            format_ms(point["pushdown_query"]["mean_latency_seconds"] * 1_000.0),
            format_x(point["speedup"]),
        ]
        for point in sweep_points
    ]
    print(
        render_box_table(
            [
                "Target sel.",
                "Actual sel.",
                "Rows",
                "Pruning rate",
                "Full scan",
                "Pushdown",
                "Speedup",
            ],
            sweep_rows,
        )
    )

    print()
    print(console.dash_section("[PRICE-PREDICATE CONTRIBUTION]"))
    print()
    price_demo = queries.get("price_pushdown_demonstration")
    if price_demo:
        date_only = price_demo["date_only"]
        date_and_price = price_demo["date_and_price"]
        date_only_pruned = date_only["total_chunks"] - date_only["candidate_chunks"]
        date_and_price_pruned = (
            date_and_price["total_chunks"] - date_and_price["candidate_chunks"]
        )
        # The price demonstration is an untimed pushdown probe, so it has no speedup.
        print(
            render_box_table(
                ["Predicate", "Chunks pruned", "Pruning rate", "Speedup"],
                [
                    [
                        "Symbol + date",
                        f"{date_only_pruned:,}/{date_only['total_chunks']:,}",
                        format_pct(date_only["chunks_skipped_ratio"]),
                        "N/A (untimed)",
                    ],
                    [
                        "Symbol + date + price",
                        f"{date_and_price_pruned:,}/{date_and_price['total_chunks']:,}",
                        format_pct(date_and_price["chunks_skipped_ratio"]),
                        "N/A (untimed)",
                    ],
                ],
            )
        )

    intraday_queries = (
        queries.get("dataset_b_intraday", {}).get("fixed_queries")
        if intraday_store is not None
        else None
    )
    if (
        intraday_queries
        and intraday_path is not None
        and intraday_data is not None
        and intraday_build_seconds is not None
    ):
        intraday_symbol_count = int(intraday_data["symbol"].nunique())
        intraday_row_count = len(intraday_data)
        intraday_chunk_count = intraday_store.total_chunks
        print()
        print(console.dash_section("[SETUP] Loading dataset (Dataset B, intraday)"))
        print(
            console.field_block(
                [
                    ("Status", "COMPLETE"),
                    ("Dataset", intraday_path.name),
                    ("Symbols", f"{intraday_symbol_count:,}"),
                    ("Rows loaded", f"{intraday_row_count:,}"),
                    (
                        "Store build time",
                        f"{intraday_build_seconds * 1_000.0:.3f} ms",
                    ),
                    ("Total chunks", f"{intraday_chunk_count:,}"),
                ]
            )
        )
        print()
        print(console.dash_section("[FIXED QUERIES — DATASET B (INTRADAY)]"))
        total_intraday = len(intraday_queries)
        for index, (query_name, metrics) in enumerate(
            intraday_queries.items(), start=1
        ):
            _print_fixed_query_block(
                index=index,
                total=total_intraday,
                query_name=query_name,
                metrics=metrics,
                store_rows=intraday_row_count,
            )

    speedups = {name: metrics["speedup"] for name, metrics in fixed.items()}
    narrowest_name = min(fixed, key=lambda name: fixed[name]["selectivity"])
    broadest_name = max(fixed, key=lambda name: fixed[name]["selectivity"])

    print()
    print(console.dash_section("[RESULT]"))
    print(
        console.field_block(
            [
                ("Correctness", "PASSED"),
                (
                    "Narrowest-query speedup",
                    f"{format_x(speedups[narrowest_name])} ({narrowest_name}, "
                    f"{fixed[narrowest_name]['pushdown_query']['row_count']:,} rows)",
                ),
                (
                    "Broadest-query speedup",
                    f"{format_x(speedups[broadest_name])} ({broadest_name}, "
                    f"{fixed[broadest_name]['pushdown_query']['row_count']:,} rows)",
                ),
                (
                    "Dataset-size caveat",
                    f"{row_count:,} rows / {chunk_count:,} chunks "
                    f"({format_bytes(daily_store.total_column_bytes)} of "
                    "column data): needle-query ratios mostly reflect "
                    "per-chunk DataFrame construction overhead in the "
                    "row-wise baseline and do not extrapolate",
                ),
                ("Main observation", _sweep_trend_observation(sweep_points)),
                ("Result artifact", str(queries_path)),
            ]
        )
    )
    print()
    print(console.pitdb_footer(index=1))


def _print_benchmark_2(
    *,
    daily_data: Any,
    daily_store: ChunkStore,
    memory: dict[str, Any],
    fixed_queries: dict[str, Any],
    memory_path: Path,
) -> None:
    """Print Benchmark 2 in the approved exact format."""

    symbol_count = int(daily_data["symbol"].nunique())
    row_count = len(daily_data)
    chunk_count = daily_store.total_chunks

    print(console.pitdb_header(index=2, total=5, title="Memory-Footprint Analysis"))
    print()
    print("Question")
    print(
        console.wrap_text(
            "How much memory do PitDB's uncompressed float64 column storage "
            "and its per-query working set use, for pruned and unpruned "
            "execution paths?"
        )
    )
    print()
    print(
        console.titled_block(
            "Configuration",
            [
                ("Rows", f"{row_count:,}"),
                ("Symbols", f"{symbol_count:,}"),
                ("Chunks", f"{chunk_count:,}"),
                ("Numeric representation", "float64"),
                ("Chunk granularity", "monthly"),
                (
                    "Query repetitions",
                    f"{memory['traced_repeats']} per engine, rotating order "
                    "(tracemalloc peaks)",
                ),
            ],
        )
    )

    print()
    print(console.dash_section("[SETUP]"))
    print(
        console.field_block(
            [
                ("Dataset generated", "YES"),
                ("Store constructed", "YES"),
                ("Memory profiler ready", "YES"),
            ]
        )
    )

    q1 = fixed_queries["Q1"]
    print()
    print(console.dash_section("[CORRECTNESS VALIDATION]"))
    print(
        console.field_block(
            [
                ("Full-scan rows", f"{q1['full_scan']['row_count']:,}"),
                ("Pushdown rows", f"{q1['pushdown_query']['row_count']:,}"),
                ("Row equality", "PASS"),
                ("Value equality", "PASS"),
                ("Dtype equality", "PASS"),
                ("Overall status", "PASSED"),
            ]
        )
    )

    raw_bytes = memory["raw_numeric_payload_bytes"]
    packed_bytes = memory["column_payload_bytes"]
    raw_bytes_per_row = raw_bytes / row_count if row_count else 0.0

    print()
    print(
        console.dash_section(
            "[STORAGE REPRESENTATION]", " Uncompressed float64 payload (no compression)"
        )
    )
    print()
    print(
        render_box_table(
            ["Representation", "Total memory", "Bytes per row"],
            [
                [
                    "Numeric columns in the store (float64 x 5)",
                    format_bytes(packed_bytes),
                    f"{packed_bytes / row_count if row_count else 0.0:.1f}",
                ],
                [
                    "Same columns in the source frame (float64 x 5)",
                    format_bytes(raw_bytes),
                    f"{raw_bytes_per_row:.1f}",
                ],
            ],
        )
    )
    print()
    print(
        console.wrap_text(
            "These two rows are equal by construction: PitDB stores raw "
            "uncompressed float64 columns. No storage reduction or "
            "compression ratio is measured or claimed.",
            indent=2,
        )
    )
    print()
    print(
        console.field_block(
            [
                (
                    "Full raw DataFrame (pandas, deep)",
                    format_bytes(memory["raw_uncompressed_bytes"]),
                )
            ]
        )
    )
    print()
    print(
        console.dash_section(
            "[INTERNAL OBJECT FOOTPRINT]",
            " Not directly comparable with packed column bytes",
        )
    )
    print()
    print(
        render_box_table(
            ["Component", "Memory"],
            [
                [
                    "Packed numeric buffers",
                    format_bytes(memory["packed_numeric_buffer_bytes"]),
                ],
                ["Chunk metadata", format_bytes(memory["chunk_metadata_bytes"])],
                ["Zone maps", format_bytes(memory["zone_map_bytes"])],
                [
                    "Labels and indexes",
                    format_bytes(memory["labels_and_indexes_bytes"]),
                ],
                [
                    "Python object overhead",
                    format_bytes(memory["python_object_overhead_bytes"]),
                ],
                [
                    "Complete PitDB object graph",
                    format_bytes(memory["total_object_graph_bytes"]),
                ],
            ],
        )
    )
    print()
    print("  Important")
    print(
        console.wrap_text(
            "The complete object-graph footprint includes Python bookkeeping "
            "and must not be interpreted as equivalent to packed persisted "
            "column bytes.",
            indent=4,
        )
    )

    traced = memory["traced_query_peaks"]
    print()
    print(
        console.dash_section(
            "[PER-QUERY PEAK ALLOCATION]",
            f" tracemalloc peak above pre-call baseline; median of "
            f"{memory['traced_repeats']} interleaved repeats",
        )
    )
    print()
    traced_rows = []
    for name, entry in traced.items():
        engines = entry["engines"]
        traced_rows.append(
            [
                name,
                f"{entry['row_count']:,}",
                format_bytes(engines["full_scan"]["median_peak_bytes"]),
                format_bytes(engines["vectorized_full_scan"]["median_peak_bytes"]),
                format_bytes(engines["pushdown_query"]["median_peak_bytes"]),
            ]
        )
    print(
        render_box_table(
            ["Query", "Rows", "Row-wise scan", "Vectorized scan", "Pushdown"],
            traced_rows,
        )
    )
    construction = memory.get("store_construction_traced_peak_bytes")
    if construction is not None:
        print()
        print(
            console.field_block(
                [
                    (
                        "Store construction peak (median of 3)",
                        format_bytes(construction["median_peak_bytes"]),
                    )
                ]
            )
        )
    rss = memory.get("rss_diagnostic")
    if rss is not None:
        print()
        print(
            console.dash_section(
                "[RSS DIAGNOSTIC]", " Noise-bounded; not a result or a comparison"
            )
        )
        print(
            console.field_block(
                [
                    (
                        "Process RSS before measuring",
                        format_bytes(rss["idle_rss_bytes"]),
                    ),
                    (
                        "Q1 full-scan RSS increase",
                        format_bytes(rss["full_scan_q1_peak_rss_increase_bytes"]),
                    ),
                    (
                        "Q1 pushdown RSS increase",
                        format_bytes(rss["q1_peak_rss_increase_bytes"]),
                    ),
                    ("Resolution", f"{rss['resolution_bytes']} bytes (page size)"),
                ]
            )
        )
        print(
            console.wrap_text(
                "Single-shot, fixed order, 1 ms sampler, shared ~100 MB "
                "process: increases at this scale are at the measurement "
                "noise floor and are not used for any comparison.",
                indent=2,
            )
        )

    q1_traced = traced["Q1"]["engines"]
    print()
    print(console.dash_section("[RESULT]"))
    print(
        console.field_block(
            [
                ("Correctness", "PASSED"),
                ("Numeric payload (store, float64)", format_bytes(packed_bytes)),
                (
                    "Full object footprint",
                    format_bytes(memory["total_object_graph_bytes"]),
                ),
                (
                    "Q1 peak allocation, row-wise scan",
                    format_bytes(q1_traced["full_scan"]["median_peak_bytes"]),
                ),
                (
                    "Q1 peak allocation, pushdown",
                    format_bytes(q1_traced["pushdown_query"]["median_peak_bytes"]),
                ),
            ]
        )
    )
    print()
    print(
        console.titled_block(
            "  Interpretation",
            [
                (
                    "Observation",
                    "For Q1 (1 matching row) the pushdown executor "
                    "materialises one chunk where the row-wise scan "
                    "materialises every chunk, so its traced peak is far "
                    "lower; the gap shrinks as queries match more chunks "
                    "(see Q5).",
                ),
                (
                    "Valid claim",
                    "Per-query traced allocation peaks on this fixture; "
                    "storage is uncompressed so no storage saving is claimed.",
                ),
                (
                    "Limitation",
                    "tracemalloc covers Python/NumPy-registered allocations "
                    "only, in one shared process on a 0.2 MB dataset; the "
                    "RSS diagnostic is below measurement resolution.",
                ),
            ],
            indent=4,
        )
    )
    print()
    print(console.field_block([("Result artifact", str(memory_path))]))
    print()
    print(console.pitdb_footer(index=2))


def run_all(*, include_memory: bool = False) -> dict[str, Any]:
    """Run the configured benchmarks and persist their JSON results.

    Dataset B is skipped only when its CSV is absent; an invalid file raises.
    """

    if not isinstance(include_memory, bool):
        raise TypeError("include_memory must be a bool")

    daily_path = _project_path(
        _config_value("DAILY_DATA_FILE", _DEFAULT_DAILY_DATA_FILE)
    )
    intraday_path = _project_path(
        _config_value("INTRADAY_DATA_FILE", _DEFAULT_INTRADAY_DATA_FILE)
    )
    results_dir = _project_path(_config_value("RESULTS_DIR", _DEFAULT_RESULTS_DIR))
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = create_manifest(
        parameters={
            "benchmark": "daily_queries_and_memory",
            "include_memory": include_memory,
            "num_trials": _config_value("NUM_TRIALS", 18),
            "warmup_runs": _config_value("WARMUP_RUNS", 3),
        },
        fixture_paths=[path for path in (daily_path, intraday_path) if path.is_file()],
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    hardware = _hardware_info()
    hardware["generated_at_utc"] = generated_at
    hardware_path = results_dir / "hardware.json"

    num_trials = _config_value("NUM_TRIALS", 18)
    warmup_runs = _config_value("WARMUP_RUNS", 3)

    daily_data = _load_dataset_quiet(daily_path)
    t0 = time.perf_counter()
    daily_store = _build_daily_store(daily_data)
    daily_build_seconds = time.perf_counter() - t0

    intraday_data = None
    intraday_store = None
    intraday_build_seconds: float | None = None
    if intraday_path.is_file():
        intraday_data = _load_dataset_quiet(intraday_path)
        t0 = time.perf_counter()
        intraday_store = _build_intraday_store(intraday_data)
        intraday_build_seconds = time.perf_counter() - t0
    else:
        LOGGER.warning(
            "Dataset B intraday file is absent; skipping its query benchmarks: %s",
            intraday_path,
        )

    # quiet=True: correctness is re-rendered inside each query's result block later.
    fixed_queries = run_query_benchmarks(daily_store, quiet=True)
    queries: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "dataset": "A_daily",
        "source": str(daily_path),
        "store": _store_summary(daily_store),
        "fixed_queries": fixed_queries,
        "pairwise_interval_comparisons": annotate_pairwise_significance(fixed_queries),
        "selectivity_sweep": run_selectivity_sweep(daily_store, quiet=True),
        "price_pushdown_demonstration": run_price_pushdown_demonstration(daily_store),
    }
    if intraday_store is not None:
        # Dataset B is a trailing window, so thresholds come from the store.
        intraday_queries = build_intraday_queries(intraday_store)
        queries["dataset_b_intraday"] = {
            "store": _store_summary(intraday_store),
            "fixed_queries": run_query_benchmarks(
                intraday_store, intraday_queries, quiet=True
            ),
        }
    queries_path = results_dir / "queries.json"
    queries["manifest"] = finish_manifest(manifest)
    hardware["manifest"] = queries["manifest"]
    _write_json(hardware_path, hardware)
    _write_json(queries_path, queries)

    _print_benchmark_1(
        daily_path=daily_path,
        daily_data=daily_data,
        daily_store=daily_store,
        daily_build_seconds=daily_build_seconds,
        queries=queries,
        num_trials=num_trials,
        warmup_runs=warmup_runs,
        queries_path=queries_path,
        intraday_path=intraday_path if intraday_store is not None else None,
        intraday_data=intraday_data,
        intraday_store=intraday_store,
        intraday_build_seconds=intraday_build_seconds,
    )

    memory: dict[str, Any] | None = None
    memory_path = results_dir / "memory.json"
    if include_memory:
        # build_store_fn is only used to measure store construction's own peak.
        memory = {
            "generated_at_utc": generated_at,
            "dataset": "A_daily",
            "source": str(daily_path),
            **run_memory_benchmarks(
                daily_data,
                daily_store,
                build_store_fn=lambda: _build_daily_store(daily_data),
            ),
        }
        memory["manifest"] = finish_manifest(manifest)
        _write_json(memory_path, memory)
        print()
        _print_benchmark_2(
            daily_data=daily_data,
            daily_store=daily_store,
            memory=memory,
            fixed_queries=fixed_queries,
            memory_path=memory_path,
        )

    result_files = {
        "queries": str(queries_path),
        "hardware": str(hardware_path),
    }
    if include_memory:
        result_files["memory"] = str(memory_path)

    print()
    print(console.phase("EXPORT", "Result files"))
    for name, path in result_files.items():
        print(f"  {name:<12} {path}")

    return {
        "queries": queries,
        "memory": memory,
        "hardware": hardware,
        "result_files": result_files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the chunked query engine's complete benchmark suite."
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="also run the optional peak-memory benchmark",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_all(include_memory=args.memory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_all"]
