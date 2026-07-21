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

# ``python benchmarks/run_all.py`` makes benchmarks/ the first import root.
# Add the repository root so that both that documented invocation and
# ``python -m benchmarks.run_all`` resolve the same project modules.
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
    # The individual benchmark modules have the same staged-build fallback.
    # Once benchmarks/config.py is added, its values take precedence here.
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
        # Additive fields -- every key above this line is unchanged from
        # this function's prior return value; logical/physical core counts
        # and the git commit are new, already-available provenance, not new
        # measurements. See BENCHMARK_OUTPUT_AUDIT.md Section 9, item 1.
        "logical_cpu_count": psutil_module.cpu_count(logical=True),
        "physical_cpu_count": psutil_module.cpu_count(logical=False),
        "git_commit": git_commit(),
    }


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write strict, portable JSON without partial result files."""

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
    """Call the unmodified ``load_dataset`` without its own stdout line.

    ``src/replay/data_loader.py:load_dataset`` always prints a
    ``"Loaded N rows | symbols (...) | date range: ..."`` line -- useful
    standalone, but this benchmark's exact-format header/Configuration/
    [SETUP] block (below) already reports row count, symbol count, and date
    range in its own place, from the same returned DataFrame. Suppressing
    the duplicate here changes nothing about what ``load_dataset`` does,
    validates, or returns -- only which of two now-redundant places prints
    the load confirmation.
    """

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
    """Decompose one query's WHERE clause into human-readable fields.

    Walks the same parsed predicate tree (``parse_where_clause``/
    ``extract_as_of``) the query engine itself executes against this exact
    SQL string -- not a separate regex or string parse -- so this
    description can never disagree with what the query actually does. Any
    clause shape this project's bounded grammar allows but this summarizer
    doesn't specifically recognize is listed under "Additional predicate(s)"
    rather than silently dropped, and the raw SQL is always included too.
    """

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
    """Return a qualitative pruning observation grounded in this query's own
    measured pruning rate -- not a fixed claim independent of the data."""

    pruning_rate = metrics["pushdown_query"]["chunks_skipped_ratio"]
    if pruning_rate >= 0.9:
        return "High pruning was achieved for this narrow query."
    if pruning_rate >= 0.3:
        return "Moderate pruning was achieved for this query."
    return (
        "Low pruning was achieved; this query needs most of the table, so a "
        "small speedup is expected here, not a regression."
    )


def _print_fixed_query_block(
    *, index: int, total: int, query_name: str, metrics: dict[str, Any]
) -> None:
    """Print one query's full exact-format result block.

    ``metrics`` is one entry of ``run_query_benchmarks``'s returned dict --
    already fully computed (including the correctness check, run with
    ``quiet=True`` by the caller) before this function is ever called.
    Reaching this function at all is only possible if that correctness
    check passed: a failure raises ``AssertionError`` from inside
    ``run_query_benchmarks`` and aborts the run before any query's block --
    including this one -- is printed. "Overall status: PASSED" below is
    therefore a truthful restatement of an already-verified outcome, not a
    second, independent check.
    """

    sql = metrics["query"]
    full_scan = metrics["full_scan"]
    pushdown = metrics["pushdown_query"]
    total_chunks = pushdown["total_chunks"]
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
    box = render_box_table(
        ["Engine", "Mean latency", "95% confidence interval"],
        [
            [
                "Full scan",
                format_ms(full_scan_ms),
                f"{full_scan_ci_low:.3f} – {full_scan_ci_high:.3f} ms",
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
            [("Full scan / pushdown", format_x(metrics["speedup"]))],
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
                ("Limitation", "Result applies to this predicate and dataset only."),
            ],
            indent=4,
        )
    )
    print()
    print("-" * 80)


def _sweep_trend_observation(sweep_points: list[dict[str, Any]]) -> str:
    """Describe the measured selectivity-sweep speedup trend.

    Computed from this run's own sweep points, not asserted in advance --
    see benchmarks/bench_bitemporal.py's ``_bitemporal_trend_observation``
    for why a fixed-direction claim independent of the actual data is a
    real communication risk this suite specifically avoids elsewhere.
    """

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
    """Print Benchmark 1 in full, start to finish, in the approved format.

    Nothing is printed before this function runs (dataset load/store-build
    happen silently first -- see ``_load_dataset_quiet`` and ``run_all``),
    so this header is always the first output the benchmark produces.
    """

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
        "  How does zone-map pushdown performance change as query selectivity narrows?"
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
                ("Timing statistic", "outlier-trimmed mean"),
                ("Confidence interval", "95%"),
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
            index=index, total=total_fixed, query_name=query_name, metrics=metrics
        )

    print()
    print(console.dash_section("[SELECTIVITY SWEEP]"))
    print()
    sweep_points = (queries.get("selectivity_sweep") or {}).get("points") or []
    sweep_rows = [
        [
            format_pct(point["target_selectivity"]),
            format_pct(point["actual_selectivity"]),
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
        # Speedup: this demonstration only ever ran an untimed pushdown
        # probe (see run_price_pushdown_demonstration) to isolate the price
        # clause's incremental chunk pruning -- there is no full-scan timing
        # comparison for it, so no speedup number exists to report. Marked
        # N/A rather than reusing an unrelated query's speedup.
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
            )

    speedups = {name: metrics["speedup"] for name, metrics in fixed.items()}
    narrowest_name = max(speedups, key=lambda name: speedups[name])
    broadest_name = min(speedups, key=lambda name: speedups[name])

    print()
    print(console.dash_section("[RESULT]"))
    print(
        console.field_block(
            [
                ("Correctness", "PASSED"),
                (
                    "Narrowest-query speedup",
                    f"{format_x(speedups[narrowest_name])} ({narrowest_name})",
                ),
                (
                    "Broadest-query speedup",
                    f"{format_x(speedups[broadest_name])} ({broadest_name})",
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
    """Print Benchmark 2 in full, in the approved format.

    ``fixed_queries`` is Benchmark 1's already-computed, already-validated
    Q1 result (same store, same query) -- reused here for the correctness
    block rather than re-running a second, redundant full-scan/pushdown
    comparison: Q1's correctness was already established once, in
    Benchmark 1, and reaching this function at all requires that check to
    have passed (see run_all()).
    """

    symbol_count = int(daily_data["symbol"].nunique())
    row_count = len(daily_data)
    chunk_count = daily_store.total_chunks

    print(console.pitdb_header(index=2, total=5, title="Memory-Footprint Analysis"))
    print()
    print("Question")
    print(
        console.wrap_text(
            "How memory-efficient is PitDB's uncompressed columnar storage "
            "compared with equivalent in-memory representations?"
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
                    "1 (RSS sampled continuously during a single execution)",
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

    raw_bytes = memory["raw_uncompressed_bytes"]
    packed_bytes = memory["column_payload_bytes"]
    raw_bytes_per_row = raw_bytes / row_count if row_count else 0.0
    packed_bytes_per_row = packed_bytes / row_count if row_count else 0.0
    packed_over_raw = packed_bytes / raw_bytes if raw_bytes else float("inf")
    storage_reduction = 1.0 - packed_over_raw

    print()
    print(
        console.dash_section(
            "[STORAGE REPRESENTATION]", " Directly comparable storage measurements"
        )
    )
    print()
    print(
        render_box_table(
            ["Representation", "Total memory", "Bytes per row"],
            [
                [
                    "Raw DataFrame columns",
                    format_bytes(raw_bytes),
                    f"{raw_bytes_per_row:.1f}",
                ],
                [
                    "PitDB packed column storage",
                    format_bytes(packed_bytes),
                    f"{packed_bytes_per_row:.1f}",
                ],
            ],
        )
    )
    print()
    print(
        console.titled_block(
            "  Packed-storage ratio",
            [
                ("PitDB / raw DataFrame", format_x(packed_over_raw)),
                ("Storage reduction", format_pct(storage_reduction)),
            ],
            indent=4,
        )
    )
    print()
    print(
        console.titled_block(
            "  Comparison status",
            [
                ("Valid comparison", "YES"),
                ("Reason", "Both measure raw column storage."),
            ],
            indent=4,
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

    idle_rss = memory["idle_rss_bytes"]
    construction_peak = memory.get("store_construction_peak_rss_bytes")
    full_scan_peak = memory["full_scan_q1_peak_rss_bytes"]
    pushdown_peak = memory["q1_peak_rss_bytes"]

    print()
    print(console.dash_section("[PEAK PROCESS MEMORY]"))
    print()
    rss_rows = [["Process idle", format_bytes(idle_rss), "—"]]
    if construction_peak is not None:
        rss_rows.append(
            [
                "Store construction",
                format_bytes(construction_peak),
                format_bytes(max(0, construction_peak - idle_rss)),
            ]
        )
    rss_rows.append(
        [
            "Full-scan query",
            format_bytes(full_scan_peak),
            format_bytes(max(0, full_scan_peak - idle_rss)),
        ]
    )
    rss_rows.append(
        [
            "Pushdown query",
            format_bytes(pushdown_peak),
            format_bytes(max(0, pushdown_peak - idle_rss)),
        ]
    )
    print(render_box_table(["Operation", "Peak RSS", "Increase from idle"], rss_rows))

    print()
    print(console.dash_section("[RESULT]"))
    print(
        console.field_block(
            [
                ("Correctness", "PASSED"),
                ("Packed storage", format_bytes(packed_bytes)),
                ("Raw DataFrame storage", format_bytes(raw_bytes)),
                ("Storage reduction", format_pct(storage_reduction)),
                (
                    "Full object footprint",
                    format_bytes(memory["total_object_graph_bytes"]),
                ),
                ("Peak query RSS", format_bytes(pushdown_peak)),
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
                    f"Packed storage uses {format_pct(storage_reduction)} less "
                    "memory than the equivalent raw DataFrame representation.",
                ),
                (
                    "Valid claim",
                    "Packed storage is compared only with raw table storage.",
                ),
                ("Limitation", "Python object overhead is reported separately."),
            ],
            indent=4,
        )
    )
    print()
    print(console.field_block([("Result artifact", str(memory_path))]))
    print()
    print(console.pitdb_footer(index=2))


def run_all(*, include_memory: bool = False) -> dict[str, Any]:
    """Run configured benchmarks and persist their structured JSON results.

    Dataset A is mandatory. Dataset B is optional only when its merged CSV is
    absent; a present but invalid intraday file raises instead of silently
    producing misleading query-benchmark inputs.

    Benchmark 1's dataset load/store-build/query computation all happen
    silently before its own header is printed (see ``_load_dataset_quiet``
    and the ``quiet=True`` calls below), so its exact-format Configuration
    block can show real symbol/row/chunk counts instead of placeholders, and
    so its header is always the very first line this function prints.
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

    generated_at = datetime.now(timezone.utc).isoformat()
    hardware = _hardware_info()
    hardware["generated_at_utc"] = generated_at
    hardware_path = results_dir / "hardware.json"
    _write_json(hardware_path, hardware)

    num_trials = _config_value("NUM_TRIALS", 15)
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

    # quiet=True: Benchmark 1's per-query correctness outcome is re-rendered
    # inside its own result block (_print_fixed_query_block) once printing
    # starts below, instead of live during this still-silent computation
    # phase -- see benchmarks/reporting/validation.py's `quiet` parameter.
    fixed_queries = run_query_benchmarks(daily_store, quiet=True)
    queries: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "dataset": "A_daily",
        "source": str(daily_path),
        "store": _store_summary(daily_store),
        "fixed_queries": fixed_queries,
        # Flags which consecutive fixed queries' pushdown-query latencies are
        # (or are not) distinguishable from noise at 95% confidence. Still
        # computed and persisted to queries.json even though Benchmark 1's
        # exact-format terminal output does not currently print this table.
        "pairwise_significance": annotate_pairwise_significance(fixed_queries),
        "selectivity_sweep": run_selectivity_sweep(daily_store, quiet=True),
        "price_pushdown_demonstration": run_price_pushdown_demonstration(daily_store),
    }
    if intraday_store is not None:
        # Thresholds are derived from the store's own labels
        # (build_intraday_queries), since Dataset B's date range is a
        # trailing window relative to fetch time, not a fixed history.
        intraday_queries = build_intraday_queries(intraday_store)
        queries["dataset_b_intraday"] = {
            "store": _store_summary(intraday_store),
            "fixed_queries": run_query_benchmarks(
                intraday_store, intraday_queries, quiet=True
            ),
        }
    queries_path = results_dir / "queries.json"
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
        # Silent, like Benchmark 1: nothing prints before Benchmark 2's own
        # header. build_store_fn lets run_memory_benchmarks measure "store
        # construction"'s own peak RSS via a second, redundant build from
        # the same already-loaded daily_data -- daily_store itself (already
        # built) is what every other measurement below uses.
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
