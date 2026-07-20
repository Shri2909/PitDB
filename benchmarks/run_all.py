"""Master runner for the chunked query engine's query and memory benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
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
from benchmarks.report import format_ms, format_pct, format_x, render_table
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


def _print_summary(
    queries: dict[str, Any],
    memory: dict[str, Any] | None,
    result_files: dict[str, str],
) -> None:
    print("\nChunked Query Engine Benchmark Summary")
    print("=" * 96)

    print("Queries - Dataset A (daily)")
    headers = [
        "Query",
        "Full-scan",
        "Pushdown (95% CI)",
        "Speedup",
        "Selectivity",
        "Skipped",
    ]
    rows = []
    for name, metrics in queries["fixed_queries"].items():
        full_scan_ms = metrics["full_scan"]["mean_latency_seconds"] * 1_000.0
        pushdown = metrics["pushdown_query"]
        pushdown_ms = pushdown["mean_latency_seconds"] * 1_000.0
        ci_low = pushdown["ci95_low_seconds"] * 1_000.0
        ci_high = pushdown["ci95_high_seconds"] * 1_000.0
        rows.append(
            [
                name,
                format_ms(full_scan_ms),
                f"{format_ms(pushdown_ms)} [{ci_low:.3f},{ci_high:.3f}]",
                format_x(metrics["speedup"]),
                format_pct(metrics["selectivity"]),
                format_pct(pushdown["chunks_skipped_ratio"]),
            ]
        )
    print(render_table(headers, rows))

    intraday_queries = queries.get("dataset_b_intraday")
    if intraday_queries:
        print("\nQueries - Dataset B (intraday)")
        intraday_rows = []
        for name, metrics in intraday_queries["fixed_queries"].items():
            full_scan_ms = metrics["full_scan"]["mean_latency_seconds"] * 1_000.0
            pushdown = metrics["pushdown_query"]
            pushdown_ms = pushdown["mean_latency_seconds"] * 1_000.0
            intraday_rows.append(
                [
                    name,
                    format_ms(full_scan_ms),
                    format_ms(pushdown_ms),
                    format_x(metrics["speedup"]),
                    format_pct(metrics["selectivity"]),
                    format_pct(pushdown["chunks_skipped_ratio"]),
                ]
            )
        print(render_table(headers[:2] + ["Pushdown"] + headers[3:], intraday_rows))

    pairwise = queries.get("pairwise_significance") or []
    if pairwise:
        print(
            "\nStatistical significance: consecutive fixed queries, "
            "pushdown-query mean latency (95% confidence)"
        )
        for comparison in pairwise:
            verdict = (
                "DISTINGUISHABLE (a real difference)"
                if comparison["distinguishable_from_noise"]
                else "NOT distinguishable (likely measurement noise)"
            )
            print(
                f"  {comparison['query_a']} ({comparison['a_mean_ms']:.3f}ms) vs "
                f"{comparison['query_b']} ({comparison['b_mean_ms']:.3f}ms): {verdict}"
            )

    price_demo = queries.get("price_pushdown_demonstration")
    if price_demo:
        print("\nPrice-pushdown demonstration: AAPL, 2024, isolating the price clause")
        date_only = price_demo["date_only"]
        date_and_price = price_demo["date_and_price"]
        print(
            f"  date only:          {date_only['candidate_chunks']}/"
            f"{date_only['total_chunks']} candidate chunks, "
            f"{date_only['row_count']} rows"
        )
        print(
            f"  date AND price>200: {date_and_price['candidate_chunks']}/"
            f"{date_and_price['total_chunks']} candidate chunks, "
            f"{date_and_price['row_count']} rows"
        )
        print(
            "  price clause pruned "
            f"{price_demo['price_clause_incremental_chunks_pruned']} "
            "additional chunk(s) beyond the date range alone"
        )

    if memory is not None:
        print("\nMemory")
        # raw_uncompressed_bytes (pandas' own memory_usage(deep=True)) and
        # chunk_store_bytes (a full Python object-graph walk) use different
        # accounting methods and are not a fair compression comparison --
        # column_payload_bytes (the actual packed float64 bytes ChunkStore
        # persists) vs. raw is the apples-to-apples one. See
        # charts/chart_memory.py's _METRICS comment for the full reasoning.
        print(
            f"raw DataFrame (pandas)={memory['raw_uncompressed_bytes']:,} bytes | "
            f"column payload (packed)={memory['column_payload_bytes']:,} bytes | "
            f"chunk store (object graph)={memory['chunk_store_bytes']:,} bytes | "
            f"Q1 peak RSS={memory['q1_peak_rss_bytes']:,} bytes"
        )

    print("\nResult files")
    for name, path in result_files.items():
        print(f"  {name:<12} {path}")


def run_all(*, include_memory: bool = False) -> dict[str, Any]:
    """Run configured benchmarks and persist their structured JSON results.

    Dataset A is mandatory. Dataset B is optional only when its merged CSV is
    absent; a present but invalid intraday file raises instead of silently
    producing misleading query-benchmark inputs.
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

    print(f"Loading Dataset A (daily): {daily_path}")
    daily_data = load_dataset(daily_path)
    daily_store = _build_daily_store(daily_data)

    intraday_data = None
    intraday_store = None
    if intraday_path.is_file():
        print(f"Loading Dataset B (intraday): {intraday_path}")
        intraday_data = load_dataset(intraday_path)
        intraday_store = _build_intraday_store(intraday_data)
    else:
        LOGGER.warning(
            "Dataset B intraday file is absent; skipping its query benchmarks: %s",
            intraday_path,
        )

    print("Running query benchmarks and selectivity sweep...")
    fixed_queries = run_query_benchmarks(daily_store)
    queries: dict[str, Any] = {
        "generated_at_utc": generated_at,
        "dataset": "A_daily",
        "source": str(daily_path),
        "store": _store_summary(daily_store),
        "fixed_queries": fixed_queries,
        # Flags which consecutive fixed queries' pushdown-query latencies are
        # (or are not) distinguishable from noise at 95% confidence, rather
        # than letting the summary table imply every mean difference is a
        # real effect.
        "pairwise_significance": annotate_pairwise_significance(fixed_queries),
        "selectivity_sweep": run_selectivity_sweep(daily_store),
        "price_pushdown_demonstration": run_price_pushdown_demonstration(daily_store),
    }
    if intraday_store is not None:
        # Thresholds are derived from the store's own labels
        # (build_intraday_queries), since Dataset B's date range is a
        # trailing window relative to fetch time, not a fixed history.
        intraday_queries = build_intraday_queries(intraday_store)
        queries["dataset_b_intraday"] = {
            "store": _store_summary(intraday_store),
            "fixed_queries": run_query_benchmarks(intraday_store, intraday_queries),
        }
    queries_path = results_dir / "queries.json"
    _write_json(queries_path, queries)

    memory: dict[str, Any] | None = None
    memory_path = results_dir / "memory.json"
    if include_memory:
        print("Running optional memory benchmarks...")
        memory = {
            "generated_at_utc": generated_at,
            "dataset": "A_daily",
            "source": str(daily_path),
            **run_memory_benchmarks(daily_data, daily_store),
        }
        _write_json(memory_path, memory)

    result_files = {
        "queries": str(queries_path),
        "hardware": str(hardware_path),
    }
    if include_memory:
        result_files["memory"] = str(memory_path)

    _print_summary(queries, memory, result_files)
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
