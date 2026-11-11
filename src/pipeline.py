"""End-to-end ingestion, query, benchmark, and chart orchestration."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from src.query.executor import QueryExecutor, QueryResult
from src.query.sql_parser import (
    UnsupportedQueryError,
    extract_as_of,
    parse_where_clause,
)
from src.replay.data_loader import load_dataset
from src.replay.replay_harness import ReplayHarness
from src.store.chunk_boundary import (
    ChunkBoundary,
    DurationBoundary,
    MonthlyBoundary,
)
from src.store.chunk_store import ChunkStore


def _normalize_queries(queries: list[str] | None) -> list[str]:
    if queries is None:
        return []
    if not isinstance(queries, list):
        raise TypeError("queries must be a list of SQL strings or None")

    normalized: list[str] = []
    for index, query in enumerate(queries):
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"queries[{index}] must be a non-empty SQL string")
        normalized.append(query)
    return normalized


def _raw_value_bytes(store: ChunkStore, row_count: int) -> int:
    """Return the raw float64 payload size for the store's value columns."""

    return row_count * len(store.value_columns) * 8


def _raw_to_stored_ratio(raw_bytes: int, stored_bytes: int) -> float:
    """Raw float64 payload bytes over stored bytes (~1.0x: a size sanity check)."""

    if raw_bytes == 0:
        return 0.0
    if stored_bytes <= 0:
        raise RuntimeError("a non-empty input produced no stored column bytes")
    return raw_bytes / stored_bytes


def _print_store_summary(summary: dict[str, int | float | str]) -> None:
    print("\nChunk Store Summary")
    print("=" * 56)
    print(f"Chunk strategy:          {summary['chunk_boundary']}")
    print(f"Total chunks:            {summary['total_chunks']:,}")
    print(f"Total rows:              {summary['total_rows']:,}")
    print(f"Raw float64 bytes:       {summary['raw_value_bytes']:,}")
    print(f"Total column bytes:      {summary['total_column_bytes']:,}")
    print(f"Raw/stored byte ratio:   {summary['raw_to_stored_ratio']:.3f}x")


def _print_query_result(index: int, sql: str, result: QueryResult) -> None:
    print(f"\nQuery {index}: {sql}")
    print(f"  rows:           {result.row_count:,}")
    print(f"  elapsed:        {result.elapsed_seconds:.6f} seconds")
    print(f"  selectivity:    {result.selectivity:.3%}")
    print(
        "  chunks skipped: "
        f"{len(result.pushdown.skipped_chunk_ids):,}/"
        f"{result.pushdown.total_chunks:,} "
        f"({result.pushdown.chunks_skipped_ratio:.3%})"
    )


def run_pipeline(
    data_csv_path: Path,
    chunk_boundary: ChunkBoundary | None = None,
    queries: list[str] | None = None,
) -> dict[str, Any]:
    """Load, sequentially ingest, optionally query, and return all results.

    Defaults to ``MonthlyBoundary`` (daily data); dense one-minute data should
    pass an hourly ``DurationBoundary``.
    """

    if not isinstance(data_csv_path, Path):
        raise TypeError("data_csv_path must be a pathlib.Path")
    selected_boundary = MonthlyBoundary() if chunk_boundary is None else chunk_boundary
    if not isinstance(selected_boundary, ChunkBoundary):
        raise TypeError("chunk_boundary must be a ChunkBoundary or None")
    sql_queries = _normalize_queries(queries)

    source_path = data_csv_path.expanduser().resolve(strict=False)
    dataframe = load_dataset(source_path)

    # The harness exposes rows in source order only: no length, indexing or lookahead.
    harness = ReplayHarness(dataframe)
    store = ChunkStore(chunk_boundary=selected_boundary)
    store.ingest_from_replay(harness)

    raw_bytes = _raw_value_bytes(store, len(dataframe))
    column_bytes = store.total_column_bytes
    summary: dict[str, int | float | str] = {
        "data_csv_path": str(source_path),
        "chunk_boundary": type(selected_boundary).__name__,
        "total_chunks": store.total_chunks,
        "total_rows": store.total_rows,
        "raw_value_bytes": raw_bytes,
        "total_column_bytes": column_bytes,
        "raw_to_stored_ratio": _raw_to_stored_ratio(raw_bytes, column_bytes),
    }
    _print_store_summary(summary)

    query_results: list[dict[str, Any]] = []
    if sql_queries:
        executor = QueryExecutor(store)
        for index, sql in enumerate(sql_queries, start=1):
            result = executor.execute(sql)
            _print_query_result(index, sql, result)
            query_results.append({"sql": sql, "result": result})

    # The source DataFrame is deliberately not returned.
    return {
        "summary": summary,
        "store": store,
        "query_results": query_results,
    }


def _boundary_from_cli(strategy: str) -> ChunkBoundary:
    if strategy == "monthly":
        return MonthlyBoundary()
    if strategy == "hourly":
        return DurationBoundary(timedelta(hours=1))
    raise ValueError(f"unsupported chunk strategy: {strategy}")


def _non_empty_query(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("query must not be empty")
    return value


def _source_checkout_root() -> Path | None:
    """Return the repository root in a source checkout (``--benchmark`` needs one)."""

    root = Path(__file__).resolve().parents[1]
    if (root / "benchmarks" / "run_suite.py").is_file() and (root / "data").is_dir():
        return root
    return None


def _preflight_queries(queries: list[str] | None) -> None:
    """Reject syntactically invalid queries before the slow ingestion."""

    for sql in queries or []:
        remaining_sql, _ = extract_as_of(sql)
        parse_where_clause(remaining_sql)


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-csv",
        type=Path,
        required=True,
        help="merged daily or one-minute OHLCV CSV",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=("monthly", "hourly"),
        default="monthly",
        help=("monthly for daily bars (default), hourly for dense one-minute bars"),
    )
    parser.add_argument(
        "--query",
        action="append",
        type=_non_empty_query,
        default=None,
        help="SQL query to execute after ingestion; repeat for multiple queries",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "run the full benchmark suite and generate available charts "
            "(requires a source checkout, not an installed wheel)"
        ),
    )
    args = parser.parse_args(argv)

    if args.benchmark and _source_checkout_root() is None:
        return _fail(
            "--benchmark requires a source checkout of PitDB (the benchmarks "
            "package and data/ directory are not part of the installed "
            "package); clone the repository and run it from there"
        )

    try:
        _preflight_queries(args.query)
        run_pipeline(
            data_csv_path=args.data_csv,
            chunk_boundary=_boundary_from_cli(args.chunk_strategy),
            queries=args.query,
        )
    except UnicodeDecodeError as exc:
        return _fail(f"{args.data_csv} is not valid UTF-8 text ({exc.reason})")
    except UnsupportedQueryError as exc:
        return _fail(f"unsupported or invalid --query: {exc}")
    except (OSError, ValueError) as exc:
        return _fail(str(exc))

    if args.benchmark:
        # Lazy import: benchmark/chart dependencies are optional.
        try:
            from benchmarks.run_suite import run_suite
        except ImportError as exc:
            return _fail(
                f"benchmark dependencies are unavailable ({exc}); "
                "install them with: pip install '.[benchmarks]'"
            )

        print("\nRunning benchmark suite...")
        return run_suite()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_pipeline"]
