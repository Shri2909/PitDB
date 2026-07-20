"""End-to-end tests for replay, compression, storage, and querying."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.pipeline import run_pipeline
from src.query.sql_parser import parse_where_clause
from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DAILY_DATA_FILE = PROJECT_ROOT / "data/cache/all_symbols_daily.csv"
INTRADAY_DATA_FILE = PROJECT_ROOT / "data/cache/all_symbols_1m.csv"
VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]

QUERIES = {
    "Q1": ("SELECT * FROM data WHERE symbol = 'AAPL' AND date = '2024-03-01'"),
    "Q2": (
        "SELECT * FROM data WHERE symbol = 'AAPL' "
        "AND date >= '2024-03-01' AND date < '2024-04-01'"
    ),
    "Q3": (
        "SELECT * FROM data WHERE symbol IN ('AAPL', 'MSFT') "
        "AND date >= '2024-03-01' AND date < '2024-03-08'"
    ),
    "Q4": (
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 150 "
        "AND date >= '2024-01-01'"
    ),
    "Q5": "SELECT * FROM data WHERE date >= '2024-01-01'",
}


def _synthetic_market_frame() -> pd.DataFrame:
    """Create 500 trading days x 10 symbols with compressible plateaus."""

    symbols = np.asarray(
        [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "META",
            "NVDA",
            "TSLA",
            "JPM",
            "XOM",
            "WMT",
        ],
        dtype=object,
    )
    timestamps = pd.bdate_range(
        "2023-01-02 16:00:00",
        periods=500,
        tz="UTC",
    )
    symbol_count = len(symbols)
    day_index = np.repeat(np.arange(len(timestamps), dtype=np.int64), symbol_count)
    symbol_index = np.tile(np.arange(symbol_count, dtype=np.int64), len(timestamps))

    # All prices use binary-exact quarter increments. Five-day plateaus model
    # unchanged observations and guarantee that compression is not assessed on
    # adversarial random mantissas alone.
    base = 140.0 + symbol_index.astype(np.float64) * 20.0
    close = base + (day_index // 5).astype(np.float64) * 0.25
    volume = (
        1_000_000.0
        + (day_index // 10).astype(np.float64) * 128.0
        + symbol_index.astype(np.float64) * 1_024.0
    )

    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps.to_numpy(), symbol_count),
            "symbol": np.tile(symbols, len(timestamps)),
            "open": close - 0.25,
            "high": close + 0.50,
            "low": close - 0.50,
            "close": close,
            "volume": volume,
        }
    )
    for column in VALUE_COLUMNS:
        frame[column] = frame[column].astype(np.float64)

    assert len(frame) == 5_000
    return frame


def _canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize cross-symbol chunk ordering without changing row values."""

    return frame.sort_values(
        ["timestamp", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def synthetic_dataset(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pd.DataFrame, Path]:
    frame = _synthetic_market_frame()
    path = tmp_path_factory.mktemp("end-to-end") / "synthetic_daily.csv"
    frame.to_csv(path, index=False)
    return frame, path


@pytest.fixture(scope="module")
def synthetic_pipeline(
    synthetic_dataset: tuple[pd.DataFrame, Path],
) -> dict[str, Any]:
    _, path = synthetic_dataset
    return run_pipeline(
        path,
        chunk_boundary=MonthlyBoundary(),
        queries=list(QUERIES.values()),
    )


def test_full_pipeline_synthetic(
    synthetic_dataset: tuple[pd.DataFrame, Path],
    synthetic_pipeline: dict[str, Any],
) -> None:
    original, _ = synthetic_dataset
    store = synthetic_pipeline["store"]
    restored = store.full_scan()

    assert store.total_rows == 5_000
    assert store.total_chunks > 1
    assert len(synthetic_pipeline["query_results"]) == len(QUERIES)
    pd.testing.assert_frame_equal(
        _canonical(restored),
        _canonical(original),
        check_exact=True,
        check_dtype=True,
        obj="full replay-compress-store-decompress round trip",
    )


def test_full_pipeline_all_queries(
    synthetic_pipeline: dict[str, Any],
) -> None:
    store = synthetic_pipeline["store"]
    pipeline_queries = synthetic_pipeline["query_results"]

    assert len(pipeline_queries) == len(QUERIES)
    for (name, expected_sql), observed in zip(
        QUERIES.items(), pipeline_queries, strict=True
    ):
        assert observed["sql"] == expected_sql
        predicate = parse_where_clause(expected_sql)
        indexed = observed["result"].data
        full_scan = store.full_scan(predicate.evaluate_against_row)

        # Both paths use the same exact predicate semantics. The comparison
        # therefore isolates the conservative pushdown invariant: skipping a
        # chunk must never remove a valid result row.
        pd.testing.assert_frame_equal(
            _canonical(indexed),
            _canonical(full_scan),
            check_exact=True,
            check_dtype=True,
            obj=f"pipeline/full-scan equivalence for {name}",
        )


@pytest.mark.parametrize(
    ("data_path", "boundary"),
    [
        pytest.param(
            DAILY_DATA_FILE,
            MonthlyBoundary(),
            marks=pytest.mark.skipif(
                not DAILY_DATA_FILE.is_file(),
                reason="Dataset A merged daily CSV is unavailable",
            ),
            id="dataset-a-daily-monthly",
        ),
        pytest.param(
            INTRADAY_DATA_FILE,
            DurationBoundary(timedelta(hours=1)),
            marks=pytest.mark.skipif(
                not INTRADAY_DATA_FILE.is_file(),
                reason="Dataset B merged one-minute CSV is unavailable",
            ),
            id="dataset-b-intraday-hourly",
        ),
    ],
)
def test_pipeline_with_real_data(
    data_path: Path,
    boundary: MonthlyBoundary | DurationBoundary,
) -> None:
    result = run_pipeline(data_path, chunk_boundary=boundary)
    store = result["store"]

    expected_rows = len(pd.read_csv(data_path, usecols=["timestamp"]))
    assert store.total_rows == expected_rows
    assert result["summary"]["total_rows"] == expected_rows
    assert store.total_chunks > 0
    assert isinstance(store.chunk_boundary, type(boundary))


def test_pipeline_stats_consistency(
    synthetic_dataset: tuple[pd.DataFrame, Path],
    synthetic_pipeline: dict[str, Any],
) -> None:
    original, _ = synthetic_dataset
    store = synthetic_pipeline["store"]
    summary = synthetic_pipeline["summary"]
    raw_value_bytes = len(original) * len(VALUE_COLUMNS) * 8

    assert store.total_rows == len(original)
    assert summary["total_rows"] == len(original)
    assert summary["raw_value_bytes"] == raw_value_bytes
    assert summary["total_column_bytes"] == store.total_column_bytes
    # Storage is raw float64 (no compression), so column bytes should exactly
    # match the raw payload size, not be strictly smaller than it.
    assert store.total_column_bytes == raw_value_bytes
    assert summary["compression_ratio"] == pytest.approx(
        raw_value_bytes / store.total_column_bytes
    )


__all__ = [
    "test_full_pipeline_all_queries",
    "test_full_pipeline_synthetic",
    "test_pipeline_stats_consistency",
    "test_pipeline_with_real_data",
]
