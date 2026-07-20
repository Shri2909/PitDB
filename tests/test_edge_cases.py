"""Edge-case integration tests for time-series chunk storage and querying."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.query.executor import QueryExecutor
from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary
from src.store.chunk_store import ChunkStore

VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _market_frame(
    timestamps: pd.DatetimeIndex,
    *,
    symbol: str = "AAPL",
    closes: np.ndarray | None = None,
) -> pd.DataFrame:
    row_count = len(timestamps)
    if closes is None:
        closes = 100.0 + np.arange(row_count, dtype=np.float64) * 0.25
    closes = np.asarray(closes, dtype=np.float64)
    if closes.shape != (row_count,):
        raise ValueError("closes must contain one value per timestamp")

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": closes - 0.25,
            "high": closes + 0.50,
            "low": closes - 0.50,
            "close": closes,
            "volume": 1_000.0 + np.arange(row_count, dtype=np.float64),
        }
    )
    for column in VALUE_COLUMNS:
        frame[column] = frame[column].astype(np.float64)
    return frame


def _assert_float64_bits_equal(expected: np.ndarray, actual: np.ndarray) -> None:
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    assert expected.shape == actual.shape
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


def test_timestamp_dst_transition() -> None:
    # The fall transition repeats 01:00 and 01:30 with different UTC offsets.
    # DurationBoundary must use real elapsed time, not naive wall-clock labels.
    timestamps = pd.date_range(
        "2024-11-03 00:30:00",
        periods=7,
        freq="30min",
        tz="America/New_York",
    )
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))

    store.ingest_dataframe(frame)

    assert store.total_chunks == 4
    assert [chunk.label.row_count for chunk in store.chunks] == [2, 2, 2, 1]
    assert store.total_rows == len(frame)
    assert all(chunk.label.time_start.utcoffset() is not None for chunk in store.chunks)
    assert all(chunk.label.time_end.utcoffset() is not None for chunk in store.chunks)

    result = QueryExecutor(store).execute(
        "SELECT * FROM data "
        "WHERE timestamp >= '2024-11-03T01:00:00-04:00' "
        "AND timestamp < '2024-11-03T01:30:00-05:00'"
    )
    expected = frame.iloc[1:4].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        result.data.reset_index(drop=True),
        expected,
        check_exact=True,
        check_dtype=True,
    )


def test_duplicate_timestamps() -> None:
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-03-01 09:30:00", tz="UTC"),
            pd.Timestamp("2024-03-01 09:31:00", tz="UTC"),
        ]
    )
    aapl = _market_frame(timestamps, symbol="AAPL")
    msft = _market_frame(
        timestamps,
        symbol="MSFT",
        closes=np.array([300.0, 300.25], dtype=np.float64),
    )
    interleaved = (
        pd.concat([aapl, msft], ignore_index=True)
        .sort_values(["timestamp", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    store = ChunkStore(MonthlyBoundary())

    store.ingest_dataframe(interleaved)

    assert store.total_rows == 4
    assert store.total_chunks == 2
    assert {chunk.label.symbol for chunk in store.chunks} == {"AAPL", "MSFT"}
    for chunk in store.chunks:
        decoded = chunk.decompress()
        assert chunk.label.row_count == 2
        assert decoded["timestamp"].tolist() == list(timestamps)
        assert decoded["symbol"].nunique() == 1
        assert decoded["symbol"].iat[0] == chunk.label.symbol


def test_market_holiday_gap() -> None:
    before_holiday = pd.Timestamp("2024-07-03 16:00:00", tz="UTC")
    after_holiday = pd.Timestamp("2024-07-08 09:30:00", tz="UTC")
    timestamps = pd.DatetimeIndex([before_holiday, after_holiday])
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(days=1)))

    store.ingest_dataframe(frame)

    # The duration strategy closes the pre-holiday chunk. No rows are invented
    # for the weekend/holiday gap, and the two labels expose the actual break.
    assert store.total_chunks == 2
    first, second = store.chunks
    assert first.label.time_start == first.label.time_end == before_holiday
    assert second.label.time_start == second.label.time_end == after_holiday
    assert second.label.time_start - first.label.time_end > timedelta(days=3)
    assert first.label.row_count == second.label.row_count == 1


def test_single_row_chunk() -> None:
    timestamp = pd.Timestamp("2024-08-01 09:30:00", tz="UTC")
    frame = _market_frame(pd.DatetimeIndex([timestamp]), closes=np.array([42.0]))
    store = ChunkStore(MonthlyBoundary())

    store.ingest_dataframe(frame)

    assert store.total_chunks == 1
    chunk = store.chunks[0]
    assert chunk.label.row_count == 1
    assert chunk.label.time_start == chunk.label.time_end == timestamp
    # Each independent float64 column contains only its raw 64-bit first value.
    assert chunk.column_byte_lengths == [8] * len(VALUE_COLUMNS)
    assert len(chunk.column_data) == 8 * len(VALUE_COLUMNS)
    pd.testing.assert_frame_equal(
        chunk.decompress(), frame, check_exact=True, check_dtype=True
    )

    result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price = 42"
    )
    assert result.row_count == 1
    pd.testing.assert_frame_equal(
        result.data.reset_index(drop=True),
        frame,
        check_exact=True,
        check_dtype=True,
    )


def test_nan_in_data() -> None:
    timestamps = pd.date_range("2024-09-02 09:30:00", periods=6, freq="min", tz="UTC")
    mixed_close = np.array(
        [np.nan, 100.0, np.nan, 90.0, 110.0, np.nan], dtype=np.float64
    )
    mixed = _market_frame(timestamps, symbol="AAPL", closes=mixed_close)
    # Exercise NaNs outside the tracked close column as well.
    mixed.loc[[1, 4], "open"] = np.nan
    all_nan = _market_frame(
        timestamps,
        symbol="MSFT",
        closes=np.full(len(timestamps), np.nan, dtype=np.float64),
    )
    frame = (
        pd.concat([mixed, all_nan], ignore_index=True)
        .sort_values(["timestamp", "symbol"], kind="mergesort")
        .reset_index(drop=True)
    )
    store = ChunkStore(MonthlyBoundary())

    store.ingest_dataframe(frame)

    labels = {chunk.label.symbol: chunk.label for chunk in store.chunks}
    assert labels["AAPL"].min_price == 90.0
    assert labels["AAPL"].max_price == 110.0
    assert labels["MSFT"].min_price == float("inf")
    assert labels["MSFT"].max_price == float("-inf")

    decoded = {chunk.label.symbol: chunk.decompress() for chunk in store.chunks}
    for symbol, expected in (("AAPL", mixed), ("MSFT", all_nan)):
        for column in VALUE_COLUMNS:
            _assert_float64_bits_equal(
                expected[column].to_numpy(dtype=np.float64),
                decoded[symbol][column].to_numpy(dtype=np.float64),
            )

    mixed_result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 100"
    )
    assert mixed_result.row_count == 1
    assert mixed_result.data["close"].iat[0] == 110.0

    all_nan_result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE symbol = 'MSFT' AND price >= 0"
    )
    assert all_nan_result.data.empty
    assert all_nan_result.row_count == 0
    assert all_nan_result.pushdown.chunks_skipped_ratio == 1.0


def test_very_large_chunk() -> None:
    row_count = 100_000
    timestamp = pd.Timestamp("2024-10-01 09:30:00", tz="UTC")
    values = np.array([125.0, 126.0, 124.0, 125.5, 10_000.0], dtype=np.float64)
    store = ChunkStore(MonthlyBoundary())

    # Direct sequential ingestion avoids constructing a second 100,000-row
    # DataFrame before compression while exercising the same store API.
    for _ in range(row_count):
        store.ingest_row(timestamp, "AAPL", values)
    store.flush_all()

    assert store.total_chunks == 1
    assert store.total_rows == row_count
    chunk = store.chunks[0]
    assert chunk.label.row_count == row_count
    restored = chunk.decompress()
    assert len(restored) == row_count
    assert restored["timestamp"].iloc[0] == timestamp
    assert restored["timestamp"].iloc[-1] == timestamp
    for index, column in enumerate(VALUE_COLUMNS):
        np.testing.assert_array_equal(
            restored[column].to_numpy(dtype=np.float64),
            np.full(row_count, values[index], dtype=np.float64),
        )


def test_zero_matching_rows() -> None:
    timestamps = pd.date_range("2024-12-02 09:30:00", periods=10, freq="min", tz="UTC")
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))
    store.ingest_dataframe(frame)

    result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE symbol = 'NONEXISTENT' AND price > 10000"
    )

    assert result.data.empty
    assert result.data.columns.tolist() == ["timestamp", "symbol", *VALUE_COLUMNS]
    assert result.row_count == 0
    assert result.selectivity == 0.0
    assert result.pushdown.candidate_count == 0
    assert result.pushdown.chunks_skipped_ratio == 1.0


__all__ = [
    "test_duplicate_timestamps",
    "test_market_holiday_gap",
    "test_nan_in_data",
    "test_single_row_chunk",
    "test_timestamp_dst_transition",
    "test_very_large_chunk",
    "test_zero_matching_rows",
]
