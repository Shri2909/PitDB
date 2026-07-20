"""Integration tests for time-bounded, lossless chunk storage."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary
from src.store.chunk_store import ChunkStore

VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _market_frame(
    timestamps: pd.DatetimeIndex,
    symbol: str = "AAPL",
    base: float = 100.0,
) -> pd.DataFrame:
    """Build deterministic float64 OHLCV data for one sequential series."""

    offsets = np.arange(len(timestamps), dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": symbol,
            "open": base + offsets * 0.25,
            "high": base + 1.0 + offsets * 0.25,
            "low": base - 1.0 + offsets * 0.25,
            "close": base + 0.5 + offsets * 0.25,
            "volume": 1_000.0 + offsets * 10.0,
        }
    )


def _ingest_without_implicit_flush(store: ChunkStore, frame: pd.DataFrame) -> None:
    """Feed rows one at a time so tests control the end-of-stream flush."""

    for row in frame.itertuples(index=False):
        values = np.asarray(
            [row.open, row.high, row.low, row.close, row.volume],
            dtype=np.float64,
        )
        store.ingest_row(row.timestamp, row.symbol, values)


def test_chunk_creation_hourly() -> None:
    timestamps = pd.date_range(
        "2024-01-02 10:00:00", periods=3 * 60, freq="min", tz="UTC"
    )
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))

    store.ingest_dataframe(frame)

    assert store.total_chunks == 3
    for hour, chunk in enumerate(store.chunks):
        expected_start = timestamps[hour * 60]
        expected_end = timestamps[hour * 60 + 59]
        assert chunk.label.time_start == expected_start
        assert chunk.label.time_end == expected_end
        assert chunk.label.row_count == 60


def test_chunk_label_correctness() -> None:
    timestamps = pd.date_range("2024-03-01 09:30:00", periods=155, freq="min", tz="UTC")
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))
    store.ingest_dataframe(frame)

    for chunk in store.chunks:
        decoded = chunk.decompress()
        close_prices = decoded["close"].to_numpy(dtype=np.float64)
        decoded_timestamps = decoded["timestamp"]

        # Zone-map labels must never exclude a value actually in the chunk.
        assert np.all(chunk.label.min_price <= close_prices)
        assert np.all(close_prices <= chunk.label.max_price)
        assert chunk.label.min_price == float(np.min(close_prices))
        assert chunk.label.max_price == float(np.max(close_prices))
        assert chunk.label.time_start == decoded_timestamps.min()
        assert chunk.label.time_end == decoded_timestamps.max()


def test_multi_symbol_chunking() -> None:
    timestamps = pd.date_range("2024-04-01 10:00:00", periods=90, freq="min", tz="UTC")
    aapl = _market_frame(timestamps, symbol="AAPL", base=100.0)
    msft = _market_frame(timestamps, symbol="MSFT", base=300.0)
    interleaved = (
        pd.concat([aapl, msft], ignore_index=True)
        .sort_values(["timestamp", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))

    store.ingest_dataframe(interleaved)

    assert {chunk.label.symbol for chunk in store.chunks} == {"AAPL", "MSFT"}
    for chunk in store.chunks:
        decoded = chunk.decompress()
        assert decoded["symbol"].nunique() == 1
        assert decoded["symbol"].iat[0] == chunk.label.symbol


def test_chunk_boundary_query() -> None:
    timestamps = pd.date_range("2024-05-01 10:00:00", periods=120, freq="min", tz="UTC")
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))
    store.ingest_dataframe(frame)
    query_start = pd.Timestamp("2024-05-01 10:55:00", tz="UTC")
    query_end = pd.Timestamp("2024-05-01 11:05:00", tz="UTC")

    result = store.full_scan(lambda row: query_start <= row["timestamp"] <= query_end)

    assert len(result) == 11
    assert result["timestamp"].tolist() == list(
        pd.date_range(query_start, query_end, freq="min")
    )
    assert set(result["timestamp"].dt.hour) == {10, 11}


def test_round_trip_through_store() -> None:
    timestamps = pd.date_range("2024-06-03 09:30:00", periods=185, freq="min", tz="UTC")
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))

    store.ingest_dataframe(frame)
    restored = store.full_scan()

    pd.testing.assert_frame_equal(
        restored,
        frame,
        check_exact=True,
        check_dtype=True,
    )


def test_flush_partial_chunk() -> None:
    timestamps = pd.date_range("2024-07-01 14:00:00", periods=30, freq="min", tz="UTC")
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))

    _ingest_without_implicit_flush(store, frame)
    assert store.total_chunks == 0

    store.flush_all()

    assert store.total_chunks == 1
    label = store.chunks[0].label
    assert label.time_start == timestamps[0]
    assert label.time_end == timestamps[-1]
    assert label.row_count == 30
    assert label.min_price == float(frame["close"].min())
    assert label.max_price == float(frame["close"].max())


def test_get_chunk_by_id() -> None:
    # get_chunk used to scan self._chunks linearly; this pins its by-ID
    # lookup contract (found chunk matches its own label, and an unknown ID
    # raises KeyError) so a future change can't silently reintroduce an O(n)
    # scan or an incorrect lookup.
    timestamps = pd.date_range("2024-09-01 10:00:00", periods=90, freq="min", tz="UTC")
    frame = _market_frame(timestamps)
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))
    store.ingest_dataframe(frame)

    assert store.total_chunks > 1
    for chunk in store.chunks:
        found = store.get_chunk(chunk.label.chunk_id)
        assert found is chunk

    unknown_id = max(chunk.label.chunk_id for chunk in store.chunks) + 1
    with pytest.raises(KeyError):
        store.get_chunk(unknown_id)


def test_empty_store() -> None:
    store = ChunkStore()

    assert store.total_chunks == 0
    assert store.total_rows == 0
    assert store.total_column_bytes == 0
    assert store.chunks == []
    assert store.labels == []
    assert store.full_scan().empty
    assert store.full_scan().columns.tolist() == [
        "timestamp",
        "symbol",
        *VALUE_COLUMNS,
    ]


def test_row_count_accuracy() -> None:
    row_count = 197
    frame = _market_frame(
        pd.date_range("2024-08-01 09:30:00", periods=row_count, freq="min", tz="UTC")
    )
    store = ChunkStore(DurationBoundary(timedelta(hours=1)))

    store.ingest_dataframe(frame)

    assert store.total_rows == row_count
    assert sum(chunk.label.row_count for chunk in store.chunks) == row_count
    assert sum(len(chunk.timestamps) for chunk in store.chunks) == row_count


def test_monthly_boundary_chunking() -> None:
    # Two complete calendar years give every month 20-23 weekday rows and cover
    # both ordinary transitions and the December 2023 -> January 2024 rollover.
    timestamps = pd.date_range("2023-01-02", "2024-12-31", freq="B", tz="UTC")
    frame = _market_frame(timestamps)
    store = ChunkStore(chunk_boundary=MonthlyBoundary())

    store.ingest_dataframe(frame)

    assert store.total_chunks == 24
    assert store.total_rows == len(frame)

    observed_months: list[tuple[int, int]] = []
    for chunk in store.chunks:
        decoded = chunk.decompress()
        timestamp_values = decoded["timestamp"]
        month_keys = {
            (timestamp.year, timestamp.month) for timestamp in timestamp_values
        }

        # No chunk may cross a calendar-month boundary.
        assert len(month_keys) == 1
        month_key = next(iter(month_keys))
        observed_months.append(month_key)
        assert month_key == (
            chunk.label.time_start.year,
            chunk.label.time_start.month,
        )
        assert month_key == (
            chunk.label.time_end.year,
            chunk.label.time_end.month,
        )
        assert chunk.label.time_start == timestamp_values.min()
        assert chunk.label.time_end == timestamp_values.max()
        assert 20 <= chunk.label.row_count <= 23

    expected_months = [(year, month) for year in (2023, 2024) for month in range(1, 13)]
    assert observed_months == expected_months

    december_index = observed_months.index((2023, 12))
    december = store.chunks[december_index]
    january = store.chunks[december_index + 1]
    assert (december.label.time_end.year, december.label.time_end.month) == (
        2023,
        12,
    )
    assert (january.label.time_start.year, january.label.time_start.month) == (
        2024,
        1,
    )
    assert MonthlyBoundary().crosses(
        december.label.time_start, january.label.time_start
    )

    # The year component is independently load-bearing when month numbers match.
    assert MonthlyBoundary().crosses(
        pd.Timestamp("2023-01-03", tz="UTC"),
        pd.Timestamp("2024-01-02", tz="UTC"),
    )


def test_ingest_correction_appends_a_labeled_chunk() -> None:
    store = ChunkStore()
    event_time = pd.Timestamp("2024-01-02 09:30:00", tz="UTC")
    store.ingest_row(event_time, "AAPL", np.array([100.0, 101.0, 99.0, 100.5, 1_000.0]))
    store.flush_all()
    assert store.total_chunks == 1
    assert store.total_rows == 1

    transaction_time = event_time + timedelta(days=1)
    store.ingest_correction(
        event_time,
        "AAPL",
        np.array([105.0, 106.0, 104.0, 105.5, 1_100.0]),
        transaction_time=transaction_time,
    )

    assert store.total_chunks == 2
    assert store.total_rows == 2
    correction_label = store.labels[-1]
    assert correction_label.transaction_time_start == transaction_time
    assert correction_label.transaction_time_end == transaction_time
    assert correction_label.symbol == "AAPL"
    assert correction_label.row_count == 1
    # Chunk IDs are drawn from the same shared, never-reused counter.
    assert correction_label.chunk_id != store.labels[0].chunk_id


def test_cross_symbol_tz_mismatch_raises_and_preserves_store_state() -> None:
    """Regression: _add_chunk used to mutate _chunks/_chunk_by_id before the
    bisect call that could raise on a tz-awareness mismatch, permanently
    desyncing the as-of index from the rest of the store -- the chunk stayed
    registered but silently vanished from every future as-of lookup.
    """

    store = ChunkStore()
    store.ingest_row(
        pd.Timestamp("2024-01-02 09:30:00"),
        "AAPL",
        np.array([100.0, 101.0, 99.0, 100.5, 1_000.0]),
    )
    store.flush_all()
    assert store.total_chunks == 1

    # The tz mismatch only surfaces once this row's chunk is actually
    # finalized (_add_chunk is where the as-of-index bisect happens) -- a
    # brand-new builder's first row never crosses its own boundary, so the
    # mismatch is only caught on the flush that finalizes it.
    store.ingest_row(
        pd.Timestamp("2024-01-02 09:30:00", tz="UTC"),
        "MSFT",
        np.array([200.0, 201.0, 199.0, 200.5, 2_000.0]),
    )
    with pytest.raises(ValueError, match="timezone awareness"):
        store.flush_all()

    # The failed chunk must leave no trace: not registered in _chunks, not in
    # _chunk_by_id, and the as-of index still exactly matches total_chunks --
    # not a chunk registered in one structure but silently missing from the
    # as-of index forever.
    assert store.total_chunks == 1
    assert store.total_rows == 1
    assert len(store.labels) == 1
    assert (
        store.labels_visible_as_of(pd.Timestamp("2024-01-02 09:30:00")) == store.labels
    )


def test_flush_all_then_out_of_order_row_raises() -> None:
    """Regression: a fresh ChunkBuilder created after flush_all() has no
    memory of a symbol's prior data, so its own non-decreasing check alone
    can't catch an out-of-order row resuming after a flush -- causality was
    silently violated across that boundary.
    """

    store = ChunkStore()
    t0 = pd.Timestamp("2024-03-01 09:30:00", tz="UTC")
    t1 = pd.Timestamp("2024-03-01 09:31:00", tz="UTC")
    store.ingest_row(t0, "AAPL", np.array([100.0, 101.0, 99.0, 100.5, 1_000.0]))
    store.ingest_row(t1, "AAPL", np.array([101.0, 102.0, 100.0, 101.5, 1_100.0]))
    store.flush_all()

    with pytest.raises(ValueError, match="non-decreasing order"):
        store.ingest_row(t0, "AAPL", np.array([99.0, 100.0, 98.0, 99.5, 900.0]))
