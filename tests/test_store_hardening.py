"""Storage hardening: tz validation, atomic ingest, locking, invariants, DST."""

from __future__ import annotations

import sys
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.query.executor import QueryExecutor
from src.store.chunk import Chunk
from src.store.chunk_boundary import ChunkBoundary, DurationBoundary, MonthlyBoundary
from src.store.chunk_builder import ChunkBuilder
from src.store.chunk_store import ChunkStore
from src.store.label import ChunkLabel

UTC = timezone.utc
T = datetime(2024, 1, 1, tzinfo=UTC)
NAIVE = datetime(2024, 1, 1)


def vals(close: float = 10.0) -> np.ndarray:
    return np.array(
        [close - 0.5, close + 1.0, close - 1.0, close, 100.0], dtype=np.float64
    )


def query(store: ChunkStore, cutoff=None) -> pd.DataFrame:
    clause = "" if cutoff is None else f" TIMESTAMP AS OF '{cutoff.isoformat()}'"
    frame = QueryExecutor(store).execute("SELECT * FROM data" + clause).data
    return frame.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


# --- 1. timezone awareness is validated before buffering ---


@pytest.mark.parametrize("first_aware", [True, False])
def test_mixed_awareness_row_rejected_immediately_store_stays_usable(first_aware):
    first = T if first_aware else NAIVE
    second = NAIVE if first_aware else T
    store = ChunkStore()
    store.ingest_row(first, "A", vals())
    with pytest.raises(ValueError, match="timezone awareness"):
        store.ingest_row(second, "B", vals(20))
    assert "B" not in store._builders

    # Later symbols still ingest and flush; flush_all() does not raise forever.
    store.ingest_row(first + timedelta(days=1), "C", vals(30))
    store.flush_all()
    assert store.total_rows == 2
    assert sorted(c.label.symbol for c in store.chunks) == ["A", "C"]


@pytest.mark.parametrize("first_aware", [True, False])
def test_mixed_awareness_rejected_when_reference_comes_from_correction(first_aware):
    first = T if first_aware else NAIVE
    second = NAIVE if first_aware else T
    store = ChunkStore()
    store.ingest_correction(first, "A", vals(), first + timedelta(days=1))
    with pytest.raises(ValueError, match="timezone awareness"):
        store.ingest_row(second, "B", vals())
    assert not store._builders
    # The wrong-awareness correction is rejected too, without leaking an ID.
    chunks_before = store.total_chunks
    with pytest.raises(ValueError, match="timezone awareness"):
        store.ingest_correction(second, "B", vals(), second + timedelta(days=1))
    assert store.total_chunks == chunks_before
    store.ingest_row(first, "C", vals())
    store.flush_all()
    assert store.total_rows == 2


def test_mixed_awareness_rejected_across_symbols_before_first_flush():
    store = ChunkStore()
    store.ingest_row(NAIVE, "A", vals())
    with pytest.raises(ValueError, match="timezone awareness"):
        store.ingest_row(T, "B", vals())
    store.flush_all()
    assert store.total_rows == 1


# --- 2. failed bulk ingest leaves the store unchanged ---


def _frame(rows):
    return pd.DataFrame(
        [
            {
                "timestamp": ts,
                "symbol": sym,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": close,
                "volume": 10.0,
            }
            for ts, sym, close in rows
        ]
    )


def test_failed_ingest_dataframe_is_atomic_and_retry_works():
    store = ChunkStore()
    # Rows already accepted through the streaming API stay untouched.
    store.ingest_row(datetime(2023, 12, 1, tzinfo=UTC), "PRE", vals())
    store.flush_all()
    store.ingest_row(datetime(2023, 12, 2, tzinfo=UTC), "OPEN", vals())
    rows_before = store.total_rows
    chunks_before = store.total_chunks

    good = [(datetime(2024, m, 1, tzinfo=UTC), "A", float(m)) for m in range(1, 5)]
    bad = _frame(good + [(datetime(2024, 1, 15, tzinfo=UTC), "A", 9.0)])
    with pytest.raises(ValueError, match="non-decreasing"):
        store.ingest_dataframe(bad)

    assert store.total_rows == rows_before
    assert store.total_chunks == chunks_before
    assert sorted(store._builders) == ["OPEN"]
    assert store._builders["OPEN"]._row_count == 1
    assert "A" not in store._last_valid_timestamp

    store.ingest_dataframe(_frame(good))
    assert store.total_rows == rows_before + 1 + len(good)  # OPEN flushed too
    assert not store._builders
    assert len(store.labels_visible_as_of(datetime(2030, 1, 1, tzinfo=UTC))) == (
        store.total_chunks
    )


def test_failed_ingest_dataframe_on_timezone_mismatch_is_atomic():
    store = ChunkStore()
    frame = _frame(
        [
            (pd.Timestamp("2024-01-01", tz="UTC"), "A", 1.0),
            (pd.Timestamp("2024-02-01"), "A", 2.0),
        ]
    )
    with pytest.raises(ValueError):
        store.ingest_dataframe(frame)
    assert store.total_rows == 0
    assert store.total_chunks == 0
    assert not store._builders
    store.ingest_dataframe(frame.iloc[:1])
    assert store.total_rows == 1


# --- 3. thread safety ---


@pytest.fixture
def aggressive_thread_switching():
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def test_concurrent_corrections_keep_index_consistent(aggressive_thread_switching):
    store = ChunkStore()
    store.ingest_row(T, "BASE", vals())
    store.flush_all()
    threads_n, per_thread = 4, 150
    errors: list[BaseException] = []
    barrier = threading.Barrier(threads_n)

    def worker(index: int) -> None:
        try:
            barrier.wait()
            for i in range(per_thread):
                # Interleaved, deliberately non-monotonic transaction times.
                event = T + timedelta(hours=i)
                tx = event + timedelta(days=(i * threads_n + index) % 97 + 1)
                store.ingest_correction(event, f"S{index}", vals(float(i)), tx)
                store.labels_visible_as_of(tx)
        except BaseException as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(threads_n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert store.total_chunks == 1 + threads_n * per_thread
    ids = [c.label.chunk_id for c in store.chunks]
    assert len(set(ids)) == len(ids)
    assert store._as_of_keys == sorted(store._as_of_keys)
    assert len(store._as_of_keys) == len(store._as_of_labels) == store.total_chunks
    assert {id(label) for label in store._as_of_labels} == {
        id(c.label) for c in store.chunks
    }
    cutoff = T + timedelta(days=10)
    visible = {label.chunk_id for label in store.labels_visible_as_of(cutoff)}
    expected = {
        c.label.chunk_id
        for c in store.chunks
        if (c.label.transaction_time_start or c.label.time_start) <= cutoff
    }
    assert visible == expected


def test_concurrent_ingest_rows_distinct_symbols(aggressive_thread_switching):
    store = ChunkStore(chunk_boundary=DurationBoundary(timedelta(hours=1)))
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            for i in range(200):
                store.ingest_row(T + timedelta(minutes=i), f"S{index}", vals(float(i)))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    store.flush_all()
    assert not errors, errors
    assert store.total_rows == 800
    ids = [c.label.chunk_id for c in store.chunks]
    assert len(set(ids)) == len(ids)
    assert store._as_of_keys == sorted(store._as_of_keys)


# --- 4. Chunk / ChunkLabel cross-field invariants ---


def _label(**overrides) -> ChunkLabel:
    fields = dict(
        symbol="A",
        time_start=T,
        time_end=T + timedelta(days=1),
        min_price=1.0,
        max_price=2.0,
        row_count=2,
        chunk_id=0,
    )
    fields.update(overrides)
    return ChunkLabel(**fields)


def _chunk(label=None, timestamps=None, txn=None, rows=2) -> Chunk:
    data = np.arange(rows, dtype=np.float64).tobytes()
    return Chunk(
        label=label if label is not None else _label(row_count=rows),
        column_data=data,
        column_byte_lengths=[len(data)],
        num_columns=1,
        column_names=["close"],
        timestamps=(
            timestamps
            if timestamps is not None
            else [T + timedelta(days=i) for i in range(rows)]
        ),
        correction_transaction_times=txn,
    )


def test_label_rejects_inverted_prices():
    with pytest.raises(ValueError, match="min_price"):
        _label(min_price=3.0, max_price=2.0)


def test_label_rejects_nan_prices():
    with pytest.raises(ValueError, match="NaN"):
        _label(min_price=float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        _label(max_price=float("nan"))


def test_label_accepts_equal_prices_and_all_nan_marker():
    _label(min_price=5.0, max_price=5.0)
    _label(min_price=float("inf"), max_price=float("-inf"))


def test_builder_all_nan_prices_and_equal_timestamps_still_valid():
    builder = ChunkBuilder("A", chunk_boundary=MonthlyBoundary())
    nan_row = vals()
    nan_row[3] = np.nan
    builder.add_row(T, nan_row)
    builder.add_row(T, nan_row)  # equal timestamps are legitimate
    chunk = builder.flush()
    assert chunk is not None and chunk.label.row_count == 2


def test_chunk_accepts_valid_base_and_correction():
    _chunk()
    tx = [T + timedelta(days=5), T + timedelta(days=3)]  # unsorted is fine
    _chunk(
        label=_label(
            transaction_time_start=T + timedelta(days=3),
            transaction_time_end=T + timedelta(days=5),
        ),
        txn=tx,
    )


def test_chunk_rejects_unsorted_timestamps():
    with pytest.raises(ValueError, match="non-decreasing"):
        _chunk(timestamps=[T + timedelta(days=1), T])


def test_chunk_rejects_time_start_not_first_timestamp():
    with pytest.raises(ValueError, match="time_start"):
        _chunk(label=_label(time_start=T - timedelta(days=1)))


def test_chunk_rejects_timestamp_beyond_time_end():
    with pytest.raises(ValueError, match="time_end"):
        _chunk(label=_label(time_end=T))


def test_chunk_rejects_mixed_awareness_timestamps():
    with pytest.raises(ValueError, match="timezone awareness"):
        _chunk(timestamps=[T, NAIVE + timedelta(days=1)])


def test_chunk_rejects_transaction_time_outside_label_range():
    label = _label(
        transaction_time_start=T + timedelta(days=2),
        transaction_time_end=T + timedelta(days=3),
    )
    with pytest.raises(ValueError, match="transaction-time range"):
        _chunk(label=label, txn=[T + timedelta(days=2), T + timedelta(days=4)])


def test_chunk_rejects_transaction_time_before_row_timestamp():
    label = _label(
        transaction_time_start=T - timedelta(days=1),
        transaction_time_end=T + timedelta(days=5),
    )
    with pytest.raises(ValueError, match="precede"):
        _chunk(label=label, txn=[T - timedelta(days=1), T + timedelta(days=5)])


# --- 5. builder state untouched when the boundary raises ---


class _RaisingBoundary(ChunkBoundary):
    def __init__(self) -> None:
        self.fail = True

    def crosses(self, time_start, current_timestamp) -> bool:
        if self.fail:
            raise RuntimeError("boundary exploded")
        return False


def test_builder_state_untouched_when_boundary_raises_on_first_row():
    boundary = _RaisingBoundary()
    builder = ChunkBuilder("A", chunk_boundary=boundary)
    with pytest.raises(RuntimeError, match="exploded"):
        builder.add_row(T, vals())
    assert builder._time_start is None
    assert builder._row_count == 0
    assert builder._raw_values == {}
    assert builder.flush() is None

    boundary.fail = False
    builder.add_row(T + timedelta(days=1), vals())
    chunk = builder.flush()
    assert chunk is not None
    assert chunk.label.time_start == T + timedelta(days=1)


def test_builder_state_untouched_when_boundary_raises_mid_window():
    boundary = _RaisingBoundary()
    boundary.fail = False
    builder = ChunkBuilder("A", chunk_boundary=boundary)
    builder.add_row(T, vals())
    boundary.fail = True
    with pytest.raises(RuntimeError):
        builder.add_row(T + timedelta(days=1), vals())
    assert builder._row_count == 1
    assert builder._time_end == T


def test_store_stays_consistent_when_boundary_raises_on_first_row():
    boundary = _RaisingBoundary()
    store = ChunkStore(chunk_boundary=boundary)
    with pytest.raises(RuntimeError):
        store.ingest_row(T, "A", vals())
    assert store._time_reference is None
    boundary.fail = False
    store.ingest_row(NAIVE, "A", vals())  # no stale tz reference left behind
    store.flush_all()
    assert store.total_rows == 1


# --- 6. compaction across DST and across differing transaction times ---

NY = "America/New_York"


def _ny(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=NY)


def _dst_store(boundary, times) -> ChunkStore:
    store = ChunkStore(chunk_boundary=boundary)
    store.ingest_row(times[0] - pd.Timedelta(days=30), "A", vals())
    store.flush_all()
    for i, ts in enumerate(times):
        store.ingest_correction(ts, "A", vals(20.0 + i), ts + pd.Timedelta(hours=1 + i))
    return store


def _assert_compaction_is_transparent(store, times, merge_boundary=None):
    cutoffs = [None]
    for ts in times:
        for hours in range(0, 8):
            cutoffs.append(ts + pd.Timedelta(hours=hours))
    before = [query(store, c) for c in cutoffs]
    chunks_before = store.total_chunks
    reduced = store.compact_corrections(boundary=merge_boundary)
    assert reduced > 0 and store.total_chunks == chunks_before - reduced
    for cutoff, expected in zip(cutoffs, before):
        pd.testing.assert_frame_equal(query(store, cutoff), expected, check_exact=True)
    assert store._as_of_keys == sorted(store._as_of_keys)
    for chunk in store.chunks:  # merged chunks are structurally valid
        if chunk.label.transaction_time_start is not None:
            assert chunk.label.transaction_time_start == min(
                chunk.correction_transaction_times
            )


def test_monthly_boundary_compaction_across_spring_forward():
    times = [
        _ny("2024-03-09 12:00"),
        _ny("2024-03-10 03:30"),  # first instant after the 2am gap
        _ny("2024-03-11 12:00"),
        _ny("2024-03-31 23:30"),  # still March locally, already April in UTC
        _ny("2024-04-01 00:30"),
    ]
    store = _dst_store(MonthlyBoundary(), times)
    _assert_compaction_is_transparent(store, times)
    corrections = [c for c in store.chunks if c.label.transaction_time_start]
    assert sorted(c.label.row_count for c in corrections) == [1, 4]


def test_duration_boundary_compaction_across_spring_forward():
    # 12:00 EST Mar 9 -> 12:00 EDT Mar 10 is only 23 elapsed hours.
    times = [
        _ny("2024-03-09 12:00"),
        _ny("2024-03-10 12:00"),
        _ny("2024-03-11 12:00"),
    ]
    store = _dst_store(DurationBoundary(timedelta(hours=24)), times)
    _assert_compaction_is_transparent(store, times)
    corrections = [c for c in store.chunks if c.label.transaction_time_start]
    assert sorted(c.label.row_count for c in corrections) == [1, 2]


def test_compaction_across_fall_back_fold_keeps_both_instants():
    first = pd.Timestamp("2024-11-03 01:30").tz_localize(NY, ambiguous=True)  # EDT
    second = pd.Timestamp("2024-11-03 01:30").tz_localize(NY, ambiguous=False)  # EST
    assert second - first == pd.Timedelta(hours=1)
    times = [first, second, _ny("2024-11-04 09:00")]
    store = _dst_store(DurationBoundary(timedelta(hours=2)), times)
    _assert_compaction_is_transparent(store, times)
    assert len(query(store)) == len(times) + 1  # base + 3 corrected instants


def test_compaction_with_monthly_boundary_across_fall_back():
    times = [
        _ny("2024-10-31 23:30"),
        _ny("2024-11-01 00:30"),
        _ny("2024-11-02 12:00"),
        _ny("2024-11-04 12:00"),
    ]
    store = _dst_store(MonthlyBoundary(), times)
    _assert_compaction_is_transparent(store, times)


def test_compaction_then_as_of_with_different_transaction_times_same_timestamp():
    store = ChunkStore()
    store.ingest_row(T, "A", vals(10.0))
    store.flush_all()
    tx = [T + timedelta(days=d) for d in (3, 1, 2)]  # unsorted on purpose
    for close, when in zip((30.0, 10.5, 20.0), tx):
        store.ingest_correction(T, "A", vals(close), when)

    cutoffs = [
        None,
        T,
        T + timedelta(hours=12),
        T + timedelta(days=1),
        T + timedelta(days=2),
        T + timedelta(days=3),
        T + timedelta(days=9),
    ]
    before = [query(store, c) for c in cutoffs]
    assert store.compact_corrections() == 2

    merged = [c for c in store.chunks if c.label.transaction_time_start][0]
    assert merged.label.row_count == 3
    assert set(merged.correction_transaction_times) == set(tx)
    assert merged.label.transaction_time_start == T + timedelta(days=1)
    assert merged.label.transaction_time_end == T + timedelta(days=3)

    for cutoff, expected in zip(cutoffs, before):
        after = query(store, cutoff)
        pd.testing.assert_frame_equal(after, expected, check_exact=True)
    assert query(store, T + timedelta(days=1))["close"].tolist() == [10.5]
    assert query(store, T + timedelta(days=2))["close"].tolist() == [20.0]
    assert query(store, T + timedelta(days=3))["close"].tolist() == [30.0]
    assert query(store, T)["close"].tolist() == [10.0]
