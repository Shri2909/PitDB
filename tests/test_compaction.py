"""Correction-chunk compaction: fewer, larger chunks, identical query results.

Compaction is a pure storage-layout optimization -- it must never change
what any query (with any AS OF cutoff) sees, only how many chunks that
answer happens to live in. The property-based test below is the actual
"proof": the equivalence is checked across a generated space of random
correction sets and random AS OF cutoffs, not just a couple of hand-picked
examples.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.query.executor import QueryExecutor
from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary
from src.store.chunk_store import ChunkStore

UTC = timezone.utc


def _values(close: float) -> np.ndarray:
    return np.array(
        [close - 0.5, close + 1.0, close - 1.0, close, 1_000.0], dtype=np.float64
    )


def _build_store_with_corrections(
    day_offsets: list[int], transaction_lag_days: list[int]
) -> tuple[ChunkStore, list[datetime]]:
    """Ingest one base row per offset, then one correction per offset."""

    store = ChunkStore()
    base_day = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [base_day + timedelta(days=offset) for offset in day_offsets]

    for timestamp in timestamps:
        store.ingest_row(timestamp, "AAPL", _values(100.0))
    store.flush_all()

    transaction_times = []
    for timestamp, lag, close in zip(
        timestamps, transaction_lag_days, range(len(timestamps)), strict=True
    ):
        transaction_time = timestamp + timedelta(days=lag)
        store.ingest_correction(
            timestamp, "AAPL", _values(200.0 + close), transaction_time=transaction_time
        )
        transaction_times.append(transaction_time)

    return store, transaction_times


def _query_at(executor: QueryExecutor, as_of: datetime | None) -> pd.DataFrame:
    if as_of is None:
        sql = "SELECT * FROM data WHERE symbol = 'AAPL'"
    else:
        sql = (
            f"SELECT * FROM data TIMESTAMP AS OF '{as_of.isoformat()}' "
            "WHERE symbol = 'AAPL'"
        )
    return (
        executor.execute(sql)
        .data.sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )


def test_compact_corrections_reduces_chunk_count() -> None:
    store, _ = _build_store_with_corrections([0, 1, 2, 3, 4], [1, 1, 1, 1, 1])
    before_chunks = store.total_chunks
    before_rows = store.total_rows

    removed = store.compact_corrections()

    assert removed > 0
    assert store.total_chunks < before_chunks
    assert store.total_rows == before_rows


def test_compact_corrections_preserves_as_of_query_results() -> None:
    store, transaction_times = _build_store_with_corrections(
        [0, 1, 2, 3, 4], [1, 2, 3, 4, 5]
    )
    executor = QueryExecutor(store)

    as_of_values = [
        None,
        transaction_times[0] - timedelta(hours=1),
        transaction_times[0],
        transaction_times[2],
        transaction_times[-1] + timedelta(days=1),
    ]
    before = [_query_at(executor, as_of) for as_of in as_of_values]

    store.compact_corrections()

    after = [_query_at(executor, as_of) for as_of in as_of_values]

    for as_of, before_frame, after_frame in zip(
        as_of_values, before, after, strict=True
    ):
        pd.testing.assert_frame_equal(
            before_frame, after_frame, check_exact=True, obj=f"AS OF {as_of!r}"
        )


@settings(deadline=None, max_examples=50)
@given(
    day_offsets=st.lists(
        st.integers(min_value=0, max_value=60), min_size=1, max_size=10, unique=True
    ),
    lag_seed=st.integers(min_value=1, max_value=10),
    as_of_offset_days=st.integers(min_value=-5, max_value=65),
)
def test_compaction_preserves_query_results_property(
    day_offsets: list[int], lag_seed: int, as_of_offset_days: int
) -> None:
    """Property: for ANY correction set and ANY AS OF cutoff, compaction is
    invisible to query results -- the core claim this whole plan rests on."""

    day_offsets = sorted(day_offsets)
    transaction_lag_days = [lag_seed + index for index in range(len(day_offsets))]
    store, _ = _build_store_with_corrections(day_offsets, transaction_lag_days)
    executor = QueryExecutor(store)

    base_day = datetime(2024, 1, 1, tzinfo=UTC)
    as_of = base_day + timedelta(days=as_of_offset_days)

    before_current = _query_at(executor, None)
    before_as_of = _query_at(executor, as_of)

    store.compact_corrections()

    after_current = _query_at(executor, None)
    after_as_of = _query_at(executor, as_of)

    pd.testing.assert_frame_equal(before_current, after_current, check_exact=True)
    pd.testing.assert_frame_equal(before_as_of, after_as_of, check_exact=True)


def test_compact_corrections_respects_chunk_boundary() -> None:
    boundary = DurationBoundary(timedelta(days=10))
    store = ChunkStore(chunk_boundary=boundary)
    base_day = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [base_day + timedelta(days=offset) for offset in range(0, 30, 3)]

    for timestamp in timestamps:
        store.ingest_row(timestamp, "AAPL", _values(100.0))
    store.flush_all()

    for timestamp in timestamps:
        store.ingest_correction(
            timestamp,
            "AAPL",
            _values(200.0),
            transaction_time=timestamp + timedelta(days=1),
        )

    store.compact_corrections()

    for chunk in store.chunks:
        if chunk.label.transaction_time_start is None:
            continue
        assert not boundary.crosses(chunk.label.time_start, chunk.label.time_end)


def test_compact_corrections_is_idempotent() -> None:
    store, _ = _build_store_with_corrections([0, 1, 2], [1, 1, 1])

    first = store.compact_corrections()
    second = store.compact_corrections()

    assert first > 0
    assert second == 0


def test_compact_corrections_does_not_disturb_base_chunks() -> None:
    store, _ = _build_store_with_corrections([0, 1, 2], [1, 1, 1])
    base_chunk_ids_before = {
        chunk.label.chunk_id
        for chunk in store.chunks
        if chunk.label.transaction_time_start is None
    }
    base_chunks_before = {
        chunk.label.chunk_id: chunk
        for chunk in store.chunks
        if chunk.label.transaction_time_start is None
    }

    store.compact_corrections()

    base_chunk_ids_after = {
        chunk.label.chunk_id
        for chunk in store.chunks
        if chunk.label.transaction_time_start is None
    }
    assert base_chunk_ids_after == base_chunk_ids_before
    for chunk_id in base_chunk_ids_before:
        assert store.get_chunk(chunk_id) is base_chunks_before[chunk_id]


def test_compact_corrections_accepts_a_coarser_boundary_override() -> None:
    """Corrections spread thinly across many MonthlyBoundary windows have
    little to merge with under the default (store) boundary, but merge
    aggressively under a deliberately coarser override -- mirroring Hudi's
    and Delta's own configurable compaction batch/target-file-size knobs."""

    store = ChunkStore(chunk_boundary=MonthlyBoundary())
    base_day = datetime(2024, 1, 1, tzinfo=UTC)
    # One correction every ~2 months, spanning 2 years -> ~12 corrections,
    # each landing in a different monthly window, so the default boundary
    # has nothing to merge.
    offsets = list(range(0, 730, 60))
    timestamps = [base_day + timedelta(days=offset) for offset in offsets]

    for timestamp in timestamps:
        store.ingest_row(timestamp, "AAPL", _values(100.0))
    store.flush_all()
    for timestamp in timestamps:
        store.ingest_correction(
            timestamp,
            "AAPL",
            _values(200.0),
            transaction_time=timestamp + timedelta(days=1),
        )

    before_chunks = store.total_chunks
    removed_default = store.compact_corrections()
    assert removed_default == 0  # nothing shares a month, nothing merges

    # Re-build identical data in a fresh store, this time compacting with an
    # explicit multi-year boundary override.
    store2 = ChunkStore(chunk_boundary=MonthlyBoundary())
    for timestamp in timestamps:
        store2.ingest_row(timestamp, "AAPL", _values(100.0))
    store2.flush_all()
    for timestamp in timestamps:
        store2.ingest_correction(
            timestamp,
            "AAPL",
            _values(200.0),
            transaction_time=timestamp + timedelta(days=1),
        )
    removed_coarse = store2.compact_corrections(
        boundary=DurationBoundary(timedelta(days=1000))
    )

    assert removed_coarse > 0
    assert store2.total_chunks < before_chunks


def test_compact_corrections_can_be_scoped_to_one_symbol() -> None:
    store = ChunkStore(chunk_boundary=MonthlyBoundary())
    base_day = datetime(2024, 1, 1, tzinfo=UTC)
    for symbol in ("AAPL", "MSFT"):
        for offset in range(3):
            timestamp = base_day + timedelta(days=offset)
            store.ingest_row(timestamp, symbol, _values(100.0))
        store.flush_all()
        for offset in range(3):
            timestamp = base_day + timedelta(days=offset)
            store.ingest_correction(
                timestamp,
                symbol,
                _values(200.0),
                transaction_time=timestamp + timedelta(days=1),
            )

    removed = store.compact_corrections(symbol="AAPL")

    assert removed > 0
    msft_correction_chunks = [
        chunk
        for chunk in store.chunks
        if chunk.label.symbol == "MSFT"
        and chunk.label.transaction_time_start is not None
    ]
    assert len(msft_correction_chunks) == 3  # untouched, still one per correction
