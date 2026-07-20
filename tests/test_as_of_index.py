"""The transaction-time index (ChunkStore.labels_visible_as_of) must produce
exactly the same candidate/skipped chunks as the plain linear scan it
replaces -- it's purely a faster way to compute the same partition, never a
different one. Proven via property-based equivalence across randomized
label sets and AS OF cutoffs, not just hand-picked examples.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from src.query.predicate import AlwaysTrue, Comparison, label_visible_as_of
from src.query.pushdown import evaluate_pushdown
from src.store.chunk_store import ChunkStore

UTC = timezone.utc


def _values(close: float) -> np.ndarray:
    return np.array(
        [close - 0.5, close + 1.0, close - 1.0, close, 1_000.0], dtype=np.float64
    )


def _build_mixed_store(
    base_day_offsets: list[int], correction_specs: list[tuple[int, int]]
) -> ChunkStore:
    """Ingest base rows at ``base_day_offsets``, then corrections at
    (day_offset, transaction_lag_days) pairs in ``correction_specs``."""

    store = ChunkStore()
    base_day = datetime(2024, 1, 1, tzinfo=UTC)

    for offset in base_day_offsets:
        store.ingest_row(
            base_day + timedelta(days=offset), "AAPL", _values(100.0 + offset)
        )
    store.flush_all()

    for offset, lag in correction_specs:
        timestamp = base_day + timedelta(days=offset)
        store.ingest_correction(
            timestamp,
            "AAPL",
            _values(200.0 + offset),
            transaction_time=timestamp + timedelta(days=lag),
        )
    return store


def test_labels_visible_as_of_matches_direct_linear_check() -> None:
    store = _build_mixed_store(
        base_day_offsets=[0, 1, 2, 3, 4],
        correction_specs=[(0, 1), (2, 3), (2, 5), (4, 1)],
    )
    base_day = datetime(2024, 1, 1, tzinfo=UTC)

    for offset_days in range(-2, 10):
        as_of = base_day + timedelta(days=offset_days)
        indexed = {label.chunk_id for label in store.labels_visible_as_of(as_of)}
        linear = {
            label.chunk_id
            for label in store.labels
            if label_visible_as_of(label, as_of)
        }
        assert indexed == linear, f"mismatch at as_of={as_of!r}"

        not_indexed = {
            label.chunk_id for label in store.labels_not_visible_as_of(as_of)
        }
        assert indexed | not_indexed == {label.chunk_id for label in store.labels}
        assert indexed.isdisjoint(not_indexed)


@settings(deadline=None, max_examples=50)
@given(
    base_day_offsets=st.lists(
        st.integers(min_value=0, max_value=30), min_size=1, max_size=8, unique=True
    ),
    correction_lags=st.lists(
        st.integers(min_value=1, max_value=10), min_size=0, max_size=8
    ),
    as_of_offset_days=st.integers(min_value=-5, max_value=45),
    price_threshold=st.floats(min_value=50.0, max_value=300.0),
)
def test_indexed_pushdown_matches_linear_pushdown_property(
    base_day_offsets: list[int],
    correction_lags: list[int],
    as_of_offset_days: int,
    price_threshold: float,
) -> None:
    """Property: the indexed evaluate_pushdown path and the linear-scan
    fallback must always partition chunks into the same candidate/skipped
    sets, IN THE SAME ORDER, for any label set and any AS OF cutoff -- this
    is the actual correctness claim behind introducing the index at all, and
    evaluate_pushdown's own docstring promises input order is preserved
    regardless of which path is used."""

    base_day_offsets = sorted(base_day_offsets)
    correction_specs = [
        (base_day_offsets[index % len(base_day_offsets)], lag)
        for index, lag in enumerate(correction_lags)
    ]
    store = _build_mixed_store(base_day_offsets, correction_specs)
    base_day = datetime(2024, 1, 1, tzinfo=UTC)
    as_of = base_day + timedelta(days=as_of_offset_days)

    for predicate in (
        AlwaysTrue(),
        Comparison("symbol", "=", "AAPL", "symbol"),
        Comparison("close", ">", price_threshold, "price"),
    ):
        linear_result = evaluate_pushdown(predicate, store.labels, as_of=as_of)
        indexed_result = evaluate_pushdown(
            predicate, store.labels, as_of=as_of, as_of_index=store
        )

        # List equality, not just set equality: evaluate_pushdown's docstring
        # promises input order is preserved regardless of which path ran.
        assert linear_result.candidate_chunk_ids == indexed_result.candidate_chunk_ids
        assert linear_result.skipped_chunk_ids == indexed_result.skipped_chunk_ids
        assert linear_result.total_chunks == indexed_result.total_chunks
        assert linear_result.candidate_count == indexed_result.candidate_count


def test_as_of_index_stays_correct_after_compaction() -> None:
    store = _build_mixed_store(
        base_day_offsets=[0, 1, 2, 3, 4],
        correction_specs=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
    )
    store.compact_corrections()

    base_day = datetime(2024, 1, 1, tzinfo=UTC)
    predicate = AlwaysTrue()

    for offset_days in range(-2, 12):
        as_of = base_day + timedelta(days=offset_days)
        linear_result = evaluate_pushdown(predicate, store.labels, as_of=as_of)
        indexed_result = evaluate_pushdown(
            predicate, store.labels, as_of=as_of, as_of_index=store
        )
        assert linear_result.candidate_chunk_ids == indexed_result.candidate_chunk_ids
        assert linear_result.skipped_chunk_ids == indexed_result.skipped_chunk_ids


def test_evaluate_pushdown_without_index_is_unaffected() -> None:
    """Backward compatibility: omitting as_of_index must be byte-identical to
    the pre-existing linear-scan-only behavior (this is what every existing
    test_pushdown.py test relies on)."""

    store = _build_mixed_store(base_day_offsets=[0, 1, 2], correction_specs=[(1, 2)])
    predicate = Comparison("symbol", "=", "AAPL", "symbol")

    result = evaluate_pushdown(predicate, store.labels)
    assert result.candidate_chunk_ids == [label.chunk_id for label in store.labels]
    assert result.skipped_chunk_ids == []
