"""Bitemporal versioning: transaction-time corrections and AS OF queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.query.predicate import AlwaysTrue
from src.query.pushdown import evaluate_pushdown
from src.store.chunk_store import ChunkStore
from src.store.label import ChunkLabel

UTC = timezone.utc


def _values(close: float) -> np.ndarray:
    return np.array(
        [close - 0.5, close + 1.0, close - 1.0, close, 1_000.0], dtype=np.float64
    )


def _as_of_sql(as_of: datetime, where: str = "") -> str:
    clause = f" WHERE {where}" if where else ""
    return f"SELECT * FROM data TIMESTAMP AS OF '{as_of.isoformat()}'{clause}"


def test_correction_overrides_base_row_after_its_transaction_time() -> None:
    store = ChunkStore()
    event_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    store.ingest_row(event_time, "AAPL", _values(100.0))
    store.flush_all()

    correction_txn_time = datetime(2024, 3, 2, 12, 0, tzinfo=UTC)
    store.ingest_correction(
        event_time, "AAPL", _values(105.0), transaction_time=correction_txn_time
    )

    executor = QueryExecutor(store)

    before = executor.execute(
        _as_of_sql(correction_txn_time - timedelta(hours=1), "symbol = 'AAPL'")
    )
    assert before.data["close"].tolist() == [100.0]

    after = executor.execute(
        _as_of_sql(correction_txn_time + timedelta(hours=1), "symbol = 'AAPL'")
    )
    assert after.data["close"].tolist() == [105.0]

    no_as_of = executor.execute("SELECT * FROM data WHERE symbol = 'AAPL'")
    assert no_as_of.data["close"].tolist() == [105.0]


def test_correction_never_visible_before_its_transaction_time() -> None:
    """The core look-ahead regression: no leakage across the exact boundary."""

    store = ChunkStore()
    event_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    store.ingest_row(event_time, "AAPL", _values(100.0))
    store.flush_all()

    correction_txn_time = datetime(2024, 3, 2, 12, 0, tzinfo=UTC)
    store.ingest_correction(
        event_time, "AAPL", _values(105.0), transaction_time=correction_txn_time
    )
    executor = QueryExecutor(store)

    just_before = executor.execute(
        _as_of_sql(correction_txn_time - timedelta(microseconds=1), "symbol = 'AAPL'")
    )
    assert just_before.data["close"].tolist() == [100.0]

    exactly_at = executor.execute(_as_of_sql(correction_txn_time, "symbol = 'AAPL'"))
    assert exactly_at.data["close"].tolist() == [105.0]


def test_chained_corrections_resolve_to_latest_visible_version() -> None:
    store = ChunkStore()
    event_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    store.ingest_row(event_time, "AAPL", _values(100.0))
    store.flush_all()

    txn_times = [
        datetime(2024, 3, 2, tzinfo=UTC),
        datetime(2024, 3, 3, tzinfo=UTC),
        datetime(2024, 3, 4, tzinfo=UTC),
    ]
    closes = [101.0, 102.0, 103.0]
    for txn_time, close in zip(txn_times, closes, strict=True):
        store.ingest_correction(
            event_time, "AAPL", _values(close), transaction_time=txn_time
        )

    executor = QueryExecutor(store)

    # Before any correction: original base value.
    result = executor.execute(
        _as_of_sql(txn_times[0] - timedelta(hours=1), "symbol = 'AAPL'")
    )
    assert result.data["close"].tolist() == [100.0]

    # Between successive corrections: the most recent one already visible.
    for index, txn_time in enumerate(txn_times):
        result = executor.execute(_as_of_sql(txn_time, "symbol = 'AAPL'"))
        assert result.data["close"].tolist() == [closes[index]]

    # No AS OF: latest correction wins.
    result = executor.execute("SELECT * FROM data WHERE symbol = 'AAPL'")
    assert result.data["close"].tolist() == [closes[-1]]


def test_ingest_correction_rejects_transaction_time_before_event() -> None:
    store = ChunkStore()
    event_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)

    with pytest.raises(ValueError):
        store.ingest_correction(
            event_time,
            "AAPL",
            _values(100.0),
            transaction_time=event_time - timedelta(days=1),
        )


def test_correction_does_not_disturb_live_builder_causality() -> None:
    """A correction for an old timestamp must not perturb ordinary streaming."""

    store = ChunkStore()
    t0 = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    t1 = datetime(2024, 3, 1, 9, 31, tzinfo=UTC)
    store.ingest_row(t0, "AAPL", _values(100.0))
    store.ingest_row(t1, "AAPL", _values(101.0))

    store.ingest_correction(
        t0, "AAPL", _values(105.0), transaction_time=t1 + timedelta(days=1)
    )

    t2 = datetime(2024, 3, 1, 9, 32, tzinfo=UTC)
    store.ingest_row(t2, "AAPL", _values(102.0))
    store.flush_all()

    assert store.total_rows == 4  # t0, t1, t2 (base) + 1 correction chunk


def test_as_of_without_corrections_matches_plain_time_filter() -> None:
    """AS OF with zero corrections in play collapses to a plain time filter."""

    store = ChunkStore()
    base_day = datetime(2024, 3, 1, tzinfo=UTC)
    for offset in range(5):
        store.ingest_row(
            base_day + timedelta(days=offset), "AAPL", _values(100.0 + offset)
        )
    store.flush_all()

    cutoff = base_day + timedelta(days=2, hours=12)
    executor = QueryExecutor(store)

    as_of_result = executor.execute(_as_of_sql(cutoff))
    plain_filter_result = executor.execute(
        f"SELECT * FROM data WHERE timestamp <= '{cutoff.isoformat()}'"
    )

    pd.testing.assert_frame_equal(
        as_of_result.data.sort_values(["timestamp", "symbol"]).reset_index(drop=True),
        plain_filter_result.data.sort_values(["timestamp", "symbol"]).reset_index(
            drop=True
        ),
        check_exact=True,
    )
    assert as_of_result.row_count == 3  # offsets 0, 1, 2


def test_pushdown_prunes_correction_chunk_by_transaction_time() -> None:
    day = datetime(2024, 3, 1, tzinfo=UTC)
    base_label = ChunkLabel(
        symbol="AAPL",
        time_start=day,
        time_end=day,
        min_price=100.0,
        max_price=100.0,
        row_count=1,
        chunk_id=0,
    )
    correction_label = ChunkLabel(
        symbol="AAPL",
        time_start=day,
        time_end=day,
        min_price=105.0,
        max_price=105.0,
        row_count=1,
        chunk_id=1,
        transaction_time_start=day + timedelta(days=5),
        transaction_time_end=day + timedelta(days=5),
    )

    result = evaluate_pushdown(
        AlwaysTrue(), [base_label, correction_label], as_of=day + timedelta(days=1)
    )

    # The base label stays a candidate; the correction's later txn time hides it.
    assert result.candidate_chunk_ids == [0]
    assert result.skipped_chunk_ids == [1]

    result_after = evaluate_pushdown(
        AlwaysTrue(), [base_label, correction_label], as_of=day + timedelta(days=10)
    )
    assert result_after.candidate_chunk_ids == [0, 1]
    assert result_after.skipped_chunk_ids == []


def test_full_scan_matches_pushdown_with_as_of_and_corrections() -> None:
    store = ChunkStore()
    day = datetime(2024, 3, 1, tzinfo=UTC)
    timestamps = [day, day + timedelta(days=1), day + timedelta(days=2)]
    for symbol in ("AAPL", "MSFT"):
        for index, timestamp in enumerate(timestamps):
            store.ingest_row(timestamp, symbol, _values(100.0 + index))
        store.flush_all()

    correction_txn_time = day + timedelta(days=10)
    store.ingest_correction(
        timestamps[1], "AAPL", _values(999.0), transaction_time=correction_txn_time
    )

    full_scan = FullScanQueryEngine(store)
    pushdown = QueryExecutor(store)

    for as_of in (
        None,
        correction_txn_time - timedelta(hours=1),
        correction_txn_time + timedelta(hours=1),
    ):
        sql = (
            _as_of_sql(as_of, "close > 0")
            if as_of is not None
            else "SELECT * FROM data WHERE close > 0"
        )
        full_scan_result = full_scan.execute(sql)
        pushdown_result = pushdown.execute(sql)

        assert full_scan_result.row_count == pushdown_result.row_count
        pd.testing.assert_frame_equal(
            full_scan_result.data.reset_index(drop=True),
            pushdown_result.data.reset_index(drop=True),
            check_exact=True,
        )


def test_correction_failing_value_predicate_still_overrides_base_row() -> None:
    """A correction's own value must never be pruned by price pushdown."""

    store = ChunkStore()
    event_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    store.ingest_row(event_time, "AAPL", _values(100.0))
    store.flush_all()

    # The correction's close (40.0) fails `close > 50`; the base row (100.0) passes.
    store.ingest_correction(
        event_time,
        "AAPL",
        _values(40.0),
        transaction_time=event_time + timedelta(days=1),
    )

    full_scan = FullScanQueryEngine(store)
    pushdown = QueryExecutor(store)
    sql = "SELECT * FROM data WHERE close > 50"

    full_scan_result = full_scan.execute(sql)
    pushdown_result = pushdown.execute(sql)

    # The key must resolve to the correction and then be excluded, not the base row.
    assert full_scan_result.row_count == 0
    assert pushdown_result.row_count == 0
    pd.testing.assert_frame_equal(
        full_scan_result.data.reset_index(drop=True),
        pushdown_result.data.reset_index(drop=True),
        check_exact=True,
    )


def test_same_instant_correction_wins_tie_over_base_row() -> None:
    """An exact-tie transaction time must favor the correction."""

    store = ChunkStore()
    event_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    store.ingest_row(event_time, "AAPL", _values(100.0))
    store.flush_all()

    store.ingest_correction(
        event_time, "AAPL", _values(999.0), transaction_time=event_time
    )

    result = QueryExecutor(store).execute("SELECT * FROM data WHERE symbol = 'AAPL'")
    assert result.data["close"].tolist() == [999.0]


def test_same_instant_ties_resolve_independently_across_symbols() -> None:
    """A same-instant tie for one symbol must never leak into another symbol."""

    store = ChunkStore()
    event_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    symbols = ["AAPL", "MSFT", "GOOG"]
    for symbol in symbols:
        store.ingest_row(event_time, symbol, _values(100.0))
    store.flush_all()

    for index, symbol in enumerate(symbols):
        store.ingest_correction(
            event_time,
            symbol,
            _values(900.0 + index),
            transaction_time=event_time,
        )

    result = QueryExecutor(store).execute("SELECT * FROM data")
    closes_by_symbol = dict(zip(result.data["symbol"], result.data["close"]))
    assert closes_by_symbol == {"AAPL": 900.0, "MSFT": 901.0, "GOOG": 902.0}


def test_engines_agree_when_correction_exists_for_unrelated_symbol() -> None:
    """A correction for another symbol must not change row order in either engine."""

    store = ChunkStore()
    aapl_time = datetime(2024, 3, 1, 9, 30, tzinfo=UTC)
    store.ingest_row(aapl_time, "AAPL", _values(100.0))
    msft_time = datetime(2024, 3, 1, 9, 31, tzinfo=UTC)
    store.ingest_row(msft_time, "MSFT", _values(200.0))
    store.flush_all()

    # A correction only for MSFT -- irrelevant to a query scoped to AAPL.
    store.ingest_correction(
        msft_time,
        "MSFT",
        _values(999.0),
        transaction_time=msft_time + timedelta(days=1),
    )

    full_scan_result = FullScanQueryEngine(store).execute(
        "SELECT * FROM data WHERE symbol = 'AAPL'"
    )
    pushdown_result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE symbol = 'AAPL'"
    )

    assert full_scan_result.row_count == 1
    pd.testing.assert_frame_equal(
        full_scan_result.data.reset_index(drop=True),
        pushdown_result.data.reset_index(drop=True),
        check_exact=True,
    )
