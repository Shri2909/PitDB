"""Independent temporal oracle and regression probes.

Expected results use only the original input records, Python comparisons,
and a dictionary of visible versions. No parser, production predicate,
version resolver, chunk label, baseline engine, or decompressor computes
the expected query result.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.query.executor import QueryExecutor
from src.query.predicate import Comparison
from src.store.chunk_boundary import DurationBoundary
from src.store.chunk_store import ChunkStore

UTC = timezone.utc
EPOCH = datetime(2024, 1, 1, tzinfo=UTC)
COLS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class Record:
    timestamp: datetime
    symbol: str
    values: tuple[float, ...]
    transaction_time: datetime
    correction: bool
    sequence: int


def values(close):
    return (
        float(close) - 0.5,
        float(close) + 1,
        float(close) - 1,
        float(close),
        float(close) * 10 + 200,
    )


def add_base(store, records, timestamp, symbol, close):
    payload = values(close)
    store.ingest_row(timestamp, symbol, np.array(payload))
    records.append(Record(timestamp, symbol, payload, timestamp, False, len(records)))


def add_correction(store, records, timestamp, symbol, close, txn):
    payload = values(close)
    store.ingest_correction(timestamp, symbol, np.array(payload), txn)
    records.append(Record(timestamp, symbol, payload, txn, True, len(records)))


def expected(records, cutoff=None, matches=lambda row: True):
    winners = {}
    for record in records:
        if cutoff is not None and record.transaction_time > cutoff:
            continue
        key = (record.symbol, record.timestamp)
        rank = (record.transaction_time, record.correction, record.sequence)
        if key not in winners or rank > winners[key][0]:
            winners[key] = (rank, record)
    rows = [record for _, record in winners.values() if matches(record)]
    return sorted((r.timestamp, r.symbol, *r.values) for r in rows)


def actual(frame):
    return sorted(tuple(row) for row in frame.itertuples(index=False, name=None))


def sql(cutoff, where=""):
    clause = f" TIMESTAMP AS OF '{cutoff.isoformat()}'" if cutoff is not None else ""
    return f"SELECT * FROM data{clause}" + (f" WHERE {where}" if where else "")


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    closes=st.lists(st.integers(-20, 200), min_size=6, max_size=6),
    corrections=st.lists(
        st.tuples(st.integers(0, 5), st.integers(0, 6), st.integers(-20, 200)),
        max_size=10,
    ),
    cutoff_day=st.integers(-1, 12),
)
def test_raw_record_oracle_before_and_after_compaction(closes, corrections, cutoff_day):
    store = ChunkStore(chunk_boundary=DurationBoundary(timedelta(days=1)))
    records = []
    keys = [(EPOCH + timedelta(days=i // 2), ("AAPL", "MSFT")[i % 2]) for i in range(6)]
    for (ts, symbol), close in zip(keys, closes):
        add_base(store, records, ts, symbol, close)
    store.flush_all()
    for key_index, lag, close in corrections:
        ts, symbol = keys[key_index]
        add_correction(store, records, ts, symbol, close, ts + timedelta(days=lag))
    queries = [
        ("", lambda r: True),
        ("close >= 100", lambda r: r.values[3] >= 100),
        (
            "symbol = 'AAPL' AND timestamp >= '2024-01-02T00:00:00+00:00'",
            lambda r: r.symbol == "AAPL" and r.timestamp >= EPOCH + timedelta(days=1),
        ),
        (
            "open < 50 AND volume IN (200, 1200, 2200)",
            lambda r: r.values[0] < 50 and r.values[4] in (200, 1200, 2200),
        ),
    ]
    cutoffs = [
        None,
        EPOCH - timedelta(microseconds=1),
        EPOCH + timedelta(days=cutoff_day),
        EPOCH + timedelta(days=12),
    ]
    for compact in (False, True):
        if compact:
            store.compact_corrections()
        for cutoff in cutoffs:
            for where, matches in queries:
                result = QueryExecutor(store).execute(sql(cutoff, where))
                assert actual(result.data) == expected(records, cutoff, matches), (
                    compact,
                    cutoff,
                    where,
                    records,
                )


@settings(max_examples=40, deadline=None, derandomize=True)
@given(
    correction_lags=st.lists(st.integers(0, 12), max_size=12),
    cutoff_offset=st.integers(-1, 14),
)
def test_as_of_index_against_independent_raw_row_visibility(
    correction_lags, cutoff_offset
):
    store = ChunkStore(chunk_boundary=DurationBoundary(timedelta(days=1)))
    records = []
    for i in range(8):
        add_base(store, records, EPOCH + timedelta(days=i), "AAPL", i)
    store.flush_all()
    for i, lag in enumerate(correction_lags):
        ts = EPOCH + timedelta(days=i % 8)
        add_correction(store, records, ts, "AAPL", 100 + i, ts + timedelta(days=lag))
    cutoff = EPOCH + timedelta(days=cutoff_offset)
    for compact in (False, True):
        if compact:
            store.compact_corrections()
        # Independently scan each chunk's actual row transaction timestamps;
        # deliberately do not call label_visible_as_of or evaluate_pushdown.
        visible = set()
        for chunk in store.chunks:
            transaction_times = (
                chunk.timestamps
                if chunk.correction_transaction_times is None
                else chunk.correction_transaction_times
            )
            if any(t <= cutoff for t in transaction_times):
                visible.add(chunk.label.chunk_id)
        indexed = {label.chunk_id for label in store.labels_visible_as_of(cutoff)}
        invisible = {label.chunk_id for label in store.labels_not_visible_as_of(cutoff)}
        assert indexed == visible
        assert invisible == {c.label.chunk_id for c in store.chunks} - visible


@pytest.mark.parametrize(
    "operator_text,operation",
    [
        ("=", operator.eq),
        (">", operator.gt),
        (">=", operator.ge),
        ("<", operator.lt),
        ("<=", operator.le),
    ],
)
def test_price_zone_map_has_no_false_negatives(operator_text, operation):
    for prices in (
        (float("nan"),),
        (1.0, float("nan"), 7.0),
        (float("-inf"), 5.0, float("inf")),
    ):
        store = ChunkStore()
        for i, price in enumerate(prices):
            store.ingest_row(
                EPOCH + timedelta(seconds=i), "AAPL", np.array(values(price))
            )
        store.flush_all()
        for threshold in (-math.inf, -1.0, 1.0, 5.0, 7.0, 100.0, math.inf, math.nan):
            predicate = Comparison("close", operator_text, threshold, "price")
            if any(operation(price, threshold) for price in prices):
                assert predicate.evaluate_against_label(store.labels[0])


def test_cutoff_boundary_and_same_instant_correction_tie():
    store, records = ChunkStore(), []
    add_base(store, records, EPOCH, "AAPL", 100)
    store.flush_all()
    add_correction(store, records, EPOCH, "AAPL", 20, EPOCH)
    add_correction(
        store, records, EPOCH, "AAPL", 200, EPOCH + timedelta(microseconds=1)
    )
    for cutoff in (
        EPOCH - timedelta(microseconds=1),
        EPOCH,
        EPOCH + timedelta(microseconds=1),
    ):
        assert actual(QueryExecutor(store).execute(sql(cutoff)).data) == expected(
            records, cutoff
        )


def test_correction_only_keys_with_out_of_order_transaction_arrival():
    store, records = ChunkStore(), []
    for lag, close in ((6, 20), (1, 200), (3, 50)):
        add_correction(
            store, records, EPOCH, "AAPL", close, EPOCH + timedelta(days=lag)
        )
    add_correction(
        store,
        records,
        EPOCH + timedelta(hours=1),
        "MSFT",
        300,
        EPOCH + timedelta(days=2),
    )
    for compact in (False, True):
        if compact:
            store.compact_corrections()
        for day in (0, 1, 2, 3, 5, 6):
            cutoff = EPOCH + timedelta(days=day)
            for where, matches in (
                ("", lambda r: True),
                ("price > 100", lambda r: r.values[3] > 100),
            ):
                frame = QueryExecutor(store).execute(sql(cutoff, where)).data
                # Direct list comparison also verifies the resolver's
                # documented timestamp/symbol result ordering.
                assert list(frame.itertuples(index=False, name=None)) == expected(
                    records, cutoff, matches
                )
                assert list(frame.columns) == ["timestamp", "symbol", *COLS]
                if not frame.empty:
                    assert all(str(frame[c].dtype) == "float64" for c in COLS)


@pytest.mark.parametrize("aware_store", [False, True])
def test_as_of_timezone_literal_alignment(aware_store):
    timestamp = EPOCH if aware_store else EPOCH.replace(tzinfo=None)
    cutoff = (EPOCH.replace(tzinfo=None) if aware_store else EPOCH) + timedelta(days=1)
    store, records = ChunkStore(), []
    add_base(store, records, timestamp, "AAPL", 100)
    store.flush_all()
    # Existing exact-filter and label logic explicitly support matching
    # awareness to the stored timestamps. AS OF should use the same rule.
    result = QueryExecutor(store).execute(sql(cutoff))
    assert actual(result.data) == expected(records, timestamp + timedelta(days=1))


@pytest.mark.parametrize("as_of", [False, True])
def test_queries_handle_aware_timestamps_with_distinct_offsets(as_of):
    store, records = ChunkStore(), []
    add_base(store, records, EPOCH, "AAPL", 100)
    india = timezone(timedelta(hours=5, minutes=30))
    add_base(
        store, records, (EPOCH + timedelta(hours=1)).astimezone(india), "MSFT", 200
    )
    store.flush_all()
    cutoff = EPOCH + timedelta(days=1)
    query = (
        sql(cutoff) if as_of else sql(None, "timestamp >= '2024-01-01T00:00:00+00:00'")
    )
    result = QueryExecutor(store).execute(query)
    assert actual(result.data) == expected(records, cutoff)


def test_current_matches_as_of_after_all_allowed_duplicate_base_rows():
    store, records = ChunkStore(), []
    add_base(store, records, EPOCH, "AAPL", 100)
    add_base(store, records, EPOCH, "AAPL", 200)
    store.flush_all()
    cutoff = EPOCH + timedelta(days=1)
    as_of = QueryExecutor(store).execute(sql(cutoff)).data
    current = QueryExecutor(store).execute(sql(None)).data
    assert actual(as_of) == expected(records, cutoff)
    assert actual(current) == expected(records)


def test_unrelated_correction_does_not_change_duplicate_base_result():
    store, records = ChunkStore(), []
    add_base(store, records, EPOCH, "AAPL", 100)
    add_base(store, records, EPOCH, "AAPL", 200)
    store.flush_all()
    before = actual(QueryExecutor(store).execute(sql(None)).data)
    add_correction(store, records, EPOCH, "MSFT", 300, EPOCH + timedelta(days=1))
    after = [
        row
        for row in actual(QueryExecutor(store).execute(sql(None)).data)
        if row[1] == "AAPL"
    ]
    assert after == before


def test_as_of_duplicate_base_price_filter_is_independent_of_flush_boundaries():
    outputs = []
    for split in (False, True):
        store, records = ChunkStore(), []
        add_base(store, records, EPOCH, "AAPL", 100)
        if split:
            store.flush_all()
        add_base(store, records, EPOCH, "AAPL", 200)
        store.flush_all()
        cutoff = EPOCH + timedelta(days=1)
        result = QueryExecutor(store).execute(sql(cutoff, "close < 150"))
        outputs.append(actual(result.data))
        assert outputs[-1] == expected(records, cutoff, lambda r: r.values[3] < 150), (
            split
        )
    assert outputs[0] == outputs[1]


def test_naive_literal_uses_actual_named_timezone_offset():
    store, records = ChunkStore(), []
    timestamp = pd.Timestamp("2024-01-01 09:30:00", tz="America/New_York")
    add_base(store, records, timestamp, "AAPL", 100)
    store.flush_all()
    result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE timestamp = '2024-01-01T09:30:00'"
    )
    assert actual(result.data) == expected(records)


def test_time_pushdown_does_not_reuse_start_offset_across_dst():
    store, records = ChunkStore(), []
    first = pd.Timestamp("2024-03-01 09:30:00", tz="America/New_York")
    last = pd.Timestamp("2024-03-15 09:30:00", tz="America/New_York")
    add_base(store, records, first, "AAPL", 100)
    add_base(store, records, last, "AAPL", 200)
    store.flush_all()
    result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE timestamp = '2024-03-15T09:30:00'"
    )
    assert actual(result.data) == expected(
        records, matches=lambda r: r.timestamp == last
    )
