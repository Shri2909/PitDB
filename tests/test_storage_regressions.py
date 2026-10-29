"""Storage invariants and regression tests from the independent audit."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.query.executor import QueryExecutor
from src.replay.data_loader import load_dataset
from src.replay.replay_harness import ReplayHarness
from src.store.chunk import Chunk
from src.store.chunk_boundary import DurationBoundary
from src.store.chunk_builder import ChunkBuilder
from src.store.chunk_store import ChunkStore

UTC = timezone.utc
T = datetime(2024, 1, 1, tzinfo=UTC)


def values(close=10.0):
    return np.array(
        [close - 0.5, close + 1.0, close - 1.0, close, 100.0], dtype=np.float64
    )


def result(store, cutoff=None):
    clause = "" if cutoff is None else f" TIMESTAMP AS OF '{cutoff.isoformat()}'"
    return QueryExecutor(store).execute("SELECT * FROM data" + clause).data


def raw_reference(records, cutoff=None):
    """Independent raw-record oracle: no production version/predicate helpers."""
    winners = {}
    for seq, (event, symbol, nums, transaction, correction) in enumerate(records):
        effective = event if transaction is None else transaction
        if cutoff is not None and effective > cutoff:
            continue
        key = (symbol, event)
        ranking = (effective, correction, seq)
        if key not in winners or ranking > winners[key][0]:
            winners[key] = (ranking, event, symbol, nums)
    entries = sorted(winners.values(), key=lambda item: (item[1], item[2]))
    return pd.DataFrame(
        [
            {
                "timestamp": event,
                "symbol": symbol,
                **dict(zip(("open", "high", "low", "close", "volume"), nums)),
            }
            for _, event, symbol, nums in entries
        ],
        columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"],
    )


@pytest.mark.parametrize("order", [(0, 1, 2, 3), (3, 1, 0, 2), (2, 0, 3, 1)])
def test_compaction_matches_independent_oracle_at_boundaries(order):
    store = ChunkStore()
    records = []
    for offset in (0, 1, 32):
        event, nums = T + timedelta(days=offset), values(10 + offset)
        store.ingest_row(event, "A", nums)
        records.append((event, "A", nums.copy(), None, False))
    store.flush_all()
    base_chunks = [chunk for chunk in store.chunks]
    corrections = [
        (T, 20.0, T + timedelta(days=2)),
        (T, 21.0, T + timedelta(days=2)),
        (T + timedelta(days=1), 30.0, T + timedelta(days=1)),
        (T + timedelta(days=40), 40.0, T + timedelta(days=41)),
    ]
    for index in order:
        event, price, transaction = corrections[index]
        nums = values(price)
        store.ingest_correction(event, "A", nums, transaction)
        records.append((event, "A", nums.copy(), transaction, True))
    cutoffs = [
        T,
        T + timedelta(days=2) - timedelta(microseconds=1),
        T + timedelta(days=2),
        T + timedelta(days=41),
        None,
    ]
    expected = [raw_reference(records, cutoff) for cutoff in cutoffs]
    for _ in range(3):
        for cutoff, frame in zip(cutoffs, expected):
            pd.testing.assert_frame_equal(
                result(store, cutoff), frame, check_exact=True
            )
        store.compact_corrections(boundary=DurationBoundary(timedelta(days=365)))
        assert all(
            store.get_chunk(chunk.label.chunk_id) is chunk for chunk in base_chunks
        )
        assert len(store._as_of_keys) == len(store._as_of_labels) == store.total_chunks
        assert store._correction_chunk_count == sum(
            ch.label.transaction_time_start is not None for ch in store.chunks
        )


def test_flush_publish_failure_retains_accepted_rows(monkeypatch):
    store = ChunkStore()
    store.ingest_row(T, "A", values())
    publish = store._add_chunk
    monkeypatch.setattr(
        store,
        "_add_chunk",
        lambda chunk: (_ for _ in ()).throw(ValueError("injected index rejection")),
    )
    with pytest.raises(ValueError, match="injected index rejection"):
        store.flush_all()
    monkeypatch.setattr(store, "_add_chunk", publish)
    store.flush_all()
    # Source flush_all docstring says failed builders accurately preserve state.
    assert store.total_rows == 1, "accepted row was erased before publication"


def test_timezone_rejected_flush_retains_buffered_rows():
    store = ChunkStore()
    store.ingest_row(T.replace(tzinfo=None), "A", values())
    store.flush_all()
    store.ingest_row(T, "B", values(20))
    with pytest.raises(ValueError, match="timezone awareness"):
        store.flush_all()
    assert store._builders["B"]._row_count == 1, (
        "rejected flush erased accepted buffered row"
    )


def test_compaction_publish_failure_preserves_historical_rows(monkeypatch):
    store = ChunkStore()
    store.ingest_row(T, "A", values())
    store.flush_all()
    store.ingest_correction(T, "A", values(20), T + timedelta(days=1))
    store.ingest_correction(T, "A", values(30), T + timedelta(days=2))
    before = result(store, T + timedelta(days=1))
    publish = store._add_chunk
    monkeypatch.setattr(
        store,
        "_add_chunk",
        lambda chunk: (_ for _ in ()).throw(ValueError("injected index rejection")),
    )
    with pytest.raises(ValueError, match="injected index rejection"):
        store.compact_corrections()
    monkeypatch.setattr(store, "_add_chunk", publish)
    pd.testing.assert_frame_equal(
        result(store, T + timedelta(days=1)), before, check_exact=True
    )


@pytest.mark.parametrize(
    "reserved", ["timestamp", "symbol", "_txn_time", "_is_correction"]
)
def test_custom_value_columns_reject_reserved_engine_columns(reserved):
    with pytest.raises(ValueError, match="reserved|column"):
        ChunkStore(value_columns=["close", reserved])


@pytest.mark.parametrize("which", ["event", "transaction"])
def test_reject_nat_ingestion(which):
    store = ChunkStore()
    with pytest.raises((ValueError, TypeError)):
        if which == "event":
            store.ingest_row(pd.NaT, "A", values())
        else:
            store.ingest_correction(T, "A", values(), pd.NaT)


def test_input_and_decoded_arrays_are_detached():
    nums = values()
    store = ChunkStore()
    store.ingest_row(T, "A", nums)
    nums[:] = 999.0
    store.flush_all()
    chunk = store.chunks[0]
    arrays = chunk.column_arrays()
    arrays["close"][:] = 888.0
    assert chunk.decompress()["close"].tolist() == [10.0]
    chunks = store.chunks
    chunks.clear()
    assert store.total_chunks == 1


def test_replay_value_copy_and_documented_shared_source():
    frame = pd.DataFrame(
        [
            dict(
                timestamp=T,
                symbol="A",
                **dict(zip(("open", "high", "low", "close", "volume"), values())),
            )
        ]
    )
    harness = ReplayHarness(frame)
    row = next(harness)
    row["values"][:] = 999.0
    assert frame["close"].tolist() == [10.0]
    harness.reset()
    frame.loc[0, "close"] = 20.0
    assert next(harness)["values"][3] == 20.0  # explicitly documented shared source


def test_per_symbol_ordering_across_flush_and_rejection():
    store = ChunkStore()
    store.ingest_row(T, "A", values())
    store.flush_all()
    with pytest.raises(ValueError, match="non-decreasing"):
        store.ingest_row(T - timedelta(seconds=1), "A", values())
    store.ingest_row(T + timedelta(seconds=1), "A", values(20))
    store.flush_all()
    assert store.full_scan()["close"].tolist() == [10.0, 20.0]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda kwargs: kwargs.update(column_data=kwargs["column_data"] + b"x"),
        lambda kwargs: kwargs.update(column_byte_lengths=[7, 9, 8, 8, 8]),
        lambda kwargs: kwargs.update(timestamps=[]),
        lambda kwargs: kwargs.update(column_names=["close"] * 5),
        lambda kwargs: kwargs.update(correction_transaction_times=[T]),
    ],
)
def test_malformed_chunk_framing_rejected(mutator):
    builder = ChunkBuilder("A")
    builder.add_row(T, values())
    good = builder.flush()
    kwargs = {
        field.name: getattr(good, field.name) for field in dataclasses.fields(Chunk)
    }
    mutator(kwargs)
    with pytest.raises((ValueError, TypeError)):
        Chunk(**kwargs).decompress()


def test_valid_custom_value_column_round_trip():
    store = ChunkStore(value_columns=["metric", "close"])
    store.ingest_row(T, "A", np.array([7.0, 10.0], dtype=np.float64))
    store.flush_all()
    data = store.full_scan()
    assert list(data.columns) == ["timestamp", "symbol", "metric", "close"]
    assert data["metric"].tolist() == [7.0]
    assert data["close"].tolist() == [10.0]


def test_loader_missing_symbol_is_rejected_before_replay(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-01", "2024-01-02"],
            "symbol": ["A", None],
            **{
                name: [10.0, 11.0]
                for name in ("open", "high", "low", "close", "volume")
            },
        }
    )
    path = tmp_path / "null-symbol.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="symbol"):
        load_dataset(path)


def test_loader_rejects_ambiguous_and_out_of_bounds_dates(tmp_path):
    for stamp in ["01/02/2024", "2500-01-01", None]:
        frame = pd.DataFrame(
            {
                "timestamp": [stamp, "2024-01-02"],
                "symbol": ["A", "A"],
                **{
                    name: [10.0, 11.0]
                    for name in ("open", "high", "low", "close", "volume")
                },
            }
        )
        path = tmp_path / "bad-date.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_dataset(path)


def test_stdlib_dst_fold_accepts_chronological_sequence():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    earlier = datetime(2024, 11, 3, 1, 45, tzinfo=zone, fold=0)
    later = datetime(2024, 11, 3, 1, 15, tzinfo=zone, fold=1)
    assert earlier.astimezone(UTC) < later.astimezone(UTC)
    store = ChunkStore()
    store.ingest_row(earlier, "A", values())
    store.ingest_row(later, "A", values(20))
    store.flush_all()
    assert store.total_rows == 2


def test_stdlib_dst_fold_rejects_reverse_absolute_order():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    later = datetime(2024, 11, 3, 1, 45, tzinfo=zone, fold=1)
    earlier = datetime(2024, 11, 3, 1, 50, tzinfo=zone, fold=0)
    assert later.astimezone(UTC) > earlier.astimezone(UTC)
    store = ChunkStore()
    store.ingest_row(later, "A", values())
    with pytest.raises(ValueError, match="non-decreasing"):
        store.ingest_row(earlier, "A", values(20))


def test_stdlib_dst_fold_correction_cannot_precede_event():
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    event = datetime(2024, 11, 3, 1, 15, tzinfo=zone, fold=1)
    learned = datetime(2024, 11, 3, 1, 45, tzinfo=zone, fold=0)
    assert learned.astimezone(UTC) < event.astimezone(UTC)
    store = ChunkStore()
    with pytest.raises(ValueError, match="cannot precede"):
        store.ingest_correction(event, "A", values(), learned)
