"""Independent precision, empty-schema and publication regression tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.store.chunk_boundary import DurationBoundary
from src.store.chunk_store import ChunkStore


def values(close):
    return np.array([close, close, close, close, 1.0], dtype=np.float64)


def test_sql_preserves_nanosecond_timestamp_equality():
    store = ChunkStore()
    first = pd.Timestamp("2024-01-01T00:00:00.000000001Z")
    store.ingest_row(first, "A", values(10.0))
    store.ingest_row(first + pd.Timedelta(nanoseconds=1), "A", values(20.0))
    store.flush_all()
    result = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE timestamp = '2024-01-01T00:00:00.000000001Z'"
    )
    assert result.data["close"].tolist() == [10.0]


def test_as_of_nanosecond_boundary_never_reveals_next_revision():
    store = ChunkStore()
    event = pd.Timestamp("2024-01-01T00:00:00Z")
    store.ingest_row(event, "A", values(10.0))
    store.flush_all()
    store.ingest_correction(
        event, "A", values(20.0), event + pd.Timedelta(nanoseconds=2)
    )
    executor = QueryExecutor(store)
    assert executor.execute(
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-01T00:00:00.000000001Z'"
    ).data["close"].tolist() == [10.0]
    assert executor.execute(
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-01T00:00:00.000000002Z'"
    ).data["close"].tolist() == [20.0]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM data WHERE symbol = 'absent'",
        "SELECT * FROM data WHERE volume < 0",
        "SELECT * FROM data TIMESTAMP AS OF '2023-01-01T00:00:00Z'",
        "SELECT * FROM data TIMESTAMP AS OF '2023-01-01T00:00:00Z' "
        "WHERE timestamp >= '2024-01-01'",
    ],
)
def test_empty_schema_and_dtypes_agree_across_all_three_engines(sql):
    store = ChunkStore()
    stamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    store.ingest_row(stamp, "A", values(10.0))
    store.flush_all()
    results = [
        engine.execute(sql).data
        for engine in (
            FullScanQueryEngine(store),
            QueryExecutor(store),
            QueryExecutor(store, enable_pruning=False),
        )
    ]
    for frame in results:
        assert frame.empty
        assert list(frame.columns) == [
            "timestamp",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert str(frame.timestamp.dtype) == "datetime64[ns, UTC]"
        assert all(
            frame[name].dtype == np.float64
            for name in ("open", "high", "low", "close", "volume")
        )
    pd.testing.assert_frame_equal(results[0], results[1])
    pd.testing.assert_frame_equal(results[0], results[2])


def test_finalized_chunks_and_their_framing_are_immutable():
    store = ChunkStore()
    store.ingest_row(datetime(2024, 1, 1), "A", values(10.0))
    store.flush_all()
    chunk = store.chunks[0]
    with pytest.raises(FrozenInstanceError):
        chunk.column_data = b"bad"
    with pytest.raises(TypeError):
        chunk.column_names[0] = "symbol"
    with pytest.raises(TypeError):
        chunk.timestamps[0] = datetime(2000, 1, 1)
    arrays = chunk.column_arrays()
    arrays["close"][:] = 1000
    assert (
        QueryExecutor(store).execute("SELECT * FROM data WHERE close = 10").row_count
        == 1
    )


def test_duration_boundary_handles_full_supported_timestamp_range():
    store = ChunkStore(chunk_boundary=DurationBoundary(timedelta(hours=1)))
    for stamp in (pd.Timestamp("1700-01-01"), pd.Timestamp("2200-01-01")):
        store.ingest_row(stamp, "A", values(10.0))
    store.flush_all()
    assert store.total_chunks == 2
    assert store.total_rows == 2


@pytest.mark.parametrize("empty", [False, True])
def test_missing_physical_sql_column_is_rejected_consistently(empty):
    store = ChunkStore(value_columns=["metric", "close"])
    if not empty:
        store.ingest_row(datetime(2024, 1, 1), "A", np.array([1.0, 2.0]))
        store.flush_all()
    for engine in (
        FullScanQueryEngine(store),
        QueryExecutor(store),
        QueryExecutor(store, enable_pruning=False),
    ):
        with pytest.raises(ValueError, match="column.*open.*store schema"):
            engine.execute("SELECT * FROM data WHERE open > 0")


def test_rollover_publication_failure_preserves_old_buffer_and_rejects_new_row(
    monkeypatch,
):
    store = ChunkStore(chunk_boundary=DurationBoundary(timedelta(hours=1)))
    stamp = datetime(2024, 1, 1)
    store.ingest_row(stamp, "A", values(10.0))
    publish = store._add_chunk
    monkeypatch.setattr(
        store,
        "_add_chunk",
        lambda chunk: (_ for _ in ()).throw(ValueError("injected publication failure")),
    )
    with pytest.raises(ValueError, match="injected"):
        store.ingest_row(stamp + timedelta(hours=1), "A", values(20.0))
    monkeypatch.setattr(store, "_add_chunk", publish)
    store.ingest_row(stamp + timedelta(hours=1), "A", values(20.0))
    store.flush_all()
    assert store.full_scan()["close"].tolist() == [10.0, 20.0]


@pytest.mark.parametrize("literal", ["2024-03-10T02:30:00", "2024-11-03T01:30:00"])
def test_ambiguous_or_nonexistent_naive_local_times_are_rejected(literal):
    store = ChunkStore()
    store.ingest_row(
        pd.Timestamp("2024-01-01", tz="America/New_York"), "A", values(10.0)
    )
    store.flush_all()
    for suffix in (f" WHERE timestamp = '{literal}'", f" TIMESTAMP AS OF '{literal}'"):
        with pytest.raises(ValueError, match="ambiguous|invalid|nonexistent"):
            QueryExecutor(store).execute("SELECT * FROM data" + suffix)
