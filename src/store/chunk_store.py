"""Top-level orchestration for streaming ingestion and chunk access."""

from __future__ import annotations

import bisect
import dataclasses
import functools
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from src.replay.replay_harness import ReplayHarness
from src.store.chunk import Chunk
from src.store.chunk_boundary import ChunkBoundary, MonthlyBoundary
from src.store.chunk_builder import DEFAULT_VALUE_COLUMNS, ChunkBuilder
from src.store.label import ChunkLabel
from src.store.validation import (
    align_timestamp,
    canonical_timestamp,
    timestamp_value,
    validate_value_columns,
)

_F = TypeVar("_F", bound=Callable[..., Any])


def _synchronized(method: _F) -> _F:
    """Run a ChunkStore method while holding the store's re-entrant lock."""

    @functools.wraps(method)
    def wrapper(self: ChunkStore, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


class ChunkStore:
    """Manage per-symbol builders and finalized raw column chunks (thread-safe)."""

    def __init__(
        self,
        chunk_boundary: ChunkBoundary = MonthlyBoundary(),
        value_columns: list[str] | None = None,
    ) -> None:
        if not callable(getattr(chunk_boundary, "crosses", None)):
            raise TypeError("chunk_boundary must provide a crosses() method")

        selected = validate_value_columns(value_columns, default=DEFAULT_VALUE_COLUMNS)

        self.chunk_boundary = chunk_boundary
        self.value_columns = selected
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._chunk_by_id: dict[int, Chunk] = {}
        self._builders: dict[str, ChunkBuilder] = {}
        self._chunk_id_counter = 0
        self._chunk_id_source = self._chunk_ids()
        # Parallel lists sorted by _as_of_key(label), for O(log n) AS OF lookups.
        self._as_of_keys: list[datetime] = []
        self._as_of_labels: list[ChunkLabel] = []
        self._correction_chunk_count = 0
        self._last_valid_timestamp: dict[str, datetime] = {}
        self._time_reference: pd.Timestamp | None = None
        self._has_base_duplicates = False

    def _chunk_ids(self) -> Iterator[int]:
        """Yield one globally unique, monotonically increasing chunk ID."""
        while True:
            chunk_id = self._chunk_id_counter
            self._chunk_id_counter += 1
            yield chunk_id

    @_synchronized
    def ingest_row(self, timestamp: datetime, symbol: str, values: np.ndarray) -> None:
        """Route one sequential row to its symbol's active builder."""
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")

        timestamp = canonical_timestamp(timestamp, self._time_reference)
        # Reject a tz mismatch before buffering: it would poison later flush_all().
        self._validate_as_of_key_tz(timestamp)
        last_timestamp = self._last_valid_timestamp.get(symbol)

        builder = self._builders.get(symbol)
        if builder is None:
            # A new builder can't see pre-flush_all() rows, so check order here.
            if last_timestamp is not None:
                try:
                    out_of_order = timestamp < last_timestamp
                except TypeError as exc:
                    raise ValueError(
                        "timestamp timezone awareness must match this "
                        "symbol's prior ingestion"
                    ) from exc
                if out_of_order:
                    raise ValueError(
                        "timestamps must be added in non-decreasing order; "
                        f"received {timestamp!r} after {last_timestamp!r} "
                        f"for symbol {symbol!r} across a flush_all() boundary"
                    )

            builder = ChunkBuilder(
                symbol=symbol,
                chunk_boundary=self.chunk_boundary,
                value_columns=self.value_columns,
                chunk_id_counter=self._chunk_id_source,
            )
        # Publish the old window before the builder discards its buffer.
        builder.add_row(timestamp, values, publish=self._add_chunk)
        self._builders[symbol] = builder
        if last_timestamp is not None and timestamp == last_timestamp:
            self._has_base_duplicates = True
        self._last_valid_timestamp[symbol] = timestamp
        if self._time_reference is None:
            self._time_reference = timestamp

    @_synchronized
    def ingest_correction(
        self,
        timestamp: datetime,
        symbol: str,
        values: np.ndarray,
        transaction_time: datetime,
    ) -> None:
        """Append a bitemporal correction for one (symbol, timestamp) row.

        Bypasses the live per-symbol builders; a correction cannot be learned
        before the event itself happened.
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        timestamp = canonical_timestamp(timestamp, self._time_reference)
        transaction_time = canonical_timestamp(
            transaction_time,
            self._time_reference if self._time_reference is not None else timestamp,
            name="transaction_time",
        )

        try:
            precedes_event = transaction_time < timestamp
        except TypeError as exc:
            raise ValueError(
                "transaction_time timezone awareness must match timestamp"
            ) from exc
        if precedes_event:
            raise ValueError(
                "transaction_time cannot precede the event's own timestamp; "
                f"received transaction_time={transaction_time!r} "
                f"for timestamp={timestamp!r}"
            )
        # Check awareness against the store before a chunk ID is consumed.
        self._validate_as_of_key_tz(timestamp)

        builder = ChunkBuilder(
            symbol=symbol,
            chunk_boundary=self.chunk_boundary,
            value_columns=self.value_columns,
            chunk_id_counter=self._chunk_id_source,
        )
        builder.add_row(timestamp, values)
        chunk = builder.flush()
        assert chunk is not None  # exactly one row was just added

        versioned_label = dataclasses.replace(
            chunk.label,
            transaction_time_start=transaction_time,
            transaction_time_end=transaction_time,
        )
        self._add_chunk(
            dataclasses.replace(
                chunk,
                label=versioned_label,
                correction_transaction_times=[transaction_time],
            )
        )
        if self._time_reference is None:
            self._time_reference = timestamp

    @_synchronized
    def compact_corrections(
        self, symbol: str | None = None, boundary: ChunkBoundary | None = None
    ) -> int:
        """Merge correction chunks per symbol and return the net chunk-count reduction.

        Base chunks are untouched. ``boundary`` defaults to the store's own;
        a coarser one merges more aggressively.
        """
        merge_boundary = boundary if boundary is not None else self.chunk_boundary
        groups: dict[str, list[Chunk]] = {}
        for chunk in self._chunks:
            if chunk.label.transaction_time_start is None:
                continue
            if symbol is not None and chunk.label.symbol != symbol:
                continue
            groups.setdefault(chunk.label.symbol, []).append(chunk)

        replacements: list[tuple[set[int], list[Chunk]]] = []
        for group_symbol, chunks in groups.items():
            if len(chunks) < 2:
                continue

            rows: list[tuple[datetime, np.ndarray, datetime]] = []
            for chunk in chunks:
                assert chunk.correction_transaction_times is not None
                # column_arrays() avoids decompress()'s costly DataFrame build.
                columns = chunk.column_arrays()
                for position, transaction_time in enumerate(
                    chunk.correction_transaction_times
                ):
                    row_values = np.array(
                        [columns[name][position] for name in self.value_columns],
                        dtype=np.float64,
                    )
                    rows.append(
                        (chunk.timestamps[position], row_values, transaction_time)
                    )
            rows.sort(key=lambda row: row[0])

            builder = ChunkBuilder(
                symbol=group_symbol,
                chunk_boundary=merge_boundary,
                value_columns=self.value_columns,
                chunk_id_counter=self._chunk_id_source,
            )
            # add_row returns the previous window's chunk, so the pending times
            # are exactly that window's transaction times.
            merged_chunks: list[Chunk] = []
            pending_transaction_times: list[datetime] = []
            for timestamp, row_values, transaction_time in rows:
                completed = builder.add_row(timestamp, row_values)
                if completed is not None:
                    assert len(pending_transaction_times) == completed.label.row_count
                    merged_chunks.append(
                        self._label_as_correction(completed, pending_transaction_times)
                    )
                    pending_transaction_times = []
                pending_transaction_times.append(transaction_time)
            tail = builder.flush()
            if tail is not None:
                assert len(pending_transaction_times) == tail.label.row_count
                merged_chunks.append(
                    self._label_as_correction(tail, pending_transaction_times)
                )

            if len(merged_chunks) < len(chunks):
                replacements.append(
                    ({chunk.label.chunk_id for chunk in chunks}, merged_chunks)
                )

        if not replacements:
            return 0
        # Work on copies so a publication failure restores the previous state.
        previous = (
            self._chunks,
            self._chunk_by_id,
            self._as_of_keys,
            self._as_of_labels,
            self._correction_chunk_count,
        )
        working = (
            list(self._chunks),
            dict(self._chunk_by_id),
            list(self._as_of_keys),
            list(self._as_of_labels),
        )
        try:
            self._chunks, self._chunk_by_id, self._as_of_keys, self._as_of_labels = (
                working
            )
            for old_ids, merged_chunks in replacements:
                self._remove_chunks(old_ids)
                for merged_chunk in merged_chunks:
                    self._add_chunk(merged_chunk)
        except BaseException:
            (
                self._chunks,
                self._chunk_by_id,
                self._as_of_keys,
                self._as_of_labels,
                self._correction_chunk_count,
            ) = previous
            raise
        return sum(len(old_ids) - len(merged) for old_ids, merged in replacements)

    @staticmethod
    def _label_as_correction(chunk: Chunk, transaction_times: list[datetime]) -> Chunk:
        versioned_label = dataclasses.replace(
            chunk.label,
            transaction_time_start=min(transaction_times),
            transaction_time_end=max(transaction_times),
        )
        return dataclasses.replace(
            chunk, label=versioned_label, correction_transaction_times=transaction_times
        )

    def _remove_chunks(self, chunk_ids: set[int]) -> None:
        """Remove many chunks in one O(n) pass."""
        if not chunk_ids:
            return
        missing = chunk_ids - self._chunk_by_id.keys()
        if missing:
            raise KeyError(f"chunk ID(s) not found: {sorted(missing)}")

        removed_correction_count = sum(
            1
            for chunk_id in chunk_ids
            if self._chunk_by_id[chunk_id].label.transaction_time_start is not None
        )

        self._chunks = [
            chunk for chunk in self._chunks if chunk.label.chunk_id not in chunk_ids
        ]
        for chunk_id in chunk_ids:
            del self._chunk_by_id[chunk_id]

        kept_keys: list[datetime] = []
        kept_labels: list[ChunkLabel] = []
        if len(self._as_of_keys) != len(self._as_of_labels):
            raise RuntimeError("AS OF index keys and labels are misaligned")
        for key, label in zip(self._as_of_keys, self._as_of_labels):
            if label.chunk_id not in chunk_ids:
                kept_keys.append(key)
                kept_labels.append(label)
        self._as_of_keys = kept_keys
        self._as_of_labels = kept_labels
        self._correction_chunk_count -= removed_correction_count

    @_synchronized
    def flush_all(self) -> None:
        """Finalize every active symbol and end the current ingestion session."""
        # Builders are removed one by one so a mid-way failure leaves accurate state.
        for symbol in list(self._builders.keys()):
            builder = self._builders[symbol]
            if builder._row_count:
                completed = builder.finalize()
                self._add_chunk(completed)
            del self._builders[symbol]

    def _add_chunk(self, chunk: Chunk) -> None:
        # Compute the insertion position first: it can raise, and a chunk left
        # out of the AS OF index would be a silent pushdown false negative.
        key = self._as_of_key(chunk.label)
        self._validate_as_of_key_tz(key)
        position = bisect.bisect_right(self._as_of_keys, key)

        self._chunks.append(chunk)
        self._chunk_by_id[chunk.label.chunk_id] = chunk
        self._as_of_keys.insert(position, key)
        self._as_of_labels.insert(position, chunk.label)
        if chunk.label.transaction_time_start is not None:
            self._correction_chunk_count += 1

    def _validate_as_of_key_tz(self, key: datetime) -> None:
        """Raise a clear error on a store-wide tz-awareness mismatch."""
        key = timestamp_value(key, name="AS OF index key")
        reference = self._as_of_keys[0] if self._as_of_keys else self._time_reference
        if reference is None:
            return
        existing_is_aware = reference.utcoffset() is not None
        new_is_aware = key.utcoffset() is not None
        if existing_is_aware != new_is_aware:
            new_kind = "timezone-aware" if new_is_aware else "naive"
            existing_kind = "timezone-aware" if existing_is_aware else "naive"
            raise ValueError(
                "timestamp timezone awareness must be consistent across the "
                f"entire store; received a {new_kind} timestamp but the "
                f"store already holds {existing_kind} timestamps"
            )

    @staticmethod
    def _as_of_key(label: ChunkLabel) -> datetime:
        return (
            label.transaction_time_start
            if label.transaction_time_start is not None
            else label.time_start
        )

    @_synchronized
    def labels_visible_as_of(self, as_of: datetime) -> list[ChunkLabel]:
        """Return every label that might hold a row visible as of a time."""
        as_of = self.align_as_of(as_of)
        cut = bisect.bisect_right(self._as_of_keys, as_of)
        return self._as_of_labels[:cut]

    @_synchronized
    def labels_not_visible_as_of(self, as_of: datetime) -> list[ChunkLabel]:
        """Return the complement of ``labels_visible_as_of``."""
        as_of = self.align_as_of(as_of)
        cut = bisect.bisect_right(self._as_of_keys, as_of)
        return self._as_of_labels[cut:]

    @_synchronized
    def align_as_of(self, as_of: datetime) -> pd.Timestamp:
        """Use the same timezone policy for indexed and exact AS OF checks."""
        reference = self._time_reference
        if reference is None and self._as_of_keys:
            reference = self._as_of_keys[0]
        return (
            timestamp_value(as_of, name="AS OF cutoff")
            if reference is None
            else align_timestamp(reference, as_of)
        )

    @property
    @_synchronized
    def has_base_duplicates(self) -> bool:
        """Whether accepted base keys require last-ingested version resolution."""
        return self._has_base_duplicates

    @_synchronized
    def ingest_dataframe(self, df: pd.DataFrame) -> None:
        """Ingest a DataFrame through the same sequential replay contract."""
        harness = ReplayHarness(df, value_columns=self.value_columns)
        self.ingest_from_replay(harness)

    @_synchronized
    def ingest_from_replay(self, harness: ReplayHarness) -> None:
        """Canonical no-lookahead ingestion path; atomic (rolls back on any failure)."""
        snapshot = self._snapshot()
        try:
            for row in harness:
                self.ingest_row(
                    row["timestamp"],
                    row["symbol"],
                    row["values"],
                )
            self.flush_all()
        except BaseException:
            self._restore(snapshot)
            raise

    def _snapshot(self) -> dict[str, Any]:
        return {
            "chunks": list(self._chunks),
            "chunk_by_id": dict(self._chunk_by_id),
            "as_of_keys": list(self._as_of_keys),
            "as_of_labels": list(self._as_of_labels),
            "correction_chunk_count": self._correction_chunk_count,
            "chunk_id_counter": self._chunk_id_counter,
            "builders": {
                symbol: (builder, builder._capture_state())
                for symbol, builder in self._builders.items()
            },
            "last_valid_timestamp": dict(self._last_valid_timestamp),
            "time_reference": self._time_reference,
            "has_base_duplicates": self._has_base_duplicates,
        }

    def _restore(self, snapshot: dict[str, Any]) -> None:
        self._chunks = snapshot["chunks"]
        self._chunk_by_id = snapshot["chunk_by_id"]
        self._as_of_keys = snapshot["as_of_keys"]
        self._as_of_labels = snapshot["as_of_labels"]
        self._correction_chunk_count = snapshot["correction_chunk_count"]
        self._chunk_id_counter = snapshot["chunk_id_counter"]
        self._builders = {}
        for symbol, (builder, state) in snapshot["builders"].items():
            builder._restore_state(state)
            self._builders[symbol] = builder
        self._last_valid_timestamp = snapshot["last_valid_timestamp"]
        self._time_reference = snapshot["time_reference"]
        self._has_base_duplicates = snapshot["has_base_duplicates"]

    @property
    @_synchronized
    def chunks(self) -> list[Chunk]:
        """Return a shallow copy so callers cannot mutate store membership."""
        return list(self._chunks)

    @property
    @_synchronized
    def labels(self) -> list[ChunkLabel]:
        return [chunk.label for chunk in self._chunks]

    @property
    @_synchronized
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    @_synchronized
    def has_corrections(self) -> bool:
        """Whether any correction chunk exists (disables price pushdown pruning)."""
        return self._correction_chunk_count > 0

    @property
    @_synchronized
    def total_rows(self) -> int:
        return sum(chunk.label.row_count for chunk in self._chunks)

    @property
    @_synchronized
    def total_column_bytes(self) -> int:
        return sum(len(chunk.column_data) for chunk in self._chunks)

    @_synchronized
    def get_chunk(self, chunk_id: int) -> Chunk:
        try:
            return self._chunk_by_id[chunk_id]
        except KeyError:
            raise KeyError(f"chunk ID not found: {chunk_id}") from None

    @_synchronized
    def empty_frame(self) -> pd.DataFrame:
        """Return the same public schema/dtypes when no chunk is selected."""
        timezone = None if self._time_reference is None else self._time_reference.tz
        data = {
            "timestamp": pd.Series(pd.DatetimeIndex([], tz=timezone)),
            "symbol": pd.Series([], dtype=object),
        }
        data.update(
            {name: pd.Series([], dtype=np.float64) for name in self.value_columns}
        )
        return pd.DataFrame(data)

    def full_scan(
        self,
        predicate_fn: Callable[[pd.Series], bool] | None = None,
    ) -> pd.DataFrame:
        """Decompress every chunk and optionally apply an exact row predicate."""

        if predicate_fn is not None and not callable(predicate_fn):
            raise TypeError("predicate_fn must be callable or None")

        # Chunks are immutable, so the lock need not be held while decompressing.
        chunks = self.chunks
        if not chunks:
            return self.empty_frame()

        # Intentionally naive: decompress everything before filtering.
        frame = pd.concat(
            [chunk.decompress() for chunk in chunks],
            ignore_index=True,
        )
        if predicate_fn is None:
            return frame

        mask = frame.apply(lambda row: bool(predicate_fn(row)), axis=1)
        return frame.loc[mask].reset_index(drop=True)


__all__ = ["ChunkStore"]
