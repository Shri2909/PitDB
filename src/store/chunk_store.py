"""Top-level orchestration for streaming ingestion and chunk access."""

from __future__ import annotations

import bisect
import dataclasses
from collections.abc import Callable, Iterator
from datetime import datetime

import numpy as np
import pandas as pd

from src.replay.replay_harness import ReplayHarness
from src.store.chunk import Chunk
from src.store.chunk_boundary import ChunkBoundary, MonthlyBoundary
from src.store.chunk_builder import DEFAULT_VALUE_COLUMNS, ChunkBuilder
from src.store.label import ChunkLabel


class ChunkStore:
    """Manage per-symbol builders and finalized compressed chunks."""

    def __init__(
        self,
        chunk_boundary: ChunkBoundary = MonthlyBoundary(),
        value_columns: list[str] | None = None,
    ) -> None:
        if not callable(getattr(chunk_boundary, "crosses", None)):
            raise TypeError("chunk_boundary must provide a crosses() method")

        selected = (
            list(DEFAULT_VALUE_COLUMNS)
            if value_columns is None
            else list(value_columns)
        )
        if not selected:
            raise ValueError("value_columns must not be empty")
        if any(not isinstance(column, str) or not column for column in selected):
            raise ValueError("every value column must be a non-empty string")
        if len(selected) != len(set(selected)):
            raise ValueError("value_columns must not contain duplicates")
        if "close" not in selected:
            raise ValueError("value_columns must include 'close' for price labels")

        self.chunk_boundary = chunk_boundary
        self.value_columns = selected
        self._chunks: list[Chunk] = []
        self._chunk_by_id: dict[int, Chunk] = {}
        self._builders: dict[str, ChunkBuilder] = {}
        self._chunk_id_counter = 0
        self._chunk_id_source = self._chunk_ids()
        # Sorted together by _as_of_key(label): transaction_time_start for a
        # correction chunk, else time_start. Lets labels_visible_as_of()
        # answer "which chunks might be visible as of T" in O(log n) via
        # bisect instead of a linear per-label scan. A 1D threshold query
        # only needs a sorted list, not a 2D structure like an R-tree --
        # this project doesn't (yet) support transaction-time range queries,
        # which is what a 2D index would be for. Assumes homogeneous
        # timezone-awareness across every label in the store (consistent
        # with ChunkBuilder, which already raises on a per-symbol tz
        # mismatch rather than silently coping).
        self._as_of_keys: list[datetime] = []
        self._as_of_labels: list[ChunkLabel] = []
        self._correction_chunk_count = 0
        self._last_valid_timestamp: dict[str, datetime] = {}

    def _chunk_ids(self) -> Iterator[int]:
        """Yield one globally unique, monotonically increasing chunk ID."""

        while True:
            chunk_id = self._chunk_id_counter
            self._chunk_id_counter += 1
            yield chunk_id

    def ingest_row(self, timestamp: datetime, symbol: str, values: np.ndarray) -> None:
        """Route one sequential row to its symbol's active builder."""

        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")

        builder = self._builders.get(symbol)
        if builder is None:
            # A brand-new builder starts with no memory of this symbol's
            # prior data, so its own non-decreasing check (ChunkBuilder
            # only compares against its own _time_end) can't catch an
            # out-of-order row resuming after a flush_all(). Check against
            # the last timestamp this symbol ever successfully ingested.
            last_timestamp = self._last_valid_timestamp.get(symbol)
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
            self._builders[symbol] = builder

        completed = builder.add_row(timestamp, values)
        if completed is not None:
            self._add_chunk(completed)
        self._last_valid_timestamp[symbol] = timestamp

    def ingest_correction(
        self,
        timestamp: datetime,
        symbol: str,
        values: np.ndarray,
        transaction_time: datetime,
    ) -> None:
        """Append a bitemporal correction for one (symbol, timestamp) row.

        Deliberately separate from ``ingest_row``: it never touches
        ``self._builders``, so the live per-symbol builder's non-decreasing
        causality contract is completely undisturbed. A correction is built
        with a throwaway single-row ``ChunkBuilder`` sharing this store's
        chunk-id source, so it inherits the exact same validation, price-zone
        computation, and byte-packing logic finalized chunks use -- no
        parallel implementation to silently diverge from it.

        The only causality rule enforced here: a correction cannot be
        learned before the event itself happened.
        """

        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(transaction_time, datetime):
            raise TypeError("transaction_time must be a datetime instance")

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

    def compact_corrections(
        self, symbol: str | None = None, boundary: ChunkBoundary | None = None
    ) -> int:
        """Merge small correction chunks into fewer, larger ones.

        Mirrors Delta Lake's ``OPTIMIZE``, Apache Hudi's compaction table
        service, and TDSQL's late (async) data migration: ``ingest_correction``
        stays cheap and never blocks on this -- corrections accumulate as
        many small chunks, and get consolidated later via this explicit,
        idempotent maintenance operation.

        Only ever touches correction chunks (``transaction_time_start is not
        None``); base chunks are never read or rewritten. Merged rows are
        replayed through a fresh ``ChunkBuilder`` in valid-timestamp order
        (required -- corrections may target any past timestamp in any
        ingestion order, but ``ChunkBuilder`` only accepts non-decreasing
        order), so a merge naturally respects the boundary the same way
        base ingestion does: a merge spanning more time than one boundary
        window allows simply produces multiple merged chunks.

        ``boundary`` defaults to ``self.chunk_boundary`` (the base store's
        own granularity) if omitted. Correction chunks don't need the same
        fine pruning precision base data does -- corrections are relatively
        rare events, so a handful spread across many of the base store's
        boundary windows (e.g. months) may have nothing to merge with even
        when there are plenty of corrections overall. Passing a deliberately
        coarser boundary here (e.g. yearly) merges far more aggressively at
        low correction volume, mirroring Hudi's and Delta's own configurable
        compaction batch/target-file-size parameters.

        Returns the net reduction in chunk count.
        """

        merge_boundary = boundary if boundary is not None else self.chunk_boundary
        groups: dict[str, list[Chunk]] = {}
        for chunk in self._chunks:
            if chunk.label.transaction_time_start is None:
                continue
            if symbol is not None and chunk.label.symbol != symbol:
                continue
            groups.setdefault(chunk.label.symbol, []).append(chunk)

        removed = 0
        for group_symbol, chunks in groups.items():
            if len(chunks) < 2:
                continue

            rows: list[tuple[datetime, np.ndarray, datetime]] = []
            for chunk in chunks:
                assert chunk.correction_transaction_times is not None
                # column_arrays() skips decompress()'s pandas DataFrame
                # construction entirely -- for the handful of rows a
                # correction chunk holds, that construction (dtype
                # inference, index building) dominated compact_corrections'
                # own runtime; reading the raw float64 arrays directly and
                # indexing them by position is the same bytes, no DataFrame.
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
            # add_row returns the *previous* window's completed chunk (never
            # including the row just added, which starts the new window --
            # see ChunkBuilder.add_row), so pending_transaction_times always
            # holds exactly that previous window's transaction times at the
            # moment a completed chunk appears.
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

            self._remove_chunks({chunk.label.chunk_id for chunk in chunks})
            removed += len(chunks)
            for merged_chunk in merged_chunks:
                self._add_chunk(merged_chunk)
                removed -= 1

        return removed

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
        """Remove many chunks in one batched pass.

        ``compact_corrections`` used to call a single-chunk ``_remove_chunk``
        once per old chunk being merged away, and that method rebuilt
        ``self._chunks`` (and re-scanned the as-of index) from scratch on
        every call -- O(n) work repeated once per removed chunk is O(k*n)
        for k removals, not O(n). Filtering every structure exactly once
        here, regardless of how many chunk_ids are being removed, is what
        makes removal O(n) overall.
        """

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

        # Filtering the two parallel index lists together in one pass keeps
        # them sorted (removing elements from an already-sorted sequence
        # can't unsort it) without needing a bisect lookup per removed key.
        kept_keys: list[datetime] = []
        kept_labels: list[ChunkLabel] = []
        for key, label in zip(self._as_of_keys, self._as_of_labels, strict=True):
            if label.chunk_id not in chunk_ids:
                kept_keys.append(key)
                kept_labels.append(label)
        self._as_of_keys = kept_keys
        self._as_of_labels = kept_labels
        self._correction_chunk_count -= removed_correction_count

    def flush_all(self) -> None:
        """Finalize every active symbol and end the current ingestion session.

        Removes each symbol's builder from ``self._builders`` only once it
        has been successfully flushed and added, not all at once at the end
        -- if ``_add_chunk`` raises partway through, ``self._builders``
        accurately reflects which symbols were and weren't processed,
        instead of a stale ``.clear()`` erasing that information regardless
        of where the failure happened.
        """

        for symbol in list(self._builders.keys()):
            builder = self._builders[symbol]
            completed = builder.flush()
            if completed is not None:
                self._add_chunk(completed)
            del self._builders[symbol]

    def _add_chunk(self, chunk: Chunk) -> None:
        # Compute the as-of-index insertion position -- which can raise on a
        # tz-awareness mismatch -- before mutating any structure. Mutating
        # _chunks/_chunk_by_id first and only then discovering the bisect
        # call fails would leave the chunk permanently registered but absent
        # from the as-of index: a silent, permanent pushdown false negative.
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
        """Raise a clear error on a store-wide tz-awareness mismatch.

        Nothing enforces homogeneous tz-awareness across symbols --
        ``ChunkBuilder`` only compares a row against its own builder's
        ``_time_end``, never store-wide, and ``ingest_correction`` uses a
        disposable one-off builder that has no memory of any other symbol
        either. Left unchecked, ``bisect.bisect_right`` below would raise a
        bare ``TypeError`` from deep inside ``_add_chunk``, after this
        method had already mutated other store state in the old ordering.
        """

        if not self._as_of_keys:
            return
        existing_is_aware = self._as_of_keys[0].utcoffset() is not None
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

    def labels_visible_as_of(self, as_of: datetime) -> list[ChunkLabel]:
        """Return every label that might hold a row visible as of a time.

        O(log n) bisect plus a cheap slice (copying already-known
        references, not recomputing anything per label) instead of a linear
        scan calling ``label_visible_as_of`` on every label -- see
        ``src/query/pushdown.py``'s ``as_of_index`` fast path, which uses
        this to avoid running predicate evaluation on chunks already known
        to be AS-OF-irrelevant.
        """

        cut = bisect.bisect_right(self._as_of_keys, as_of)
        return self._as_of_labels[:cut]

    def labels_not_visible_as_of(self, as_of: datetime) -> list[ChunkLabel]:
        """The complement of ``labels_visible_as_of`` -- also O(log n) + slice."""

        cut = bisect.bisect_right(self._as_of_keys, as_of)
        return self._as_of_labels[cut:]

    def ingest_dataframe(self, df: pd.DataFrame) -> None:
        """Ingest a DataFrame through the same sequential replay contract."""

        harness = ReplayHarness(df, value_columns=self.value_columns)
        self.ingest_from_replay(harness)

    def ingest_from_replay(self, harness: ReplayHarness) -> None:
        """Canonical no-lookahead ingestion path."""

        for row in harness:
            self.ingest_row(
                row["timestamp"],
                row["symbol"],
                row["values"],
            )
        self.flush_all()

    @property
    def chunks(self) -> list[Chunk]:
        """Return a shallow copy so callers cannot mutate store membership."""

        return list(self._chunks)

    @property
    def labels(self) -> list[ChunkLabel]:
        return [chunk.label for chunk in self._chunks]

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def has_corrections(self) -> bool:
        """O(1) check maintained incrementally by _add_chunk/_remove_chunks.

        Used to disable price-based pushdown pruning (see
        ``evaluate_pushdown``'s ``price_pruning_enabled``): once any
        correction chunk exists anywhere in the store, a correction's own
        price zone map failing a value predicate must never hide it from
        candidacy, or a stale base row could silently win over it.
        """

        return self._correction_chunk_count > 0

    @property
    def total_rows(self) -> int:
        return sum(chunk.label.row_count for chunk in self._chunks)

    @property
    def total_column_bytes(self) -> int:
        return sum(len(chunk.column_data) for chunk in self._chunks)

    def get_chunk(self, chunk_id: int) -> Chunk:
        # This used to scan self._chunks linearly. QueryExecutor calls
        # get_chunk once per candidate chunk, so a broad query (most/all
        # chunks are candidates) turned that into an O(total_chunks^2)
        # pattern -- measured at 24,000 chunks, a 0%-skip query took 22.5s
        # through the pushdown-query path against 6.8s for the full-scan
        # baseline that never calls get_chunk. Chunk IDs are assigned once,
        # at creation, and never reused, so a plain dict gives O(1) lookup
        # with no invalidation concerns.
        try:
            return self._chunk_by_id[chunk_id]
        except KeyError:
            raise KeyError(f"chunk ID not found: {chunk_id}") from None

    def full_scan(
        self,
        predicate_fn: Callable[[pd.Series], bool] | None = None,
    ) -> pd.DataFrame:
        """Decompress every chunk and optionally apply an exact row predicate."""

        if predicate_fn is not None and not callable(predicate_fn):
            raise TypeError("predicate_fn must be callable or None")

        if not self._chunks:
            return pd.DataFrame(columns=["timestamp", "symbol", *self.value_columns])

        # Decompress before filtering: this is intentionally the naive
        # full-scan baseline, with no label pruning or skipped chunks.
        frame = pd.concat(
            [chunk.decompress() for chunk in self._chunks],
            ignore_index=True,
        )
        if predicate_fn is None:
            return frame

        mask = frame.apply(lambda row: bool(predicate_fn(row)), axis=1)
        return frame.loc[mask].reset_index(drop=True)


__all__ = ["ChunkStore"]
