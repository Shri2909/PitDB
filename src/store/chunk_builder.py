"""Streaming construction of independently framed chunks."""

from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import datetime
from itertools import count

import numpy as np

from src.store.chunk import Chunk
from src.store.chunk_boundary import ChunkBoundary, MonthlyBoundary
from src.store.label import ChunkLabel

DEFAULT_VALUE_COLUMNS = ("open", "high", "low", "close", "volume")


class ChunkBuilder:
    """Accumulate one symbol's rows into a chunk of raw column data."""

    def __init__(
        self,
        symbol: str,
        chunk_boundary: ChunkBoundary = MonthlyBoundary(),
        value_columns: list[str] | None = None,
        chunk_id_counter: Iterator[int] | None = None,
    ) -> None:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("symbol must be a non-empty string")
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

        self.symbol = symbol
        self.chunk_boundary = chunk_boundary
        self.value_columns = selected
        self._chunk_id_counter = (
            iter(chunk_id_counter) if chunk_id_counter is not None else count()
        )
        self._price_column_index = self.value_columns.index("close")
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._raw_values: dict[str, list[float]] = {}
        self._timestamps: list[datetime] = []
        self._min_price = float("inf")
        self._max_price = float("-inf")
        self._time_start: datetime | None = None
        self._time_end: datetime | None = None
        self._row_count = 0

    def _start_window(self, timestamp: datetime) -> None:
        self._time_start = timestamp
        self._raw_values = {column: [] for column in self.value_columns}

    def _validate_row(self, timestamp: datetime, values: np.ndarray) -> None:
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance")
        if not isinstance(values, np.ndarray):
            raise TypeError("values must be a NumPy array")
        if values.ndim != 1:
            raise ValueError("values must be a one-dimensional array")
        if values.dtype != np.float64:
            raise ValueError(f"values must have float64 dtype; found {values.dtype}")
        if len(values) != len(self.value_columns):
            raise ValueError(
                f"expected {len(self.value_columns)} values, found {len(values)}"
            )
        if self._time_end is not None:
            try:
                out_of_order = timestamp < self._time_end
            except TypeError as exc:
                raise ValueError(
                    "timestamp timezone awareness must match the current chunk"
                ) from exc
            if out_of_order:
                raise ValueError(
                    "timestamps must be added in non-decreasing order; "
                    f"received {timestamp!r} after {self._time_end!r}"
                )

    def _append_row(self, timestamp: datetime, values: np.ndarray) -> None:
        for index, column in enumerate(self.value_columns):
            self._raw_values[column].append(float(values[index]))

        # Bare min/max with NaN is argument-order dependent because every NaN
        # comparison is false. Skipping NaN preserves valid zone-map bounds; an
        # all-NaN chunk intentionally remains (+inf, -inf).
        price = float(values[self._price_column_index])
        if not math.isnan(price):
            self._min_price = min(self._min_price, price)
            self._max_price = max(self._max_price, price)

        self._time_end = timestamp
        self._timestamps.append(timestamp)
        self._row_count += 1

    def add_row(self, timestamp: datetime, values: np.ndarray) -> Chunk | None:
        """Consume one row and return the previous chunk on a boundary crossing."""

        self._validate_row(timestamp, values)
        if self._time_start is None:
            self._start_window(timestamp)

        assert self._time_start is not None
        if self.chunk_boundary.crosses(self._time_start, timestamp):
            completed = self.finalize()
            self._reset_accumulators()
            self._start_window(timestamp)
            self._append_row(timestamp, values)
            return completed

        self._append_row(timestamp, values)
        return None

    def finalize(self) -> Chunk:
        """Finalize the active window without resetting builder state."""

        if (
            self._row_count == 0
            or self._time_start is None
            or self._time_end is None
            or not self._raw_values
        ):
            raise ValueError("cannot finalize an empty chunk")

        try:
            chunk_id = next(self._chunk_id_counter)
        except StopIteration as exc:
            raise RuntimeError("chunk_id_counter was exhausted") from exc
        if not isinstance(chunk_id, int) or isinstance(chunk_id, bool):
            raise TypeError("chunk_id_counter must yield integers")

        streams: list[bytes] = []
        column_byte_lengths: list[int] = []
        for column in self.value_columns:
            raw_array = np.asarray(self._raw_values[column], dtype=np.float64)
            if len(raw_array) != self._row_count:
                raise RuntimeError(
                    f"buffered {len(raw_array)} values for {column!r}; "
                    f"expected {self._row_count}"
                )
            column_bytes = raw_array.tobytes()
            streams.append(column_bytes)
            column_byte_lengths.append(len(column_bytes))

        label = ChunkLabel(
            symbol=self.symbol,
            time_start=self._time_start,
            time_end=self._time_end,
            min_price=self._min_price,
            max_price=self._max_price,
            row_count=self._row_count,
            chunk_id=chunk_id,
        )
        return Chunk(
            label=label,
            column_data=b"".join(streams),
            column_byte_lengths=column_byte_lengths,
            num_columns=len(self.value_columns),
            column_names=list(self.value_columns),
            timestamps=list(self._timestamps),
        )

    def flush(self) -> Chunk | None:
        """Finalize the in-progress chunk at end-of-stream, if one exists."""

        if self._row_count == 0:
            return None
        completed = self.finalize()
        self._reset_accumulators()
        return completed


__all__ = ["ChunkBuilder"]
