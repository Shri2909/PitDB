"""Raw column streams and queryable metadata for one time chunk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.store.label import ChunkLabel
from src.store.validation import RESERVED_VALUE_COLUMNS, timestamp_value


@dataclass(frozen=True)
class Chunk:
    """One independently queryable group of raw, column-oriented rows."""

    label: ChunkLabel
    column_data: bytes
    column_byte_lengths: tuple[int, ...] | list[int]
    num_columns: int
    column_names: tuple[str, ...] | list[str]
    timestamps: tuple[datetime, ...] | list[datetime]
    # None for a base chunk; for a correction chunk, one entry per row.
    correction_transaction_times: tuple[datetime, ...] | list[datetime] | None = None

    def __post_init__(self) -> None:
        # Detach sequence inputs into immutable tuples.
        object.__setattr__(self, "column_byte_lengths", tuple(self.column_byte_lengths))
        object.__setattr__(self, "column_names", tuple(self.column_names))
        object.__setattr__(
            self, "timestamps", tuple(timestamp_value(t) for t in self.timestamps)
        )
        if self.correction_transaction_times is not None:
            object.__setattr__(
                self,
                "correction_transaction_times",
                tuple(
                    timestamp_value(t, name="transaction_time")
                    for t in self.correction_transaction_times
                ),
            )
        self._validate_structure()
        self._validate_invariants()

    def _validate_invariants(self) -> None:
        """Check cross-field invariants between rows, label and transaction times."""
        timestamps = self.timestamps
        if timestamps:
            try:
                for earlier, later in zip(timestamps, timestamps[1:]):
                    if later < earlier:
                        raise ValueError(
                            "chunk timestamps must be in non-decreasing order"
                        )
                if self.label.time_start != timestamps[0]:
                    raise ValueError(
                        "label.time_start must equal the first row timestamp"
                    )
                if timestamps[-1] > self.label.time_end:
                    raise ValueError("row timestamps must not exceed label.time_end")
            except TypeError as exc:
                raise ValueError(
                    "chunk timestamps and label time range must share "
                    "timezone awareness"
                ) from exc

        if self.correction_transaction_times is None:
            return
        start = self.label.transaction_time_start
        end = self.label.transaction_time_end
        assert start is not None and end is not None
        try:
            for timestamp, transaction_time in zip(
                timestamps, self.correction_transaction_times
            ):
                if transaction_time < start or transaction_time > end:
                    raise ValueError(
                        "each correction transaction time must lie within "
                        "the label's transaction-time range"
                    )
                if transaction_time < timestamp:
                    raise ValueError(
                        "a correction's transaction time cannot precede its "
                        "row timestamp"
                    )
        except TypeError as exc:
            raise ValueError(
                "transaction times and row timestamps must share timezone awareness"
            ) from exc

    def _validate_structure(self) -> None:
        if not isinstance(self.label, ChunkLabel):
            raise TypeError("label must be a ChunkLabel")
        if not isinstance(self.column_data, bytes):
            raise TypeError("column_data must be bytes")
        if not isinstance(self.num_columns, int) or isinstance(self.num_columns, bool):
            raise TypeError("num_columns must be an integer")
        if self.num_columns < 0:
            raise ValueError("num_columns must be non-negative")
        if len(self.column_names) != self.num_columns:
            raise ValueError("num_columns does not match the number of column names")
        if len(self.column_byte_lengths) != self.num_columns:
            raise ValueError(
                "num_columns does not match the number of column byte lengths"
            )
        if len(set(self.column_names)) != len(self.column_names):
            raise ValueError("column_names must be unique")
        if any(not isinstance(name, str) or not name for name in self.column_names):
            raise ValueError("every column name must be a non-empty string")
        if RESERVED_VALUE_COLUMNS.intersection(self.column_names):
            raise ValueError("numeric column names must not use reserved engine names")

        for byte_length in self.column_byte_lengths:
            if not isinstance(byte_length, int) or isinstance(byte_length, bool):
                raise TypeError("column byte lengths must be integers")
            if byte_length < 0:
                raise ValueError("column byte lengths must be non-negative")
            if byte_length % 8:
                raise ValueError(
                    "column byte lengths must contain complete float64 values"
                )

        framed_size = sum(self.column_byte_lengths)
        if framed_size != len(self.column_data):
            raise ValueError(
                "column byte lengths must consume column_data exactly; "
                f"declared {framed_size} bytes, found "
                f"{len(self.column_data)}"
            )

        if self.label.row_count < 0:
            raise ValueError("label.row_count must be non-negative")
        if len(self.timestamps) != self.label.row_count:
            raise ValueError(
                "timestamp count must equal label.row_count; "
                f"found {len(self.timestamps)} timestamps for "
                f"{self.label.row_count} rows"
            )
        if any(not isinstance(value, datetime) for value in self.timestamps):
            raise TypeError("every timestamp must be a datetime instance")

        has_transaction_range = self.label.transaction_time_start is not None
        has_correction_times = self.correction_transaction_times is not None
        if has_transaction_range != has_correction_times:
            raise ValueError(
                "label.transaction_time_start and correction_transaction_times "
                "must be set together (correction chunk) or both left unset "
                "(base chunk)"
            )
        if has_correction_times:
            assert self.correction_transaction_times is not None
            if len(self.correction_transaction_times) != self.label.row_count:
                raise ValueError(
                    "correction_transaction_times count must equal "
                    f"label.row_count; found "
                    f"{len(self.correction_transaction_times)} for "
                    f"{self.label.row_count} rows"
                )
            if any(
                not isinstance(value, datetime)
                for value in self.correction_transaction_times
            ):
                raise TypeError(
                    "every correction transaction time must be a datetime instance"
                )

    def column_arrays(self) -> dict[str, np.ndarray]:
        """Reassemble each column's raw bytes into plain float64 arrays."""
        self._validate_structure()
        columns: dict[str, np.ndarray] = {}
        offset = 0

        for column_name, byte_length in zip(
            self.column_names, self.column_byte_lengths
        ):
            end = offset + byte_length
            # frombuffer is read-only; copy to give callers a writable array.
            column_values = np.frombuffer(
                self.column_data[offset:end], dtype=np.float64
            ).copy()
            if len(column_values) != self.label.row_count:
                raise ValueError(
                    f"stored column {column_name!r} has "
                    f"{len(column_values)} rows; expected {self.label.row_count}"
                )
            columns[column_name] = column_values
            offset = end

        return columns

    def decompress(self) -> pd.DataFrame:
        """Reassemble each column's raw bytes and reconstruct row order."""
        columns = self.column_arrays()

        # Insertion order fixes the schema: timestamp, symbol, then value columns.
        frame_data: dict[str, object] = {
            "timestamp": list(self.timestamps),
            "symbol": [self.label.symbol] * self.label.row_count,
        }
        frame_data.update(columns)
        return pd.DataFrame(frame_data)


__all__ = ["Chunk"]
