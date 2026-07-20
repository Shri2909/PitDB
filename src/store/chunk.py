"""Raw column streams and queryable metadata for one time chunk."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.store.label import ChunkLabel


@dataclass
class Chunk:
    """One independently queryable group of raw, column-oriented rows.

    ``column_data`` contains one column's raw float64 bytes after another;
    ``column_byte_lengths`` is the framing needed to split those streams
    again.
    """

    label: ChunkLabel
    column_data: bytes
    column_byte_lengths: list[int]
    num_columns: int
    column_names: list[str]
    timestamps: list[datetime]
    # None for a base chunk (transaction time is derived from each row's own
    # timestamp -- see src/query/versioning.py). A list, positionally aligned
    # with `timestamps` and the same length, for a correction chunk (single
    # or merged by ChunkStore.compact_corrections): a chunk merging several
    # corrections holds rows learned at different transaction times, so a
    # single per-chunk scalar can no longer answer "when was this row
    # learned" -- only a per-row list can.
    correction_transaction_times: list[datetime] | None = None

    def __post_init__(self) -> None:
        # Detach mutable framing lists from their caller. Chunk itself remains a
        # mutable dataclass, so decompress() validates again before trusting it.
        self.column_byte_lengths = list(self.column_byte_lengths)
        self.column_names = list(self.column_names)
        self.timestamps = list(self.timestamps)
        if self.correction_transaction_times is not None:
            self.correction_transaction_times = list(self.correction_transaction_times)
        self._validate_structure()

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

        for byte_length in self.column_byte_lengths:
            if not isinstance(byte_length, int) or isinstance(byte_length, bool):
                raise TypeError("column byte lengths must be integers")
            if byte_length < 0:
                raise ValueError("column byte lengths must be non-negative")

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
        """Reassemble each column's raw bytes into plain float64 arrays.

        This is the byte-unpacking core both ``decompress()`` and callers
        that only need the underlying arrays (not a full DataFrame, e.g.
        ``ChunkStore.compact_corrections``) rely on. Skipping straight to
        this instead of ``decompress()`` avoids pandas' DataFrame
        construction machinery (dtype inference, index building, column
        management) entirely -- real, measured overhead when all a caller
        wants is the raw values for a chunk with few rows.
        """

        self._validate_structure()
        columns: dict[str, np.ndarray] = {}
        offset = 0

        for column_name, byte_length in zip(
            self.column_names, self.column_byte_lengths, strict=True
        ):
            end = offset + byte_length
            # np.frombuffer returns a read-only view into the immutable bytes
            # object; .copy() gives callers an owned, writable array matching
            # the previous decode path's behavior.
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

        # Dict insertion order makes the reconstructed schema deterministic:
        # timestamp, symbol, then the original ordered value columns.
        frame_data: dict[str, object] = {
            "timestamp": list(self.timestamps),
            "symbol": [self.label.symbol] * self.label.row_count,
        }
        frame_data.update(columns)
        return pd.DataFrame(frame_data)


__all__ = ["Chunk"]
