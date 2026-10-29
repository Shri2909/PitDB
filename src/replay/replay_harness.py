"""Strictly sequential replay adapters for streaming compression.

The public surface deliberately omits length, indexing, peeking, and access to
the source DataFrame. Consumers can observe only the current row when it is
yielded, matching a live feed where future values do not yet exist.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from src.replay.data_loader import _validate_symbol_column
from src.store.validation import RESERVED_VALUE_COLUMNS

DEFAULT_VALUE_COLUMNS = ("open", "high", "low", "close", "volume")


class ReplayHarness(Iterator[dict[str, Any]]):
    """Yield DataFrame rows once, in their existing order, without lookahead.

    The DataFrame is retained by reference rather than copied, keeping the
    loader/replay boundary memory-efficient. Callers must therefore treat their
    source DataFrame as immutable for the lifetime of the harness.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        value_columns: list[str] | None = None,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not df.columns.is_unique:
            duplicates = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
            raise ValueError(
                "DataFrame contains duplicate column names: " + ", ".join(duplicates)
            )

        selected = (
            list(DEFAULT_VALUE_COLUMNS)
            if value_columns is None
            else list(value_columns)
        )
        if any(
            not isinstance(column, str) or not column.strip() for column in selected
        ):
            raise ValueError("value_columns must contain non-empty string column names")
        if len(selected) != len(set(selected)):
            raise ValueError("value_columns must not contain duplicates")
        conflicts = sorted(RESERVED_VALUE_COLUMNS.intersection(selected))
        if conflicts:
            raise ValueError(
                "value_columns contain reserved columns: " + ", ".join(conflicts)
            )

        required = {"timestamp", "symbol", *selected}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError("missing replay columns: " + ", ".join(missing))

        # Validate all keys before exposing the first row so a later malformed
        # symbol cannot cause a partially ingested replay.
        _validate_symbol_column(df)

        self._df = df
        self._value_columns = selected
        self._index = 0

    def has_next(self) -> bool:
        """Return whether one more sequential row can be consumed."""

        return self._index < len(self._df)

    def next_row(self) -> dict[str, Any]:
        """Return and consume the current row.

        The numeric array is a private float64 copy, so downstream compression
        cannot mutate the source DataFrame accidentally.
        """

        if not self.has_next():
            raise StopIteration

        row = self._df.iloc[self._index]
        symbol = row["symbol"]
        if not isinstance(symbol, str):
            raise ValueError(
                "symbol must be a string at replay index "
                f"{self._index}; found {type(symbol).__name__}"
            )
        if not symbol.strip():
            raise ValueError(
                "symbol must not be empty or whitespace-only at replay index "
                f"{self._index}"
            )

        try:
            values = row.loc[self._value_columns].to_numpy(dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"value columns cannot be converted to float64 at replay "
                f"index {self._index}"
            ) from exc

        result = {
            "timestamp": row["timestamp"],
            "symbol": symbol,
            "values": values,
        }
        self._index += 1
        return result

    def __iter__(self) -> ReplayHarness:
        return self

    def __next__(self) -> dict[str, Any]:
        return self.next_row()

    def reset(self) -> None:
        """Replay the same immutable source again from its first row."""

        self._index = 0


__all__ = ["ReplayHarness"]
