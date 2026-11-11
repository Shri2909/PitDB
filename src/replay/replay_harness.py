"""Strictly sequential, no-lookahead replay of a DataFrame."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from src.replay.data_loader import _validate_symbol_column
from src.store.chunk_builder import DEFAULT_VALUE_COLUMNS
from src.store.validation import RESERVED_VALUE_COLUMNS


class ReplayHarness(Iterator[dict[str, Any]]):
    """Yield DataFrame rows in order without lookahead.

    The DataFrame is held by reference, so callers must not mutate it.
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

        if isinstance(value_columns, (str, bytes)):
            raise TypeError(
                "value_columns must be a list of column names, not a bare string; "
                f"use [{value_columns!r}]"
            )
        selected = (
            list(DEFAULT_VALUE_COLUMNS)
            if value_columns is None
            else list(value_columns)
        )
        if not selected:
            raise ValueError("value_columns must not be empty")
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

        # Fail early: the store derives price labels from `close`.
        if "close" not in selected:
            raise ValueError("value_columns must include 'close' for price labels")

        required = {"timestamp", "symbol", *selected}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError("missing replay columns: " + ", ".join(missing))

        # Validate every key up front to avoid a partially ingested replay.
        _validate_symbol_column(df)

        self._df = df
        self._value_columns = selected
        self._index = 0

    def has_next(self) -> bool:
        """Return whether one more sequential row can be consumed."""
        return self._index < len(self._df)

    def next_row(self) -> dict[str, Any]:
        """Return and consume the current row (values are a private float64 copy)."""
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
