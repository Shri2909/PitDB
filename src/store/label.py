"""Immutable chunk metadata (zone maps) for conservative predicate pushdown."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from datetime import datetime

from src.store.validation import timestamp_value

_SUPPORTED_OPERATORS = frozenset({"=", ">", ">=", "<", "<="})


def _require_operator(op: str) -> None:
    if op not in _SUPPORTED_OPERATORS:
        supported = ", ".join(sorted(_SUPPORTED_OPERATORS))
        raise ValueError(
            f"unsupported predicate operator {op!r}; expected one of: {supported}"
        )


@dataclass(frozen=True)
class ChunkLabel:
    """Small, immutable summary attached to one raw column chunk."""

    symbol: str
    time_start: datetime
    time_end: datetime
    min_price: float
    max_price: float
    row_count: int
    chunk_id: int
    # Both None for a base chunk; for a correction chunk, the min/max transaction time.
    transaction_time_start: datetime | None = None
    transaction_time_end: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        for field in ("row_count", "chunk_id"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        for field in ("time_start", "time_end"):
            object.__setattr__(
                self, field, timestamp_value(getattr(self, field), name=field)
            )
        try:
            if self.time_end < self.time_start:
                raise ValueError("time_end must not precede time_start")
        except TypeError as exc:
            raise ValueError("time range timezone awareness must match") from exc
        self._validate_price_range()
        if (self.transaction_time_start is None) != (self.transaction_time_end is None):
            raise ValueError("both transaction-time bounds must be set together")
        if self.transaction_time_start is not None:
            for field in ("transaction_time_start", "transaction_time_end"):
                object.__setattr__(
                    self, field, timestamp_value(getattr(self, field), name=field)
                )
            try:
                if self.transaction_time_end < self.transaction_time_start:
                    raise ValueError("transaction-time range must be ordered")
            except TypeError as exc:
                raise ValueError(
                    "transaction-time timezone awareness must match"
                ) from exc

    def _validate_price_range(self) -> None:
        """Require ordered price bounds, except the all-NaN marker (+inf, -inf)."""
        for field in ("min_price", "max_price"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise TypeError(f"{field} must be a real number")
            if math.isnan(value):
                raise ValueError(f"{field} must not be NaN")
        if self.min_price > self.max_price and not (
            self.min_price == math.inf and self.max_price == -math.inf
        ):
            raise ValueError(
                "min_price must not exceed max_price "
                "(only the all-NaN marker (+inf, -inf) may be inverted)"
            )

    def can_contain_symbol(self, symbol: str) -> bool:
        """Return whether this chunk holds the given symbol."""
        return self.symbol == symbol

    def can_contain_time(self, op: str, value: datetime) -> bool:
        """Return whether this inclusive time range might satisfy a predicate."""
        _require_operator(op)
        if op == "=":
            return self.time_start <= value <= self.time_end
        if op == ">=":
            return self.time_end >= value
        if op == "<=":
            return self.time_start <= value
        if op == ">":
            return self.time_end > value
        return self.time_start < value  # op == "<"

    def can_contain_price(self, op: str, value: float) -> bool:
        """Return whether this min/max price range might satisfy a predicate."""
        _require_operator(op)
        if op == "=":
            return self.min_price <= value <= self.max_price
        if op == ">=":
            return self.max_price >= value
        if op == "<=":
            return self.min_price <= value
        if op == ">":
            return self.max_price > value
        return self.min_price < value  # op == "<"


__all__ = ["ChunkLabel"]
