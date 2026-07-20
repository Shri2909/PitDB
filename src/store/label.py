"""Immutable chunk metadata for conservative predicate pushdown.

The min/max fields form a compact zone map: returning ``False`` means a chunk
is provably unable to match and may be skipped, while ``True`` means it remains
a candidate for exact row-level evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

_SUPPORTED_OPERATORS = frozenset({"=", ">", ">=", "<", "<="})


def _require_operator(op: str) -> None:
    if op not in _SUPPORTED_OPERATORS:
        supported = ", ".join(sorted(_SUPPORTED_OPERATORS))
        raise ValueError(
            f"unsupported predicate operator {op!r}; expected one of: {supported}"
        )


@dataclass(frozen=True)
class ChunkLabel:
    """Small, immutable summary attached to one compressed chunk."""

    symbol: str
    time_start: datetime
    time_end: datetime
    min_price: float
    max_price: float
    row_count: int
    chunk_id: int
    # None means "no revision": this chunk's rows were recorded at their own
    # valid timestamp (a live feed's ordinary print), so transaction-time
    # visibility can be derived from time_start/time_end directly. A
    # concrete (start, end) pair marks a correction chunk -- a zone map over
    # each row's own transaction time, exactly like time_start/time_end is a
    # zone map over valid time. A single, unmerged correction has
    # start == end (a degenerate point range); a chunk merging several
    # corrections (see ChunkStore.compact_corrections) gets the true
    # min/max across the rows it actually contains.
    transaction_time_start: datetime | None = None
    transaction_time_end: datetime | None = None

    def can_contain_symbol(self, symbol: str) -> bool:
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
        """Return whether this min/max price range might satisfy a predicate.

        An all-NaN chunk is represented naturally as ``(+inf, -inf)``. These
        formulas then return False for every finite or NaN query value without
        requiring a special branch.
        """

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
