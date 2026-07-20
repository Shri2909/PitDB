"""Pluggable strategies for deciding when a time-series chunk ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta


class ChunkBoundary(ABC):
    """Strategy for deciding when the current chunk should be finalized."""

    @abstractmethod
    def crosses(self, time_start: datetime, current_timestamp: datetime) -> bool:
        """Return whether ``current_timestamp`` belongs in a new chunk."""


@dataclass
class MonthlyBoundary(ChunkBoundary):
    """Use one calendar-month chunk per symbol.

    This is the default strategy for daily OHLCV data. Daily rows are roughly
    24 hours apart, so an hourly boundary would create one-row chunks --
    multiplying per-chunk label/bookkeeping overhead across the store without
    any pruning-precision benefit, since a one-row chunk can't be split any
    finer than its own single row already is.
    """

    def crosses(self, time_start: datetime, current_timestamp: datetime) -> bool:
        # Compare the pair, not only the month number. The year check handles
        # same-month values from different years and December-to-January rollover.
        return (current_timestamp.year, current_timestamp.month) != (
            time_start.year,
            time_start.month,
        )


@dataclass
class DurationBoundary(ChunkBoundary):
    """Use a fixed wall-clock duration per symbol.

    Appropriate for dense intraday inputs, such as one-minute data grouped into
    hourly chunks, where each window contains enough adjacent values to compress.
    """

    duration: timedelta

    def __post_init__(self) -> None:
        if not isinstance(self.duration, timedelta):
            raise TypeError("duration must be a datetime.timedelta")
        if self.duration <= timedelta(0):
            raise ValueError("duration must be greater than zero")

    def crosses(self, time_start: datetime, current_timestamp: datetime) -> bool:
        return (current_timestamp - time_start) >= self.duration


__all__ = ["ChunkBoundary", "MonthlyBoundary", "DurationBoundary"]
