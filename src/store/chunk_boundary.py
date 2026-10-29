"""Pluggable strategies for deciding when a time-series chunk ends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.store.validation import timestamp_value


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
    """Use a fixed elapsed duration per symbol, including across DST changes.

    Appropriate for dense intraday inputs, such as one-minute data grouped into
    hourly chunks, where each window contains adjacent values.
    """

    duration: timedelta

    def __post_init__(self) -> None:
        if not isinstance(self.duration, timedelta):
            raise TypeError("duration must be a datetime.timedelta")
        if self.duration <= timedelta(0):
            raise ValueError("duration must be greater than zero")

    def crosses(self, time_start: datetime, current_timestamp: datetime) -> bool:
        start = timestamp_value(time_start)
        current = timestamp_value(current_timestamp)
        if (start.tzinfo is None) != (current.tzinfo is None):
            raise TypeError("boundary timestamps must have matching timezone awareness")
        # Each timestamp fits int64 nanoseconds, but their difference may not.
        # Python integer arithmetic also supports timedelta.max boundaries.
        duration_ns = (
            (self.duration.days * 86400 + self.duration.seconds) * 1_000_000_000
            + self.duration.microseconds * 1000
            + getattr(self.duration, "nanoseconds", 0)
        )
        return current.value - start.value >= duration_ns


__all__ = ["ChunkBoundary", "MonthlyBoundary", "DurationBoundary"]
