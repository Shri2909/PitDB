"""Shared validation at the boundary of mutable ingestion state."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

RESERVED_VALUE_COLUMNS = frozenset(
    {"timestamp", "symbol", "_txn_time", "_is_correction"}
)


def validate_value_columns(
    value_columns: list[str] | None, *, default: tuple[str, ...]
) -> list[str]:
    selected = list(default) if value_columns is None else list(value_columns)
    if not selected:
        raise ValueError("value_columns must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in selected):
        raise ValueError("every value column must be a non-empty string")
    if len(selected) != len(set(selected)):
        raise ValueError("value_columns must not contain duplicates")
    reserved = RESERVED_VALUE_COLUMNS.intersection(selected)
    if reserved:
        raise ValueError(f"reserved value column name(s): {sorted(reserved)}")
    if "close" not in selected:
        raise ValueError("value_columns must include 'close' for price labels")
    return selected


def timestamp_value(value: datetime, *, name: str = "timestamp") -> pd.Timestamp:
    """Validate a real, nanosecond-representable datetime without losing fold."""
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime instance")
    if pd.isna(value):
        raise ValueError(f"{name} must not be a missing datetime")
    try:
        return pd.Timestamp(value).as_unit("ns")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be representable as datetime64[ns]") from exc


def align_timestamp(reference: datetime, value: datetime) -> pd.Timestamp:
    """Align a query literal to the store's timezone semantics.

    Naive literals are localized in an aware store's zone (ambiguous or
    nonexistent times are rejected); offset literals on naive data compare
    wall time; aware/aware comparisons keep absolute instants.
    """
    ref = timestamp_value(reference, name="reference timestamp")
    literal = timestamp_value(value, name="datetime literal")
    try:
        if ref.tz is not None and literal.tz is None:
            localized = literal.tz_localize(ref.tz, ambiguous="NaT", nonexistent="NaT")
            if pd.isna(localized):
                raise ValueError("datetime literal is ambiguous or nonexistent")
            return localized
        if ref.tz is None and literal.tz is not None:
            return literal.tz_localize(None)
        return literal
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "datetime literal is ambiguous or invalid in the stored timezone"
        ) from exc


def canonical_timestamp(
    value: datetime, reference: datetime | None = None, *, name: str = "timestamp"
) -> pd.Timestamp:
    """Convert aware timestamps to the reference zone (mismatches are left as is)."""
    timestamp = timestamp_value(value, name=name)
    if reference is not None:
        ref = timestamp_value(reference, name="reference timestamp")
        if timestamp.tz is not None and ref.tz is not None:
            timestamp = timestamp.tz_convert(ref.tz)
    return timestamp
