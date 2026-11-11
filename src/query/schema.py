"""Canonical query schema, SQL aliases, and typed literal conversion."""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import Enum

import pandas as pd


class ColumnType(Enum):
    STRING = "string"
    DATETIME = "datetime"
    FLOAT = "float"
    INTEGER = "integer"


SCHEMA: dict[str, ColumnType] = {
    "symbol": ColumnType.STRING,
    "timestamp": ColumnType.DATETIME,
    "open": ColumnType.FLOAT,
    "high": ColumnType.FLOAT,
    "low": ColumnType.FLOAT,
    "close": ColumnType.FLOAT,
    "volume": ColumnType.FLOAT,
}


ALIAS_MAP: dict[str, str] = {
    "date": "timestamp",
    "price": "close",
}

VALID_COLUMNS = set(SCHEMA) | set(ALIAS_MAP)


# "price" is the label category backed by min_price/max_price, not a SQL alias.
COLUMN_LABEL_MAP: dict[str, str | None] = {
    "symbol": "symbol",
    "timestamp": "time",
    "open": None,
    "high": None,
    "low": None,
    "close": "price",
    "volume": None,
}


# Extended (2024-01-01) or compact (20240101) date; ASCII digits only.
_DATE_ONLY_PATTERN = re.compile(r"(?:[0-9]{4}-[0-9]{2}-[0-9]{2}|[0-9]{8})\Z")
_FLOAT_PATTERN = re.compile(
    r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z"
)
_INTEGER_PATTERN = re.compile(r"[+-]?[0-9]+\Z")

# Out-of-range literals are rejected, not clamped (unsound for <, >=, =).
MIN_TIMESTAMP = pd.Timestamp.min
MAX_TIMESTAMP = pd.Timestamp.max
SUPPORTED_TIMESTAMP_RANGE = (
    f"{MIN_TIMESTAMP.isoformat()} to {MAX_TIMESTAMP.isoformat()}"
)


def _unquote_sql_literal(value: str) -> str:
    """Remove one matching SQL quote pair and unescape doubled quotes."""

    literal = value.strip()
    if len(literal) < 2 or literal[0] != literal[-1]:
        return literal

    quote = literal[0]
    if quote not in {"'", '"'}:
        return literal

    inner = literal[1:-1]
    return inner.replace(quote * 2, quote)


def validate_column(name: str) -> str:
    """Validate a SQL column reference and return its canonical DataFrame name."""

    if not isinstance(name, str):
        raise ValueError("column name must be a string")

    normalized = name.strip().lower()
    canonical = ALIAS_MAP.get(normalized, normalized)
    if canonical not in SCHEMA:
        valid = ", ".join(sorted(VALID_COLUMNS))
        raise ValueError(f"unknown column {name!r}; expected one of: {valid}")
    return canonical


def is_date_only(value: str, *, decoded: bool = False) -> bool:
    """Return whether a SQL literal is ``YYYY-MM-DD`` or ``YYYYMMDD``.

    With ``decoded=True`` the value is already unquoted and inspected as is.
    """

    if not isinstance(value, str):
        raise TypeError("date literal must be a string")
    literal = value if decoded else _unquote_sql_literal(value)
    return _DATE_ONLY_PATTERN.fullmatch(literal) is not None


def parse_literal(
    value: str, column: str, *, decoded: bool = False
) -> datetime | float | str:
    """Convert a SQL literal for an already-canonical column.

    ``decoded=True`` means the value's SQL quoting was already removed.
    """

    if not isinstance(value, str):
        raise ValueError("SQL literal must be a string")
    if not isinstance(column, str) or column not in SCHEMA:
        raise ValueError(f"literal column must be canonical; received {column!r}")

    literal = value if decoded else _unquote_sql_literal(value)
    column_type = SCHEMA[column]

    if column_type is ColumnType.STRING:
        return literal

    if column_type is ColumnType.DATETIME:
        normalized = literal[:-1] + "+00:00" if literal.endswith("Z") else literal
        try:
            # fromisoformat validates syntax; pd.Timestamp below keeps nanoseconds.
            datetime.fromisoformat(normalized)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"invalid datetime literal {value!r} for column {column!r}"
            ) from exc
        try:
            return pd.Timestamp(normalized).as_unit("ns")
        except pd.errors.OutOfBoundsDatetime as exc:
            raise ValueError(
                f"datetime literal {value!r} for column {column!r} is outside the "
                f"supported range ({SUPPORTED_TIMESTAMP_RANGE})"
            ) from exc
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"invalid datetime literal {value!r} for column {column!r}"
            ) from exc

    if column_type is ColumnType.FLOAT:
        # float() would accept "nan", "inf" and "1_0"; only finite decimals are allowed.
        stripped = literal.strip()
        number = float(stripped) if _FLOAT_PATTERN.fullmatch(stripped) else math.nan
        if not math.isfinite(number):
            raise ValueError(
                f"invalid floating-point literal {value!r} for column {column!r}; "
                "only finite decimal numbers are supported"
            )
        return number

    if column_type is ColumnType.INTEGER:
        stripped = literal.strip()
        if not _INTEGER_PATTERN.fullmatch(stripped):
            raise ValueError(f"invalid integer literal {value!r} for column {column!r}")
        return int(stripped)

    raise AssertionError(f"unhandled column type: {column_type!r}")


__all__ = [
    "ALIAS_MAP",
    "MAX_TIMESTAMP",
    "MIN_TIMESTAMP",
    "SUPPORTED_TIMESTAMP_RANGE",
    "COLUMN_LABEL_MAP",
    "ColumnType",
    "SCHEMA",
    "VALID_COLUMNS",
    "is_date_only",
    "parse_literal",
    "validate_column",
]
