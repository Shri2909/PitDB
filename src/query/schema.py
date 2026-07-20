"""Canonical query schema, SQL aliases, and typed literal conversion.

Aliases exist only at the SQL-facing boundary. Once :func:`validate_column`
returns, every parser and executor component works exclusively with real
columns produced by ``Chunk.decompress()``.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum


class ColumnType(Enum):
    STRING = "string"
    DATETIME = "datetime"
    FLOAT = "float"
    INTEGER = "integer"


# These names exactly match the columns reconstructed by Chunk.decompress().
SCHEMA: dict[str, ColumnType] = {
    "symbol": ColumnType.STRING,
    "timestamp": ColumnType.DATETIME,
    "open": ColumnType.FLOAT,
    "high": ColumnType.FLOAT,
    "low": ColumnType.FLOAT,
    "close": ColumnType.FLOAT,
    "volume": ColumnType.FLOAT,
}


# SQL conveniences are resolved before a Comparison/In node is constructed.
ALIAS_MAP: dict[str, str] = {
    "date": "timestamp",
    "price": "close",
}

VALID_COLUMNS = set(SCHEMA) | set(ALIAS_MAP)


# Keys are canonical DataFrame columns only. ``price`` below names the label
# category backed by ChunkLabel.min_price/max_price; it is not a SQL alias.
COLUMN_LABEL_MAP: dict[str, str | None] = {
    "symbol": "symbol",
    "timestamp": "time",
    "open": None,
    "high": None,
    "low": None,
    "close": "price",
    "volume": None,
}


_DATE_ONLY_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}\Z")


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
    """Validate a SQL column reference and return its canonical DataFrame name.

    This function must be called while parsing, immediately after extracting a
    column name from the SQL AST. Downstream query nodes must never retain an
    alias such as ``date`` or ``price``.
    """

    if not isinstance(name, str):
        raise ValueError("column name must be a string")

    normalized = name.strip().lower()
    canonical = ALIAS_MAP.get(normalized, normalized)
    if canonical not in SCHEMA:
        valid = ", ".join(sorted(VALID_COLUMNS))
        raise ValueError(f"unknown column {name!r}; expected one of: {valid}")
    return canonical


def is_date_only(value: str) -> bool:
    """Return whether a SQL literal has exactly ``YYYY-MM-DD`` syntax.

    Quotes and surrounding whitespace are ignored. This syntactic flag is kept
    separate from :func:`parse_literal` so that function can retain its required
    ``datetime | float | str`` return contract. The SQL parser uses the flag to
    expand date equality into the appropriate whole-day interval.
    """

    if not isinstance(value, str):
        raise TypeError("date literal must be a string")
    return _DATE_ONLY_PATTERN.fullmatch(_unquote_sql_literal(value)) is not None


def parse_literal(value: str, column: str) -> datetime | float | str:
    """Convert a SQL literal according to an already-canonical column.

    ``column`` is intentionally not passed through :func:`validate_column`:
    accepting an alias here would allow deferred alias resolution to creep into
    execution. Callers must canonicalize once, at parse time, before invoking
    this function.
    """

    if not isinstance(value, str):
        raise ValueError("SQL literal must be a string")
    if not isinstance(column, str) or column not in SCHEMA:
        raise ValueError(f"literal column must be canonical; received {column!r}")

    literal = _unquote_sql_literal(value)
    column_type = SCHEMA[column]

    if column_type is ColumnType.STRING:
        return literal

    if column_type is ColumnType.DATETIME:
        # datetime.fromisoformat accepts date-only values as midnight. Normalize
        # the common SQL/ISO UTC suffix explicitly for portable behavior.
        normalized = literal[:-1] + "+00:00" if literal.endswith("Z") else literal
        try:
            return datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValueError(
                f"invalid datetime literal {value!r} for column {column!r}"
            ) from exc

    if column_type is ColumnType.FLOAT:
        try:
            return float(literal)
        except ValueError as exc:
            raise ValueError(
                f"invalid floating-point literal {value!r} for column {column!r}"
            ) from exc

    if column_type is ColumnType.INTEGER:
        try:
            return int(literal)
        except ValueError as exc:
            raise ValueError(
                f"invalid integer literal {value!r} for column {column!r}"
            ) from exc

    raise AssertionError(f"unhandled column type: {column_type!r}")


__all__ = [
    "ALIAS_MAP",
    "COLUMN_LABEL_MAP",
    "ColumnType",
    "SCHEMA",
    "VALID_COLUMNS",
    "is_date_only",
    "parse_literal",
    "validate_column",
]
