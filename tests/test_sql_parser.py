"""Tests for SQL-to-predicate conversion and bounded-query validation."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.query.predicate import AlwaysTrue, And, Comparison, In
from src.query.sql_parser import (
    UnsupportedQueryError,
    extract_as_of,
    parse_where_clause,
)


def test_parse_simple_equality() -> None:
    predicate = parse_where_clause("SELECT * FROM data WHERE symbol = 'AAPL'")

    assert isinstance(predicate, Comparison)
    assert predicate.column == "symbol"
    assert predicate.operator == "="
    assert predicate.value == "AAPL"
    assert predicate.label_field == "symbol"


def test_parse_greater_than() -> None:
    predicate = parse_where_clause("SELECT * FROM data WHERE price > 150")

    assert isinstance(predicate, Comparison)
    assert predicate.column == "close"
    assert predicate.operator == ">"
    assert predicate.value == 150.0
    assert predicate.label_field == "price"


def test_parse_less_than_equal() -> None:
    predicate = parse_where_clause("SELECT * FROM data WHERE date <= '2024-03-01'")

    assert isinstance(predicate, Comparison)
    assert predicate.column == "timestamp"
    assert predicate.operator == "<="
    assert predicate.value == datetime(2024, 3, 1)
    assert predicate.label_field == "time"


def test_parse_and() -> None:
    predicate = parse_where_clause(
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 150"
    )

    assert isinstance(predicate, And)
    assert len(predicate.children) == 2
    symbol, price = predicate.children
    assert isinstance(symbol, Comparison)
    assert (symbol.column, symbol.operator, symbol.value) == (
        "symbol",
        "=",
        "AAPL",
    )
    assert isinstance(price, Comparison)
    assert (price.column, price.operator, price.value) == (
        "close",
        ">",
        150.0,
    )


def test_parse_triple_and() -> None:
    predicate = parse_where_clause(
        "SELECT * FROM data WHERE symbol = 'AAPL' "
        "AND price > 150 AND date >= '2024-01-01'"
    )

    assert isinstance(predicate, And)
    assert len(predicate.children) == 3
    assert all(isinstance(child, Comparison) for child in predicate.children)
    assert [child.column for child in predicate.children] == [
        "symbol",
        "close",
        "timestamp",
    ]
    assert [child.operator for child in predicate.children] == ["=", ">", ">="]
    assert [child.value for child in predicate.children] == [
        "AAPL",
        150.0,
        datetime(2024, 1, 1),
    ]


def test_parse_in() -> None:
    predicate = parse_where_clause(
        "SELECT * FROM data WHERE symbol IN ('AAPL', 'MSFT')"
    )

    assert isinstance(predicate, In)
    assert predicate.column == "symbol"
    assert predicate.values == ["AAPL", "MSFT"]
    assert predicate.label_field == "symbol"


def test_parse_no_where() -> None:
    predicate = parse_where_clause("SELECT * FROM data")
    assert isinstance(predicate, AlwaysTrue)


def test_reject_or() -> None:
    with pytest.raises(
        UnsupportedQueryError,
        match="OR across columns is not supported",
    ):
        parse_where_clause("SELECT * FROM data WHERE symbol = 'AAPL' OR price > 150")


def test_reject_subquery() -> None:
    with pytest.raises(
        UnsupportedQueryError,
        match="Subqueries are not supported",
    ):
        parse_where_clause(
            "SELECT * FROM data WHERE price IN (SELECT price FROM historical_data)"
        )


def test_reject_join() -> None:
    with pytest.raises(
        UnsupportedQueryError,
        match="JOINs are not supported",
    ):
        parse_where_clause(
            "SELECT * FROM a JOIN b ON a.symbol = b.symbol WHERE a.price > 150"
        )


def test_reject_aggregation() -> None:
    expected = (
        "Aggregation functions are not supported; only WHERE-clause "
        "filtering is executed."
    )
    with pytest.raises(UnsupportedQueryError, match=expected):
        parse_where_clause("SELECT AVG(price) FROM data WHERE symbol = 'AAPL'")


def test_column_validation() -> None:
    with pytest.raises(ValueError, match="unknown column"):
        parse_where_clause("SELECT * FROM data WHERE nonexistent_column = 5")


def test_reversed_comparison() -> None:
    predicate = parse_where_clause("SELECT * FROM data WHERE 150 < price")

    assert isinstance(predicate, Comparison)
    assert predicate.column == "close"
    assert predicate.operator == ">"
    assert predicate.value == 150.0
    assert predicate.label_field == "price"


def test_date_equality_expands_to_range() -> None:
    predicate = parse_where_clause("SELECT * FROM data WHERE date = '2024-03-01'")

    assert isinstance(predicate, And)
    assert len(predicate.children) == 2
    lower, upper = predicate.children
    assert isinstance(lower, Comparison)
    assert isinstance(upper, Comparison)
    assert (lower.column, lower.operator, lower.value, lower.label_field) == (
        "timestamp",
        ">=",
        datetime(2024, 3, 1, 0, 0, 0),
        "time",
    )
    assert (upper.column, upper.operator, upper.value, upper.label_field) == (
        "timestamp",
        "<",
        datetime(2024, 3, 2, 0, 0, 0),
        "time",
    )


def test_datetime_equality_with_time_not_expanded() -> None:
    predicate = parse_where_clause(
        "SELECT * FROM data WHERE timestamp = '2024-03-01 09:31:00'"
    )

    assert isinstance(predicate, Comparison)
    assert predicate.column == "timestamp"
    assert predicate.operator == "="
    assert predicate.value == datetime(2024, 3, 1, 9, 31, 0)
    assert predicate.label_field == "time"


def test_alias_columns_resolve_to_canonical_names() -> None:
    date_predicate = parse_where_clause("SELECT * FROM data WHERE date >= '2024-01-01'")
    price_predicate = parse_where_clause("SELECT * FROM data WHERE price > 150")

    assert isinstance(date_predicate, Comparison)
    assert isinstance(price_predicate, Comparison)
    assert date_predicate.column == "timestamp"
    assert price_predicate.column == "close"
    assert date_predicate.column != "date"
    assert price_predicate.column != "price"


def test_extract_as_of_strips_clause_and_parses_timestamp() -> None:
    remaining_sql, as_of = extract_as_of(
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-01T00:00:00Z' "
        "WHERE symbol = 'AAPL'"
    )

    assert as_of == datetime(2024, 1, 1, tzinfo=as_of.tzinfo)
    predicate = parse_where_clause(remaining_sql)
    assert isinstance(predicate, Comparison)
    assert predicate.column == "symbol"
    assert predicate.value == "AAPL"


def test_extract_as_of_absent_returns_sql_unchanged() -> None:
    sql = "SELECT * FROM data WHERE symbol = 'AAPL'"
    remaining_sql, as_of = extract_as_of(sql)

    assert as_of is None
    assert remaining_sql == sql
    # Backward compatibility: every existing query must parse identically
    # whether or not it's first passed through extract_as_of.
    assert parse_where_clause(remaining_sql) == parse_where_clause(sql)


def test_extract_as_of_rejects_version_number_syntax() -> None:
    with pytest.raises(UnsupportedQueryError):
        extract_as_of("SELECT * FROM data VERSION AS OF 123 WHERE symbol = 'AAPL'")


def test_extract_as_of_preserves_quoted_identifier_in_where_clause() -> None:
    """Regression: the old sqlglot-databricks-dialect extraction re-serialized
    the entire statement, and Databricks treats a double-quoted string as a
    literal rather than a quoted identifier -- silently turning a quoted
    column reference into two competing string literals whenever AS OF was
    present, even though the same WHERE clause parses fine on its own.
    """

    remaining_sql, as_of = extract_as_of(
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-01T00:00:00Z' "
        "WHERE \"symbol\" = 'AAPL'"
    )

    assert as_of is not None
    predicate = parse_where_clause(remaining_sql)
    assert isinstance(predicate, Comparison)
    assert predicate.column == "symbol"
    assert predicate.value == "AAPL"


def test_extract_as_of_preserves_escaped_quote_literal() -> None:
    """Regression: Databricks treats adjacent quoted strings as implicit
    concatenation, so a legitimately escaped literal like 'O''Reilly' used
    to be corrupted into CONCAT('O', 'Reilly') whenever AS OF was present.
    """

    remaining_sql, as_of = extract_as_of(
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-01T00:00:00Z' "
        "WHERE symbol = 'O''Reilly'"
    )

    assert as_of is not None
    predicate = parse_where_clause(remaining_sql)
    assert isinstance(predicate, Comparison)
    assert predicate.value == "O'Reilly"


def test_extract_as_of_not_defeated_by_where_inside_a_leading_comment() -> None:
    """Regression: a prior regex-based extraction searched for TIMESTAMP AS OF
    only in the text before the first bare "WHERE" match -- defeated by any
    earlier occurrence of the ordinary English word "where", such as inside a
    SQL comment, which silently truncated the search before it ever reached
    the real clause (leaving as_of=None and the leftover AS OF text corrupting
    the downstream parse).
    """

    sql = (
        "-- pull rows where the AS OF snapshot matters\n"
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-01T00:00:00Z' "
        "WHERE symbol = 'AAPL'"
    )
    remaining_sql, as_of = extract_as_of(sql)

    assert as_of == datetime(2024, 1, 1, tzinfo=as_of.tzinfo)
    predicate = parse_where_clause(remaining_sql)
    assert isinstance(predicate, Comparison)
    assert predicate.column == "symbol"
    assert predicate.value == "AAPL"


def test_extract_as_of_ignores_the_phrase_inside_an_alias() -> None:
    """Regression: the same fragile boundary search also falsely rejected
    valid queries that merely contained the phrase "TIMESTAMP AS OF" inside a
    column alias with no real AS OF clause present at all.
    """

    sql = "SELECT close AS \"TIMESTAMP AS OF trick\" FROM data WHERE symbol = 'AAPL'"
    remaining_sql, as_of = extract_as_of(sql)

    assert as_of is None
    assert remaining_sql == sql
    predicate = parse_where_clause(remaining_sql)
    assert isinstance(predicate, Comparison)
    assert predicate.column == "symbol"


def test_extract_as_of_ordinary_timestamp_column_reference_is_unaffected() -> None:
    """The tokenizer sees "timestamp" as TokenType.TIMESTAMP even as a plain
    column reference (this project's own queries write this constantly) --
    extraction must only match that token immediately after the table name,
    never elsewhere, or every ordinary date-range query would misfire.
    """

    sql = "SELECT * FROM data WHERE timestamp >= '2024-01-01'"
    remaining_sql, as_of = extract_as_of(sql)

    assert as_of is None
    assert remaining_sql == sql


def test_extract_as_of_rejects_non_literal_marker_without_leaking_raw_error() -> None:
    """A TIMESTAMP token immediately after the table name that isn't followed
    by a complete AS OF '<literal>' sequence is a malformed attempt at this
    syntax, in that specific structural position -- must raise the project's
    own UnsupportedQueryError, never a raw exception from a downstream parser.
    """

    with pytest.raises(UnsupportedQueryError, match="AS OF requires a literal"):
        extract_as_of("SELECT * FROM data TIMESTAMP AS OF symbol WHERE symbol = 'AAPL'")


def test_in_rejects_date_only_literal() -> None:
    """Regression: `=` expands a date-only literal into a whole-day range,
    but IN never did -- silently matching only exact midnight and dropping
    every intraday row for that date. Reject explicitly instead.
    """

    with pytest.raises(
        UnsupportedQueryError, match="IN with date-only literals is not supported"
    ):
        parse_where_clause(
            "SELECT * FROM data WHERE timestamp IN ('2024-03-01', '2024-03-02')"
        )


def test_in_still_accepts_exact_timestamp_literals() -> None:
    predicate = parse_where_clause(
        "SELECT * FROM data WHERE timestamp IN "
        "('2024-03-01 09:30:00', '2024-03-02 09:30:00')"
    )
    assert isinstance(predicate, In)
    assert predicate.column == "timestamp"
