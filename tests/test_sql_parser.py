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
    # An exact (time-bearing) literal keeps the plain operator.
    predicate = parse_where_clause(
        "SELECT * FROM data WHERE date <= '2024-03-01 00:00:00'"
    )

    assert isinstance(predicate, Comparison)
    assert predicate.column == "timestamp"
    assert predicate.operator == "<="
    assert predicate.value == datetime(2024, 3, 1)
    assert predicate.label_field == "time"


def test_parse_less_than_equal_date_only_is_day_aware() -> None:
    # A date-only literal names the whole day, so `<= day` is `< next day`.
    predicate = parse_where_clause("SELECT * FROM data WHERE date <= '2024-03-01'")

    assert isinstance(predicate, Comparison)
    assert predicate.column == "timestamp"
    assert predicate.operator == "<"
    assert predicate.value == datetime(2024, 3, 2)
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
    # Parsing must be identical whether or not extract_as_of ran first.
    assert parse_where_clause(remaining_sql) == parse_where_clause(sql)


def test_extract_as_of_rejects_version_number_syntax() -> None:
    with pytest.raises(UnsupportedQueryError):
        extract_as_of("SELECT * FROM data VERSION AS OF 123 WHERE symbol = 'AAPL'")


def test_extract_as_of_preserves_quoted_identifier_in_where_clause() -> None:
    """A double-quoted column in WHERE must stay an identifier when AS OF is present."""

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
    """An escaped quote like 'O''Reilly' must survive extraction intact."""

    remaining_sql, as_of = extract_as_of(
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-01T00:00:00Z' "
        "WHERE symbol = 'O''Reilly'"
    )

    assert as_of is not None
    predicate = parse_where_clause(remaining_sql)
    assert isinstance(predicate, Comparison)
    assert predicate.value == "O'Reilly"


def test_extract_as_of_not_defeated_by_where_inside_a_leading_comment() -> None:
    """The word "where" in a leading comment must not hide the AS OF clause."""

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
    """ "TIMESTAMP AS OF" inside a column alias is not an AS OF clause."""

    sql = "SELECT close AS \"TIMESTAMP AS OF trick\" FROM data WHERE symbol = 'AAPL'"
    remaining_sql, as_of = extract_as_of(sql)

    assert as_of is None
    assert remaining_sql == sql
    # Extraction ignores the projection; parsing must still reject it.
    with pytest.raises(UnsupportedQueryError, match="SELECT \\*"):
        parse_where_clause(remaining_sql)


def test_extract_as_of_ordinary_timestamp_column_reference_is_unaffected() -> None:
    """A plain ``timestamp`` column reference is never mistaken for AS OF."""

    sql = "SELECT * FROM data WHERE timestamp >= '2024-01-01'"
    remaining_sql, as_of = extract_as_of(sql)

    assert as_of is None
    assert remaining_sql == sql


def test_extract_as_of_rejects_non_literal_marker_without_leaking_raw_error() -> None:
    """A malformed AS OF raises UnsupportedQueryError, not a raw parser error."""

    with pytest.raises(UnsupportedQueryError, match="AS OF requires a literal"):
        extract_as_of("SELECT * FROM data TIMESTAMP AS OF symbol WHERE symbol = 'AAPL'")


def test_in_rejects_date_only_literal() -> None:
    """IN has no whole-day expansion, so date-only literals are rejected."""

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
