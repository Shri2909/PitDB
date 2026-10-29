"""Regression coverage for the supported complete SQL statement contract."""

from datetime import datetime, timezone

import numpy as np
import pytest

from src.query.executor import QueryExecutor
from src.query.predicate import And, Comparison, In
from src.query.sql_parser import (
    UnsupportedQueryError,
    extract_as_of,
    parse_where_clause,
)
from src.store.chunk_store import ChunkStore


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM data LIMIT 0",
        "SELECT * FROM data ORDER BY close DESC",
        "SELECT close FROM data",
        "SELECT no_such_column FROM data",
        "SELECT DISTINCT symbol FROM data",
        "SELECT * FROM data GROUP BY symbol",
        "SELECT * FROM data UNION SELECT * FROM other",
        "DELETE FROM data WHERE symbol = 'AAPL'",
        "DROP TABLE data",
        "SELECT * FROM data; SELECT * FROM data WHERE symbol = 'MSFT'",
        "SELECT * FROM data HAVING close > 100000",
        "SELECT * FROM data FOR TIMESTAMP AS OF '2024-03-01T12:00:00Z'",
        "SELECT * FROM other",
        "SELECT * FROM database.data",
        "SELECT * FROM catalog.database.data",
        "SELECT * FROM data, data",
        "SELECT * FROM data TABLESAMPLE (10 PERCENT)",
        "WITH d AS (SELECT * FROM data) SELECT * FROM d",
        "SELECT * FROM data WHERE other.symbol = 'AAPL'",
        "SELECT * FROM data WHERE database.data.symbol = 'AAPL'",
        "SELECT * FROM data WHERE \"\".symbol = 'AAPL'",
        'SELECT * FROM data AS ""',
        "SELECT * FROM data AS d WHERE data.symbol = 'AAPL'",
        "SELECT data.* FROM data",
        "SELECT * EXCEPT (close) FROM data",
        "SELECT * FROM data AS d(other)",
        "SELECT * FROM data OFFSET 1",
        "SELECT * FROM data WHERE close = 11 QUALIFY close > 0",
        "SELECT * FROM data WINDOW w AS (PARTITION BY symbol)",
        "SELECT * FROM data WHERE symbol IN ()",
    ],
)
def test_unused_statement_semantics_are_rejected(sql: str) -> None:
    store = ChunkStore()
    store.ingest_row(
        datetime(2024, 3, 1, 12, tzinfo=timezone.utc),
        "AAPL",
        np.array([10, 12, 9, 11, 100], dtype=np.float64),
    )
    store.flush_all()
    with pytest.raises(UnsupportedQueryError):
        QueryExecutor(store).execute(sql)


@pytest.mark.parametrize(
    ("stored", "literal"),
    [
        (" AAPL ", "' AAPL '"),
        ("'AAPL'", "'''AAPL'''"),
        ('"AAPL"', "'\"AAPL\"'"),
        ("O''Reilly", "'O''''Reilly'"),
        ("   ", "'   '"),
        ("", "''"),
        ("-- WHERE; TIMESTAMP AS OF", "'-- WHERE; TIMESTAMP AS OF'"),
    ],
)
def test_decoded_string_literal_is_preserved(stored: str, literal: str) -> None:
    comparison = parse_where_clause(f"SELECT * FROM data WHERE symbol = {literal}")
    membership = parse_where_clause(f"SELECT * FROM data WHERE symbol IN ({literal})")
    assert isinstance(comparison, Comparison)
    assert comparison.value == stored
    assert comparison.evaluate_against_row({"symbol": stored})
    assert isinstance(membership, In)
    assert membership.values == [stored]
    assert membership.evaluate_against_row({"symbol": stored})


@pytest.mark.parametrize(
    ("symbol", "literal"),
    [
        (" AAPL ", "' AAPL '"),
        ("'AAPL'", "'''AAPL'''"),
        ('"AAPL"', "'\"AAPL\"'"),
        ("O''Reilly", "'O''''Reilly'"),
    ],
)
def test_exact_string_literal_matches_stored_row(symbol: str, literal: str) -> None:
    store = ChunkStore()
    store.ingest_row(
        datetime(2024, 3, 1, 12, tzinfo=timezone.utc),
        symbol,
        np.array([10, 12, 9, 11, 100], dtype=np.float64),
    )
    store.flush_all()
    executor = QueryExecutor(store)
    for temporal_clause in ("", " TIMESTAMP AS OF '2024-03-02T00:00:00Z'"):
        result = executor.execute(
            f"SELECT * FROM data{temporal_clause} WHERE symbol = {literal}"
        )
        assert result.data["symbol"].tolist() == [symbol]


@pytest.mark.parametrize("qualifier", ["data", "DATA", "d", '"d"'])
def test_matching_qualifier_is_supported(qualifier: str) -> None:
    alias = " AS d" if qualifier.lower() in {"d", '"d"'} else ""
    predicate = parse_where_clause(
        f"SELECT * FROM data{alias} WHERE {qualifier}.price = 11"
    )
    assert isinstance(predicate, Comparison)
    assert (predicate.column, predicate.value) == ("close", 11.0)


@pytest.mark.parametrize("sql", ["", " ", "-- comment", "SELECT * FROM", "SELECT ("])
def test_invalid_sql_raises_project_error(sql: str) -> None:
    with pytest.raises(UnsupportedQueryError):
        parse_where_clause(sql)


def test_one_trailing_semicolon_and_comments_are_supported() -> None:
    predicate = parse_where_clause(
        "/* SELECT * FROM other; */ SELECT * FROM data "
        "WHERE symbol = 'WHERE; FROM' /* ORDER BY close */; -- extra ;"
    )
    assert isinstance(predicate, Comparison)
    assert predicate.value == "WHERE; FROM"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM data; ;",
        ";SELECT * FROM data",
        "SELECT * FROM data WHERE " + "(" * 200 + "close = 1" + ")" * 200,
        "SELECT * FROM data WHERE " + " AND ".join(["close = 1"] * 3000),
        "SELECT * FROM data WHERE symbol = '" + "a" * 70000 + "'",
    ],
    ids=["extra-statement", "leading-semicolon", "nesting", "and-count", "length"],
)
def test_out_of_bounds_sql_raises_project_error(sql: str) -> None:
    with pytest.raises(UnsupportedQueryError):
        parse_where_clause(sql)
    with pytest.raises(UnsupportedQueryError):
        extract_as_of(sql)


def test_bounded_nested_and_expression_is_supported() -> None:
    predicate = parse_where_clause(
        "SELECT * FROM data WHERE "
        + "(" * 20
        + " AND ".join(["close >= 1"] * 40)
        + ")" * 20
    )
    assert isinstance(predicate, And)
    assert len(predicate.children) == 40


def test_unrepresentable_date_equality_is_rejected() -> None:
    with pytest.raises(UnsupportedQueryError):
        parse_where_clause("SELECT * FROM data WHERE date = '9999-12-31'")
