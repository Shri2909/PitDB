"""End-to-end equivalence and diagnostics tests for indexed SQL execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.query.executor as executor_module
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.query.sql_parser import UnsupportedQueryError, parse_where_clause
from src.store.chunk_store import ChunkStore

QUERIES = {
    "Q1": ("SELECT * FROM data WHERE symbol = 'AAPL' AND date = '2024-03-01'"),
    "Q2": (
        "SELECT * FROM data WHERE symbol = 'AAPL' "
        "AND date >= '2024-03-01' AND date < '2024-04-01'"
    ),
    "Q3": (
        "SELECT * FROM data WHERE symbol IN ('AAPL', 'MSFT') "
        "AND date >= '2024-03-01' AND date < '2024-03-08'"
    ),
    "Q4": (
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 150 "
        "AND date >= '2024-01-01'"
    ),
    "Q5": "SELECT * FROM data WHERE date >= '2024-01-01'",
}


@pytest.fixture(scope="module")
def query_store() -> ChunkStore:
    # Four calendar months and two symbols yield eight independently prunable
    # chunks. March 1 deliberately has non-midnight observations to exercise
    # date-only equality rather than exact-instant equality.
    timestamps = pd.to_datetime(
        [
            "2024-01-02 09:30:00Z",
            "2024-01-15 09:30:00Z",
            "2024-02-01 09:30:00Z",
            "2024-02-15 09:30:00Z",
            "2024-03-01 09:30:00Z",
            "2024-03-01 15:30:00Z",
            "2024-03-05 09:30:00Z",
            "2024-04-01 09:30:00Z",
            "2024-04-15 09:30:00Z",
        ],
        utc=True,
    )

    rows: list[dict[str, object]] = []
    for position, timestamp in enumerate(timestamps):
        for symbol in ("AAPL", "MSFT"):
            close = (
                145.0 + position * 2.0 if symbol == "AAPL" else 300.0 + position * 2.0
            )
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": close - 0.25,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000.0 + position,
                }
            )

    frame = pd.DataFrame(rows)
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = frame[column].astype(np.float64)

    store = ChunkStore()
    store.ingest_dataframe(frame)
    assert store.total_chunks == 8
    assert store.total_rows == len(frame) == 18
    return store


def test_result_equivalence_Q1_through_Q5(
    query_store: ChunkStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = QueryExecutor(query_store)

    for name, sql in QUERIES.items():
        predicate = parse_where_clause(sql)

        # QueryExecutor normally parses internally. Injecting this exact object
        # ensures indexed and brute-force paths share one predicate tree, so the
        # only variable under test is whether chunk skipping loses valid rows.
        monkeypatch.setattr(
            executor_module,
            "parse_where_clause",
            lambda _sql, predicate=predicate: predicate,
        )
        indexed = executor.execute(sql).data
        full_scan = query_store.full_scan(predicate.evaluate_against_row)

        pd.testing.assert_frame_equal(
            indexed.reset_index(drop=True),
            full_scan.reset_index(drop=True),
            check_exact=True,
            obj=f"indexed/full-scan equivalence for {name}",
        )


def test_full_scan_matches_pushdown_query(query_store: ChunkStore) -> None:
    full_scan = FullScanQueryEngine(query_store)
    indexed = QueryExecutor(query_store)

    for name, sql in QUERIES.items():
        expected = full_scan.execute(sql)
        actual = indexed.execute(sql)

        pd.testing.assert_frame_equal(
            actual.data.reset_index(drop=True),
            expected.data.reset_index(drop=True),
            check_exact=True,
            obj=f"full-scan/pushdown-query equivalence for {name}",
        )
        assert actual.row_count == expected.row_count


def test_full_scan_matches_pushdown_for_datetime_in_clause(
    query_store: ChunkStore,
) -> None:
    # Regression test: In.evaluate_against_row used to skip the tz coercion
    # In.evaluate_vectorized applies, so a tz-aware stored timestamp compared
    # against the naive datetimes an IN-clause literal parses to (no SQL
    # literal here carries an explicit UTC offset) silently returned False
    # under Python's `in` semantics -- the naive full-scan baseline would
    # undercount matches for any timestamp/price IN query. symbol IN doesn't
    # exercise this (strings need no tz coercion), so QUERIES above never
    # caught it.
    # Exact stored-row instants (not date-only literals -- IN does exact
    # equality, it doesn't get the whole-day expansion `=` gets), each held
    # by two rows (AAPL and MSFT) in the query_store fixture.
    sql = (
        "SELECT * FROM data WHERE timestamp IN "
        "('2024-03-01 09:30:00', '2024-04-01 09:30:00')"
    )

    full_scan_result = FullScanQueryEngine(query_store).execute(sql)
    pushdown_result = QueryExecutor(query_store).execute(sql)

    assert full_scan_result.row_count > 0
    pd.testing.assert_frame_equal(
        full_scan_result.data.reset_index(drop=True),
        pushdown_result.data.reset_index(drop=True),
        check_exact=True,
        obj="full-scan/pushdown-query equivalence for date IN (...)",
    )


def test_query_returns_correct_row_count(query_store: ChunkStore) -> None:
    # The synthetic dataset has exactly three AAPL observations in March:
    # March 1 at 09:30 and 15:30, plus March 5 at 09:30.
    result = QueryExecutor(query_store).execute(QUERIES["Q2"])

    assert result.row_count == 3
    assert len(result.data) == 3


def test_selectivity_calculation(query_store: ChunkStore) -> None:
    result = QueryExecutor(query_store).execute(QUERIES["Q3"])

    assert result.row_count == 6
    assert result.selectivity == result.row_count / query_store.total_rows
    assert result.selectivity == 6 / 18


def test_chunks_skipped_positive(query_store: ChunkStore) -> None:
    result = QueryExecutor(query_store).execute(QUERIES["Q1"])

    assert result.pushdown.candidate_count == 1
    assert result.pushdown.total_chunks == 8
    assert result.pushdown.chunks_skipped_ratio == 7 / 8
    assert result.pushdown.chunks_skipped_ratio > 0.5


def test_broad_query_skips_few_chunks(query_store: ChunkStore) -> None:
    result = QueryExecutor(query_store).execute(QUERIES["Q5"])

    assert result.pushdown.candidate_count == query_store.total_chunks
    assert result.pushdown.skipped_chunk_ids == []
    assert result.pushdown.chunks_skipped_ratio == 0.0


@pytest.mark.parametrize(
    "sql",
    [
        ("SELECT * FROM data WHERE symbol = 'AAPL' OR price > 150"),
        (
            "SELECT * FROM data AS a JOIN other AS b "
            "ON a.symbol = b.symbol WHERE a.price > 150"
        ),
        "SELECT AVG(price) FROM data WHERE symbol = 'AAPL'",
    ],
    ids=["or", "join", "aggregate"],
)
def test_unsupported_query_handling(query_store: ChunkStore, sql: str) -> None:
    with pytest.raises(UnsupportedQueryError):
        QueryExecutor(query_store).execute(sql)


def test_date_equality_matches_whole_day(query_store: ChunkStore) -> None:
    result = QueryExecutor(query_store).execute(QUERIES["Q1"])

    assert not result.data.empty
    assert result.row_count == 2
    assert result.data["symbol"].tolist() == ["AAPL", "AAPL"]

    timestamps = result.data["timestamp"]
    lower = pd.Timestamp("2024-03-01 00:00:00", tz="UTC")
    upper = pd.Timestamp("2024-03-02 00:00:00", tz="UTC")
    assert ((timestamps >= lower) & (timestamps < upper)).all()
    assert timestamps.tolist() == [
        pd.Timestamp("2024-03-01 09:30:00", tz="UTC"),
        pd.Timestamp("2024-03-01 15:30:00", tz="UTC"),
    ]
    assert all(timestamp.hour != 0 for timestamp in timestamps)
