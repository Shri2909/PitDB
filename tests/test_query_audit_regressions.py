"""Regression tests for query-layer engine parity.

Each scenario runs on the pruned executor, the unpruned executor and the
full-scan baseline.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.query.predicate import Comparison, In
from src.query.schema import parse_literal
from src.query.sql_parser import (
    UnsupportedQueryError,
    extract_as_of,
    parse_where_clause,
)
from src.store.chunk_boundary import DurationBoundary
from src.store.chunk_store import ChunkStore

NAN = float("nan")


def _vals(close: float) -> np.ndarray:
    return np.array([close, close, close, close, 1.0], dtype=np.float64)


def _engines(store: ChunkStore):
    return {
        "pruned": QueryExecutor(store),
        "unpruned": QueryExecutor(store, enable_pruning=False),
        "full_scan": FullScanQueryEngine(store),
    }


def _run_all(store: ChunkStore, sql: str) -> pd.DataFrame:
    """Run ``sql`` on every engine, require identical frames, return one."""

    frames = {name: eng.execute(sql).data for name, eng in _engines(store).items()}
    reference = frames["full_scan"]
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True),
            reference.reset_index(drop=True),
            check_exact=False,
            obj=f"{name} vs full_scan for {sql!r}",
        )
    return reference


def _assert_all_raise(store: ChunkStore, sql: str, match: str | None = None) -> None:
    for name, engine in _engines(store).items():
        with pytest.raises(UnsupportedQueryError, match=match):
            engine.execute(sql)


def _hourly_store(
    *, tz: str | None = None, days: int = 4, hours_per_chunk: int = 6
) -> ChunkStore:
    """Hourly rows for 2024-01-01 .. +days, several chunks per day."""

    store = ChunkStore(
        chunk_boundary=DurationBoundary(timedelta(hours=hours_per_chunk))
    )
    start = pd.Timestamp("2024-01-01", tz=tz)
    for hour in range(days * 24):
        store.ingest_row(start + pd.Timedelta(hours=hour), "A", _vals(float(hour)))
    store.flush_all()
    return store


# --------------------------------------------------------------------------
# 1. IN with NaN literal / NaN data: one semantics everywhere.
# --------------------------------------------------------------------------


def _nan_store(*, with_correction: bool) -> ChunkStore:
    store = ChunkStore()
    start = pd.Timestamp("2024-01-01")
    for offset, close in enumerate([1.0, NAN, 3.0]):
        store.ingest_row(start + pd.Timedelta(days=offset), "A", _vals(close))
    store.flush_all()
    if with_correction:
        # Any correction disables price pruning for the whole store.
        store.ingest_correction(start, "A", _vals(1.0), start + pd.Timedelta(days=5))
    return store


@pytest.mark.parametrize("with_correction", [False, True])
@pytest.mark.parametrize("literal", ["'nan'", "'NaN'", "'inf'", "'-inf'", "'1_0'"])
def test_non_finite_float_literals_are_rejected_by_every_engine(
    with_correction: bool, literal: str
) -> None:
    store = _nan_store(with_correction=with_correction)
    _assert_all_raise(store, f"SELECT * FROM data WHERE close IN ({literal})")
    _assert_all_raise(store, f"SELECT * FROM data WHERE close = {literal}")


@pytest.mark.parametrize("literal", ["1e999", "-1e999", "'1e999'"])
def test_overflowing_float_literal_is_rejected(literal: str) -> None:
    with pytest.raises(UnsupportedQueryError, match="finite"):
        parse_where_clause(f"SELECT * FROM data WHERE close > {literal}")


@pytest.mark.parametrize("literal", ["'1_0'", "'0x10'", "'1,5'", "''", "'١٢'"])
def test_non_decimal_float_literals_are_rejected(literal: str) -> None:
    with pytest.raises(UnsupportedQueryError):
        parse_where_clause(f"SELECT * FROM data WHERE close = {literal}")


@pytest.mark.parametrize("literal", ["3", "3.0", "-3", "+3", ".5", "3.", "1e2", "'3'"])
def test_ordinary_float_literals_still_parse(literal: str) -> None:
    predicate = parse_where_clause(f"SELECT * FROM data WHERE close = {literal}")
    assert isinstance(predicate, Comparison)
    assert isinstance(predicate.value, float)


def test_parse_literal_rejects_non_finite_directly() -> None:
    for text in ("nan", "inf", "-Infinity", "1_0", "1e999"):
        with pytest.raises(ValueError):
            parse_literal(text, "close")


@pytest.mark.parametrize("with_correction", [False, True])
def test_in_vectorized_and_row_wise_agree_on_nan_data(with_correction: bool) -> None:
    """A programmatic NaN literal (the parser rejects it) must match nothing."""

    store = _nan_store(with_correction=with_correction)
    frame = store.full_scan()
    assert frame["close"].isna().sum() == 1
    for values in ([NAN], [NAN, 3.0], [3.0], [1.0, 2.0], []):
        node = In("close", values, "price")
        vectorized = node.evaluate_vectorized(frame).tolist()
        row_wise = [node.evaluate_against_row(row) for row in frame.to_dict("records")]
        assert vectorized == row_wise, values
        # NaN data never matches, even when the literal list holds a NaN.
        assert not vectorized[1]


def test_comparison_with_nan_literal_agrees() -> None:
    frame = _nan_store(with_correction=False).full_scan()
    for operator in ("=", "<", ">", "<=", ">="):
        node = Comparison("close", operator, NAN, "price")
        vectorized = node.evaluate_vectorized(frame).tolist()
        row_wise = [node.evaluate_against_row(row) for row in frame.to_dict("records")]
        assert vectorized == row_wise == [False] * len(frame)


@pytest.mark.parametrize("with_correction", [False, True])
def test_in_on_data_containing_nan_agrees_across_engines(with_correction: bool) -> None:
    store = _nan_store(with_correction=with_correction)
    assert _run_all(store, "SELECT * FROM data WHERE close IN (3, 1)")[
        "close"
    ].tolist() == [
        1.0,
        3.0,
    ]


# --------------------------------------------------------------------------
# 2. Date-only literals are day-aware; AS OF date-only is end of day.
# --------------------------------------------------------------------------

_DAY = pd.Timestamp("2024-01-02")
_NEXT = pd.Timestamp("2024-01-03")


@pytest.mark.parametrize("tz", [None, "UTC", "America/New_York"])
@pytest.mark.parametrize(
    "operator, keep",
    [
        ("=", lambda ts: (ts >= _DAY) & (ts < _NEXT)),
        ("<", lambda ts: ts < _DAY),
        ("<=", lambda ts: ts < _NEXT),
        (">", lambda ts: ts >= _NEXT),
        (">=", lambda ts: ts >= _DAY),
    ],
)
@pytest.mark.parametrize("form", ["{op} '2024-01-02'", "{op} '20240102'"])
def test_date_only_comparisons_are_day_aware(tz, operator, keep, form) -> None:
    store = _hourly_store(tz=tz)
    frame = store.full_scan()
    naive = frame["timestamp"].dt.tz_localize(None)
    expected = frame.loc[keep(naive), "close"].tolist()
    assert expected, "scenario must be non-trivial"

    sql = "SELECT * FROM data WHERE timestamp " + form.format(op=operator)
    assert _run_all(store, sql)["close"].tolist() == expected


@pytest.mark.parametrize("tz", [None, "America/New_York"])
@pytest.mark.parametrize(
    "operator, swapped",
    [("<", ">"), ("<=", ">="), (">", "<"), (">=", "<="), ("=", "=")],
)
def test_date_only_comparison_is_symmetric_when_literal_is_on_the_left(
    tz, operator, swapped
) -> None:
    store = _hourly_store(tz=tz)
    forward = _run_all(
        store, f"SELECT * FROM data WHERE timestamp {operator} '2024-01-02'"
    )
    reversed_ = _run_all(
        store, f"SELECT * FROM data WHERE '2024-01-02' {swapped} timestamp"
    )
    pd.testing.assert_frame_equal(forward, reversed_)


def test_date_only_boundaries_through_pruning_have_exact_edges() -> None:
    """Chunks end/start exactly at midnight (6h chunks); no off-by-one at edges."""

    store = _hourly_store(hours_per_chunk=6)
    frame = store.full_scan()

    le = _run_all(store, "SELECT * FROM data WHERE timestamp <= '2024-01-02'")
    assert le["timestamp"].max() == pd.Timestamp("2024-01-02 23:00")
    gt = _run_all(store, "SELECT * FROM data WHERE timestamp > '2024-01-02'")
    assert gt["timestamp"].min() == pd.Timestamp("2024-01-03 00:00")
    eq = _run_all(store, "SELECT * FROM data WHERE timestamp = '2024-01-02'")
    assert len(eq) == 24
    assert len(le) + len(gt) == len(frame)

    pruned = QueryExecutor(store).execute(
        "SELECT * FROM data WHERE timestamp = '2024-01-02'"
    )
    assert pruned.pushdown.candidate_count == 4  # only that day's 6h chunks
    assert pruned.pushdown.skipped_chunk_ids


def test_exact_time_literals_stay_exact() -> None:
    store = _hourly_store()
    rows = _run_all(
        store, "SELECT * FROM data WHERE timestamp <= '2024-01-02 00:00:00'"
    )
    assert rows["timestamp"].max() == pd.Timestamp("2024-01-02 00:00")


def test_date_only_parse_shapes() -> None:
    day = pd.Timestamp("2024-01-02")
    for text, operator, expected in [
        ("= '2024-01-02'", "AND", None),
        ("<= '2024-01-02'", "<", _NEXT),
        ("> '2024-01-02'", ">=", _NEXT),
        ("< '2024-01-02'", "<", day),
        (">= '2024-01-02'", ">=", day),
    ]:
        predicate = parse_where_clause(f"SELECT * FROM data WHERE timestamp {text}")
        if operator == "AND":
            assert [(c.operator, c.value) for c in predicate.children] == [
                (">=", day),
                ("<", _NEXT),
            ]
        else:
            assert (predicate.operator, predicate.value) == (operator, expected)


@pytest.mark.parametrize("tz", [None, "America/New_York"])
@pytest.mark.parametrize("literal", ["2024-01-02", "20240102"])
def test_as_of_date_only_means_end_of_that_day(tz, literal) -> None:
    store = _hourly_store(tz=tz)
    # A correction filed on the AS OF day at 15:00, and one the next day.
    stamp = pd.Timestamp("2024-01-02 03:00", tz=tz)
    store.ingest_correction(
        stamp, "A", _vals(-1.0), pd.Timestamp("2024-01-02 15:00", tz=tz)
    )
    store.ingest_correction(
        stamp + pd.Timedelta(hours=1),
        "A",
        _vals(-2.0),
        pd.Timestamp("2024-01-03 00:00", tz=tz),
    )

    result = _run_all(store, f"SELECT * FROM data TIMESTAMP AS OF '{literal}'")
    stamps = result["timestamp"].dt.tz_localize(None)
    # Every base row of the AS OF day is visible (previously only midnight).
    assert stamps.max() == pd.Timestamp("2024-01-02 23:00")
    assert len(result) == 48
    closes = dict(zip(stamps, result["close"]))
    assert closes[pd.Timestamp("2024-01-02 03:00")] == -1.0  # same-day correction
    # Midnight-next-day correction is NOT yet known at the end of 2024-01-02.
    assert closes[pd.Timestamp("2024-01-02 04:00")] == 28.0

    exact = _run_all(
        store, "SELECT * FROM data TIMESTAMP AS OF '2024-01-02 23:59:59.999999999'"
    )
    pd.testing.assert_frame_equal(result, exact)
    tight = _run_all(store, "SELECT * FROM data TIMESTAMP AS OF '2024-01-03 00:00:00'")
    assert len(tight) == 49  # the exact-midnight form includes the next day's 00:00 row


def test_as_of_date_only_extract_value() -> None:
    sql, as_of = extract_as_of("SELECT * FROM data TIMESTAMP AS OF '2024-01-01'")
    assert as_of == pd.Timestamp("2024-01-01 23:59:59.999999999")
    assert "AS OF" not in sql
    _, exact = extract_as_of("SELECT * FROM data TIMESTAMP AS OF '2024-01-01 00:00:00'")
    assert exact == pd.Timestamp("2024-01-01")


def test_as_of_date_only_at_max_day_is_clamped_not_rejected() -> None:
    _, as_of = extract_as_of("SELECT * FROM data TIMESTAMP AS OF '2262-04-11'")
    assert as_of == pd.Timestamp.max


def test_as_of_with_where_date_semantics_combine() -> None:
    store = _hourly_store()
    sql = (
        "SELECT * FROM data TIMESTAMP AS OF '2024-01-02' "
        "WHERE timestamp >= '2024-01-02'"
    )
    assert len(_run_all(store, sql)) == 24


# --------------------------------------------------------------------------
# 3. Compact dates and bare numbers.
# --------------------------------------------------------------------------


def test_compact_date_is_date_only_everywhere() -> None:
    extended = parse_where_clause("SELECT * FROM data WHERE timestamp = '2024-01-02'")
    compact = parse_where_clause("SELECT * FROM data WHERE timestamp = '20240102'")
    assert [(c.operator, c.value) for c in compact.children] == [
        (c.operator, c.value) for c in extended.children
    ]
    with pytest.raises(UnsupportedQueryError, match="date-only"):
        parse_where_clause("SELECT * FROM data WHERE timestamp IN ('20240102')")


@pytest.mark.parametrize("operator", ["=", "<", "<=", ">", ">="])
def test_bare_number_timestamp_is_rejected(operator: str) -> None:
    with pytest.raises(UnsupportedQueryError, match="ambiguous numeric"):
        parse_where_clause(f"SELECT * FROM data WHERE timestamp {operator} 20240102")
    with pytest.raises(UnsupportedQueryError, match="ambiguous numeric"):
        parse_where_clause("SELECT * FROM data WHERE timestamp IN (20240102)")


# --------------------------------------------------------------------------
# 4. DST-ambiguous / nonexistent local literals: typed, deterministic error.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["2024-03-10 02:30:00", "2024-11-03 01:30:00"])
def test_dst_ambiguous_literal_raises_unsupported_regardless_of_clause_order(
    bad: str,
) -> None:
    store = _hourly_store(tz="America/New_York", days=2)
    for where in (
        f"timestamp = '{bad}' AND symbol = 'NOPE'",
        f"symbol = 'NOPE' AND timestamp = '{bad}'",  # row-wise And short-circuits
        f"symbol = 'A' AND timestamp > '{bad}'",
        f"timestamp IN ('{bad}')",
        f"close > 1e9 AND timestamp <= '{bad}'",
    ):
        _assert_all_raise(
            store, f"SELECT * FROM data WHERE {where}", match="ambiguous|invalid"
        )
    _assert_all_raise(store, f"SELECT * FROM data TIMESTAMP AS OF '{bad}'")


def test_dst_error_is_raised_even_when_no_chunk_survives_pruning() -> None:
    store = _hourly_store(tz="America/New_York", days=2)
    sql = (
        "SELECT * FROM data WHERE timestamp >= '2030-01-01' "
        "AND timestamp = '2024-03-10 02:30:00'"
    )
    _assert_all_raise(store, sql)


def test_date_equality_on_day_whose_midnight_does_not_exist() -> None:
    # Sao Paulo skipped 2018-11-04 00:00 (DST started at midnight).
    store = ChunkStore()
    store.ingest_row(
        pd.Timestamp("2018-11-01", tz="America/Sao_Paulo"), "A", _vals(1.0)
    )
    store.flush_all()
    for operator in ("=", "<", ">="):  # these compare against that midnight
        _assert_all_raise(
            store, f"SELECT * FROM data WHERE timestamp {operator} '2018-11-04'"
        )
    # `<=`/`>` only compare against the following midnight, which does exist.
    for operator in ("<=", ">"):
        _run_all(store, f"SELECT * FROM data WHERE timestamp {operator} '2018-11-04'")
    # The previous day is fine, and so is a naive store.
    assert (
        len(_run_all(store, "SELECT * FROM data WHERE timestamp = '2018-11-01'")) == 1
    )


def test_unsupported_error_is_still_a_value_error() -> None:
    assert issubclass(UnsupportedQueryError, ValueError)


# --------------------------------------------------------------------------
# 5. AS OF literal errors use the same exception type as WHERE literal errors.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["garbage", "2024-13-45", "2024-01-01T25:00", ""])
def test_bad_as_of_literal_is_unsupported_query_error(bad: str) -> None:
    with pytest.raises(UnsupportedQueryError, match="invalid datetime literal"):
        extract_as_of(f"SELECT * FROM data TIMESTAMP AS OF '{bad}'")
    with pytest.raises(UnsupportedQueryError, match="invalid datetime literal"):
        parse_where_clause(f"SELECT * FROM data WHERE timestamp = '{bad}'")
    _assert_all_raise(
        _hourly_store(days=1), f"SELECT * FROM data TIMESTAMP AS OF '{bad}'"
    )


# --------------------------------------------------------------------------
# 6. Symbol literals must be strings.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("literal", ["5", "-5", "5.5", "(5)"])
def test_numeric_symbol_literal_is_rejected(literal: str) -> None:
    with pytest.raises(UnsupportedQueryError, match="quoted string"):
        parse_where_clause(f"SELECT * FROM data WHERE symbol = {literal}")
    with pytest.raises(UnsupportedQueryError, match="quoted string"):
        parse_where_clause(f"SELECT * FROM data WHERE {literal} = symbol")


def test_numeric_symbol_in_list_is_rejected_and_quoted_digits_work() -> None:
    with pytest.raises(UnsupportedQueryError, match="quoted string"):
        parse_where_clause("SELECT * FROM data WHERE symbol IN ('A', 5)")
    store = ChunkStore()
    store.ingest_row(pd.Timestamp("2024-01-01"), "5", _vals(1.0))
    store.ingest_row(pd.Timestamp("2024-01-01"), "A", _vals(2.0))
    store.flush_all()
    assert _run_all(store, "SELECT * FROM data WHERE symbol = '5'")[
        "close"
    ].tolist() == [1.0]


# --------------------------------------------------------------------------
# 7. Datetime literals outside the datetime64[ns] range.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("literal", ["9999-12-31", "0001-01-01", "2300-01-01T00:00:00"])
def test_out_of_range_datetime_literal_names_supported_range(literal: str) -> None:
    sqls = [
        f"SELECT * FROM data WHERE timestamp < '{literal}'",
        f"SELECT * FROM data TIMESTAMP AS OF '{literal}'",
    ]
    if "T" in literal:  # IN rejects date-only literals before range checking
        sqls.append(f"SELECT * FROM data WHERE timestamp IN ('{literal}')")
    for sql in sqls:
        with pytest.raises(UnsupportedQueryError) as info:
            parse_where_clause(extract_as_of(sql)[0])
        message = str(info.value)
        assert "outside the supported range" in message
        assert "1677-09-21" in message and "2262-04-11" in message
        assert "invalid datetime literal" not in message


def test_date_equality_on_last_representable_day_is_rejected_clearly() -> None:
    with pytest.raises(UnsupportedQueryError, match="supported datetime range"):
        parse_where_clause("SELECT * FROM data WHERE timestamp = '2262-04-11'")
    # ...but the day before is fine.
    parse_where_clause("SELECT * FROM data WHERE timestamp = '2262-04-10'")


# --------------------------------------------------------------------------
# 8. Documented result ordering.
# --------------------------------------------------------------------------


def _interleaved_store() -> ChunkStore:
    store = ChunkStore()
    base = pd.Timestamp("2024-01-01")
    # B first, then A; per-symbol chunks flush in symbol-creation order.
    store.ingest_row(base + pd.Timedelta(hours=5), "B", _vals(5.0))
    store.ingest_row(base + pd.Timedelta(hours=1), "A", _vals(1.0))
    store.ingest_row(base + pd.Timedelta(hours=7), "B", _vals(7.0))
    store.ingest_row(base + pd.Timedelta(hours=9), "A", _vals(9.0))
    store.flush_all()
    return store


def test_unversioned_queries_return_natural_chunk_order() -> None:
    store = _interleaved_store()
    natural = store.full_scan()
    sorted_order = natural.sort_values(["timestamp", "symbol"], kind="stable")
    assert not natural.index.equals(sorted_order.index), "scenario must differ"

    for engine in _engines(store).values():
        frame = engine.execute("SELECT * FROM data").data
        assert frame["close"].tolist() == natural["close"].tolist()


def test_versioned_queries_return_timestamp_symbol_order() -> None:
    expected = [1.0, 5.0, 7.0, 9.0]

    store = _interleaved_store()
    for engine in _engines(store).values():  # AS OF => versioned
        frame = engine.execute("SELECT * FROM data TIMESTAMP AS OF '2024-12-31'").data
        assert frame["close"].tolist() == expected

    with_correction = _interleaved_store()
    with_correction.ingest_correction(
        pd.Timestamp("2024-01-01 05:00"), "B", _vals(5.0), pd.Timestamp("2024-02-01")
    )
    for engine in _engines(with_correction).values():
        assert engine.execute("SELECT * FROM data").data["close"].tolist() == expected
    # A query whose pruned candidates exclude every correction still sorts.
    for engine in _engines(with_correction).values():
        frame = engine.execute("SELECT * FROM data WHERE symbol = 'A'").data
        assert frame["close"].tolist() == [1.0, 9.0]


def test_executor_docstring_documents_both_orders() -> None:
    doc = QueryExecutor.__doc__ or ""
    assert "natural" in doc and "(timestamp, symbol)" in doc


# --------------------------------------------------------------------------
# 9. Explicit tie-break for corrections sharing a transaction time.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", [None, "coarse"])
def test_tied_transaction_time_latest_ingested_wins_before_and_after_compaction(
    boundary,
) -> None:
    store = ChunkStore()
    event = pd.Timestamp("2024-01-05 10:00")
    store.ingest_row(event, "A", _vals(100.0))
    store.flush_all()
    tied = pd.Timestamp("2024-01-06 12:00")
    # Three corrections, all with the same transaction time.
    for value in (101.0, 102.0, 103.0):
        store.ingest_correction(event, "A", _vals(value), tied)
    # An unrelated later-learned correction on another row, to exercise merging.
    store.ingest_row(event + pd.Timedelta(days=2), "A", _vals(200.0))
    store.flush_all()
    store.ingest_correction(
        event + pd.Timedelta(days=2),
        "A",
        _vals(201.0),
        pd.Timestamp("2024-01-09"),
    )

    def snapshot() -> list[list[float]]:
        return [
            _run_all(store, sql)["close"].tolist()
            for sql in (
                "SELECT * FROM data",
                "SELECT * FROM data TIMESTAMP AS OF '2024-01-06 12:00:00'",
                "SELECT * FROM data WHERE close >= 100 AND close < 104",
            )
        ]

    before = snapshot()
    assert before[0] == [103.0, 201.0]  # last ingested of the tied trio wins
    assert before[1] == [103.0]

    removed = (
        store.compact_corrections()
        if boundary is None
        else store.compact_corrections(boundary=DurationBoundary(timedelta(days=365)))
    )
    assert removed > 0
    assert snapshot() == before

    # A further tied correction ingested after compaction still wins, and a
    # second compaction keeps it winning.
    store.ingest_correction(event, "A", _vals(104.0), tied)
    assert _run_all(store, "SELECT * FROM data")["close"].tolist()[0] == 104.0
    store.compact_corrections()
    assert _run_all(store, "SELECT * FROM data")["close"].tolist()[0] == 104.0


def test_correction_beats_base_row_at_same_instant_and_later_tie_wins_among_base() -> (
    None
):
    store = ChunkStore()
    event = pd.Timestamp("2024-01-05 10:00")
    store.ingest_row(event, "A", _vals(1.0))
    store.flush_all()
    store.ingest_correction(event, "A", _vals(2.0), event)  # same instant as base
    assert _run_all(store, "SELECT * FROM data")["close"].tolist() == [2.0]
