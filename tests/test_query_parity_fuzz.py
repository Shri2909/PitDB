"""Randomized, deterministic parity tests across every query path.

Predicate nodes must agree between vectorized and row-wise evaluation, and all
engines must match an independent pure-Python bitemporal oracle on random
stores. A literal in a DST gap/fold must make every engine raise
``UnsupportedQueryError``.
"""

from __future__ import annotations

import random
from datetime import timedelta

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.query.predicate import And, Comparison, In
from src.query.sql_parser import UnsupportedQueryError
from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary
from src.store.chunk_store import ChunkStore

NAN = float("nan")

# --------------------------------------------------------------------------
# Layer 1: vectorized vs row-wise predicate evaluation.
# --------------------------------------------------------------------------

_closes = st.sampled_from([1.0, 2.0, 3.0, 4.5, NAN, 0.0, -1.0])
_literals = st.sampled_from([1.0, 2.0, 3.0, 4.5, NAN, 0.0, -1.0, 100.0])


@st.composite
def _frames(draw):
    size = draw(st.integers(0, 12))
    closes = [draw(_closes) for _ in range(size)]
    symbols = [draw(st.sampled_from(["A", "B", "C"])) for _ in range(size)]
    stamps = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        [draw(st.integers(0, 72)) for _ in range(size)], unit="h"
    )
    frame = pd.DataFrame(
        {
            "timestamp": pd.Series(stamps, dtype="datetime64[ns]"),
            "symbol": pd.Series(symbols, dtype="object"),
            "close": pd.Series(closes, dtype="float64"),
        }
    )
    return frame


@st.composite
def _nodes(draw):
    kind = draw(
        st.sampled_from(["cmp_close", "in_close", "cmp_ts", "in_ts", "cmp_sym"])
    )
    if kind == "cmp_close":
        return Comparison(
            "close",
            draw(st.sampled_from(["=", "<", ">", "<=", ">="])),
            draw(_literals),
            "price",
        )
    if kind == "in_close":
        return In("close", draw(st.lists(_literals, max_size=4)), "price")
    stamps = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(hours=draw(st.integers(0, 72)))
        for _ in range(draw(st.integers(1, 3)))
    ]
    if kind == "cmp_ts":
        return Comparison(
            "timestamp",
            draw(st.sampled_from(["=", "<", ">", "<=", ">="])),
            stamps[0],
            "time",
        )
    if kind == "in_ts":
        return In("timestamp", stamps, "time")
    return Comparison(
        "symbol", draw(st.sampled_from(["=", "<", ">", "<=", ">="])), "B", "symbol"
    )


@settings(max_examples=600, deadline=None, derandomize=True)
@given(frame=_frames(), nodes=st.lists(_nodes(), min_size=1, max_size=2))
def test_vectorized_matches_row_wise_including_nan(frame, nodes) -> None:
    predicate = nodes[0] if len(nodes) == 1 else And(nodes)
    vectorized = predicate.evaluate_vectorized(frame).tolist()
    row_wise = [
        bool(predicate.evaluate_against_row(row)) for row in frame.to_dict("records")
    ]
    assert vectorized == row_wise


# --------------------------------------------------------------------------
# Layer 2: engines vs each other vs an independent oracle.
# --------------------------------------------------------------------------

_ZONES = [None, "UTC", "America/New_York"]
_CLOSE_CHOICES = [1.0, 2.0, 3.0, 4.0, 5.0, NAN]
_BASE_DAY = "2024-03-08"


def _vals(close: float) -> np.ndarray:
    return np.array([close, close, close, close, 1.0], dtype=np.float64)


def _build_store(rng: random.Random, tz: str | None):
    base = pd.Timestamp(_BASE_DAY, tz=tz)
    boundary = rng.choice(
        [MonthlyBoundary(), DurationBoundary(timedelta(hours=rng.choice([6, 24])))]
    )
    store = ChunkStore(chunk_boundary=boundary)
    events: list[tuple[str, pd.Timestamp, float]] = []
    for symbol in ("A", "B"):
        offset = 0
        for _ in range(rng.randint(1, 8)):
            offset += rng.choice([0, 0, 1, 1, 2, 5, 40 * 24])  # dups, month jumps
            events.append(
                (symbol, base + pd.Timedelta(hours=offset), rng.choice(_CLOSE_CHOICES))
            )
    events.sort(key=lambda event: event[1])
    for symbol, stamp, close in events:
        store.ingest_row(stamp, symbol, _vals(close))
    store.flush_all()

    corrections: list[tuple[str, pd.Timestamp, pd.Timestamp, float]] = []
    for _ in range(rng.randint(0, 5)):
        symbol, stamp, _close = rng.choice(events)
        txn = stamp + pd.Timedelta(hours=rng.choice([0, 0, 1, 5, 100]))
        close = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0])
        store.ingest_correction(stamp, symbol, _vals(close), txn)
        corrections.append((symbol, stamp, txn, close))
    return store, events, corrections


def _oracle(events, corrections, visible):
    """Latest-wins resolution; ties: later txn, then correction, then later ingest."""

    best: dict[tuple[str, pd.Timestamp], tuple] = {}
    sequence = 0
    for symbol, stamp, close in events:
        sequence += 1
        if visible(stamp):
            rank = (stamp, 0, sequence)
            key = (symbol, stamp)
            if key not in best or rank >= best[key][0]:
                best[key] = (rank, close)
    for symbol, stamp, txn, close in corrections:
        sequence += 1
        if visible(txn):
            rank = (txn, 1, sequence)
            key = (symbol, stamp)
            if key not in best or rank >= best[key][0]:
                best[key] = (rank, close)
    return {key: value[1] for key, value in best.items()}


def _localize(text: str, tz: str | None) -> pd.Timestamp | None:
    """Localize like the engines do; ``None`` means a DST gap/fold (unbindable)."""

    try:
        return pd.Timestamp(text, tz=tz)
    except Exception:  # AmbiguousTimeError / NonExistentTimeError
        return None


def _day_forms(rng: random.Random) -> tuple[str, pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(_BASE_DAY) + pd.Timedelta(days=rng.choice([0, 1, 2, 3, 5, 20]))
    text = day.strftime("%Y%m%d") if rng.random() < 0.3 else day.strftime("%Y-%m-%d")
    return text, day, day + pd.Timedelta(days=1)


_OPS = {
    "=": lambda a, b: a == b,
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}


def _make_predicate(rng: random.Random, tz: str | None):
    """Return ``(where_sql, keep, bindable)``; unbindable means a DST gap/fold."""

    x = rng.choice([1.0, 2.0, 2.5, 3.0, 4.0, 5.0])
    y = rng.choice([1.0, 3.0, 4.5])
    kind = rng.choice(
        [
            "all",
            "cmp",
            "cmp",
            "sym_and",
            "in",
            "in",
            "ts_exact",
            "day",
            "day",
            "sym_day",
        ]
    )
    if kind == "all":
        return "", lambda c, s, t: True, True
    if kind == "cmp":
        op = rng.choice(list(_OPS))
        return f"WHERE close {op} {x}", lambda c, s, t: _OPS[op](c, x), True
    if kind == "sym_and":
        op = rng.choice([">=", "<", "="])
        return (
            f"WHERE symbol = 'A' AND close {op} {x}",
            lambda c, s, t: s == "A" and _OPS[op](c, x),
            True,
        )
    if kind == "in":
        # NaN data never matches an IN list.
        return f"WHERE close IN ({x}, {y})", lambda c, s, t: c == x or c == y, True
    if kind == "ts_exact":
        stamp = pd.Timestamp(_BASE_DAY) + pd.Timedelta(hours=rng.randint(0, 120))
        op = rng.choice(list(_OPS))
        text = f"{stamp:%Y-%m-%d %H:%M:%S}"
        literal = _localize(text, tz)
        return (
            f"WHERE close < 5 AND timestamp {op} '{text}'",
            lambda c, s, t: c < 5 and _OPS[op](t, literal),
            literal is not None,
        )

    text, day, next_day = _day_forms(rng)
    op = rng.choice(list(_OPS))
    # Only the day boundaries an operator really compares against must bind.
    start = None if op in ("<=", ">") else _localize(f"{day:%Y-%m-%d}", tz)
    end = None if op in ("<", ">=") else _localize(f"{next_day:%Y-%m-%d}", tz)
    bindable = (op in ("<=", ">") or start is not None) and (
        op in ("<", ">=") or end is not None
    )
    in_day = {
        "=": lambda t: start <= t < end,
        "<": lambda t: t < start,
        "<=": lambda t: t < end,
        ">": lambda t: t >= end,
        ">=": lambda t: t >= start,
    }[op]
    if kind == "day":
        return f"WHERE timestamp {op} '{text}'", lambda c, s, t: in_day(t), bindable
    return (
        f"WHERE symbol IN ('B', 'C') AND timestamp {op} '{text}'",
        lambda c, s, t: s in ("B", "C") and in_day(t),
        bindable,
    )


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True), right.reset_index(drop=True), check_exact=True
    )


def _normalize(rows):
    return sorted((s, t, "nan" if c != c else c) for s, t, c in rows)


def _run_engines(store: ChunkStore, sql: str):
    engines = (
        QueryExecutor(store),
        QueryExecutor(store, enable_pruning=False),
        FullScanQueryEngine(store),
    )
    outcomes = []
    for engine in engines:
        try:
            outcomes.append(engine.execute(sql).data)
        except UnsupportedQueryError as exc:
            outcomes.append(exc)
    return outcomes


@settings(max_examples=150, deadline=None, derandomize=True)
@given(seed=st.integers(0, 2**32 - 1))
def test_all_engines_agree_with_each_other_and_with_the_oracle(seed: int) -> None:
    rng = random.Random(seed)
    tz = rng.choice(_ZONES)
    store, events, corrections = _build_store(rng, tz)

    where, keep, bindable = _make_predicate(rng, tz)
    as_of_kind = rng.choice(["none", "none", "exact", "day"])
    cutoff = None
    as_of_clause = ""
    if as_of_kind == "exact":
        stamp = pd.Timestamp(_BASE_DAY) + pd.Timedelta(hours=rng.randint(0, 140))
        text = f"{stamp:%Y-%m-%d %H:%M:%S}"
        as_of_clause = f" TIMESTAMP AS OF '{text}'"
        cutoff = _localize(text, tz)
        bindable = bindable and cutoff is not None
    elif as_of_kind == "day":
        text, _day, next_day = _day_forms(rng)
        as_of_clause = f" TIMESTAMP AS OF '{text}'"
        # Date-only AS OF: everything strictly before the next day's midnight.
        cutoff = _localize(f"{next_day:%Y-%m-%d}", tz)
        bindable = bindable and _localize(f"{next_day:%Y-%m-%d} 00:00", tz) is not None

    sql = f"SELECT * FROM data{as_of_clause}" + (f" {where}" if where else "")

    if not bindable:
        for outcome in _run_engines(store, sql):
            assert isinstance(outcome, UnsupportedQueryError), (sql, tz, seed)
        return

    def visible(txn: pd.Timestamp) -> bool:
        if cutoff is None:
            return True
        return txn < cutoff if as_of_kind == "day" else txn <= cutoff

    resolved = _oracle(events, corrections, visible)
    expected = _normalize(
        (symbol, stamp, close)
        for (symbol, stamp), close in resolved.items()
        if keep(close, symbol, stamp)
    )

    def check(label: str) -> None:
        pruned, unpruned, full = _run_engines(store, sql)
        for outcome in (pruned, unpruned, full):
            assert not isinstance(outcome, Exception), (label, sql, outcome)
        _frames_equal(pruned, unpruned)
        _frames_equal(pruned, full)
        actual = _normalize(
            (row.symbol, row.timestamp, row.close) for row in pruned.itertuples()
        )
        assert actual == expected, (label, sql, tz, seed)

    check("initial")
    if corrections and rng.random() < 0.5:
        store.compact_corrections()
        check("after compaction")
