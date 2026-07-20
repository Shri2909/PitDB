"""Tests for conservative chunk-label predicate pushdown."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.query.predicate import AlwaysTrue, And, Comparison, In
from src.query.pushdown import evaluate_pushdown
from src.store.label import ChunkLabel


def _label(
    chunk_id: int,
    *,
    symbol: str = "AAPL",
    time_start: datetime | None = None,
    time_end: datetime | None = None,
    min_price: float = 100.0,
    max_price: float = 200.0,
) -> ChunkLabel:
    start = time_start or datetime(2024, 3, 1)
    end = time_end or start + timedelta(hours=1) - timedelta(microseconds=1)
    return ChunkLabel(
        symbol=symbol,
        time_start=start,
        time_end=end,
        min_price=min_price,
        max_price=max_price,
        row_count=60,
        chunk_id=chunk_id,
    )


def _mixed_symbol_labels() -> list[ChunkLabel]:
    return [
        *[_label(index, symbol="AAPL") for index in range(5)],
        *[_label(index, symbol="MSFT") for index in range(5, 10)],
    ]


def test_symbol_pushdown_skips_non_matching() -> None:
    labels = _mixed_symbol_labels()
    predicate = Comparison("symbol", "=", "AAPL", "symbol")

    result = evaluate_pushdown(predicate, labels)

    assert result.candidate_chunk_ids == [0, 1, 2, 3, 4]
    assert result.skipped_chunk_ids == [5, 6, 7, 8, 9]
    assert result.candidate_count == 5
    assert result.total_chunks == 10
    assert result.chunks_skipped_ratio == 0.5


def test_time_range_pushdown() -> None:
    day_start = datetime(2024, 3, 1)
    labels = []
    for hour in range(24):
        start = day_start + timedelta(hours=hour)
        labels.append(
            _label(
                hour,
                time_start=start,
                time_end=start + timedelta(hours=1) - timedelta(microseconds=1),
            )
        )
    predicate = And(
        [
            Comparison(
                "timestamp",
                ">=",
                datetime(2024, 3, 1, 10),
                "time",
            ),
            Comparison(
                "timestamp",
                "<",
                datetime(2024, 3, 1, 14),
                "time",
            ),
        ]
    )

    result = evaluate_pushdown(predicate, labels)

    assert result.candidate_chunk_ids == [10, 11, 12, 13]
    assert result.candidate_count == 4
    assert result.skipped_chunk_ids == [
        *range(0, 10),
        *range(14, 24),
    ]
    assert len(result.skipped_chunk_ids) == 20
    assert result.total_chunks == 24
    assert result.chunks_skipped_ratio == 20 / 24


def test_price_pushdown() -> None:
    labels = [
        _label(0, min_price=100.0, max_price=150.0),
        _label(1, min_price=150.0, max_price=200.0),
        _label(2, min_price=190.0, max_price=200.01),
        _label(3, min_price=210.0, max_price=250.0),
    ]
    predicate = Comparison("close", ">", 200.0, "price")

    result = evaluate_pushdown(predicate, labels)

    # Strict > means max_price == 200 cannot contain a matching value.
    assert result.candidate_chunk_ids == [2, 3]
    assert result.skipped_chunk_ids == [0, 1]
    assert result.candidate_count == 2
    assert result.chunks_skipped_ratio == 0.5


def test_combined_pushdown() -> None:
    day = datetime(2024, 3, 1)
    labels = [
        _label(
            0,
            symbol="AAPL",
            time_start=day + timedelta(hours=10),
            max_price=160.0,
        ),
        _label(
            1,
            symbol="AAPL",
            time_start=day + timedelta(hours=11),
            max_price=140.0,
        ),
        _label(
            2,
            symbol="AAPL",
            time_start=day + timedelta(hours=15),
            max_price=300.0,
        ),
        _label(
            3,
            symbol="MSFT",
            time_start=day + timedelta(hours=10),
            max_price=300.0,
        ),
        _label(
            4,
            symbol="AAPL",
            time_start=day + timedelta(hours=13),
            max_price=201.0,
        ),
        _label(
            5,
            symbol="AAPL",
            time_start=day + timedelta(hours=9),
            max_price=300.0,
        ),
    ]
    predicate = And(
        [
            Comparison("symbol", "=", "AAPL", "symbol"),
            Comparison("timestamp", ">=", day + timedelta(hours=10), "time"),
            Comparison("timestamp", "<", day + timedelta(hours=14), "time"),
            Comparison("close", ">", 150.0, "price"),
        ]
    )

    result = evaluate_pushdown(predicate, labels)

    assert result.candidate_chunk_ids == [0, 4]
    assert result.skipped_chunk_ids == [1, 2, 3, 5]
    assert result.candidate_count == 2
    assert result.total_chunks == 6
    assert result.chunks_skipped_ratio == 4 / 6


def test_no_filter_selects_all() -> None:
    labels = _mixed_symbol_labels()

    result = evaluate_pushdown(AlwaysTrue(), labels)

    assert result.candidate_chunk_ids == list(range(10))
    assert result.skipped_chunk_ids == []
    assert result.candidate_count == 10
    assert result.total_chunks == 10
    assert result.chunks_skipped_ratio == 0.0


def test_impossible_filter_skips_all() -> None:
    labels = _mixed_symbol_labels()
    predicate = Comparison("symbol", "=", "NONEXISTENT", "symbol")

    result = evaluate_pushdown(predicate, labels)

    assert result.candidate_chunk_ids == []
    assert result.skipped_chunk_ids == list(range(10))
    assert result.candidate_count == 0
    assert result.total_chunks == 10
    assert result.chunks_skipped_ratio == 1.0


def test_conservative_label_evaluation() -> None:
    labels = [_label(42, min_price=145.0, max_price=155.0)]
    predicate = Comparison("close", "=", 150.0, "price")

    result = evaluate_pushdown(predicate, labels)

    # A range can prove absence but not presence. The chunk might contain 150,
    # so exact row filtering must make the final decision after decompression.
    assert result.candidate_chunk_ids == [42]
    assert result.skipped_chunk_ids == []
    assert result.candidate_count == 1
    assert result.chunks_skipped_ratio == 0.0


def test_label_pushdown_handles_aware_literal_against_naive_label() -> None:
    # Regression test: Comparison/In.evaluate_against_label used to call
    # _coerce_row_operands(label.time_start, self.value) and keep only the
    # second return value, assuming the literal is always the side that gets
    # tz-adjusted. That's backwards when the label side is naive (as here)
    # and the literal is aware -- the helper adjusted the discarded first
    # value instead, leaving an aware literal to be compared directly against
    # the label's real naive time_start/time_end, raising
    # "TypeError: can't compare offset-naive and offset-aware datetimes".
    label = _label(0)  # naive time_start=2024-03-01 00:00, time_end=00:59:59.999999

    matching = Comparison(
        "timestamp", "=", datetime(2024, 3, 1, 0, 30, tzinfo=timezone.utc), "time"
    )
    non_matching = Comparison(
        "timestamp", "=", datetime(2024, 3, 1, 5, 0, tzinfo=timezone.utc), "time"
    )
    assert matching.evaluate_against_label(label) is True
    assert non_matching.evaluate_against_label(label) is False

    in_predicate = In(
        "timestamp",
        [
            datetime(2024, 3, 1, 0, 30, tzinfo=timezone.utc),
            datetime(2024, 3, 1, 5, 0, tzinfo=timezone.utc),
        ],
        "time",
    )
    assert in_predicate.evaluate_against_label(label) is True
