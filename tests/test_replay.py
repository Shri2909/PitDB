"""Tests for strict, no-lookahead DataFrame replay."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.replay.replay_harness import ReplayHarness

VALUE_COLUMNS = ["open", "high", "low", "close", "volume"]


@pytest.fixture
def replay_frame() -> pd.DataFrame:
    index = np.arange(10, dtype=np.float64)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-02 09:30:00", periods=10, freq="min", tz="UTC"
            ),
            "symbol": np.where(np.arange(10) % 2 == 0, "AAPL", "MSFT"),
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.5 + index,
            "volume": 1_000.0 + index * 10.0,
        }
    )


def test_sequential_access_only(replay_frame: pd.DataFrame) -> None:
    harness = ReplayHarness(replay_frame)

    assert harness._index == 0
    for expected_index in range(len(replay_frame)):
        harness.next_row()
        assert harness._index == expected_index + 1

    # Internal state uses underscore-prefixed names only. Consumers receive no
    # public DataFrame, length, row-count, random-access, or lookahead surface.
    assert not {name for name in vars(harness) if not name.startswith("_")}
    for forbidden in (
        "df",
        "dataframe",
        "length",
        "row_count",
        "remaining_count",
        "peek",
        "__getitem__",
        "__len__",
    ):
        assert not hasattr(harness, forbidden)

    with pytest.raises(TypeError):
        len(harness)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        harness[0]  # type: ignore[index]


def test_replay_produces_correct_data(replay_frame: pd.DataFrame) -> None:
    rows = list(ReplayHarness(replay_frame))

    assert len(rows) == len(replay_frame)
    for position, replayed in enumerate(rows):
        original = replay_frame.iloc[position]
        assert replayed["timestamp"] == original["timestamp"]
        assert replayed["symbol"] == original["symbol"]
        np.testing.assert_array_equal(
            replayed["values"],
            original.loc[VALUE_COLUMNS].to_numpy(dtype=np.float64),
        )


def test_replay_reset(replay_frame: pd.DataFrame) -> None:
    harness = ReplayHarness(replay_frame)
    midpoint = len(replay_frame) // 2

    first_half = [next(harness) for _ in range(midpoint)]
    assert len(first_half) == midpoint
    assert harness._index == midpoint

    harness.reset()
    assert harness._index == 0
    second_pass = list(harness)

    assert len(second_pass) == len(replay_frame)
    for position, replayed in enumerate(second_pass):
        original = replay_frame.iloc[position]
        assert replayed["timestamp"] == original["timestamp"]
        assert replayed["symbol"] == original["symbol"]
        np.testing.assert_array_equal(
            replayed["values"],
            original.loc[VALUE_COLUMNS].to_numpy(dtype=np.float64),
        )


def test_replay_preserves_types(replay_frame: pd.DataFrame) -> None:
    row = ReplayHarness(replay_frame).next_row()

    assert isinstance(row["timestamp"], datetime)
    assert isinstance(row["symbol"], str)
    assert isinstance(row["values"], np.ndarray)
    assert row["values"].dtype == np.float64
    assert row["values"].ndim == 1


def test_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ReplayHarness([{"timestamp": 1, "symbol": "AAPL"}])  # type: ignore[arg-type]


def test_rejects_duplicate_columns(replay_frame: pd.DataFrame) -> None:
    duplicated = replay_frame.copy()
    duplicated.columns = ["timestamp", "symbol", "open", "high", "low", "close", "open"]

    with pytest.raises(ValueError, match="duplicate column names"):
        ReplayHarness(duplicated)


def test_rejects_missing_required_column(replay_frame: pd.DataFrame) -> None:
    missing_symbol = replay_frame.drop(columns=["symbol"])

    with pytest.raises(ValueError, match="missing replay columns"):
        ReplayHarness(missing_symbol)


def test_rejects_duplicate_value_columns(replay_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="value_columns must not contain duplicates"):
        ReplayHarness(replay_frame, value_columns=["close", "close"])


def test_next_row_rejects_non_string_symbol(replay_frame: pd.DataFrame) -> None:
    bad_frame = replay_frame.copy()
    bad_frame.loc[0, "symbol"] = 123

    with pytest.raises(ValueError, match="symbol must be a string"):
        ReplayHarness(bad_frame).next_row()


def test_next_row_rejects_non_numeric_values(replay_frame: pd.DataFrame) -> None:
    bad_frame = replay_frame.copy()
    bad_frame["close"] = bad_frame["close"].astype(object)
    bad_frame.loc[0, "close"] = "not-a-number"

    with pytest.raises(ValueError, match="cannot be converted to float64"):
        ReplayHarness(bad_frame).next_row()
