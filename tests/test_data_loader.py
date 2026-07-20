"""Tests for CSV loading and schema validation (src/replay/data_loader.py).

This module previously had zero test coverage anywhere in the suite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.replay.data_loader import load_dataset, load_single_symbol, validate_schema


def _single_symbol_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC"),
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [1_000.0, 1_100.0, 1_200.0],
        }
    )


def _merged_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-02", periods=2, freq="D", tz="UTC"
            ).tolist()
            * 2,
            "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "open": [100.0, 101.0, 200.0, 201.0],
            "high": [101.0, 102.0, 201.0, 202.0],
            "low": [99.0, 100.0, 199.0, 200.0],
            "close": [100.5, 101.5, 200.5, 201.5],
            "volume": [1_000.0, 1_100.0, 2_000.0, 2_100.0],
        }
    )


def test_validate_schema_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        validate_schema([1, 2, 3])  # type: ignore[arg-type]


def test_validate_schema_rejects_missing_columns() -> None:
    frame = _merged_frame().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(frame)


def test_validate_schema_rejects_non_datetime_timestamp() -> None:
    frame = _merged_frame()
    frame["timestamp"] = frame["timestamp"].astype(str)
    with pytest.raises(ValueError, match="datetime64 dtype"):
        validate_schema(frame)


def test_validate_schema_coerces_numeric_columns_with_warning() -> None:
    frame = _merged_frame()
    frame["volume"] = frame["volume"].astype(np.int64)

    with pytest.warns(RuntimeWarning, match="casting numeric column 'volume'"):
        validate_schema(frame)

    assert frame["volume"].dtype == np.float64


def test_validate_schema_rejects_non_convertible_numeric_column() -> None:
    frame = _merged_frame()
    frame["close"] = ["not", "numeric", "values", "here"]
    with pytest.raises(ValueError, match="cannot be converted to float64"):
        validate_schema(frame)


def test_validate_schema_rejects_non_string_symbol() -> None:
    frame = _merged_frame()
    frame["symbol"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="symbol column must use"):
        validate_schema(frame)


def test_load_single_symbol_round_trip(tmp_path: Path) -> None:
    frame = _single_symbol_frame()
    path = tmp_path / "aapl.csv"
    frame.to_csv(path, index=False)

    loaded = load_single_symbol(path)

    assert list(loaded["close"]) == list(frame["close"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])


def test_load_single_symbol_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_single_symbol(tmp_path / "does-not-exist.csv")


def test_load_single_symbol_rejects_null_column(tmp_path: Path) -> None:
    frame = _single_symbol_frame()
    frame["volume"] = np.nan
    path = tmp_path / "aapl.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="completely null columns"):
        load_single_symbol(path)


def test_load_single_symbol_rejects_duplicate_timestamps(tmp_path: Path) -> None:
    frame = _single_symbol_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    path = tmp_path / "aapl.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate timestamps"):
        load_single_symbol(path)


def test_load_single_symbol_rejects_unsorted_timestamps(tmp_path: Path) -> None:
    frame = _single_symbol_frame().iloc[::-1].reset_index(drop=True)
    path = tmp_path / "aapl.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="sorted in ascending order"):
        load_single_symbol(path)


def test_load_dataset_round_trip(tmp_path: Path) -> None:
    frame = _merged_frame()
    path = tmp_path / "merged.csv"
    frame.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert set(loaded["symbol"].unique()) == {"AAPL", "MSFT"}
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])


def test_load_dataset_rejects_duplicate_timestamp_symbol_pairs(tmp_path: Path) -> None:
    frame = _merged_frame()
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    frame.loc[1, "symbol"] = frame.loc[0, "symbol"]
    path = tmp_path / "merged.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate \\(timestamp, symbol\\) pairs"):
        load_dataset(path)


def test_load_dataset_rejects_unsorted_timestamps_within_symbol(tmp_path: Path) -> None:
    frame = _merged_frame()
    # Swap AAPL's two rows so AAPL's own timestamps go backward, while the
    # file's overall row order (grouped by symbol) stays otherwise valid.
    frame.loc[[0, 1], "timestamp"] = frame.loc[[1, 0], "timestamp"].to_numpy()
    path = tmp_path / "merged.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="unsorted symbols"):
        load_dataset(path)
