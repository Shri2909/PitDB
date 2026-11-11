"""Tests for CSV loading and schema validation (src/replay/data_loader.py)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.replay.data_loader import (
    DATASET_COLUMNS,
    load_dataset,
    load_single_symbol,
    validate_schema,
)


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


def test_validate_schema_lossless_integer_cast_is_silent() -> None:
    frame = _merged_frame()
    frame["volume"] = frame["volume"].astype(np.int64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_schema(frame)

    assert frame["volume"].dtype == np.float64


def test_validate_schema_lossy_integer_cast_warns() -> None:
    frame = _merged_frame()
    frame["volume"] = np.array([2**53 + 1, 1, 2, 3], dtype=np.int64)

    with pytest.warns(RuntimeWarning, match="casting numeric column 'volume'"):
        validate_schema(frame)

    assert frame["volume"].dtype == np.float64


def test_validate_schema_object_dtype_cast_warns() -> None:
    frame = _merged_frame()
    frame["volume"] = pd.Series(["1", "2", "3", "4"], dtype=object)

    with pytest.warns(RuntimeWarning, match="casting numeric column 'volume'"):
        validate_schema(frame)


def test_normal_csv_load_emits_no_warnings(tmp_path: Path) -> None:
    frame = _merged_frame()
    frame["volume"] = frame["volume"].astype(np.int64)
    path = tmp_path / "ints.csv"
    frame.to_csv(path, index=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = load_dataset(path)

    assert loaded["volume"].dtype == np.float64


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


@pytest.mark.parametrize("invalid", [None, np.nan, pd.NA, "", "   ", "\t\n"])
def test_validate_schema_rejects_missing_or_blank_symbol(invalid: object) -> None:
    frame = _merged_frame()
    frame.loc[1, "symbol"] = invalid

    with pytest.raises(ValueError, match="symbol"):
        validate_schema(frame)


@pytest.mark.parametrize("invalid", [None, "", "   "])
def test_load_dataset_rejects_missing_or_blank_symbol(
    tmp_path: Path, invalid: str | None
) -> None:
    frame = _merged_frame()
    frame.loc[1, "symbol"] = invalid
    path = tmp_path / "invalid-symbol.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="symbol"):
        load_dataset(path)


def test_load_dataset_preserves_exact_nonblank_symbol(tmp_path: Path) -> None:
    frame = _merged_frame()
    frame["symbol"] = [" AAPL ", " AAPL ", "'MSFT'", "'MSFT'"]
    path = tmp_path / "exact-symbol.csv"
    frame.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert list(loaded["symbol"]) == list(frame["symbol"])


@pytest.mark.parametrize("symbol", ["NA", "123", "null", "nan", "00123"])
@pytest.mark.parametrize("column_name", ["symbol", " SYMBOL "])
def test_load_dataset_preserves_symbol_tokens_without_csv_inference(
    tmp_path: Path, symbol: str, column_name: str
) -> None:
    frame = _merged_frame().iloc[:2].copy()
    frame["symbol"] = symbol
    frame = frame.rename(columns={"symbol": column_name})
    path = tmp_path / "literal-symbol.csv"
    frame.to_csv(path, index=False)

    loaded = load_dataset(path)

    assert loaded["symbol"].tolist() == [symbol] * len(frame)


def test_symbol_converter_keeps_default_numeric_null_handling(tmp_path: Path) -> None:
    frame = _merged_frame()
    frame["symbol"] = "NA"
    frame["volume"] = "NA"
    path = tmp_path / "null-numeric.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="completely null columns.*volume"):
        load_dataset(path)


def test_validate_schema_accepts_pandas_string_symbols() -> None:
    frame = _merged_frame()
    frame["symbol"] = frame["symbol"].astype("string")

    validate_schema(frame)

    assert frame["symbol"].tolist() == ["AAPL", "AAPL", "MSFT", "MSFT"]


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
    # Swap AAPL's rows so its timestamps go backward.
    frame.loc[[0, 1], "timestamp"] = frame.loc[[1, 0], "timestamp"].to_numpy()
    path = tmp_path / "merged.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="unsorted symbols"):
        load_dataset(path)


# --- non-finite values and booleans -------------------------------------------


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_validate_schema_rejects_infinite_values(column: str, bad: float) -> None:
    frame = _merged_frame()
    frame.loc[2, column] = bad

    with pytest.raises(ValueError, match=rf"'{column}'.*non-finite.*row 2"):
        validate_schema(frame)


@pytest.mark.parametrize("column", ["open", "high", "low", "close"])
def test_validate_schema_rejects_nan_prices_naming_first_row(column: str) -> None:
    frame = _merged_frame()
    frame.loc[[1, 3], column] = np.nan

    with pytest.raises(ValueError, match=rf"price column '{column}'.*row 1"):
        validate_schema(frame)


def test_validate_schema_allows_missing_volume() -> None:
    frame = _merged_frame()
    frame.loc[1, "volume"] = np.nan

    validate_schema(frame)

    assert np.isnan(frame.loc[1, "volume"])


@pytest.mark.parametrize("dtype", ["bool", "object"])
def test_validate_schema_rejects_boolean_columns(dtype: str) -> None:
    frame = _merged_frame()
    frame["volume"] = pd.Series([True, False, True, True], dtype=dtype)

    with pytest.raises(ValueError, match="'volume'.*boolean"):
        validate_schema(frame)


def test_load_dataset_rejects_inf_and_boolean_csv_values(tmp_path: Path) -> None:
    frame = _merged_frame()
    frame.loc[1, "close"] = np.inf
    path = tmp_path / "inf.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="'close'.*non-finite.*row 1"):
        load_dataset(path)

    frame = _merged_frame()
    frame["open"] = [True, False, True, False]
    path = tmp_path / "bool.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="'open'.*boolean"):
        load_dataset(path)


def test_load_single_symbol_rejects_inf(tmp_path: Path) -> None:
    frame = _single_symbol_frame()
    frame.loc[2, "high"] = np.inf
    path = tmp_path / "inf.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="'high'.*non-finite.*row 2"):
        load_single_symbol(path)


# --- CSV shape edge cases ------------------------------------------------------


def test_header_only_csv_reports_no_data_rows(tmp_path: Path) -> None:
    path = tmp_path / "header-only.csv"
    path.write_text("timestamp,symbol,open,high,low,close,volume\n")

    with pytest.raises(ValueError, match="no data rows"):
        load_dataset(path)
    with pytest.raises(ValueError, match="no data rows"):
        load_single_symbol(path)


def test_trailing_comma_in_header_drops_unnamed_null_column(tmp_path: Path) -> None:
    path = tmp_path / "trailing-comma.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume,\n"
        "2024-01-02,AAPL,1,2,0.5,1.5,100,\n"
        "2024-01-03,AAPL,1,2,0.5,1.5,100,\n"
    )

    loaded = load_dataset(path)

    assert list(loaded.columns) == list(DATASET_COLUMNS)
    assert len(loaded) == 2


def test_trailing_comma_in_header_only_without_row_commas(tmp_path: Path) -> None:
    path = tmp_path / "trailing-header-comma.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume,\n"
        "2024-01-02,AAPL,1,2,0.5,1.5,100\n"
    )

    assert list(load_dataset(path).columns) == list(DATASET_COLUMNS)


def test_unnamed_column_with_data_is_not_silently_dropped(tmp_path: Path) -> None:
    path = tmp_path / "unnamed-data.csv"
    path.write_text(
        "timestamp,symbol,open,high,low,close,volume,\n"
        "2024-01-02,AAPL,1,2,0.5,1.5,100,7\n"
    )

    assert "unnamed: 7" in load_dataset(path).columns


def test_utf8_bom_file_loads(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(
        b"\xef\xbb\xbfsymbol,timestamp,open,high,low,close,volume\n"
        b"AAPL,2024-01-02,1,2,0.5,1.5,100\n"
    )
    assert load_dataset(path)["symbol"].tolist() == ["AAPL"]

    quoted = tmp_path / "bom-quoted.csv"
    quoted.write_bytes(
        b'\xef\xbb\xbf"timestamp","symbol","open","high","low","close","volume"\n'
        b'2024-01-02,"AAPL",1,2,0.5,1.5,100\n'
    )
    assert load_dataset(quoted)["symbol"].tolist() == ["AAPL"]


# --- load_dataset reporting ----------------------------------------------------


def test_load_dataset_truncates_long_symbol_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    count = 500
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-02"] * count,
            "symbol": [f"SYM{i:04d}" for i in range(count)],
            **{name: [1.0] * count for name in ("open", "high", "low", "close")},
            "volume": [1.0] * count,
        }
    )
    path = tmp_path / "many.csv"
    frame.to_csv(path, index=False)

    load_dataset(path)

    out = capsys.readouterr().out
    assert "symbols (500)" in out
    assert "SYM0000" in out
    assert "SYM0499" not in out
    assert "(+490 more)" in out
    assert len(out) < 400


def test_load_dataset_verbose_false_is_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "ok.csv"
    _merged_frame().to_csv(path, index=False)

    load_dataset(path, verbose=False)

    assert capsys.readouterr().out == ""
