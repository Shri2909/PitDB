"""Load and validate cached OHLCV CSV data (timestamps normalized to UTC)."""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
SINGLE_SYMBOL_COLUMNS = (TIMESTAMP_COLUMN, *NUMERIC_COLUMNS)
DATASET_COLUMNS = (TIMESTAMP_COLUMN, SYMBOL_COLUMN, *NUMERIC_COLUMNS)
_MAX_SYMBOLS_SHOWN = 10


def _read_csv(filepath: Path) -> pd.DataFrame:
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    # Read symbols as literal text so keys like "NA" or "00123" survive.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(
            (row for row in csv.reader(stream) if any(field.strip() for field in row)),
            [],
        )
    converters = {name: str for name in header if name.strip().lower() == SYMBOL_COLUMN}
    try:
        df = pd.read_csv(path, converters=converters, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty and has no columns: {path}") from exc

    # Drop all-null "Unnamed: N" columns created by a trailing header comma.
    stray = [
        column
        for column in df.columns
        if isinstance(column, str)
        and column.startswith("Unnamed:")
        and df[column].isna().all()
    ]
    if stray:
        df = df.drop(columns=stray)
    if len(df) == 0:
        raise ValueError(f"CSV file has a header but no data rows: {path}")
    return df


def _normalize_columns(df: pd.DataFrame) -> None:
    normalized = [str(column).strip().lower() for column in df.columns]
    duplicates = sorted(
        {column for column in normalized if normalized.count(column) > 1}
    )
    if duplicates:
        raise ValueError(
            "column names collide after lowercase normalization: "
            + ", ".join(duplicates)
        )
    df.columns = normalized


def _require_columns(df: pd.DataFrame, required: tuple[str, ...]) -> None:
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))


def _reject_completely_null_columns(df: pd.DataFrame) -> None:
    null_columns = [column for column in df.columns if df[column].isna().all()]
    if null_columns:
        raise ValueError(
            "completely null columns are not allowed: " + ", ".join(null_columns)
        )


def _parse_timestamp_utc(df: pd.DataFrame) -> None:
    try:
        # ISO8601, not "mixed": "mixed" can silently swap day/month per row.
        parsed = pd.to_datetime(
            df[TIMESTAMP_COLUMN], errors="raise", utc=True, format="ISO8601"
        )
        # Pin nanosecond resolution regardless of pandas' input-dependent default.
        parsed = parsed.astype("datetime64[ns, UTC]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "timestamp column contains values that cannot be parsed as UTC "
            "datetime64[ns]"
        ) from exc

    if parsed.isna().any():
        count = int(parsed.isna().sum())
        raise ValueError(f"timestamp column contains {count} null timestamp(s)")
    df[TIMESTAMP_COLUMN] = parsed


# Largest integer magnitude float64 represents exactly.
_FLOAT64_EXACT_INT = 2**53
# NaN is tolerated only in volume (a missing volume is common in real feeds);
# a missing price is rejected because every price column feeds chunk labels.
_PRICE_COLUMNS = ("open", "high", "low", "close")


def _reject_boolean_column(series: pd.Series, column: str) -> None:
    is_bool = pd.api.types.is_bool_dtype(series.dtype)
    if not is_bool and series.dtype == object:
        is_bool = bool(series.map(lambda v: isinstance(v, (bool, np.bool_))).any())
    if is_bool:
        raise ValueError(
            f"numeric column '{column}' contains boolean values; "
            "booleans are not accepted as numbers"
        )


def _reject_non_finite(values: pd.Series, column: str) -> None:
    array = values.to_numpy(dtype=np.float64, na_value=np.nan)
    infinite = np.isinf(array)
    if infinite.any():
        row = int(np.argmax(infinite))
        raise ValueError(
            f"numeric column '{column}' contains non-finite values "
            f"(+/-inf); first at data row {row}: {array[row]}"
        )
    if column in _PRICE_COLUMNS:
        missing = np.isnan(array)
        if missing.any():
            row = int(np.argmax(missing))
            raise ValueError(
                f"price column '{column}' contains missing/NaN values; "
                f"first at data row {row} ({int(missing.sum())} total)"
            )


def _cast_is_lossy(series: pd.Series) -> bool:
    """Return whether casting a non-float64 column to float64 can lose data."""
    dtype = series.dtype
    if pd.api.types.is_float_dtype(dtype):
        # float16/float32 widen exactly to float64.
        return False
    if pd.api.types.is_integer_dtype(dtype):
        non_null = series.dropna()
        if non_null.empty:
            return False
        magnitude = non_null.abs().max()
        return bool(magnitude > _FLOAT64_EXACT_INT)
    # Text/object/other inputs are an unexpected upstream dtype drift.
    return True


def _coerce_numeric_columns(df: pd.DataFrame) -> None:
    for column in NUMERIC_COLUMNS:
        series = df[column]
        _reject_boolean_column(series, column)

        if series.dtype != np.dtype("float64"):
            original_dtype = series.dtype
            lossy = _cast_is_lossy(series)
            try:
                converted = pd.to_numeric(series, errors="raise").astype(np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"numeric column '{column}' cannot be converted to float64"
                ) from exc
            if lossy:
                warnings.warn(
                    f"casting numeric column '{column}' from {original_dtype} "
                    "to float64 may lose precision or change type",
                    RuntimeWarning,
                    stacklevel=3,
                )
            df[column] = converted
            series = converted

        _reject_non_finite(series, column)


def _validate_symbol_column(df: pd.DataFrame) -> None:
    """Reject unusable keys without changing valid symbol strings."""
    symbol = df[SYMBOL_COLUMN]
    if symbol.isna().any():
        count = int(symbol.isna().sum())
        raise ValueError(f"symbol column contains {count} null symbol(s)")

    if not (
        pd.api.types.is_object_dtype(symbol.dtype)
        or pd.api.types.is_string_dtype(symbol.dtype)
    ):
        raise ValueError(
            "symbol column must use a pandas string or object dtype; "
            f"found {symbol.dtype}"
        )

    invalid_mask = ~symbol.map(lambda value: isinstance(value, str))
    if invalid_mask.any():
        example = symbol[invalid_mask].iloc[0]
        raise ValueError("symbol column contains a non-string value: " + repr(example))
    if symbol.map(lambda value: not value.strip()).any():
        raise ValueError("symbol column contains an empty or whitespace-only symbol")


def validate_schema(df: pd.DataFrame) -> None:
    """Validate the merged OHLCV schema and cast numeric columns to float64 in place.

    Lossy casts warn; booleans, infinities and NaN prices raise ``ValueError``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate_schema expects a pandas DataFrame")

    _require_columns(df, DATASET_COLUMNS)
    if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COLUMN].dtype):
        raise ValueError(
            "timestamp column must have a datetime64 dtype; "
            f"found {df[TIMESTAMP_COLUMN].dtype}"
        )

    _validate_symbol_column(df)
    _coerce_numeric_columns(df)


def load_single_symbol(filepath: Path) -> pd.DataFrame:
    """Load one symbol CSV into a clean, UTC-normalized DataFrame."""
    df = _read_csv(filepath)
    _normalize_columns(df)
    _require_columns(df, SINGLE_SYMBOL_COLUMNS)
    _reject_completely_null_columns(df)
    _parse_timestamp_utc(df)
    _coerce_numeric_columns(df)

    duplicate_mask = df.duplicated(subset=[TIMESTAMP_COLUMN], keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        example = df.loc[duplicate_mask, TIMESTAMP_COLUMN].iloc[0]
        raise ValueError(
            f"found {count} rows with duplicate timestamps; example: {example}"
        )
    if not df[TIMESTAMP_COLUMN].is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order")

    return df


def load_dataset(merged_csv_path: Path, *, verbose: bool = True) -> pd.DataFrame:
    """Load one merged dataset CSV for ``ReplayHarness``."""
    df = _read_csv(merged_csv_path)
    _normalize_columns(df)
    _require_columns(df, DATASET_COLUMNS)
    _reject_completely_null_columns(df)
    _parse_timestamp_utc(df)
    validate_schema(df)

    duplicate_mask = df.duplicated(subset=[TIMESTAMP_COLUMN, SYMBOL_COLUMN], keep=False)
    if duplicate_mask.any():
        count = int(duplicate_mask.sum())
        example = df.loc[duplicate_mask, [TIMESTAMP_COLUMN, SYMBOL_COLUMN]].iloc[0]
        raise ValueError(
            "found "
            f"{count} rows with duplicate (timestamp, symbol) pairs; "
            f"example: ({example[TIMESTAMP_COLUMN]}, {example[SYMBOL_COLUMN]!r})"
        )

    unsorted_symbols: list[str] = []
    for symbol, group in df.groupby(SYMBOL_COLUMN, sort=False, dropna=False):
        if not group[TIMESTAMP_COLUMN].is_monotonic_increasing:
            unsorted_symbols.append(repr(symbol))
    if unsorted_symbols:
        raise ValueError(
            "timestamps must be sorted ascending within each symbol; "
            "unsorted symbols: " + ", ".join(unsorted_symbols)
        )

    symbols = sorted(df[SYMBOL_COLUMN].dropna().unique().tolist())
    date_start = df[TIMESTAMP_COLUMN].min().isoformat()
    date_end = df[TIMESTAMP_COLUMN].max().isoformat()
    if verbose:
        shown = ", ".join(symbols[:_MAX_SYMBOLS_SHOWN])
        if len(symbols) > _MAX_SYMBOLS_SHOWN:
            shown += f", ... (+{len(symbols) - _MAX_SYMBOLS_SHOWN:,} more)"
        print(
            f"Loaded {len(df):,} rows | "
            f"symbols ({len(symbols):,}): {shown} | "
            f"date range: {date_start} to {date_end}"
        )
    return df


__all__ = ["load_single_symbol", "load_dataset", "validate_schema"]
