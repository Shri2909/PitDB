"""Load and validate cached OHLCV CSV data for sequential replay.

Timestamps are normalized to timezone-aware ``datetime64[ns, UTC]``. Choosing
UTC gives daily and intraday inputs one unambiguous timeline before they reach
the replay and chunking layers.

The merged loader accepts one explicit file rather than scanning a directory.
This prevents per-symbol intermediates from being counted alongside merged
files and prevents daily and one-minute datasets from being mixed accidentally.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_COLUMN = "timestamp"
SYMBOL_COLUMN = "symbol"
NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
SINGLE_SYMBOL_COLUMNS = (TIMESTAMP_COLUMN, *NUMERIC_COLUMNS)
DATASET_COLUMNS = (TIMESTAMP_COLUMN, SYMBOL_COLUMN, *NUMERIC_COLUMNS)


def _read_csv(filepath: Path) -> pd.DataFrame:
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty and has no columns: {path}") from exc


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
        # format="ISO8601" (not "mixed"): "mixed" infers each row's date
        # format independently, which silently swaps day/month on ambiguous
        # non-ISO input (e.g. mixed DD/MM data can parse as inconsistent
        # months row to row). The actual data source (yfinance via
        # data/download_data.py) always produces ISO 8601 timestamps, so
        # requiring that format removes the ambiguity instead of guessing.
        parsed = pd.to_datetime(
            df[TIMESTAMP_COLUMN], errors="raise", utc=True, format="ISO8601"
        )
        # Make the resolution part of the public loader contract rather than
        # depending on pandas' input-dependent datetime resolution.
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


def _coerce_numeric_columns(df: pd.DataFrame) -> None:
    for column in NUMERIC_COLUMNS:
        if df[column].dtype == np.dtype("float64"):
            continue

        original_dtype = df[column].dtype
        try:
            converted = pd.to_numeric(df[column], errors="raise").astype(np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"numeric column '{column}' cannot be converted to float64"
            ) from exc

        warnings.warn(
            f"casting numeric column '{column}' from {original_dtype} to float64",
            RuntimeWarning,
            stacklevel=3,
        )
        df[column] = converted


def _validate_symbol_column(df: pd.DataFrame) -> None:
    symbol = df[SYMBOL_COLUMN]
    if not (
        pd.api.types.is_object_dtype(symbol.dtype)
        or pd.api.types.is_string_dtype(symbol.dtype)
    ):
        raise ValueError(
            "symbol column must use a pandas string or object dtype; "
            f"found {symbol.dtype}"
        )

    non_null = symbol.dropna()
    invalid_mask = ~non_null.map(lambda value: isinstance(value, str))
    if invalid_mask.any():
        example = non_null[invalid_mask].iloc[0]
        raise ValueError("symbol column contains a non-string value: " + repr(example))


def validate_schema(df: pd.DataFrame) -> None:
    """Validate and normalize the merged OHLCV schema in place.

    Required numeric columns are cast to ``float64`` when conversion is safe;
    every such cast emits ``RuntimeWarning`` so upstream dtype drift is visible.
    All structural or non-convertible schema problems raise ``ValueError``.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("validate_schema expects a pandas DataFrame")

    _require_columns(df, DATASET_COLUMNS)
    if not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COLUMN].dtype):
        raise ValueError(
            "timestamp column must have a datetime64 dtype; "
            f"found {df[TIMESTAMP_COLUMN].dtype}"
        )

    _coerce_numeric_columns(df)
    _validate_symbol_column(df)


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


def load_dataset(merged_csv_path: Path) -> pd.DataFrame:
    """Load one named merged dataset for direct use by ``ReplayHarness``.

    No directory globbing is performed: one daily or one intraday merged file
    is loaded at a time, preserving its intended granularity and chunking model.
    """

    df = _read_csv(merged_csv_path)
    _normalize_columns(df)
    _require_columns(df, DATASET_COLUMNS)
    _reject_completely_null_columns(df)
    _parse_timestamp_utc(df)

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

    validate_schema(df)

    symbols = sorted(df[SYMBOL_COLUMN].dropna().unique().tolist())
    date_start = df[TIMESTAMP_COLUMN].min().isoformat()
    date_end = df[TIMESTAMP_COLUMN].max().isoformat()
    print(
        f"Loaded {len(df):,} rows | "
        f"symbols ({len(symbols)}): {', '.join(symbols)} | "
        f"date range: {date_start} to {date_end}"
    )
    return df


__all__ = ["load_single_symbol", "load_dataset", "validate_schema"]
