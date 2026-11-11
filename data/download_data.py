"""Download Dataset A (daily, 2023-2024) and Dataset B (1-minute, trailing 5 days).

Dataset B is not a fixed date range (yfinance caps 1-minute bars); see
data/README.md.
"""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("yfinance is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]

CACHE_DIR = Path(__file__).parent / "cache"

DATASET_A_START = "2023-01-01"
DATASET_A_END = "2024-12-31"
DATASET_A_MERGED = CACHE_DIR / "all_symbols_daily.csv"

DATASET_B_PERIOD = "5d"
DATASET_B_MERGED = CACHE_DIR / "all_symbols_1m.csv"

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _manual_download_hint(symbol: str) -> str:
    return (
        f"  Could not download data for {symbol} via yfinance.\n"
        f"  Manual fallback: go to https://finance.yahoo.com/quote/{symbol}/history\n"
        f"  and use the 'Download' link to get a CSV, then place it at\n"
        f"  data/cache/{symbol}_daily.csv (or _1m.csv) with columns: "
        f"{', '.join(COLUMNS)}"
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten yfinance's output into this project's schema.

    "close" comes from the dividend-unadjusted "Close"; extra columns are dropped.
    """
    df = df.reset_index()
    # Single-ticker output still has a (field, ticker) MultiIndex; keep the field level.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df = df.rename(
        columns={
            date_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df[COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _download_one(
    symbol: str, interval: str, start=None, end=None, period=None
) -> pd.DataFrame | None:
    # auto_adjust=False drops the dividend adjustment; prices stay split-adjusted.
    try:
        if period is not None:
            raw = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                actions=True,
                progress=False,
            )
        else:
            raw = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                actions=True,
                progress=False,
            )
    except Exception as exc:
        print(f"  ERROR downloading {symbol}: {exc}")
        print(_manual_download_hint(symbol))
        return None

    if raw is None or raw.empty:
        print(f"  WARNING: no data returned for {symbol}.")
        print(_manual_download_hint(symbol))
        return None

    return _normalize(raw)


def _validate(df: pd.DataFrame, symbol: str, label: str) -> None:
    if df.empty:
        raise ValueError(f"{label}: empty DataFrame for {symbol}")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(f"{label}: timestamps not sorted for {symbol}")


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically via a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        df.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _download_dataset(
    label: str,
    merged_path: Path,
    interval: str,
    start=None,
    end=None,
    period=None,
    force: bool = False,
    allow_partial: bool = False,
) -> bool:
    """Download every symbol and write the merged CSV; return success.

    Nothing is written unless every symbol downloaded, or ``allow_partial`` is set.
    """
    if merged_path.exists() and not force:
        existing = pd.read_csv(merged_path)
        print(
            f"[{label}] Data already cached at {merged_path} "
            f"({len(existing)} rows). Skipping. Use --force to re-download."
        )
        return True

    print(f"\n[{label}] Downloading {len(SYMBOLS)} symbols (interval={interval})...")
    frames = []
    failed: list[str] = []
    for i, symbol in enumerate(SYMBOLS):
        print(f"  [{i + 1}/{len(SYMBOLS)}] {symbol} ...")
        df = _download_one(symbol, interval, start=start, end=end, period=period)
        if df is None:
            failed.append(symbol)
            continue
        _validate(df, symbol, label)
        df["symbol"] = symbol
        frames.append(df)
        if i < len(SYMBOLS) - 1:
            time.sleep(2)  # be gentle with yfinance

    if failed and not (allow_partial and frames):
        print(
            f"\n[{label}] ERROR: no files were written. Failed symbols: "
            f"{', '.join(failed)}. Fix the download or pass --allow-partial to "
            "accept a merged file that omits them.",
            file=sys.stderr,
        )
        return False
    if failed:
        print(
            f"\n[{label}] WARNING: --allow-partial; merged file OMITS "
            f"{', '.join(failed)}.",
            file=sys.stderr,
        )

    suffix = "daily" if interval == "1d" else "1m"
    for frame in frames:
        _atomic_write_csv(frame, CACHE_DIR / f"{frame['symbol'].iloc[0]}_{suffix}.csv")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    _atomic_write_csv(merged, merged_path)

    print(f"\n[{label}] Summary:")
    for symbol, group in merged.groupby("symbol"):
        print(
            f"  {symbol}: {len(group)} rows, "
            f"{group['timestamp'].min()} -> {group['timestamp'].max()}"
        )
        if interval == "1d" and len(group) < 400:
            print(
                f"    WARNING: {symbol} has fewer rows than expected (~500). "
                "Holiday gaps are normal, but check this."
            )
    print(f"  Total rows: {len(merged)}")
    print(
        f"  Saved merged file: {merged_path} "
        f"({merged_path.stat().st_size / 1024:.1f} KB)"
    )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download this project's datasets.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if cached files exist."
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Write the merged CSV even if some symbols failed to download "
            "(default: any failed symbol is an error, exit 1, nothing written)."
        ),
    )
    args = parser.parse_args(argv)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    ok_a = _download_dataset(
        label="Dataset A (daily)",
        merged_path=DATASET_A_MERGED,
        interval="1d",
        start=DATASET_A_START,
        end=DATASET_A_END,
        force=args.force,
        allow_partial=args.allow_partial,
    )

    ok_b = _download_dataset(
        label="Dataset B (1-minute)",
        merged_path=DATASET_B_MERGED,
        interval="1m",
        period=DATASET_B_PERIOD,
        force=args.force,
        allow_partial=args.allow_partial,
    )

    if not (ok_a and ok_b):
        print("\nDownload INCOMPLETE; see errors above.", file=sys.stderr)
        return 1
    print(
        "\nDone. Remember to record Dataset B's actual achieved date range "
        "in data/README.md."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
