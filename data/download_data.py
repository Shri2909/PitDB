"""
One-shot download script for the project's two datasets.

Dataset A (daily, 2023-01-01 -> 2024-12-31): drives Q1-Q5 benchmark queries
    and the P0 charts. Chunked monthly in later phases.
Dataset B (1-minute, trailing 5 trading days): drives the Dataset B query
    benchmarks. Chunked hourly in later phases. 1-minute bars are capped by
    yfinance's free tier to the trailing 7 days, so this window is NOT a
    fixed date range -- see data/README.md for why the CSVs are committed
    to the repo instead of relying on re-download.
"""

import argparse
import sys
import time
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


def _normalize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten yfinance's output into this project's expected schema.

    With a single ticker, yfinance's columns are still a (field, ticker)
    MultiIndex; keeping that shape past this point produces a corrupted CSV
    (the ticker level gets written as a bogus extra header/data row). Level 0
    (the field name) is dropped down to a flat Index before anything else.

    Requested with auto_adjust=False, actions=True (see _download_one): the
    raw frame therefore also carries "Adj Close", "Dividends", and
    "Stock Splits" columns alongside "Close". "close" is deliberately taken
    from "Close" (dividend-unadjusted), not "Adj Close" -- this is the whole
    point of the auto_adjust=False change. "Adj Close"/"Dividends"/
    "Stock Splits" are dropped rather than added to the merged schema.
    """
    df = df.reset_index()
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
    # auto_adjust=False, actions=True: fetch dividend-unadjusted OHLC (plus
    # Dividends/Stock Splits history, dropped in _normalize) instead of
    # yfinance's default auto-adjusted-for-dividends prices. Note this does
    # NOT produce fully "raw as-traded" prices -- Yahoo's OHLC is still
    # split-adjusted at the source regardless of this flag; auto_adjust=False
    # only removes the dividend adjustment, not the split adjustment.
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

    return _normalize(raw, symbol)


def _validate(df: pd.DataFrame, symbol: str, label: str) -> None:
    assert not df.empty, f"{label}: empty DataFrame for {symbol}"
    ts = df["timestamp"]
    assert ts.is_monotonic_increasing, f"{label}: timestamps not sorted for {symbol}"


def _download_dataset(
    label: str,
    merged_path: Path,
    interval: str,
    start=None,
    end=None,
    period=None,
    force: bool = False,
) -> None:
    if merged_path.exists() and not force:
        existing = pd.read_csv(merged_path)
        print(
            f"[{label}] Data already cached at {merged_path} "
            f"({len(existing)} rows). Skipping. Use --force to re-download."
        )
        return

    print(f"\n[{label}] Downloading {len(SYMBOLS)} symbols (interval={interval})...")
    frames = []
    for i, symbol in enumerate(SYMBOLS):
        print(f"  [{i + 1}/{len(SYMBOLS)}] {symbol} ...")
        df = _download_one(symbol, interval, start=start, end=end, period=period)
        if df is None:
            continue
        _validate(df, symbol, label)
        df["symbol"] = symbol
        per_symbol_path = (
            CACHE_DIR / f"{symbol}_{'daily' if interval == '1d' else '1m'}.csv"
        )
        df.to_csv(per_symbol_path, index=False)
        frames.append(df)
        if i < len(SYMBOLS) - 1:
            time.sleep(2)  # be gentle with yfinance

    if not frames:
        print(f"[{label}] No data downloaded for any symbol. Aborting merge.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    merged.to_csv(merged_path, index=False)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download this project's datasets.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if cached files exist."
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    _download_dataset(
        label="Dataset A (daily)",
        merged_path=DATASET_A_MERGED,
        interval="1d",
        start=DATASET_A_START,
        end=DATASET_A_END,
        force=args.force,
    )

    _download_dataset(
        label="Dataset B (1-minute)",
        merged_path=DATASET_B_MERGED,
        interval="1m",
        period=DATASET_B_PERIOD,
        force=args.force,
    )

    print(
        "\nDone. Remember to record Dataset B's actual achieved date range "
        "in data/README.md."
    )


if __name__ == "__main__":
    main()
