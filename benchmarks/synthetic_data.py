"""Synthetic OHLCV data generation for the scale sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_ohlcv(
    num_symbols: int,
    rows_per_symbol: int,
    *,
    seed: int = 0,
    start: str = "2023-01-03",
) -> pd.DataFrame:
    """Build a timestamp-and-symbol-sorted synthetic OHLCV DataFrame."""
    if num_symbols < 1:
        raise ValueError("num_symbols must be >= 1")
    if rows_per_symbol < 1:
        raise ValueError("rows_per_symbol must be >= 1")

    rng = np.random.default_rng(seed)
    timestamps = pd.bdate_range(start=start, periods=rows_per_symbol, tz="UTC")

    frames = []
    for symbol_index in range(num_symbols):
        symbol = f"SYN{symbol_index:06d}"
        start_price = float(rng.uniform(20.0, 500.0))
        daily_returns = rng.normal(0.0, 0.02, size=rows_per_symbol)
        close = start_price * np.cumprod(1.0 + daily_returns)
        close = np.maximum(close, 0.01)

        prev_close = np.roll(close, 1)
        prev_close[0] = start_price
        gap = rng.normal(0.0, 0.005, size=rows_per_symbol)
        open_ = prev_close * (1.0 + gap)

        intraday_spread = np.abs(rng.normal(0.0, 0.01, size=rows_per_symbol))
        high = np.maximum(open_, close) * (1.0 + intraday_spread)
        low = np.minimum(open_, close) * (1.0 - intraday_spread)

        volume = rng.lognormal(mean=15.5, sigma=0.6, size=rows_per_symbol)

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": symbol,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["timestamp", "symbol"], kind="stable").reset_index(
        drop=True
    )
    return merged


__all__ = ["generate_synthetic_ohlcv"]
