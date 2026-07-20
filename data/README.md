# Data

StreamXor uses two datasets, both sourced from Yahoo Finance via `yfinance`.

## Dataset A — Daily bars (primary)
- Symbols: AAPL, MSFT, GOOG, AMZN, TSLA, META, NVDA, JPM, V, JNJ
- Date range: 2023-01-01 to 2024-12-31
- Interval: 1 day
- Expected size: ~5,000 rows (10 symbols x ~500 trading days)
- Drives: benchmark queries Q1-Q5, P0 charts
- Chunking strategy: one chunk per calendar month per symbol
- **Price adjustment:** downloaded with `auto_adjust=False, actions=True` --
  prices are split-adjusted (a deterministic, zero-entropy transform) but
  *not* dividend-adjusted. A previous version of this fixture used
  yfinance's default adjusted mode (`auto_adjust=True`), where `close` was
  actually "Adj Close"; anyone diffing against historical benchmark results
  from before this change should account for that.

## Dataset B — 1-minute bars (secondary)
- Symbols: same 10 as Dataset A
- Date range: 2026-07-10 09:30:00-04:00 -> 2026-07-16 11:24:00-04:00
  when `download_data.py` was last run (yfinance's free tier only serves
  1-minute data for the trailing 7 days, so this is not a fixed range)
- Interval: 1 minute
- Expected size: ~390 rows/trading day/symbol
- Drives: Dataset B query benchmarks (narrow/price-threshold/broad queries)
- Chunking strategy: hourly
- Price adjustment: also re-pulled with `auto_adjust=False, actions=True`
  for consistency with Dataset A, though dividends rarely fall inside a
  trailing 5-day window so the practical effect here is expected to be small.

## Why the CSVs are committed to the repo

`data/cache/all_symbols_daily.csv` and `data/cache/all_symbols_1m.csv` are
checked directly into version control (small, a few MB total combined) rather
than `.gitignore`'d. This keeps benchmark results reproducible without
depending on `yfinance` still working, its rate limits, or Dataset B's
"trailing 5 days" window drifting between runs. `download_data.py` exists to
*regenerate* these files on demand — it is not required to reproduce the
benchmarks.

## Regenerating the data

```bash
python data/download_data.py            # skips datasets already cached
python data/download_data.py --force    # re-downloads everything
```

## Source / license

Data sourced from Yahoo Finance via the `yfinance` Python package, for
research/educational use only.
