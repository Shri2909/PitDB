# Data fixtures

PitDB's default real-data benchmarks use two committed merged CSV fixtures
originally obtained through `yfinance`. Benchmarks read the local files and
never refresh provider data during a measurement.

## Fixture identities

Counts, ranges and SHA-256 hashes were read from the local CSV bytes (checked
2026-10-29). They are facts about these files, not predictions of what a
fresh download returns.

| File | Rows | Symbols | First timestamp | Last timestamp |
|---|---:|---:|---|---|
| `cache/all_symbols_daily.csv` | 5,010 | 10 | `2023-01-03` | `2024-12-30` |
| `cache/all_symbols_1m.csv` | 16,739 | 10 | `2026-07-10 09:30:00-04:00` | `2026-07-16 11:24:00-04:00` |

Symbols in both: `AAPL`, `AMZN`, `GOOG`, `JNJ`, `JPM`, `META`, `MSFT`,
`NVDA`, `TSLA`, `V`.

```text
all_symbols_daily.csv  8992e0ab010bf0453554cefea7e9ca1125cd99cc2e658525e6ca397aaba0de54
all_symbols_1m.csv     bf07d8f1140fe78f7e9e401df91e47155dbb7808705f4a16ae2047f11044a32d
```

The daily fixture drives fixed queries Q1-Q5 with monthly chunks; the minute
fixture drives intraday queries with hourly chunks. Neither file is in the
Python wheel, so run the benchmarks from a checkout.

## Known fixture quirks

- The 1-minute fixture ends mid-session: every symbol's last bar is
  `2026-07-16 11:24:00-04:00`, because the trailing download was taken while
  that session was open. The first four sessions look complete.
- AMZN has 1,673 one-minute rows, every other symbol 1,674. The missing bar
  is `2026-07-13 09:33:00-04:00`; it is a provider gap that the loader does
  not repair.
- Daily closes are float32-derived (e.g. `259.0199890136719`), so printed
  prices carry float32 noise.
- The datasets are tiny (0.2 MB of float64 for the daily set once loaded), so
  benchmark ratios on them reflect fixed per-chunk overhead more than data
  volume; see [METHODOLOGY.md](../METHODOLOGY.md).

## Loader contract

Merged CSV columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`,
`volume`. Column names are stripped and lowercased (collisions are
rejected). Timestamps must be ISO 8601 and become UTC at nanosecond
resolution; numeric columns convert to `float64` with a warning. The loader
rejects `inf` in any price/volume column, `NaN` in open/high/low/close (a
`NaN` volume is allowed), boolean columns, missing/non-string/empty/
whitespace-only symbols (before replay starts), duplicate
`(timestamp, symbol)` pairs, and decreasing timestamps within a symbol.
Other symbol text is kept exactly.

Direct `ChunkStore.ingest_row()` has its own policy: it accepts
nondecreasing duplicate keys and resolves ties by last-ingested order.
Single-symbol CSVs may omit `symbol`; `load_single_symbol()` validates them
separately. The merged loader reads one explicit file, so per-symbol
intermediates in `cache/` are never counted alongside it.

## Provenance limits

The downloader requests `auto_adjust=False, actions=True`, uses the
provider's `Close` (not `Adj Close`) and drops action columns. That describes
the request and local transformation; it does not certify as-traded prices or
the provider's adjustment and revision history.

The fixtures carry no original publication time per value. PitDB models base
transaction time as event time, so a downloaded, already-revised series
cannot show what a backtest knew at a past cutoff. Revision-aware experiments
must supply explicit correction transaction times with real source
provenance.

## Optional refresh

```bash
.venv/bin/python -m pip install '.[download]'
.venv/bin/python data/download_data.py                # reuse existing caches
.venv/bin/python data/download_data.py --force        # re-download
.venv/bin/python data/download_data.py --force --allow-partial
```

If any symbol fails, the script names it, writes nothing and exits 1;
`--allow-partial` accepts a merged file without the failed symbols (with a
warning). Writes go through a temporary file and an atomic replace, so a
failure never truncates an existing CSV. The daily request uses fixed
start/end dates; the intraday request uses a trailing window, so a refresh
changes its dates and rows. Provider output can change: keep the original
bytes and hashes when comparing with historical measurements.

Data comes from Yahoo Finance through the optional `yfinance` package. The
repository's code metadata grants no rights to redistribute third-party
market data beyond the source's terms.
