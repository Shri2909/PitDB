from pathlib import Path

# Dataset A — primary, daily granularity, drives Q1-Q5 and the P0 charts
DATA_DIR = Path("data/cache")
DAILY_DATA_FILE = DATA_DIR / "all_symbols_daily.csv"
SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]

# Dataset B — secondary, intraday granularity. Drives the Dataset B query
# benchmarks in benchmarks/results/queries.json (run_all.py); no dedicated
# chart currently reads it.
INTRADAY_DATA_FILE = DATA_DIR / "all_symbols_1m.csv"
INTRADAY_CHUNK_BOUNDARY_HOURS = (
    1  # DurationBoundary(timedelta(hours=1)) — ~60 rows/chunk
)

NUM_TRIALS = 15  # number of repeated runs per measurement (>= 10 per brief)
WARMUP_RUNS = 3  # discarded warm-up runs before timing

# Fixed benchmark queries Q1–Q5 — run against Dataset A.
# Note: 'date = <literal>' is intentionally a whole-day equality (expands
# internally to a half-open [day, day+1) range against the timestamp column,
# in sql_parser.py). Without that expansion, Q1 would match zero rows against
# any dataset with a real time-of-day component, and would only
# coincidentally work for date-only timestamps.
# Note: Q4's price threshold (200) is chosen to actually split AAPL's 2024
# chunks on the price clause -- AAPL's 2024 close never drops below 165, so a
# threshold of 150 (a prior value here) would trivially match every chunk and
# report zero incremental pruning from the price clause, silently attributing
# 100% of Q4's chunk-skip ratio to the date clause alone. Re-verify against
# data/cache/all_symbols_daily.csv before changing this threshold.
BENCHMARK_QUERIES = {
    "Q1": "SELECT * FROM data WHERE symbol = 'AAPL' AND date = '2024-03-01'",
    "Q2": (
        "SELECT * FROM data WHERE symbol = 'AAPL' "
        "AND date >= '2024-03-01' AND date < '2024-04-01'"
    ),
    "Q3": (
        "SELECT * FROM data WHERE symbol IN ('AAPL', 'MSFT') "
        "AND date >= '2024-03-01' AND date < '2024-03-08'"
    ),
    "Q4": (
        "SELECT * FROM data WHERE symbol = 'AAPL' AND price > 200 "
        "AND date >= '2024-01-01'"
    ),
    "Q5": "SELECT * FROM data WHERE date >= '2024-01-01'",
}

# Selectivity sweep target values for Graph 1 & 3 — treated as APPROXIMATE
# targets, not guaranteed exact hits. bench_query.py's sweep records the
# actual achieved selectivity for each constructed query and plots that (not
# the target), since hitting an exact selectivity from an arbitrary
# date-range or price threshold generally requires a small search rather
# than a closed-form threshold.
SELECTIVITY_SWEEP_TARGETS = [1.0, 0.5, 0.1, 0.01, 0.001]

RESULTS_DIR = Path("benchmarks/results")
