from pathlib import Path

# Dataset A: daily granularity, drives Q1-Q5.
DATA_DIR = Path("data/cache")
DAILY_DATA_FILE = DATA_DIR / "all_symbols_daily.csv"
SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]

# Dataset B: intraday granularity, used only for the Dataset B query benchmarks.
INTRADAY_DATA_FILE = DATA_DIR / "all_symbols_1m.csv"
INTRADAY_CHUNK_BOUNDARY_HOURS = 1  # ~60 rows/chunk

# Multiple of 6 (3! engine orderings); time_interleaved rounds other values up.
NUM_TRIALS = 18
WARMUP_RUNS = 3  # discarded warm-up runs before timing

# Fixed queries Q1-Q5 against Dataset A. Q1's `date = <literal>` is a whole-day
# equality. Q4's price threshold must split AAPL's 2024 closes (165-259) so the
# price clause prunes chunks; re-verify against the fixture before changing it.
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

# Approximate targets; the sweep records and plots the achieved selectivity.
SELECTIVITY_SWEEP_TARGETS = [1.0, 0.5, 0.1, 0.01, 0.001]

RESULTS_DIR = Path("benchmarks/results")
