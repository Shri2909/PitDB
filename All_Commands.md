# PitDB commands

Run everything from the repository root. Commands call `.venv/bin/python`
explicitly so the intended interpreter is used even if another environment is
active (and to avoid stale script shebangs in a moved virtualenv).

## Install and verify

```bash
make setup      # .venv + pinned requirements.txt + editable install
make test       # pytest tests/ -v --tb=short
```

Equivalent manual steps, plus lint:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q tests
.venv/bin/python -m ruff check src tests benchmarks charts setup.py
```

PitDB needs Python 3.11 or newer. `make setup` accepts only 3.11 or 3.12
(`make setup SYSTEM_PYTHON=python3.12`) because the exact pins in
`requirements.txt` have wheels only there. Smaller installs use extras:

```bash
.venv/bin/python -m pip install .                    # core engine only
.venv/bin/python -m pip install '.[benchmarks]'      # runners and charts
.venv/bin/python -m pip install '.[dev]'             # tests and lint
.venv/bin/python -m pip install '.[download]'        # market-data downloader
```

A wheel ships only the `src` package. Benchmarks, charts and data fixtures
need a source checkout; `python -m src.pipeline --benchmark` exits 1 with a
clear error outside one.

## Ingest and query (`src.pipeline`)

| Flag | Meaning |
|---|---|
| `--data-csv PATH` | Required. Merged daily or one-minute OHLCV CSV |
| `--chunk-strategy {monthly,hourly}` | `monthly` (default) for daily bars, `hourly` for one-minute bars |
| `--query SQL` | Query to run after ingestion; repeatable; must be nonempty |
| `--benchmark` | After the queries, run `benchmarks/run_suite.py` (full suite, source checkout only) |

```bash
.venv/bin/python -m src.pipeline \
  --data-csv data/cache/all_symbols_daily.csv \
  --query "SELECT * FROM data WHERE symbol = 'AAPL' AND date = '2024-03-01'"

.venv/bin/python -m src.pipeline \
  --data-csv data/cache/all_symbols_1m.csv --chunk-strategy hourly \
  --query "SELECT * FROM data WHERE symbol = 'AAPL'"

.venv/bin/python -m src.pipeline \
  --data-csv data/cache/all_symbols_daily.csv \
  --query "SELECT * FROM data TIMESTAMP AS OF '2024-06-01T00:00:00Z' WHERE symbol = 'AAPL'"
```

Errors print one `error: ...` line on stderr and exit 1 (an empty `--query`
is an argparse error, exit 2). The summary's "Raw/stored byte ratio" is a
byte-count sanity check on uncompressed storage, not a compression result.
`--benchmark` does not turn `--data-csv` into a benchmark fixture. Query
grammar and literal rules: [ARCHITECTURE.md](ARCHITECTURE.md#query-contract).

## Benchmark environment

The published run used the tuning below. It reduces noise; it is not
required to run the code, but timings from an untuned machine should be
disclosed as such. Do the following before a session.

1. **Quiet the machine.** Close browsers, IDE indexing and other CPU-heavy
   programs (`ps aux --sort=-%cpu | head`), use AC power, and avoid running
   the test suite alongside.
2. **Performance CPU governor.** Note the current policy so you can revert:

   ```bash
   cpupower frequency-info -p                 # note the current governor
   sudo cpupower frequency-set -g performance
   cpupower frequency-info -p                 # confirm
   ```

3. **Single-threaded native libraries and fixed hashing, in the shell that
   launches the run:**

   ```bash
   export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
   export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
   export PYTHONHASHSEED=0 MALLOC_ARENA_MAX=1
   ```

4. **Pin to one isolated or idle core.** Find one:

   ```bash
   cat /sys/devices/system/cpu/isolated     # cores isolated at boot, if any
   lscpu -e                                  # core/thread layout
   ```

   Prefer an isolated core; otherwise pick an idle one (check with `top`
   then `1`, or `mpstat -P ALL 1 3`) and avoid its SMT sibling. Launch under
   `taskset -c <core>`; children inherit the pinning.

5. **Do not edit `src/`, `benchmarks/` or `charts/` files during a run.**
   The manifest refuses to publish if the source changed mid-run.

**Reference machine (example only).** The published results used an Intel
i5-1235U laptop on AC power with kernel parameters `isolcpus=4,5
nohz_full=4,5` (set in the bootloader; this needs a reboot and is optional),
the performance governor, everything pinned with `taskset -c 4` (an isolated
E-core), and the variables above. The result provenance also records turbo
on, SMT on, cpu 4 locked at 3.3 GHz, THP off, `nmi_watchdog` 0, ASLR off,
swappiness 10 and IRQ affinity excluding cpus 4-5; these extras are not
needed to reproduce the method. Exact values are in the provenance section
of [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

**Revert after the session.**

```bash
sudo cpupower frequency-set -g <previous governor>   # e.g. powersave or schedutil
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS
unset VECLIB_MAXIMUM_THREADS PYTHONHASHSEED MALLOC_ARENA_MAX
```

The variables only exist in the shell you exported them in, so closing that
terminal also reverts them. Undo any other sysctl or `/sys` change you made
by restoring the value you noted beforehand, and remove `isolcpus` and
`nohz_full` from the bootloader configuration (then regenerate it and
reboot) if you added them.

## Complete suite

```bash
taskset -c <core> .venv/bin/python benchmarks/run_suite.py   # same as: make bench-all
```

Runs all five benchmarks and the charts under one run ID: queries and
memory, the scale sweep (up to 10,000 symbols, 5,010,000 rows, 240,000
chunks), the bitemporal sweep and chunk granularity. It took about 2 h 40 min
on the reference machine (scale sweep about 90 min, bitemporal about 60 min,
the rest a few minutes each) and needs several GB of RAM at the 1000x point.
It writes `benchmarks/results/*.json` and `suite_run.log`, then
`charts/output/*.png` and `charts/output/manifest.json`.

If sources or fixtures change mid-run, the run refuses to publish results.
An interrupted or failed run is not a complete suite and must be rerun from
the start; do not report partial output.

## Individual benchmarks

| Target | Command | Writes |
|---|---|---|
| `make bench` | `benchmarks/run_all.py --memory` (queries, selectivity sweep, memory; `--memory` is optional on the script) | `queries.json`, `memory.json` |
| `make bench-scale` | `benchmarks/bench_scale.py` (about 90 min) | `scale_sweep.json` |
| `make bench-bitemporal` | `benchmarks/bench_bitemporal.py` (about 60 min) | `bitemporal_sweep.json` |
| `make bench-chunk-granularity` | `benchmarks/bench_chunk_granularity.py` | `chunk_granularity_sweep.json` |

Standalone runners form one compatible result set only if they share
`PITDB_BENCHMARK_RUN_ID` (the suite sets one automatically) and the source is
unchanged until all finish:

```bash
export PITDB_BENCHMARK_RUN_ID="$(.venv/bin/python -c 'import uuid; print(uuid.uuid4().hex)')"
.venv/bin/python benchmarks/run_all.py --memory
.venv/bin/python benchmarks/bench_scale.py
.venv/bin/python benchmarks/bench_bitemporal.py
.venv/bin/python benchmarks/bench_chunk_granularity.py
.venv/bin/python charts/generate_all.py
unset PITDB_BENCHMARK_RUN_ID
```

Use a fresh results directory rather than mixing JSON from earlier runs, and
archive old outputs first if you need them.

## Charts and the results report

```bash
make charts      # charts/generate_all.py; reads existing results, runs no benchmark
```

Chart generation fails with "missing or incomplete run manifest" if results
lack a complete manifest or come from mixed runs. It accepts `--results-dir`
and `--output-dir` (defaults `benchmarks/results/` and `charts/output/`).

Regenerate `BENCHMARK_RESULTS.md` from the result JSON. Both notes are
required free text, printed verbatim in the provenance section: the machine
load observed at start, and how the run was isolated.

```bash
.venv/bin/python benchmarks/render_results.py \
  --load-note "1-min load ~0.3, no other benchmarks running" \
  --run-note "Governor performance; taskset -c 4; thread env vars=1; PYTHONHASHSEED=0"
```

Optional flags: `--results-dir`, `--charts-dir`, `--output`.

## Optional data refresh

Committed fixtures are enough to reproduce the default benchmarks;
refreshing changes the inputs (especially the trailing intraday window).
Preserve existing fixture files and hashes first if comparability matters.

```bash
.venv/bin/python -m pip install '.[download]'
make download-data                                          # reuses existing caches
.venv/bin/python data/download_data.py --force              # re-download
.venv/bin/python data/download_data.py --force --allow-partial
```

If any symbol fails, the script exits 1 and writes nothing unless
`--allow-partial` is given (the merged file then omits the failed symbols and
a warning names them). Writes are atomic. Schema and fixture identities:
[data/README.md](data/README.md).

## Makefile targets

| Target | Runs |
|---|---|
| `make setup` | Creates `.venv` (Python 3.11 or 3.12; `SYSTEM_PYTHON=` overrides), installs `requirements.txt` and the package |
| `make download-data` | `data/download_data.py` |
| `make test` | `pytest tests/ -v --tb=short` |
| `make bench` | `benchmarks/run_all.py --memory` |
| `make bench-scale` | `benchmarks/bench_scale.py` |
| `make bench-bitemporal` | `benchmarks/bench_bitemporal.py` |
| `make bench-chunk-granularity` | `benchmarks/bench_chunk_granularity.py` |
| `make bench-all` | `benchmarks/run_suite.py` (every benchmark plus charts) |
| `make charts` | `charts/generate_all.py` only |
| `make clean` | Removes build and cache files, `benchmarks/results/*.json`, `suite_run.log`, the chart manifest and untracked chart PNGs; tracked PNGs stay |
| `make all` | `setup`, `download-data`, `test`, `bench-all` |

There is no make target for `render_results.py`; run it as shown above.
