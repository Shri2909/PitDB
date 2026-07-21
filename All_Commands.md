# All Commands

Every command used to prepare the machine and run PitDB's benchmark suite,
in the order they're meant to be run. Nothing here is fabricated -- this is
the exact, minimal ("essential") tuning set settled on during benchmarking:
enough to get stable, reproducible numbers without slowing the machine down
or requiring a reboot (no `isolcpus`, no turbo/SMT disabling, no `nice`
pinning -- those were tried and explicitly walked back as unnecessary for
this workload).

Commands assume the repo root (`/home/abc/Desktop/project`) as the working
directory and the project's virtualenv at `.venv/`.

---

## 1. Essential CPU tuning (run before every benchmark session)

### 1.1 Check for competing processes

Kill or close anything CPU-heavy (browsers, IDEs with active indexing,
other terminals) before benchmarking -- background load is the single
biggest source of noisy results.

```bash
ps aux --sort=-%cpu | head -10
```

### 1.2 Set the CPU governor to performance

Prevents the CPU from clocking down between measurements. This is the one
setting that matters most for run-to-run consistency.

```bash
sudo cpupower frequency-set -g performance
```

Verify it took effect:

```bash
cpupower frequency-info | grep "current policy"
```

### 1.3 Pin thread pools to a single thread

NumPy/pandas' BLAS backends (OpenBLAS/MKL) silently multithread otherwise,
which adds scheduling noise to single-threaded benchmark measurements.

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### 1.4 Pin allocator/hash behavior

```bash
export MALLOC_ARENA_MAX=1
export PYTHONHASHSEED=0
```

These four `export` lines (1.3 + 1.4) only apply to the shell session they're
run in -- run them in the same terminal you'll launch the benchmarks from,
every time you open a new terminal.

---

## 2. Benchmark run commands

Each benchmark can be run directly with the venv's Python, or via the
matching `make` target (both are equivalent -- the Makefile targets call
`python`, resolved from whatever's first on `PATH`; the direct form pins it
to this project's venv explicitly).

### Benchmark 1 & 2 -- Fixed queries/selectivity sweep + memory footprint

```bash
.venv/bin/python benchmarks/run_all.py --memory
```

```bash
make bench
```

(`make bench` runs `run_all.py` without `--memory` -- add the flag directly
if Benchmark 2's memory-footprint section is needed.)

### Benchmark 3 -- Scale sweep

Not included in the session 2 consolidated results by explicit request
(sweeps up to 10,000 symbols / ~240,000 chunks and takes substantially
longer than the others). Command is still valid and documented here for
completeness.

```bash
.venv/bin/python benchmarks/bench_scale.py
```

```bash
make bench-scale
```

### Benchmark 4 -- Bitemporal corrections and compaction

```bash
.venv/bin/python benchmarks/bench_bitemporal.py
```

```bash
make bench-bitemporal
```

### Benchmark 5 -- Chunk-granularity trade-off

```bash
.venv/bin/python benchmarks/bench_chunk_granularity.py
```

```bash
make bench-chunk-granularity
```

### Full suite -- all five benchmarks + charts + consolidated summary

```bash
.venv/bin/python benchmarks/run_suite.py
```

```bash
make bench-all
```

---

## 3. Supporting commands

### Regenerate charts from existing results JSON (no benchmarks re-run)

```bash
.venv/bin/python charts/generate_all.py
```

```bash
make charts
```

### Run the test suite

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

```bash
make test
```

### Clear stale results before a fresh run

```bash
rm -f benchmarks/results/*.json charts/output/*.png
```

(`make clean` does this plus removing build artifacts and `__pycache__`.)
