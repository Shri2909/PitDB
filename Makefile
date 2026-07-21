.PHONY: setup download-data test bench bench-scale bench-bitemporal bench-chunk-granularity bench-all charts clean all

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

download-data:
	python data/download_data.py

test:
	pytest tests/ -v --tb=short

bench:
	python benchmarks/run_all.py

# Separate from `bench`: sweeps up to 10,000 symbols / ~240,000 chunks and
# takes substantially longer, so it's opt-in rather than part of the default
# benchmark run. Writes benchmarks/results/scale_sweep.json, consumed by
# `make charts` (Graph 9) if present.
bench-scale:
	python benchmarks/bench_scale.py

# Previously had no Makefile target at all -- had to be invoked as a bare
# `python benchmarks/bench_*.py`. Writes benchmarks/results/bitemporal_sweep.json,
# consumed by `make charts` (Graph 6) if present.
bench-bitemporal:
	python benchmarks/bench_bitemporal.py

# Previously had no Makefile target at all -- see bench-bitemporal above.
# Writes benchmarks/results/chunk_granularity_sweep.json, consumed by
# `make charts` (Graph 7) if present.
bench-chunk-granularity:
	python benchmarks/bench_chunk_granularity.py

# Runs all five benchmarks behind one entry point with one suite header and
# one truthful cross-benchmark summary -- see benchmarks/run_suite.py.
# Includes bench-scale's 1000x point and bench-bitemporal's full sweep, so
# this is substantially slower than `make bench` alone; run it when you want
# the complete picture, not for a quick check.
bench-all:
	python benchmarks/run_suite.py

charts:
	python charts/generate_all.py

clean:
	rm -rf build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f benchmarks/results/*.json charts/output/*.png

all: setup download-data test bench charts
