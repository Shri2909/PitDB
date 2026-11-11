# Interpreter used to run everything after `make setup`.
PYTHON ?= .venv/bin/python
# Interpreter used by `make setup` to create .venv, e.g.
# `make setup SYSTEM_PYTHON=python3.12`. Must be 3.11 or 3.12: 3.11 is the
# supported floor (setup.py python_requires) and the exact pins in
# requirements.txt only have wheels up to 3.12.
SYSTEM_PYTHON ?= python3
PYTHON_FLOOR := 3.11
PYTHON_PINNED_MAX := 3.12

.PHONY: setup download-data test bench bench-scale bench-bitemporal bench-chunk-granularity bench-all charts clean all

setup:
	@$(SYSTEM_PYTHON) -c 'import sys; lo = tuple(map(int, "$(PYTHON_FLOOR)".split("."))); hi = tuple(map(int, "$(PYTHON_PINNED_MAX)".split("."))); v = sys.version_info[:2]; ok = lo <= v <= hi; ok or sys.exit("error: make setup needs Python $(PYTHON_FLOOR)-$(PYTHON_PINNED_MAX) (found %d.%d from $(SYSTEM_PYTHON)); rerun with SYSTEM_PYTHON=python3.12 or similar" % v)'
	$(SYSTEM_PYTHON) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

download-data:
	$(PYTHON) data/download_data.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

# Query + memory benchmarks: writes queries.json and memory.json.
bench:
	$(PYTHON) benchmarks/run_all.py --memory

# Sweeps up to 10,000 symbols / ~240,000 chunks (about 90 min); not part of
# `bench`. Writes benchmarks/results/scale_sweep.json.
bench-scale:
	$(PYTHON) benchmarks/bench_scale.py

# Writes benchmarks/results/bitemporal_sweep.json.
bench-bitemporal:
	$(PYTHON) benchmarks/bench_bitemporal.py

# Writes benchmarks/results/chunk_granularity_sweep.json.
bench-chunk-granularity:
	$(PYTHON) benchmarks/bench_chunk_granularity.py

# All five benchmarks plus charts under one run ID (about 2 h 40 min on the
# reference machine) -- see benchmarks/run_suite.py.
bench-all:
	$(PYTHON) benchmarks/run_suite.py

# Regenerates charts from existing results; runs no benchmark.
charts:
	$(PYTHON) charts/generate_all.py

# Removes generated artifacts; chart PNGs tracked in git are kept.
clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find src tests benchmarks charts data -type d -name "__pycache__" -exec rm -rf {} +
	find src tests benchmarks charts data -type f -name "*.pyc" -delete
	rm -f benchmarks/results/*.json benchmarks/results/suite_run.log charts/output/manifest.json
	@for f in charts/output/*.png; do \
		[ -e "$$f" ] || continue; \
		git ls-files --error-unmatch "$$f" >/dev/null 2>&1 || rm -f "$$f"; \
	done

# `bench-all` already generates the charts.
all: setup download-data test bench-all
