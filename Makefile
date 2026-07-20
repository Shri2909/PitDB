.PHONY: setup download-data test bench bench-scale charts clean all

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

charts:
	python charts/generate_all.py

clean:
	rm -rf build *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f benchmarks/results/*.json charts/output/*.png

all: setup download-data test bench charts
