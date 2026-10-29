"""Common three-engine measurement contract for the synthetic sweeps."""

from __future__ import annotations

from typing import Any

from benchmarks.bench_query import _benchmark_pair
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor, QueryResult
from src.store.chunk_store import ChunkStore


def measure_query_paths(
    store: ChunkStore,
    sql: str,
    *,
    num_trials: int,
    warmup: int,
    query_name: str,
    quiet: bool = True,
) -> tuple[dict[str, Any], QueryResult]:
    full, push, speedup, result = _benchmark_pair(
        FullScanQueryEngine(store),
        QueryExecutor(store),
        sql,
        query_name=query_name,
        quiet=quiet,
        num_trials=num_trials,
        warmup=warmup,
    )
    # Legacy flattened fields remain readable; complete per-engine evidence is
    # also exported. All means and intervals below refer to raw observations.
    metrics = {
        "full_scan": full,
        "pushdown_query": push,
        "vectorized_full_scan": push["vectorized_full_scan"],
        "vectorized_full_scan_speedup": push["vectorized_full_scan_speedup"],
        "acquisition_order": push["acquisition_order"],
        "timing_method": push["timing_method"],
        "query_num_trials": num_trials,
        "query_warmup": warmup,
        "query_pushdown_mean_ms": push["mean_latency_seconds"] * 1000,
        "query_pushdown_ci95_low_ms": push["ci95_low_seconds"] * 1000,
        "query_pushdown_ci95_high_ms": push["ci95_high_seconds"] * 1000,
        "query_full_scan_mean_ms": full["mean_latency_seconds"] * 1000,
        "query_full_scan_ci95_low_ms": full["ci95_low_seconds"] * 1000,
        "query_full_scan_ci95_high_ms": full["ci95_high_seconds"] * 1000,
        "query_speedup": speedup,
        "row_count": result.row_count,
    }
    return metrics, result
