"""Memory-footprint benchmarks for raw data, chunk storage, and query paths."""

from __future__ import annotations

import gc
import statistics
import sys
import tracemalloc
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from itertools import permutations
from threading import Event, Thread
from typing import Any

import numpy as np
import pandas as pd

from benchmarks.config import BENCHMARK_QUERIES
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.store.chunk_store import ChunkStore

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised without optional dep
    psutil = None  # type: ignore[assignment]


_RSS_SAMPLE_INTERVAL_SECONDS = 0.001


def _require_psutil() -> Any:
    if psutil is None:
        raise RuntimeError(
            "psutil is required for RSS measurement; install the project's "
            "pinned dependencies before running memory benchmarks"
        )
    return psutil


def _measure_peak_rss(fn: Callable[[], Any]) -> tuple[int, int]:
    """Return ``(baseline_rss, peak_rss)`` while retaining ``fn``'s result."""
    if not callable(fn):
        raise TypeError("fn must be callable")

    psutil_module = _require_psutil()
    gc.collect()
    process = psutil_module.Process()
    baseline = int(process.memory_info().rss)
    peak = [baseline]
    stop = Event()
    ready = Event()
    sampler_error: list[BaseException] = []

    def sample_once() -> None:
        rss = int(process.memory_info().rss)
        if rss > peak[0]:
            peak[0] = rss

    def sample_until_stopped() -> None:
        try:
            sample_once()
            ready.set()
            while not stop.wait(_RSS_SAMPLE_INTERVAL_SECONDS):
                sample_once()
        except BaseException as exc:  # propagate sampler failures to caller
            sampler_error.append(exc)
            ready.set()

    sampler = Thread(
        target=sample_until_stopped,
        name="pitdb-rss-sampler",
        daemon=True,
    )
    sampler.start()
    ready.wait()

    result: Any = None
    function_failed = False
    try:
        # Keep the result alive until the final sample.
        result = fn()
        sample_once()
    except BaseException:
        function_failed = True
        raise
    finally:
        stop.set()
        sampler.join()
        if not function_failed and not sampler_error:
            sample_once()
        del result

    if sampler_error:
        raise RuntimeError("RSS sampler failed") from sampler_error[0]
    return baseline, peak[0]


def measure_peak_memory(fn: Callable[[], Any]) -> int:
    """Run ``fn`` and return the highest observed absolute process RSS in bytes."""

    _, peak = _measure_peak_rss(fn)
    return peak


def measure_traced_peak(fn: Callable[[], Any]) -> int:
    """Return the tracemalloc peak (bytes) allocated by ``fn`` above its start."""
    if not callable(fn):
        raise TypeError("fn must be callable")
    gc.collect()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        start, _ = tracemalloc.get_traced_memory()
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
        del result
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return max(0, int(peak - start))


def measure_traced_peaks_interleaved(
    calls: Mapping[str, Callable[[], Any]], repeats: int
) -> dict[str, dict[str, Any]]:
    """Repeat ``measure_traced_peak`` per engine in rotating order.

    ``repeats`` is rounded up to a multiple of the permutation count.
    """
    if not calls:
        raise ValueError("at least one callable is required")
    names = tuple(calls)
    orders = tuple(permutations(names))
    cycle = len(orders)
    repeats = -(-max(1, int(repeats)) // cycle) * cycle
    for name in names:
        calls[name]()
    samples: dict[str, list[int]] = {name: [] for name in names}
    for repeat in range(repeats):
        for name in orders[repeat % cycle]:
            samples[name].append(measure_traced_peak(calls[name]))
    return {
        name: {
            "peak_bytes_samples": values,
            "repeats": len(values),
            "median_peak_bytes": int(statistics.median(values)),
            "min_peak_bytes": min(values),
            "max_peak_bytes": max(values),
        }
        for name, values in samples.items()
    }


def _raw_size_bytes(raw_data: pd.DataFrame | np.ndarray) -> int:
    if isinstance(raw_data, pd.DataFrame):
        return int(raw_data.memory_usage(index=True, deep=True).sum())
    if isinstance(raw_data, np.ndarray):
        return int(raw_data.nbytes)
    raise TypeError("raw_data must be a pandas DataFrame or NumPy array")


def _deep_size_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate the retained Python object graph without double-counting."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)

    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        return size + sum(
            _deep_size_bytes(key, seen) + _deep_size_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_size_bytes(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(
            _deep_size_bytes(getattr(value, field.name), seen)
            for field in fields(value)
        )
    if hasattr(value, "__dict__"):
        return size + _deep_size_bytes(vars(value), seen)
    return size


# Zone-map label fields vs. plain chunk-identity/bookkeeping metadata.
_ZONE_MAP_LABEL_FIELDS = ("min_price", "max_price", "time_start", "time_end")
_METADATA_LABEL_FIELDS = (
    "symbol",
    "row_count",
    "chunk_id",
    "transaction_time_start",
    "transaction_time_end",
)


def _memory_breakdown(store: ChunkStore) -> dict[str, int]:
    """Split the store's deep size into disjoint components summing to the total."""
    seen: set[int] = set()

    packed_bytes = sum(
        _deep_size_bytes(chunk.column_data, seen) for chunk in store.chunks
    )

    metadata_bytes = 0
    zone_map_bytes = 0
    for chunk in store.chunks:
        label = chunk.label
        if id(label) not in seen:
            seen.add(id(label))
            # The instance shell counts as bookkeeping overhead.
            metadata_bytes += sys.getsizeof(label)
        metadata_bytes += sum(
            _deep_size_bytes(getattr(label, name), seen)
            for name in _METADATA_LABEL_FIELDS
        )
        zone_map_bytes += sum(
            _deep_size_bytes(getattr(label, name), seen)
            for name in _ZONE_MAP_LABEL_FIELDS
        )

    total_bytes = _deep_size_bytes(store)  # fresh walk: the ground truth

    # Private ChunkStore containers (no public accessor); the shared `seen` set
    # keeps already-counted labels from being counted twice.
    index_containers = [
        getattr(store, "_chunks", []),
        getattr(store, "_chunk_by_id", {}),
        getattr(store, "_as_of_keys", []),
        getattr(store, "_as_of_labels", []),
        getattr(store, "_builders", {}),
    ]
    labels_and_indexes_bytes = sum(
        _deep_size_bytes(container, seen) for container in index_containers
    )

    accounted_bytes = (
        packed_bytes + metadata_bytes + zone_map_bytes + labels_and_indexes_bytes
    )
    python_object_overhead_bytes = max(0, total_bytes - accounted_bytes)

    return {
        "packed_numeric_buffer_bytes": packed_bytes,
        "chunk_metadata_bytes": metadata_bytes,
        "zone_map_bytes": zone_map_bytes,
        "labels_and_indexes_bytes": labels_and_indexes_bytes,
        "python_object_overhead_bytes": python_object_overhead_bytes,
        "total_object_graph_bytes": total_bytes,
    }


_TRACED_REPEATS = 18  # multiple of 3! so the 3-engine rotation is balanced
_RSS_RESOLUTION_BYTES = 4096


def _rss_diagnostic(
    store: ChunkStore,
    q1: str,
    build_store_fn: Callable[[], ChunkStore] | None,
) -> dict[str, Any]:
    """Single-shot process-RSS observations; a noise-bounded diagnostic only."""

    idle_rss, _ = _measure_peak_rss(lambda: None)
    out: dict[str, Any] = {
        "status": "noise-bounded diagnostic; not a comparison between engines",
        "resolution_bytes": _RSS_RESOLUTION_BYTES,
        "sampling_interval_microseconds": int(_RSS_SAMPLE_INTERVAL_SECONDS * 1_000_000),
        "order": "fixed: store construction, full scan, then pushdown; one shot each",
        "scope": (
            "Observed process RSS in one shared process holding source data "
            "and existing stores; allocator retention and sampler timing "
            "dominate any difference this small"
        ),
        "idle_rss_bytes": idle_rss,
    }
    if build_store_fn is not None:
        base, peak = _measure_peak_rss(build_store_fn)
        out["store_construction_peak_rss_increase_bytes"] = max(0, peak - base)
    base, peak = _measure_peak_rss(lambda: FullScanQueryEngine(store).execute(q1))
    out["full_scan_q1_peak_rss_increase_bytes"] = max(0, peak - base)
    executor = QueryExecutor(store)
    base, peak = _measure_peak_rss(lambda: executor.execute(q1))
    out["q1_peak_rss_increase_bytes"] = max(0, peak - base)
    out["q1_increases_below_resolution_or_noise"] = (
        out["q1_peak_rss_increase_bytes"] <= 256 * _RSS_RESOLUTION_BYTES
        and out["full_scan_q1_peak_rss_increase_bytes"] <= 256 * _RSS_RESOLUTION_BYTES
    )
    return out


def run_memory_benchmarks(
    raw_data: pd.DataFrame | np.ndarray,
    store: ChunkStore,
    *,
    build_store_fn: Callable[[], ChunkStore] | None = None,
    queries: Mapping[str, str] | None = None,
    repeats: int = _TRACED_REPEATS,
    include_rss_diagnostic: bool = True,
) -> dict[str, Any]:
    """Measure retained representations and per-query traced peak allocation."""
    if not isinstance(store, ChunkStore):
        raise TypeError("store must be a ChunkStore")
    raw_bytes = _raw_size_bytes(raw_data)
    numeric_columns = list(store.value_columns)
    if isinstance(raw_data, pd.DataFrame):
        missing = set(numeric_columns) - set(raw_data.columns)
        if missing:
            raise ValueError(
                f"raw_data is missing stored numeric columns: {sorted(missing)}"
            )
        numeric_bytes = sum(
            int(raw_data[column].to_numpy(dtype=np.float64).nbytes)
            for column in numeric_columns
        )
    else:
        if raw_data.ndim != 2 or raw_data.shape[1] != len(numeric_columns):
            raise ValueError(
                "raw numeric arrays must have one column per stored value column"
            )
        numeric_bytes = int(np.asarray(raw_data, dtype=np.float64).nbytes)
    if len(raw_data) != store.total_rows:
        raise ValueError(
            "raw_data and store must contain the same number of physical rows"
        )
    store_bytes = _deep_size_bytes(store)
    breakdown = _memory_breakdown(store)

    if queries is None:
        queries = {
            name: sql for name, sql in BENCHMARK_QUERIES.items() if isinstance(sql, str)
        }
    q1 = queries.get("Q1") if queries else None
    if not isinstance(q1, str) or not q1:
        raise ValueError("queries must define a non-empty Q1 string")

    full_scan_engine = FullScanQueryEngine(store)
    pushdown_engine = QueryExecutor(store)
    vectorized_engine = QueryExecutor(store, enable_pruning=False)
    traced: dict[str, Any] = {}
    for name, sql in queries.items():
        peaks = measure_traced_peaks_interleaved(
            {
                "full_scan": lambda sql=sql: full_scan_engine.execute(sql),
                "vectorized_full_scan": lambda sql=sql: vectorized_engine.execute(sql),
                "pushdown_query": lambda sql=sql: pushdown_engine.execute(sql),
            },
            repeats,
        )
        row_count = pushdown_engine.execute(sql).row_count
        traced[name] = {"query": sql, "row_count": row_count, "engines": peaks}

    result: dict[str, Any] = {
        "raw_uncompressed_bytes": raw_bytes,
        "raw_numeric_payload_bytes": numeric_bytes,
        "numeric_payload_columns": numeric_columns,
        "numeric_payload_dtype": "float64",
        "numeric_payload_excludes": [
            "timestamp",
            "symbol",
            "index",
            "metadata",
            "Python object overhead",
        ],
        "numeric_payload_note": (
            "Storage is uncompressed float64: the raw numeric payload and the "
            "store column payload are equal by construction (an invariant "
            "check, not a result). No compression or storage reduction is "
            "measured or claimed."
        ),
        "numeric_payload_matches_store": numeric_bytes == store.total_column_bytes,
        "retained_size_accounting": (
            "Raw DataFrame uses pandas memory_usage(index=True, deep=True); "
            "store is an approximate Python object-graph walk. These differ "
            "in accounting and do not establish equivalent-content "
            "total-memory reduction."
        ),
        "chunk_store_bytes": store_bytes,
        "column_payload_bytes": store.total_column_bytes,
        **breakdown,
        "traced_query_peaks": traced,
        "traced_query_method": (
            "tracemalloc peak above the pre-call traced total; engines "
            "rotate through every ordering; gc.collect() before each call; "
            "one un-recorded warm pass; same process, not a fresh "
            "subprocess; covers Python and NumPy-registered allocations only"
        ),
        "traced_repeats": next(iter(traced.values()))["engines"]["pushdown_query"][
            "repeats"
        ],
    }
    if build_store_fn is not None:
        builds = [measure_traced_peak(build_store_fn) for _ in range(3)]
        result["store_construction_traced_peak_bytes"] = {
            "peak_bytes_samples": builds,
            "median_peak_bytes": int(statistics.median(builds)),
            "note": "a second, redundant build while source and store stay resident",
        }
    if include_rss_diagnostic:
        result["rss_diagnostic"] = _rss_diagnostic(store, q1, build_store_fn)
    return result


__all__ = [
    "measure_peak_memory",
    "measure_traced_peak",
    "measure_traced_peaks_interleaved",
    "run_memory_benchmarks",
]
