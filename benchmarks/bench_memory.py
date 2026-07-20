"""Memory-footprint benchmarks for raw data, chunk storage, and query Q1.

RSS is sampled at process level so native allocations made by NumPy and
pandas are included.  ``tracemalloc`` alone cannot observe those
allocations, so it is not used as the headline peak-memory metric.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from threading import Event, Thread
from typing import Any

import numpy as np
import pandas as pd

from src.query.executor import QueryExecutor
from src.store.chunk_store import ChunkStore

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - exercised without optional dep
    psutil = None  # type: ignore[assignment]

try:
    from benchmarks.config import BENCHMARK_QUERIES
except ModuleNotFoundError:
    BENCHMARK_QUERIES = {
        "Q1": ("SELECT * FROM data WHERE symbol = 'AAPL' AND date = '2024-03-01'")
    }


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
        name="streamxor-rss-sampler",
        daemon=True,
    )
    sampler.start()
    ready.wait()

    result: Any = None
    function_failed = False
    try:
        # Keep the result alive until the final sample. This captures callables
        # that create and return the object whose retained RSS is being tested.
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
        # Do not release a returned query DataFrame before its RSS is sampled.
        del result

    if sampler_error:
        raise RuntimeError("RSS sampler failed") from sampler_error[0]
    return baseline, peak[0]


def measure_peak_memory(fn: Callable[[], Any]) -> int:
    """Run ``fn`` and return the highest observed process RSS in bytes.

    Sampling occurs every millisecond and immediately before and after the
    call. Exceptions raised by ``fn`` propagate unchanged. The returned value
    is absolute RSS, as required by the benchmark contract, rather than a
    Python-allocation-only value or merely ``after - before``.
    """

    _, peak = _measure_peak_rss(fn)
    return peak


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


def run_memory_benchmarks(
    raw_data: pd.DataFrame | np.ndarray,
    store: ChunkStore,
) -> dict[str, int | str]:
    """Measure retained representations and peak RSS of the indexed Q1 path.

    Raw and chunk-store retained sizes are representation-level measurements;
    unlike two absolute process RSS readings, they remain comparable even when
    the caller already holds both objects. Query metrics report absolute peak
    RSS as well as growth above a garbage-collected pre-query baseline.
    """

    if not isinstance(store, ChunkStore):
        raise TypeError("store must be a ChunkStore")
    raw_bytes = _raw_size_bytes(raw_data)
    store_bytes = _deep_size_bytes(store)

    q1 = BENCHMARK_QUERIES.get("Q1")
    if not isinstance(q1, str) or not q1:
        raise ValueError("BENCHMARK_QUERIES must define a non-empty Q1 string")
    executor = QueryExecutor(store)
    baseline_rss, peak_rss = _measure_peak_rss(lambda: executor.execute(q1))

    return {
        "raw_uncompressed_bytes": raw_bytes,
        "chunk_store_bytes": store_bytes,
        "column_payload_bytes": store.total_column_bytes,
        "q1_peak_rss_bytes": peak_rss,
        "q1_baseline_rss_bytes": baseline_rss,
        "q1_peak_rss_increase_bytes": max(0, peak_rss - baseline_rss),
        "q1_query": q1,
        "rss_sampling_interval_microseconds": int(
            _RSS_SAMPLE_INTERVAL_SECONDS * 1_000_000
        ),
    }


__all__ = ["measure_peak_memory", "run_memory_benchmarks"]
