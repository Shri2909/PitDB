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

from src.baselines.full_scan import FullScanQueryEngine
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
        name="pitdb-rss-sampler",
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


# ChunkLabel fields that form the pruning-relevant zone map (see
# src/store/label.py's own docstring: "the min/max fields form a compact
# zone map"), vs. every other label field, which this breakdown treats as
# plain chunk-identity/bookkeeping metadata.
_ZONE_MAP_LABEL_FIELDS = ("min_price", "max_price", "time_start", "time_end")
_METADATA_LABEL_FIELDS = (
    "symbol",
    "row_count",
    "chunk_id",
    "transaction_time_start",
    "transaction_time_end",
)


def _memory_breakdown(store: ChunkStore) -> dict[str, int]:
    """Decompose the store's total deep object-graph size into disjoint,
    reconciling components, using ``_deep_size_bytes``'s own ``seen``
    id-set walked in a fixed order (packed buffers, then label metadata
    and zone-map fields, then everything else) so an object reachable from
    more than one place is only ever counted once anywhere in this
    breakdown, and the components always sum exactly to
    ``chunk_store_bytes`` (``_deep_size_bytes(store)`` with a fresh set).

    This mirrors the codebase's own architecture rather than inventing
    categories: ``ChunkLabel`` is documented as carrying both plain
    metadata and the zone map in the same object (see
    ``_ZONE_MAP_LABEL_FIELDS``/``_METADATA_LABEL_FIELDS`` above), so this
    is a field-level split of that one object, not two separate structures.
    "Labels and indexes" below is what's left of the store's own top-level
    containers (``chunks``, plus internal id/as-of indexes) after packed
    buffers and label fields are already accounted for; "Python object
    overhead" is the final residual (the ``ChunkStore`` instance itself and
    any container-shell bytes not attributed elsewhere) -- computed as a
    subtraction, not an independent estimate, so rounding or an
    unanticipated attribute can never make the parts exceed the whole.
    """

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
            # The dataclass instance's own shell -- not attributable to any
            # single field -- is counted as identity/bookkeeping overhead.
            metadata_bytes += sys.getsizeof(label)
        metadata_bytes += sum(
            _deep_size_bytes(getattr(label, name), seen)
            for name in _METADATA_LABEL_FIELDS
        )
        zone_map_bytes += sum(
            _deep_size_bytes(getattr(label, name), seen)
            for name in _ZONE_MAP_LABEL_FIELDS
        )

    total_bytes = _deep_size_bytes(store)  # fresh, independent walk -- the ground truth

    # Store-level index/bookkeeping containers, reachable only through
    # private ChunkStore attributes (there is no public accessor for them,
    # and adding one would mean changing src/store/chunk_store.py, which
    # this presentation-only change does not touch). Walked through the
    # same `seen` set as the label fields above, so labels already counted
    # as metadata/zone-map bytes (e.g. reachable again via _as_of_labels)
    # contribute zero further bytes here -- only each container's own
    # shell and any not-yet-visited entries (like the as-of index's
    # datetime keys) are newly counted.
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


def run_memory_benchmarks(
    raw_data: pd.DataFrame | np.ndarray,
    store: ChunkStore,
    *,
    build_store_fn: Callable[[], ChunkStore] | None = None,
) -> dict[str, int | str]:
    """Measure retained representations and peak RSS of the indexed Q1 path.

    Raw and chunk-store retained sizes are representation-level measurements;
    unlike two absolute process RSS readings, they remain comparable even when
    the caller already holds both objects. Query metrics report absolute peak
    RSS as well as growth above a garbage-collected pre-query baseline.

    ``build_store_fn``, if given, is a zero-argument callable that
    constructs a fresh ``ChunkStore`` equivalent to ``store`` from the same
    source data -- used only to measure "store construction"'s own peak RSS
    in isolation (a second, redundant build; every other measurement here
    uses the already-built ``store`` the caller passed in). When omitted,
    construction-RSS fields are not included in the returned dict.
    """

    if not isinstance(store, ChunkStore):
        raise TypeError("store must be a ChunkStore")
    raw_bytes = _raw_size_bytes(raw_data)
    store_bytes = _deep_size_bytes(store)
    breakdown = _memory_breakdown(store)

    q1 = BENCHMARK_QUERIES.get("Q1")
    if not isinstance(q1, str) or not q1:
        raise ValueError("BENCHMARK_QUERIES must define a non-empty Q1 string")

    # "Idle" here means this process's RSS immediately before this
    # function's own construction/query measurements begin -- it already
    # carries whatever the interpreter, pandas/numpy imports, and any
    # earlier benchmark in the same run left resident, not a
    # freshly-launched process's baseline.
    idle_rss, _ = _measure_peak_rss(lambda: None)

    construction_baseline_rss = None
    construction_peak_rss = None
    if build_store_fn is not None:
        construction_baseline_rss, construction_peak_rss = _measure_peak_rss(
            build_store_fn
        )

    full_scan_engine = FullScanQueryEngine(store)
    full_scan_baseline_rss, full_scan_peak_rss = _measure_peak_rss(
        lambda: full_scan_engine.execute(q1)
    )

    executor = QueryExecutor(store)
    baseline_rss, peak_rss = _measure_peak_rss(lambda: executor.execute(q1))

    result: dict[str, Any] = {
        "raw_uncompressed_bytes": raw_bytes,
        "chunk_store_bytes": store_bytes,
        "column_payload_bytes": store.total_column_bytes,
        **breakdown,
        "idle_rss_bytes": idle_rss,
        "full_scan_q1_baseline_rss_bytes": full_scan_baseline_rss,
        "full_scan_q1_peak_rss_bytes": full_scan_peak_rss,
        "q1_peak_rss_bytes": peak_rss,
        "q1_baseline_rss_bytes": baseline_rss,
        "q1_peak_rss_increase_bytes": max(0, peak_rss - baseline_rss),
        "q1_query": q1,
        "rss_sampling_interval_microseconds": int(
            _RSS_SAMPLE_INTERVAL_SECONDS * 1_000_000
        ),
    }
    if build_store_fn is not None:
        result["store_construction_baseline_rss_bytes"] = construction_baseline_rss
        result["store_construction_peak_rss_bytes"] = construction_peak_rss
    return result


__all__ = ["measure_peak_memory", "run_memory_benchmarks"]
