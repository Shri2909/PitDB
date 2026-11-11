"""Latency summaries (raw mean, descriptive Student-t interval) and interleaved timing.

SciPy is not a dependency, so t quantiles come from an embedded table.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Mapping
from itertools import permutations
from typing import Any

import numpy as np

Z_95 = 1.959963984540054  # large-sample limit of the t quantile below
LATENCY_TRIM_FRACTION = 0.15

# Two-sided 95% Student-t critical values t_{0.975, df} for df = 1..30.
_T_975_BY_DF = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip
# Sparser anchors beyond df=30; interpolated linearly in 1/df.
_T_975_TAIL = ((30, 2.042), (40, 2.021), (60, 2.000), (120, 1.980))


def t_critical_95(degrees_of_freedom: int) -> float:
    """Two-sided 95% Student-t critical value for ``degrees_of_freedom``."""
    if degrees_of_freedom < 1:
        raise ValueError("degrees_of_freedom must be >= 1")
    if degrees_of_freedom <= 30:
        return _T_975_BY_DF[degrees_of_freedom - 1]
    if degrees_of_freedom >= 120:
        weight = 120.0 / degrees_of_freedom
        return Z_95 + (1.980 - Z_95) * weight
    for (low_df, low_t), (high_df, high_t) in zip(_T_975_TAIL, _T_975_TAIL[1:]):
        if low_df <= degrees_of_freedom <= high_df:
            span = 1.0 / low_df - 1.0 / high_df
            weight = (1.0 / low_df - 1.0 / degrees_of_freedom) / span
            return low_t + (high_t - low_t) * weight
    raise AssertionError("unreachable")  # pragma: no cover


def confidence_interval_95(
    mean: float, std: float, num_trials: int
) -> tuple[float, float]:
    """Return a Student-t interval on the raw mean; coverage is not established."""
    if num_trials <= 1:
        return mean, mean
    half_width = t_critical_95(num_trials - 1) * std / math.sqrt(num_trials)
    return mean - half_width, mean + half_width


def intervals_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    """Describe interval overlap; this is not a test of a mean difference."""
    return a_low <= b_high and b_low <= a_high


def _samples(times: np.ndarray) -> np.ndarray:
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("latency samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("latency samples must be finite and non-negative")
    return values


def trimmed_mean_and_std(
    times: np.ndarray, trim_fraction: float = LATENCY_TRIM_FRACTION
) -> tuple[float, float, int]:
    """Describe the fastest retained fraction; no interval is inferred from it."""
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be within [0.0, 0.5)")
    sorted_times = np.sort(_samples(times))
    keep = max(1, int(round(len(sorted_times) * (1.0 - trim_fraction))))
    trimmed = sorted_times[:keep]
    return (
        float(np.mean(trimmed)),
        float(np.std(trimmed, ddof=1)) if keep > 1 else 0.0,
        keep,
    )


def latency_summary(times: np.ndarray) -> dict[str, Any]:
    """Keep all observations in acquisition order, and identify every estimator."""
    values = _samples(times)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    low, high = confidence_interval_95(mean, std, len(values))
    selected_mean, selected_std, kept = trimmed_mean_and_std(values)
    return {
        "raw_latency_samples_seconds": values.tolist(),
        "sample_count": len(values),
        "mean_latency_seconds": mean,
        "std_latency_seconds": std,
        "median_latency_seconds": float(np.median(values)),
        "mean_estimand": "raw untrimmed arithmetic mean of all measured trials",
        "ci95_low_seconds": low,
        "ci95_high_seconds": high,
        "uncertainty_method": (
            "descriptive Student-t interval, raw mean +/- t(0.975, n-1) * "
            f"sample SE (t={t_critical_95(len(values) - 1):.3f} for "
            f"n={len(values)}); coverage not established; no significance "
            "test"
            if len(values) > 1
            else "single observation; no interval"
        ),
        "upper_trim_fraction": LATENCY_TRIM_FRACTION,
        "retained_sample_count": kept,
        "upper_trimmed_mean_seconds": selected_mean,
        "upper_trimmed_std_seconds": selected_std,
        "trimmed_estimand": (
            "descriptive fastest retained fraction; no inferential interval"
        ),
    }


def balanced_trial_count(num_trials: int, num_engines: int) -> int:
    """Round ``num_trials`` up to a whole number of engine-permutation cycles."""
    cycle = math.factorial(num_engines)
    return -(-num_trials // cycle) * cycle


def time_interleaved(
    calls: Mapping[str, Callable[[], Any]], num_trials: int, warmup: int
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Time engines over rotating permutations, with GC disabled around each call.

    ``num_trials`` is rounded up to whole permutation cycles; read the actual
    count from ``len(samples[name])``. Returns samples, last results and order.
    """
    import time

    if (
        isinstance(num_trials, bool)
        or not isinstance(num_trials, int)
        or num_trials < 1
    ):
        raise ValueError("num_trials must be a positive integer")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not calls:
        raise ValueError("at least one timed callable is required")
    names = tuple(calls)
    orders = tuple(permutations(names))
    num_trials = balanced_trial_count(num_trials, len(names))
    samples = {name: np.empty(num_trials, dtype=np.float64) for name in names}
    results: dict[str, Any] = {}
    acquisition_order: list[dict[str, Any]] = []
    gc_was_enabled = gc.isenabled()
    try:
        for trial in range(warmup):
            for name in orders[trial % len(orders)]:
                gc.collect()
                gc.disable()
                calls[name]()
        for trial in range(num_trials):
            for position, name in enumerate(orders[trial % len(orders)]):
                gc.collect()
                gc.disable()
                started = time.perf_counter()
                results[name] = calls[name]()
                samples[name][trial] = time.perf_counter() - started
                acquisition_order.append(
                    {"round": trial, "position": position, "engine": name}
                )
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples, results, acquisition_order


__all__ = [
    "Z_95",
    "balanced_trial_count",
    "confidence_interval_95",
    "t_critical_95",
    "intervals_overlap",
    "trimmed_mean_and_std",
    "latency_summary",
    "time_interleaved",
]
