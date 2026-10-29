"""Recoverable latency observations and explicitly descriptive statistics.

The headline mean uses every observation. Normal-approximation intervals
are shown only as a descriptive uncertainty summary: wall-clock trials in
one process need not be independent or normally distributed. No calibrated
coverage or significance test is claimed. Upper-tail trimming is reported
separately and never used to compute an interval or headline speedup.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from itertools import permutations
from typing import Any

import numpy as np

Z_95 = 1.959963984540054
LATENCY_TRIM_FRACTION = 0.15


def confidence_interval_95(
    mean: float, std: float, num_trials: int
) -> tuple[float, float]:
    """Return a normal-approximation interval; coverage is not established."""
    if num_trials <= 1:
        return mean, mean
    half_width = Z_95 * std / math.sqrt(num_trials)
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
            "descriptive normal approximation, raw mean +/- 1.959964 * sample"
            " SE; coverage not established; no significance test"
        ),
        "upper_trim_fraction": LATENCY_TRIM_FRACTION,
        "retained_sample_count": kept,
        "upper_trimmed_mean_seconds": selected_mean,
        "upper_trimmed_std_seconds": selected_std,
        "trimmed_estimand": (
            "descriptive fastest retained fraction; no inferential interval"
        ),
    }


def time_interleaved(
    calls: Mapping[str, Callable[[], Any]], num_trials: int, warmup: int
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Rotate all engine permutations across paired rounds and retain order.

    Complete permutation cycles balance both position and pair precedence.
    A partial final cycle is disclosed by the acquisition records. This is
    one warm process; cache state and independent-trial assumptions are not
    controlled. The clock covers only the callable, including SQL execution.
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
    for trial in range(warmup):
        for name in orders[trial % len(orders)]:
            calls[name]()
    samples = {name: np.empty(num_trials, dtype=np.float64) for name in names}
    results: dict[str, Any] = {}
    acquisition_order: list[dict[str, Any]] = []
    for trial in range(num_trials):
        for position, name in enumerate(orders[trial % len(orders)]):
            started = time.perf_counter()
            results[name] = calls[name]()
            samples[name][trial] = time.perf_counter() - started
            acquisition_order.append(
                {"round": trial, "position": position, "engine": name}
            )
    return samples, results, acquisition_order


__all__ = [
    "Z_95",
    "confidence_interval_95",
    "intervals_overlap",
    "trimmed_mean_and_std",
    "latency_summary",
    "time_interleaved",
]
