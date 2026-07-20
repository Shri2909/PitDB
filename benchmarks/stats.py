"""Shared statistical helpers for benchmark reporting.

scipy is not a project dependency, so confidence intervals use the normal
rather than the exact Student's-t critical value. With NUM_TRIALS=15 (14
degrees of freedom) the t critical value is ~2.145 vs. 1.96 used here, a ~9%
wider interval than reported -- this is a slightly optimistic (narrow)
approximation, not an exact interval.
"""

from __future__ import annotations

import math

import numpy as np

Z_95 = 1.959963984540054


def confidence_interval_95(
    mean: float, std: float, num_trials: int
) -> tuple[float, float]:
    """Return an approximate 95% CI for a mean of ``num_trials`` samples."""

    if num_trials <= 1:
        return mean, mean
    half_width = Z_95 * std / math.sqrt(num_trials)
    return mean - half_width, mean + half_width


def intervals_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    """Return whether two closed intervals share any point."""

    return a_low <= b_high and b_low <= a_high


def trimmed_mean_and_std(
    times: np.ndarray, trim_fraction: float = 0.15
) -> tuple[float, float, int]:
    """Return (mean, sample std, kept-sample count) after dropping the
    slowest ``trim_fraction`` of trials.

    Latency outliers in wall-clock benchmarking are one-directional: a GC
    pause, OS scheduling hiccup, or background process makes a trial
    slower -- nothing makes one impossibly faster. Symmetric trimming (the
    classic trimmed mean, dropping both tails) would discard good fast
    trials for no reason; this only trims the upper (slow) tail, then
    reports plain mean/std (ddof=1) over what's left. Feed the returned
    sample count into confidence_interval_95, not the original trial count
    -- the CI is over the trimmed sample, not the raw one.
    """

    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be within [0.0, 0.5)")

    sorted_times = np.sort(times)
    keep = max(1, int(round(len(sorted_times) * (1.0 - trim_fraction))))
    trimmed = sorted_times[:keep]
    mean = float(np.mean(trimmed))
    std = float(np.std(trimmed, ddof=1)) if len(trimmed) > 1 else 0.0
    return mean, std, len(trimmed)


__all__ = [
    "Z_95",
    "confidence_interval_95",
    "intervals_overlap",
    "trimmed_mean_and_std",
]
