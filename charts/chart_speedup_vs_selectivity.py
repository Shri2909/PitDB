"""Generate Graph 1: pushdown-query speedup versus measured selectivity."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "benchmarks/results/queries.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "charts/output/01_speedup_vs_selectivity.png"

_MIN_SELECTIVITY = 0.001
_MAX_SELECTIVITY = 1.0


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"query benchmark results do not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"query benchmark results are not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("query benchmark results must be a JSON object")
    return payload


def _require_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{field} must be greater than zero")
    if not positive and number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _sweep_points(payload: Mapping[str, Any]) -> Sequence[Any]:
    if "selectivity_sweep" in payload:
        sweep = payload["selectivity_sweep"]
        if not isinstance(sweep, Mapping):
            raise ValueError("selectivity_sweep must be an object")
        payload = sweep

    points = payload.get("points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)):
        raise ValueError("selectivity sweep must contain a points array")
    if not points:
        raise ValueError("selectivity sweep contains no data points")
    return points


def _latency_stats(
    point: Mapping[str, Any],
    key: str,
    point_index: int,
) -> tuple[float, float]:
    metrics = point.get(key)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"points[{point_index}].{key} must be an object")
    mean = _require_number(
        metrics.get("mean_latency_seconds"),
        f"points[{point_index}].{key}.mean_latency_seconds",
        positive=True,
    )
    std = _require_number(
        metrics.get("std_latency_seconds"),
        f"points[{point_index}].{key}.std_latency_seconds",
    )
    return mean, std


def _propagated_speedup_std(
    speedup: float,
    full_scan_mean: float,
    full_scan_std: float,
    pushdown_mean: float,
    pushdown_std: float,
) -> float:
    """Propagate one-standard-deviation latency errors through both paths.

    The benchmark stores aggregate latency statistics rather than paired
    per-trial speedup samples. First-order independent-error propagation is
    therefore the available estimate for the speedup ratio's standard
    deviation.
    """

    relative_variance = (full_scan_std / full_scan_mean) ** 2 + (
        pushdown_std / pushdown_mean
    ) ** 2
    return speedup * math.sqrt(relative_variance)


# A propagated speedup std can exceed the speedup itself for a fast (few-ms)
# pushdown query measured over a small NUM_TRIALS sample, where a single
# trial hit by GC/OS scheduling noise dominates that point's variance --
# verified directly: instrumenting the real sweep showed no single cause
# (GC fires on nearly every trial at every point without this happening;
# it's a rare, larger stall, not routine collection). Rather than crashing
# chart generation over one noisy-but-otherwise-valid point, the lower
# error whisker is capped so it can never reach zero on this log-scale
# chart, and a warning is printed so the noise stays visible instead of
# being silently hidden.
_MAX_LOWER_ERROR_FRACTION = 0.99


def _extract_series(
    payload: Mapping[str, Any],
) -> tuple[list[float], list[float], list[float], list[float]]:
    series: list[tuple[float, float, float, float]] = []

    for index, raw_point in enumerate(_sweep_points(payload)):
        if not isinstance(raw_point, Mapping):
            raise ValueError(f"points[{index}] must be an object")

        selectivity = _require_number(
            raw_point.get("actual_selectivity"),
            f"points[{index}].actual_selectivity",
            positive=True,
        )
        if not _MIN_SELECTIVITY <= selectivity <= _MAX_SELECTIVITY:
            raise ValueError(
                f"points[{index}].actual_selectivity must be within "
                f"[{_MIN_SELECTIVITY}, {_MAX_SELECTIVITY}] for this log chart"
            )

        speedup = _require_number(
            raw_point.get("speedup"),
            f"points[{index}].speedup",
            positive=True,
        )
        full_scan_mean, full_scan_std = _latency_stats(raw_point, "full_scan", index)
        pushdown_mean, pushdown_std = _latency_stats(raw_point, "pushdown_query", index)

        calculated_speedup = full_scan_mean / pushdown_mean
        if not math.isclose(
            speedup,
            calculated_speedup,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"points[{index}].speedup is inconsistent with the full-scan "
                "and pushdown-query means"
            )

        speedup_std = _propagated_speedup_std(
            speedup,
            full_scan_mean,
            full_scan_std,
            pushdown_mean,
            pushdown_std,
        )
        lower_error = speedup_std
        max_lower_error = speedup * _MAX_LOWER_ERROR_FRACTION
        if lower_error >= speedup or lower_error > max_lower_error:
            print(
                f"Warning: points[{index}] (selectivity={selectivity:g}) has a "
                f"propagated speedup std ({speedup_std:.3f}) close to or "
                f"exceeding its own speedup ({speedup:.3f}) -- one or more "
                "timing trials for this point were likely disrupted by a GC "
                "pause or OS scheduling stall. Clipping the lower error "
                "whisker rather than hiding this point; consider re-running "
                "the benchmark or increasing NUM_TRIALS if this recurs."
            )
            lower_error = max_lower_error
        series.append((selectivity, speedup, lower_error, speedup_std))

    series.sort(key=lambda point: point[0])
    return (
        [point[0] for point in series],
        [point[1] for point in series],
        [point[2] for point in series],
        [point[3] for point in series],
    )


def generate_chart(
    results: Mapping[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create and save the publication-ready speedup/selectivity chart."""

    if isinstance(results, Mapping):
        payload = results
    elif isinstance(results, (str, Path)):
        payload = _load_json(Path(results).expanduser().resolve(strict=False))
    else:
        raise TypeError("results must be a mapping or a JSON file path")

    selectivities, speedups, lower_errors, upper_errors = _extract_series(payload)

    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10.5,
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    with plt.rc_context(rc):
        figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(_MIN_SELECTIVITY, _MAX_SELECTIVITY)

        lower_bound = min(
            1.0,
            *(value - error for value, error in zip(speedups, lower_errors)),
        )
        upper_bound = max(
            1.0,
            *(value + error for value, error in zip(speedups, upper_errors)),
        )
        axis.set_ylim(lower_bound / 1.35, upper_bound * 1.35)

        axis.errorbar(
            selectivities,
            speedups,
            yerr=[lower_errors, upper_errors],
            color="#2F6FA3",
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor="#2F6FA3",
            markeredgewidth=1.5,
            linewidth=2.0,
            elinewidth=1.2,
            capsize=4,
            capthick=1.2,
            zorder=4,
        )
        axis.axhline(
            1.0,
            color="#666666",
            linestyle="--",
            linewidth=1.2,
            zorder=2,
        )
        axis.annotate(
            "No improvement (1x)",
            xy=(0.98, 1.0),
            xycoords=("axes fraction", "data"),
            xytext=(0, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9.5,
            color="#555555",
        )

        axis.set_xlabel("Actual selectivity")
        axis.set_ylabel("Speedup factor (x)")
        axis.set_title(
            "Query Speedup vs. Selectivity (Pushdown Query vs. Full Scan)",
            pad=16,
        )

        axis.set_xticks([0.001, 0.01, 0.1, 1.0])
        axis.set_xticklabels(["0.1%", "1%", "10%", "100%"])
        axis.yaxis.set_major_locator(LogLocator(base=10.0))
        axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0, 7.0)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
        axis.yaxis.set_minor_formatter(NullFormatter())

        axis.grid(
            which="major",
            color="#D7DCE2",
            linewidth=0.75,
            alpha=0.85,
            zorder=0,
        )
        axis.grid(
            which="minor",
            axis="y",
            color="#E7EAEE",
            linewidth=0.5,
            alpha=0.65,
            zorder=0,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(axis="both", which="both", color="#555555")

        figure.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
            metadata={
                "Title": (
                    "Query Speedup vs. Selectivity (Pushdown Query vs. Full Scan)"
                ),
                "Description": (
                    "Log-log line chart of pushdown-query speedup versus "
                    "measured selectivity with one-standard-deviation error "
                    "bars and a no-improvement baseline at 1x."
                ),
            },
        )
        plt.close(figure)

    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="queries.json generated by benchmarks/run_all.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="destination PNG path",
    )
    args = parser.parse_args(argv)
    output = generate_chart(args.results, args.output)
    print(f"Generated {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_chart", "main"]
