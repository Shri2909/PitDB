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
    """First-order propagation of latency SDs (descriptive, ignores covariance)."""
    relative_variance = (full_scan_std / full_scan_mean) ** 2 + (
        pushdown_std / pushdown_mean
    ) ** 2
    return speedup * math.sqrt(relative_variance)


# Propagated SD can exceed the speedup itself; cap the lower log-axis whisker.
_MAX_LOWER_ERROR_FRACTION = 0.99


def _clip_lower(
    label: str, selectivity: float, speedup: float, speedup_std: float
) -> float:
    if speedup_std > speedup * _MAX_LOWER_ERROR_FRACTION:
        print(
            f"Warning: {label} at selectivity={selectivity:g} has a propagated "
            f"speedup SD ({speedup_std:.3f}) close to or exceeding its own "
            f"speedup ({speedup:.3f}); clipping the lower whisker (log axis) "
            "rather than hiding the point."
        )
        return speedup * _MAX_LOWER_ERROR_FRACTION
    return speedup_std


def _extract_series(
    payload: Mapping[str, Any],
) -> tuple[dict[str, list[tuple[float, float, float, float]]], list[tuple[int, float]]]:
    """Return ``{series: [(selectivity, speedup, lower, upper)]}`` plus skipped."""
    series: dict[str, list[tuple[float, float, float, float]]] = {
        "row_wise": [],
        "vectorized": [],
    }
    skipped: list[tuple[int, float]] = []

    for index, raw_point in enumerate(_sweep_points(payload)):
        if not isinstance(raw_point, Mapping):
            raise ValueError(f"points[{index}] must be an object")

        selectivity = _require_number(
            raw_point.get("actual_selectivity"),
            f"points[{index}].actual_selectivity",
        )
        if selectivity > _MAX_SELECTIVITY:
            raise ValueError(
                f"points[{index}].actual_selectivity must not exceed {_MAX_SELECTIVITY}"
            )
        if selectivity < _MIN_SELECTIVITY:
            skipped.append((index, selectivity))
            continue

        speedup = _require_number(
            raw_point.get("speedup"), f"points[{index}].speedup", positive=True
        )
        full_scan_mean, full_scan_std = _latency_stats(raw_point, "full_scan", index)
        pushdown_mean, pushdown_std = _latency_stats(raw_point, "pushdown_query", index)
        if not math.isclose(
            speedup, full_scan_mean / pushdown_mean, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"points[{index}].speedup is inconsistent with the full-scan "
                "and pushdown-query means"
            )
        std = _propagated_speedup_std(
            speedup, full_scan_mean, full_scan_std, pushdown_mean, pushdown_std
        )
        series["row_wise"].append(
            (
                selectivity,
                speedup,
                _clip_lower("row-wise", selectivity, speedup, std),
                std,
            )
        )

        pushdown = raw_point["pushdown_query"]
        vectorized = pushdown.get("vectorized_full_scan")
        if not isinstance(vectorized, Mapping):
            raise ValueError(
                f"points[{index}].pushdown_query.vectorized_full_scan is missing; "
                "rerun the benchmarks"
            )
        vec_mean = _require_number(
            vectorized.get("mean_latency_seconds"),
            f"points[{index}] vectorized mean",
            positive=True,
        )
        vec_std = _require_number(
            vectorized.get("std_latency_seconds"), f"points[{index}] vectorized std"
        )
        vec_speedup = vec_mean / pushdown_mean
        vec_std_speedup = _propagated_speedup_std(
            vec_speedup, vec_mean, vec_std, pushdown_mean, pushdown_std
        )
        series["vectorized"].append(
            (
                selectivity,
                vec_speedup,
                _clip_lower("vectorized", selectivity, vec_speedup, vec_std_speedup),
                vec_std_speedup,
            )
        )

    if not series["row_wise"]:
        raise ValueError("no sweep point lies within the plotted selectivity range")
    for values in series.values():
        values.sort(key=lambda point: point[0])
    return series, skipped


def generate_chart(
    results: Mapping[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create and save the speedup/selectivity chart."""
    if isinstance(results, Mapping):
        payload = results
    elif isinstance(results, (str, Path)):
        payload = _load_json(Path(results).expanduser().resolve(strict=False))
    else:
        raise TypeError("results must be a mapping or a JSON file path")

    series, skipped = _extract_series(payload)
    all_points = [*series["row_wise"], *series["vectorized"]]

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

        lower_bound = min(1.0, *(p[1] - p[2] for p in all_points))
        upper_bound = max(1.0, *(p[1] + p[3] for p in all_points))
        axis.set_ylim(lower_bound / 1.9, upper_bound * 1.6)

        styles = {
            "row_wise": ("#2F6FA3", "o", "Row-wise full scan / pushdown"),
            "vectorized": (
                "#E07A2D",
                "s",
                "Vectorized full scan (no pruning) / pushdown",
            ),
        }
        for name, (color, marker, label) in styles.items():
            points = series[name]
            axis.errorbar(
                [p[0] for p in points],
                [p[1] for p in points],
                yerr=[[p[2] for p in points], [p[3] for p in points]],
                color=color,
                marker=marker,
                markersize=6.5,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.5,
                linewidth=2.0,
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                label=label,
                zorder=4,
            )
            for selectivity, speedup, _, _ in points:
                axis.annotate(
                    f"{speedup:.1f}x",
                    xy=(selectivity, speedup),
                    xytext=(0, 9 if name == "row_wise" else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=color,
                )
        axis.legend(loc="lower left", frameon=False, fontsize=9.5)
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
            "Query Speedup vs. Selectivity (Row-wise Baseline and Vectorized Control)",
            pad=16,
        )
        note = "Error bars: +/-1 SD, first-order propagation of per-engine latency SD."
        if skipped:
            note += (
                f" {len(skipped)} sweep point(s) with selectivity below "
                f"{_MIN_SELECTIVITY:.1%} not shown "
                f"({', '.join(f'{sel:.3%}' for _, sel in skipped)})."
            )
        figure.text(
            0.01, -0.02, note, ha="left", va="top", fontsize=8.5, color="#555555"
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
                "Title": "Query Speedup vs. Selectivity",
                "Description": (
                    "Log-log line chart of pushdown speedup versus measured "
                    "selectivity against both the row-wise full-scan baseline "
                    "and the vectorized no-pruning control, with +/-1 SD "
                    "(propagated) error bars and a no-improvement line at 1x."
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
