"""Generate Graph 3: chunk-skipping effectiveness versus selectivity."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "benchmarks/results/queries.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "charts/output/03_chunks_skipped.png"

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


def _require_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
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


def _extract_series(
    payload: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    series: list[tuple[float, float]] = []

    for index, raw_point in enumerate(_sweep_points(payload)):
        if not isinstance(raw_point, Mapping):
            raise ValueError(f"points[{index}] must be an object")

        # Plot achieved row selectivity rather than the nominal sweep target;
        # real time and price distributions do not guarantee exact target hits.
        selectivity = _require_number(
            raw_point.get("actual_selectivity"),
            f"points[{index}].actual_selectivity",
        )
        if not _MIN_SELECTIVITY <= selectivity <= _MAX_SELECTIVITY:
            raise ValueError(
                f"points[{index}].actual_selectivity must be within "
                f"[{_MIN_SELECTIVITY}, {_MAX_SELECTIVITY}] for this log chart"
            )

        skipped_ratio = _require_number(
            raw_point.get("chunks_skipped_ratio"),
            f"points[{index}].chunks_skipped_ratio",
        )
        if not 0.0 <= skipped_ratio <= 1.0:
            raise ValueError(
                f"points[{index}].chunks_skipped_ratio must be between 0 and 1"
            )
        series.append((selectivity, skipped_ratio))

    series.sort(key=lambda point: point[0])
    return (
        [point[0] for point in series],
        [point[1] for point in series],
    )


def generate_chart(
    results: Mapping[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create and save the publication-ready chunk-skipping chart."""

    if isinstance(results, Mapping):
        payload = results
    elif isinstance(results, (str, Path)):
        payload = _load_json(Path(results).expanduser().resolve(strict=False))
    else:
        raise TypeError("results must be a mapping or a JSON file path")

    selectivities, skipped_ratios = _extract_series(payload)

    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator

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
        axis.set_xlim(_MIN_SELECTIVITY, _MAX_SELECTIVITY)
        axis.set_ylim(0.0, 1.0)

        axis.plot(
            selectivities,
            skipped_ratios,
            color="#2F6FA3",
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor="#2F6FA3",
            markeredgewidth=1.5,
            linewidth=2.0,
            zorder=4,
        )

        axis.set_xlabel("Actual selectivity")
        axis.set_ylabel("Chunks-skipped ratio")
        axis.set_title(
            "Predicate Pushdown Effectiveness: Chunks Skipped vs. Selectivity",
            pad=16,
        )
        axis.set_xticks([0.001, 0.01, 0.1, 1.0])
        axis.set_xticklabels(["0.1%", "1%", "10%", "100%"])
        axis.yaxis.set_major_locator(MultipleLocator(0.2))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        axis.grid(
            which="major",
            color="#D7DCE2",
            linewidth=0.75,
            alpha=0.85,
            zorder=0,
        )
        axis.grid(
            which="minor",
            axis="x",
            color="#E7EAEE",
            linewidth=0.5,
            alpha=0.6,
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
                    "Predicate Pushdown Effectiveness: Chunks Skipped vs. Selectivity"
                ),
                "Description": (
                    "Line chart of the fraction of compressed chunks skipped "
                    "by conservative predicate pushdown at each measured "
                    "query selectivity."
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
