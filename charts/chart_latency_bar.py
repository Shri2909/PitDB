"""Generate Graph 2: full-scan and pushdown-query latency for Q1-Q5."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "benchmarks/results/queries.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "charts/output/02_latency_q1_q5.png"

_QUERY_ORDER = ("Q1", "Q2", "Q3", "Q4", "Q5")


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


def _fixed_queries(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if "fixed_queries" in payload:
        fixed = payload["fixed_queries"]
        if not isinstance(fixed, Mapping):
            raise ValueError("fixed_queries must be an object")
        payload = fixed
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


def _latency_stats(
    query: Mapping[str, Any],
    query_name: str,
    engine_key: str,
) -> tuple[float, float]:
    engine = query.get(engine_key)
    if not isinstance(engine, Mapping):
        raise ValueError(f"{query_name}.{engine_key} must be an object")
    mean = _require_number(
        engine.get("mean_latency_seconds"),
        f"{query_name}.{engine_key}.mean_latency_seconds",
        positive=True,
    )
    std = _require_number(
        engine.get("std_latency_seconds"),
        f"{query_name}.{engine_key}.std_latency_seconds",
    )
    return mean, std


def _extract_series(
    payload: Mapping[str, Any],
) -> tuple[list[float], list[float], list[float], list[float]]:
    queries = _fixed_queries(payload)
    full_scan_means: list[float] = []
    full_scan_stds: list[float] = []
    pushdown_means: list[float] = []
    pushdown_stds: list[float] = []

    for query_name in _QUERY_ORDER:
        query = queries.get(query_name)
        if not isinstance(query, Mapping):
            raise ValueError(f"query benchmark results are missing {query_name}")
        full_scan_mean, full_scan_std = _latency_stats(
            query,
            query_name,
            "full_scan",
        )
        pushdown_mean, pushdown_std = _latency_stats(
            query,
            query_name,
            "pushdown_query",
        )
        full_scan_means.append(full_scan_mean)
        full_scan_stds.append(full_scan_std)
        pushdown_means.append(pushdown_mean)
        pushdown_stds.append(pushdown_std)

    return full_scan_means, full_scan_stds, pushdown_means, pushdown_stds


def _display_units(
    full_scan_means: list[float],
    full_scan_stds: list[float],
    pushdown_means: list[float],
    pushdown_stds: list[float],
) -> tuple[float, str]:
    """Use milliseconds when every observed mean is below one second."""

    if max((*full_scan_means, *pushdown_means)) < 1.0:
        return 1_000.0, "Latency (ms)"
    return 1.0, "Latency (seconds)"


def generate_chart(
    results: Mapping[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create and save the publication-ready grouped latency chart."""

    if isinstance(results, Mapping):
        payload = results
    elif isinstance(results, (str, Path)):
        payload = _load_json(Path(results).expanduser().resolve(strict=False))
    else:
        raise TypeError("results must be a mapping or a JSON file path")

    full_scan_means, full_scan_stds, pushdown_means, pushdown_stds = _extract_series(
        payload
    )
    scale, y_label = _display_units(
        full_scan_means,
        full_scan_stds,
        pushdown_means,
        pushdown_stds,
    )
    full_scan_means = [value * scale for value in full_scan_means]
    full_scan_stds = [value * scale for value in full_scan_stds]
    pushdown_means = [value * scale for value in pushdown_means]
    pushdown_stds = [value * scale for value in pushdown_stds]

    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination = destination.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "font.size": 10.5,
        "axes.titlesize": 15,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    with plt.rc_context(rc):
        figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
        positions = list(range(len(_QUERY_ORDER)))
        bar_width = 0.34
        full_scan_positions = [position - bar_width / 2.0 for position in positions]
        pushdown_positions = [position + bar_width / 2.0 for position in positions]
        error_style = {
            "elinewidth": 1.1,
            "ecolor": "#3F444A",
            "capsize": 3.5,
            "capthick": 1.1,
        }

        axis.bar(
            full_scan_positions,
            full_scan_means,
            width=bar_width,
            yerr=full_scan_stds,
            label="Full scan",
            color="#7B8492",
            edgecolor="white",
            linewidth=0.8,
            error_kw=error_style,
            zorder=3,
        )
        axis.bar(
            pushdown_positions,
            pushdown_means,
            width=bar_width,
            yerr=pushdown_stds,
            label="Pushdown query",
            color="#2F6FA3",
            edgecolor="white",
            linewidth=0.8,
            error_kw=error_style,
            zorder=3,
        )

        highest = max(
            *(mean + std for mean, std in zip(full_scan_means, full_scan_stds)),
            *(mean + std for mean, std in zip(pushdown_means, pushdown_stds)),
        )
        axis.set_ylim(0.0, highest * 1.18)
        axis.set_xticks(positions, _QUERY_ORDER)
        axis.set_ylabel(y_label)
        axis.set_title(
            "Query Latency: Full Scan vs. Pushdown Query (Q1–Q5)",
            pad=16,
        )
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncols=2,
            frameon=False,
            handlelength=1.5,
            columnspacing=2.0,
        )
        axis.grid(
            axis="y",
            color="#D7DCE2",
            linewidth=0.75,
            alpha=0.85,
            zorder=0,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(axis="x", length=0, pad=8)
        axis.tick_params(axis="y", width=0.8, color="#555555")

        figure.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
            metadata={
                "Title": "Query Latency: Full Scan vs. Pushdown Query (Q1–Q5)",
                "Description": (
                    "Grouped bar chart comparing full-scan and pushdown-query "
                    "mean query latency for Q1 through Q5, with one-standard-"
                    "deviation error bars."
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
