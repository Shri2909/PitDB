"""Generate Graph 2: row-wise, vectorized and pushdown latency for Q1-Q5."""

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


_ENGINES = (
    ("full_scan", "Row-wise full scan (baseline)", "#7B8492"),
    ("vectorized_full_scan", "Vectorized full scan (no pruning)", "#E0A83A"),
    ("pushdown_query", "Zone-map pushdown", "#2F6FA3"),
)


def _engine_block(query: Mapping[str, Any], engine_key: str) -> Mapping[str, Any]:
    if engine_key == "vectorized_full_scan":
        block = query.get("pushdown_query", {}).get("vectorized_full_scan")
    else:
        block = query.get(engine_key)
    return block if isinstance(block, Mapping) else {}


def _latency_stats(
    query: Mapping[str, Any],
    query_name: str,
    engine_key: str,
) -> tuple[float, float]:
    engine = _engine_block(query, engine_key)
    if not engine:
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
) -> tuple[dict[str, tuple[list[float], list[float]]], list[int], list[float]]:
    """Return per-engine (means, stds) in seconds, row counts and speedups."""
    queries = _fixed_queries(payload)
    data: dict[str, tuple[list[float], list[float]]] = {
        key: ([], []) for key, _, _ in _ENGINES
    }
    rows: list[int] = []
    speedups: list[float] = []
    for query_name in _QUERY_ORDER:
        query = queries.get(query_name)
        if not isinstance(query, Mapping):
            raise ValueError(f"query benchmark results are missing {query_name}")
        for key, _, _ in _ENGINES:
            mean, std = _latency_stats(query, query_name, key)
            data[key][0].append(mean)
            data[key][1].append(std)
        rows.append(int(query["pushdown_query"]["row_count"]))
        speedups.append(data["full_scan"][0][-1] / data["pushdown_query"][0][-1])
    return data, rows, speedups


def _format_latency(milliseconds: float) -> str:
    if milliseconds >= 1000.0:
        return f"{milliseconds / 1000.0:.2f} s"
    return f"{milliseconds:.2f} ms"


def generate_chart(
    results: Mapping[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create and save the grouped three-engine latency chart (log axis)."""
    if isinstance(results, Mapping):
        payload = results
    elif isinstance(results, (str, Path)):
        payload = _load_json(Path(results).expanduser().resolve(strict=False))
    else:
        raise TypeError("results must be a mapping or a JSON file path")

    data, rows, speedups = _extract_series(payload)
    data = {
        key: ([m * 1000.0 for m in means], [d * 1000.0 for d in stds])
        for key, (means, stds) in data.items()
    }

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
        "axes.titlesize": 14,
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
        figure, axis = plt.subplots(figsize=(11.0, 6.0), constrained_layout=True)
        positions = list(range(len(_QUERY_ORDER)))
        bar_width = 0.26
        error_style = {
            "elinewidth": 1.1,
            "ecolor": "#3F444A",
            "capsize": 3.0,
            "capthick": 1.1,
        }
        for offset, (key, label, color) in zip((-1, 0, 1), _ENGINES):
            means, stds = data[key]
            # Log axis: clip the lower whisker so it cannot reach zero.
            lower = [min(std, mean * 0.95) for mean, std in zip(means, stds)]
            xs = [position + offset * bar_width for position in positions]
            axis.bar(
                xs,
                means,
                width=bar_width,
                yerr=[lower, stds],
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                error_kw=error_style,
                zorder=3,
            )
            for x, mean, std in zip(xs, means, stds):
                axis.text(
                    x,
                    (mean + std) * 1.12,
                    _format_latency(mean),
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    rotation=90,
                    color="#222222",
                )

        axis.set_yscale("log")
        highest = max(
            mean + std for key, _, _ in _ENGINES for mean, std in zip(*data[key])
        )
        lowest = min(mean for key, _, _ in _ENGINES for mean in data[key][0])
        axis.set_ylim(lowest / 3.0, highest * 8.0)
        axis.set_xticks(
            positions,
            [
                f"{name}\n{count:,} row{'s' if count != 1 else ''}\n"
                f"row-wise/pushdown {speedup:.1f}x"
                for name, count, speedup in zip(_QUERY_ORDER, rows, speedups)
            ],
        )
        axis.set_ylabel("Mean latency (ms, log scale)")
        axis.set_title("Query Latency by Engine, Q1-Q5 (mean, +/-1 SD)", pad=14)
        axis.legend(loc="upper left", ncols=3, frameon=False, fontsize=9.5)
        axis.grid(axis="y", which="major", color="#D7DCE2", linewidth=0.75, zorder=0)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", length=0, pad=8)
        figure.text(
            0.01,
            -0.02,
            "Error bars: +/-1 sample SD of the trials (lower whisker clipped on the "
            "log axis); bars start at the axis floor.\n"
            "Small committed fixture (5,010 rows, 240 chunks): needle-query ratios "
            "mostly reflect per-chunk DataFrame construction in the row-wise "
            "baseline and do not extrapolate.",
            ha="left",
            va="top",
            fontsize=8.5,
            color="#555555",
        )

        figure.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
            metadata={
                "Title": "Query Latency by Engine, Q1-Q5",
                "Description": (
                    "Grouped bar chart on a log axis comparing the row-wise "
                    "full-scan baseline, the vectorized no-pruning control and "
                    "zone-map pushdown mean latency for Q1 through Q5, with "
                    "+/-1 SD error bars and result row counts."
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
