"""Generate Graph 5: dataset-scale sweep of ingestion and query performance.

Plots how ingestion throughput, isolated pushdown-evaluation cost, and
pushdown-query latency for both the narrow symbol-equality query and the
broad full-range query behave as synthetic dataset scale increases from 1x
to 1000x the size of Dataset A.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "benchmarks/results/scale_sweep.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "charts/output/05_scale_sweep.png"


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"scale sweep results do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "points" not in payload:
        raise ValueError(
            "scale sweep results must be a JSON object with a 'points' array"
        )
    return payload["points"]


def generate_chart(
    results: list[dict[str, Any]] | dict[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    if isinstance(results, (str, Path)):
        points = _load_json(Path(results).expanduser().resolve(strict=False))
    elif isinstance(results, dict):
        points = results["points"]
    else:
        points = results

    labels = [point["label"] for point in points]
    ingest_rows_per_sec = [point["ingest_rows_per_second"] for point in points]
    ns_per_label = [point["pushdown_eval"]["ns_per_label_broad"] for point in points]
    narrow_pushdown_ms = [
        point["queries"]["narrow_symbol_eq"]["pushdown_mean_ms"] for point in points
    ]
    broad_pushdown_ms = [
        point["queries"]["broad_full_range"]["pushdown_mean_ms"] for point in points
    ]
    broad_full_scan_ms = [
        point["queries"]["broad_full_range"].get("full_scan_mean_ms")
        for point in points
    ]

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
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }

    color_ingest = "#2F6FA3"
    color_pushdown = "#4C9F70"
    color_narrow = "#E07A2D"
    color_broad = "#B23A48"
    color_full_scan = "#8A94A3"

    with plt.rc_context(rc):
        figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
        positions = list(range(len(labels)))

        ax0 = axes[0]
        ax0.plot(
            positions,
            ingest_rows_per_sec,
            color=color_ingest,
            marker="o",
            linewidth=2.0,
        )
        ax0.set_title("Ingestion throughput vs scale")
        ax0.set_ylabel("Rows/second")
        ax0.set_xticks(positions, labels)
        ax0.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)

        ax1 = axes[1]
        ax1.plot(
            positions, ns_per_label, color=color_pushdown, marker="o", linewidth=2.0
        )
        ax1.set_title("Pushdown label-eval cost vs scale")
        ax1.set_ylabel("ns/label (broad query)")
        ax1.set_xticks(positions, labels)
        ax1.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)

        ax2 = axes[2]
        ax2.plot(
            positions,
            narrow_pushdown_ms,
            color=color_narrow,
            marker="o",
            linewidth=2.0,
            label="Pushdown query, narrow symbol filter",
        )
        ax2.plot(
            positions,
            broad_pushdown_ms,
            color=color_broad,
            marker="s",
            linewidth=2.0,
            label="Pushdown query, broad 0%-skip filter",
        )
        if all(value is not None for value in broad_full_scan_ms):
            ax2.plot(
                positions,
                broad_full_scan_ms,
                color=color_full_scan,
                marker="^",
                linewidth=1.6,
                linestyle="--",
                label="Full scan, broad filter",
            )
        ax2.set_yscale("log")
        ax2.set_title("Query latency vs scale (log scale)")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_xticks(positions, labels)
        ax2.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)
        ax2.legend(fontsize=8.5, frameon=False, loc="upper left")

        for axis in axes:
            axis.set_xlabel("Scale (symbol count multiplier)")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_axisbelow(True)

        figure.suptitle(
            "Scale Sweep — Synthetic Dataset at 1x/10x/100x/1000x",
            fontsize=14.5,
            fontweight="semibold",
        )

        figure.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
            metadata={
                "Title": "Scale Sweep",
                "Description": (
                    "Ingestion, pushdown-evaluation, and query-latency metrics "
                    "across increasing synthetic dataset scale (chunk/row count)."
                ),
            },
        )
        plt.close(figure)

    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args(argv)
    output = generate_chart(args.results, args.output)
    print(f"Generated {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_chart", "main"]
