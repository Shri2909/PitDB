"""Generate Graph 6: the bitemporal correction-volume sweep chart.

Three panels, all reading benchmarks/results/bitemporal_sweep.json (see
benchmarks/bench_bitemporal.py, which measures every scenario both before
and after compact_corrections() so the compaction's real effect is always
directly visible, never requiring a separate before/after run to see):

1. Correction chunks pruned by AS OF (post-compaction) -- transaction-time
   pushdown pruning still works after chunks have been merged.
2. The "current" (no AS OF) scenario's pushdown latency, pre- vs.
   post-compaction, against full-scan -- this is the actual proof that
   compaction helps: fewer, larger chunks mean less decompression/concat
   overhead for a query that can't prune on transaction time at all.
3. compact_corrections() itself: wall-clock cost vs. correction volume --
   this should stay roughly linear (it did not, before the O(n^2) chunk
   removal bug in ChunkStore._remove_chunk was fixed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "benchmarks/results/bitemporal_sweep.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "charts/output/06_bitemporal_sweep.png"


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"bitemporal sweep results do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "points" not in payload:
        raise ValueError(
            "bitemporal sweep results must be a JSON object with a 'points' array"
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
    positions = list(range(len(labels)))

    as_of_positions = [
        index
        for index, point in enumerate(points)
        if "as_of_post_compaction" in point["scenarios"]
    ]
    as_of_skipped_ratio = [
        points[index]["scenarios"]["as_of_post_compaction"]["chunks_skipped_ratio"]
        for index in as_of_positions
    ]

    current_pre_ms = [
        point["scenarios"]["current_pre_compaction"]["query_pushdown_mean_ms"]
        for point in points
    ]
    current_post_ms = [
        point["scenarios"]["current_post_compaction"]["query_pushdown_mean_ms"]
        for point in points
    ]
    current_full_scan_ms = [
        point["scenarios"]["current_pre_compaction"]["query_full_scan_mean_ms"]
        for point in points
    ]
    compaction_ms = [point["compaction_seconds"] * 1000.0 for point in points]
    correction_counts = [point["num_corrections"] for point in points]

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

    color_skipped = "#4C9F70"
    color_pre = "#B23A48"
    color_post = "#4C9F70"
    color_full_scan = "#8A94A3"
    color_compaction = "#2F6FA3"

    with plt.rc_context(rc):
        figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)

        ax0 = axes[0]
        ax0.bar(
            [labels[index] for index in as_of_positions],
            [ratio * 100.0 for ratio in as_of_skipped_ratio],
            color=color_skipped,
        )
        ax0.set_title("Correction chunks pruned by AS OF\n(after compaction)")
        ax0.set_ylabel("Chunks skipped (%)")
        ax0.set_xlabel("Correction rate")
        ax0.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)

        ax1 = axes[1]
        ax1.plot(
            positions,
            current_pre_ms,
            color=color_pre,
            marker="o",
            linewidth=2.0,
            label="Pushdown, pre-compaction",
        )
        ax1.plot(
            positions,
            current_post_ms,
            color=color_post,
            marker="s",
            linewidth=2.0,
            label="Pushdown, post-compaction",
        )
        ax1.plot(
            positions,
            current_full_scan_ms,
            color=color_full_scan,
            marker="^",
            linewidth=1.6,
            linestyle="--",
            label="Full scan (pre-compaction)",
        )
        ax1.set_yscale("log")
        ax1.set_title("'Current' (no AS OF) query latency:\ncompaction's real effect")
        ax1.set_ylabel("Latency (ms, log scale)")
        ax1.set_xlabel("Correction rate")
        ax1.set_xticks(positions, labels)
        ax1.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)
        ax1.legend(fontsize=8.0, frameon=False, loc="upper left")

        ax2 = axes[2]
        ax2.plot(
            correction_counts,
            compaction_ms,
            color=color_compaction,
            marker="o",
            linewidth=2.0,
        )
        ax2.set_title("compact_corrections() cost\nvs. correction volume")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_xlabel("Number of corrections")
        ax2.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)

        for axis in axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_axisbelow(True)

        figure.suptitle(
            "Bitemporal Correction-Volume Sweep",
            fontsize=14.5,
            fontweight="semibold",
        )

        figure.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
            metadata={
                "Title": "Bitemporal Correction-Volume Sweep",
                "Description": (
                    "AS OF pruning, compaction's pre/post effect on query "
                    "latency, and compact_corrections() cost, as correction "
                    "volume grows relative to base data."
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
