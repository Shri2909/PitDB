"""Generate Graph 7: the chunk-granularity tradeoff chart.

Plots chunk count, chunks-skipped-ratio, and query speedup for a fixed
combined date-range + price-band query across a few ChunkBoundary
granularities -- the actual measured Pareto curve behind "finer chunks
prune better but cost more chunk-management overhead," rather than an
unverified claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "benchmarks/results/chunk_granularity_sweep.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "charts/output/07_chunk_granularity.png"


def _load_json(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"chunk granularity sweep results do not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "points" not in payload:
        raise ValueError(
            "chunk granularity sweep results must be a JSON object with a "
            "'points' array"
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
    total_chunks = [point["total_chunks"] for point in points]
    skipped_ratio = [point["chunks_skipped_ratio"] * 100.0 for point in points]
    speedup = [point["query_speedup"] for point in points]

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

    color_chunks = "#2F6FA3"
    color_skipped = "#4C9F70"
    color_speedup = "#E07A2D"

    with plt.rc_context(rc):
        figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), constrained_layout=True)

        ax0 = axes[0]
        ax0.bar(labels, total_chunks, color=color_chunks)
        ax0.set_title("Chunk count vs granularity")
        ax0.set_ylabel("Total chunks")
        ax0.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)

        ax1 = axes[1]
        ax1.bar(labels, skipped_ratio, color=color_skipped)
        ax1.set_title("Pruning precision vs granularity")
        ax1.set_ylabel("Chunks skipped (%)")
        ax1.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)

        ax2 = axes[2]
        ax2.plot(positions, speedup, color=color_speedup, marker="o", linewidth=2.0)
        ax2.set_title("Query speedup vs granularity")
        ax2.set_ylabel("Speedup (full-scan / pushdown)")
        ax2.set_xticks(positions, labels)
        ax2.grid(axis="y", color="#D7DCE2", linewidth=0.7, zorder=0)

        for axis in axes:
            axis.set_xlabel("Chunk boundary granularity")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.set_axisbelow(True)
            for tick in axis.get_xticklabels():
                tick.set_rotation(10)

        figure.suptitle(
            "Chunk-Granularity Tradeoff — Combined Date-Range + Price-Band Query",
            fontsize=14.5,
            fontweight="semibold",
        )

        figure.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
            metadata={
                "Title": "Chunk-Granularity Tradeoff",
                "Description": (
                    "Chunk count, pruning precision, and query speedup for a "
                    "combined date-range + price-band query across a few "
                    "ChunkBoundary granularities."
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
