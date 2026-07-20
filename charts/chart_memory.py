"""Generate Graph 4: retained storage and peak query memory usage."""

from __future__ import annotations

import argparse
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "benchmarks/results/memory.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "charts/output/04_memory.png"

_BYTES_PER_MEGABYTE = 1_000_000.0
# raw_uncompressed_bytes and chunk_store_bytes are NOT an apples-to-apples
# compression comparison: the former is pandas' own memory_usage(deep=True)
# on the source DataFrame (all 7 columns, including the object-dtype symbol
# column and index overhead); the latter is a full Python object-graph walk
# of the whole ChunkStore (every datetime object, every dataclass, every
# dict/list container -- overhead that has nothing to do with how much
# actual data is stored). column_payload_bytes -- the packed float64 column
# bytes ChunkStore actually persists -- is the fair comparison against raw
# data; it was already computed by bench_memory.py but never charted here.
# Labels spell out the differing accounting methods so this chart can't be
# misread as a compression-ratio claim (that claim lives in
# src/pipeline.py's _compression_ratio, a genuinely apples-to-apples metric
# unit-tested in tests/test_end_to_end.py -- unrelated to this chart).
_METRICS = (
    ("raw_uncompressed_bytes", "Raw DataFrame\n(pandas, deep)"),
    ("column_payload_bytes", "Column payload\n(packed floats)"),
    ("chunk_store_bytes", "Chunk store\n(Python object graph)"),
    ("q1_peak_rss_bytes", "Peak during\nQ1 query"),
)


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"memory benchmark results do not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"memory benchmark results are not valid JSON: {path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("memory benchmark results must be a JSON object")
    return payload


def _memory_metrics(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept memory.json directly or the object returned by run_all()."""

    if "memory" in payload:
        memory = payload["memory"]
        if memory is None:
            raise ValueError(
                "memory benchmarks were not run; invoke run_all with "
                "include_memory=True or --memory"
            )
        if not isinstance(memory, Mapping):
            raise ValueError("memory must be an object or null")
        return memory
    return payload


def _require_bytes(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a numeric byte count")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or not number.is_integer():
        raise ValueError(f"{field} must be a non-negative integer byte count")
    return int(number)


def _extract_values(
    payload: Mapping[str, Any],
) -> tuple[list[str], list[float]]:
    metrics = _memory_metrics(payload)
    labels: list[str] = []
    values_mb: list[float] = []

    for field, label in _METRICS:
        if field not in metrics:
            raise ValueError(f"memory benchmark results are missing {field}")
        value_bytes = _require_bytes(metrics[field], field)
        labels.append(label)
        values_mb.append(value_bytes / _BYTES_PER_MEGABYTE)

    baseline = metrics.get("q1_baseline_rss_bytes")
    if baseline is not None:
        baseline_bytes = _require_bytes(baseline, "q1_baseline_rss_bytes")
        peak_bytes = _require_bytes(metrics["q1_peak_rss_bytes"], "q1_peak_rss_bytes")
        if peak_bytes < baseline_bytes:
            raise ValueError("q1_peak_rss_bytes cannot be below its RSS baseline")

    if max(values_mb, default=0.0) <= 0.0:
        raise ValueError("at least one measured memory value must be positive")
    return labels, values_mb


def _value_label(value_mb: float) -> str:
    if value_mb >= 100.0:
        return f"{value_mb:,.0f} MB"
    if value_mb >= 10.0:
        return f"{value_mb:,.1f} MB"
    return f"{value_mb:,.2f} MB"


def generate_chart(
    results: Mapping[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create and save the raw/store/query memory comparison."""

    if isinstance(results, Mapping):
        payload = results
    elif isinstance(results, (str, Path)):
        payload = _load_json(Path(results).expanduser().resolve(strict=False))
    else:
        raise TypeError("results must be a mapping or a JSON file path")

    labels, values = _extract_values(payload)

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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
    colors = ["#7B8492", "#3F9142", "#2F6FA3", "#E07A2D"]

    with plt.rc_context(rc):
        figure, axis = plt.subplots(figsize=(9.0, 5.4), constrained_layout=True)
        positions = list(range(len(values)))
        bars = axis.bar(
            positions,
            values,
            width=0.58,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        maximum = max(values)
        headroom = max(1.0, maximum * 0.18)
        axis.set_ylim(0.0, maximum + headroom)
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Memory (MB)")
        axis.set_title(
            "Memory Footprint: Raw Data, Column Payload, Chunk Store, and Query Peak",
            pad=16,
        )
        axis.grid(
            axis="y",
            color="#D7DCE2",
            linewidth=0.7,
            alpha=0.8,
            zorder=0,
        )
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#555555")
        axis.spines["bottom"].set_color("#555555")
        axis.tick_params(axis="x", length=0, pad=8)
        axis.tick_params(axis="y", width=0.8, color="#555555")

        label_offset = max(0.25, maximum * 0.02)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + label_offset,
                _value_label(value),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
                color="#222222",
                clip_on=False,
            )

        figure.savefig(
            destination,
            dpi=300,
            bbox_inches="tight",
            metadata={
                "Title": (
                    "Memory Footprint: Raw Data, Column Payload, Chunk Store, "
                    "and Query Peak"
                ),
                "Description": (
                    "Bar chart comparing retained raw-DataFrame memory "
                    "(pandas), the actual packed column-payload bytes "
                    "ChunkStore persists, the chunk store's full Python "
                    "object-graph footprint, and absolute peak process RSS "
                    "observed while executing Q1 via the pushdown-query "
                    "path. Raw DataFrame and chunk store use different "
                    "accounting methods and are not a compression-ratio "
                    "comparison; column payload vs. raw DataFrame is the "
                    "fair one."
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
        help="memory.json generated by benchmarks/run_all.py --memory",
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
