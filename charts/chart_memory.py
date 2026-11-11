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
# Raw and store payloads are equal by construction, so they share one bar.
_RETAINED = (
    (
        "column_payload_bytes",
        "Numeric payload\n(float64 x5; source ==\nstore, uncompressed)",
    ),
    ("raw_uncompressed_bytes", "Full source DataFrame\n(pandas, deep)"),
    ("chunk_store_bytes", "Full store\n(object-graph estimate)"),
)
_ENGINES = (
    ("full_scan", "Row-wise full scan", "#7B8492"),
    ("vectorized_full_scan", "Vectorized (no pruning)", "#E0A83A"),
    ("pushdown_query", "Zone-map pushdown", "#2F6FA3"),
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
) -> tuple[list[str], list[float], Mapping[str, Any]]:
    metrics = _memory_metrics(payload)
    labels: list[str] = []
    values_mb: list[float] = []
    for field, label in _RETAINED:
        if field not in metrics:
            raise ValueError(f"memory benchmark results are missing {field}")
        labels.append(label)
        values_mb.append(_require_bytes(metrics[field], field) / _BYTES_PER_MEGABYTE)
    if max(values_mb, default=0.0) <= 0.0:
        raise ValueError("at least one measured memory value must be positive")
    traced = metrics.get("traced_query_peaks")
    if not isinstance(traced, Mapping) or not traced:
        raise ValueError(
            "memory results lack traced_query_peaks; rerun benchmarks/run_all.py "
            "--memory with the current code"
        )
    return labels, values_mb, traced


def _value_label(value_mb: float) -> str:
    if value_mb >= 100.0:
        return f"{value_mb:,.0f} MB"
    if value_mb >= 10.0:
        return f"{value_mb:,.1f} MB"
    return f"{value_mb:,.2f} MB"


def _kb_label(value_bytes: float) -> str:
    if value_bytes >= 1_000_000:
        return f"{value_bytes / 1e6:.2f} MB"
    return f"{value_bytes / 1e3:.0f} kB"


def generate_chart(
    results: Mapping[str, Any] | str | Path = DEFAULT_RESULTS_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """Create the retained-size and per-query peak-allocation panels."""
    if isinstance(results, Mapping):
        payload = results
    elif isinstance(results, (str, Path)):
        payload = _load_json(Path(results).expanduser().resolve(strict=False))
    else:
        raise TypeError("results must be a mapping or a JSON file path")

    labels, values, traced = _extract_values(payload)

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

    with plt.rc_context(rc):
        figure, (left, right) = plt.subplots(
            1,
            2,
            figsize=(15.0, 5.8),
            gridspec_kw={"width_ratios": [1, 1.6]},
            constrained_layout=True,
        )

        positions = list(range(len(values)))
        bars = left.bar(
            positions,
            values,
            width=0.58,
            color=["#3F9142", "#8699A7", "#2F6FA3"],
            edgecolor="white",
            zorder=3,
        )
        left.set_ylim(0.0, max(values) * 1.18)
        left.set_xticks(positions, labels, fontsize=9)
        left.set_ylabel("Retained size (MB)")
        left.set_title("Retained representations (different accounting methods)")
        for bar, value in zip(bars, values):
            left.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + max(values) * 0.02,
                _value_label(value),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="semibold",
            )

        names = list(traced)
        group = list(range(len(names)))
        width = 0.26
        for offset, (key, label, color) in zip((-1, 0, 1), _ENGINES):
            medians, lows, highs = [], [], []
            for name in names:
                engine = traced[name]["engines"][key]
                median = float(engine["median_peak_bytes"])
                medians.append(median)
                lows.append(max(0.0, median - float(engine["min_peak_bytes"])))
                highs.append(max(0.0, float(engine["max_peak_bytes"]) - median))
            xs = [g + offset * width for g in group]
            right.bar(
                xs,
                medians,
                width=width,
                yerr=[lows, highs],
                label=label,
                color=color,
                edgecolor="white",
                error_kw={"elinewidth": 1.0, "ecolor": "#3F444A", "capsize": 3},
                zorder=3,
            )
            for x, median in zip(xs, medians):
                right.text(
                    x,
                    median * 1.5,
                    _kb_label(median),
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    rotation=90,
                )
        right.set_yscale("log")
        top = max(
            float(traced[n]["engines"][k]["max_peak_bytes"])
            for n in names
            for k, _, _ in _ENGINES
        )
        bottom = min(
            float(traced[n]["engines"][k]["median_peak_bytes"])
            for n in names
            for k, _, _ in _ENGINES
        )
        right.set_ylim(max(1.0, bottom / 4.0), top * 12.0)
        right.set_xticks(
            group,
            [f"{n}\n{traced[n]['row_count']:,} row(s)" for n in names],
        )
        right.set_ylabel("Peak traced allocation per query (bytes, log scale)")
        right.set_title(
            "Per-query peak allocation (tracemalloc, median; whiskers min-max)"
        )
        right.legend(loc="upper left", ncols=3, frameon=False, fontsize=9.5)

        for axis in (left, right):
            axis.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8, zorder=0)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        figure.suptitle(
            "Memory: Retained Sizes and Per-Query Peak Allocation",
            fontsize=14.5,
            fontweight="semibold",
        )
        figure.text(
            0.01,
            -0.03,
            "Storage is uncompressed float64: no compression or storage reduction "
            "is measured or claimed.\nProcess RSS is omitted: its observed increases "
            f"are at the {payload_note(payload)} measurement noise floor. "
            "tracemalloc covers Python/NumPy-registered allocations only;\n"
            "same process, 0.2 MB dataset.",
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
                "Title": "Memory: Retained Sizes and Per-Query Peak Allocation",
                "Description": (
                    "Left: numeric payload (equal in source and store; "
                    "uncompressed), full DataFrame and store object-graph "
                    "estimates. Right: tracemalloc peak allocation per query "
                    "for row-wise, vectorized and pushdown paths (median of "
                    "interleaved repeats, min-max whiskers). No RSS and no "
                    "storage-reduction claim."
                ),
            },
        )
        plt.close(figure)

    return destination


def payload_note(payload: Mapping[str, Any]) -> str:
    metrics = _memory_metrics(payload)
    rss = metrics.get("rss_diagnostic")
    if isinstance(rss, Mapping) and "resolution_bytes" in rss:
        return f"{int(rss['resolution_bytes'])}-byte-resolution"
    return "page-resolution"


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
