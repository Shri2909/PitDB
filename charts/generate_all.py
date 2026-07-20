"""Generate every available chart from benchmark results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks/results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "charts/output"

from charts.chart_bitemporal import generate_chart as generate_graph_6
from charts.chart_chunk_granularity import generate_chart as generate_graph_7
from charts.chart_chunks_skipped import generate_chart as generate_graph_3
from charts.chart_latency_bar import generate_chart as generate_graph_2
from charts.chart_memory import generate_chart as generate_graph_4
from charts.chart_scale_sweep import generate_chart as generate_graph_5
from charts.chart_speedup_vs_selectivity import generate_chart as generate_graph_1

JsonObject = Mapping[str, Any]
ChartFunction = Callable[[Any, str | Path], Path]


def _resolve_project_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve(strict=False)


def _load_json(path: Path) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark result is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark result must contain a JSON object: {path}")
    return payload


def _load_all_results(results_dir: Path) -> dict[str, JsonObject]:
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"benchmark results directory does not exist: {results_dir}"
        )

    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        raise FileNotFoundError(f"no benchmark JSON files were found in {results_dir}")

    # Read every result file up front. This catches stale or malformed optional
    # results instead of silently generating a partial chart set from bad data.
    loaded: dict[str, JsonObject] = {}
    for path in result_files:
        if path.stem in loaded:
            raise ValueError(f"duplicate benchmark result name: {path.stem}")
        loaded[path.stem] = _load_json(path)
    return loaded


def _require_result(
    results: Mapping[str, JsonObject], name: str, results_dir: Path
) -> JsonObject:
    payload = results.get(name)
    if payload is None:
        raise FileNotFoundError(
            f"required P0 benchmark result is missing: {results_dir / (name + '.json')}"
        )
    return payload


def _generate_required(
    name: str,
    function: ChartFunction,
    data: Any,
    destination: Path,
    generated: dict[str, Path],
) -> None:
    output = function(data, destination).resolve(strict=False)
    generated[name] = output
    print(f"Generated {name}: {output}")


def _generate_optional(
    name: str,
    function: ChartFunction,
    data: Any,
    destination: Path,
    generated: dict[str, Path],
) -> None:
    try:
        output = function(data, destination).resolve(strict=False)
    except (FileNotFoundError, ImportError, ValueError) as exc:
        # P1/P2 benchmark inputs are optional. Missing data is reported clearly;
        # malformed required P0 results and unexpected programming errors still
        # fail the command rather than being hidden.
        print(f"Skipped {name}: {exc}")
        return
    generated[name] = output
    print(f"Generated {name}: {output}")


def generate_all(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Generate all P0 charts and every P1/P2 chart with available data."""

    results_path = _resolve_project_path(results_dir)
    output_path = _resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = _load_all_results(results_path)

    queries = _require_result(results, "queries", results_path)
    generated: dict[str, Path] = {}

    # P0 charts are part of the primary evaluation and therefore required.
    _generate_required(
        "Graph 1 - speedup vs selectivity",
        generate_graph_1,
        queries,
        output_path / "01_speedup_vs_selectivity.png",
        generated,
    )
    _generate_required(
        "Graph 2 - query latency",
        generate_graph_2,
        queries,
        output_path / "02_latency_q1_q5.png",
        generated,
    )
    _generate_required(
        "Graph 3 - chunks skipped",
        generate_graph_3,
        queries,
        output_path / "03_chunks_skipped.png",
        generated,
    )

    memory = results.get("memory")
    if memory is None:
        print(
            f"Skipped Graph 4 - memory: {results_path / 'memory.json'} is unavailable"
        )
    else:
        _generate_optional(
            "Graph 4 - memory",
            generate_graph_4,
            memory,
            output_path / "04_memory.png",
            generated,
        )

    scale_sweep = results.get("scale_sweep")
    if scale_sweep is None:
        print(
            "Skipped Graph 5 - scale sweep: "
            f"{results_path / 'scale_sweep.json'} is unavailable"
        )
    else:
        _generate_optional(
            "Graph 5 - scale sweep",
            generate_graph_5,
            scale_sweep,
            output_path / "05_scale_sweep.png",
            generated,
        )

    bitemporal_sweep = results.get("bitemporal_sweep")
    if bitemporal_sweep is None:
        print(
            "Skipped Graph 6 - bitemporal sweep: "
            f"{results_path / 'bitemporal_sweep.json'} is unavailable"
        )
    else:
        _generate_optional(
            "Graph 6 - bitemporal sweep",
            generate_graph_6,
            bitemporal_sweep,
            output_path / "06_bitemporal_sweep.png",
            generated,
        )

    chunk_granularity_sweep = results.get("chunk_granularity_sweep")
    if chunk_granularity_sweep is None:
        print(
            "Skipped Graph 7 - chunk granularity sweep: "
            f"{results_path / 'chunk_granularity_sweep.json'} is unavailable"
        )
    else:
        _generate_optional(
            "Graph 7 - chunk granularity sweep",
            generate_graph_7,
            chunk_granularity_sweep,
            output_path / "07_chunk_granularity.png",
            generated,
        )

    print(f"Generated {len(generated)} chart(s) in {output_path}")
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory containing benchmark JSON results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for generated PNG charts",
    )
    args = parser.parse_args(argv)
    generate_all(args.results_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_all", "main"]
