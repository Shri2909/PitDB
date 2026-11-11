"""Generate every available chart from benchmark results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks/results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "charts/output"

from benchmarks.reporting.provenance import file_sha256
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


def _load_json(path: Path, fingerprints: dict[str, str] | None = None) -> JsonObject:
    try:
        contents = path.read_bytes()
        payload = json.loads(contents)
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark result is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"benchmark result must contain a JSON object: {path}")
    if fingerprints is not None:
        fingerprints[path.name] = hashlib.sha256(contents).hexdigest()
    return payload


def _load_all_results(
    results_dir: Path, fingerprints: dict[str, str] | None = None
) -> dict[str, JsonObject]:
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"benchmark results directory does not exist: {results_dir}"
        )

    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        raise FileNotFoundError(f"no benchmark JSON files were found in {results_dir}")

    # Load everything up front so stale optional results fail loudly.
    loaded: dict[str, JsonObject] = {}
    for path in result_files:
        if path.stem in loaded:
            raise ValueError(f"duplicate benchmark result name: {path.stem}")
        loaded[path.stem] = _load_json(path, fingerprints)
    validate_result_manifests(loaded)
    return loaded


def validate_result_manifests(results: Mapping[str, JsonObject]) -> str:
    """Require one complete, source-identical run before creating any chart."""
    identities = set()
    for name, payload in results.items():
        manifest = payload.get("manifest")
        if not isinstance(manifest, Mapping) or manifest.get("complete") is not True:
            raise ValueError(
                f"{name}: missing or incomplete run manifest; rerun benchmarks"
            )
        source = manifest.get("source")
        run_id = manifest.get("run_id")
        if not isinstance(source, Mapping) or not source.get("sha256") or not run_id:
            raise ValueError(f"{name}: manifest lacks source hash or run identity")
        identities.add((run_id, source["sha256"]))
    if len(identities) != 1:
        raise ValueError(
            (
                "benchmark results mix different runs or source snapshots; rerun "
                "as one suite"
            )
        )
    return next(iter(identities))[0]


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


_CHARTS = (
    (
        "queries",
        "Graph 1 - speedup vs selectivity",
        generate_graph_1,
        "01_speedup_vs_selectivity.png",
    ),
    ("queries", "Graph 2 - query latency", generate_graph_2, "02_latency_q1_q5.png"),
    ("queries", "Graph 3 - chunks skipped", generate_graph_3, "03_chunks_skipped.png"),
    ("memory", "Graph 4 - memory", generate_graph_4, "04_memory.png"),
    ("scale_sweep", "Graph 5 - scale sweep", generate_graph_5, "05_scale_sweep.png"),
    (
        "bitemporal_sweep",
        "Graph 6 - bitemporal sweep",
        generate_graph_6,
        "06_bitemporal_sweep.png",
    ),
    (
        "chunk_granularity_sweep",
        "Graph 7 - chunk granularity sweep",
        generate_graph_7,
        "07_chunk_granularity.png",
    ),
)


def generate_all(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Publish charts for one consistent run once every selected chart succeeds."""
    results_path = _resolve_project_path(results_dir)
    output_path = _resolve_project_path(output_dir)
    input_hashes: dict[str, str] = {}
    results = _load_all_results(results_path, input_hashes)
    _require_result(results, "queries", results_path)
    run_id = validate_result_manifests(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}
    with TemporaryDirectory(
        prefix=".pitdb-charts-", dir=output_path.parent
    ) as temporary:
        stage = Path(temporary)
        selected = []
        for result_name, chart_name, function, filename in _CHARTS:
            if result_name not in results:
                print(
                    f"Skipped {chart_name}: {result_name}.json unavailable in this run"
                )
                continue
            _generate_required(
                chart_name, function, results[result_name], stage / filename, generated
            )
            selected.append(filename)
        if input_hashes != {
            path.name: file_sha256(path) for path in sorted(results_path.glob("*.json"))
        }:
            raise ValueError("benchmark result files changed during chart generation")
        artifact_manifest = {
            "run_id": run_id,
            "input_sha256": input_hashes,
            "chart_sha256": {
                path.name: file_sha256(path) for path in generated.values()
            },
            "selected_result_families": sorted(
                {item[0] for item in _CHARTS if item[0] in results}
            ),
            "missing_result_families": sorted(
                {item[0] for item in _CHARTS if item[0] not in results}
            ),
            "selected_charts_complete": True,
            "full_suite_charts_complete": len(selected) == len(_CHARTS),
        }
        output_path.mkdir(parents=True, exist_ok=True)
        # The manifest is replaced last and binds all published PNG bytes.
        for filename in selected:
            (stage / filename).replace(output_path / filename)
        for _, _, _, filename in _CHARTS:
            if filename not in selected:
                (output_path / filename).unlink(missing_ok=True)
        manifest_stage = stage / "manifest.json"
        manifest_stage.write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_stage.replace(output_path / "manifest.json")
    published = {name: output_path / path.name for name, path in generated.items()}
    print(f"Generated {len(published)} chart(s) in {output_path}")
    return published


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
