"""Run all benchmark producers under one run identity and export charts."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import bench_bitemporal, bench_chunk_granularity, bench_scale, run_all
from benchmarks.report import format_bytes, format_x, render_box_table
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit, hardware_snapshot
from benchmarks.reporting.provenance import RUN_ID_ENV
from charts.generate_all import generate_all as generate_all_charts

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
CHARTS_DIR = PROJECT_ROOT / "charts" / "output"
LOG_PATH = RESULTS_DIR / "suite_run.log"


class _Tee(io.TextIOBase):
    """Write to the real stdout and an in-memory buffer at the same time."""

    def __init__(self, real_stdout: Any) -> None:
        self._real_stdout = real_stdout
        self.buffer = io.StringIO()

    def write(self, text: str) -> int:
        self.buffer.write(text)
        return self._real_stdout.write(text)

    def flush(self) -> None:
        self._real_stdout.flush()


@contextlib.contextmanager
def _tee_stdout():
    tee = _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        yield tee


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cpu_description(hardware: dict[str, Any]) -> str:
    # platform.processor() is commonly empty on Linux.
    base = hardware.get("processor") or hardware.get("machine") or "unknown"
    logical = hardware.get("logical_cpu_count")
    physical = hardware.get("physical_cpu_count")
    if logical is not None and physical is not None:
        return f"{base} ({logical} logical / {physical} physical cores)"
    return base


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)}m {remainder:.1f}s"


def _run_benchmarks_1_and_2() -> tuple[list[str], list[str]]:
    """Run benchmarks 1 and 2 via ``run_all``; a correctness failure fails both."""

    try:
        result = run_all.run_all(include_memory=True)
    except AssertionError:
        return (
            ["1", "Fixed queries and selectivity", "FAILED", "SUPPRESSED"],
            ["2", "Memory footprint", "FAILED", "SUPPRESSED"],
        )

    speedups = [
        metrics["speedup"] for metrics in result["queries"]["fixed_queries"].values()
    ]
    row_1 = [
        "1",
        "Fixed queries and selectivity",
        "PASSED",
        f"Max speedup: {format_x(max(speedups))}" if speedups else "N/A",
    ]

    memory = result.get("memory")
    if memory is not None:
        q1 = memory["traced_query_peaks"]["Q1"]["engines"]
        row_2 = [
            "2",
            "Memory footprint",
            "PASSED",
            "Q1 traced peak: "
            f"{format_bytes(q1['pushdown_query']['median_peak_bytes'])} pushdown vs "
            f"{format_bytes(q1['full_scan']['median_peak_bytes'])} row-wise "
            "(storage is uncompressed; no reduction claimed)",
        ]
    else:
        row_2 = ["2", "Memory footprint", "N/A", "N/A (not run)"]

    return row_1, row_2


def _run_benchmark_3() -> list[str]:
    try:
        bench_scale.main()
    except AssertionError:
        return ["3", "Scale sweep", "FAILED", "SUPPRESSED"]

    payload = _load_json(RESULTS_DIR / "scale_sweep.json")
    points = (payload or {}).get("points") or []
    if not points:
        return ["3", "Scale sweep", "PASSED", "N/A"]
    largest = max(points, key=lambda point: point["total_rows"])
    return [
        "3",
        "Scale sweep",
        "PASSED",
        f"Max scale: {largest['total_rows']:,} rows",
    ]


def _run_benchmark_4() -> list[str]:
    try:
        bench_bitemporal.main()
    except AssertionError:
        return ["4", "Corrections and compaction", "FAILED", "SUPPRESSED"]

    payload = _load_json(RESULTS_DIR / "bitemporal_sweep.json")
    points = (payload or {}).get("points") or []
    primary = "AS OF: N/A (no corrections)"
    for entry in reversed(points):
        sweep = entry.get("as_of_sweep") or []
        if not sweep:
            continue
        speedups = [point["scenario"]["query_speedup"] for point in sweep]
        mid = next((p for p in sweep if p["is_primary"]), sweep[0])
        primary = (
            f"AS OF vs row-wise at {entry['label']} corrected: "
            f"{format_x(min(speedups))}-{format_x(max(speedups))} across "
            f"{len(sweep)} cutoffs (mid-history cutoff "
            f"{format_x(mid['scenario']['query_speedup'])})"
        )
        break
    return ["4", "Corrections and compaction", "PASSED", primary]


def _run_benchmark_5() -> list[str]:
    try:
        bench_chunk_granularity.main()
    except AssertionError:
        return ["5", "Chunk-granularity trade-off", "FAILED", "SUPPRESSED"]

    payload = _load_json(RESULTS_DIR / "chunk_granularity_sweep.json")
    findings = (payload or {}).get("findings")
    if findings:
        primary = (
            f"Lowest pushdown latency: {findings['lowest_pushdown_label']} "
            f"(engine default: {findings['configured_default_label']}; "
            "single synthetic grid, no general default claimed)"
        )
    else:
        primary = "N/A"
    return ["5", "Chunk-granularity trade-off", "PASSED", primary]


@contextlib.contextmanager
def _suite_run_identity():
    previous = os.environ.get(RUN_ID_ENV)
    os.environ[RUN_ID_ENV] = uuid.uuid4().hex
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(RUN_ID_ENV, None)
        else:
            os.environ[RUN_ID_ENV] = previous


def run_suite() -> int:
    """Run all five benchmarks and charts; return 0 only if everything succeeded."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)

    with _suite_run_identity(), _tee_stdout() as tee:
        rows: list[list[str]] = []
        row_1, row_2 = _run_benchmarks_1_and_2()
        rows.append(row_1)
        rows.append(row_2)
        rows.append(_run_benchmark_3())
        rows.append(_run_benchmark_4())
        rows.append(_run_benchmark_5())

        charts_generated: dict[str, Path] | None = None
        if all(row[2] == "PASSED" for row in rows):
            try:
                charts_generated = generate_all_charts(
                    results_dir=RESULTS_DIR, output_dir=CHARTS_DIR
                )
            except (FileNotFoundError, ImportError, ValueError) as exc:
                print(console.phase("WARNING", f"Chart generation failed: {exc}"))
        else:
            print(
                console.phase(
                    "WARNING",
                    (
                        "Charts suppressed: benchmark run failed or is incomplete; "
                        "prior "
                        "artifacts were not used."
                    ),
                )
            )

        finished_at = datetime.now(timezone.utc)
        hardware = hardware_snapshot()

        passed_count = sum(1 for row in rows if row[2] == "PASSED")
        failed_count = sum(1 for row in rows if row[2] == "FAILED")
        suite_failed = passed_count != len(rows) or not charts_generated

        print()
        print(console.pitdb_title("PITDB BENCHMARK SUITE SUMMARY"))
        print()
        print(
            console.titled_block(
                "Environment",
                [
                    ("Git commit", git_commit()),
                    ("Python", hardware["python_version"]),
                    (
                        "Operating system",
                        f"{hardware['system']} {hardware['release']} "
                        f"({hardware['machine']})",
                    ),
                    ("CPU", _cpu_description(hardware)),
                    (
                        "Random seed",
                        "seed=1 (synthetic benchmarks 3-5, data generation); "
                        "N/A (real market data, benchmarks 1-2)",
                    ),
                    (
                        "Run started",
                        started_at.isoformat(timespec="seconds"),
                    ),
                    (
                        "Run completed",
                        finished_at.isoformat(timespec="seconds"),
                    ),
                    (
                        "Total runtime",
                        _format_duration((finished_at - started_at).total_seconds()),
                    ),
                ],
            )
        )
        print()
        print(
            render_box_table(
                ["No.", "Benchmark", "Correctness", "Primary result"], rows
            )
        )
        print()
        print(
            console.titled_block(
                "Validation",
                [
                    ("Benchmarks passed", f"{passed_count}/{len(rows)}"),
                    ("Correctness failures", str(failed_count)),
                    (
                        "Warnings",
                        str(tee.buffer.getvalue().count("[WARNING]")),
                    ),
                    (
                        "All differential equality gates passed",
                        "YES" if passed_count == len(rows) else "NO",
                    ),
                ],
            )
        )
        print()
        print(
            console.titled_block(
                "Generated artifacts",
                [
                    ("CSV results", "N/A (this suite exports JSON only)"),
                    ("JSON results", str(RESULTS_DIR)),
                    (
                        "Charts",
                        str(CHARTS_DIR)
                        if charts_generated
                        else "N/A (charts suppressed or failed; see warning)",
                    ),
                    ("Plain-text log", str(LOG_PATH)),
                ],
            )
        )
        print()
        print("Final status")
        overall_status = (
            "BENCHMARK SUITE COMPLETED WITH FAILURES"
            if suite_failed
            else "ALL BENCHMARKS COMPLETED SUCCESSFULLY"
        )
        print(f"  {overall_status}")
        print()
        print("=" * 80)

        transcript = tee.buffer.getvalue()

    LOG_PATH.write_text(transcript, encoding="utf-8")

    return 1 if suite_failed else 0


def main() -> int:
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_suite"]
