"""Run all benchmark producers under one identity and export consistent charts.

Correctness failures are collected so the summary can show every benchmark.
Charts are suppressed if a benchmark fails or is incomplete. Chart failures
are reported and return a nonzero suite status; prior artifacts are never
used to fill a failed producer's output.
"""

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
from benchmarks.bench_chunk_granularity import DEFAULT_GRANULARITY_KEY
from benchmarks.report import format_pct, format_x, render_box_table
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit, hardware_snapshot
from benchmarks.reporting.provenance import RUN_ID_ENV
from charts.generate_all import generate_all as generate_all_charts

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
CHARTS_DIR = PROJECT_ROOT / "charts" / "output"
LOG_PATH = RESULTS_DIR / "suite_run.log"


class _Tee(io.TextIOBase):
    """Write to the real stdout and an in-memory buffer at the same time.

    Exists so the suite runner can (a) count how many ``[WARNING]``/
    ``[ERROR]`` lines the run produced and (b) save a complete plain-text
    transcript to disk, without changing what the user sees on the
    terminal -- every line still reaches the real stdout immediately, in
    the same order, as it always did.
    """

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
    # platform.processor() is commonly empty on Linux -- fall back to
    # platform.machine() (the architecture, e.g. "x86_64") rather than
    # printing a blank field. Neither this project nor its pinned
    # dependencies include a CPU-model-name library (e.g. py-cpuinfo), so
    # a specific model name is not reliably available without adding one.
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
    """Fixed queries/selectivity sweep + memory footprint, via run_all.run_all.

    Both share one existing entry point and therefore one correctness gate
    outcome: a failure here can only originate from Benchmark 1's query
    correctness check (memory has no correctness gate of its own), so a
    caught failure marks both rows FAILED -- Benchmark 2 never got to run.
    """

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
        raw_bytes = memory["raw_numeric_payload_bytes"]
        packed_bytes = memory["column_payload_bytes"]
        reduction = 1.0 - (packed_bytes / raw_bytes if raw_bytes else 0.0)
        row_2 = [
            "2",
            "Memory footprint",
            "PASSED",
            f"Numeric-buffer reduction: {format_pct(reduction)}",
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
    as_of_speedup = None
    for entry in reversed(points):
        as_of_post = entry["scenarios"].get("as_of_post_compaction")
        if as_of_post is not None:
            as_of_speedup = as_of_post["query_speedup"]
            break
    primary = (
        f"AS OF speedup: {format_x(as_of_speedup)}"
        if as_of_speedup is not None
        else "AS OF speedup: N/A (no corrections)"
    )
    return ["4", "Corrections and compaction", "PASSED", primary]


def _run_benchmark_5() -> list[str]:
    try:
        bench_chunk_granularity.main()
    except AssertionError:
        return ["5", "Chunk-granularity trade-off", "FAILED", "SUPPRESSED"]

    payload = _load_json(RESULTS_DIR / "chunk_granularity_sweep.json")
    points = (payload or {}).get("points") or []
    default_point = next(
        (point for point in points if point.get("key") == DEFAULT_GRANULARITY_KEY),
        None,
    )
    primary = f"Default: {default_point['window']}" if default_point else "N/A"
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
    """Run all five benchmarks, generate charts, and print one consolidated
    suite summary. Returns 0 only when all benchmark gates and chart export
    succeed; failures or incomplete runs return 1.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)

    # One run identity links all producers; restore the caller's environment.
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
        # Suite-wide count of benchmarks whose correctness gate raised --
        # not a count of individual queries/scenarios within a benchmark,
        # which this suite-level summary does not track separately.
        correctness_failures = failed_count
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
                    ("Correctness failures", str(correctness_failures)),
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
        # 80-column, matching pitdb_title's own bar width above -- not
        # console.rule() (96 columns), which belongs to the older style
        # bench_scale.py still uses. See console.py's module docstring.
        print("=" * 80)

        transcript = tee.buffer.getvalue()

    LOG_PATH.write_text(transcript, encoding="utf-8")

    return 1 if suite_failed else 0


def main() -> int:
    return run_suite()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_suite"]
