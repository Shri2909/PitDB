"""Bitemporal correction-volume sweep.

Measures how the cost of transaction-time versioning grows as the fraction
of corrected rows increases relative to base data: ingest_correction
throughput, compact_corrections() throughput, isolated AS OF
pushdown-pruning cost, and end-to-end AS OF query latency (pushdown vs.
full-scan) -- each measured **both before and after** compact_corrections(),
so the actual effect of compaction is always visible directly in this
benchmark's own output, never requiring an ad hoc script to see it.

Mirrors benchmarks/bench_scale.py's conventions -- ddof=1 sample std,
stats.py's confidence_interval_95 normal approximation, and the same
full-scan/pushdown correctness gate (_assert_equivalent) every other
benchmark in this suite uses before reporting any speedup number. Also
mirrors bench_query.py's annotate_pairwise_significance: every pre/post
comparison is flagged as statistically distinguishable or not via
stats.py's intervals_overlap, rather than letting point estimates alone
imply an effect that might just be noise.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.bench_query import _assert_equivalent
from benchmarks.bench_scale import _time_calls
from benchmarks.config import NUM_TRIALS as DEFAULT_NUM_TRIALS
from benchmarks.report import format_ms, format_pct, format_x, render_box_table
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit, hardware_snapshot
from benchmarks.reporting.validation import assert_equivalent_and_report
from benchmarks.stats import (
    confidence_interval_95,
    intervals_overlap,
    trimmed_mean_and_std,
)
from benchmarks.synthetic_data import generate_synthetic_ohlcv
from src.baselines.full_scan import FullScanQueryEngine
from src.query.executor import QueryExecutor
from src.query.predicate import PredicateNode
from src.query.pushdown import evaluate_pushdown
from src.query.sql_parser import parse_where_clause
from src.store.chunk_boundary import MonthlyBoundary
from src.store.chunk_store import ChunkStore

NUM_SYMBOLS = 20
ROWS_PER_SYMBOL = 501  # matches bench_scale.py's Dataset-A-shaped row count
CORRECTION_RATES = [0.0, 0.01, 0.05, 0.20, 0.50]
QUERY_NUM_TRIALS = DEFAULT_NUM_TRIALS  # matches benchmarks/config.py's convention
QUERY_WARMUP = 2
# Drops the slowest 15% of query-latency trials before reporting mean/std/CI
# -- see trimmed_mean_and_std's docstring. Decompression-heavy queries in
# this benchmark are far more exposed to one-off OS/GC hiccups (multi-hundred
# ms per trial) than the microsecond-scale isolated pushdown-eval loop below,
# which is left on a plain mean.
LATENCY_TRIM_FRACTION = 0.15
PUSHDOWN_ISOLATED_TRIALS = 50


def apply_corrections(
    df: pd.DataFrame, correction_rate: float, seed: int
) -> list[dict[str, Any]]:
    """Select a deterministic fraction of rows and build correction events.

    Every selected (symbol, timestamp) row is corrected to a bumped close
    price, with a fixed transaction-time lag (one day after the dataset's
    last observed timestamp) -- every correction is uniformly "learned"
    well after every base row.
    """

    if not 0.0 <= correction_rate <= 1.0:
        raise ValueError("correction_rate must be within [0.0, 1.0]")

    num_corrections = int(round(len(df) * correction_rate))
    if num_corrections == 0:
        return []

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(df), size=num_corrections, replace=False)
    transaction_time = df["timestamp"].max() + pd.Timedelta(days=1)

    corrections: list[dict[str, Any]] = []
    for position in selected:
        row = df.iloc[position]
        values = np.array(
            [row.open, row.high, row.low, row.close * 1.05, row.volume],
            dtype=np.float64,
        )
        corrections.append(
            {
                "timestamp": row.timestamp,
                "symbol": row.symbol,
                "values": values,
                "transaction_time": transaction_time,
            }
        )
    return corrections


def measure_scenario(
    *,
    store: ChunkStore,
    predicate: PredicateNode,
    as_of_value: Any,
    sql: str,
    query_name: str,
) -> dict[str, Any]:
    """Measure isolated pushdown-eval cost and end-to-end query latency for
    one (AS OF value, SQL) scenario against the store's *current* state.

    Called twice per scenario (before and after compact_corrections()) so
    the caller can compare the two dicts this returns directly.
    """

    pushdown_times = np.empty(PUSHDOWN_ISOLATED_TRIALS, dtype=np.float64)
    for trial in range(PUSHDOWN_ISOLATED_TRIALS):
        t0 = time.perf_counter()
        evaluate_pushdown(predicate, store.labels, as_of=as_of_value, as_of_index=store)
        pushdown_times[trial] = time.perf_counter() - t0
    pushdown_result = evaluate_pushdown(
        predicate, store.labels, as_of=as_of_value, as_of_index=store
    )

    pushdown_engine = QueryExecutor(store)
    full_scan_engine = FullScanQueryEngine(store)

    pushdown_query_times, pushdown_query_result = _time_calls(
        lambda: pushdown_engine.execute(sql), QUERY_NUM_TRIALS, QUERY_WARMUP
    )
    full_scan_query_times, full_scan_query_result = _time_calls(
        lambda: full_scan_engine.execute(sql), QUERY_NUM_TRIALS, QUERY_WARMUP
    )
    # Same correctness gate every other benchmark in this suite uses: full
    # row/value/dtype equality between the naive and pushdown paths, now
    # with corrections and an AS OF cutoff in play, before any latency is
    # reported. quiet=True: re-rendered inside each correction level's own
    # "Correctness validation"/"Correctness revalidation" block once
    # printing starts (see _print_correction_level), not live here during
    # this still-silent computation phase.
    assert_equivalent_and_report(
        lambda: _assert_equivalent(
            full_scan_query_result, pushdown_query_result, query_name=query_name
        ),
        name=query_name,
        full_scan_row_count=full_scan_query_result.row_count,
        pushdown_row_count=pushdown_query_result.row_count,
        quiet=True,
    )

    # Trimmed, not plain, mean: latency outliers here are one-directional
    # (a GC pause or OS scheduling hiccup only ever makes a trial slower),
    # and a single such trial can otherwise blow out the reported CI far
    # wider than every other measurement's -- see LATENCY_TRIM_FRACTION.
    pushdown_mean, pushdown_std, pushdown_n = trimmed_mean_and_std(
        pushdown_query_times, LATENCY_TRIM_FRACTION
    )
    ci_low, ci_high = confidence_interval_95(pushdown_mean, pushdown_std, pushdown_n)
    full_scan_mean, _, _ = trimmed_mean_and_std(
        full_scan_query_times, LATENCY_TRIM_FRACTION
    )

    return {
        "total_chunks": store.total_chunks,
        "pushdown_eval_ns_per_label": (
            float(np.mean(pushdown_times)) * 1e9 / store.total_chunks
        ),
        "chunks_skipped_ratio": pushdown_result.chunks_skipped_ratio,
        "candidate_chunks": pushdown_result.candidate_count,
        "query_pushdown_mean_ms": pushdown_mean * 1000.0,
        "query_pushdown_ci95_low_ms": ci_low * 1000.0,
        "query_pushdown_ci95_high_ms": ci_high * 1000.0,
        "query_full_scan_mean_ms": full_scan_mean * 1000.0,
        "query_speedup": full_scan_mean / pushdown_mean if pushdown_mean else 0.0,
        "row_count": pushdown_query_result.row_count,
    }


def _compare(
    label: str, before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Flag whether a before/after pushdown-latency delta is real or noise,
    the same way bench_query.py's annotate_pairwise_significance does.

    Computed for every correction level's JSON output regardless of
    whether the current terminal format prints it (it currently doesn't --
    see _print_correction_level); still available to anyone reading
    bitemporal_sweep.json directly.
    """

    overlap = intervals_overlap(
        before["query_pushdown_ci95_low_ms"],
        before["query_pushdown_ci95_high_ms"],
        after["query_pushdown_ci95_low_ms"],
        after["query_pushdown_ci95_high_ms"],
    )
    return {
        "comparison": label,
        "before_mean_ms": before["query_pushdown_mean_ms"],
        "after_mean_ms": after["query_pushdown_mean_ms"],
        "ci95_overlap": overlap,
        "distinguishable_from_noise": not overlap,
    }


def run_correction_rate(*, label: str, correction_rate: float) -> dict[str, Any]:
    """Compute one correction level's full before/after measurement set.

    Silent: nothing is printed here (see quiet=True inside
    measure_scenario) -- this function only computes and returns data, so
    the exact-format per-level block (_print_correction_level) can be
    printed as one complete, self-contained unit once every value it needs
    already exists.
    """

    df = generate_synthetic_ohlcv(
        num_symbols=NUM_SYMBOLS, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )
    store = ChunkStore(chunk_boundary=MonthlyBoundary())
    store.ingest_dataframe(df)
    base_chunks = store.total_chunks
    base_rows = store.total_rows

    corrections = apply_corrections(df, correction_rate, seed=2)

    t0 = time.perf_counter()
    for correction in corrections:
        store.ingest_correction(
            correction["timestamp"],
            correction["symbol"],
            correction["values"],
            transaction_time=correction["transaction_time"],
        )
    correction_seconds = time.perf_counter() - t0
    correction_rows_per_second = (
        len(corrections) / correction_seconds
        if corrections and correction_seconds > 0
        else 0.0
    )
    chunks_before_compaction = store.total_chunks

    where_clause = "timestamp >= '2023-01-01'"
    predicate = parse_where_clause(f"SELECT * FROM data WHERE {where_clause}")
    current_sql = f"SELECT * FROM data WHERE {where_clause}"
    as_of_value = df["timestamp"].max() if corrections else None
    as_of_sql = (
        f"SELECT * FROM data TIMESTAMP AS OF '{as_of_value.isoformat()}' "
        f"WHERE {where_clause}"
        if corrections
        else None
    )

    # Pre-compaction measurement, taken before compact_corrections() is ever
    # called, so the effect of compaction below is a real before/after
    # comparison -- not two already-compacted runs compared against each
    # other, which is not the same thing.
    current_pre = measure_scenario(
        store=store,
        predicate=predicate,
        as_of_value=None,
        sql=current_sql,
        query_name=f"bitemporal-{label}-current-pre",
    )

    as_of_pre = None
    if corrections:
        as_of_pre = measure_scenario(
            store=store,
            predicate=predicate,
            as_of_value=as_of_value,
            sql=as_of_sql,
            query_name=f"bitemporal-{label}-as_of-pre",
        )

    # Reflects the recommended production workflow: ingest_correction stays
    # cheap and never blocks (measured above), consolidation happens later
    # via this explicit, idempotent maintenance call. Timed directly, not
    # inferred.
    t0 = time.perf_counter()
    compacted = store.compact_corrections()
    compaction_seconds = time.perf_counter() - t0
    # Rows actually replayed through compact_corrections() -- every
    # correction row gets re-read once per compaction pass, regardless of
    # how many chunks that collapses into. 0 when there is nothing to
    # compact (guards divide-by-zero rather than reporting a fabricated
    # rate).
    compaction_rows_per_second = (
        len(corrections) / compaction_seconds
        if corrections and compaction_seconds > 0
        else 0.0
    )

    current_post = measure_scenario(
        store=store,
        predicate=predicate,
        as_of_value=None,
        sql=current_sql,
        query_name=f"bitemporal-{label}-current-post",
    )
    current_comparison = _compare(
        "current pre-vs-post compaction", current_pre, current_post
    )

    as_of_post = None
    as_of_comparison = None
    if corrections:
        as_of_post = measure_scenario(
            store=store,
            predicate=predicate,
            as_of_value=as_of_value,
            sql=as_of_sql,
            query_name=f"bitemporal-{label}-as_of-post",
        )
        as_of_comparison = _compare(
            "as_of pre-vs-post compaction", as_of_pre, as_of_post
        )

    entry = {
        "label": label,
        "correction_rate": correction_rate,
        "as_of_cutoff": as_of_value.isoformat() if as_of_value is not None else None,
        "base_rows": base_rows,
        "base_chunks": base_chunks,
        "num_corrections": len(corrections),
        "correction_chunks": chunks_before_compaction - base_chunks,
        "chunks_before_compaction": chunks_before_compaction,
        "correction_seconds": correction_seconds,
        "correction_rows_per_second": correction_rows_per_second,
        "compacted_chunks": compacted,
        "compaction_seconds": compaction_seconds,
        "compaction_rows_per_second": compaction_rows_per_second,
        "total_chunks": store.total_chunks,
        "correction_chunks_after_compaction": store.total_chunks - base_chunks,
        "scenarios": {
            "current_pre_compaction": current_pre,
            "current_post_compaction": current_post,
            **({"as_of_pre_compaction": as_of_pre} if as_of_pre else {}),
            **({"as_of_post_compaction": as_of_post} if as_of_post else {}),
        },
        "compaction_significance": {
            "current": current_comparison,
            **({"as_of": as_of_comparison} if as_of_comparison else {}),
        },
    }
    return entry


def _correctness_status(scenario: dict[str, Any] | None) -> str:
    """PASSED if the scenario was measured (and therefore already passed
    the correctness gate inside measure_scenario -- reaching this point at
    all requires that), else an explicit reason it wasn't measured."""

    return "PASSED" if scenario is not None else "N/A (no corrections exist yet)"


def _matrix_row(
    name: str, storage_state: str, scenario: dict[str, Any] | None
) -> list[str]:
    if scenario is None:
        return [name, storage_state, "N/A", "N/A", "N/A"]
    return [
        name,
        storage_state,
        format_ms(scenario["query_full_scan_mean_ms"]),
        format_ms(scenario["query_pushdown_mean_ms"]),
        format_x(scenario["query_speedup"]),
    ]


def _pruning_row(name: str, scenario: dict[str, Any] | None) -> list[str]:
    if scenario is None:
        return [name, "N/A", "N/A", "N/A"]
    pruned = scenario["total_chunks"] - scenario["candidate_chunks"]
    return [
        name,
        f"{scenario['total_chunks']:,}",
        f"{pruned:,}",
        format_pct(scenario["chunks_skipped_ratio"]),
    ]


def _print_correction_level(*, index: int, total: int, entry: dict[str, Any]) -> None:
    """Print one correction level's complete before/after block.

    Every value used here is already fully computed in ``entry`` (see
    ``run_correction_rate``) -- this function only formats and lays it out.
    """

    scenarios = entry["scenarios"]
    current_pre = scenarios["current_pre_compaction"]
    as_of_pre = scenarios.get("as_of_pre_compaction")
    current_post = scenarios["current_post_compaction"]
    as_of_post = scenarios.get("as_of_post_compaction")

    print()
    print(console.dash_section(f"[CORRECTION LEVEL {index}/{total}]"))

    print()
    print(
        console.titled_block(
            "Correction workload",
            [
                ("Corrections", f"{entry['num_corrections']:,}"),
                ("Corrected-row percentage", format_pct(entry["correction_rate"])),
                ("Ingestion time", f"{entry['correction_seconds'] * 1000:.3f} ms"),
                (
                    "Ingestion throughput",
                    f"{entry['correction_rows_per_second']:,.0f} corrections/second"
                    if entry["num_corrections"]
                    else "N/A (no corrections at this level)",
                ),
            ],
        )
    )
    print()
    print(
        console.titled_block(
            "Storage state before compaction",
            [
                ("Total chunks", f"{entry['chunks_before_compaction']:,}"),
                ("Base chunks", f"{entry['base_chunks']:,}"),
                ("Correction chunks", f"{entry['correction_chunks']:,}"),
            ],
        )
    )
    print()
    print(
        console.titled_block(
            "Correctness validation",
            [
                ("Current-state query", "PASSED"),
                ("Historical AS OF query", _correctness_status(as_of_pre)),
            ],
        )
    )
    print()
    print("Performance matrix")
    print()
    print(
        render_box_table(
            ["Query", "Storage state", "Full scan", "Pushdown", "Speedup"],
            [
                _matrix_row("Current state", "Before compaction", current_pre),
                _matrix_row("Historical AS OF", "Before compaction", as_of_pre),
            ],
        )
    )
    print()
    print("Pruning before compaction")
    print()
    print(
        render_box_table(
            ["Query", "Total chunks", "Chunks pruned", "Pruning rate"],
            [
                _pruning_row("Current state", current_pre),
                _pruning_row("Historical AS OF", as_of_pre),
            ],
        )
    )

    print()
    print(console.dash_section("[COMPACTION]"))
    print(
        console.field_block(
            [
                ("Input chunks", f"{entry['chunks_before_compaction']:,}"),
                ("Output chunks", f"{entry['total_chunks']:,}"),
                ("Chunks merged", f"{entry['compacted_chunks']:,}"),
                (
                    "Compaction runtime",
                    f"{entry['compaction_seconds'] * 1000:.3f} ms",
                ),
                (
                    "Processing throughput",
                    f"{entry['compaction_rows_per_second']:,.0f} rows/second"
                    if entry["compacted_chunks"]
                    else "N/A (nothing to compact)",
                ),
                ("Status", "COMPLETE"),
            ]
        )
    )
    print()
    print(
        console.titled_block(
            "Storage state after compaction",
            [
                ("Total chunks", f"{entry['total_chunks']:,}"),
                ("Base chunks", f"{entry['base_chunks']:,}"),
                (
                    "Correction chunks",
                    f"{entry['correction_chunks_after_compaction']:,}",
                ),
            ],
        )
    )
    print()
    print(
        console.titled_block(
            "Correctness revalidation",
            [
                ("Current-state query", "PASSED"),
                ("Historical AS OF query", _correctness_status(as_of_post)),
            ],
        )
    )
    print()
    print("Performance matrix")
    print()
    print(
        render_box_table(
            ["Query", "Storage state", "Full scan", "Pushdown", "Speedup"],
            [
                _matrix_row("Current state", "After compaction", current_post),
                _matrix_row("Historical AS OF", "After compaction", as_of_post),
            ],
        )
    )


def _sweep_summary_rows(results: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for entry in results:
        current_pre = entry["scenarios"]["current_pre_compaction"]["query_speedup"]
        current_post = entry["scenarios"]["current_post_compaction"]["query_speedup"]
        as_of_pre = entry["scenarios"].get("as_of_pre_compaction")
        as_of_post = entry["scenarios"].get("as_of_post_compaction")
        as_of_cell = (
            f"{format_x(as_of_pre['query_speedup'])} / "
            f"{format_x(as_of_post['query_speedup'])}"
            if as_of_pre and as_of_post
            else "-- / --"
        )
        rows.append(
            [
                f"{entry['num_corrections']:,}",
                format_pct(entry["correction_rate"]),
                f"{format_x(current_pre)} / {format_x(current_post)}",
                as_of_cell,
                f"{entry['compaction_seconds'] * 1000:.3f} ms",
            ]
        )
    return rows


def _current_state_trend(results: list[dict[str, Any]]) -> str:
    first, last = results[0], results[-1]
    first_speedup = first["scenarios"]["current_pre_compaction"]["query_speedup"]
    last_speedup = last["scenarios"]["current_pre_compaction"]["query_speedup"]
    direction = "grew" if last_speedup >= first_speedup else "fell"
    return (
        f"Pre-compaction speedup {direction} from {format_x(first_speedup)} at "
        f"{first['label']} corrected to {format_x(last_speedup)} at "
        f"{last['label']} corrected."
    )


def _as_of_trend(results: list[dict[str, Any]]) -> str:
    with_corrections = [entry for entry in results if entry["num_corrections"] > 0]
    if not with_corrections:
        return "No corrections were applied in this sweep; AS OF was never measured."
    first, last = with_corrections[0], with_corrections[-1]
    first_speedup = first["scenarios"]["as_of_pre_compaction"]["query_speedup"]
    last_speedup = last["scenarios"]["as_of_pre_compaction"]["query_speedup"]
    direction = "grew" if last_speedup >= first_speedup else "fell"
    return (
        f"Pre-compaction speedup {direction} from {format_x(first_speedup)} at "
        f"{first['label']} corrected to {format_x(last_speedup)} at "
        f"{last['label']} corrected."
    )


def _compaction_trend(results: list[dict[str, Any]]) -> str:
    compacted = [entry for entry in results if entry["compacted_chunks"] > 0]
    if not compacted:
        return "No correction chunks were compacted at any level in this sweep."
    first, last = compacted[0], compacted[-1]
    direction = (
        "grew" if last["compaction_seconds"] >= first["compaction_seconds"] else "fell"
    )
    return (
        f"Runtime {direction} from "
        f"{first['compaction_seconds'] * 1000:.3f} ms at {first['label']} "
        f"corrected to {last['compaction_seconds'] * 1000:.3f} ms at "
        f"{last['label']} corrected."
    )


def main() -> int:
    # Silent, like Benchmarks 1 and 2: every correction level is fully
    # computed (quiet=True inside measure_scenario) before anything prints,
    # so the header and Configuration block below can report real base-row
    # counts and the actual AS OF cutoff instead of placeholders.
    results = [
        run_correction_rate(label=f"{rate:.0%}", correction_rate=rate)
        for rate in CORRECTION_RATES
    ]

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "bitemporal_sweep.json"
    # "points" is byte-identical to what this file has always written;
    # "generated_at_utc"/"git_commit"/"hardware" are new, additive
    # provenance keys only -- see BENCHMARK_OUTPUT_AUDIT.md Section 9, item 1.
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hardware": hardware_snapshot(),
        "points": results,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        console.pitdb_header(
            index=4, total=5, title="Bitemporal Corrections and Compaction"
        )
    )
    print()
    print("Question")
    print(
        console.wrap_text(
            "How do correction volume and compaction affect current-state and "
            "historical AS OF query performance?"
        )
    )

    as_of_cutoffs = {
        entry["as_of_cutoff"] for entry in results if entry["as_of_cutoff"]
    }
    if len(as_of_cutoffs) == 1:
        as_of_cutoff_display = next(iter(as_of_cutoffs))
    elif as_of_cutoffs:
        as_of_cutoff_display = "varies by correction level"
    else:
        as_of_cutoff_display = "N/A (no corrections in this sweep)"

    print()
    print(
        console.titled_block(
            "Configuration",
            [
                ("Base rows", f"{results[0]['base_rows']:,}"),
                ("Correction levels", str(len(CORRECTION_RATES))),
                ("Historical AS OF cutoff", as_of_cutoff_display),
                ("Compaction policy", "monthly (same granularity as base store)"),
                ("Warmup runs", str(QUERY_WARMUP)),
                ("Measured runs", str(QUERY_NUM_TRIALS)),
            ],
        )
    )

    for index, entry in enumerate(results, start=1):
        _print_correction_level(index=index, total=len(results), entry=entry)

    print()
    print(console.dash_section("[CORRECTION SWEEP SUMMARY]"))
    print()
    print(
        render_box_table(
            [
                "Corrections",
                "Percent",
                "Current-state speedup (before/after)",
                "Historical AS OF speedup (before/after)",
                "Compaction runtime",
            ],
            _sweep_summary_rows(results),
        )
    )

    print()
    print(console.dash_section("[RESULT]"))
    print(
        console.field_block(
            [
                ("Correctness", "PASSED AT ALL CORRECTION LEVELS"),
                ("Current-state trend", _current_state_trend(results)),
                ("Historical AS OF trend", _as_of_trend(results)),
                ("Compaction trend", _compaction_trend(results)),
                ("Result artifact", str(output_path)),
            ]
        )
    )
    print()
    print(
        console.titled_paragraph(
            "Interpretation",
            "Historical AS OF queries can reject correction chunks whose "
            "transaction times occur after the requested cutoff.",
        )
    )
    print()
    print(
        console.titled_paragraph(
            "Limitation",
            "Performance conclusions apply only to the tested correction "
            "distributions and compaction policy.",
        )
    )
    print()
    print(console.pitdb_footer(index=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
