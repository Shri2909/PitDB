"""Bitemporal correction-volume sweep, measured before and after compaction."""

from __future__ import annotations

import bisect
import math
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

from benchmarks.config import NUM_TRIALS as DEFAULT_NUM_TRIALS
from benchmarks.measurement import measure_query_paths
from benchmarks.report import format_ms, format_pct, format_x, render_box_table
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit, hardware_snapshot
from benchmarks.reporting.provenance import (
    create_manifest,
    dataframe_fingerprint,
    write_result,
)
from benchmarks.stats import intervals_overlap
from benchmarks.synthetic_data import generate_synthetic_ohlcv
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
# Fractions of correction versions made visible by each swept AS OF cutoff.
VISIBLE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.0)
PRIMARY_VISIBLE_FRACTION = 0.50
MAX_VERSIONS_PER_KEY = 3
MAX_LAG_DAYS = 90
PUSHDOWN_ISOLATED_TRIALS = 50


def apply_corrections(
    df: pd.DataFrame, correction_rate: float, seed: int
) -> list[dict[str, Any]]:
    """Build correction events for a deterministic fraction of base rows.

    Each key gets 1-3 versions (60/30/10%), each learned 1-90 days after the last.
    """
    if not 0.0 <= correction_rate <= 1.0:
        raise ValueError("correction_rate must be within [0.0, 1.0]")

    num_keys = int(round(len(df) * correction_rate))
    if num_keys == 0:
        return []

    rng = np.random.default_rng(seed)
    selected = rng.choice(len(df), size=num_keys, replace=False)
    version_counts = rng.choice(
        np.arange(1, MAX_VERSIONS_PER_KEY + 1), size=num_keys, p=[0.6, 0.3, 0.1]
    )

    corrections: list[dict[str, Any]] = []
    for key_index, (position, versions) in enumerate(zip(selected, version_counts)):
        row = df.iloc[position]
        transaction_time = row.timestamp
        for version in range(1, int(versions) + 1):
            transaction_time = transaction_time + pd.Timedelta(
                days=int(rng.integers(1, MAX_LAG_DAYS + 1))
            )
            values = np.array(
                [
                    row.open,
                    row.high,
                    row.low,
                    row.close * (1.0 + 0.05 * version),
                    row.volume,
                ],
                dtype=np.float64,
            )
            corrections.append(
                {
                    "timestamp": row.timestamp,
                    "symbol": row.symbol,
                    "values": values,
                    "transaction_time": transaction_time,
                    "key_index": key_index,
                    "version": version,
                }
            )
    return corrections


def as_of_cutoffs(
    df: pd.DataFrame, corrections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Cutoffs making ~0/25/50/75/100% of versions visible, plus end of base data."""

    transaction_times = sorted(c["transaction_time"] for c in corrections)
    count = len(transaction_times)
    cutoffs: list[dict[str, Any]] = []
    for fraction in VISIBLE_FRACTIONS:
        if fraction == 0.0:
            cutoff = transaction_times[0] - pd.Timedelta(days=1)
            label = "0% of versions"
        else:
            cutoff = transaction_times[max(0, math.ceil(fraction * count) - 1)]
            label = f"{fraction:.0%} of versions"
        cutoffs.append(_cutoff_entry(cutoff, label, fraction, transaction_times))
    cutoffs.append(
        _cutoff_entry(
            df["timestamp"].max(), "end of base data", None, transaction_times
        )
    )
    return cutoffs


def _cutoff_entry(
    cutoff: pd.Timestamp,
    label: str,
    target_fraction: float | None,
    sorted_transaction_times: list[Any],
) -> dict[str, Any]:
    visible = bisect.bisect_right(sorted_transaction_times, cutoff)
    return {
        "label": label,
        "target_visible_fraction": target_fraction,
        "cutoff": cutoff,
        "visible_versions": visible,
        "visible_fraction": visible / len(sorted_transaction_times),
    }


def measure_scenario(
    *,
    store: ChunkStore,
    predicate: PredicateNode,
    as_of_value: Any,
    sql: str,
    query_name: str,
) -> dict[str, Any]:
    """Measure isolated pushdown-eval cost and query latency for one scenario."""
    pushdown_times = np.empty(PUSHDOWN_ISOLATED_TRIALS, dtype=np.float64)
    for trial in range(PUSHDOWN_ISOLATED_TRIALS):
        t0 = time.perf_counter()
        evaluate_pushdown(predicate, store.labels, as_of=as_of_value, as_of_index=store)
        pushdown_times[trial] = time.perf_counter() - t0
    pushdown_result = evaluate_pushdown(
        predicate, store.labels, as_of=as_of_value, as_of_index=store
    )

    timing_metrics, _query_result = measure_query_paths(
        store,
        sql,
        num_trials=QUERY_NUM_TRIALS,
        warmup=QUERY_WARMUP,
        query_name=query_name,
    )
    return {
        "total_chunks": store.total_chunks,
        "pushdown_eval_ns_per_label": float(np.mean(pushdown_times))
        * 1e9
        / store.total_chunks,
        "pushdown_eval_raw_samples_seconds": pushdown_times.tolist(),
        "pushdown_eval_num_trials": PUSHDOWN_ISOLATED_TRIALS,
        "pushdown_eval_method": (
            "evaluate_pushdown(predicate, store.labels, as_of=cutoff, "
            "as_of_index=store); includes labels materialization and full ID "
            "outputs"
        ),
        "chunks_skipped_ratio": pushdown_result.chunks_skipped_ratio,
        "candidate_chunks": pushdown_result.candidate_count,
        **timing_metrics,
    }


def _compare(
    label: str, before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Describe before/after estimates; overlapping intervals are not a test."""
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
        "inference": "none; interval overlap is descriptive",
    }


def run_correction_rate(*, label: str, correction_rate: float) -> dict[str, Any]:
    """Compute one correction level's full before/after measurement set."""
    df = generate_synthetic_ohlcv(
        num_symbols=NUM_SYMBOLS, rows_per_symbol=ROWS_PER_SYMBOL, seed=1
    )
    fixture = dataframe_fingerprint(df)
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

    def as_of_sql_for(cutoff: pd.Timestamp) -> str:
        return (
            f"SELECT * FROM data TIMESTAMP AS OF '{cutoff.isoformat()}' "
            f"WHERE {where_clause}"
        )

    cutoffs = as_of_cutoffs(df, corrections) if corrections else []
    primary_cutoff = next(
        (
            entry
            for entry in cutoffs
            if entry["target_visible_fraction"] == PRIMARY_VISIBLE_FRACTION
        ),
        None,
    )
    as_of_value = primary_cutoff["cutoff"] if primary_cutoff else None
    as_of_sql = as_of_sql_for(as_of_value) if primary_cutoff else None

    current_pre = measure_scenario(
        store=store,
        predicate=predicate,
        as_of_value=None,
        sql=current_sql,
        query_name=f"bitemporal-{label}-current-pre",
    )

    as_of_pre = None
    as_of_sweep: list[dict[str, Any]] = []
    for cutoff_entry in cutoffs:
        is_primary = cutoff_entry is primary_cutoff
        scenario = measure_scenario(
            store=store,
            predicate=predicate,
            as_of_value=cutoff_entry["cutoff"],
            sql=as_of_sql_for(cutoff_entry["cutoff"]),
            query_name=f"bitemporal-{label}-as_of-pre-{cutoff_entry['label']}",
        )
        scenario["as_of_cutoff"] = cutoff_entry["cutoff"].isoformat()
        scenario["visible_correction_versions"] = cutoff_entry["visible_versions"]
        scenario["visible_correction_fraction"] = cutoff_entry["visible_fraction"]
        as_of_sweep.append(
            {
                "label": cutoff_entry["label"],
                "target_visible_fraction": cutoff_entry["target_visible_fraction"],
                "cutoff": cutoff_entry["cutoff"].isoformat(),
                "visible_correction_versions": cutoff_entry["visible_versions"],
                "visible_correction_fraction": cutoff_entry["visible_fraction"],
                "is_primary": is_primary,
                "storage_state": "before compaction",
                "scenario": scenario,
            }
        )
        if is_primary:
            as_of_pre = scenario

    t0 = time.perf_counter()
    compacted = store.compact_corrections()
    compaction_seconds = time.perf_counter() - t0
    # Every correction row is replayed once per pass; 0 when nothing to compact.
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
        as_of_post["as_of_cutoff"] = as_of_value.isoformat()
        as_of_post["visible_correction_versions"] = primary_cutoff["visible_versions"]
        as_of_post["visible_correction_fraction"] = primary_cutoff["visible_fraction"]
        as_of_comparison = _compare(
            "as_of pre-vs-post compaction", as_of_pre, as_of_post
        )

    highest_version: dict[int, int] = {}
    for correction in corrections:
        key = correction["key_index"]
        highest_version[key] = max(highest_version.get(key, 0), correction["version"])
    versions_per_key = {
        str(version): sum(1 for top in highest_version.values() if top == version)
        for version in range(1, MAX_VERSIONS_PER_KEY + 1)
    }

    entry = {
        "fixture": fixture,
        "synthetic_seed": 1,
        "correction_seed": 2,
        "query_num_trials": QUERY_NUM_TRIALS,
        "query_warmup": QUERY_WARMUP,
        "label": label,
        "correction_rate": correction_rate,
        "as_of_cutoff": as_of_value.isoformat() if as_of_value is not None else None,
        "as_of_scenario": (
            "mid-history: the primary cutoff makes "
            f"{PRIMARY_VISIBLE_FRACTION:.0%} of correction versions visible "
            "(and hides base rows learned after it); corrections are 1-"
            f"{MAX_VERSIONS_PER_KEY} versions per key with 1-{MAX_LAG_DAYS} day "
            "lags. See as_of_sweep for the other cutoffs."
            if corrections
            else None
        ),
        "as_of_sweep": as_of_sweep,
        "base_rows": base_rows,
        "base_chunks": base_chunks,
        "num_corrections": len(corrections),
        "num_corrected_keys": len({c["key_index"] for c in corrections}),
        "versions_per_key": versions_per_key,
        "correction_chunks": chunks_before_compaction - base_chunks,
        "chunks_before_compaction": chunks_before_compaction,
        "correction_seconds": correction_seconds,
        "correction_rows_per_second": correction_rows_per_second,
        "compacted_chunks": compacted,
        "compaction_seconds": compaction_seconds,
        "compaction_num_trials": 1,
        "compaction_measurement_method": (
            "single compact_corrections() call; no repeated distribution "
            "or causal significance test"
        ),
        "compaction_rows_per_second": compaction_rows_per_second,
        "total_chunks": store.total_chunks,
        "correction_chunks_after_compaction": store.total_chunks - base_chunks,
        "scenarios": {
            "current_pre_compaction": current_pre,
            "current_post_compaction": current_post,
            **({"as_of_pre_compaction": as_of_pre} if as_of_pre else {}),
            **({"as_of_post_compaction": as_of_post} if as_of_post else {}),
        },
        "compaction_comparisons": {
            "current": current_comparison,
            **({"as_of": as_of_comparison} if as_of_comparison else {}),
        },
    }
    return entry


def _correctness_status(scenario: dict[str, Any] | None) -> str:
    """PASSED if measured (it already cleared the correctness gate), else N/A."""
    return "PASSED" if scenario is not None else "N/A (no corrections exist yet)"


def _matrix_row(
    name: str, storage_state: str, scenario: dict[str, Any] | None
) -> list[str]:
    if scenario is None:
        return [name, storage_state, "N/A", "N/A", "N/A", "N/A"]
    return [
        name,
        storage_state,
        format_ms(scenario["query_full_scan_mean_ms"]),
        format_ms(scenario["query_pushdown_mean_ms"]),
        format_x(scenario["query_speedup"]),
        format_x(scenario["vectorized_full_scan_speedup"]),
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
    """Print one correction level's complete before/after block."""
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
                ("Correction versions", f"{entry['num_corrections']:,}"),
                ("Corrected keys", f"{entry['num_corrected_keys']:,}"),
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
            [
                "Query",
                "Storage state",
                "Row-wise scan",
                "Pushdown",
                "vs. row-wise",
                "vs. vectorized",
            ],
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

    if entry["as_of_sweep"]:
        print()
        print("Historical AS OF cutoff sweep (before compaction)")
        print()
        print(
            render_box_table(
                [
                    "Cutoff makes visible",
                    "Versions visible",
                    "Rows returned",
                    "Pruning rate",
                    "Row-wise",
                    "Pushdown",
                    "vs. row-wise",
                    "vs. vectorized",
                ],
                [
                    [
                        point["label"] + (" (primary)" if point["is_primary"] else ""),
                        f"{point['visible_correction_versions']:,} "
                        f"({format_pct(point['visible_correction_fraction'])})",
                        f"{point['scenario']['row_count']:,}",
                        format_pct(point["scenario"]["chunks_skipped_ratio"]),
                        format_ms(point["scenario"]["query_full_scan_mean_ms"]),
                        format_ms(point["scenario"]["query_pushdown_mean_ms"]),
                        format_x(point["scenario"]["query_speedup"]),
                        format_x(point["scenario"]["vectorized_full_scan_speedup"]),
                    ]
                    for point in entry["as_of_sweep"]
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
            [
                "Query",
                "Storage state",
                "Row-wise scan",
                "Pushdown",
                "vs. row-wise",
                "vs. vectorized",
            ],
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
    """Describe the measured AS OF speedup range across the cutoff sweep."""

    with_corrections = [entry for entry in results if entry["num_corrections"] > 0]
    if not with_corrections:
        return "No corrections were applied in this sweep; AS OF was never measured."
    entry = with_corrections[-1]
    points = entry["as_of_sweep"]
    speedups = [(point["scenario"]["query_speedup"], point) for point in points]
    low, low_point = min(speedups, key=lambda item: item[0])
    high, high_point = max(speedups, key=lambda item: item[0])
    primary = next(point for point in points if point["is_primary"])
    return (
        f"At {entry['label']} corrected ({entry['num_corrections']:,} versions "
        f"over {entry['num_corrected_keys']:,} keys), AS OF speedup vs. the "
        f"row-wise scan ranged {format_x(low)} (cutoff showing "
        f"{low_point['label']}) to {format_x(high)} (cutoff showing "
        f"{high_point['label']}) across {len(points)} cutoffs; the primary "
        f"mid-history cutoff ({primary['label']} visible) gave "
        f"{format_x(primary['scenario']['query_speedup'])} with "
        f"{format_pct(primary['scenario']['chunks_skipped_ratio'])} of chunks "
        "pruned."
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
    # Everything is computed before anything prints.
    manifest = create_manifest(
        parameters={
            "benchmark": "bitemporal_sweep",
            "visible_fractions": list(VISIBLE_FRACTIONS),
            "primary_visible_fraction": PRIMARY_VISIBLE_FRACTION,
            "max_versions_per_key": MAX_VERSIONS_PER_KEY,
            "max_lag_days": MAX_LAG_DAYS,
            "num_symbols": NUM_SYMBOLS,
            "rows_per_symbol": ROWS_PER_SYMBOL,
            "correction_rates": CORRECTION_RATES,
            "num_trials": QUERY_NUM_TRIALS,
            "warmup": QUERY_WARMUP,
            "synthetic_seed": 1,
            "correction_seed": 2,
        }
    )
    results = [
        run_correction_rate(label=f"{rate:.0%}", correction_rate=rate)
        for rate in CORRECTION_RATES
    ]

    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "bitemporal_sweep.json"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "hardware": hardware_snapshot(),
        "points": results,
    }
    write_result(output_path, payload, manifest)

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

    if any(entry["as_of_cutoff"] for entry in results):
        as_of_cutoff_display = (
            f"sweep of {len(VISIBLE_FRACTIONS) + 1} cutoffs per level; primary = "
            f"{PRIMARY_VISIBLE_FRACTION:.0%} of correction versions visible "
            "(mid-history, varies by level)"
        )
    else:
        as_of_cutoff_display = "N/A (no corrections in this sweep)"

    print()
    print(
        console.titled_block(
            "Configuration",
            [
                ("Base rows", f"{results[0]['base_rows']:,}"),
                ("Correction levels", str(len(CORRECTION_RATES))),
                (
                    "Correction model",
                    f"1-{MAX_VERSIONS_PER_KEY} versions per corrected key, each "
                    f"learned 1-{MAX_LAG_DAYS} days after the previous",
                ),
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
                "Versions",
                "Corrected rows",
                "Current-state speedup (before/after)",
                "AS OF speedup, primary cutoff (before/after)",
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
            "transaction times occur after the requested cutoff. The pruning "
            "and speedup actually obtained depend on how many correction "
            "versions the cutoff leaves visible (see the AS OF sweep tables "
            "and the trend line above); no single AS OF speedup describes "
            "every cutoff.",
        )
    )
    print()
    print(
        console.titled_paragraph(
            "Limitation",
            "Performance conclusions apply only to the tested synthetic "
            "correction model (random lags of 1-90 days, up to 3 versions per "
            "key, 20 symbols, 501 rows each) and compaction policy.",
        )
    )
    print()
    print(console.pitdb_footer(index=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
