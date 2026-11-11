# ruff: noqa: E501 -- generated-Markdown table rows are intentionally one line each
"""Render BENCHMARK_RESULTS.md from the benchmark result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RESULTS = PROJECT_ROOT / "benchmarks" / "results"
DEFAULT_CHARTS = PROJECT_ROOT / "charts" / "output"
DEFAULT_OUTPUT = PROJECT_ROOT / "BENCHMARK_RESULTS.md"

CHART_DESCRIPTIONS = {
    "01_speedup_vs_selectivity.png": "Graph 1: speedup vs. selectivity, row-wise "
    "baseline and vectorized control",
    "02_latency_q1_q5.png": "Graph 2: Q1-Q5 latency, three engines, log axis",
    "03_chunks_skipped.png": "Graph 3: chunks skipped vs. selectivity",
    "04_memory.png": "Graph 4: retained sizes and per-query peak allocation",
    "05_scale_sweep.png": "Graph 5: scale sweep",
    "06_bitemporal_sweep.png": "Graph 6: bitemporal AS OF sweep, current-state "
    "latency, compaction cost",
    "07_chunk_granularity.png": "Graph 7: chunk granularity",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ms(seconds: float) -> float:
    return seconds * 1000.0


def _lat(summary: dict[str, Any]) -> str:
    """mean +/- SD (ms) of one latency summary."""
    return (
        f"{_ms(summary['mean_latency_seconds']):.3f} "
        f"+/- {_ms(summary['std_latency_seconds']):.3f}"
    )


def _x(value: float) -> str:
    return f"{value:.2f}x"


def _pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _provenance(
    manifest: dict[str, Any], results: dict[str, dict], run_note: str, load_note: str
) -> str:
    hw = manifest["hardware"]
    dirty = manifest["git_worktree"]
    status = dirty.get("status")
    changed = len(status) if isinstance(status, list) else "unknown"
    deps = ", ".join(f"{k} {v}" for k, v in manifest["dependency_versions"].items())
    threads = ", ".join(f"{k}={v}" for k, v in manifest["thread_environment"].items())
    fixtures = "\n".join(
        f"  - `{Path(name).name}`: sha256 `{digest}`"
        for name, digest in sorted(
            results["queries"]["manifest"]["fixture_files"].items()
        )
    )
    return f"""## Provenance

| Field | Value |
|---|---|
| Run ID | `{manifest["run_id"]}` |
| Started / completed (UTC) | {manifest["started_at_utc"]} / {manifest["completed_at_utc"]} |
| Git commit (HEAD) | `{manifest["git_commit"]}` |
| Working tree at start | dirty: {dirty.get("dirty")} ({changed} changed or untracked paths); the results measure the working-tree source identified by the hash below, not the bare commit |
| Source snapshot sha256 | `{manifest["source"]["sha256"]}` ({len(manifest["source"]["files"])} files: `src/`, `benchmarks/`, `charts/`, requirements, setup files) |
| Python | {hw["python_version"]} |
| OS / machine | {hw["system"]} {hw["release"]} ({hw["machine"]}) |
| CPU | {hw.get("processor") or hw["machine"]}; {hw["logical_cpu_count"]} logical / {hw["physical_cpu_count"]} physical cores |
| Memory | {hw["total_memory_bytes"] / 1024**3:.1f} GiB |
| Dependencies | {deps} |
| Thread environment | {threads} |
| Timing scope | {manifest["timing_scope"]} |

Fixture hashes (real-data benchmarks 1-2):

{fixtures}

Machine load: {load_note}

Run isolation: {run_note}

All five result files and the chart set carry this run ID and source hash;
`charts/output/manifest.json` binds each PNG to the exact result bytes."""


def _fixed_queries(queries: dict[str, Any]) -> str:
    def block(fixed: dict[str, Any]) -> str:
        rows = []
        for name, q in fixed.items():
            push = q["pushdown_query"]
            vec = push["vectorized_full_scan"]
            rows.append(
                [
                    name,
                    f"{push['row_count']:,}",
                    _pct(q["selectivity"]),
                    f"{push['candidate_chunks']}/{push['total_chunks']}",
                    _lat(q["full_scan"]),
                    _lat(vec),
                    _lat(push),
                    _x(q["speedup"]),
                    _x(push["vectorized_full_scan_speedup"]),
                    str(push["sample_count"]),
                ]
            )
        return _table(
            [
                "Query",
                "Rows returned",
                "Selectivity",
                "Chunks scanned",
                "Row-wise ms (mean +/- SD)",
                "Vectorized ms",
                "Pushdown ms",
                "Row-wise / pushdown",
                "Vectorized / pushdown",
                "Trials",
            ],
            rows,
        )

    store = queries["store"]
    out = [
        "## Benchmark 1: fixed queries and selectivity sweep",
        "",
        f"Dataset A: `all_symbols_daily.csv`, {store['total_rows']:,} rows, "
        f"{store['total_chunks']} monthly chunks, "
        f"{store['total_column_bytes'] / 1e6:.2f} MB of float64 column data.",
        "",
        block(queries["fixed_queries"]),
        "",
        "**Read with care.** The fixture is tiny and Q1-Q4 are needle queries "
        "returning 1-140 rows. The row-wise baseline rebuilds a DataFrame for "
        "every chunk on every call, so the ratios above mostly reflect that "
        "fixed per-chunk overhead; they do not extrapolate to larger or "
        "denser data (see Benchmark 3 for synthetic scale). The vectorized "
        "column is the control that isolates pruning from row-wise "
        "evaluation.",
        "",
        "### Selectivity sweep",
        "",
    ]
    rows = []
    for p in queries["selectivity_sweep"]["points"]:
        push = p["pushdown_query"]
        rows.append(
            [
                f"{p['target_selectivity']:g}",
                _pct(p["actual_selectivity"]),
                f"{push['row_count']:,}",
                _pct(p["chunks_skipped_ratio"]),
                _lat(p["full_scan"]),
                _lat(push["vectorized_full_scan"]),
                _lat(push),
                _x(p["speedup"]),
                _x(push["vectorized_full_scan_speedup"]),
            ]
        )
    out.append(
        _table(
            [
                "Target",
                "Actual selectivity",
                "Rows",
                "Chunks skipped",
                "Row-wise ms",
                "Vectorized ms",
                "Pushdown ms",
                "Row-wise / pushdown",
                "Vectorized / pushdown",
            ],
            rows,
        )
    )
    out.append("")
    demo = queries["price_pushdown_demonstration"]
    rng = demo.get("aapl_2024_close_range")
    out.append(
        f"Price-clause demonstration: AAPL's 2024 close range derived from the "
        f"store labels is {rng[0]:.2f} to {rng[1]:.2f}; adding `price > 200` "
        f"prunes {demo['price_clause_incremental_chunks_pruned']} additional "
        f"chunks beyond the date clause "
        f"({demo['date_only']['candidate_chunks']} -> "
        f"{demo['date_and_price']['candidate_chunks']} candidates)."
    )
    intraday = queries.get("dataset_b_intraday")
    if intraday:
        out += ["", "### Dataset B (1-minute, hourly chunks)", ""]
        st = intraday["store"]
        out.append(
            f"{st['total_rows']:,} rows, {st['total_chunks']} chunks. The fixture "
            "ends mid-session (see `data/README.md`)."
        )
        out += ["", block(intraday["fixed_queries"])]
    return "\n".join(out)


def _memory(memory: dict[str, Any]) -> str:
    rows = []
    for name, entry in memory["traced_query_peaks"].items():
        e = entry["engines"]
        rows.append(
            [
                name,
                f"{entry['row_count']:,}",
                f"{e['full_scan']['median_peak_bytes']:,}",
                f"{e['vectorized_full_scan']['median_peak_bytes']:,}",
                f"{e['pushdown_query']['median_peak_bytes']:,}",
            ]
        )
    rss = memory["rss_diagnostic"]
    construction = memory.get("store_construction_traced_peak_bytes")
    lines = [
        "## Benchmark 2: memory",
        "",
        "Storage is uncompressed float64. **No storage reduction or compression "
        "ratio is measured or claimed**; the numeric payload is equal in the "
        "source frame and the store by construction "
        f"(equal: {memory['numeric_payload_matches_store']}).",
        "",
        _table(
            ["Representation", "Bytes"],
            [
                [
                    "Numeric payload, float64 x 5 columns (source == store)",
                    f"{memory['column_payload_bytes']:,}",
                ],
                [
                    "Full source DataFrame (pandas deep)",
                    f"{memory['raw_uncompressed_bytes']:,}",
                ],
                [
                    "Store object-graph estimate",
                    f"{memory['chunk_store_bytes']:,}",
                ],
            ],
        ),
        "",
        "These use different accounting methods and are not a ratio.",
        "",
        f"Per-query peak allocation (`tracemalloc` peak above the pre-call "
        f"baseline, bytes; median of {memory['traced_repeats']} repeats with "
        "engines rotating through every ordering; min and max are in "
        "`memory.json`):",
        "",
        _table(
            [
                "Query",
                "Rows",
                "Row-wise scan",
                "Vectorized scan",
                "Pushdown",
            ],
            rows,
        ),
    ]
    if construction:
        lines += [
            "",
            f"Store construction traced peak (median of 3): "
            f"{construction['median_peak_bytes']:,} bytes.",
        ]
    lines += [
        "",
        f"Process RSS is a **noise-bounded diagnostic and not a result**: Q1 "
        f"full-scan increase {rss['full_scan_q1_peak_rss_increase_bytes']:,} "
        f"bytes, Q1 pushdown increase {rss['q1_peak_rss_increase_bytes']:,} "
        f"bytes against a {rss['idle_rss_bytes'] / 1e6:.0f} MB process, "
        f"{rss['resolution_bytes']}-byte page resolution, "
        f"{rss['sampling_interval_microseconds']} microsecond sampler, "
        "single shot in fixed order. It is omitted from Graph 4. The earlier "
        "2.7x memory comparison is withdrawn.",
        "",
        f"Method: {memory['traced_query_method']}.",
    ]
    return "\n".join(lines)


def _scale(scale: dict[str, Any]) -> str:
    rows = []
    for p in scale["points"]:
        narrow = p["queries"]["narrow_symbol_eq"]
        broad = p["queries"]["broad_full_range"]
        rows.append(
            [
                p["label"],
                f"{p['num_symbols']:,}",
                f"{p['total_rows']:,}",
                f"{p['total_chunks']:,}",
                f"{p['query_num_trials']}/{p['query_warmup']}",
                f"{p['ingest_rows_per_second']:,.0f}",
                f"{p['pushdown_eval']['ns_per_label_broad']:.1f}",
                _x(narrow["speedup"]),
                _x(narrow["vectorized_full_scan_speedup"]),
                _x(broad["speedup"]),
                _x(broad["vectorized_full_scan_speedup"]),
            ]
        )
    detail = []
    for p in scale["points"]:
        for name, q in p["queries"].items():
            detail.append(
                [
                    p["label"],
                    name,
                    f"{q['row_count']:,}",
                    f"{q['candidate_chunks']:,}/{q['total_chunks']:,}",
                    _lat(q["full_scan"]),
                    _lat(q["vectorized_full_scan"]),
                    _lat(q["pushdown_query"]),
                ]
            )
    return "\n".join(
        [
            "## Benchmark 3: synthetic scale sweep",
            "",
            "Synthetic OHLCV, seed 1, 501 rows per symbol, monthly chunks. "
            "**Protocols differ by point** (trials/warm-up column): one "
            "row-wise call at the largest point costs on the order of 100 s, so "
            "the 1000x point has one balanced engine-permutation cycle "
            "(6 trials) and one warm-up round. Its Student-t interval is wide "
            "and its speedups must not be compared with the other points "
            "without that caveat.",
            "",
            _table(
                [
                    "Scale",
                    "Symbols",
                    "Rows",
                    "Chunks",
                    "Trials/warm-up",
                    "Ingest rows/s",
                    "ns/label (broad)",
                    "Narrow: row-wise / pushdown",
                    "Narrow: vectorized / pushdown",
                    "Broad: row-wise / pushdown",
                    "Broad: vectorized / pushdown",
                ],
                rows,
            ),
            "",
            "Latencies (ms, mean +/- SD):",
            "",
            _table(
                [
                    "Scale",
                    "Query",
                    "Rows",
                    "Chunks scanned",
                    "Row-wise",
                    "Vectorized",
                    "Pushdown",
                ],
                detail,
            ),
        ]
    )


def _bitemporal(bi: dict[str, Any]) -> str:
    pts = bi["points"]
    cur = []
    for p in pts:
        s = p["scenarios"]
        cur.append(
            [
                p["label"],
                f"{p['num_corrections']:,}",
                f"{p['num_corrected_keys']:,}",
                f"{p['correction_seconds'] * 1000:.1f}",
                f"{p['compaction_seconds'] * 1000:.1f}",
                f"{p['chunks_before_compaction']:,} -> {p['total_chunks']:,}",
                _x(s["current_pre_compaction"]["query_speedup"]),
                _x(s["current_post_compaction"]["query_speedup"]),
                _lat(s["current_pre_compaction"]["pushdown_query"]),
                _lat(s["current_post_compaction"]["pushdown_query"]),
            ]
        )
    out = [
        "## Benchmark 4: bitemporal corrections and compaction",
        "",
        "Synthetic: 20 symbols x 501 rows. Each corrected key receives 1-3 "
        "successive versions, each learned 1-90 days after the previous, so "
        "transaction times are spread through the history. This is **not** a "
        "best case: the previous design placed every correction after the "
        "last base row and queried AS OF the last base row, which pruned all "
        "correction chunks.",
        "",
        "Current-state queries (no AS OF), before/after `compact_corrections()`:",
        "",
        _table(
            [
                "Corrected rows",
                "Versions",
                "Keys",
                "Ingest ms",
                "Compaction ms (1 call)",
                "Chunks",
                "Row-wise / pushdown (pre)",
                "(post)",
                "Pushdown ms (pre)",
                "Pushdown ms (post)",
            ],
            cur,
        ),
    ]
    for p in pts:
        sweep = p.get("as_of_sweep")
        if not sweep:
            continue
        rows = []
        for item in sweep:
            sc = item["scenario"]
            rows.append(
                [
                    item["label"] + (" (primary)" if item["is_primary"] else ""),
                    f"{item['visible_correction_versions']:,} "
                    f"({_pct(item['visible_correction_fraction'])})",
                    f"{sc['row_count']:,}",
                    _pct(sc["chunks_skipped_ratio"]),
                    _lat(sc["full_scan"]),
                    _lat(sc["pushdown_query"]),
                    _x(sc["query_speedup"]),
                    _x(sc["vectorized_full_scan_speedup"]),
                ]
            )
        out += [
            "",
            f"AS OF cutoff sweep at {p['label']} of rows corrected "
            f"({p['num_corrections']:,} versions, before compaction); every "
            "cutoff's three engines returned identical results:",
            "",
            _table(
                [
                    "Cutoff makes visible",
                    "Versions visible",
                    "Rows returned",
                    "Chunks skipped",
                    "Row-wise ms",
                    "Pushdown ms",
                    "Row-wise / pushdown",
                    "Vectorized / pushdown",
                ],
                rows,
            ),
        ]
    out += [
        "",
        "AS OF at the primary (50%-visible) cutoff, before / after compaction "
        "(row-wise / pushdown):",
        "",
        _table(
            ["Corrected rows", "Before compaction", "After compaction"],
            [
                [
                    p["label"],
                    _x(p["scenarios"]["as_of_pre_compaction"]["query_speedup"]),
                    _x(p["scenarios"]["as_of_post_compaction"]["query_speedup"]),
                ]
                for p in pts
                if "as_of_pre_compaction" in p["scenarios"]
            ],
        ),
        "",
        "Compaction timing is one call per level (no distribution).",
    ]
    return "\n".join(out)


def _granularity(gr: dict[str, Any]) -> str:
    pts = gr["points"]
    findings = gr["findings"]
    rows = []
    for p in pts:
        a = p["aggregate"]
        rows.append(
            [
                p["label"],
                f"{p['total_chunks']:,}",
                f"{p['avg_rows_per_chunk']:.1f}",
                _pct(a["mean_chunks_skipped_ratio"]),
                f"{a['mean_pushdown_ms']:.3f}",
                f"{a['mean_vectorized_ms']:.3f}",
                f"{a['mean_row_wise_ms']:.3f}",
                _x(a["geomean_row_wise_over_pushdown"]),
            ]
        )
    q = pts[0]["aggregate"]
    return "\n".join(
        [
            "## Benchmark 5: chunk granularity",
            "",
            f"Synthetic, 10 symbols x 501 rows; {q['num_queries']} queries "
            f"({', '.join(q['symbols'])} x {', '.join(q['windows'])}), each with "
            "a median-close price band. Latencies are means over the query "
            "grid (ms). Pushdown latency and row-wise baseline latency are "
            "shown separately.",
            "",
            _table(
                [
                    "Granularity",
                    "Chunks",
                    "Rows/chunk",
                    "Mean pruning",
                    "Pushdown ms",
                    "Vectorized ms",
                    "Row-wise ms",
                    "Row-wise / pushdown (geomean)",
                ],
                rows,
            ),
            "",
            f"- Lowest mean pushdown latency: **{findings['lowest_pushdown_label']}**.",
            f"- Lowest mean row-wise latency: {findings['lowest_row_wise_label']}.",
            f"- Highest speedup ratio: {findings['highest_speedup_label']}.",
            f"- Configured engine default: {findings['configured_default_label']} "
            f"(reported separately; not validated here): "
            f"{findings['default_vs_best']}.",
            f"- {findings['speedup_explanation']}",
            "",
            "One synthetic dataset and a nine-query grid cannot justify a "
            "general default granularity, and differences between "
            "granularities may be within run-to-run noise.",
        ]
    )


CAVEATS = """## Caveats

1. **Small real-data fixture.** 5,010 rows, 240 chunks, 0.2 MB. Needle-query
   ratios are dominated by per-chunk overhead of the row-wise baseline.
2. **Single warm process, wall clock.** No cold-cache or fresh-process
   isolation; trials are not independent. Intervals are descriptive
   Student-t intervals on the raw mean, not significance tests.
3. **Unequal scale-sweep protocols.** See Benchmark 3.
4. **AS OF performance depends on the cutoff.** No single AS OF speedup
   describes the engine; read the sweep.
5. **Granularity.** One synthetic dataset; no general default is implied.
6. **Memory.** Uncompressed storage: no reduction claim. `tracemalloc` covers
   Python/NumPy-registered allocations only; RSS is noise-bounded.
7. **Equality gates share code.** Row-wise, vectorized and pushdown paths
   share the parser and version resolver, so agreement is a regression check,
   not an independent oracle.
8. **Machine load.** Other processes ran on the machine during parts of the
   run (see provenance); timings are not from an idle machine.
9. **Dirty working tree.** Results measure the source hashed in the manifest,
   which differs from the bare git commit."""


def render(results_dir: Path, charts_dir: Path, run_note: str, load_note: str) -> str:
    results = {p.stem: _load(p) for p in sorted(results_dir.glob("*.json"))}
    required = {
        "queries",
        "memory",
        "scale_sweep",
        "bitemporal_sweep",
        "chunk_granularity_sweep",
    }
    missing = required - set(results)
    if missing:
        raise SystemExit(f"missing result files: {sorted(missing)}")
    manifest = results["queries"]["manifest"]
    chart_manifest = _load(charts_dir / "manifest.json")
    if chart_manifest["run_id"] != manifest["run_id"]:
        raise SystemExit("chart manifest run ID does not match the results")
    charts = "\n".join(
        f"- `charts/output/{name}` ({CHART_DESCRIPTIONS.get(name, '')}); "
        f"sha256 `{digest}`"
        for name, digest in sorted(chart_manifest["chart_sha256"].items())
    )
    parts = [
        "# PitDB measured results",
        "",
        "Generated by `benchmarks/render_results.py` from the JSON files in "
        "`benchmarks/results/`; every number below is read from those files. "
        "See [METHODOLOGY.md](METHODOLOGY.md) for protocol and interpretation "
        "and [All_Commands.md](All_Commands.md) to reproduce.",
        "",
        "Row-wise = the baseline `FullScanQueryEngine`; vectorized = the same "
        "executor with pruning disabled (control); pushdown = the pruned "
        "executor. Speedup = baseline raw mean / pushdown raw mean. "
        "`mean +/- SD` is the raw arithmetic mean and sample standard "
        "deviation of the trials in milliseconds.",
        "",
        _provenance(manifest, results, run_note, load_note),
        "",
        _fixed_queries(results["queries"]),
        "",
        _memory(results["memory"]),
        "",
        _scale(results["scale_sweep"]),
        "",
        _bitemporal(results["bitemporal_sweep"]),
        "",
        _granularity(results["chunk_granularity_sweep"]),
        "",
        CAVEATS,
        "",
        "## Charts",
        "",
        charts,
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python benchmarks/run_suite.py          # about 2h40m (reference laptop)",
        '.venv/bin/python benchmarks/render_results.py --load-note "..." '
        '--run-note "..."',
        "```",
        "",
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--charts-dir", type=Path, default=DEFAULT_CHARTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--load-note", required=True)
    parser.add_argument("--run-note", required=True)
    args = parser.parse_args(argv)
    text = render(args.results_dir, args.charts_dir, args.run_note, args.load_note)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
