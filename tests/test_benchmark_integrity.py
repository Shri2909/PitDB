"""Regression checks for measurement scope, controls and artifact provenance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks import bench_query, run_suite
from benchmarks.bench_memory import run_memory_benchmarks
from benchmarks.synthetic_data import generate_synthetic_ohlcv
from src.query.executor import QueryResult
from src.query.pushdown import PushdownResult
from src.store.chunk_store import ChunkStore


def _empty_result() -> QueryResult:
    return QueryResult(pd.DataFrame(), 0, PushdownResult([], [], 0, 0.0, 0), 0.0, 0.0)


def test_latency_keeps_acquisition_order_and_separates_selected_estimator():
    samples = np.array([0.100, *([0.001] * 13), 0.101])
    metrics = bench_query._latency_metrics(samples, _empty_result())
    assert metrics["raw_latency_samples_seconds"] == samples.tolist()
    assert metrics["retained_sample_count"] == 13
    assert metrics["mean_latency_seconds"] == pytest.approx(float(samples.mean()))
    assert metrics["upper_trimmed_mean_seconds"] == pytest.approx(0.001)
    assert metrics["median_latency_seconds"] == pytest.approx(0.001)
    assert "raw untrimmed" in metrics["mean_estimand"]


def test_interval_overlap_does_not_claim_significance():
    measurements = {
        name: {
            "pushdown_query": {
                "mean_latency_seconds": mean,
                "ci95_low_seconds": low,
                "ci95_high_seconds": high,
            }
        }
        for name, mean, low, high in [("A", 1, 0, 2), ("B", 2.8, 1.8, 3.8)]
    }
    comparison = bench_query.annotate_pairwise_significance(measurements)[0]
    assert comparison["ci95_overlap"] is True
    assert "distinguishable_from_noise" not in comparison
    assert comparison["inference"] == "none; interval overlap is descriptive"


def test_memory_benchmark_reports_no_storage_reduction_and_traced_peaks():
    frame = generate_synthetic_ohlcv(num_symbols=3, rows_per_symbol=40, seed=1)
    store = ChunkStore()
    store.ingest_dataframe(frame)
    memory = run_memory_benchmarks(
        frame,
        store,
        queries={
            "Q1": "SELECT * FROM data WHERE symbol = 'SYN000000'",
            "Q2": "SELECT * FROM data",
        },
        repeats=6,
        include_rss_diagnostic=False,
    )
    expected = sum(frame[column].to_numpy().nbytes for column in store.value_columns)
    assert memory["raw_numeric_payload_bytes"] == expected
    assert memory["column_payload_bytes"] == expected
    assert memory["numeric_payload_matches_store"] is True
    assert memory["raw_uncompressed_bytes"] > expected
    # The tautological ratio / "reduction" claim must not be exported.
    assert "numeric_payload_ratio" not in memory
    assert not any("reduction" in key for key in memory)
    assert "no compression" in memory["numeric_payload_note"].lower()
    for entry in memory["traced_query_peaks"].values():
        assert set(entry["engines"]) == {
            "full_scan",
            "vectorized_full_scan",
            "pushdown_query",
        }
        for peaks in entry["engines"].values():
            assert peaks["repeats"] == 6 == len(peaks["peak_bytes_samples"])
            assert peaks["min_peak_bytes"] <= peaks["median_peak_bytes"]
            assert peaks["median_peak_bytes"] <= peaks["max_peak_bytes"]
    q1 = memory["traced_query_peaks"]["Q1"]["engines"]
    # One symbol's chunks vs every chunk: pruning must allocate less.
    assert (
        q1["pushdown_query"]["median_peak_bytes"] < q1["full_scan"]["median_peak_bytes"]
    )


def test_traced_peak_sees_numpy_allocations_and_ignores_prior_memory():
    import numpy as np

    from benchmarks.bench_memory import measure_traced_peak

    resident = np.zeros(2_000_000)  # live before the call: must not be counted
    peak = measure_traced_peak(lambda: np.ones(1_000_000))
    assert 7_000_000 <= peak <= 9_000_000
    assert resident.nbytes == 16_000_000


def test_rss_is_labelled_as_a_noise_bounded_diagnostic(monkeypatch):
    frame = generate_synthetic_ohlcv(num_symbols=1, rows_per_symbol=4, seed=1)
    store = ChunkStore()
    store.ingest_dataframe(frame)
    monkeypatch.setattr(
        "benchmarks.bench_memory._measure_peak_rss", lambda fn: (100, 4196)
    )
    memory = run_memory_benchmarks(
        frame, store, queries={"Q1": "SELECT * FROM data"}, repeats=6
    )
    rss = memory["rss_diagnostic"]
    assert "noise-bounded" in rss["status"]
    assert rss["resolution_bytes"] == 4096
    assert "q1_peak_rss_bytes" not in memory


def test_suite_memory_row_makes_no_reduction_claim(monkeypatch):
    memory = {
        "traced_query_peaks": {
            "Q1": {
                "engines": {
                    "pushdown_query": {"median_peak_bytes": 1000},
                    "full_scan": {"median_peak_bytes": 2000},
                }
            }
        }
    }
    queries = {"fixed_queries": {"Q1": {"speedup": 2.0}}}
    monkeypatch.setattr(
        run_suite.run_all,
        "run_all",
        lambda include_memory: {"queries": queries, "memory": memory},
    )
    _, row_2 = run_suite._run_benchmarks_1_and_2()
    assert "reduction" not in row_2[3].lower().replace("no reduction claimed", "")
    assert "traced peak" in row_2[3]


def test_suite_suppresses_charts_after_correctness_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(run_suite, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_suite, "LOG_PATH", tmp_path / "suite.log")
    monkeypatch.setattr(
        run_suite,
        "_run_benchmarks_1_and_2",
        lambda: (
            ["1", "Queries", "FAILED", "SUPPRESSED"],
            ["2", "Memory", "FAILED", "SUPPRESSED"],
        ),
    )
    for number in (3, 4, 5):
        monkeypatch.setattr(
            run_suite,
            f"_run_benchmark_{number}",
            lambda: ["x", "other", "PASSED", "ok"],
        )
    chart_calls = []
    monkeypatch.setattr(
        run_suite, "generate_all_charts", lambda **kw: chart_calls.append(kw)
    )
    assert run_suite.run_suite() == 1
    assert chart_calls == []


def test_interleaved_controls_balance_position_and_preserve_sample_alignment():
    from collections import Counter

    from benchmarks.stats import time_interleaved

    seen = []
    samples, results, order = time_interleaved(
        {name: lambda name=name: seen.append(name) or name for name in ("a", "b", "c")},
        num_trials=6,
        warmup=0,
    )
    assert [entry["engine"] for entry in order] == seen
    assert len(order) == 18
    assert all(len(values) == 6 for values in samples.values())
    assert results == {name: name for name in ("a", "b", "c")}
    positions = Counter((entry["engine"], entry["position"]) for entry in order)
    assert set(positions.values()) == {2}
    rounds = [
        [entry["engine"] for entry in order if entry["round"] == i] for i in range(6)
    ]
    assert sum(row.index("a") < row.index("b") for row in rounds) == 3


def test_interleaved_rounds_trials_up_to_a_balanced_cycle_and_restores_gc():
    import gc
    from collections import Counter

    from benchmarks.stats import balanced_trial_count, time_interleaved

    assert balanced_trial_count(15, 3) == 18
    assert balanced_trial_count(18, 3) == 18
    assert balanced_trial_count(1, 2) == 2
    assert gc.isenabled()
    states = []
    samples, _, order = time_interleaved(
        {n: lambda n=n: states.append(gc.isenabled()) for n in ("a", "b", "c")},
        num_trials=4,
        warmup=1,
    )
    assert gc.isenabled()  # restored afterwards
    assert states and not any(states)  # disabled during every call
    assert all(len(values) == 6 for values in samples.values())
    positions = Counter((e["engine"], e["position"]) for e in order)
    assert set(positions.values()) == {2}  # perfectly balanced, no 4/5/6 split


def test_student_t_interval_is_wider_than_normal_at_small_n():
    from benchmarks.stats import (
        Z_95,
        confidence_interval_95,
        latency_summary,
        t_critical_95,
    )

    assert t_critical_95(2) == pytest.approx(4.303)
    assert t_critical_95(17) == pytest.approx(2.110)
    assert t_critical_95(10_000) == pytest.approx(Z_95, abs=0.01)
    low, high = confidence_interval_95(10.0, 1.0, 3)
    assert (high - low) / 2 == pytest.approx(4.303 / 3**0.5)
    summary = latency_summary(np.array([0.010, 0.011, 0.012]))
    assert "Student-t" in summary["uncertainty_method"]
    half = (summary["ci95_high_seconds"] - summary["ci95_low_seconds"]) / 2
    sd, n = summary["std_latency_seconds"], 3
    assert half == pytest.approx(4.303 * sd / n**0.5)


def test_three_query_paths_agree_for_broad_and_narrow_queries(monkeypatch):
    frame = generate_synthetic_ohlcv(num_symbols=2, rows_per_symbol=10, seed=1)
    store = ChunkStore()
    store.ingest_dataframe(frame)
    monkeypatch.setattr(bench_query, "NUM_TRIALS", 2)
    monkeypatch.setattr(bench_query, "WARMUP_RUNS", 0)
    measured = bench_query.run_query_benchmarks(
        store,
        {
            "broad": "SELECT * FROM data",
            "narrow": "SELECT * FROM data WHERE symbol = 'SYN000000'",
        },
        quiet=True,
    )
    broad = measured["broad"]["pushdown_query"]
    assert broad["chunks_skipped_ratio"] == 0.0
    for entry in measured.values():
        control = entry["pushdown_query"]["vectorized_full_scan"]
        assert control["row_count"] == entry["full_scan"]["row_count"]
        # Requested 2 trials is rounded up to one full 3! cycle.
        assert len(control["raw_latency_samples_seconds"]) == 6
        assert entry["num_trials"] == 6
        assert len(entry["pushdown_query"]["acquisition_order"]) == 18
        assert entry["pushdown_query"]["vectorized_full_scan_speedup"] > 0


def test_run_manifest_hashes_source_fixture_and_parameters(tmp_path):
    from benchmarks.reporting.provenance import create_manifest, finish_manifest

    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "engine.py"
    source.write_text("value = 1\n")
    fixture = tmp_path / "fixture.csv"
    fixture.write_text("x\n1\n")
    manifest = create_manifest(
        parameters={"trials": 3, "seed": 1}, fixture_paths=[fixture], root=tmp_path
    )
    assert manifest["complete"] is False
    assert manifest["parameters"] == {"trials": 3, "seed": 1}
    assert "src/engine.py" in manifest["source"]["files"]
    assert manifest["fixture_files"][str(fixture)]
    assert manifest["dependency_versions"]["numpy"]
    assert "OPENBLAS_NUM_THREADS" in manifest["thread_environment"]
    assert finish_manifest(manifest, root=tmp_path)["complete"] is True
    source.write_text("value = 2\n")
    with pytest.raises(RuntimeError, match="source changed"):
        finish_manifest(manifest, root=tmp_path)


def test_run_manifest_rejects_fixture_mutation(tmp_path):
    from benchmarks.reporting.provenance import create_manifest, finish_manifest

    fixture = tmp_path / "fixture.csv"
    fixture.write_text("x\n1\n")
    manifest = create_manifest(parameters={}, fixture_paths=[fixture], root=tmp_path)
    fixture.write_text("x\n2\n")
    with pytest.raises(RuntimeError, match="fixture changed"):
        finish_manifest(manifest, root=tmp_path)


def test_chart_generation_rejects_mixed_or_incomplete_manifests():
    from charts.generate_all import validate_result_manifests

    good = {
        "manifest": {"complete": True, "run_id": "one", "source": {"sha256": "abc"}}
    }
    other = {
        "manifest": {"complete": True, "run_id": "two", "source": {"sha256": "abc"}}
    }
    assert validate_result_manifests({"queries": good, "memory": good}) == "one"
    with pytest.raises(ValueError, match="mix different runs"):
        validate_result_manifests({"queries": good, "memory": other})
    with pytest.raises(ValueError, match="incomplete run manifest"):
        validate_result_manifests({"queries": {}})


def test_chart_render_failure_preserves_previous_artifacts(monkeypatch, tmp_path):
    import json

    import charts.generate_all as charts

    inputs, output = tmp_path / "results", tmp_path / "charts"
    inputs.mkdir()
    output.mkdir()
    manifest = {"complete": True, "run_id": "one", "source": {"sha256": "abc"}}
    (inputs / "queries.json").write_text(json.dumps({"manifest": manifest}))
    (output / "first.png").write_bytes(b"previous chart")
    (output / "manifest.json").write_text("previous manifest")

    def render(_data, destination):
        destination.write_bytes(b"new chart")
        return destination

    def fail(_data, _destination):
        raise ValueError("invalid chart data")

    monkeypatch.setattr(
        charts,
        "_CHARTS",
        (
            ("queries", "first", render, "first.png"),
            ("queries", "second", fail, "second.png"),
        ),
    )
    with pytest.raises(ValueError, match="invalid chart data"):
        charts.generate_all(inputs, output)
    assert (output / "first.png").read_bytes() == b"previous chart"
    assert (output / "manifest.json").read_text() == "previous manifest"
    assert not (output / "second.png").exists()


def test_chart_manifest_binds_exact_inputs_and_removes_unselected_old_chart(
    monkeypatch, tmp_path
):
    import json

    import charts.generate_all as charts
    from benchmarks.reporting.provenance import file_sha256

    inputs, output = tmp_path / "results", tmp_path / "charts"
    inputs.mkdir()
    output.mkdir()
    manifest = {"complete": True, "run_id": "one", "source": {"sha256": "abc"}}
    (inputs / "queries.json").write_text(json.dumps({"manifest": manifest}))
    (output / "memory.png").write_bytes(b"old run memory chart")

    def render(_data, destination):
        destination.write_bytes(b"current run chart")
        return destination

    monkeypatch.setattr(
        charts,
        "_CHARTS",
        (
            ("queries", "query", render, "query.png"),
            ("memory", "memory", render, "memory.png"),
        ),
    )
    generated = charts.generate_all(inputs, output)
    exported = json.loads((output / "manifest.json").read_text())
    assert generated == {"query": output / "query.png"}
    assert exported["input_sha256"]["queries.json"] == file_sha256(
        inputs / "queries.json"
    )
    assert exported["chart_sha256"]["query.png"] == file_sha256(output / "query.png")
    assert exported["missing_result_families"] == ["memory"]
    assert exported["selected_charts_complete"] is True
    assert exported["full_suite_charts_complete"] is False
    assert not (output / "memory.png").exists()


def test_small_sweeps_export_all_control_observations(monkeypatch):
    from benchmarks import bench_bitemporal, bench_chunk_granularity, bench_scale
    from src.store.chunk_boundary import MonthlyBoundary

    monkeypatch.setattr(bench_scale, "ROWS_PER_SYMBOL", 10)
    scale = bench_scale.run_scale_point(
        label="test", num_symbols=2, num_trials=2, warmup=0
    )
    assert scale["fixture"]["rows"] == 20
    assert scale["query_requested_trials"] == 2
    assert scale["query_num_trials"] == 6
    for metrics in scale["queries"].values():
        assert len(metrics["acquisition_order"]) == 18
        assert len(metrics["full_scan"]["raw_latency_samples_seconds"]) == 6
        assert len(metrics["vectorized_full_scan"]["raw_latency_samples_seconds"]) == 6

    monkeypatch.setattr(bench_bitemporal, "NUM_SYMBOLS", 2)
    monkeypatch.setattr(bench_bitemporal, "ROWS_PER_SYMBOL", 10)
    monkeypatch.setattr(bench_bitemporal, "QUERY_NUM_TRIALS", 2)
    monkeypatch.setattr(bench_bitemporal, "QUERY_WARMUP", 0)
    monkeypatch.setattr(bench_bitemporal, "PUSHDOWN_ISOLATED_TRIALS", 2)
    bitemporal = bench_bitemporal.run_correction_rate(label="test", correction_rate=0.5)
    assert set(bitemporal["scenarios"]) == {
        "current_pre_compaction",
        "current_post_compaction",
        "as_of_pre_compaction",
        "as_of_post_compaction",
    }
    for metrics in bitemporal["scenarios"].values():
        assert len(metrics["acquisition_order"]) == 18
        assert len(metrics["pushdown_eval_raw_samples_seconds"]) == 2
    assert (
        "distinguishable_from_noise"
        not in bitemporal["compaction_comparisons"]["current"]
    )

    monkeypatch.setattr(bench_chunk_granularity, "NUM_SYMBOLS", 2)
    monkeypatch.setattr(bench_chunk_granularity, "ROWS_PER_SYMBOL", 150)
    monkeypatch.setattr(bench_chunk_granularity, "QUERY_SYMBOL", "SYN000000")
    monkeypatch.setattr(bench_chunk_granularity, "QUERY_NUM_TRIALS", 2)
    monkeypatch.setattr(bench_chunk_granularity, "QUERY_WARMUP", 0)
    granularity = bench_chunk_granularity.run_granularity_point(
        {
            "key": "monthly",
            "label": "Monthly",
            "window": "1 month",
            "boundary": MonthlyBoundary(),
        }
    )
    assert granularity["fixture"]["rows"] == 300
    assert len(granularity["acquisition_order"]) == 18
    assert len(granularity["vectorized_full_scan"]["raw_latency_samples_seconds"]) == 6
    assert granularity["aggregate"]["num_queries"] == 3  # 1 present symbol x 3 windows


def test_chart_export_rejects_results_replaced_during_render(monkeypatch, tmp_path):
    import json

    import charts.generate_all as charts

    inputs, output = tmp_path / "results", tmp_path / "charts"
    inputs.mkdir()
    manifest = {"complete": True, "run_id": "one", "source": {"sha256": "abc"}}
    path = inputs / "queries.json"
    path.write_text(json.dumps({"manifest": manifest, "value": "old"}))

    def render(data, destination):
        assert data["value"] == "old"
        path.write_text(json.dumps({"manifest": manifest, "value": "new"}))
        destination.write_bytes(b"chart from old data")
        return destination

    monkeypatch.setattr(charts, "_CHARTS", (("queries", "query", render, "query.png"),))
    with pytest.raises(ValueError, match="changed during chart generation"):
        charts.generate_all(inputs, output)
    assert not (output / "query.png").exists()
    assert not (output / "manifest.json").exists()


def test_suite_reports_failure_when_chart_export_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(run_suite, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_suite, "LOG_PATH", tmp_path / "suite.log")
    monkeypatch.setattr(
        run_suite,
        "_run_benchmarks_1_and_2",
        lambda: (["1", "Queries", "PASSED", "ok"], ["2", "Memory", "PASSED", "ok"]),
    )
    for number in (3, 4, 5):
        monkeypatch.setattr(
            run_suite,
            f"_run_benchmark_{number}",
            lambda: ["x", "other", "PASSED", "ok"],
        )

    def fail(**_kwargs):
        raise ValueError("result provenance mismatch")

    monkeypatch.setattr(run_suite, "generate_all_charts", fail)
    assert run_suite.run_suite() == 1
    assert "COMPLETED WITH FAILURES" in (tmp_path / "suite.log").read_text()


# Bitemporal AS OF scenario is not best-case


def test_corrections_have_multiple_versions_and_spread_transaction_times():
    from benchmarks import bench_bitemporal as bb

    frame = generate_synthetic_ohlcv(num_symbols=4, rows_per_symbol=200, seed=1)
    corrections = bb.apply_corrections(frame, 0.25, seed=2)
    keys = {c["key_index"] for c in corrections}
    assert len(corrections) > len(keys)  # some keys have >1 version
    by_key: dict[int, list] = {}
    for c in corrections:
        by_key.setdefault(c["key_index"], []).append(c)
        assert c["transaction_time"] > c["timestamp"]
    for versions in by_key.values():
        times = [v["transaction_time"] for v in versions]
        assert times == sorted(set(times))  # strictly increasing per key
    # Not parked after the last base row, as in the old best-case design.
    last_base = frame["timestamp"].max()
    assert any(c["transaction_time"] <= last_base for c in corrections)
    assert any(c["transaction_time"] > last_base for c in corrections)


def test_as_of_sweep_has_mixed_visibility_and_verified_equivalence(monkeypatch):
    from benchmarks import bench_bitemporal as bb

    monkeypatch.setattr(bb, "NUM_SYMBOLS", 3)
    monkeypatch.setattr(bb, "ROWS_PER_SYMBOL", 150)
    monkeypatch.setattr(bb, "QUERY_NUM_TRIALS", 2)
    monkeypatch.setattr(bb, "QUERY_WARMUP", 0)
    monkeypatch.setattr(bb, "PUSHDOWN_ISOLATED_TRIALS", 2)
    entry = bb.run_correction_rate(label="30%", correction_rate=0.30)
    sweep = entry["as_of_sweep"]
    assert len(sweep) == len(bb.VISIBLE_FRACTIONS) + 1
    visible = [p["visible_correction_versions"] for p in sweep[:5]]
    assert visible[0] == 0
    assert visible == sorted(visible)
    assert visible[-1] == entry["num_corrections"]
    # Genuinely mixed: some cutoff leaves corrections both visible and hidden.
    assert any(0 < v < entry["num_corrections"] for v in visible)
    primary = [p for p in sweep if p["is_primary"]]
    assert len(primary) == 1
    assert primary[0]["scenario"] is entry["scenarios"]["as_of_pre_compaction"]
    # Pruning differs across cutoffs (it is not a single best-case number).
    skipped = {round(p["scenario"]["chunks_skipped_ratio"], 6) for p in sweep}
    assert len(skipped) > 1
    for point in sweep:
        # measure_query_paths asserts exact 3-way equality before reporting.
        assert (
            point["scenario"]["full_scan"]["row_count"]
            == (point["scenario"]["row_count"])
        )


def test_suite_as_of_summary_is_data_driven_range(monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(run_suite, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_suite.bench_bitemporal, "main", lambda: 0)

    def point(label, speedup, primary):
        return {
            "label": label,
            "is_primary": primary,
            "scenario": {"query_speedup": speedup},
        }

    (tmp_path / "bitemporal_sweep.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "label": "20%",
                        "as_of_sweep": [
                            point("0%", 9.0, False),
                            point("50%", 1.7, True),
                            point("100%", 1.0, False),
                        ],
                    }
                ]
            }
        )
    )
    row = run_suite._run_benchmark_4()
    assert "1.00x-9.00x" in row[3]
    assert "mid-history cutoff 1.70x" in row[3]


# Chunk-granularity output is data-driven


def _granularity_point(key, label, chunks, push, row_wise):
    return {
        "key": key,
        "label": label,
        "total_chunks": chunks,
        "aggregate": {
            "mean_pushdown_ms": push,
            "mean_row_wise_ms": row_wise,
            "mean_vectorized_ms": row_wise * 0.9,
            "mean_chunks_skipped_ratio": 0.9,
            "geomean_row_wise_over_pushdown": row_wise / push,
        },
    }


def test_granularity_findings_do_not_hardwire_the_default():
    from benchmarks.bench_chunk_granularity import summarize_granularity

    points = [
        _granularity_point("weekly", "Weekly", 800, 4.0, 150.0),
        _granularity_point("monthly", "Monthly", 200, 2.6, 60.0),
        _granularity_point("quarterly", "Quarterly", 80, 2.4, 30.0),
    ]
    findings = summarize_granularity(points)
    assert findings["lowest_pushdown_label"] == "Quarterly"
    assert findings["configured_default_label"] == "Monthly"
    assert findings["default_matches_lowest_pushdown"] is False
    assert "did NOT have the lowest" in findings["default_vs_best"]
    assert findings["highest_speedup_label"] == "Weekly"
    # Baseline and pushdown latencies are both reported in the explanation.
    assert "row-wise baseline latency" in findings["speedup_explanation"]
    assert "not evidence" in findings["speedup_explanation"]
    points[1]["aggregate"]["mean_pushdown_ms"] = 1.0
    assert summarize_granularity(points)["default_matches_lowest_pushdown"] is True


# Charts


def _timing(mean, std=None):
    return {
        "mean_latency_seconds": mean,
        "std_latency_seconds": mean * 0.1 if std is None else std,
    }


def _sweep_point(selectivity, full, push, vec):
    return {
        "actual_selectivity": selectivity,
        "speedup": full / push,
        "chunks_skipped_ratio": 0.5,
        "full_scan": _timing(full),
        "pushdown_query": {**_timing(push), "vectorized_full_scan": _timing(vec)},
    }


def test_selectivity_charts_skip_sub_threshold_points_instead_of_aborting(tmp_path):
    from charts import chart_chunks_skipped, chart_speedup_vs_selectivity

    payload = {
        "selectivity_sweep": {
            "points": [
                _sweep_point(1.0, 0.06, 0.02, 0.05),
                _sweep_point(0.1, 0.06, 0.004, 0.05),
                _sweep_point(0.0004, 0.06, 0.001, 0.05),  # below 0.1%
                _sweep_point(0.0, 0.06, 0.001, 0.05),  # zero rows
            ]
        }
    }
    series, skipped = chart_speedup_vs_selectivity._extract_series(payload)
    assert [i for i, _ in skipped] == [2, 3]
    assert len(series["row_wise"]) == len(series["vectorized"]) == 2
    # Vectorized control is a distinct series, not a copy of the baseline.
    assert series["vectorized"][0][1] != series["row_wise"][0][1]
    one = chart_speedup_vs_selectivity.generate_chart(payload, tmp_path / "one.png")
    three = chart_chunks_skipped.generate_chart(payload, tmp_path / "three.png")
    assert one.stat().st_size > 5_000 and three.stat().st_size > 5_000
    assert chart_chunks_skipped._extract_series(payload)[2] == [0.0004, 0.0]


def test_latency_chart_includes_vectorized_control_and_row_counts(tmp_path):
    from charts import chart_latency_bar

    fixed = {}
    for index, name in enumerate(("Q1", "Q2", "Q3", "Q4", "Q5"), start=1):
        point = _sweep_point(0.01 * index, 0.06, 0.001 * index, 0.04)
        point["pushdown_query"]["row_count"] = index
        fixed[name] = point
    data, rows, speedups = chart_latency_bar._extract_series({"fixed_queries": fixed})
    assert set(data) == {"full_scan", "vectorized_full_scan", "pushdown_query"}
    assert rows == [1, 2, 3, 4, 5]
    assert speedups[0] == pytest.approx(60.0)
    out = chart_latency_bar.generate_chart(
        {"fixed_queries": fixed}, tmp_path / "latency.png"
    )
    assert out.stat().st_size > 5_000


def test_memory_chart_excludes_rss_and_requires_traced_peaks(tmp_path):
    from charts import chart_memory

    engines = {
        key: {
            "median_peak_bytes": value,
            "min_peak_bytes": value - 1,
            "max_peak_bytes": value + 1,
        }
        for key, value in (
            ("full_scan", 2_000_000),
            ("vectorized_full_scan", 1_500_000),
            ("pushdown_query", 40_000),
        )
    }
    memory = {
        "column_payload_bytes": 200_400,
        "raw_uncompressed_bytes": 500_000,
        "chunk_store_bytes": 1_300_000,
        "q1_peak_rss_bytes": 103_000_000,  # must NOT appear on the chart
        "traced_query_peaks": {"Q1": {"row_count": 1, "engines": engines}},
    }
    labels, values, _ = chart_memory._extract_values(memory)
    assert max(values) < 2.0  # no 103 MB RSS bar
    assert not any("RSS" in label for label in labels)
    assert (
        chart_memory.generate_chart(memory, tmp_path / "m.png").stat().st_size > 5_000
    )
    with pytest.raises(ValueError, match="traced_query_peaks"):
        chart_memory._extract_values(
            {k: v for k, v in memory.items() if k != "traced_query_peaks"}
        )


def test_chunks_skipped_chart_metadata_does_not_claim_compression():
    import inspect

    from charts import chart_chunks_skipped

    source = inspect.getsource(chart_chunks_skipped)
    assert "compressed" not in source.lower()


# data/download_data.py


@pytest.fixture
def downloader(monkeypatch, tmp_path):
    import importlib.util
    import sys
    import types
    from pathlib import Path

    monkeypatch.setitem(sys.modules, "yfinance", types.ModuleType("yfinance"))
    path = Path(__file__).resolve().parent.parent / "data" / "download_data.py"
    spec = importlib.util.spec_from_file_location("pitdb_download_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(module, "DATASET_A_MERGED", tmp_path / "all_symbols_daily.csv")
    monkeypatch.setattr(module, "DATASET_B_MERGED", tmp_path / "all_symbols_1m.csv")
    monkeypatch.setattr(module, "SYMBOLS", ["AAA", "BBB"])
    monkeypatch.setattr(module.time, "sleep", lambda _s: None)
    return module


def _fake_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1, 2],
        }
    )


def test_download_failure_exits_nonzero_and_preserves_existing_files(
    downloader, tmp_path, monkeypatch
):
    merged = downloader.DATASET_A_MERGED
    merged.write_text("original fixture bytes\n")
    monkeypatch.setattr(
        downloader,
        "_download_one",
        lambda symbol, *a, **k: _fake_frame() if symbol == "AAA" else None,
    )
    assert downloader.main(["--force"]) == 1
    assert merged.read_text() == "original fixture bytes\n"
    assert not (tmp_path / "AAA_daily.csv").exists()  # nothing written at all


def test_download_allow_partial_writes_atomically_and_names_missing(
    downloader, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        downloader,
        "_download_one",
        lambda symbol, *a, **k: _fake_frame() if symbol == "AAA" else None,
    )
    assert downloader.main(["--force", "--allow-partial"]) == 0
    assert "OMITS BBB" in capsys.readouterr().err
    merged = pd.read_csv(downloader.DATASET_A_MERGED)
    assert set(merged["symbol"]) == {"AAA"}
    assert not list(tmp_path.glob("*.tmp"))


def test_download_success_exits_zero(downloader, monkeypatch):
    monkeypatch.setattr(downloader, "_download_one", lambda *a, **k: _fake_frame())
    assert downloader.main(["--force"]) == 0
    assert set(pd.read_csv(downloader.DATASET_B_MERGED)["symbol"]) == {"AAA", "BBB"}
