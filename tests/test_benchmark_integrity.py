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


def test_numeric_memory_comparison_has_identical_columns(monkeypatch):
    frame = generate_synthetic_ohlcv(num_symbols=1, rows_per_symbol=4, seed=1)
    store = ChunkStore()
    store.ingest_dataframe(frame)
    monkeypatch.setattr(
        "benchmarks.bench_memory._measure_peak_rss", lambda fn: (100, 110)
    )
    memory = run_memory_benchmarks(frame, store)
    expected = sum(frame[column].to_numpy().nbytes for column in store.value_columns)
    assert memory["raw_numeric_payload_bytes"] == expected
    assert memory["column_payload_bytes"] == expected
    assert memory["numeric_payload_columns"] == list(store.value_columns)
    assert memory["numeric_payload_ratio"] == 1.0
    assert memory["raw_uncompressed_bytes"] > expected


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
        assert len(control["raw_latency_samples_seconds"]) == 2
        assert len(entry["pushdown_query"]["acquisition_order"]) == 6
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
    for metrics in scale["queries"].values():
        assert len(metrics["acquisition_order"]) == 6
        assert len(metrics["full_scan"]["raw_latency_samples_seconds"]) == 2
        assert len(metrics["vectorized_full_scan"]["raw_latency_samples_seconds"]) == 2

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
        assert len(metrics["acquisition_order"]) == 6
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
    assert len(granularity["acquisition_order"]) == 6
    assert len(granularity["vectorized_full_scan"]["raw_latency_samples_seconds"]) == 2


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
