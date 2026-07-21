"""Tests for the benchmark suite's presentation-only reporting layer.

These tests cover formatting (benchmarks/report.py), structural rendering
(benchmarks/reporting/console.py), the correctness-gate wrapper
(benchmarks/reporting/validation.py), and provenance snapshotting
(benchmarks/reporting/environment.py). None of them exercise benchmark
computation, timing, or the query engine -- see tests/test_pushdown.py,
tests/test_executor.py, etc. for that. These tests exist to prove formatting
changes never touch benchmark semantics: every formatter here takes a
plain value in and returns a string, nothing else.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.report import (
    format_bytes,
    format_ms,
    format_pct,
    format_x,
    render_box_table,
    render_table,
)
from benchmarks.reporting import console
from benchmarks.reporting.environment import git_commit, hardware_snapshot
from benchmarks.reporting.validation import (
    CorrectnessOutcome,
    assert_equivalent_and_report,
    render_correctness_block,
    speedup_cell,
)

# ---------------------------------------------------------------------------
# benchmarks/report.py -- render_table
# ---------------------------------------------------------------------------


class TestRenderTable:
    def test_aligns_first_column_left_and_rest_right(self) -> None:
        table = render_table(["Name", "Value"], [["a", "1"], ["bb", "22"]])
        lines = table.splitlines()
        header, separator, *rows = lines
        assert header.startswith("Name")
        assert set(separator) <= {"-", " "}
        assert len(rows) == 2
        # Every line is padded to the same total width...
        assert len({len(line) for line in lines}) == 1
        # ...and the right-aligned "Value" column ends at the same column
        # in the header and in every data row.
        assert header.rstrip().endswith("Value")
        assert rows[0].rstrip().endswith("1")
        assert rows[1].rstrip().endswith("22")
        assert len(header) == len(rows[0]) == len(rows[1])

    def test_column_width_follows_longest_cell(self) -> None:
        table = render_table(["X"], [["short"], ["a very long cell value"]])
        for line in table.splitlines():
            assert len(line) == len("a very long cell value")

    def test_empty_rows_still_renders_header_and_separator(self) -> None:
        table = render_table(["A", "B"], [])
        assert table.splitlines() == ["A  B", "-  -"]

    def test_rejects_empty_headers(self) -> None:
        with pytest.raises(ValueError):
            render_table([], [])

    def test_rejects_mismatched_row_length(self) -> None:
        with pytest.raises(ValueError):
            render_table(["A", "B"], [["only one cell"]])


# ---------------------------------------------------------------------------
# benchmarks/report.py -- number formatters
# ---------------------------------------------------------------------------


class TestFormatMs:
    def test_typical_value(self) -> None:
        assert format_ms(1.23456) == "1.235ms"

    def test_zero(self) -> None:
        assert format_ms(0.0) == "0.000ms"

    def test_negative(self) -> None:
        assert format_ms(-1.5) == "-1.500ms"

    def test_nan_does_not_raise(self) -> None:
        assert format_ms(float("nan")) == "nanms"

    def test_positive_infinity_does_not_raise(self) -> None:
        assert format_ms(float("inf")) == "infms"

    def test_negative_infinity_does_not_raise(self) -> None:
        assert format_ms(float("-inf")) == "-infms"


class TestFormatX:
    def test_typical_value(self) -> None:
        assert format_x(40.783) == "40.78x"

    def test_zero(self) -> None:
        assert format_x(0.0) == "0.00x"

    def test_infinity_does_not_raise(self) -> None:
        assert format_x(math.inf) == "infx"


class TestFormatPct:
    def test_typical_fraction(self) -> None:
        assert format_pct(0.9958) == "99.58%"

    def test_zero(self) -> None:
        assert format_pct(0.0) == "0.00%"

    def test_full(self) -> None:
        assert format_pct(1.0) == "100.00%"

    def test_nan_does_not_raise(self) -> None:
        assert format_pct(float("nan")) == "nan%"


class TestFormatBytes:
    def test_bytes_under_1024(self) -> None:
        assert format_bytes(512) == "512 bytes"

    def test_zero(self) -> None:
        assert format_bytes(0) == "0 bytes"

    def test_kib(self) -> None:
        assert format_bytes(2048) == "2.00 KiB"

    def test_mib(self) -> None:
        assert format_bytes(5 * 1024 * 1024) == "5.00 MiB"

    def test_gib(self) -> None:
        assert format_bytes(3 * 1024**3) == "3.00 GiB"

    def test_negative(self) -> None:
        assert format_bytes(-2048) == "-2.00 KiB"

    def test_nan_does_not_raise(self) -> None:
        assert format_bytes(float("nan")) == "nan"

    def test_infinity_does_not_raise(self) -> None:
        assert format_bytes(float("inf")) == "inf"


# ---------------------------------------------------------------------------
# benchmarks/reporting/console.py
# ---------------------------------------------------------------------------


class TestPhase:
    def test_known_label_produces_tagged_line(self) -> None:
        line = console.phase("BUILD", "ingested 5,010 rows")
        assert line.startswith("[BUILD]")
        assert "ingested 5,010 rows" in line

    def test_unknown_label_rejected(self) -> None:
        with pytest.raises(ValueError):
            console.phase("NOT_A_PHASE", "message")

    def test_every_declared_phase_is_accepted(self) -> None:
        for label in console.PHASES:
            assert console.phase(label, "x").startswith(f"[{label}]")


class TestRule:
    def test_default_character(self) -> None:
        assert set(console.rule()) == {"="}

    def test_custom_character(self) -> None:
        assert set(console.rule("-")) == {"-"}

    def test_rejects_multi_character_input(self) -> None:
        with pytest.raises(ValueError):
            console.rule("==")


class TestBenchmarkHeader:
    def test_contains_index_name_and_question(self) -> None:
        header = console.benchmark_header(
            index=3,
            total=5,
            name="Scale Sweep",
            question="Does it scale?",
            config_lines=["Trials: 15"],
        )
        assert "BENCHMARK 3/5" in header
        assert "Scale Sweep" in header
        assert "Does it scale?" in header
        assert "Trials: 15" in header


class TestRenderBoxTable:
    def test_all_columns_left_aligned_with_unicode_borders(self) -> None:
        table = render_box_table(["Engine", "Latency"], [["Full scan", "52ms"]])
        lines = table.splitlines()
        assert lines[0].startswith("┌") and lines[0].endswith("┐")
        assert lines[-1].startswith("└") and lines[-1].endswith("┘")
        assert "│ Engine" in lines[1]
        assert "│ Full scan" in lines[3]

    def test_column_width_follows_longest_cell(self) -> None:
        table = render_box_table(["X"], [["a very long cell value"]])
        for line in table.splitlines():
            assert len(line) == len(table.splitlines()[0])

    def test_rejects_empty_headers(self) -> None:
        with pytest.raises(ValueError):
            render_box_table([], [])

    def test_rejects_mismatched_row_length(self) -> None:
        with pytest.raises(ValueError):
            render_box_table(["A", "B"], [["only one cell"]])


class TestPitdbHeaderTitleFooter:
    def test_pitdb_header_contains_index_total_and_title(self) -> None:
        header = console.pitdb_header(
            index=2, total=5, title="Memory-Footprint Analysis"
        )
        assert "PITDB BENCHMARK 2/5" in header
        assert "Memory-Footprint Analysis" in header
        assert header.startswith("=" * 80)

    def test_pitdb_title_is_a_bare_double_rule_block(self) -> None:
        block = console.pitdb_title("PITDB BENCHMARK SUITE SUMMARY")
        lines = block.splitlines()
        assert lines[0] == lines[2] == "=" * 80
        assert "PITDB BENCHMARK SUITE SUMMARY" in lines[1]

    def test_pitdb_footer_names_the_completed_benchmark(self) -> None:
        footer = console.pitdb_footer(index=4)
        assert "BENCHMARK 4 COMPLETE" in footer


class TestDashSection:
    def test_single_line_tag(self) -> None:
        block = console.dash_section("[FIXED QUERIES]")
        lines = block.splitlines()
        assert lines[0] == lines[2] == "-" * 80
        assert lines[1] == "[FIXED QUERIES]"

    def test_multi_line_tag_plus_subtitle(self) -> None:
        block = console.dash_section("[STORAGE REPRESENTATION]", " subtitle")
        lines = block.splitlines()
        assert lines[1] == "[STORAGE REPRESENTATION]"
        assert lines[2] == " subtitle"

    def test_rejects_no_lines(self) -> None:
        with pytest.raises(ValueError):
            console.dash_section()


class TestFieldBlockAndTitledBlock:
    def test_field_block_aligns_colons_to_longest_label_in_this_block_only(
        self,
    ) -> None:
        block = console.field_block([("A", "1"), ("Longer label", "2")])
        lines = block.splitlines()
        # Both colons land at the same column, derived from "Longer label".
        colon_positions = {line.index(":") for line in lines}
        assert len(colon_positions) == 1

    def test_field_block_rejects_empty_pairs(self) -> None:
        with pytest.raises(ValueError):
            console.field_block([])

    def test_titled_block_prefixes_title_line(self) -> None:
        block = console.titled_block("Configuration", [("Rows", "5,010")])
        lines = block.splitlines()
        assert lines[0] == "Configuration"
        assert "Rows" in lines[1] and "5,010" in lines[1]


class TestTitledParagraphAndWrapText:
    def test_wrap_text_indents_every_line(self) -> None:
        wrapped = console.wrap_text("a " * 60, indent=4, width=40)
        for line in wrapped.splitlines():
            assert line.startswith("    ")

    def test_titled_paragraph_indents_body_deeper_than_heading(self) -> None:
        block = console.titled_paragraph("Limitation", "Some text here.", indent=2)
        lines = block.splitlines()
        assert lines[0] == "  Limitation"
        assert lines[1].startswith("    ")


class TestInterpretation:
    def test_contains_all_three_labeled_fields(self) -> None:
        block = console.interpretation(
            observation="obs", evidence="evi", limitation="lim"
        )
        assert "Observation : obs" in block
        assert "Evidence    : evi" in block
        assert "Limitation  : lim" in block


# ---------------------------------------------------------------------------
# benchmarks/reporting/validation.py
# ---------------------------------------------------------------------------


class TestCorrectnessBlock:
    def test_pass_block_shows_passed_status(self) -> None:
        outcome = CorrectnessOutcome(
            name="Q1", full_scan_row_count=21, pushdown_row_count=21, passed=True
        )
        block = render_correctness_block(outcome)
        assert "PASSED" in block
        assert "FAILED" not in block
        assert "21" in block

    def test_fail_block_shows_failed_status_and_detail(self) -> None:
        outcome = CorrectnessOutcome(
            name="Q1",
            full_scan_row_count=21,
            pushdown_row_count=19,
            passed=False,
            detail="row counts differ",
        )
        block = render_correctness_block(outcome)
        assert "FAILED" in block
        assert "row counts differ" in block


class TestAssertEquivalentAndReport:
    def test_pass_path_prints_and_returns_passed_outcome(self, capsys) -> None:
        outcome = assert_equivalent_and_report(
            lambda: None,
            name="Q1",
            full_scan_row_count=21,
            pushdown_row_count=21,
        )
        assert outcome.passed is True
        printed = capsys.readouterr().out
        assert "PASSED" in printed
        assert "[VALIDATE]" in printed

    def test_fail_path_prints_and_reraises_same_exception_type(self, capsys) -> None:
        def _fail() -> None:
            raise AssertionError("row counts differ")

        with pytest.raises(AssertionError, match="row counts differ"):
            assert_equivalent_and_report(
                _fail, name="Q1", full_scan_row_count=21, pushdown_row_count=19
            )
        printed = capsys.readouterr().out
        assert "FAILED" in printed
        assert "[ERROR]" in printed

    def test_only_assertion_error_is_intercepted(self) -> None:
        def _fail_differently() -> None:
            raise TypeError("not a correctness failure")

        with pytest.raises(TypeError):
            assert_equivalent_and_report(
                _fail_differently,
                name="Q1",
                full_scan_row_count=21,
                pushdown_row_count=21,
            )


class TestSpeedupCell:
    def test_passed_outcome_uses_formatter(self) -> None:
        outcome = CorrectnessOutcome("Q1", 21, 21, passed=True)
        assert speedup_cell(outcome, 40.78, format_x) == "40.78x"

    def test_failed_outcome_is_suppressed_not_formatted(self) -> None:
        outcome = CorrectnessOutcome("Q1", 21, 19, passed=False, detail="mismatch")
        assert speedup_cell(outcome, 999.0, format_x) == "SUPPRESSED"


# ---------------------------------------------------------------------------
# benchmarks/reporting/environment.py
# ---------------------------------------------------------------------------


class TestEnvironment:
    def test_git_commit_returns_a_non_empty_string(self) -> None:
        commit = git_commit()
        assert isinstance(commit, str)
        assert commit != ""

    def test_hardware_snapshot_has_expected_keys(self) -> None:
        snapshot = hardware_snapshot()
        for key in (
            "processor",
            "system",
            "release",
            "python_version",
            "machine",
            "total_memory_bytes",
            "logical_cpu_count",
            "physical_cpu_count",
        ):
            assert key in snapshot
        assert snapshot["total_memory_bytes"] > 0
        assert snapshot["logical_cpu_count"] >= 1
