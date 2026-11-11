"""CLI benchmark orchestration must preserve one run and failure status."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks import run_all, run_suite
from charts import generate_all
from src import pipeline
from src.store.chunk_boundary import DurationBoundary, MonthlyBoundary


@pytest.mark.parametrize("status", [0, 1])
def test_benchmark_flag_uses_coordinated_suite_and_returns_its_status(
    monkeypatch, status
):
    ingest = Mock()
    benchmark = Mock(return_value=status)
    monkeypatch.setattr(pipeline, "run_pipeline", ingest)
    monkeypatch.setattr(run_suite, "run_suite", benchmark)
    monkeypatch.setattr(run_all, "run_all", Mock())
    monkeypatch.setattr(generate_all, "generate_all", Mock())
    assert pipeline.main(["--data-csv", "example.csv", "--benchmark"]) == status
    ingest.assert_called_once()
    benchmark.assert_called_once_with()


# --- argument parsing, execution, and error exits -----------------------------


def _write_csv(path: Path, timestamps: list[str], symbols: list[str]) -> Path:
    rows = ["timestamp,symbol,open,high,low,close,volume"]
    for index, (stamp, symbol) in enumerate(zip(timestamps, symbols)):
        price = 100.0 + index
        rows.append(f"{stamp},{symbol},{price},{price + 1},{price - 1},{price},1000")
    path.write_text("\n".join(rows) + "\n")
    return path


@pytest.fixture
def daily_csv(tmp_path: Path) -> Path:
    stamps = ["2024-01-02", "2024-01-03", "2024-02-01", "2024-01-02"]
    symbols = ["AAPL", "AAPL", "AAPL", "MSFT"]
    # Per-symbol ascending order is required; the interleaving is arbitrary.
    return _write_csv(tmp_path / "daily.csv", stamps, symbols)


@pytest.fixture
def minute_csv(tmp_path: Path) -> Path:
    stamps = [
        "2024-01-02T14:30:00Z",
        "2024-01-02T14:31:00Z",
        "2024-01-02T15:30:00Z",
        "2024-01-02T15:31:00Z",
    ]
    return _write_csv(tmp_path / "minute.csv", stamps, ["AAPL"] * 4)


def test_parsing_defaults_to_monthly_boundary_and_no_queries(monkeypatch, tmp_path):
    ingest = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline", ingest)

    assert pipeline.main(["--data-csv", str(tmp_path / "x.csv")]) == 0

    kwargs = ingest.call_args.kwargs
    assert kwargs["data_csv_path"] == tmp_path / "x.csv"
    assert isinstance(kwargs["chunk_boundary"], MonthlyBoundary)
    assert kwargs["queries"] is None


def test_parsing_hourly_strategy_and_repeated_queries(monkeypatch):
    ingest = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline", ingest)

    status = pipeline.main(
        [
            "--data-csv",
            "x.csv",
            "--chunk-strategy",
            "hourly",
            "--query",
            "SELECT * FROM data",
            "--query",
            "SELECT * FROM data WHERE symbol = 'A'",
        ]
    )

    assert status == 0
    kwargs = ingest.call_args.kwargs
    assert isinstance(kwargs["chunk_boundary"], DurationBoundary)
    assert kwargs["chunk_boundary"].duration == timedelta(hours=1)
    assert kwargs["queries"] == [
        "SELECT * FROM data",
        "SELECT * FROM data WHERE symbol = 'A'",
    ]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--data-csv", "x.csv", "--chunk-strategy", "weekly"],
        ["--data-csv", "x.csv", "--query", ""],
        ["--data-csv", "x.csv", "--query", "   "],
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as raised:
        pipeline.main(argv)

    assert raised.value.code == 2
    assert "Traceback" not in capsys.readouterr().err


def test_main_executes_queries_and_prints_summary(daily_csv, capsys):
    status = pipeline.main(
        [
            "--data-csv",
            str(daily_csv),
            "--query",
            "SELECT * FROM data WHERE symbol = 'AAPL'",
            "--query",
            "SELECT * FROM data WHERE symbol = 'MSFT'",
        ]
    )

    out = capsys.readouterr()
    assert status == 0
    assert out.err == ""
    assert "Chunk strategy:          MonthlyBoundary" in out.out
    assert "Query 1: SELECT * FROM data WHERE symbol = 'AAPL'" in out.out
    assert "Query 2:" in out.out
    assert "rows:           3" in out.out
    assert "rows:           1" in out.out


def test_main_hourly_strategy_end_to_end(minute_csv, capsys):
    status = pipeline.main(
        ["--data-csv", str(minute_csv), "--chunk-strategy", "hourly"]
    )

    out = capsys.readouterr().out
    assert status == 0
    assert "DurationBoundary" in out
    assert "Total chunks:            2" in out


def test_summary_reports_raw_to_stored_ratio_not_compression(daily_csv, capsys):
    summary = pipeline.run_pipeline(daily_csv)["summary"]

    assert summary["raw_to_stored_ratio"] == pytest.approx(1.0)
    assert "compression_ratio" not in summary
    assert "Compression" not in capsys.readouterr().out


def test_run_pipeline_still_raises_programmatically(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.run_pipeline(tmp_path / "missing.csv")


def _assert_clean_failure(capsys, *needles):
    captured = capsys.readouterr()
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("error: ")
    assert "Traceback" not in captured.err
    for needle in needles:
        assert needle in lines[0]


def test_main_missing_csv_exits_nonzero_without_traceback(tmp_path, capsys):
    assert pipeline.main(["--data-csv", str(tmp_path / "missing.csv")]) == 1
    _assert_clean_failure(capsys, "does not exist")


def test_main_directory_as_csv_exits_nonzero(tmp_path, capsys):
    assert pipeline.main(["--data-csv", str(tmp_path)]) == 1
    _assert_clean_failure(capsys)


def test_main_non_utf8_csv_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "binary.csv"
    path.write_bytes(b"timestamp,symbol\n\xff\xfe\x80,\xc3\x28\n")

    assert pipeline.main(["--data-csv", str(path)]) == 1
    _assert_clean_failure(capsys, "UTF-8")


def test_main_header_only_csv_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "header.csv"
    path.write_text("timestamp,symbol,open,high,low,close,volume\n")

    assert pipeline.main(["--data-csv", str(path)]) == 1
    _assert_clean_failure(capsys, "no data rows")


def test_main_invalid_data_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,symbol,open\n2024-01-02,AAPL,1\n")

    assert pipeline.main(["--data-csv", str(path)]) == 1
    _assert_clean_failure(capsys, "missing required columns")


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE data",
        "SELECT * FROM data JOIN other ON 1 = 1",
        "SELECT * FROM data WHERE ((",
    ],
)
def test_main_unsupported_query_exits_nonzero(daily_csv, capsys, sql):
    assert pipeline.main(["--data-csv", str(daily_csv), "--query", sql]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: unsupported or invalid --query")
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err
    # Syntax errors are caught before the slow ingestion starts.
    assert "Chunk Store Summary" not in captured.out


def test_benchmark_requires_source_checkout(monkeypatch, daily_csv, capsys):
    ingest = Mock()
    monkeypatch.setattr(pipeline, "run_pipeline", ingest)
    monkeypatch.setattr(pipeline, "_source_checkout_root", lambda: None)

    assert pipeline.main(["--data-csv", str(daily_csv), "--benchmark"]) == 1

    _assert_clean_failure(capsys, "--benchmark", "source checkout")
    ingest.assert_not_called()


def test_source_checkout_detected_from_repository() -> None:
    assert pipeline._source_checkout_root() == Path(__file__).resolve().parents[1]
