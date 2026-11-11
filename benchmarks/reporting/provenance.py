"""Self-contained benchmark provenance and result/chart integrity checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.reporting.environment import git_commit, hardware_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_ENV = "PITDB_BENCHMARK_RUN_ID"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "PYTHONHASHSEED",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paths = sorted(
        {
            *root.glob("src/**/*.py"),
            *root.glob("benchmarks/**/*.py"),
            *root.glob("charts/**/*.py"),
            *root.glob("requirements*.txt"),
            *(
                path
                for path in (root / "pyproject.toml", root / "setup.py")
                if path.is_file()
            ),
        }
    )
    files = {str(path.relative_to(root)): file_sha256(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def dataframe_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Hash ordered values, index, columns and dtypes; record the hash method."""
    schema = [(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]
    digest = hashlib.sha256(json.dumps(schema).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return {
        "sha256": digest.hexdigest(),
        "rows": len(frame),
        "columns_and_dtypes": schema,
        "method": (
            "sha256(schema JSON + pandas hash_pandas_object ordered "
            "rows/index bytes); pandas version recorded in manifest"
        ),
    }


def _git_state(root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"dirty": None, "status": "unknown (git unavailable)"}
    if completed.returncode:
        return {"dirty": None, "status": "unknown (not a git checkout)"}
    return {"dirty": bool(completed.stdout), "status": completed.stdout.splitlines()}


def create_manifest(
    *,
    parameters: dict[str, Any],
    fixture_paths: list[Path] | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    dependencies = {}
    for name in ("numpy", "pandas", "sqlglot", "psutil", "matplotlib"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not installed"
    return {
        "schema_version": 1,
        "run_id": os.environ.get(RUN_ID_ENV) or uuid.uuid4().hex,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "git_commit": git_commit(short=False, cwd=root),
        "git_worktree": _git_state(root),
        "source": source_snapshot(root),
        "fixture_files": {
            str(path): file_sha256(path) for path in (fixture_paths or [])
        },
        "hardware": hardware_snapshot(),
        "dependency_versions": dependencies,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "parameters": parameters,
        "timing_scope": (
            "one warm process, perf_counter wall clock; "
            "parsing/decompression/filtering included; no fresh-process "
            "isolation or significance claims"
        ),
    }


def finish_manifest(
    manifest: dict[str, Any], *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    if source_snapshot(root)["sha256"] != manifest["source"]["sha256"]:
        raise RuntimeError(
            "benchmark source changed during measurement; refusing to publish results"
        )
    for name, digest in manifest["fixture_files"].items():
        if file_sha256(Path(name)) != digest:
            raise RuntimeError(
                "benchmark fixture changed during measurement; "
                "refusing to publish results"
            )
    return {
        **manifest,
        "complete": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_result(path: Path, payload: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Atomically publish strict JSON only after a complete successful run."""
    complete = finish_manifest(manifest)
    serialized = json.dumps(
        {**payload, "manifest": complete}, indent=2, sort_keys=True, allow_nan=False
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)
