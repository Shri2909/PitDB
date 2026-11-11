"""Run-provenance snapshot: git commit, interpreter, OS, CPU, memory."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil


def git_commit(*, short: bool = True, cwd: Path | None = None) -> str:
    """Return the current git commit hash, or an "unknown (...)" reason."""

    args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown (git unavailable)"
    if completed.returncode != 0:
        return "unknown (not a git checkout)"
    commit = completed.stdout.strip()
    return commit or "unknown (not a git checkout)"


def hardware_snapshot() -> dict[str, Any]:
    """Return interpreter/OS/CPU/memory fields for result files."""

    return {
        "processor": platform.processor(),
        "system": platform.system(),
        "release": platform.release(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "total_memory_bytes": int(psutil.virtual_memory().total),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
    }


__all__ = ["git_commit", "hardware_snapshot"]
