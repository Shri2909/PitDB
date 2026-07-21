"""Run-provenance snapshot: git commit, interpreter, OS, CPU, memory.

Every value here is read directly from the running process, the filesystem,
or a subprocess call -- never invented. A value this module cannot determine
reliably comes back as an explicit "unknown (...)" string rather than being
omitted or guessed, per this suite's rule against fabricating metadata.

``psutil`` is a pinned project dependency (see requirements.txt) already
imported unconditionally by benchmarks/bench_scale.py; this module follows
that same convention rather than adding a new optional-import branch.
"""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any

import psutil


def git_commit(*, short: bool = True, cwd: Path | None = None) -> str:
    """Return the current git commit hash, or an explicit "unknown" reason.

    Never fabricates a hash: outside a git checkout, or if the ``git``
    executable itself is unavailable, this returns a descriptive
    "unknown (...)" string instead.
    """

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
    """Return already-available interpreter/OS/CPU/memory fields.

    Mirrors the field set benchmarks/run_all.py's pre-existing
    ``_hardware_info()`` collects (that function is untouched -- this is a
    separate, additive snapshot for the other benchmark scripts' result
    files and the suite header), plus logical/physical core counts, which
    were not previously collected anywhere in this suite despite being an
    already-available ``psutil`` call, not a new measurement.
    """

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
