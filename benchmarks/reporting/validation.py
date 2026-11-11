"""Render the outcome of an already-defined full-scan/pushdown correctness assertion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CorrectnessOutcome:
    """The rendered outcome of one already-executed correctness assertion."""

    name: str
    full_scan_row_count: int
    pushdown_row_count: int
    passed: bool
    detail: str | None = None  # failure only


def render_correctness_block(outcome: CorrectnessOutcome) -> str:
    """Return a PASS or FAIL block for one correctness comparison."""

    if outcome.passed:
        return "\n".join(
            [
                f"[VALIDATE] Correctness check: {outcome.name}",
                f"  Full-scan rows   : {outcome.full_scan_row_count:,}",
                f"  Pushdown rows    : {outcome.pushdown_row_count:,}",
                "  Row equality     : PASS",
                "  Value equality   : PASS",
                "  Dtype equality   : PASS",
                "  Overall status   : PASSED",
            ]
        )

    return "\n".join(
        [
            f"[ERROR] Correctness check FAILED: {outcome.name}",
            f"  Full-scan rows   : {outcome.full_scan_row_count:,}",
            f"  Pushdown rows    : {outcome.pushdown_row_count:,}",
            f"  Failure detail   : {outcome.detail or 'unknown'}",
            "  Overall status   : FAILED -- no speedup will be reported for "
            "this comparison",
        ]
    )


def assert_equivalent_and_report(
    assert_fn: Callable[[], None],
    *,
    name: str,
    full_scan_row_count: int,
    pushdown_row_count: int,
    quiet: bool = False,
) -> CorrectnessOutcome:
    """Run ``assert_fn`` and print its outcome; a failure prints and re-raises.

    ``quiet=True`` suppresses only the PASS-path print.
    """

    try:
        assert_fn()
    except AssertionError as exc:
        outcome = CorrectnessOutcome(
            name=name,
            full_scan_row_count=full_scan_row_count,
            pushdown_row_count=pushdown_row_count,
            passed=False,
            detail=str(exc),
        )
        print(render_correctness_block(outcome))
        raise

    outcome = CorrectnessOutcome(
        name=name,
        full_scan_row_count=full_scan_row_count,
        pushdown_row_count=pushdown_row_count,
        passed=True,
    )
    if not quiet:
        print(render_correctness_block(outcome))
    return outcome


def speedup_cell(
    outcome: CorrectnessOutcome, value: float, formatter: Callable[[float], str]
) -> str:
    """Return the formatted speedup, or "SUPPRESSED" for a failed comparison."""

    if not outcome.passed:
        return "SUPPRESSED"
    return formatter(value)


__all__ = [
    "CorrectnessOutcome",
    "assert_equivalent_and_report",
    "render_correctness_block",
    "speedup_cell",
]
