"""Correctness-gate rendering: makes an already-executed full-scan/pushdown
comparison visible on both the pass and fail path.

This module performs no comparison of its own. Every benchmark file already
calls ``benchmarks.bench_query._assert_equivalent`` (a full
``pd.testing.assert_frame_equal`` between the naive full-scan baseline and
the zone-map pushdown path) before computing or printing any speedup --
that assertion's raise/no-raise outcome is the *entire* correctness contract
this project relies on, and it is not touched here. ``assert_equivalent_and_report``
only wraps that existing call so its outcome prints instead of staying silent
on success, while re-raising exactly the same exception, unchanged, on
failure -- the run still aborts exactly as it does today.
"""

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
    detail: str | None = None  # populated only on failure


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
    """Run an existing zero-argument correctness assertion and print its
    outcome, preserving its exception behavior exactly.

    ``assert_fn`` must be a callable that raises ``AssertionError`` on
    mismatch and returns ``None`` otherwise -- in every current call site,
    ``benchmarks.bench_query._assert_equivalent`` bound with its arguments
    already applied via a lambda. On failure, the FAIL block is always
    printed (regardless of ``quiet``) and the original ``AssertionError`` is
    re-raised unchanged (not wrapped, not swallowed), so the run aborts
    exactly as it did before this wrapper existed -- no speedup is ever
    computed after a failed comparison, because the caller never gets
    control back to compute one.

    ``quiet=True`` suppresses only the PASS-path print: for callers that
    intend to re-render the returned ``CorrectnessOutcome`` themselves at a
    different point in the output (e.g. embedded inside a per-query result
    block further down), so correctness isn't reported twice for the same
    comparison. The assertion itself, and its exception behavior, are
    identical either way.
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
    """Return a formatted speedup string, or the suppression marker.

    Exists for callers that build a results table from a list of
    ``CorrectnessOutcome``s after the fact (rather than aborting
    immediately) -- not currently exercised by this suite's own scripts
    (which all abort on the first correctness failure, matching their prior
    behavior), but kept as the documented, tested suppression path the
    implementation spec requires: a speedup must never be printed opposite a
    failed comparison.
    """

    if not outcome.passed:
        return "SUPPRESSED"
    return formatter(value)


__all__ = [
    "CorrectnessOutcome",
    "assert_equivalent_and_report",
    "render_correctness_block",
    "speedup_cell",
]
