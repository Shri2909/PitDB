"""Shared, presentation-only console rendering for the benchmark suite.

Every function in this module formats values its caller already computed --
none of them execute a query, run a timing loop, or perform a statistical
calculation. Computation lives in ``benchmarks/stats.py``; the underlying
table/number primitives these functions build on live in
``benchmarks/report.py``. This module holds two header/section styles side
by side, deliberately: ``rule``/``benchmark_header``/``section_header``/
``interpretation`` (96-column) is this suite's original convention, still
used by ``bench_scale.py`` (Benchmark 3), which has not yet been migrated to
the exact-format spec the rest of the suite follows. ``pitdb_header``/
``pitdb_title``/``pitdb_footer``/``dash_section``/``field_block``/
``titled_block``/``titled_paragraph``/``wrap_text`` (80-column) is that
exact-format spec, used by Benchmarks 1, 2, 4, 5, and the suite summary in
``run_suite.py``. Once Benchmark 3 is migrated, the 96-column style can be
retired.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

# Fixed execution-phase vocabulary, in the order a benchmark run visits them.
# WARNING/ERROR are exceptions -- they appear only when something actually
# needs flagging, never as a required step of every run.
PHASES: tuple[str, ...] = (
    "SETUP",
    "BUILD",
    "VALIDATE",
    "WARMUP",
    "MEASURE",
    "ANALYZE",
    "RESULT",
    "EXPORT",
    "WARNING",
    "ERROR",
)

_TAG_WIDTH = max(len(name) for name in PHASES) + 2  # "[NAME]" + one space
_RULE_WIDTH = 96  # matches this suite's pre-existing "=" * 96 convention


def phase(label: str, message: str) -> str:
    """Return one phase-tagged line, e.g. ``"[BUILD]   ingested 5,010 rows"``."""

    if label not in PHASES:
        raise ValueError(f"unknown phase label: {label!r}; expected one of {PHASES}")
    tag = f"[{label}]"
    return f"{tag.ljust(_TAG_WIDTH)}{message}"


def rule(character: str = "=") -> str:
    """Return one full-width horizontal rule."""

    if not character or len(character) != 1:
        raise ValueError("character must be a single character")
    return character * _RULE_WIDTH


def benchmark_header(
    *,
    index: int,
    total: int,
    name: str,
    question: str,
    config_lines: Sequence[str],
) -> str:
    """Return one benchmark's opening banner: number/name, question, config.

    ``config_lines`` are already-formatted strings (e.g. "Trials: 15 measured,
    3 warmup") -- this function lays them out, it does not derive them.
    """

    lines = [
        rule(),
        f" BENCHMARK {index}/{total} — {name}",
        rule(),
        f" Question : {question}",
    ]
    lines.extend(f" {line}" for line in config_lines)
    return "\n".join(lines)


def section_header(title: str, meta_lines: Sequence[str] = ()) -> str:
    """Return a lighter-weight sub-section header (a scenario/scale point/
    correction rate within one benchmark), replacing this suite's previous
    ad hoc ``"=== ... ==="`` banners with one consistent style."""

    lines = [f"--- {title} {'-' * max(0, _RULE_WIDTH - len(title) - 5)}"]
    lines.extend(f"{line}" for line in meta_lines)
    return "\n".join(lines)


def interpretation(*, observation: str, evidence: str, limitation: str) -> str:
    """Return the standard post-benchmark interpretation block.

    Every field must be a statement about values already measured elsewhere
    in the same benchmark run -- this function only lays the three lines
    out, it computes nothing.
    """

    return "\n".join(
        [
            "Interpretation",
            f"  Observation : {observation}",
            f"  Evidence    : {evidence}",
            f"  Limitation  : {limitation}",
        ]
    )


# ---------------------------------------------------------------------------
# Exact-format primitives for the approved per-benchmark layout (currently
# wired into benchmarks/run_all.py's Benchmark 1 output only -- see
# BENCHMARK_OUTPUT_AUDIT.md). Deliberately a distinct 80-column style from
# the 96-column rule()/benchmark_header() above rather than a shared one:
# this is the approved spec's own width, not this suite's older convention.
# ---------------------------------------------------------------------------

_EXACT_RULE_WIDTH = 80


def pitdb_header(*, index: int, total: int, title: str) -> str:
    """Return the ``PITDB BENCHMARK N/total`` / title double-rule header."""

    bar = "=" * _EXACT_RULE_WIDTH
    return f"{bar}\n PITDB BENCHMARK {index}/{total}\n {title}\n{bar}"


def pitdb_title(title: str) -> str:
    """Return a double-rule block with a single title line -- for
    suite-level headers/footers that aren't a numbered per-benchmark
    header (see :func:`pitdb_header`) or its matching per-benchmark
    "BENCHMARK N COMPLETE" footer (see :func:`pitdb_footer`)."""

    bar = "=" * _EXACT_RULE_WIDTH
    return f"{bar}\n {title}\n{bar}"


def pitdb_footer(*, index: int) -> str:
    """Return the closing ``BENCHMARK N COMPLETE`` double-rule footer."""

    bar = "=" * _EXACT_RULE_WIDTH
    return f"{bar}\n BENCHMARK {index} COMPLETE\n{bar}"


def dash_section(*lines: str) -> str:
    """Return a dash-ruled section tag, e.g. around ``[FIXED QUERIES]`` --
    one line for a simple tag, or several (tag plus subtitle) when the
    approved layout calls for it, e.g. ``[STORAGE REPRESENTATION]`` /
    ``Directly comparable storage measurements``."""

    if not lines:
        raise ValueError("dash_section requires at least one line")
    bar = "-" * _EXACT_RULE_WIDTH
    body = "\n".join(lines)
    return f"{bar}\n{body}\n{bar}"


def titled_paragraph(title: str, text: str, *, indent: int = 2, width: int = 78) -> str:
    """Return a heading line followed by a wrapped, more-indented paragraph.

    Distinct from :func:`titled_block`: this is for a single free-text
    passage (an Interpretation or Limitation note), not colon-aligned
    label:value pairs.
    """

    heading = f"{' ' * indent}{title}"
    body = wrap_text(text, indent=indent + 2, width=width)
    return f"{heading}\n{body}"


def wrap_text(text: str, *, indent: int = 2, width: int = 78) -> str:
    """Return ``text`` word-wrapped to ``width`` columns, each line indented.

    Used for free-text fields (a Question, a cautionary note) long enough to
    need wrapping at this suite's 80-column layout -- never for tabular or
    already-short label:value content, which must stay on one line.
    """

    return textwrap.fill(
        text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent
    )


def field_block(pairs: Sequence[tuple[str, str]], *, indent: int = 2) -> str:
    """Return an aligned ``label : value`` block.

    Colon alignment is computed from the longest label in *this* block
    only -- each block (Configuration, Predicate, Correctness validation,
    ...) chooses its own column, rather than sharing one document-wide
    label width, matching how the approved layout's blocks are each
    independently aligned.
    """

    if not pairs:
        raise ValueError("pairs must not be empty")
    label_width = max(len(label) for label, _ in pairs)
    prefix = " " * indent
    return "\n".join(
        f"{prefix}{label.ljust(label_width)} : {value}" for label, value in pairs
    )


def titled_block(
    title: str, pairs: Sequence[tuple[str, str]], *, indent: int = 2
) -> str:
    """Return a ``title`` line followed by its :func:`field_block`."""

    return f"{title}\n{field_block(pairs, indent=indent)}"


__all__ = [
    "PHASES",
    "benchmark_header",
    "dash_section",
    "field_block",
    "interpretation",
    "phase",
    "pitdb_footer",
    "pitdb_header",
    "pitdb_title",
    "rule",
    "section_header",
    "titled_block",
    "titled_paragraph",
    "wrap_text",
]
