"""Presentation-only console rendering for the benchmark suite.

Two styles coexist: the 96-column ``rule``/``benchmark_header``/
``section_header``/``interpretation`` (still used by Benchmark 3) and the
80-column exact-format helpers (``pitdb_*``, ``dash_section``, ``field_block``,
``titled_*``, ``wrap_text``) used everywhere else.
"""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

# Execution-phase vocabulary in run order; WARNING/ERROR appear only when needed.
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
_RULE_WIDTH = 96


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
    """Return one benchmark's opening banner from already-formatted config lines."""

    lines = [
        rule(),
        f" BENCHMARK {index}/{total} — {name}",
        rule(),
        f" Question : {question}",
    ]
    lines.extend(f" {line}" for line in config_lines)
    return "\n".join(lines)


def section_header(title: str, meta_lines: Sequence[str] = ()) -> str:
    """Return a lighter-weight sub-section header."""

    lines = [f"--- {title} {'-' * max(0, _RULE_WIDTH - len(title) - 5)}"]
    lines.extend(f"{line}" for line in meta_lines)
    return "\n".join(lines)


def interpretation(*, observation: str, evidence: str, limitation: str) -> str:
    """Return the standard post-benchmark interpretation block."""

    return "\n".join(
        [
            "Interpretation",
            f"  Observation : {observation}",
            f"  Evidence    : {evidence}",
            f"  Limitation  : {limitation}",
        ]
    )


# Exact-format (80-column) primitives.
_EXACT_RULE_WIDTH = 80


def pitdb_header(*, index: int, total: int, title: str) -> str:
    """Return the ``PITDB BENCHMARK N/total`` / title double-rule header."""

    bar = "=" * _EXACT_RULE_WIDTH
    return f"{bar}\n PITDB BENCHMARK {index}/{total}\n {title}\n{bar}"


def pitdb_title(title: str) -> str:
    """Return a double-rule block with a single title line."""

    bar = "=" * _EXACT_RULE_WIDTH
    return f"{bar}\n {title}\n{bar}"


def pitdb_footer(*, index: int) -> str:
    """Return the closing ``BENCHMARK N COMPLETE`` double-rule footer."""

    bar = "=" * _EXACT_RULE_WIDTH
    return f"{bar}\n BENCHMARK {index} COMPLETE\n{bar}"


def dash_section(*lines: str) -> str:
    """Return one or more lines (tag plus optional subtitle) between dash rules."""

    if not lines:
        raise ValueError("dash_section requires at least one line")
    bar = "-" * _EXACT_RULE_WIDTH
    body = "\n".join(lines)
    return f"{bar}\n{body}\n{bar}"


def titled_paragraph(title: str, text: str, *, indent: int = 2, width: int = 78) -> str:
    """Return a heading line followed by a wrapped, more-indented paragraph."""

    heading = f"{' ' * indent}{title}"
    body = wrap_text(text, indent=indent + 2, width=width)
    return f"{heading}\n{body}"


def wrap_text(text: str, *, indent: int = 2, width: int = 78) -> str:
    """Return ``text`` word-wrapped to ``width`` columns, each line indented."""

    return textwrap.fill(
        text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent
    )


def field_block(pairs: Sequence[tuple[str, str]], *, indent: int = 2) -> str:
    """Return an aligned ``label : value`` block, aligned per block."""

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
