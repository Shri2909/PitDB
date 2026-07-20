"""Shared console-output formatting for the benchmark suite.

Every bench_*.py script used to hand-roll its own print statements
independently, with no shared column alignment and inconsistent number
formatting (some scripts round to 2 decimal places, others to 4; some print
milliseconds, others seconds). This module is the one place table rendering
and number formatting live, so every script's final summary reads the same
way.
"""

from __future__ import annotations


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return an aligned table as a single string, ready to print.

    The first column (treated as the row label -- a query name, a
    correction rate, a scale point) is left-aligned; every other column is
    right-aligned, matching how numeric data reads naturally. Column widths
    are derived from the longest cell -- header or data -- in that column.
    """

    if not headers:
        raise ValueError("headers must not be empty")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every row must have as many cells as headers")

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(cells: list[str]) -> str:
        first, rest = cells[0], cells[1:]
        rest_widths = widths[1:]
        aligned_rest = [
            cell.rjust(width) for cell, width in zip(rest, rest_widths, strict=True)
        ]
        return "  ".join([first.ljust(widths[0]), *aligned_rest])

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def format_ms(value_ms: float) -> str:
    """Format a millisecond value with a fixed, consistent precision."""

    return f"{value_ms:.3f}ms"


def format_x(ratio: float) -> str:
    """Format a speedup ratio with a fixed, consistent precision."""

    return f"{ratio:.2f}x"


def format_pct(fraction: float) -> str:
    """Format a 0-1 fraction as a percentage with a fixed precision."""

    return f"{fraction * 100:.2f}%"


__all__ = ["render_table", "format_ms", "format_x", "format_pct"]
