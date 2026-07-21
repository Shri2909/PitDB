"""Shared console-output formatting for the benchmark suite.

Every bench_*.py script used to hand-roll its own print statements
independently, with no shared column alignment and inconsistent number
formatting (some scripts round to 2 decimal places, others to 4; some print
milliseconds, others seconds). This module is the one place table rendering
and number formatting live, so every script's final summary reads the same
way.
"""

from __future__ import annotations

import math


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


def render_box_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a Unicode box-drawing table (``┌─┬─┐`` style) as one string.

    Every column, including the first, is left-aligned within its own
    one-space padding -- this is a distinct visual convention from
    :func:`render_table`'s right-aligned data columns, matching the
    box-table style used in the per-query/per-benchmark result blocks that
    call this function directly rather than through ``render_table``.
    Column widths are derived from the longest cell -- header or data -- in
    that column, exactly like ``render_table``.
    """

    if not headers:
        raise ValueError("headers must not be empty")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("every row must have as many cells as headers")

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def border(left: str, junction: str, right: str) -> str:
        return left + junction.join("─" * (width + 2) for width in widths) + right

    def render_row(cells: list[str]) -> str:
        padded = [cell.ljust(width) for cell, width in zip(cells, widths, strict=True)]
        return "│ " + " │ ".join(padded) + " │"

    lines = [
        border("┌", "┬", "┐"),
        render_row(headers),
        border("├", "┼", "┤"),
    ]
    lines.extend(render_row(row) for row in rows)
    lines.append(border("└", "┴", "┘"))
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


def format_bytes(num_bytes: float) -> str:
    """Format a byte count using binary (KiB/MiB/GiB/TiB) units.

    Values under 1024 bytes are shown as a plain integer count (no unit
    subdivision below whole bytes -- this project never measures fractional
    bytes). NaN/+-inf pass through Python's own float formatting (``"nan"``,
    ``"inf"``, ``"-inf"``) rather than raising, matching every other
    formatter in this module.
    """

    if not math.isfinite(num_bytes):
        return f"{num_bytes}"

    sign = "-" if num_bytes < 0 else ""
    remaining = abs(float(num_bytes))
    for unit in ("bytes", "KiB", "MiB", "GiB", "TiB"):
        if remaining < 1024.0 or unit == "TiB":
            if unit == "bytes":
                return f"{sign}{int(remaining):,} {unit}"
            return f"{sign}{remaining:,.2f} {unit}"
        remaining /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "render_table",
    "render_box_table",
    "format_ms",
    "format_x",
    "format_pct",
    "format_bytes",
]
