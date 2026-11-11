"""Table rendering and number formatting shared by the benchmark scripts."""

from __future__ import annotations

import math


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return an aligned table: first column left-aligned, the rest right-aligned."""

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
        aligned_rest = [cell.rjust(width) for cell, width in zip(rest, rest_widths)]
        return "  ".join([first.ljust(widths[0]), *aligned_rest])

    separator = "  ".join("-" * width for width in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def render_box_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a Unicode box-drawing table with every column left-aligned."""

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
        padded = [cell.ljust(width) for cell, width in zip(cells, widths)]
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
    """Format a byte count in binary units; non-finite values pass through as text."""

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
