"""Row-level bitemporal version resolution, shared by every query path.

A chunk's rows are, by construction, all visible at the same transaction
time -- either implicitly (a base chunk, learned one row at a time as it
happened) or explicitly (a correction chunk, a single row with its own
recorded transaction time). Resolving duplicates requires that transaction
time to travel alongside each row once chunks are decompressed and
concatenated; these two functions are the only place that happens, so the
pushdown-query path (QueryExecutor) and the naive baseline (FullScanQueryEngine)
can never silently diverge on bitemporal semantics the way two independently
written implementations could.
"""

from __future__ import annotations

import pandas as pd

from src.store.chunk import Chunk

TRANSACTION_TIME_COLUMN = "_txn_time"
_IS_CORRECTION_COLUMN = "_is_correction"


def attach_effective_transaction_time(
    frame: pd.DataFrame, chunk: Chunk
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a per-row effective transaction time.

    Base chunks (``chunk.correction_transaction_times is None``) were
    learned at their own valid time, row by row, so each row's effective
    transaction time is simply its own ``timestamp`` value. Correction
    chunks (single or merged by ``ChunkStore.compact_corrections``) carry an
    explicit, per-row transaction time list -- required once a chunk can
    merge corrections learned at different times, since no single per-chunk
    scalar could answer "when was this specific row learned" anymore.

    Also attaches ``_IS_CORRECTION_COLUMN``, a same-instant tie-break signal
    ``resolve_versions`` uses (see its docstring) -- private to this module's
    resolution logic, dropped again before ``resolve_versions`` returns.
    """

    frame = frame.copy()
    if chunk.correction_transaction_times is None:
        frame[TRANSACTION_TIME_COLUMN] = frame["timestamp"]
        frame[_IS_CORRECTION_COLUMN] = False
    else:
        frame[TRANSACTION_TIME_COLUMN] = chunk.correction_transaction_times
        frame[_IS_CORRECTION_COLUMN] = True
    return frame


def resolve_versions(
    df: pd.DataFrame, txn_col: str = TRANSACTION_TIME_COLUMN
) -> pd.DataFrame:
    """Collapse duplicate (symbol, timestamp) rows to the newest version.

    Among rows sharing a (symbol, timestamp) key, keeps only the one with
    the maximum effective transaction time -- "latest wins" -- which is the
    row-level analog of the AS OF query's "what do we currently know"
    default. Callers wanting a specific AS OF cutoff must first drop rows
    whose effective transaction time exceeds it; this function only
    resolves ties among whatever rows are already visible.

    ``ingest_correction`` only rejects a transaction time strictly before its
    own event's timestamp, so a correction filed at the exact same instant
    as the row it corrects is legal -- an exact tie on ``txn_col`` between a
    base row and its correction. Sorting by ``_IS_CORRECTION_COLUMN`` as a
    tie-break (ascending: base before correction) and taking the last row
    per group makes that tie resolve to the correction, not the base row it
    was meant to override.
    """

    if df.empty:
        return df

    # groupby(["symbol", "timestamp"]) below partitions by key regardless of
    # the frame's physical row order -- only txn_col (and the tie-break)
    # affect which row within a group survives .tail(1), so sorting by
    # symbol/timestamp here first would be wasted work never observed by the
    # grouping itself. Confirmed byte-identical output either way.
    has_tie_break = _IS_CORRECTION_COLUMN in df.columns
    sort_columns = [txn_col]
    if has_tie_break:
        sort_columns.append(_IS_CORRECTION_COLUMN)
    ordered = df.sort_values(sort_columns, kind="stable")
    newest = ordered.groupby(["symbol", "timestamp"], sort=False).tail(1)
    if has_tie_break:
        newest = newest.drop(columns=[_IS_CORRECTION_COLUMN])
    return newest.sort_values(["timestamp", "symbol"], kind="stable").reset_index(
        drop=True
    )


__all__ = [
    "TRANSACTION_TIME_COLUMN",
    "attach_effective_transaction_time",
    "resolve_versions",
]
