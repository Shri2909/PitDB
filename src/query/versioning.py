"""Row-level bitemporal version resolution shared by every query path."""

from __future__ import annotations

import pandas as pd

from src.store.chunk import Chunk

TRANSACTION_TIME_COLUMN = "_txn_time"
_IS_CORRECTION_COLUMN = "_is_correction"
_ORDER_COLUMN = "_order"


def attach_effective_transaction_time(
    frame: pd.DataFrame, chunk: Chunk
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a per-row effective transaction time.

    Base chunks use each row's own ``timestamp``; correction chunks carry an
    explicit per-row transaction time list.
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

    The row with the greatest ``txn_col`` wins. Ties go to a correction over a
    base row, then to the row appearing last in ``df`` (an explicit position
    key, so the rule does not depend on sort stability). Callers apply any AS OF
    cutoff beforehand.
    """

    if df.empty:
        return df.drop(columns=[_IS_CORRECTION_COLUMN], errors="ignore")

    has_tie_break = _IS_CORRECTION_COLUMN in df.columns
    order = pd.Series(range(len(df)), index=df.index, name=_ORDER_COLUMN)
    sort_columns = [txn_col]
    if has_tie_break:
        sort_columns.append(_IS_CORRECTION_COLUMN)
    sort_columns.append(_ORDER_COLUMN)
    ordered = df.assign(**{_ORDER_COLUMN: order}).sort_values(sort_columns)
    newest = ordered.groupby(["symbol", "timestamp"], sort=False).tail(1)
    newest = newest.drop(columns=[_ORDER_COLUMN])
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
