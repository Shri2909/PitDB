"""End-to-end SQL execution over conservatively selected chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.query.predicate import (
    PredicateNode,
    _coerce_column_literal,
    iter_datetime_literals,
    validate_predicate_columns,
)
from src.query.pushdown import PushdownResult, evaluate_pushdown
from src.query.sql_parser import (
    UnsupportedQueryError,
    extract_as_of,
    parse_where_clause,
)
from src.query.versioning import (
    TRANSACTION_TIME_COLUMN,
    attach_effective_transaction_time,
    resolve_versions,
)
from src.store.chunk_store import ChunkStore


@dataclass
class QueryResult:
    """Rows and diagnostics produced by a completed query."""

    data: pd.DataFrame
    row_count: int
    pushdown: PushdownResult
    elapsed_seconds: float
    selectivity: float


def bind_query_literals(
    store: ChunkStore, predicate: PredicateNode, as_of: datetime | None
) -> pd.Timestamp | None:
    """Localize every datetime literal and the AS OF cutoff to the store's timezone.

    Doing it up front makes a failure deterministic and typed. Returns the
    aligned AS OF cutoff (or ``None``).
    """

    try:
        for literal in iter_datetime_literals(predicate):
            store.align_as_of(literal)
        return None if as_of is None else store.align_as_of(as_of)
    except ValueError as exc:
        raise UnsupportedQueryError(
            f"datetime literal cannot be used with this store's timezone: {exc}"
        ) from exc


class QueryExecutor:
    """Parse, prune, decompress candidates, and apply an exact row filter.

    Non-versioned queries return rows in natural chunk order; versioned ones
    (AS OF, corrections or duplicate base keys) are version-resolved and sorted
    by ``(timestamp, symbol)``. The orders are not unified so the unversioned
    fast path stays free of an O(n log n) sort.
    """

    def __init__(self, store: ChunkStore, *, enable_pruning: bool = True) -> None:
        if not isinstance(store, ChunkStore):
            raise TypeError("store must be a ChunkStore")
        self.store = store
        if not isinstance(enable_pruning, bool):
            raise TypeError("enable_pruning must be a bool")
        self.enable_pruning = enable_pruning

    def execute(self, sql: str) -> QueryResult:
        """Execute the supported SQL subset against finalized store chunks."""

        started = time.perf_counter()
        remaining_sql, as_of = extract_as_of(sql)
        predicate = parse_where_clause(remaining_sql)
        validate_predicate_columns(
            predicate, {"timestamp", "symbol", *self.store.value_columns}
        )
        as_of = bind_query_literals(self.store, predicate, as_of)
        versioned = self.store.has_corrections or self.store.has_base_duplicates
        pushdown_result = (
            evaluate_pushdown(
                predicate,
                self.store.labels,
                as_of=as_of,
                as_of_index=self.store,
                price_pruning_enabled=not versioned,
            )
            if self.enable_pruning
            else PushdownResult(
                candidate_chunk_ids=[label.chunk_id for label in self.store.labels],
                skipped_chunk_ids=[],
                total_chunks=self.store.total_chunks,
                chunks_skipped_ratio=0.0,
                candidate_count=self.store.total_chunks,
            )
        )

        # The mask is applied once to the concatenation of all candidates: per-chunk
        # masks paid pandas' fixed call overhead thousands of times at scale.
        chunks = [
            self.store.get_chunk(chunk_id)
            for chunk_id in pushdown_result.candidate_chunk_ids
        ]
        needs_resolution = as_of is not None or versioned

        if chunks and needs_resolution:
            # Resolve before filtering, or a correction failing the predicate is lost.
            combined = pd.concat(
                [
                    attach_effective_transaction_time(chunk.decompress(), chunk)
                    for chunk in chunks
                ],
                ignore_index=True,
            )
            if as_of is not None:
                aligned_as_of = _coerce_column_literal(
                    combined[TRANSACTION_TIME_COLUMN], as_of
                )
                combined = combined[combined[TRANSACTION_TIME_COLUMN] <= aligned_as_of]
            # resolve_versions sorts by (timestamp, symbol); masking keeps that order.
            resolved = resolve_versions(combined).drop(
                columns=[TRANSACTION_TIME_COLUMN]
            )
            data = (
                resolved
                if resolved.empty
                else resolved[predicate.evaluate_vectorized(resolved)]
            )
        elif chunks:
            combined = pd.concat(
                [chunk.decompress() for chunk in chunks], ignore_index=True
            )
            data = combined[predicate.evaluate_vectorized(combined)]
        else:
            # Avoid decompressing a chunk just to learn the schema.
            data = self.store.empty_frame()

        data = data.reset_index(drop=True)

        row_count = len(data)
        total_rows = self.store.total_rows
        selectivity = row_count / total_rows if total_rows else 0.0
        elapsed_seconds = time.perf_counter() - started
        return QueryResult(
            data=data,
            row_count=row_count,
            pushdown=pushdown_result,
            elapsed_seconds=elapsed_seconds,
            selectivity=selectivity,
        )


__all__ = ["QueryExecutor", "QueryResult"]
