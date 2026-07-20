"""Naive full-scan baseline: exact SQL filtering after decompressing every chunk."""

from __future__ import annotations

import time

import pandas as pd

from src.query.executor import QueryResult
from src.query.predicate import _coerce_column_literal
from src.query.pushdown import PushdownResult
from src.query.sql_parser import extract_as_of, parse_where_clause
from src.query.versioning import (
    TRANSACTION_TIME_COLUMN,
    attach_effective_transaction_time,
    resolve_versions,
)
from src.store.chunk_store import ChunkStore


class FullScanQueryEngine:
    """Expose brute-force scanning through the indexed executor's interface."""

    def __init__(self, store: ChunkStore) -> None:
        if not isinstance(store, ChunkStore):
            raise TypeError("store must be a ChunkStore")
        self.store = store

    def execute(self, sql: str) -> QueryResult:
        """Parse and evaluate a query without consulting chunk labels."""

        started = time.perf_counter()
        remaining_sql, as_of = extract_as_of(sql)
        predicate = parse_where_clause(remaining_sql)

        chunks = self.store.chunks
        # Same gate QueryExecutor uses: only decompress-and-resolve on
        # transaction time when it could matter. Otherwise this stays
        # exactly the naive baseline it always was -- decompress everything,
        # apply one exact row predicate, no label or pushdown decision
        # consulted anywhere. Unlike QueryExecutor (whose needs_resolution is
        # scoped to pushdown-surviving candidates, a possible strict subset
        # of the store), chunks here is always the *whole* store, so this
        # condition is exactly self.store.has_corrections -- using the O(1)
        # property directly instead of re-deriving it via a scan.
        needs_resolution = as_of is not None or self.store.has_corrections

        if chunks and needs_resolution:
            # Resolve before filtering -- see QueryExecutor's identical
            # reordering and its comment for why filtering first can
            # silently discard a correction that fails the exact predicate.
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
            # resolve_versions already ends with a sort by
            # ["timestamp", "symbol"], and boolean-mask filtering below
            # preserves row order -- data is already in final order here.
            # No trailing sort needed: unlike QueryExecutor, this branch and
            # needs_resolution's own condition are the same
            # (self.store.has_corrections), so the "no resolution" branch
            # below is only ever taken when there's nothing to sort for
            # anyway (see the comment above needs_resolution).
            resolved = resolve_versions(combined).drop(
                columns=[TRANSACTION_TIME_COLUMN]
            )
            mask = resolved.apply(
                lambda row: bool(predicate.evaluate_against_row(row)), axis=1
            )
            data = resolved[mask]
        else:
            data = self.store.full_scan(predicate.evaluate_against_row)

        data = data.reset_index(drop=True)

        candidate_chunk_ids = [label.chunk_id for label in self.store.labels]
        total_chunks = len(candidate_chunk_ids)
        pushdown = PushdownResult(
            candidate_chunk_ids=candidate_chunk_ids,
            skipped_chunk_ids=[],
            total_chunks=total_chunks,
            chunks_skipped_ratio=0.0,
            candidate_count=total_chunks,
        )

        row_count = len(data)
        total_rows = self.store.total_rows
        selectivity = row_count / total_rows if total_rows else 0.0
        elapsed_seconds = time.perf_counter() - started
        return QueryResult(
            data=data,
            row_count=row_count,
            pushdown=pushdown,
            elapsed_seconds=elapsed_seconds,
            selectivity=selectivity,
        )


__all__ = ["FullScanQueryEngine"]
