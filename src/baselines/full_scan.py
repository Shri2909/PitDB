"""Naive full-scan baseline: exact SQL filtering after decompressing every chunk."""

from __future__ import annotations

import time

import pandas as pd

from src.query.executor import QueryResult, bind_query_literals
from src.query.predicate import _coerce_column_literal, validate_predicate_columns
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
        validate_predicate_columns(
            predicate, {"timestamp", "symbol", *self.store.value_columns}
        )
        as_of = bind_query_literals(self.store, predicate, as_of)

        chunks = self.store.chunks
        needs_resolution = (
            as_of is not None
            or self.store.has_corrections
            or self.store.has_base_duplicates
        )

        if chunks and needs_resolution:
            # Resolve before filtering, as QueryExecutor does.
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
            resolved = resolve_versions(combined).drop(
                columns=[TRANSACTION_TIME_COLUMN]
            )
            if resolved.empty:
                data = resolved
            else:
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
