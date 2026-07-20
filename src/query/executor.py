"""End-to-end SQL execution over conservatively selected chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.query.predicate import _coerce_column_literal
from src.query.pushdown import PushdownResult, evaluate_pushdown
from src.query.sql_parser import extract_as_of, parse_where_clause
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


class QueryExecutor:
    """Parse, prune, decompress candidates, and apply an exact row filter."""

    def __init__(self, store: ChunkStore) -> None:
        if not isinstance(store, ChunkStore):
            raise TypeError("store must be a ChunkStore")
        self.store = store

    def execute(self, sql: str) -> QueryResult:
        """Execute the supported SQL subset against finalized store chunks."""

        started = time.perf_counter()
        remaining_sql, as_of = extract_as_of(sql)
        predicate = parse_where_clause(remaining_sql)
        pushdown_result = evaluate_pushdown(
            predicate,
            self.store.labels,
            as_of=as_of,
            as_of_index=self.store,
            price_pruning_enabled=not self.store.has_corrections,
        )

        # Exact row evaluation runs predicate.evaluate_vectorized, which
        # translates the predicate tree into pandas boolean-mask operations,
        # rather than looping row-by-row with frame.apply(axis=1) (a
        # Python-level function call per row).
        #
        # That mask is applied once, on the concatenation of every candidate
        # chunk, rather than once per chunk: a scale sweep at 24,000 chunks
        # showed a 0%-skip query still ran ~1.4x slower than the full-scan
        # baseline when the mask was applied per chunk, because calling a
        # pandas boolean-mask operation once per ~20-row chunk pays that
        # call's fixed overhead thousands of times instead of once. Matching
        # the full-scan baseline's own structure (concatenate everything,
        # then filter once) removed that gap.
        chunks = [
            self.store.get_chunk(chunk_id)
            for chunk_id in pushdown_result.candidate_chunk_ids
        ]
        # Version resolution only runs when it could possibly matter: an
        # explicit AS OF cutoff, or at least one correction chunk among the
        # candidates. Every existing query -- no AS OF, no corrections ever
        # ingested -- takes the exact prior code path, byte-for-byte.
        needs_resolution = as_of is not None or any(
            chunk.label.transaction_time_start is not None for chunk in chunks
        )

        if chunks and needs_resolution:
            # Resolve each (symbol, timestamp) to its current/AS-OF-visible
            # version BEFORE applying the exact predicate, not after: a
            # correction's own value might fail the predicate the base row
            # happened to pass (or vice versa), and filtering first would
            # silently discard the correction, letting the stale base row
            # leak through the final result.
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
            # preserves row order -- so data is already in final order here.
            # No further sort needed (see the elif branch's comment for why
            # that one still needs one).
            resolved = resolve_versions(combined).drop(
                columns=[TRANSACTION_TIME_COLUMN]
            )
            data = resolved[predicate.evaluate_vectorized(resolved)]
        elif chunks:
            combined = pd.concat(
                [chunk.decompress() for chunk in chunks], ignore_index=True
            )
            data = combined[predicate.evaluate_vectorized(combined)]
            # This branch skips resolve_versions entirely, so rows are still
            # in natural chunk order. FullScanQueryEngine's own
            # needs_resolution is scoped to the whole store (it has no
            # narrower scope to check), so it can legitimately take the
            # resolved branch above -- which does end sorted -- for a query
            # whose own candidates here don't need resolution. Without
            # sorting here too, otherwise row-identical results could differ
            # only in order, failing the full-scan/pushdown equivalence
            # every benchmark and test in this project relies on. Gated
            # behind the cheap has_corrections check (not unconditional): a
            # store that has never seen a correction takes the exact prior
            # code path, byte-for-byte, including natural chunk-order --
            # required for parity with ChunkStore.full_scan()'s own unsorted
            # natural order.
            if not data.empty and self.store.has_corrections:
                data = data.sort_values(["timestamp", "symbol"], kind="stable")
        else:
            # Do not decompress a skipped chunk merely to discover the schema.
            data = pd.DataFrame(
                columns=["timestamp", "symbol", *self.store.value_columns]
            )

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
