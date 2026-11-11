"""Conservative chunk selection using predicate trees and chunk labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.query.predicate import PredicateNode, label_visible_as_of
from src.store.label import ChunkLabel


class AsOfIndex(Protocol):
    """AS OF fast-path interface; ``ChunkStore`` satisfies it by duck typing."""

    def labels_visible_as_of(self, as_of: datetime) -> list[ChunkLabel]: ...
    def labels_not_visible_as_of(self, as_of: datetime) -> list[ChunkLabel]: ...


@dataclass
class PushdownResult:
    """Outcome and diagnostics for one label-pushdown pass."""

    candidate_chunk_ids: list[int]
    skipped_chunk_ids: list[int]
    total_chunks: int
    chunks_skipped_ratio: float
    candidate_count: int


def evaluate_pushdown(
    predicate: PredicateNode,
    labels: list[ChunkLabel],
    as_of: datetime | None = None,
    as_of_index: AsOfIndex | None = None,
    price_pruning_enabled: bool = True,
) -> PushdownResult:
    """Partition chunks into possible matches and definite non-matches.

    Only a ``False`` from ``evaluate_against_label`` skips a chunk; input order
    is preserved in both result lists. ``as_of`` additionally skips chunks with
    no row visible at that transaction time (via ``as_of_index`` when given).
    ``price_pruning_enabled=False`` is required when the store has corrections,
    so a correction chunk's price zone map can never hide it from candidacy.
    """

    if not isinstance(predicate, PredicateNode):
        raise TypeError("predicate must be a PredicateNode")

    total_chunks = len(labels)
    candidate_chunk_ids: list[int] = []
    skipped_chunk_ids: list[int] = []

    if as_of is not None and as_of_index is not None:
        invisible_labels = as_of_index.labels_not_visible_as_of(as_of)
        invisible_ids = {label.chunk_id for label in invisible_labels}

        for label in labels:
            if not isinstance(label, ChunkLabel):
                raise TypeError("every label must be a ChunkLabel")
            if label.chunk_id in invisible_ids:
                skipped_chunk_ids.append(label.chunk_id)
            elif predicate.evaluate_against_label(label, price_pruning_enabled):
                candidate_chunk_ids.append(label.chunk_id)
            else:
                skipped_chunk_ids.append(label.chunk_id)
    else:
        for label in labels:
            if not isinstance(label, ChunkLabel):
                raise TypeError("every label must be a ChunkLabel")

            if as_of is not None and not label_visible_as_of(label, as_of):
                skipped_chunk_ids.append(label.chunk_id)
            elif predicate.evaluate_against_label(label, price_pruning_enabled):
                candidate_chunk_ids.append(label.chunk_id)
            else:
                skipped_chunk_ids.append(label.chunk_id)

    chunks_skipped_ratio = (
        len(skipped_chunk_ids) / total_chunks if total_chunks else 0.0
    )
    return PushdownResult(
        candidate_chunk_ids=candidate_chunk_ids,
        skipped_chunk_ids=skipped_chunk_ids,
        total_chunks=total_chunks,
        chunks_skipped_ratio=chunks_skipped_ratio,
        candidate_count=len(candidate_chunk_ids),
    )


__all__ = ["PushdownResult", "evaluate_pushdown"]
