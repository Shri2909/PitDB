"""Conservative chunk selection using predicate trees and chunk labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from src.query.predicate import PredicateNode, label_visible_as_of
from src.store.label import ChunkLabel


class AsOfIndex(Protocol):
    """Minimal interface for the AS OF fast path -- ``ChunkStore`` satisfies
    this by duck typing; pushdown.py deliberately doesn't import it, so this
    module still only depends on ``list[ChunkLabel]``, not a concrete store.
    """

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

    ``PredicateNode.evaluate_against_label`` owns the conservative decision:
    ``True`` means the chunk remains a candidate, including cases where its
    label lacks enough information; only ``False`` authorizes skipping it.
    Input order (the order of ``labels``) is preserved in both result lists,
    regardless of whether the ``as_of_index`` fast path below is used.

    ``as_of`` layers an additional, independent pruning check on top: a
    chunk none of whose rows could be visible as of that transaction time is
    skipped regardless of what the predicate says. ``as_of=None`` (no AS OF
    clause) disables this check entirely, preserving prior behavior exactly.

    ``as_of_index`` is an optional fast path (see ``ChunkStore.labels_visible_as_of``):
    when both it and ``as_of`` are given, the AS-OF-invisible labels are
    identified by a single O(log n) bisect against a maintained sorted index
    instead of a per-label ``label_visible_as_of`` computation, and --
    crucially -- ``predicate.evaluate_against_label`` (which can recurse
    through multiple AND'd clauses) is only ever called on labels that
    survive that cut, not on every label regardless of AS OF visibility.
    The result of that one bisect is then consulted as an O(1)
    membership test while iterating ``labels`` in their original order, so
    the fast path costs one index lookup plus a linear scan of cheap checks
    -- not the O(n) per-label computation it's replacing -- while still
    producing results in the same order the plain linear scan below would.
    Omitting ``as_of_index`` (the default) is byte-for-byte identical to
    that plain linear scan; every existing caller that only passes ``as_of``
    is unaffected.

    ``price_pruning_enabled=False`` (pass whenever the store has any
    correction chunk -- see ``ChunkStore.has_corrections``) disables price
    pruning specifically, since a correction chunk's own price zone map
    failing a value predicate must never hide it from candidacy: doing so
    would silently let a stale base row win over its correction. Symbol and
    time pruning are unaffected either way.
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
