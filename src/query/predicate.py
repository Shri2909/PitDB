"""Predicate-tree nodes for conservative chunk pruning and exact row tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from src.query.schema import ALIAS_MAP, COLUMN_LABEL_MAP, SCHEMA
from src.store.label import ChunkLabel
from src.store.validation import align_timestamp, timestamp_value

_COMPARISON_OPERATORS = frozenset({"=", "<", ">", "<=", ">="})

_VECTORIZED_COMPARISONS: dict[str, Any] = {
    "=": lambda column, literal: column == literal,
    "<": lambda column, literal: column < literal,
    ">": lambda column, literal: column > literal,
    "<=": lambda column, literal: column <= literal,
    ">=": lambda column, literal: column >= literal,
}


def _coerce_column_literal(column: pd.Series, literal: Any) -> Any:
    """Vectorized counterpart to :func:`_coerce_row_operands`.

    A chunk's column has one dtype for every row, so the tz/float alignment
    that per-row code repeats for each value only needs to happen once here,
    against the literal, before comparing it with the whole column at once.
    """

    if isinstance(literal, datetime):
        literal = timestamp_value(literal, name="datetime literal")
        if column.empty:
            return literal
        return align_timestamp(column.iloc[0], literal)

    if isinstance(literal, float):
        return literal

    return literal


def _validate_canonical_binding(column: str, label_field: str | None) -> None:
    """Protect the parser/executor boundary from aliases or unsafe mappings."""

    if not isinstance(column, str):
        raise ValueError("predicate column must be a canonical column name")
    if column in ALIAS_MAP:
        raise ValueError(
            f"predicate column {column!r} is an SQL alias; use {ALIAS_MAP[column]!r}"
        )
    if column not in SCHEMA:
        raise ValueError(f"unknown canonical predicate column: {column!r}")

    expected = COLUMN_LABEL_MAP[column]
    if label_field != expected:
        raise ValueError(
            f"label_field for {column!r} must be {expected!r}; received {label_field!r}"
        )


def _coerce_row_operands(row_value: Any, literal: Any) -> tuple[Any, Any]:
    """Align supported scalar representations before exact comparison."""

    if isinstance(literal, datetime) and isinstance(row_value, datetime):
        row_value = timestamp_value(row_value)
        return row_value, align_timestamp(row_value, literal)

    if isinstance(literal, float):
        try:
            return float(row_value), literal
        except (TypeError, ValueError, OverflowError):
            # Let the literal comparison below produce its normal result/error
            # for malformed rows rather than inventing a pushdown decision.
            return row_value, literal

    return row_value, literal


def _align_literal_to_reference(reference: datetime, literal: datetime) -> datetime:
    """Return ``literal`` with tz-awareness matched to ``reference``.

    Unlike :func:`_coerce_row_operands` (which aligns whichever of its two
    arguments is naive to match the other, for a row-vs-literal comparison
    where both returned values matter), label-pushdown call sites only want
    one thing: a literal comparable against a label's own time_start/time_end
    fields, which must never themselves be altered. Always adjusting the
    literal -- never the reference -- keeps that call pattern correct in both
    directions.
    """

    return align_timestamp(reference, literal)


def label_visible_as_of(label: ChunkLabel, as_of: datetime) -> bool:
    """Conservatively decide whether a chunk might hold rows visible as of a time.

    A base chunk (``transaction_time_start is None``) was learned at its own
    valid time, one row at a time, so its earliest possible transaction time
    is ``time_start``. A correction chunk carries a transaction-time zone map
    (``transaction_time_start``/``_end``); ``transaction_time_start`` (the
    conservative minimum) answers "could any row here be visible as of
    ``as_of``," exactly mirroring how ``time_start`` alone answers the same
    question for valid time in ``can_contain_time``'s ``"<="`` branch.
    Either way, ``False`` here must mean "provably no row in this chunk can
    be visible as of ``as_of``" -- the same conservative contract as every
    other ``evaluate_against_label`` check.
    """

    if label.transaction_time_start is None:
        aligned = _align_literal_to_reference(label.time_start, as_of)
        return label.can_contain_time("<=", aligned)

    aligned = _align_literal_to_reference(label.transaction_time_start, as_of)
    return label.transaction_time_start <= aligned


def _apply_comparison(operator: str, left: Any, right: Any) -> bool:
    if operator == "=":
        return bool(left == right)
    if operator == "<":
        return bool(left < right)
    if operator == ">":
        return bool(left > right)
    if operator == "<=":
        return bool(left <= right)
    if operator == ">=":
        return bool(left >= right)
    raise ValueError(f"unsupported comparison operator: {operator!r}")


class PredicateNode(ABC):
    """Base class for all predicate tree nodes."""

    @abstractmethod
    def evaluate_against_label(
        self, label: ChunkLabel, price_pruning_enabled: bool = True
    ) -> bool:
        """Return whether a chunk might contain matching rows.

        This decision must be conservative: false positives are acceptable,
        but a false negative would silently discard valid results.

        ``price_pruning_enabled=False`` disables pruning based on a label's
        price zone map only -- symbol and time pruning remain safe and
        unaffected. A correction chunk's whole purpose is to override a base
        row's value regardless of whether the correction's own value happens
        to satisfy a value predicate the base row's original value did (or
        didn't); pruning a correction chunk by price can hide it from
        version resolution entirely, silently letting a stale base row win.
        Callers pass ``False`` once a store has any correction chunk
        anywhere (see ``ChunkStore.has_corrections`` / ``evaluate_pushdown``).
        """

    @abstractmethod
    def evaluate_against_row(self, row: dict[str, Any]) -> bool:
        """Return the exact predicate result for one decompressed row."""

    @abstractmethod
    def evaluate_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask equivalent to evaluate_against_row per-row.

        Used by QueryExecutor (the pushdown-query path) to avoid a
        Python-level loop over rows. FullScanQueryEngine deliberately keeps
        using evaluate_against_row instead, since it exists as the naive
        baseline that the pushdown-query path is measured against.
        """


@dataclass
class Comparison(PredicateNode):
    column: str
    operator: str
    value: Any
    label_field: str | None

    def __post_init__(self) -> None:
        _validate_canonical_binding(self.column, self.label_field)
        if self.operator not in _COMPARISON_OPERATORS:
            supported = ", ".join(sorted(_COMPARISON_OPERATORS))
            raise ValueError(
                f"unsupported comparison operator {self.operator!r}; "
                f"expected one of: {supported}"
            )

    def evaluate_against_label(
        self, label: ChunkLabel, price_pruning_enabled: bool = True
    ) -> bool:
        if self.label_field is None:
            return True

        if self.label_field == "symbol":
            if self.operator == "=":
                return label.can_contain_symbol(self.value)
            # A label has only one symbol identity, but string range pushdown is
            # intentionally unsupported. Keep every chunk as a candidate and
            # let exact row evaluation implement <, >, <=, or >=.
            return True

        if self.label_field == "time":
            value = _align_literal_to_reference(label.time_start, self.value)
            return label.can_contain_time(self.operator, value)

        if self.label_field == "price":
            if not price_pruning_enabled:
                return True
            return label.can_contain_price(self.operator, self.value)

        # Defensive conservative fallback if a future label category is added.
        return True

    def evaluate_against_row(self, row: dict[str, Any]) -> bool:
        row_value, literal = _coerce_row_operands(row[self.column], self.value)
        return _apply_comparison(self.operator, row_value, literal)

    def evaluate_vectorized(self, df: pd.DataFrame) -> pd.Series:
        column = df[self.column]
        literal = _coerce_column_literal(column, self.value)
        return _VECTORIZED_COMPARISONS[self.operator](column, literal)


@dataclass
class And(PredicateNode):
    children: list[PredicateNode]

    def __post_init__(self) -> None:
        self.children = list(self.children)
        if any(not isinstance(child, PredicateNode) for child in self.children):
            raise TypeError("every And child must be a PredicateNode")

    def evaluate_against_label(
        self, label: ChunkLabel, price_pruning_enabled: bool = True
    ) -> bool:
        return all(
            child.evaluate_against_label(label, price_pruning_enabled)
            for child in self.children
        )

    def evaluate_against_row(self, row: dict[str, Any]) -> bool:
        return all(child.evaluate_against_row(row) for child in self.children)

    def evaluate_vectorized(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for child in self.children:
            mask &= child.evaluate_vectorized(df)
        return mask


@dataclass
class In(PredicateNode):
    column: str
    values: list[Any]
    label_field: str | None

    def __post_init__(self) -> None:
        _validate_canonical_binding(self.column, self.label_field)
        self.values = list(self.values)

    def evaluate_against_label(
        self, label: ChunkLabel, price_pruning_enabled: bool = True
    ) -> bool:
        if self.label_field is None:
            return True

        if self.label_field == "symbol":
            return label.symbol in self.values

        if self.label_field == "time":
            return any(
                label.can_contain_time(
                    "=", _align_literal_to_reference(label.time_start, value)
                )
                for value in self.values
            )

        if self.label_field == "price":
            if not price_pruning_enabled:
                return True
            return any(label.can_contain_price("=", value) for value in self.values)

        return True

    def evaluate_against_row(self, row: dict[str, Any]) -> bool:
        return any(
            _apply_comparison("=", *_coerce_row_operands(row[self.column], value))
            for value in self.values
        )

    def evaluate_vectorized(self, df: pd.DataFrame) -> pd.Series:
        column = df[self.column]
        values = [_coerce_column_literal(column, value) for value in self.values]
        return column.isin(values)


@dataclass
class AlwaysTrue(PredicateNode):
    def evaluate_against_label(
        self, label: ChunkLabel, price_pruning_enabled: bool = True
    ) -> bool:
        return True

    def evaluate_against_row(self, row: dict[str, Any]) -> bool:
        return True

    def evaluate_vectorized(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=df.index)


def validate_predicate_columns(predicate: PredicateNode, columns: set[str]) -> None:
    """Reject references absent from the physical schema before pruning."""

    if isinstance(predicate, And):
        for child in predicate.children:
            validate_predicate_columns(child, columns)
    elif isinstance(predicate, (Comparison, In)) and predicate.column not in columns:
        raise ValueError(f"column {predicate.column!r} is absent from the store schema")


__all__ = [
    "AlwaysTrue",
    "And",
    "Comparison",
    "In",
    "PredicateNode",
    "label_visible_as_of",
    "validate_predicate_columns",
]
