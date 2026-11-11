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
    """Vectorized counterpart to :func:`_coerce_row_operands`: align once per column."""

    if isinstance(literal, datetime):
        literal = timestamp_value(literal, name="datetime literal")
        if column.empty:
            return literal
        return align_timestamp(column.iloc[0], literal)

    return literal


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and value != value


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
            return row_value, literal

    return row_value, literal


def _align_literal_to_reference(reference: datetime, literal: datetime) -> datetime:
    """Match ``literal``'s tz-awareness to ``reference``, never altering the label."""

    return align_timestamp(reference, literal)


def label_visible_as_of(label: ChunkLabel, as_of: datetime) -> bool:
    """Return False only if no row in the chunk can be visible as of ``as_of``.

    Base chunks are learned at their own valid time, so ``time_start`` is their
    earliest transaction time; correction chunks use ``transaction_time_start``.
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
        """Return whether a chunk might contain matching rows (never a false negative).

        ``price_pruning_enabled=False`` disables price zone-map pruning only; it
        is required once a store has corrections, which must stay visible to
        version resolution.
        """

    @abstractmethod
    def evaluate_against_row(self, row: dict[str, Any]) -> bool:
        """Return the exact predicate result for one decompressed row."""

    @abstractmethod
    def evaluate_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask equivalent to ``evaluate_against_row`` per row."""


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
            # String range pushdown is unsupported; exact row evaluation decides.
            return True

        if self.label_field == "time":
            value = _align_literal_to_reference(label.time_start, self.value)
            return label.can_contain_time(self.operator, value)

        if self.label_field == "price":
            if not price_pruning_enabled:
                return True
            return label.can_contain_price(self.operator, self.value)

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
        # Series.isin matches NaN to NaN, unlike row-wise equality; drop NaN literals.
        values = [
            _coerce_column_literal(column, value)
            for value in self.values
            if not _is_nan(value)
        ]
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


def iter_datetime_literals(predicate: PredicateNode):
    """Yield every datetime literal held by a predicate tree."""

    if isinstance(predicate, And):
        for child in predicate.children:
            yield from iter_datetime_literals(child)
    elif isinstance(predicate, Comparison):
        if isinstance(predicate.value, datetime):
            yield predicate.value
    elif isinstance(predicate, In):
        for value in predicate.values:
            if isinstance(value, datetime):
                yield value


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
    "iter_datetime_literals",
    "label_visible_as_of",
    "validate_predicate_columns",
]
