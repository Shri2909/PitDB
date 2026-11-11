"""Translate the supported SQL WHERE subset into predicate-tree nodes.

A date-only literal (``'2024-01-01'`` or ``'20240101'``) names the whole day,
so comparisons are rewritten onto day boundaries and ``TIMESTAMP AS OF`` means
the end of that day. A literal with a time component is an exact instant.
``IN`` rejects date-only datetimes; bare numbers are never timestamps or
symbols. All literal problems raise :class:`UnsupportedQueryError`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import expressions as exp
from sqlglot.tokens import Token, Tokenizer, TokenType

from src.query.predicate import AlwaysTrue, And, Comparison, In, PredicateNode
from src.query.schema import (
    COLUMN_LABEL_MAP,
    MAX_TIMESTAMP,
    SCHEMA,
    SUPPORTED_TIMESTAMP_RANGE,
    ColumnType,
    is_date_only,
    parse_literal,
    validate_column,
)


class UnsupportedQueryError(ValueError):
    """Raised for SQL outside the bounded grammar or a literal that cannot be bound.

    Subclasses ``ValueError``.
    """


_SWAPPED_OPERATOR = {
    "=": "=",
    "<": ">",
    ">": "<",
    "<=": ">=",
    ">=": "<=",
}

_AS_OF_TOKENIZER = Tokenizer()
_MAX_SQL_CHARACTERS = 65_536
_MAX_SQL_TOKENS = 8_192
_MAX_PARENTHESIS_DEPTH = 32
_MAX_CONJUNCTIONS = 255


def _tokenize_bounded(sql: str) -> list[Token]:
    """Apply lexical resource and single-statement limits before parsing."""

    if not isinstance(sql, str):
        raise UnsupportedQueryError("SQL must be a string")
    if len(sql) > _MAX_SQL_CHARACTERS:
        raise UnsupportedQueryError("SQL exceeds the 65536-character limit")
    try:
        tokens = _AS_OF_TOKENIZER.tokenize(sql)
    except sqlglot.errors.SqlglotError as exc:
        raise UnsupportedQueryError("Invalid SQL tokenization") from exc
    if len(tokens) > _MAX_SQL_TOKENS:
        raise UnsupportedQueryError("SQL exceeds the 8192-token limit")

    depth = 0
    conjunctions = 0
    for index, token in enumerate(tokens):
        if token.token_type == TokenType.SEMICOLON and index != len(tokens) - 1:
            raise UnsupportedQueryError("Exactly one SQL statement is supported")
        if token.token_type == TokenType.L_PAREN:
            depth += 1
            if depth > _MAX_PARENTHESIS_DEPTH:
                raise UnsupportedQueryError("SQL exceeds the 32-level nesting limit")
        elif token.token_type == TokenType.R_PAREN:
            depth -= 1
            if depth < 0:
                raise UnsupportedQueryError("Unbalanced SQL parentheses")
        elif token.token_type == TokenType.AND:
            conjunctions += 1
            if conjunctions > _MAX_CONJUNCTIONS:
                raise UnsupportedQueryError("SQL exceeds the 256-predicate limit")
    if depth:
        raise UnsupportedQueryError("Unbalanced SQL parentheses")
    return tokens


def parse_where_clause(sql: str) -> PredicateNode:
    """Parse a complete SQL statement into a conservative predicate tree."""

    _tokenize_bounded(sql)
    try:
        statements = sqlglot.parse(sql)
        if len(statements) != 1 or statements[0] is None:
            raise UnsupportedQueryError("Exactly one SELECT statement is required")
        parsed = statements[0]
        _validate_statement_scope(parsed)

        where_clause = parsed.args.get("where")
        if where_clause is None:
            return AlwaysTrue()
        return _convert_expression(where_clause.this)
    except sqlglot.errors.SqlglotError as exc:
        raise UnsupportedQueryError("Invalid or unsupported SQL syntax") from exc
    except RecursionError as exc:
        raise UnsupportedQueryError("SQL expression nesting is too deep") from exc


def _is_of_token(token: Token) -> bool:
    # sqlglot tokenizes OF as a plain VAR, not a keyword.
    return token.token_type == TokenType.VAR and token.text.upper() == "OF"


def extract_as_of(sql: str) -> tuple[str, datetime | None]:
    """Strip a ``TIMESTAMP AS OF '<literal>'`` clause, returning (sql, as_of).

    The clause is only recognized immediately after ``FROM data`` by tokenizing
    (never fully parsing) the raw SQL, so comments, quoting and string literals
    elsewhere in the query are left untouched.
    """

    tokens = _tokenize_bounded(sql)

    from_index = next(
        (i for i, token in enumerate(tokens) if token.token_type == TokenType.FROM),
        None,
    )
    if from_index is None:
        return sql, None

    marker_index = from_index + 2  # FROM, <table name>, <marker>
    if marker_index >= len(tokens):
        return sql, None
    marker = tokens[marker_index]

    if marker.token_type == TokenType.TIMESTAMP:
        literal_index = marker_index + 3
        has_as_of_literal = (
            marker_index + 2 < len(tokens)
            and tokens[marker_index + 1].token_type == TokenType.ALIAS
            and _is_of_token(tokens[marker_index + 2])
            and literal_index < len(tokens)
            and tokens[literal_index].token_type == TokenType.STRING
        )
        if not has_as_of_literal:
            raise UnsupportedQueryError("AS OF requires a literal timestamp")

        literal_token = tokens[literal_index]
        as_of = _parse_literal_or_raise(literal_token.text, "timestamp")
        if not isinstance(as_of, datetime):
            raise UnsupportedQueryError("AS OF literal did not parse to a timestamp")
        if is_date_only(literal_token.text, decoded=True):
            # AS OF a calendar day means "as known at the end of that day".
            as_of = _end_of_day(as_of)

        start, end = marker.start, literal_token.end
        return sql[:start] + sql[end + 1 :], as_of

    is_version_as_of = (
        marker.token_type == TokenType.VAR
        and marker.text.upper() == "VERSION"
        and marker_index + 2 < len(tokens)
        and tokens[marker_index + 1].token_type == TokenType.ALIAS
        and _is_of_token(tokens[marker_index + 2])
    )
    if is_version_as_of:
        raise UnsupportedQueryError(
            "only TIMESTAMP AS OF is supported, not VERSION AS OF"
        )

    return sql, None


def _validate_statement_scope(parsed: exp.Expression) -> None:
    """Reject full-statement features that the executor cannot implement."""

    if parsed.find(exp.Subquery) is not None or parsed.find(exp.Exists) is not None:
        raise UnsupportedQueryError("Subqueries are not supported.")

    if parsed.find(exp.Join) is not None:
        raise UnsupportedQueryError("JOINs are not supported.")

    if isinstance(parsed, exp.Select):
        for projection in parsed.expressions:
            if projection.find(exp.AggFunc) is not None:
                raise UnsupportedQueryError(
                    "Aggregation functions are not supported; only "
                    "WHERE-clause filtering is executed."
                )

    if not isinstance(parsed, exp.Select):
        raise UnsupportedQueryError("Only SELECT * FROM data is supported")
    _reject_unused_arguments(parsed, {"expressions", "from", "from_", "where"})
    if (
        len(parsed.expressions) != 1
        or not isinstance(parsed.expressions[0], exp.Star)
        or any(parsed.expressions[0].args.values())
    ):
        raise UnsupportedQueryError(
            "Only the unmodified SELECT * projection is supported"
        )

    source = parsed.args.get("from") or parsed.args.get("from_")
    if not isinstance(source, exp.From):
        raise UnsupportedQueryError("FROM data is required")
    _reject_unused_arguments(source, {"this"})
    table = source.this
    if (
        not isinstance(table, exp.Table)
        or not isinstance(table.this, exp.Identifier)
        or table.name.lower() != "data"
    ):
        raise UnsupportedQueryError("Only the data table is supported")
    _reject_unused_arguments(table, {"this", "alias"})
    qualifier = "data"
    alias = table.args.get("alias")
    if alias is not None:
        if not isinstance(alias, exp.TableAlias) or not isinstance(
            alias.this, exp.Identifier
        ):
            raise UnsupportedQueryError("Only a simple table alias is supported")
        _reject_unused_arguments(alias, {"this"})
        if not alias.name:
            raise UnsupportedQueryError("Table aliases must not be empty")
        qualifier = alias.name.lower()

    for column in parsed.find_all(exp.Column):
        _reject_unused_arguments(column, {"this", "table"})
        if not isinstance(column.this, exp.Identifier):
            raise UnsupportedQueryError("Expected a simple column name")
        table_identifier = column.args.get("table")
        if table_identifier is not None and (
            not isinstance(table_identifier, exp.Identifier)
            or not column.table
            or column.table.lower() != qualifier
        ):
            raise UnsupportedQueryError(
                "Column qualifier must match the data table alias"
            )


def _reject_unused_arguments(node: exp.Expression, allowed: set[str]) -> None:
    """Reject every active AST field that the bounded executor does not use."""

    for key, value in node.args.items():
        if key not in allowed and value:
            raise UnsupportedQueryError(
                f"Unsupported {type(node).__name__} clause: {key}"
            )


def _convert_expression(node: exp.Expression) -> PredicateNode:
    """Recursively convert one sqlglot expression into the bounded IR."""

    if isinstance(node, exp.Paren):
        return _convert_expression(node.this)

    if isinstance(node, (exp.Subquery, exp.Exists)):
        raise UnsupportedQueryError("Subqueries are not supported.")
    if node.find(exp.Subquery) is not None or node.find(exp.Exists) is not None:
        raise UnsupportedQueryError("Subqueries are not supported.")

    if isinstance(node, exp.Or):
        raise UnsupportedQueryError(
            "OR across columns is not supported. Use IN for multi-value "
            "single-column filters."
        )

    if isinstance(node, exp.And):
        children: list[PredicateNode] = []
        pending = [node]
        while pending:
            current = _unwrap_parentheses(pending.pop())
            if isinstance(current, exp.And):
                pending.extend((current.right, current.left))
                continue
            child = _convert_expression(current)
            if isinstance(child, And):
                children.extend(child.children)
            else:
                children.append(child)
        return And(children)

    comparisons: tuple[tuple[type[exp.Expression], str], ...] = (
        (exp.EQ, "="),
        (exp.GT, ">"),
        (exp.GTE, ">="),
        (exp.LT, "<"),
        (exp.LTE, "<="),
    )
    for expression_type, operator in comparisons:
        if isinstance(node, expression_type):
            return _convert_comparison(node, operator)

    if isinstance(node, exp.In):
        if node.args.get("query") is not None:
            raise UnsupportedQueryError("Subqueries are not supported.")
        _reject_unused_arguments(node, {"this", "expressions"})
        if not node.expressions:
            raise UnsupportedQueryError("IN requires at least one literal")

        column = _extract_column(node.this)
        label_field = COLUMN_LABEL_MAP[column]
        if SCHEMA[column] is ColumnType.DATETIME:
            # No whole-day expansion exists for IN without OR, so reject explicitly.
            for value_node in node.expressions:
                if _literal_is_date_only(value_node):
                    raise UnsupportedQueryError(
                        "IN with date-only literals is not supported; use an "
                        "explicit range or exact timestamps"
                    )
        values = [
            _extract_literal(value_node, column) for value_node in node.expressions
        ]
        return In(column, values, label_field)

    raise UnsupportedQueryError(f"Unsupported expression type: {type(node).__name__}")


def _convert_comparison(node: exp.Expression, operator: str) -> PredicateNode:
    column, literal_node, normalized_operator = _comparison_parts(node, operator)
    label_field = COLUMN_LABEL_MAP[column]
    value = _extract_literal(literal_node, column)

    # A date-only literal means the whole day, so rewrite onto day boundaries.
    if SCHEMA[column] is ColumnType.DATETIME and _literal_is_date_only(literal_node):
        if not isinstance(value, datetime):
            raise UnsupportedQueryError(
                "date-only comparison did not produce a datetime literal"
            )
        if normalized_operator in {"<", ">="}:
            return Comparison(column, normalized_operator, value, label_field)
        next_day = _next_day(value)
        if normalized_operator == "=":
            return And(
                [
                    Comparison(column, ">=", value, label_field),
                    Comparison(column, "<", next_day, label_field),
                ]
            )
        if normalized_operator == "<=":
            return Comparison(column, "<", next_day, label_field)
        return Comparison(column, ">=", next_day, label_field)  # ">"

    return Comparison(column, normalized_operator, value, label_field)


def _comparison_parts(
    node: exp.Expression, operator: str
) -> tuple[str, exp.Expression, str]:
    """Return canonical column, literal node, and correctly oriented operator."""

    left = _unwrap_parentheses(node.left)
    right = _unwrap_parentheses(node.right)
    left_is_column = isinstance(left, exp.Column)
    right_is_column = isinstance(right, exp.Column)

    if left_is_column and not right_is_column:
        return _extract_column(left), right, operator
    if right_is_column and not left_is_column:
        return _extract_column(right), left, _SWAPPED_OPERATOR[operator]

    raise UnsupportedQueryError(
        "comparisons require exactly one column and one literal"
    )


def _unwrap_parentheses(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _extract_column(node: exp.Expression) -> str:
    """Extract and canonicalize a possibly table-qualified column reference."""

    node = _unwrap_parentheses(node)
    if not isinstance(node, exp.Column):
        raise UnsupportedQueryError(
            f"expected a column reference, found {type(node).__name__}"
        )
    return validate_column(node.name)


def _literal_text(node: exp.Expression) -> tuple[str, bool]:
    """Return a literal's decoded text and whether it was a quoted string."""

    node = _unwrap_parentheses(node)
    if isinstance(node, exp.Literal):
        return str(node.this), bool(node.is_string)

    if isinstance(node, exp.Neg):
        inner = _unwrap_parentheses(node.this)
        if not isinstance(inner, exp.Literal) or inner.is_string:
            raise UnsupportedQueryError(
                "unary minus is supported only for numeric literals"
            )
        return f"-{inner.this}", False

    raise UnsupportedQueryError(f"expected a literal, found {type(node).__name__}")


def _parse_literal_or_raise(text: str, column: str) -> Any:
    """Parse decoded literal text, raising UnsupportedQueryError on any failure."""

    try:
        return parse_literal(text, column, decoded=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedQueryError(str(exc)) from exc


def _extract_literal(node: exp.Expression, column: str) -> Any:
    """Parse a literal using the canonical column's declared schema type."""

    text, is_string = _literal_text(node)
    if not is_string and SCHEMA[column] is ColumnType.STRING:
        raise UnsupportedQueryError(
            f"column {column!r} requires a quoted string literal; received "
            f"the number {text}"
        )
    if not is_string and SCHEMA[column] is ColumnType.DATETIME:
        raise UnsupportedQueryError(
            f"ambiguous numeric literal {text} for column {column!r}; "
            "write a quoted ISO 8601 date or timestamp such as '2024-01-01'"
        )
    return _parse_literal_or_raise(text, column)


def _next_day(value: datetime) -> datetime:
    """Midnight that ends the calendar day starting at ``value``."""

    try:
        result = pd.Timestamp(value) + pd.Timedelta(days=1)
        return result.as_unit("ns")
    except (ValueError, OverflowError) as exc:
        raise UnsupportedQueryError(
            "date literal is too close to the end of the supported datetime "
            f"range ({SUPPORTED_TIMESTAMP_RANGE}) to compare by whole day"
        ) from exc


def _end_of_day(value: datetime) -> datetime:
    """Last representable nanosecond of the day starting at ``value``.

    Clamped to the largest representable timestamp: an AS OF cutoff at or past
    it already makes every transaction visible, so clamping is exact.
    """

    try:
        return _next_day(value) - pd.Timedelta(1, unit="ns")
    except UnsupportedQueryError:
        return MAX_TIMESTAMP


def _literal_is_date_only(node: exp.Expression) -> bool:
    node = _unwrap_parentheses(node)
    return (
        isinstance(node, exp.Literal)
        and bool(node.is_string)
        and is_date_only(str(node.this), decoded=True)
    )


__all__ = [
    "UnsupportedQueryError",
    "extract_as_of",
    "parse_where_clause",
]
