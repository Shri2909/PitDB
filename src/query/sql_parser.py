"""Translate the supported SQL WHERE subset into predicate-tree nodes."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import sqlglot
from sqlglot import expressions as exp
from sqlglot.tokens import Token, Tokenizer, TokenType

from src.query.predicate import AlwaysTrue, And, Comparison, In, PredicateNode
from src.query.schema import (
    COLUMN_LABEL_MAP,
    SCHEMA,
    ColumnType,
    is_date_only,
    parse_literal,
    validate_column,
)


class UnsupportedQueryError(Exception):
    """Raised when SQL falls outside this project's bounded query grammar."""


_SWAPPED_OPERATOR = {
    "=": "=",
    "<": ">",
    ">": "<",
    "<=": ">=",
    ">=": "<=",
}

_AS_OF_TOKENIZER = Tokenizer()


def parse_where_clause(sql: str) -> PredicateNode:
    """Parse a complete SQL statement into a conservative predicate tree."""

    parsed = sqlglot.parse_one(sql)
    _validate_statement_scope(parsed)

    where_clause = parsed.find(exp.Where)
    if where_clause is None:
        return AlwaysTrue()
    return _convert_expression(where_clause.this)


def _is_of_token(token: Token) -> bool:
    # "OF" is not a reserved keyword in sqlglot's tokenizer -- it comes
    # through as a plain TokenType.VAR, same as any identifier.
    return token.token_type == TokenType.VAR and token.text.upper() == "OF"


def extract_as_of(sql: str) -> tuple[str, datetime | None]:
    """Strip a ``TIMESTAMP AS OF '<literal>'`` clause, returning (sql, as_of).

    This is the same syntax and placement Delta Lake and SQL Server use for
    point-in-time queries (``FROM table TIMESTAMP AS OF ts`` / ``FOR
    SYSTEM_TIME AS OF ts``): a modifier on the table reference, not a WHERE
    predicate. That placement is exactly how this function locates it: it
    tokenizes the raw SQL (lexical analysis only -- never a full parse, and
    never a re-serialization through a different dialect) and looks at the
    single token immediately following the table name. Anywhere else, a bare
    ``TIMESTAMP`` token is just an ordinary column reference (this project's
    own queries write ``WHERE timestamp >= ...`` constantly, and "timestamp"
    tokenizes as ``TokenType.TIMESTAMP`` in that position too) -- so matching
    only makes sense pinned to this one structural position.

    An earlier version located the clause with a regex restricted to the SQL
    text before the first bare ``WHERE`` match. That was defeated by a
    ``WHERE`` before the real one -- e.g. a leading SQL comment containing
    the ordinary English word "where" -- silently truncating the search
    before it ever reached the real clause. Before that, an even earlier
    version parsed the *entire* statement under the Databricks dialect and
    re-serialized the remainder, which corrupted quoted identifiers and
    escaped-quote literals in the WHERE clause. Tokenizing sidesteps both
    failure modes: comments are discarded before tokens are ever produced,
    and a string literal (however it's quoted or escaped) always comes
    through as one opaque ``STRING`` token, never text a keyword search
    could be fooled by. The matched span's exact character offsets in the
    *original* SQL text are used to excise it, leaving every other character
    -- including exact quoting elsewhere in the query -- untouched before
    handing the remainder to the unmodified, default-dialect
    ``parse_where_clause``.
    """

    tokens = _AS_OF_TOKENIZER.tokenize(sql)

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
        as_of = parse_literal(f"'{literal_token.text}'", "timestamp")
        if not isinstance(as_of, datetime):
            raise UnsupportedQueryError("AS OF literal did not parse to a timestamp")

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

    # Reject subqueries even if their WHERE is the first exp.Where found. This
    # keeps parse_one(...).find(exp.Where) from accidentally treating an inner
    # query's predicate as the outer statement's predicate.
    if parsed.find(exp.Subquery) is not None or parsed.find(exp.Exists) is not None:
        raise UnsupportedQueryError("Subqueries are not supported.")

    if parsed.find(exp.Join) is not None:
        raise UnsupportedQueryError("JOINs are not supported.")

    select = parsed if isinstance(parsed, exp.Select) else parsed.find(exp.Select)
    if select is not None:
        for projection in select.expressions:
            if projection.find(exp.AggFunc) is not None:
                raise UnsupportedQueryError(
                    "Aggregation functions are not supported; only "
                    "WHERE-clause filtering is executed."
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
        left = _convert_expression(node.left)
        right = _convert_expression(node.right)
        children: list[PredicateNode] = []
        for child in (left, right):
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
        # sqlglot stores IN (SELECT ...) under the query argument rather than
        # expressions. It was rejected above; this guard documents the shape.
        if node.args.get("query") is not None:
            raise UnsupportedQueryError("Subqueries are not supported.")

        column = _extract_column(node.this)
        label_field = COLUMN_LABEL_MAP[column]
        if SCHEMA[column] is ColumnType.DATETIME:
            # Unlike `=` (see _convert_comparison), IN has no whole-day
            # expansion for a date-only literal -- that would need an OR of
            # per-value ranges, which this project's grammar intentionally
            # never constructs (OR across columns is explicitly rejected
            # above). Silently matching only exact midnight would drop every
            # intraday row for that date, so reject explicitly instead.
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

    # SQL date equality means the complete calendar day. An exact midnight
    # comparison would incorrectly return no intraday rows.
    if (
        normalized_operator == "="
        and SCHEMA[column] is ColumnType.DATETIME
        and _literal_is_date_only(literal_node)
    ):
        if not isinstance(value, datetime):
            raise UnsupportedQueryError(
                "date-only equality did not produce a datetime literal"
            )
        return And(
            [
                Comparison(column, ">=", value, label_field),
                Comparison(
                    column,
                    "<",
                    value + timedelta(days=1),
                    label_field,
                ),
            ]
        )

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
    # exp.Column.name deliberately omits any table/catalog qualifier.
    return validate_column(node.name)


def _literal_text(node: exp.Expression) -> str:
    node = _unwrap_parentheses(node)
    if isinstance(node, exp.Literal):
        return str(node.this)

    if isinstance(node, exp.Neg):
        inner = _unwrap_parentheses(node.this)
        if not isinstance(inner, exp.Literal) or inner.is_string:
            raise UnsupportedQueryError(
                "unary minus is supported only for numeric literals"
            )
        return f"-{inner.this}"

    raise UnsupportedQueryError(f"expected a literal, found {type(node).__name__}")


def _extract_literal(node: exp.Expression, column: str) -> Any:
    """Parse a literal using the canonical column's declared schema type."""

    return parse_literal(_literal_text(node), column)


def _literal_is_date_only(node: exp.Expression) -> bool:
    node = _unwrap_parentheses(node)
    return isinstance(node, exp.Literal) and is_date_only(str(node.this))


__all__ = [
    "UnsupportedQueryError",
    "extract_as_of",
    "parse_where_clause",
]
