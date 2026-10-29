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
    # "OF" is not a reserved keyword in sqlglot's tokenizer -- it comes
    # through as a plain TokenType.VAR, same as any identifier.
    return token.token_type == TokenType.VAR and token.text.upper() == "OF"


def extract_as_of(sql: str) -> tuple[str, datetime | None]:
    """Strip a ``TIMESTAMP AS OF '<literal>'`` clause, returning (sql, as_of).

    This project's sole supported time-travel form is a modifier immediately
    after ``FROM data``, rather than a WHERE predicate. That placement is how
    this function locates it: it
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
        as_of = parse_literal(literal_token.text, "timestamp", decoded=True)
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
        # sqlglot stores IN (SELECT ...) under the query argument rather than
        # expressions. It was rejected above; this guard documents the shape.
        if node.args.get("query") is not None:
            raise UnsupportedQueryError("Subqueries are not supported.")
        _reject_unused_arguments(node, {"this", "expressions"})
        if not node.expressions:
            raise UnsupportedQueryError("IN requires at least one literal")

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
        try:
            next_day = value + timedelta(days=1)
        except (ValueError, OverflowError) as exc:
            raise UnsupportedQueryError(
                "Date equality exceeds the datetime range"
            ) from exc
        return And(
            [
                Comparison(column, ">=", value, label_field),
                Comparison(
                    column,
                    "<",
                    next_day,
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
    # Qualifiers have already been validated against the sole FROM source.
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

    try:
        return parse_literal(_literal_text(node), column, decoded=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnsupportedQueryError(str(exc)) from exc


def _literal_is_date_only(node: exp.Expression) -> bool:
    node = _unwrap_parentheses(node)
    return isinstance(node, exp.Literal) and is_date_only(str(node.this), decoded=True)


__all__ = [
    "UnsupportedQueryError",
    "extract_as_of",
    "parse_where_clause",
]
