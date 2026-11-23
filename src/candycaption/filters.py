"""Review filters combined with AND, OR and NOT, in the manner of TagGUI.

Examples::

    signal:tags AND NOT reviewed:yes
    field:clothing OR caption:"red scarf"
    words>70 AND (file:cat OR tokens>=512)

A bare word matches the caption text. Terms next to each other are ANDed.
"""

from __future__ import annotations

import re
from typing import Any, Callable

TOKEN_RE = re.compile(r'\s*(\(|\)|"[^"]*"|[^\s()"]+(?:"[^"]*")?)')
TERM_RE = re.compile(r"^(\w+)(:|>=|<=|>|<|=)(.*)$")
SIGNAL_ALIASES = {"uncertain": "uncertain", "words": "uncertain", "mixed": "uncertain", "checks": "checks", "check": "checks",
                  "taste": "checks", "tags": "tags", "tag": "tags", "clash": "tags", "template": "template", "same": "template"}

Predicate = Callable[[dict[str, Any]], bool]


class FilterError(ValueError):
    pass


def _number_test(op: str, wanted: float) -> Callable[[float], bool]:
    return {":": lambda v: v == wanted, "=": lambda v: v == wanted, ">": lambda v: v > wanted,
            "<": lambda v: v < wanted, ">=": lambda v: v >= wanted, "<=": lambda v: v <= wanted}[op]


def _term(token: str) -> Predicate:
    match = TERM_RE.match(token)
    if not match:
        text = token.strip('"').lower()
        return lambda row: text in row["caption"].lower()
    key, op, value = match.group(1).lower(), match.group(2), match.group(3).strip('"').lower()
    if key in ("words", "tokens", "score"):
        try:
            test = _number_test(op, float(value))
        except ValueError:
            raise FilterError(f"{key} needs a number, as in {key}>70")
        return lambda row: test(row[key])
    if op != ":":
        raise FilterError(f"Use a colon with {key}, as in {key}:value")
    if key in ("signal", "flag"):
        if value in ("any", "*"):
            return lambda row: bool(row["reasons"])
        signal = SIGNAL_ALIASES.get(value)
        if not signal:
            raise FilterError("signal is one of uncertain, checks, tags, template or any")
        return lambda row: any(r["signal"] == signal for r in row["reasons"])
    if key == "field":
        return lambda row: any(value in (r.get("field") or "") or value in (r.get("field_name") or "").lower() for r in row["reasons"])
    if key == "caption":
        return lambda row: value in row["caption"].lower()
    if key in ("file", "filename", "name"):
        return lambda row: value in row["rel"].lower()
    if key in ("tag", "tags"):
        return lambda row: value in row["tags"].lower()
    if key == "reviewed":
        want = value in ("yes", "true", "1", "y")
        return lambda row: bool(row["reviewed"]) == want
    if key == "status":
        return lambda row: row["status"] == value
    if key == "rating":
        return lambda row: (row["fields"] or {}).get("rating", "").lower() == value
    if key == "health":
        return lambda row: any(value in h["check"] for h in row.get("health") or [])
    raise FilterError(f"Unknown filter \"{key}\". Try signal, field, caption, file, tags, words, tokens, reviewed, status, rating or health.")


def compile_filter(query: str) -> Predicate:
    tokens = TOKEN_RE.findall(query or "")
    if not tokens:
        return lambda row: True
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take() -> str:
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def parse_or() -> Predicate:
        left = parse_and()
        while peek() == "OR":
            take()
            right, prev = parse_and(), left
            left = lambda row, a=prev, b=right: a(row) or b(row)
        return left

    def parse_and() -> Predicate:
        left = parse_not()
        while peek() not in (None, ")", "OR"):
            if peek() == "AND":
                take()
            right, prev = parse_not(), left
            left = lambda row, a=prev, b=right: a(row) and b(row)
        return left

    def parse_not() -> Predicate:
        token = peek()
        if token is None:
            raise FilterError("The filter ends too early.")
        if token == "NOT":
            take()
            inner = parse_not()
            return lambda row: not inner(row)
        if token == "(":
            take()
            inner = parse_or()
            if peek() != ")":
                raise FilterError("A bracket is not closed.")
            take()
            return inner
        if token in (")", "AND", "OR"):
            raise FilterError(f"Unexpected {token} in the filter.")
        return _term(take())

    result = parse_or()
    if pos < len(tokens):
        raise FilterError(f"Unexpected {tokens[pos]} in the filter.")
    return result
