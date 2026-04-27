from __future__ import annotations

import re
from typing import Iterable, List, Set


IDENTIFIER_SPLIT_PATTERN = re.compile(r"(?<!^)(?=[A-Z])|[_\.\-/\s]+")
REQUIREMENT_HINTS = {
    "must",
    "shall",
    "required",
    "cannot",
    "should",
    "when",
    "if",
    "only",
}


def split_identifier(value: str) -> List[str]:
    if not value:
        return []
    return [part for part in IDENTIFIER_SPLIT_PATTERN.split(value) if part]


def singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_token(token: str) -> str:
    lowered = re.sub(r"[^a-z0-9]", "", token.lower())
    return singularize(lowered)


def normalized_terms(*values: str) -> Set[str]:
    terms: Set[str] = set()
    for value in values:
        for raw in split_identifier(value):
            normalized = normalize_token(raw)
            if normalized:
                terms.add(normalized)
    return terms


def collect_terms(values: Iterable[str]) -> Set[str]:
    terms: Set[str] = set()
    for value in values:
        terms.update(normalized_terms(value))
    return terms


def line_is_requirement(line: str) -> bool:
    tokens = normalized_terms(line)
    return bool(tokens & REQUIREMENT_HINTS)


def line_is_section_like(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) <= 90 and (
        stripped.isupper() or re.match(r"^\d+(?:\.\d+)*\.?\s+\S", stripped)
    ):
        return True
    return stripped.endswith(":") and len(stripped) <= 90


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
