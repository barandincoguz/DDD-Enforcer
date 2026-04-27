from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from core.AST.ast_signal_types import GroundingMatch
from core.AST.ast_signal_utils import line_is_requirement, line_is_section_like, normalized_terms


def rank_groundings(term: str, docs: List[Dict[str, Any]], limit: int = 3) -> List[GroundingMatch]:
    if not term:
        return []

    exact = re.compile(rf"\b{re.escape(term)}s?\b", flags=re.IGNORECASE)
    target_terms = normalized_terms(term)
    matches: List[GroundingMatch] = []
    seen: Set[tuple[str, int, str]] = set()

    for doc in docs:
        lines = (doc.get("content") or "").splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or "do not use" in line.lower():
                continue

            line_terms = normalized_terms(line)
            overlap = len(target_terms & line_terms)
            exact_hit = bool(exact.search(line))
            if not exact_hit and overlap == 0:
                continue

            score = overlap + (1.5 if target_terms and target_terms.issubset(line_terms) else 0.0)
            rule = "SRS_GROUNDING_NORMALIZED"
            if exact_hit:
                score += 5.0
                rule = "SRS_GROUNDING_EXACT"
            if line_is_requirement(line):
                score += 1.0
            if line_is_section_like(line):
                score += 0.5

            key = (doc.get("path", "srs_document"), line_number, line)
            if key in seen:
                continue
            seen.add(key)
            matches.append(GroundingMatch(file=key[0], line=line_number, snippet=line, score=score, rule=rule))

    matches.sort(key=lambda item: (item.score, -item.line, -len(item.snippet)), reverse=True)
    return matches[:limit]
