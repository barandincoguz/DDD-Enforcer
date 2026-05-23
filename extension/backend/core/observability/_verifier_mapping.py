"""Verifier issue → canonical check_id mapping (Codex C-5).

Source of truth for resolving both legacy `VerifierIssue.issue_type` (deterministic
checks D1..D5 + S1 use semantic strings) and contract `VerifierIssue.check_id`
(D6..D8 use explicit "D6"/"D7"/"D8") into one consistent D1..D8/S1 namespace.

Used by `StageEmitter.record_verifier_result` to bucket counts per rule.
"""

from __future__ import annotations

from typing import Any


# Built from `grep -n issue_type= core/verifier/checks_*.py` at WP-CORE-20
# implementation time. Re-verify when verifier check set evolves.
_ISSUE_TYPE_TO_CHECK_ID = {
    "ungrounded_context": "D1",
    "missing_evidence": "D2",
    "duplicate_entity_across_contexts": "D3",
    "invalid_aggregate_member": "D4",
    "unknown_dependency": "D5",
    "evidence_indices_out_of_range": "S1",
    "semantic_ungrounded": "S1",
    "D6": "D6",
    "D7": "D7",
    "D8": "D8",
}


def canonical_check_id(issue: Any) -> str:
    """Resolve a VerifierIssue (legacy dataclass OR contract Pydantic) to D1..D8/S1.

    Precedence: explicit `check_id` field with a known D-code wins over
    `issue_type`. If neither matches, return "unknown" (defensive; an unknown
    bucket lets the aggregator surface unexpected rule additions instead of
    silently dropping them).
    """
    check_id = getattr(issue, "check_id", None)
    if check_id in ("D6", "D7", "D8"):
        return check_id
    issue_type = getattr(issue, "issue_type", None) or check_id or ""
    return _ISSUE_TYPE_TO_CHECK_ID.get(issue_type, "unknown")


# Canonical bucket order for stable manifest output.
CANONICAL_CHECK_IDS = ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "S1")


def empty_issue_counts() -> dict[str, int]:
    """Return a zero-initialized counts dict in canonical order."""
    return {cid: 0 for cid in CANONICAL_CHECK_IDS}
