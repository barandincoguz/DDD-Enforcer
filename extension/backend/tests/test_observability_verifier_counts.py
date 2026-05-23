"""WP-CORE-20 — Verifier issue-counter mapping tests (RED phase).

Covers Codex C-5: legacy VerifierIssue (issue_type) and contract VerifierIssue
(check_id) both canonicalize to D1..D8/S1 via the same mapping.
"""
from __future__ import annotations

from typing import Optional

import pytest


class _FakeIssue:
    """Typed duck for both legacy (issue_type) and contract (check_id) VerifierIssue."""
    def __init__(self, issue_type: Optional[str] = None, check_id: Optional[str] = None):
        self.issue_type = issue_type
        self.check_id = check_id


@pytest.mark.parametrize(
    "issue_type, expected",
    [
        ("ungrounded_context", "D1"),
        ("missing_evidence", "D2"),
        ("duplicate_entity_across_contexts", "D3"),
        ("invalid_aggregate_member", "D4"),
        ("unknown_dependency", "D5"),
        ("evidence_indices_out_of_range", "S1"),
        ("semantic_ungrounded", "S1"),
        ("D6", "D6"),
        ("D7", "D7"),
        ("D8", "D8"),
    ],
)
def test_t_verifier_counts_1_legacy_issue_type_to_check_id(issue_type, expected):
    """T-VERIFIER-COUNTS-1: legacy issue_type strings map to canonical D-codes."""
    from core.observability._verifier_mapping import canonical_check_id

    issue = _FakeIssue(issue_type=issue_type)
    assert canonical_check_id(issue) == expected


def test_t_verifier_counts_2_contract_check_id_passthrough():
    """T-VERIFIER-COUNTS-2: D6/D7/D8 contract VerifierIssue check_id passes through."""
    from core.observability._verifier_mapping import canonical_check_id

    issue = _FakeIssue(check_id="D7")
    assert canonical_check_id(issue) == "D7"


def test_t_verifier_counts_3_unknown_issue_type_returns_unknown():
    """T-VERIFIER-COUNTS-3: unknown issue_type returns 'unknown' (defensive fallback)."""
    from core.observability._verifier_mapping import canonical_check_id

    issue = _FakeIssue(issue_type="something_brand_new")
    assert canonical_check_id(issue) == "unknown"


def test_t_verifier_counts_4_emitter_record_verifier_result_buckets():
    """T-VERIFIER-COUNTS-4: emitter.record_verifier_result buckets issues by canonical check_id."""
    from core.observability import RunManifest, StageEmitter

    class FakeVerifierIssue:
        def __init__(self, issue_type: Optional[str] = None, check_id: Optional[str] = None):
            self.issue_type = issue_type
            self.check_id = check_id

    class FakeVerifierResult:
        def __init__(self, issues):
            self.issues = issues

    m = RunManifest()
    em = StageEmitter(m)
    issues = [
        FakeVerifierIssue(issue_type="ungrounded_context"),  # D1
        FakeVerifierIssue(issue_type="ungrounded_context"),  # D1 again
        FakeVerifierIssue(issue_type="missing_evidence"),    # D2
        FakeVerifierIssue(check_id="D6"),                    # D6
        FakeVerifierIssue(check_id="D8"),                    # D8
    ]
    with em.stage("verifier"):
        em.record_verifier_result(FakeVerifierResult(issues=issues))

    counts = m.stages["verifier"].metrics["issue_counts_by_check"]
    assert counts["D1"] == 2
    assert counts["D2"] == 1
    assert counts["D6"] == 1
    assert counts["D8"] == 1
    # Absent buckets default to 0 for aggregator stability.
    assert counts.get("D3", 0) == 0
