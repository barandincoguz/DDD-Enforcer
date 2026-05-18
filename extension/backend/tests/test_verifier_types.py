"""Phase C1: VerifierIssue + IssueSeverity + VerifierResult interfaces."""

from core.verifier.types import VerifierIssue, IssueSeverity, VerifierResult


def test_issue_severity_has_error_and_warn():
    assert IssueSeverity.ERROR == "error"
    assert IssueSeverity.WARN == "warn"


def test_verifier_issue_construct():
    issue = VerifierIssue(
        stage="specialist",
        location="specialist:OrderMgmt.entities[2]",
        issue_type="missing_evidence",
        severity=IssueSeverity.ERROR,
        message="Entity 'Order' has no evidence_sentence_indices",
    )
    assert issue.stage == "specialist"
    assert issue.severity == IssueSeverity.ERROR


def test_verifier_result_ok_when_no_issues():
    result = VerifierResult(ok=True, issues=[])
    assert result.ok
    assert len(result.issues) == 0


def test_verifier_result_not_ok_with_error_issue():
    result = VerifierResult(
        ok=False,
        issues=[
            VerifierIssue(
                stage="architect",
                location="architect:contexts[0]",
                issue_type="ungrounded",
                severity=IssueSeverity.ERROR,
                message="Context name not in Scout output",
            )
        ],
    )
    assert not result.ok
    assert result.issues[0].issue_type == "ungrounded"
