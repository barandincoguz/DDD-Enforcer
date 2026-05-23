"""WP-CORE-8 — `_build_pipeline_error_response` helper tests.

T-HELPER-1..5: helper converts PipelineError subclasses into JSON-serializable
response dicts preserving typed taxonomy (Codex W-1 + W-4 dispositions).

Imports of `_build_pipeline_error_response` live INSIDE test bodies per the
WP-CORE-7 pattern — RED commit succeeds collection.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# T-HELPER-1 — ArchitectGroundingError
# =============================================================================


def test_build_pipeline_error_response_handles_architect_grounding_error():
    """T-HELPER-1: helper preserves srs_path + issues + residual_issues +
    cycles_attempted from ArchitectGroundingError."""
    from main import _build_pipeline_error_response  # noqa: E402
    from core.orchestration.errors import ArchitectGroundingError

    arch_issue = {"check_id": "D1", "target": "architect:contexts[X]", "message": "no IDs"}
    spec_issue = {"check_id": "D2", "target": "specialist:X.entities[0]", "message": "no evidence"}

    exc = ArchitectGroundingError(
        srs_path="inputs/SRS.docx",
        issues=[arch_issue],
        residual_issues=[spec_issue],
        cycles_attempted=1,
    )

    payload = _build_pipeline_error_response(exc)
    assert payload["success"] is False
    assert payload["error_type"] == "ArchitectGroundingError"
    assert "SRS.docx" in payload["error"]
    assert payload["srs_path"] == "inputs/SRS.docx"
    assert payload["cycles_attempted"] == 1
    assert isinstance(payload["issues"], list)
    assert len(payload["issues"]) == 1
    assert isinstance(payload["residual_issues"], list)
    assert len(payload["residual_issues"]) == 1


# =============================================================================
# T-HELPER-2 — SynthesizerEmptyModelError
# =============================================================================


def test_build_pipeline_error_response_handles_synthesizer_empty_model_error():
    """T-HELPER-2: helper preserves srs_path + input_summary."""
    from main import _build_pipeline_error_response
    from core.orchestration.errors import SynthesizerEmptyModelError

    exc = SynthesizerEmptyModelError(
        input_summary="0 SpecialistAnalysis from upstream pipeline",
        srs_path="inputs/SRS.docx",
    )

    payload = _build_pipeline_error_response(exc)
    assert payload["error_type"] == "SynthesizerEmptyModelError"
    assert payload["srs_path"] == "inputs/SRS.docx"
    assert payload["input_summary"] == "0 SpecialistAnalysis from upstream pipeline"


# =============================================================================
# T-HELPER-3 — IntermediateSaveError
# =============================================================================


def test_build_pipeline_error_response_handles_intermediate_save_error():
    """T-HELPER-3: helper preserves stage + filepath + srs_path."""
    from main import _build_pipeline_error_response
    from core.orchestration.errors import IntermediateSaveError

    exc = IntermediateSaveError(
        stage="2_architect",
        filepath="/tmp/intermediate/20260523_arch.json",
        cause=OSError("disk full"),
        srs_path="inputs/SRS.docx",
    )

    payload = _build_pipeline_error_response(exc)
    assert payload["error_type"] == "IntermediateSaveError"
    assert payload["stage"] == "2_architect"
    assert payload["filepath"] == "/tmp/intermediate/20260523_arch.json"
    assert payload["srs_path"] == "inputs/SRS.docx"


# =============================================================================
# T-HELPER-4 — issue serialization round-trip (Codex W-4 strengthened)
# =============================================================================


def test_build_pipeline_error_response_round_trip_preserves_issue_fields():
    """T-HELPER-4 (Codex W-4): json.loads(json.dumps(payload)) preserves
    expected keys + values for both legacy and contract VerifierIssue.
    Specifically: severity normalizes to string (not enum repr),
    message preserved, no `{"repr": ...}` fallback path used."""
    from main import _build_pipeline_error_response
    from core.orchestration.errors import ArchitectGroundingError
    from core.verifier.types import VerifierIssue as LegacyVerifierIssue, IssueSeverity
    from core.pipeline_contracts import VerifierIssue as ContractVerifierIssue

    legacy = LegacyVerifierIssue(
        stage="architect",
        location="architect:contexts[X].supporting_sentence_ids",
        issue_type="ungrounded_context",
        severity=IssueSeverity.ERROR,
        message="legacy-message-payload",
    )
    contract = ContractVerifierIssue(
        severity="ERROR",
        check_id="D1",
        target="architect:contexts[Y].supporting_sentence_ids",
        message="contract-message-payload",
    )

    exc = ArchitectGroundingError(
        srs_path="srs.docx",
        issues=[legacy, contract],
        residual_issues=[],
        cycles_attempted=1,
    )

    payload = _build_pipeline_error_response(exc)
    # Must be JSON-roundtrip safe (no exception)
    roundtripped = json.loads(json.dumps(payload))

    assert isinstance(roundtripped["issues"], list)
    assert len(roundtripped["issues"]) == 2

    for issue_dict in roundtripped["issues"]:
        assert isinstance(issue_dict, dict)
        # No fallback {"repr": "..."} — that means duck-typing failed.
        assert "repr" not in issue_dict, (
            f"Issue serialized via repr-fallback (helper duck-typing failed): {issue_dict}"
        )
        # message preserved verbatim
        assert "message" in issue_dict
        assert issue_dict["message"] in (
            "legacy-message-payload", "contract-message-payload",
        )
        # severity is a string (not enum repr like "IssueSeverity.ERROR")
        sev = issue_dict.get("severity")
        if sev is not None:
            assert isinstance(sev, str), f"severity should be string, got {type(sev)}: {sev!r}"
            assert "IssueSeverity" not in sev, f"severity contains enum repr: {sev!r}"


# =============================================================================
# T-HELPER-5 — SpecialistShapeError (Codex W-1 NEW)
# =============================================================================


def test_build_pipeline_error_response_handles_specialist_shape_error():
    """T-HELPER-5 (Codex W-1): SpecialistShapeError carries validation_errors
    + raw_excerpt; helper must preserve both in JSON-safe form."""
    from main import _build_pipeline_error_response
    from core.orchestration.errors import SpecialistShapeError

    exc = SpecialistShapeError(
        context_name="OrderMgmt",
        errors=[{"loc": ["entities", 0, "name"], "msg": "missing", "type": "missing"}],
        raw_excerpt='{"context": "OrderMgmt", "entities": []}',
    )

    payload = _build_pipeline_error_response(exc)
    assert payload["error_type"] == "SpecialistShapeError"
    assert payload["context_name"] == "OrderMgmt"
    assert payload["raw_excerpt"] == '{"context": "OrderMgmt", "entities": []}'
    assert isinstance(payload["validation_errors"], list)
    assert len(payload["validation_errors"]) == 1
    # validation_errors element is dict-shaped after serialization (not a raw pydantic ErrorDetails)
    err0 = payload["validation_errors"][0]
    assert isinstance(err0, dict)
    # Round-trip safe
    json.loads(json.dumps(payload))
