"""Phase C6: 5-stage pipeline driver. Mock LLM, mock verifier, mock refiner."""

import pytest
from unittest.mock import MagicMock
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.verifier.types import VerifierResult, VerifierIssue, IssueSeverity
from core.orchestration.errors import (
    ArchitectExtractionError,
    SpecialistFailureError,
)


def _ok():
    return VerifierResult(ok=True, issues=[])


def _make_deps_happy_path():
    scout = MagicMock(return_value=[
        {"section_id": "1", "section_title": "Intro", "text": "..."}
    ])
    architect = MagicMock(return_value=[
        {"name": "OrderMgmt", "supporting_sentence_ids": [0]}
    ])
    specialist = MagicMock(return_value=[
        {
            "context_name": "OrderMgmt",
            "entities": [{
                "name": "Order",
                "description": "An order",
                "confidence": 0.9,
                "justification": "test",
                "evidence_sentence_indices": [0],
            }],
            "value_objects": [],
            "services": [],
            "aggregates": [{"name": "Order", "members": ["Order"]}],
            "domain_events": [],
        }
    ])
    synthesizer = MagicMock(return_value={
        "project_name": "Test",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [{
            "context_name": "OrderMgmt",
            "description": "Manages orders",
            "ubiquitous_language": {
                "entities": [{
                    "name": "Order",
                    "description": "An order",
                    "confidence": 0.9,
                    "justification": "test",
                    "evidence_sentence_indices": [0],
                }],
                "value_objects": [],
                "services": [],
                "aggregates": [{"name": "Order", "description": "agg", "members": ["Order"]}],
                "domain_events": [],
            },
            "supporting_sentence_ids": [0],
            "business_rules": None,
        }],
        "global_rules": None,
    })
    verifier = MagicMock(return_value=_ok())
    return PipelineDeps(
        scout=scout,
        architect=architect,
        specialist=specialist,
        synthesizer=synthesizer,
        verifier=verifier,
    )


def test_pipeline_happy_path_produces_domain_model():
    deps = _make_deps_happy_path()
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert model.project_name == "Test"
    assert len(model.bounded_contexts) == 1


def test_pipeline_propagates_architect_extraction_error():
    deps = _make_deps_happy_path()
    deps.architect.side_effect = ArchitectExtractionError(srs_path="x")
    with pytest.raises(ArchitectExtractionError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_propagates_specialist_failure():
    deps = _make_deps_happy_path()
    deps.specialist.side_effect = SpecialistFailureError(context_name="OrderMgmt")
    with pytest.raises(SpecialistFailureError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_invokes_refiner_when_verifier_finds_issues():
    deps = _make_deps_happy_path()
    deps.verifier.side_effect = [
        VerifierResult(ok=False, issues=[VerifierIssue(
            stage="specialist", location="specialist:x.entities[0]",
            issue_type="missing_evidence", severity=IssueSeverity.ERROR,
            message="missing"
        )]),
        VerifierResult(ok=True, issues=[]),  # second call after refine
    ]
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    # Refiner re-runs Specialist once
    assert deps.specialist.call_count == 2
    assert model is not None
