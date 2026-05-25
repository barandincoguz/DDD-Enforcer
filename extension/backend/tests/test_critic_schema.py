"""Persisted critic schema: additive, backward-compatible."""
import pytest
from pydantic import ValidationError
from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
    CritiqueFinding, CriticReport, CriticLoopTrace,
)


def _minimal_model() -> DomainModel:
    return DomainModel(
        project_name="P",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name="Ctx",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="An order.", confidence=0.9,
                    justification="cited", evidence_sentence_indices=[0],
                )],
                value_objects=None, domain_events=None,
            ),
        )],
        global_rules=None,
    )


def test_domain_model_critic_report_defaults_none():
    assert _minimal_model().critic_report is None


def test_critique_finding_requires_known_type_and_priority():
    f = CritiqueFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Ctx.Order", rationale="no behavior",
        proposed_revision="add methods",
    )
    assert f.evidence_sentence_indices == []
    with pytest.raises(ValidationError):
        CritiqueFinding(
            finding_type="NONSENSE", priority="high",
            target_ref="x", rationale="y", proposed_revision="z",
        )


def test_critic_report_attaches_to_model():
    m = _minimal_model()
    m.critic_report = CriticReport(
        model_id="gemini-3.1-pro-preview",
        findings=[],
        loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"),
    )
    assert m.critic_report.loop.outcome == "converged"
    assert m.critic_report.score == 0.0


def test_old_model_json_without_critic_report_deserializes():
    payload = _minimal_model().model_dump()
    payload.pop("critic_report", None)
    restored = DomainModel.model_validate(payload)
    assert restored.critic_report is None
