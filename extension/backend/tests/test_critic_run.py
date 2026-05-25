import pytest
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
from core.llm.base import LLMResponse, TokenUsage
from core.critic.types import CriticResponse, ProposedFinding
from core.critic.critic import run_critic
from core.critic.errors import CriticError
from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


@dataclass
class _StageCfg:
    model_id: str = "gemini-3.1-pro-preview"
    temperature: float = 0.05
    seed: Optional[int] = 42


class _FakeClient:
    """Returns a canned LLMResponse for structured_output."""
    def __init__(self, parsed: Optional[BaseModel], json_failed: bool = False):
        self._parsed = parsed
        self._json_failed = json_failed
        self.calls = 0

    def structured_output(self, messages, schema, model, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="{}", parsed=self._parsed,
            usage=TokenUsage(1, 1, 2), model_id=model, provider="fake",
            json_failed=self._json_failed,
            json_fail_reason="schema_mismatch" if self._json_failed else None,
        )


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name="Ordering",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="An order.", confidence=0.9,
                    justification="j", evidence_sentence_indices=[0],
                )],
                value_objects=None, domain_events=None,
            ),
        )],
        global_rules=None,
    )


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="A customer places an order.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=27),
    )


def test_run_critic_maps_valid_findings():
    resp = CriticResponse(analysis="a", findings=[ProposedFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Ordering.Order", rationale="no behavior",
        proposed_revision="add place()", evidence_sentence_indices=[0],
    )])
    report = run_critic(_model(), _scout(), [], client=_FakeClient(resp), stage_cfg=_StageCfg())
    assert report.model_id == "gemini-3.1-pro-preview"
    assert len(report.findings) == 1
    assert report.malformed_findings == 0


def test_run_critic_drops_unresolvable_target_ref():
    resp = CriticResponse(analysis="a", findings=[ProposedFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Nope.Ghost", rationale="x", proposed_revision="y",
    )])
    report = run_critic(_model(), _scout(), [], client=_FakeClient(resp), stage_cfg=_StageCfg())
    assert report.findings == []
    assert report.malformed_findings == 1


def test_run_critic_drops_out_of_range_evidence_but_keeps_finding():
    resp = CriticResponse(analysis="a", findings=[ProposedFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Ordering.Order", rationale="x", proposed_revision="y",
        evidence_sentence_indices=[0, 99],
    )])
    report = run_critic(_model(), _scout(), [], client=_FakeClient(resp), stage_cfg=_StageCfg())
    assert report.findings[0].evidence_sentence_indices == [0]


def test_run_critic_raises_on_json_failed():
    with pytest.raises(CriticError):
        run_critic(_model(), _scout(), [], client=_FakeClient(None, json_failed=True), stage_cfg=_StageCfg())
