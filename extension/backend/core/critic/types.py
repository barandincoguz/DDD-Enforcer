"""LLM-facing critic schema + loop memory.

ProposedFinding/CriticResponse are what the LLM emits (no provenance).
run_critic maps them to the persisted core.schemas.CritiqueFinding/CriticReport.
"""
from typing import List, Literal
from pydantic import BaseModel, Field


class ProposedFinding(BaseModel):
    finding_type: Literal[
        "CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "BOUNDARY_SMELL",
        "ANEMIC_ENTITY", "ANEMIC_MODEL", "MISSING_AGGREGATE",
        "MISPLACED_ENTITY", "NAMING_SMELL", "LOW_CONFIDENCE", "OTHER",
    ]
    priority: Literal["high", "medium", "low"]
    target_ref: str
    rationale: str
    proposed_revision: str
    evidence_sentence_indices: List[int] = Field(default_factory=list)


class CriticResponse(BaseModel):
    """Schema-enforced critic output. `analysis` is the CoT scratchpad the
    model fills before listing findings (kept short, not persisted verbatim)."""
    analysis: str = Field(default="", description="Step-by-step DDD reasoning before findings.")
    findings: List[ProposedFinding] = Field(default_factory=list)


class CritiqueCycleMemory(BaseModel):
    """Reflexion memory for one prior cycle, fed back into the next critique."""
    cycle: int
    findings_summary: List[str] = Field(default_factory=list)
    diff_summary: str = ""
