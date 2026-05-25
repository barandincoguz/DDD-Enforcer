"""LLM-facing context-mapper schema. ProposedRelationship is what the LLM emits;
mapper.run_context_mapper maps it to core.schemas.ContextRelationship."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ProposedRelationship(BaseModel):
    context_a: str
    context_b: str
    relationship_type: Literal[
        "PARTNERSHIP", "SHARED_KERNEL", "CUSTOMER_SUPPLIER", "CONFORMIST",
        "ANTI_CORRUPTION_LAYER", "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE",
        "SEPARATE_WAYS", "BIG_BALL_OF_MUD",
    ]
    upstream: Optional[str] = None
    rationale: str = ""
    evidence_sentence_indices: List[int] = Field(default_factory=list)


class ContextMapResponse(BaseModel):
    """Schema-enforced mapper output. `analysis` is the CoT scratchpad."""
    analysis: str = Field(default="", description="Step-by-step DDD reasoning.")
    relationships: List[ProposedRelationship] = Field(default_factory=list)
