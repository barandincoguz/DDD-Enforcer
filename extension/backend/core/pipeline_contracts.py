"""Typed contracts for stage boundaries in the domain-model pipeline.

Each stage produces and consumes a typed envelope. Boundary validation
is enforced via Pydantic .model_validate() at every transition. A
schema mismatch raises ValidationError that the stage retry loop
converts into targeted LLM feedback — not a stack-trace crash.

Content classes (Entity, ValueObject, etc.) live in core.schemas and
are reused unchanged. This module adds the stage-envelope wrappers
ONLY.
"""

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator

from core.schemas import (
    Entity,
    ValueObject,
    Service,
    Aggregate,
    DomainEvent,
)


# =============================================================================
# REFINER SIGNAL POLICY
# =============================================================================
# Pydantic by default ignores unknown fields. We need to ignore COSMETIC
# extras (LLMs sometimes emit "_metadata", "_reasoning") but DETECT
# semantic refiner signals like "_unresolved", "_needs_review",
# "_refiner_note" so the retry loop can act on them instead of swallowing.

_REFINER_SIGNAL_PREFIXES = ("_unresolved", "_needs_review", "_refiner_")


def _check_refiner_signals(values: Any) -> None:
    """Raise if any extra field name starts with a refiner-signal prefix."""
    if not isinstance(values, dict):
        return
    for key in values.keys():
        if not isinstance(key, str):
            continue
        if any(key.startswith(p) for p in _REFINER_SIGNAL_PREFIXES):
            raise ValueError(
                f"refiner signal {key!r} surfaced as an extra field; "
                f"this must be handled by the retry loop, not ignored"
            )


# =============================================================================
# SCOUT STAGE
# =============================================================================


class SectionedSentence(BaseModel):
    """A single Scout-extracted sentence with section provenance."""
    index: int = Field(ge=0)
    text: str
    section: Optional[str] = None


class ChunkMetadata(BaseModel):
    """Scout-pass diagnostic info."""
    chunk_count: int = Field(ge=0)
    total_chars: int = Field(ge=0)
    truncated_chunks: int = Field(default=0, ge=0)


class ScoutOutput(BaseModel):
    """Output of the Scout stage: numbered domain-relevant sentences
    with chunk-pass diagnostics."""
    sentences: List[SectionedSentence]
    chunk_metadata: ChunkMetadata

    @model_validator(mode="before")
    @classmethod
    def _detect_refiner_signals(cls, values: Any) -> Any:
        _check_refiner_signals(values)
        return values


# =============================================================================
# ARCHITECT STAGE
# =============================================================================


class ContextHypothesis(BaseModel):
    """Architect's per-context proposal."""
    context_name: str
    description: str = ""
    supporting_sentence_ids: List[int] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _detect_refiner_signals(cls, values: Any) -> Any:
        _check_refiner_signals(values)
        return values


class ArchitectOutput(BaseModel):
    """Output of the Architect stage: identified contexts +
    architect-flagged ambiguities (informational, not fail-fast)."""
    contexts: List[ContextHypothesis]
    open_questions: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _detect_refiner_signals(cls, values: Any) -> Any:
        _check_refiner_signals(values)
        return values


# =============================================================================
# SPECIALIST STAGE
# =============================================================================


class Ambiguity(BaseModel):
    """Specialist-flagged uncertainty about an emission."""
    target: str
    reason: str


class SpecialistAnalysis(BaseModel):
    """Per-context Specialist output. extract_per_context_details
    returns a List[SpecialistAnalysis] (one per Architect-identified
    context)."""
    context: ContextHypothesis
    entities: List[Entity] = Field(default_factory=list)
    value_objects: List[ValueObject] = Field(default_factory=list)
    services: List[Service] = Field(default_factory=list)
    aggregates: List[Aggregate] = Field(default_factory=list)
    domain_events: List[DomainEvent] = Field(default_factory=list)
    business_rules: List[str] = Field(default_factory=list)
    ambiguities: List[Ambiguity] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _detect_refiner_signals(cls, values: Any) -> Any:
        _check_refiner_signals(values)
        return values


# =============================================================================
# VERIFIER STAGE
# =============================================================================


# VerifierIssue and VerifierResult are constructed in code (verifier/
# checks_*.py), never deserialized from LLM JSON, so the refiner-signal
# detector is intentionally omitted.
class VerifierIssue(BaseModel):
    """One issue surfaced by a Verifier check.

    severity is Literal-constrained. check_id stays open `str` because
    new check identifiers are added as the Verifier grows; downstream
    aggregators must defensive-default on unknown ids rather than
    assume the set is closed.
    """
    severity: Literal["ERROR", "WARN"]
    check_id: str  # "D1" | "D2" | ... | "S1" | "D6" | "D7" | "D8"
    target: str
    message: str


class VerifierResult(BaseModel):
    """Verifier output: deterministic + semantic issues across stages.

    API-compatible with the legacy core.verifier.types.VerifierResult
    dataclass that the Refiner loop (core/refiner/loop.py) consumes:
    same `ok` field name + `error_count()` helper method.
    """
    ok: bool
    issues: List[VerifierIssue] = Field(default_factory=list)

    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")
