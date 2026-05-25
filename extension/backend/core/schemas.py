"""
Domain Model Schemas

Pydantic models defining the structure of DDD domain models.
Used for validation and serialization of domain model data.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# HELPER MODELS (Used by architect.py pipeline)
# =============================================================================

class RelevanceCheck(BaseModel):
    """Result of checking if text contains domain logic."""
    is_relevant: bool = Field(description="True if the text contains domain logic")
    summary: str = Field(description="Summary of domain concepts found")


class BoundedContextList(BaseModel):
    """List of identified bounded contexts."""
    contexts: List[str] = Field(description="List of Bounded Context names")


class InferenceSource(BaseModel):
    """Traceable source record for a domain inference."""
    file: str = Field(description="Source file path")
    line: int = Field(default=1, ge=1, description="1-based line number")
    rule: str = Field(description="Inference rule identifier")
    evidence: str = Field(description="Short evidence snippet")


# =============================================================================
# CORE DOMAIN BUILDING BLOCKS
# =============================================================================

class Entity(BaseModel):
    """Domain entity definition."""
    name: str = Field(description="Name of the domain entity (e.g., Customer)")
    description: str = Field(description="Brief description of the entity's role")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM-emitted confidence in this inference (0.0-1.0). Required."
    )
    justification: str = Field(
        description="LLM-emitted reason for this entity (e.g. supporting sentence count, role)."
    )
    evidence_sentence_indices: List[int] = Field(
        min_length=1,
        description="Scout sentence indices that ground this entity. Required from Phase D1 onward."
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list populated by AST enrichment (file/line/rule)."
    )
    synonyms_to_avoid: Optional[List[str]] = Field(
        default=None,
        description="Terms forbidden for this entity (e.g., Client, User)."
    )


class ValueObject(BaseModel):
    """Value object definition."""
    name: str = Field(description="Name of the value object")
    attributes: List[str] = Field(description="List of attributes")
    description: Optional[str] = Field(description="Description of purpose")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for this inference (0.0-1.0)"
    )
    sources: List[InferenceSource] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)"
    )
    # WP-CORE-27 (D9): AST-derived signal — True iff the class backing
    # this VO has mutation methods or non-frozen state. Mismatch with
    # the LLM's value-object claim triggers a D9 ERROR. Default False
    # preserves backward compatibility for callers that do not yet
    # populate this field; AST cross-reference wiring is a follow-up
    # WP-CORE-27a.
    is_mutable_in_code: bool = Field(
        default=False,
        description=(
            "True iff the underlying class has mutation methods; "
            "claimed-VO mismatch (LLM-claimed VO + mutable code) triggers D9."
        ),
    )


class Service(BaseModel):
    """Domain service candidate definition."""
    name: str = Field(description="Name of the domain service")
    description: str = Field(description="Service responsibility summary")
    # WP-CORE-33 (V9): optional service-kind discriminator. None on legacy
    # payloads. AST-inferred via dependency shape:
    #   * "domain"         — no injected deps, pure logic.
    #   * "application"    — at least one repository dep (orchestrates
    #                        a use case across persistence boundaries).
    #   * "infrastructure" — deps are exclusively external clients /
    #                        gateways / adapters (no repo) — service is
    #                        a thin wrapper around an external system.
    kind: Optional[Literal["domain", "application", "infrastructure"]] = Field(
        default=None,
        description=(
            "Service kind discriminator. None preserves backward compatibility "
            "for payloads written before WP-CORE-33."
        ),
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for this inference (0.0-1.0)"
    )
    sources: List[InferenceSource] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)"
    )


class Aggregate(BaseModel):
    """Aggregate root candidate definition."""
    name: str = Field(description="Name of the aggregate root")
    description: str = Field(description="Aggregate consistency boundary")
    members: List[str] = Field(
        description="Entity names that live inside this aggregate. Required."
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for this inference (0.0-1.0)"
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)"
    )
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that ground this aggregate."
    )


class DomainEvent(BaseModel):
    """Domain event definition."""
    name: str = Field(description="Name of the event (e.g., OrderPlaced)")
    description: Optional[str] = Field(description="When does this event happen")


class Repository(BaseModel):
    """Repository candidate. WP-CORE-22: previously masked by the
    INFRASTRUCTURE_SUFFIXES penalty in the AST classifier; now first-class.

    aggregate_root is derived by stripping the Repository/Repo suffix from
    the class name. If the AST classifier cannot derive a meaningful
    aggregate root, the field falls back to the class name itself.
    """
    name: str = Field(description="Name of the repository class")
    aggregate_root: str = Field(
        description="Aggregate root that this repository serves"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of the repository's responsibility",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="AST-detection confidence in this candidate",
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)",
    )
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that ground this repository (if any)",
    )


class Factory(BaseModel):
    """Factory candidate. WP-CORE-22: previously masked by the
    INFRASTRUCTURE_SUFFIXES penalty in the AST classifier; now first-class.

    produces is derived by stripping the Factory/Builder suffix from the
    class name. Used by downstream validators to enforce factory-bypass
    detection.
    """
    name: str = Field(description="Name of the factory class")
    produces: str = Field(
        description="Domain type this factory creates (derived from name)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief description of what this factory creates",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="AST-detection confidence in this candidate",
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)",
    )
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that ground this factory (if any)",
    )


class AntiCorruptionLayer(BaseModel):
    """Anti-Corruption Layer candidate. WP-CORE-33 (V7): classes whose
    purpose is to translate between the bounded context's domain model
    and an external system's representation. Detected by suffix
    (Translator / Adapter / Mapper / Gateway / ACL / AntiCorruption) and
    translate-shaped methods (translate_*, convert_*, to_domain,
    from_external, adapt). Optional `description` and traceable sources
    follow the same conventions as Repository / Factory.
    """
    name: str = Field(description="Name of the ACL class")
    description: Optional[str] = Field(
        default=None,
        description="Brief description of what this ACL translates",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="AST-detection confidence in this candidate",
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)",
    )
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that ground this ACL (if any)",
    )


class Specification(BaseModel):
    """Specification candidate. WP-CORE-33 (V8): predicate-as-object
    classes that encapsulate a domain rule. Detected by suffix
    (Specification / Spec / Rule / Predicate) and the canonical
    `is_satisfied_by(...)` method, optionally with combinator helpers
    (`and_`, `or_`, `not_`).
    """
    name: str = Field(description="Name of the specification class")
    description: Optional[str] = Field(
        default=None,
        description="Brief description of the rule this specification encodes",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="AST-detection confidence in this candidate",
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)",
    )
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that ground this spec (if any)",
    )


class UbiquitousLanguage(BaseModel):
    """Collection of domain terminology for a bounded context."""
    entities: List[Entity] = Field(description="List of entities in this context")
    value_objects: Optional[List[ValueObject]] = Field(
        description="Value objects in this context"
    )
    services: Optional[List[Service]] = Field(
        default=None,
        description="Domain services in this context"
    )
    aggregates: Optional[List[Aggregate]] = Field(
        default=None,
        description="Aggregate roots in this context"
    )
    domain_events: Optional[List[str]] = Field(description="List of domain events")
    # WP-CORE-22: Repository + Factory first-class detection. Optional
    # with default None preserves backward compatibility for model.json
    # documents that predate this WP.
    repositories: Optional[List[Repository]] = Field(
        default=None,
        description="Repository candidates discovered for this context"
    )
    factories: Optional[List[Factory]] = Field(
        default=None,
        description="Factory candidates discovered for this context"
    )
    # WP-CORE-33: V7 Anti-Corruption Layer + V8 Specification first-class
    # buckets. Optional with default None preserves backward compat for
    # model.json documents that predate this WP.
    anti_corruption_layers: Optional[List[AntiCorruptionLayer]] = Field(
        default=None,
        description="Anti-corruption layer candidates discovered for this context",
    )
    specifications: Optional[List[Specification]] = Field(
        default=None,
        description="Specification-pattern candidates discovered for this context",
    )


# =============================================================================
# CONTEXT AND RULES
# =============================================================================

class BoundedContext(BaseModel):
    """Definition of a bounded context."""
    context_name: str = Field(description="Name of the bounded context")
    description: str = Field(
        default="",
        description=(
            "What this context is responsible for. Empty post-merge is "
            "acceptable; downstream enrich step LLM-populates. WP-CORE-14 "
            "removed the synthetic f'{name} context' placeholder that "
            "previously masked intermediate-vs-final-description mismatch."
        ),
    )
    allowed_dependencies: Optional[List[str]] = Field(
        default=None,
        description="List of other contexts this context can depend on"
    )
    supporting_sentence_ids: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that justify identifying this context."
    )
    business_rules: Optional[List[str]] = Field(
        default=None,
        description="Context-specific business rules surfaced by Specialist."
    )
    ubiquitous_language: "UbiquitousLanguage" = Field(
        description="The language and models specific to this context"
    )


class GlobalRules(BaseModel):
    """Project-wide architectural rules."""
    naming_convention: Optional[str] = Field(
        default="PascalCase",
        description="Preferred naming convention"
    )
    banned_global_terms: Optional[List[str]] = Field(
        default_factory=list,
        description="Terms banned across the entire project"
    )


# =============================================================================
# MAIN DOMAIN MODEL
# =============================================================================

class ProjectMetadata(BaseModel):
    """Metadata about the domain model generation."""
    version: str = Field(description="Project version (e.g., 1.0.0)")
    generated_at: str = Field(description="Generation timestamp")
    description: Optional[str] = Field(
        default="Domain model generated from requirements",
        description="High level project description"
    )


class CritiqueFinding(BaseModel):
    """One DDD design-quality finding emitted by the Critic (persisted form)."""
    finding_type: Literal[
        "CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "BOUNDARY_SMELL",
        "ANEMIC_ENTITY", "ANEMIC_MODEL", "MISSING_AGGREGATE",
        "MISPLACED_ENTITY", "NAMING_SMELL", "LOW_CONFIDENCE", "OTHER",
    ]
    priority: Literal["high", "medium", "low"]
    target_ref: str = Field(description="e.g. 'context:Ordering' | 'entity:Ordering.Order'")
    rationale: str
    proposed_revision: str
    evidence_sentence_indices: List[int] = Field(default_factory=list)


class CriticLoopTrace(BaseModel):
    """Per-document trace of the critique loop."""
    cycles_used: int
    best_cycle: int
    outcome: Literal["converged", "exhausted", "flapped", "failed"]
    score_per_cycle: List[float] = Field(default_factory=list)
    findings_count_per_cycle: List[int] = Field(default_factory=list)


class CriticReport(BaseModel):
    """Best cycle's critique + loop trace, attached to the DomainModel."""
    model_id: str
    findings: List[CritiqueFinding] = Field(default_factory=list)
    score: float = 0.0
    malformed_findings: int = 0
    loop: CriticLoopTrace
    error: Optional[str] = None


class DomainModel(BaseModel):
    """Complete domain model for a project."""
    project_name: str = Field(description="Name of the project")
    project_metadata: ProjectMetadata = Field(description="Generation metadata")
    bounded_contexts: List[BoundedContext] = Field(
        description="List of all identified Bounded Contexts. Must be non-empty."
    )
    global_rules: Optional[GlobalRules] = Field(
        description="Project-wide architectural rules"
    )
    critic_report: Optional["CriticReport"] = Field(
        default=None,
        description="Holistic Critic loop output (best cycle). None when the loop did not run.",
    )

    @field_validator("bounded_contexts")
    @classmethod
    def _non_empty(cls, v: List[BoundedContext]) -> List[BoundedContext]:
        if not v:
            raise ValueError(
                "bounded_contexts must be non-empty; an empty DomainModel "
                "indicates upstream pipeline failure and must raise instead."
            )
        return v
