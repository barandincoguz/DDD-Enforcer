"""
Domain Model Schemas

Pydantic models defining the structure of DDD domain models.
Used for validation and serialization of domain model data.
"""

from typing import List, Optional

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


class Service(BaseModel):
    """Domain service candidate definition."""
    name: str = Field(description="Name of the domain service")
    description: str = Field(description="Service responsibility summary")
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

    @field_validator("bounded_contexts")
    @classmethod
    def _non_empty(cls, v: List[BoundedContext]) -> List[BoundedContext]:
        if not v:
            raise ValueError(
                "bounded_contexts must be non-empty; an empty DomainModel "
                "indicates upstream pipeline failure and must raise instead."
            )
        return v
