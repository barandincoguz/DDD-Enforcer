"""
Domain Model Schemas

Pydantic models defining the structure of DDD domain models.
Used for validation and serialization of domain model data.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


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


# =============================================================================
# CORE DOMAIN BUILDING BLOCKS
# =============================================================================

class Entity(BaseModel):
    """Domain entity definition."""
    name: str = Field(description="Name of the domain entity (e.g., Customer)")
    description: str = Field(description="Brief description of the entity's role")
    synonyms_to_avoid: Optional[List[str]] = Field(
        default=None,
        description="Terms forbidden for this entity (e.g., Client, User)"
    )


class ValueObject(BaseModel):
    """Value object definition."""
    name: str = Field(description="Name of the value object")
    attributes: List[str] = Field(description="List of attributes")
    description: Optional[str] = Field(description="Description of purpose")


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
    domain_events: Optional[List[str]] = Field(description="List of domain events")


# =============================================================================
# CONTEXT AND RULES
# =============================================================================

class BoundedContext(BaseModel):
    """Definition of a bounded context."""
    context_name: str = Field(description="Name of the bounded context")
    description: str = Field(description="What this context is responsible for")
    allowed_dependencies: Optional[List[str]] = Field(
        default=None,
        description="List of other contexts this context can depend on"
    )
    ubiquitous_language: UbiquitousLanguage = Field(
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
        description="List of all identified Bounded Contexts"
    )
    global_rules: Optional[GlobalRules] = Field(
        description="Project-wide architectural rules"
    )
