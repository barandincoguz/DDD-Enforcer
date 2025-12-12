from typing import List, Optional

from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 0. YARDIMCI MODELLER (LLM Pipeline Ara Adımları İçin)
# -----------------------------------------------------------------------------
# Bu sınıflar architect.py tarafından import ediliyor


class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(description="True if the text contains domain logic.")
    summary: str = Field(
        description="Summary of domain concepts found, or empty string."
    )


class BoundedContextList(BaseModel):
    contexts: List[str] = Field(
        description="List of Bounded Context names identified in the document."
    )


# -----------------------------------------------------------------------------
# 1. TEMEL YAPI TAŞLARI (ENTITY, VALUE OBJECT VS.)
# -----------------------------------------------------------------------------


class Entity(BaseModel):
    name: str = Field(description="Name of the domain entity (e.g., Customer).")
    description: str = Field(description="Brief description of the entity's role.")
    synonyms_to_avoid: Optional[List[str]] = Field(
        default=None,
        description="List of terms forbidden for this entity (e.g., Client, User).",
    )


class ValueObject(BaseModel):
    name: str = Field(description="Name of the value object")
    attributes: List[str] = Field(
        description="List of attributes defining this value object"
    )
    description: Optional[str] = Field(description="Description of purpose")


class DomainEvent(BaseModel):
    name: str = Field(description="Name of the event (e.g., OrderPlaced)")
    description: Optional[str] = Field(description="When does this event happen?")


class UbiquitousLanguage(BaseModel):
    entities: List[Entity] = Field(description="List of entities in this context")
    value_objects: Optional[List[ValueObject]] = Field(
        description="Value objects in this context"
    )
    domain_events: Optional[List[str]] = Field(description="List of domain events")


# -----------------------------------------------------------------------------
# 2. CONTEXT VE KURALLAR
# -----------------------------------------------------------------------------


class BoundedContext(BaseModel):
    context_name: str = Field(
        description="Name of the bounded context (e.g., SalesContext)."
    )
    description: str = Field(description="What this context is responsible for.")
    allowed_dependencies: Optional[List[str]] = Field(
        default=None, description="List of other contexts this context interacts with."
    )
    ubiquitous_language: UbiquitousLanguage = Field(
        description="The language and models specific to this context"
    )


class GlobalRules(BaseModel):
    naming_convention: Optional[str] = Field(
        default="PascalCase",
        description="Preferred naming convention (e.g., camelCase)",
    )
    banned_global_terms: Optional[List[str]] = Field(
        default_factory=list, description="Terms banned across the entire project"
    )


# -----------------------------------------------------------------------------
# 3. METADATA VE ANA MODEL
# -----------------------------------------------------------------------------


class ProjectMetadata(BaseModel):
    version: str = Field(description="Project version (e.g., 1.0.0)")
    generated_at: str = Field(description="Generation timestamp")
    description: Optional[str] = Field(
        default="Domain model generated from requirements",
        description="High level project description",
    )


class DomainModel(BaseModel):
    project_name: str = Field(description="Name of the project")
    project_metadata: ProjectMetadata = Field(
        description="Metadata about the generation"
    )

    # Context listesi
    bounded_contexts: List[BoundedContext] = Field(
        description="List of all identified Bounded Contexts"
    )

    # Global kurallar
    global_rules: Optional[GlobalRules] = Field(
        description="Project-wide architectural rules"
    )
