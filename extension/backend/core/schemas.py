"""
Domain model and generation pipeline schemas.

These models cover:
- final domain model artifacts used by validation
- deterministic SRS parsing outputs
- typed intermediate outputs for the generation pipeline
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# SOURCE AND EVIDENCE MODELS
# =============================================================================


class InferenceSource(BaseModel):
    """Traceable source record for a domain inference."""

    file: str = Field(description="Source file path")
    line: int = Field(default=1, ge=1, description="1-based line number")
    rule: str = Field(description="Inference rule identifier")
    evidence: str = Field(description="Short evidence snippet")
    document: Optional[str] = Field(default=None, description="Document name when available")
    section: Optional[str] = Field(default=None, description="Section heading when available")
    requirement_id: Optional[str] = Field(default=None, description="Parsed requirement identifier")


class EvidenceSpan(BaseModel):
    """Canonical evidence entry extracted from an SRS document."""

    evidence_id: str = Field(description="Stable evidence identifier")
    document: str = Field(description="Source document name")
    section: str = Field(description="Section heading")
    excerpt: str = Field(description="Short excerpt from the source")
    line: int = Field(default=1, ge=1, description="Approximate 1-based source line")
    requirement_id: Optional[str] = Field(default=None, description="Requirement identifier when applicable")


# =============================================================================
# DETERMINISTIC PARSER OUTPUTS
# =============================================================================


class ParsedSection(BaseModel):
    """Structured section from a source SRS document."""

    section_id: str = Field(description="Stable parsed section identifier")
    heading: str = Field(description="Human-readable section heading")
    category: str = Field(description="High-level section category")
    content: str = Field(description="Section content")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence ids belonging to this section")


class RequirementRecord(BaseModel):
    """Deterministically parsed requirement or structured fact."""

    requirement_id: str = Field(description="Stable requirement identifier")
    category: str = Field(description="Requirement category")
    title: str = Field(description="Requirement title or label")
    description: str = Field(description="Requirement description")
    actor: Optional[str] = Field(default=None, description="Actor associated with the requirement")
    section: str = Field(description="Source section heading")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence ids supporting this requirement")


class ParsedSRSDocument(BaseModel):
    """Structured parsed representation of an SRS source document."""

    file_path: str = Field(description="Absolute or relative source file path")
    document_name: str = Field(description="Source document file name")
    clean_text: str = Field(description="Cleaned text version of the source")
    sections: List[ParsedSection] = Field(default_factory=list, description="Parsed sections")
    requirements: List[RequirementRecord] = Field(default_factory=list, description="Parsed requirement records")
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list, description="All extracted evidence spans")


# =============================================================================
# INTERMEDIATE GENERATION PIPELINE OUTPUTS
# =============================================================================


class RequirementSummary(BaseModel):
    """Normalized requirement summary produced by Scout."""

    requirement_id: str = Field(description="Requirement identifier")
    title: str = Field(description="Normalized requirement title")
    category: str = Field(description="Requirement category")
    description: str = Field(description="Normalized requirement description")
    actor: Optional[str] = Field(default=None, description="Actor tied to the requirement")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class ActorCandidate(BaseModel):
    """Actor candidate extracted from the SRS."""

    name: str = Field(description="Actor name")
    description: str = Field(description="Actor responsibility summary")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class EntityCandidate(BaseModel):
    """Entity candidate extracted from the SRS."""

    name: str = Field(description="Entity name")
    description: str = Field(description="Entity summary")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class ValueObjectCandidate(BaseModel):
    """Value object candidate extracted from the SRS."""

    name: str = Field(description="Value object name")
    description: str = Field(description="Value object summary")
    attributes: List[str] = Field(default_factory=list, description="Attributes inferred from the SRS")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class CapabilityCandidate(BaseModel):
    """Capability candidate extracted from the SRS."""

    name: str = Field(description="Capability name")
    description: str = Field(description="Capability summary")
    actor: Optional[str] = Field(default=None, description="Primary actor for the capability")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class ConstraintCandidate(BaseModel):
    """Constraint or rule candidate extracted from the SRS."""

    text: str = Field(description="Constraint text")
    category: str = Field(description="Constraint category")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class TableCandidate(BaseModel):
    """Database or information storage candidate extracted from the SRS."""

    name: str = Field(description="Table or store name")
    description: str = Field(description="Description of what it stores")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class ScoutExtraction(BaseModel):
    """Typed Scout output for requirement normalization and evidence collection."""

    requirements: List[RequirementSummary] = Field(default_factory=list)
    actors: List[ActorCandidate] = Field(default_factory=list)
    entities: List[EntityCandidate] = Field(default_factory=list)
    constraints: List[ConstraintCandidate] = Field(default_factory=list)
    tables: List[TableCandidate] = Field(default_factory=list)
    capabilities: List[CapabilityCandidate] = Field(default_factory=list)
    evidence_spans: List[EvidenceSpan] = Field(default_factory=list)


class ContextProposal(BaseModel):
    """Architect stage context proposal."""

    context_name: str = Field(description="Proposed bounded context name")
    description: str = Field(description="Context summary")
    ownership_rationale: str = Field(description="Why this context owns its responsibilities")
    included_capabilities: List[str] = Field(default_factory=list, description="Capabilities owned by this context")
    excluded_capabilities: List[str] = Field(default_factory=list, description="Capabilities explicitly not owned")
    primary_entities: List[str] = Field(default_factory=list, description="Entities primarily owned by this context")
    allowed_dependencies: List[str] = Field(default_factory=list, description="Potential allowed dependencies")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")


class ContextMap(BaseModel):
    """Architect stage output."""

    contexts: List[ContextProposal] = Field(default_factory=list)


class ContextAnalysis(BaseModel):
    """Specialist stage output for a single context."""

    context: str = Field(description="Bounded context name")
    description: str = Field(description="Context summary")
    actors: List[ActorCandidate] = Field(default_factory=list)
    capabilities: List[CapabilityCandidate] = Field(default_factory=list)
    aggregate_roots: List[EntityCandidate] = Field(default_factory=list)
    entities: List[EntityCandidate] = Field(default_factory=list)
    value_objects: List[ValueObjectCandidate] = Field(default_factory=list)
    business_rules: List[ConstraintCandidate] = Field(default_factory=list)
    domain_events: List[EntityCandidate] = Field(default_factory=list)
    domain_services: List[EntityCandidate] = Field(default_factory=list)
    allowed_dependencies: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class ContextAnalysisBatch(BaseModel):
    """Batch Specialist output."""

    analyses: List[ContextAnalysis] = Field(default_factory=list)


class VerificationReport(BaseModel):
    """Verifier output and deterministic coverage report."""

    passed: bool = Field(description="True when the final model passes verification")
    missing_requirement_ids: List[str] = Field(default_factory=list)
    uncovered_capabilities: List[str] = Field(default_factory=list)
    uncovered_actors: List[str] = Field(default_factory=list)
    uncovered_entities: List[str] = Field(default_factory=list)
    evidence_less_items: List[str] = Field(default_factory=list)
    duplicate_entities: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# =============================================================================
# FINAL DOMAIN MODEL BUILDING BLOCKS
# =============================================================================


class DomainActor(BaseModel):
    """Actor represented in the final model."""

    name: str = Field(description="Actor name")
    description: str = Field(description="Actor responsibility summary")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class BusinessRule(BaseModel):
    """Business rule or invariant."""

    text: str = Field(description="Rule text")
    category: Optional[str] = Field(default=None, description="Rule category")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class Capability(BaseModel):
    """Capability supported by a bounded context."""

    name: str = Field(description="Capability name")
    description: str = Field(description="Capability summary")
    actor: Optional[str] = Field(default=None, description="Primary actor")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class ExternalReference(BaseModel):
    """Cross-context or external dependency reference."""

    name: str = Field(description="Reference name")
    relationship: str = Field(description="Relationship type")
    target_context: Optional[str] = Field(default=None, description="Referenced context")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class Entity(BaseModel):
    """Domain entity definition."""

    name: str = Field(description="Name of the domain entity")
    description: str = Field(description="Brief description of the entity's role")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)
    synonyms_to_avoid: Optional[List[str]] = Field(
        default=None,
        description="Terms forbidden for this entity"
    )


class ValueObject(BaseModel):
    """Value object definition."""

    name: str = Field(description="Name of the value object")
    attributes: List[str] = Field(default_factory=list, description="List of attributes")
    description: Optional[str] = Field(default=None, description="Description of purpose")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class Service(BaseModel):
    """Domain service candidate definition."""

    name: str = Field(description="Name of the domain service")
    description: str = Field(description="Service responsibility summary")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class Aggregate(BaseModel):
    """Aggregate root candidate definition."""

    name: str = Field(description="Name of the aggregate root")
    description: str = Field(description="Aggregate consistency boundary")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class DomainEvent(BaseModel):
    """Domain event definition."""

    name: str = Field(description="Name of the event")
    description: Optional[str] = Field(default=None, description="When does this event happen")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_ids: List[str] = Field(default_factory=list)
    sources: List[InferenceSource] = Field(default_factory=list)


class UbiquitousLanguage(BaseModel):
    """Collection of domain terminology for a bounded context."""

    entities: List[Entity] = Field(default_factory=list, description="Entities in this context")
    value_objects: List[ValueObject] = Field(default_factory=list, description="Value objects in this context")
    services: List[Service] = Field(default_factory=list, description="Domain services in this context")
    aggregates: List[Aggregate] = Field(default_factory=list, description="Aggregate roots in this context")
    domain_events: List[DomainEvent] = Field(default_factory=list, description="Domain events in this context")


class BoundedContext(BaseModel):
    """Definition of a bounded context."""

    context_name: str = Field(description="Name of the bounded context")
    description: str = Field(description="What this context is responsible for")
    allowed_dependencies: List[str] = Field(default_factory=list, description="Contexts this context can depend on")
    actors: List[DomainActor] = Field(default_factory=list, description="Actors relevant to this context")
    capabilities: List[Capability] = Field(default_factory=list, description="Capabilities owned by this context")
    ubiquitous_language: UbiquitousLanguage = Field(description="The language and models specific to this context")
    business_rules: List[BusinessRule] = Field(default_factory=list, description="Context-specific rules")
    external_references: List[ExternalReference] = Field(default_factory=list, description="Cross-context references")
    evidence_ids: List[str] = Field(default_factory=list, description="Supporting evidence ids")
    evidence: List[InferenceSource] = Field(default_factory=list, description="Supporting evidence records")


class GlobalRules(BaseModel):
    """Project-wide architectural rules."""

    naming_convention: Optional[str] = Field(default="PascalCase", description="Preferred naming convention")
    banned_global_terms: List[str] = Field(default_factory=list, description="Terms banned across the entire project")
    cross_cutting_constraints: List[str] = Field(default_factory=list, description="Cross-context rules")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions extracted from the SRS")


class ProjectMetadata(BaseModel):
    """Metadata about the domain model generation."""

    version: str = Field(default="1.0.0", description="Project version")
    generated_at: str = Field(description="Generation timestamp")
    description: Optional[str] = Field(default="Domain model generated from requirements", description="High level project description")


class DomainModel(BaseModel):
    """Complete domain model for a project."""

    schema_version: str = Field(default="2.0.0", description="Domain model schema version")
    project_name: str = Field(description="Name of the project")
    project_metadata: ProjectMetadata = Field(description="Generation metadata")
    bounded_contexts: List[BoundedContext] = Field(default_factory=list, description="All identified bounded contexts")
    global_rules: Optional[GlobalRules] = Field(default=None, description="Project-wide architectural rules")
