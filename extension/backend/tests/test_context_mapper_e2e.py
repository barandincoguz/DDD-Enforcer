"""E2E pipeline test: context-mapper wiring inside _generate_once.

Entry point: _generate_once (imported directly from core.orchestration.pipeline).
run_pipeline is thin glue on top and doesn't add logic between _generate_once
and _apply_context_map, so testing at _generate_once level is correct and avoids
the complexity of the full critic loop.

Two behaviours proven:
  1. context_mapper set  → model.context_map populated + allowed_dependencies derived.
  2. context_mapper=None → model.context_map is None.
"""

from datetime import datetime
from typing import List, Optional

from core.orchestration.pipeline import _generate_once, PipelineDeps
from core.pipeline_contracts import (
    ArchitectOutput,
    ChunkMetadata,
    ContextHypothesis,
    ScoutOutput,
    SectionedSentence,
    SpecialistAnalysis,
)
from core.schemas import (
    BoundedContext,
    ContextMap,
    ContextRelationship,
    DomainModel,
    GlobalRules,
    ProjectMetadata,
    UbiquitousLanguage,
)
from core.verifier.types import VerifierResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ul() -> UbiquitousLanguage:
    return UbiquitousLanguage(entities=[], value_objects=None, domain_events=None)


def _make_scout() -> ScoutOutput:
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="Ordering depends on Inventory.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=32),
    )


def _make_model() -> DomainModel:
    """2-context model returned by the fake synthesizer."""
    return DomainModel(
        project_name="TestProject",
        project_metadata=ProjectMetadata(
            version="1.0.0",
            generated_at=datetime.now().isoformat(),
        ),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", description="Ordering ctx", ubiquitous_language=_ul()),
            BoundedContext(context_name="Inventory", description="Inventory ctx", ubiquitous_language=_ul()),
        ],
        global_rules=None,
    )


def _context_mapper_fn(model: DomainModel, scout: ScoutOutput, feedback: Optional[list]) -> ContextMap:
    """Fake context-mapper: one CUSTOMER_SUPPLIER relationship, Inventory upstream."""
    return ContextMap(
        model_id="test-map",
        relationships=[
            ContextRelationship(
                context_a="Ordering",
                context_b="Inventory",
                relationship_type="CUSTOMER_SUPPLIER",
                upstream="Inventory",
                rationale="r",
            )
        ],
    )


def _base_deps(*, context_mapper=None) -> PipelineDeps:
    """Minimal fake PipelineDeps that passes through _generate_once cleanly."""

    def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
        return ArchitectOutput(
            contexts=[
                ContextHypothesis(context_name="Ordering", description="Ordering ctx"),
                ContextHypothesis(context_name="Inventory", description="Inventory ctx"),
            ]
        )

    def specialist_fn(arch: ArchitectOutput, scout: ScoutOutput) -> List[SpecialistAnalysis]:
        return [
            SpecialistAnalysis(context=ctx)
            for ctx in arch.contexts
        ]

    def synthesizer_fn(analyses: List[SpecialistAnalysis]) -> DomainModel:
        return _make_model()

    def verifier_fn(snap) -> VerifierResult:
        return VerifierResult(ok=True, issues=[])

    return PipelineDeps(
        scout=lambda t: _make_scout(),
        architect=architect_fn,
        architect_with_feedback=lambda s, issues: architect_fn(s),
        specialist=specialist_fn,
        specialist_with_feedback=None,
        synthesizer=synthesizer_fn,
        verifier=verifier_fn,
        critic=None,
        context_mapper=context_mapper,
    )


# ---------------------------------------------------------------------------
# Test 1: context_mapper set → context_map populated + allowed_dependencies derived
# ---------------------------------------------------------------------------

def test_generate_once_with_context_mapper_produces_map_and_deps():
    deps = _base_deps(context_mapper=_context_mapper_fn)
    scout = _make_scout()

    model, _arch, _specialist = _generate_once(scout, deps, srs_path="fake.docx")

    # context_map must be attached
    assert model.context_map is not None, "context_map should be populated when context_mapper is set"
    assert len(model.context_map.relationships) == 1

    rel = model.context_map.relationships[0]
    assert rel.context_a == "Ordering"
    assert rel.context_b == "Inventory"
    assert rel.relationship_type == "CUSTOMER_SUPPLIER"
    assert rel.upstream == "Inventory"

    # allowed_dependencies derived: Ordering (downstream) → ["Inventory"]
    ordering_ctx = next(bc for bc in model.bounded_contexts if bc.context_name == "Ordering")
    inventory_ctx = next(bc for bc in model.bounded_contexts if bc.context_name == "Inventory")

    assert ordering_ctx.allowed_dependencies == ["Inventory"], (
        f"Expected Ordering.allowed_dependencies=['Inventory'], got {ordering_ctx.allowed_dependencies!r}"
    )
    # Inventory is upstream-only in this relationship → no outgoing edges → None
    assert inventory_ctx.allowed_dependencies is None, (
        f"Expected Inventory.allowed_dependencies=None, got {inventory_ctx.allowed_dependencies!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: context_mapper=None → context_map stays None
# ---------------------------------------------------------------------------

def test_generate_once_without_context_mapper_leaves_map_none():
    deps = _base_deps(context_mapper=None)
    scout = _make_scout()

    model, _arch, _specialist = _generate_once(scout, deps, srs_path="fake.docx")

    assert model.context_map is None, (
        f"context_map should be None when context_mapper is not wired, got {model.context_map!r}"
    )
    # allowed_dependencies untouched (None from synthesizer)
    for bc in model.bounded_contexts:
        assert bc.allowed_dependencies is None, (
            f"{bc.context_name}.allowed_dependencies should be None, got {bc.allowed_dependencies!r}"
        )
