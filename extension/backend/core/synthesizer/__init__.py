"""Deterministic Synthesizer package.

Replaces the LLM-rewrite Synthesizer at the old architect.py:766-944.
"""

from typing import List
from core.pipeline_contracts import SpecialistAnalysis
from core.schemas import DomainModel
from core.synthesizer.merge import build_deterministic_skeleton
from core.synthesizer.enrich import enrich_synonyms_and_dependencies
from core.synthesizer.errors import SynthesizerInvariantError
from core.verifier.checks_semantic_d6_d7_d8 import (
    check_d6_entity_count_preservation,
    check_d7_entity_name_traceability,
    check_d8_aggregate_member_referential_integrity,
)


def synthesize_domain_model(
    analyses: List[SpecialistAnalysis],
    llm_client,
    project_name: str = "DomainModel",
    skip_enrich: bool = False,
) -> DomainModel:
    """Build a DomainModel from typed Specialist analyses.

    Pipeline:
      1. Deterministic merge → DomainModel skeleton with entities,
         value_objects, services, aggregates, domain_events preserved
         from analyses.
      2. Optional LLM enrichment → fill Entity.synonyms_to_avoid and
         BoundedContext.allowed_dependencies. One narrow LLM call per
         bounded context.
      3. Verifier D6/D7/D8 invariants — code-bug detectors, no retry.
    """
    skeleton = build_deterministic_skeleton(analyses, project_name=project_name)
    if not skip_enrich:
        skeleton = enrich_synonyms_and_dependencies(skeleton, analyses, llm_client)

    # Invariants — code-bug detectors, no retry
    issues = []
    issues.extend(check_d6_entity_count_preservation(analyses, skeleton))
    issues.extend(check_d7_entity_name_traceability(analyses, skeleton))
    issues.extend(check_d8_aggregate_member_referential_integrity(skeleton))

    errors = [i for i in issues if i.severity == "ERROR"]
    if errors:
        raise SynthesizerInvariantError(
            check_id=",".join(dict.fromkeys(i.check_id for i in errors)),
            message=f"{len(errors)} invariant failure(s); first: {errors[0].message}",
            details=[i.model_dump() for i in errors],
        )
    return skeleton
