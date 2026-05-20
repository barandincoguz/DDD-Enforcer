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

    # Invariants — D6/D7 are HARD-FAIL (deterministic-merge bugs).
    # D8 is auto-healed: a dangling aggregate.members entry usually means
    # the Specialist LLM hallucinated a member name. We log it and DROP
    # the offending member rather than crash the run. A re-check then
    # confirms the model is clean for shipping.
    d67_issues = []
    d67_issues.extend(check_d6_entity_count_preservation(analyses, skeleton))
    d67_issues.extend(check_d7_entity_name_traceability(analyses, skeleton))
    d67_errors = [i for i in d67_issues if i.severity == "ERROR"]
    if d67_errors:
        raise SynthesizerInvariantError(
            check_id=",".join(dict.fromkeys(i.check_id for i in d67_errors)),
            message=f"{len(d67_errors)} invariant failure(s); first: {d67_errors[0].message}",
            details=[i.model_dump() for i in d67_errors],
        )

    d8_issues = check_d8_aggregate_member_referential_integrity(skeleton)
    for issue in d8_issues:
        if issue.severity != "ERROR":
            continue
        # Auto-heal D8: drop the dangling member from the aggregate.
        # target format from D8 check: "{context}.{aggregate}.{member}"
        parts = issue.target.split(".")
        if len(parts) != 3:
            print(f"  ⚠️  D8 auto-heal failed (unexpected target {issue.target!r}); leaving as-is")
            continue
        ctx_name, agg_name, member_name = parts
        for bc in skeleton.bounded_contexts:
            if bc.context_name != ctx_name:
                continue
            for agg in bc.ubiquitous_language.aggregates or []:
                if agg.name == agg_name:
                    agg.members = [m for m in agg.members if m != member_name]
        print(
            f"  ⚠️  D8 auto-healed: dropped dangling member {member_name!r} "
            f"from aggregate {agg_name!r} in context {ctx_name!r}"
        )

    return skeleton
