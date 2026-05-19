"""Deterministic Synthesizer package.

Replaces the LLM-rewrite Synthesizer at the old architect.py:766-944.
"""

from typing import List
from core.pipeline_contracts import SpecialistAnalysis
from core.schemas import DomainModel
from core.synthesizer.merge import build_deterministic_skeleton
from core.synthesizer.enrich import enrich_synonyms_and_dependencies
from core.synthesizer.errors import SynthesizerInvariantError

# D6/D7/D8 invariants are wired in WP-CORE-1 T5.


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
      3. Verifier D6/D7/D8 invariants (wired in T5).
    """
    skeleton = build_deterministic_skeleton(analyses, project_name=project_name)
    if skip_enrich:
        return skeleton
    return enrich_synonyms_and_dependencies(skeleton, analyses, llm_client)
