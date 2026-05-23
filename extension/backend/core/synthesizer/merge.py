"""Pure-function deterministic merge.

No network calls. No LLM. Pure Python only.
"""

from typing import List
from core.pipeline_contracts import SpecialistAnalysis
from core.schemas import (
    DomainModel, BoundedContext, UbiquitousLanguage, GlobalRules,
    DomainEvent,
)
from core.synthesizer.metadata import build_default_metadata, build_default_global_rules


def build_deterministic_skeleton(
    analyses: List[SpecialistAnalysis],
    project_name: str,
) -> DomainModel:
    """Merge typed analyses into a DomainModel skeleton.

    Every entity / VO / service / aggregate / domain_event from each
    SpecialistAnalysis is copied verbatim into the corresponding
    BoundedContext slot. No LLM, no field synthesis.

    synonyms_to_avoid stays None (filled later by enrich step).
    allowed_dependencies stays None (filled later by enrich step).
    """
    bounded_contexts = []
    for analysis in analyses:
        ul = UbiquitousLanguage(
            entities=list(analysis.entities),
            value_objects=list(analysis.value_objects) or None,
            services=list(analysis.services) or None,
            aggregates=list(analysis.aggregates) or None,
            domain_events=[e.name for e in analysis.domain_events] or None,
        )
        bc = BoundedContext(
            context_name=analysis.context.context_name,
            # WP-CORE-14 (F-18): pass description through; empty if upstream
            # didn't produce one (downstream enrich step LLM-populates).
            # No synthetic f"{name} context" placeholder that previously
            # masked intermediate-vs-final mismatch in debugging.
            description=analysis.context.description,
            allowed_dependencies=None,  # filled by enrich
            supporting_sentence_ids=list(analysis.context.supporting_sentence_ids),
            business_rules=list(analysis.business_rules) or None,
            ubiquitous_language=ul,
        )
        bounded_contexts.append(bc)

    return DomainModel(
        project_name=project_name,
        project_metadata=build_default_metadata(),
        bounded_contexts=bounded_contexts,
        global_rules=build_default_global_rules(),
    )
