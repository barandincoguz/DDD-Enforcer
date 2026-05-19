"""D6/D7/D8 hard-fail invariants on the deterministic Synthesizer output.

These are code-bug detectors, NOT Refiner-loopable.  If a check fails,
the deterministic merge is broken and an LLM retry will not help.

Distinct from the future CRITIC stage (M2 in the architecture menu):
D6/D7/D8 are deterministic code assertions; CRITIC would be an
LLM-based semantic critique. Different contracts, different cost profiles.
"""

from typing import List

from core.pipeline_contracts import SpecialistAnalysis, VerifierIssue
from core.schemas import DomainModel


def check_d6_entity_count_preservation(
    analyses: List[SpecialistAnalysis],
    model: DomainModel,
) -> List[VerifierIssue]:
    """D6: sum(analyses entities) == sum(model.bounded_contexts entities).

    The deterministic merge must never drop entities.  A count mismatch
    indicates a code bug in merge.py, not an LLM hallucination.
    """
    analysis_count = sum(len(a.entities) for a in analyses)
    model_count = sum(
        len(bc.ubiquitous_language.entities)
        for bc in model.bounded_contexts
    )
    if analysis_count == model_count:
        return []
    return [
        VerifierIssue(
            severity="ERROR",
            check_id="D6",
            target="synthesizer:entity_count",
            message=(
                f"D6 entity-count mismatch: analyses contained "
                f"{analysis_count} entities but model has {model_count}. "
                f"The deterministic merge dropped or fabricated entities."
            ),
        )
    ]


def check_d7_entity_name_traceability(
    analyses: List[SpecialistAnalysis],
    model: DomainModel,
) -> List[VerifierIssue]:
    """D7: every model entity must trace by name (case-insensitive) to a
    Specialist analysis entry.  Fabricated names are a merge code bug.
    """
    analysis_names = {
        e.name.lower()
        for a in analyses
        for e in a.entities
    }
    issues: List[VerifierIssue] = []
    for bc in model.bounded_contexts:
        for entity in bc.ubiquitous_language.entities:
            if entity.name.lower() not in analysis_names:
                issues.append(
                    VerifierIssue(
                        severity="ERROR",
                        check_id="D7",
                        target=f"synthesizer:{bc.context_name}.entities({entity.name})",
                        message=(
                            f"D7 traceability failure: entity {entity.name!r} "
                            f"in context {bc.context_name!r} has no matching "
                            f"entry in any SpecialistAnalysis. Fabricated name."
                        ),
                    )
                )
    return issues


def check_d8_aggregate_member_referential_integrity(
    model: DomainModel,
) -> List[VerifierIssue]:
    """D8: every aggregate.members entry must reference an entity that
    exists in the same bounded context.  A dangling reference is a merge
    code bug.
    """
    issues: List[VerifierIssue] = []
    for bc in model.bounded_contexts:
        entity_names = {e.name.lower() for e in bc.ubiquitous_language.entities}
        aggregates = bc.ubiquitous_language.aggregates or []
        for agg in aggregates:
            for member in agg.members:
                if member.lower() not in entity_names:
                    issues.append(
                        VerifierIssue(
                            severity="ERROR",
                            check_id="D8",
                            target=(
                                f"synthesizer:{bc.context_name}"
                                f".aggregates({agg.name}).members"
                            ),
                            message=(
                                f"D8 referential integrity failure: aggregate "
                                f"{agg.name!r} in context {bc.context_name!r} "
                                f"lists member {member!r} which is not an "
                                f"entity in that context."
                            ),
                        )
                    )
    return issues
