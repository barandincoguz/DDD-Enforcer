"""Deterministic D1-D5 checks. Pure functions over stage output dicts."""

from typing import Dict, List, Optional, Set
from core.verifier.types import IssueSeverity, VerifierIssue


def check_d1_supporting_sentence_ids_subset(
    contexts: List[Dict],
    scout_sentence_indices: Set[int],
    srs_path: Optional[str] = None,
) -> List[VerifierIssue]:
    """D1: every BC.supporting_sentence_ids is (a) non-empty AND (b) ⊆ Scout-emitted indices.

    Non-emptiness clause added per WP-CORE-6 to close the F-21 vacuous-pass
    defect: prior to WP-CORE-6, Architect never populated this field, so the
    subset check passed trivially (empty list ⊆ any set) for every project
    run. The non-empty clause is an honest-signal defense; full pipeline-level
    enforcement requires F-22 (Refiner stage-aware re-runs).

    WP-CORE-13 (F-24): optional `srs_path` threaded into emitted VerifierIssue
    instances so issue-level run-manifest provenance is preserved.
    """
    issues: List[VerifierIssue] = []
    for ctx in contexts:
        ids = ctx.get("supporting_sentence_ids", [])
        if not ids:
            issues.append(VerifierIssue(
                stage="architect",
                location=f"architect:contexts[{ctx.get('name')}].supporting_sentence_ids",
                issue_type="ungrounded_context",
                severity=IssueSeverity.ERROR,
                message=(
                    f"Context {ctx.get('name')!r} has no supporting_sentence_ids "
                    f"— cannot verify SRS grounding"
                ),
                suggestion="Architect must cite ≥1 Scout-emitted sentence index per context",
                srs_path=srs_path,
            ))
            continue
        bad = [i for i in ids if i not in scout_sentence_indices]
        if bad:
            issues.append(VerifierIssue(
                stage="architect",
                location=f"architect:contexts[{ctx.get('name')}].supporting_sentence_ids",
                issue_type="ungrounded_context",
                severity=IssueSeverity.ERROR,
                message=(
                    f"Context {ctx.get('name')!r} cites sentence ids {bad} "
                    f"that Scout did not emit"
                ),
                suggestion=f"Drop sentence ids {bad} or revise the context",
                srs_path=srs_path,
            ))
    return issues


def check_d2_entity_evidence_nonempty(
    context_name: str,
    entities: List[Dict],
    phase: str = "D",
) -> List[VerifierIssue]:
    """D2: every Entity has ≥1 evidence_sentence_index.

    In Phases A-C the field is optional, so emit WARN. From Phase D
    onward the field is required (min_length=1 on the schema) and
    missing evidence is an ERROR.
    """
    severity = IssueSeverity.ERROR if phase >= "D" else IssueSeverity.WARN
    issues: List[VerifierIssue] = []
    for idx, entity in enumerate(entities):
        if not entity.get("evidence_sentence_indices"):
            issues.append(VerifierIssue(
                stage="specialist",
                location=f"specialist:{context_name}.entities[{idx}]({entity.get('name')})",
                issue_type="missing_evidence",
                severity=severity,
                message=(
                    f"Entity {entity.get('name')!r} has no "
                    f"evidence_sentence_indices"
                ),
                suggestion="Cite at least one Scout sentence id that names this entity",
            ))
    return issues


def check_d3_entity_names_unique_across_contexts(
    entities_by_context: Dict[str, List[Dict]],
) -> List[VerifierIssue]:
    """D3: every entity name appears in exactly one bounded context."""
    issues: List[VerifierIssue] = []
    seen: Dict[str, str] = {}
    for ctx_name, entities in entities_by_context.items():
        for entity in entities:
            name = entity.get("name")
            if name in seen and seen[name] != ctx_name:
                issues.append(VerifierIssue(
                    stage="specialist",
                    location=f"specialist:{ctx_name}.entities({name})",
                    issue_type="duplicate_entity_across_contexts",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Entity {name!r} appears in both {seen[name]!r} "
                        f"and {ctx_name!r}"
                    ),
                    suggestion="Pick one context for this entity",
                ))
            else:
                seen[name] = ctx_name
    return issues


def check_d4_aggregate_members_exist_in_context(
    context_name: str,
    entities: List[Dict],
    aggregates: List[Dict],
) -> List[VerifierIssue]:
    """D4: every Aggregate.members entry exists as an Entity in the same context."""
    entity_names = {e.get("name") for e in entities}
    issues: List[VerifierIssue] = []
    for agg in aggregates:
        for member in agg.get("members", []):
            if member not in entity_names:
                issues.append(VerifierIssue(
                    stage="specialist",
                    location=(
                        f"specialist:{context_name}.aggregates({agg.get('name')}).members"
                    ),
                    issue_type="invalid_aggregate_member",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Aggregate {agg.get('name')!r} lists member "
                        f"{member!r} which is not an entity in {context_name!r}"
                    ),
                    suggestion=(
                        f"Either add an entity {member!r} or drop it from members"
                    ),
                ))
    return issues


def check_d5_allowed_dependencies_reference_existing_contexts(
    contexts: List[Dict],
) -> List[VerifierIssue]:
    """D5: every context.allowed_dependencies references an existing context."""
    known = {ctx.get("name") for ctx in contexts}
    issues: List[VerifierIssue] = []
    for ctx in contexts:
        for dep in ctx.get("allowed_dependencies") or []:
            if dep not in known:
                issues.append(VerifierIssue(
                    stage="synthesizer",
                    location=f"synthesizer:contexts[{ctx.get('name')}].allowed_dependencies",
                    issue_type="unknown_dependency",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Context {ctx.get('name')!r} declares dependency "
                        f"{dep!r} which is not a known context"
                    ),
                    suggestion=f"Drop {dep!r} or add a corresponding bounded context",
                ))
    return issues
