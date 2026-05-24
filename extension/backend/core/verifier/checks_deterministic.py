"""Deterministic D1-D5 + D9 + S3 + D11 checks. Pure functions over stage output dicts."""

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
            if not isinstance(name, str):
                # Skip entities with missing/non-string names; duplicate-name
                # detection is meaningless without a stable key. Schema-level
                # required-field validation belongs in the Pydantic contract,
                # not here.
                continue
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


def check_d9_value_object_mutability_consistency(
    context_name: str,
    value_objects: List[Dict],
) -> List[VerifierIssue]:
    """D9 (WP-CORE-27): claimed Value Object whose backing class has
    mutation methods → ERROR (claim mismatch).

    The `is_mutable_in_code` field is set by AST cross-reference when the
    class corresponding to the VO has mutation methods or non-frozen
    state. v1 ships the deterministic check; the AST→VO field
    population is a follow-up wiring task (WP-CORE-27a).

    A True flag here means: LLM said "this is a Value Object" but
    AST shows the class is mutable — semantic contradiction.
    """
    issues: List[VerifierIssue] = []
    for idx, vo in enumerate(value_objects):
        if vo.get("is_mutable_in_code", False):
            name = vo.get("name", "<unknown>")
            issues.append(VerifierIssue(
                stage="specialist",
                location=f"specialist:{context_name}.value_objects[{idx}]({name})",
                issue_type="value_object_mutable",
                severity=IssueSeverity.ERROR,
                message=(
                    f"Value Object {name!r} claimed in {context_name!r} "
                    f"but AST evidence shows the backing class has "
                    f"mutation methods (D9 mismatch)"
                ),
                suggestion=(
                    "Either reclassify the concept as an Entity, or remove "
                    "the mutation methods from the backing class to make it "
                    "a true Value Object"
                ),
            ))
    return issues


def check_s3_aggregate_members_nonempty(
    context_name: str,
    aggregates: List[Dict],
) -> List[VerifierIssue]:
    """S3 (WP-CORE-27): Aggregate with empty `members` list → WARN.

    Companion to D4 (which checks members exist in the context). D4
    passes vacuously for an aggregate with members=[] (empty loop). S3
    closes that gap with a WARN (not ERROR — empty aggregates may be
    valid for nascent designs in early iterations).
    """
    issues: List[VerifierIssue] = []
    for idx, agg in enumerate(aggregates):
        members = agg.get("members") or []
        if not members:
            name = agg.get("name", "<unknown>")
            issues.append(VerifierIssue(
                stage="specialist",
                location=f"specialist:{context_name}.aggregates[{idx}]({name})",
                issue_type="empty_aggregate",
                severity=IssueSeverity.WARN,
                message=(
                    f"Aggregate {name!r} in {context_name!r} has no members; "
                    f"D4 passes vacuously"
                ),
                suggestion=(
                    "List the entities that belong inside this aggregate's "
                    "consistency boundary"
                ),
            ))
    return issues


def check_d11_dependency_cycle_free(
    contexts: List[Dict],
) -> List[VerifierIssue]:
    """D11 (WP-CORE-27): no cycle in BoundedContext.allowed_dependencies graph.

    DFS with WHITE/GRAY/BLACK coloring detects back-edges. Self-loops
    (A→A) and multi-node cycles (A→B→…→A) both ERROR. Unknown
    dependencies (referenced but not in the context list) are silently
    ignored — D5 reports those separately.

    Each cycle is reported at most once via canonical (sorted-tuple)
    deduplication so a multi-rooted DFS does not emit the same cycle
    twice.
    """
    graph: Dict[str, List[str]] = {}
    for ctx in contexts:
        name = ctx.get("name")
        if not isinstance(name, str):
            continue
        deps = ctx.get("allowed_dependencies") or []
        graph[name] = [d for d in deps if isinstance(d, str)]

    if not graph:
        return []

    issues: List[VerifierIssue] = []
    seen_cycles: Set[tuple] = set()

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {node: WHITE for node in graph}

    def _emit_cycle(path: List[str], back_to: str) -> None:
        try:
            start = path.index(back_to)
        except ValueError:
            return
        cycle = path[start:] + [back_to]
        # Canonical form for dedup: rotate to the lex-smallest start.
        body = cycle[:-1]
        if not body:
            return
        min_idx = body.index(min(body))
        rotated = tuple(body[min_idx:] + body[:min_idx])
        if rotated in seen_cycles:
            return
        seen_cycles.add(rotated)
        issues.append(VerifierIssue(
            stage="synthesizer",
            location="synthesizer:contexts.allowed_dependencies",
            issue_type="dependency_cycle",
            severity=IssueSeverity.ERROR,
            message=(
                f"Dependency cycle detected in allowed_dependencies: "
                f"{' → '.join(cycle)}"
            ),
            suggestion=(
                "Break the cycle by removing one allowed_dependency edge "
                "or introducing an anti-corruption layer between the two "
                "contexts"
            ),
        ))

    def _dfs(node: str, path: List[str]) -> None:
        color[node] = GRAY
        path.append(node)
        for nbr in graph.get(node, []):
            if nbr not in color:
                continue  # unknown context → D5 catches it
            if color[nbr] == GRAY:
                _emit_cycle(path, nbr)
                continue
            if color[nbr] == WHITE:
                _dfs(nbr, path)
        path.pop()
        color[node] = BLACK

    for node in list(graph.keys()):
        if color[node] == WHITE:
            _dfs(node, [])

    return issues
