"""Route critic findings to producer stages + summarize model diffs.

Structural findings re-derive the architecture (Architect); content findings
re-extract within existing contexts (Specialist). Low-priority findings are
advisory only (never trigger a regeneration).
"""
from dataclasses import dataclass
from typing import List, Tuple
from core.schemas import CritiqueFinding, DomainModel

_STRUCTURAL = {"CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "BOUNDARY_SMELL"}
_CONTENT = {
    "ANEMIC_ENTITY", "ANEMIC_MODEL", "MISSING_AGGREGATE",
    "MISPLACED_ENTITY", "NAMING_SMELL", "LOW_CONFIDENCE",
}
_RELATIONSHIP = {"WRONG_RELATIONSHIP_TYPE", "ILLEGAL_DEPENDENCY", "MISSING_RELATIONSHIP"}


def partition_findings(
    findings: List[CritiqueFinding],
) -> Tuple[List[CritiqueFinding], List[CritiqueFinding], List[CritiqueFinding], List[CritiqueFinding]]:
    """Return (structural, content, relationship, advisory). Only high/medium
    findings are routable; low priority and OTHER are advisory."""
    structural: List[CritiqueFinding] = []
    content: List[CritiqueFinding] = []
    relationship: List[CritiqueFinding] = []
    advisory: List[CritiqueFinding] = []
    for f in findings:
        if f.priority == "low" or f.finding_type == "OTHER":
            advisory.append(f)
        elif f.finding_type in _STRUCTURAL:
            structural.append(f)
        elif f.finding_type in _CONTENT:
            content.append(f)
        elif f.finding_type in _RELATIONSHIP:
            relationship.append(f)
        else:
            advisory.append(f)
    return structural, content, relationship, advisory


@dataclass
class _CritiqueIssue:
    """Adapter so a CritiqueFinding is consumable by the producer feedback
    paths. Architect feedback (_build_grounding_feedback_block) reads
    .target/.location/.message generically. Specialist feedback requires the
    location to start with 'specialist:<ctx>' so _parse_target_ctx
    (architect.py) flags the affected context, and render_refinement_prompt
    reads .suggestion. All fields are populated to satisfy both consumers."""
    severity: str
    target: str
    location: str
    message: str
    suggestion: str


def _context_of(target_ref: str) -> str:
    """'context:Ordering' -> 'Ordering'; 'entity:Ordering.Order' -> 'Ordering'."""
    body = target_ref.split(":", 1)[-1]
    return body.split(".")[0].strip()


def adapt_structural_to_issues(findings: List[CritiqueFinding]) -> List[_CritiqueIssue]:
    """Architect-bound feedback: generic target/message (no stage prefix needed)."""
    out: List[_CritiqueIssue] = []
    for f in findings:
        out.append(_CritiqueIssue(
            severity="ERROR" if f.priority == "high" else "WARN",
            target=f.target_ref, location=f.target_ref,
            message=f"{f.rationale} | suggested: {f.proposed_revision}",
            suggestion=f.proposed_revision,
        ))
    return out


def adapt_content_to_issues(findings: List[CritiqueFinding]) -> List[_CritiqueIssue]:
    """Specialist-bound feedback: target/location MUST carry the
    'specialist:<ctx>' prefix so _specialist_with_feedback flags the right
    context as affected (architect.py:_parse_target_ctx splits on ':' then '.')."""
    out: List[_CritiqueIssue] = []
    for f in findings:
        ctx = _context_of(f.target_ref)
        out.append(_CritiqueIssue(
            severity="ERROR" if f.priority == "high" else "WARN",
            target=f"specialist:{ctx}", location=f"specialist:{ctx}",
            message=f"{f.rationale} | suggested: {f.proposed_revision}",
            suggestion=f.proposed_revision,
        ))
    return out


def _entity_names(model: DomainModel) -> dict:
    return {
        bc.context_name: {e.name for e in bc.ubiquitous_language.entities}
        for bc in model.bounded_contexts
    }


def model_diff_summary(before: DomainModel, after: DomainModel) -> str:
    """Compact deterministic diff for Reflexion memory."""
    before_ctx = {bc.context_name for bc in before.bounded_contexts}
    after_ctx = {bc.context_name for bc in after.bounded_contexts}
    parts: List[str] = []
    added_ctx = sorted(after_ctx - before_ctx)
    removed_ctx = sorted(before_ctx - after_ctx)
    if added_ctx:
        parts.append(f"contexts added: {', '.join(added_ctx)}")
    if removed_ctx:
        parts.append(f"contexts removed: {', '.join(removed_ctx)}")
    be, ae = _entity_names(before), _entity_names(after)
    for ctx in sorted(after_ctx & before_ctx):
        added_e = sorted(ae.get(ctx, set()) - be.get(ctx, set()))
        removed_e = sorted(be.get(ctx, set()) - ae.get(ctx, set()))
        if added_e:
            parts.append(f"{ctx}: entities added: {', '.join(added_e)}")
        if removed_e:
            parts.append(f"{ctx}: entities removed: {', '.join(removed_e)}")
    def _rels(m: DomainModel) -> dict:
        cm = getattr(m, "context_map", None)
        if not cm:
            return {}
        return {tuple(sorted((r.context_a, r.context_b))): (r.relationship_type, r.upstream)
                for r in cm.relationships}
    rb, ra = _rels(before), _rels(after)
    for pair in sorted(set(ra) - set(rb)):
        parts.append(f"relationship added: {pair[0]}/{pair[1]} = {ra[pair][0]}")
    for pair in sorted(set(rb) - set(ra)):
        parts.append(f"relationship removed: {pair[0]}/{pair[1]}")
    for pair in sorted(set(ra) & set(rb)):
        if ra[pair] != rb[pair]:
            parts.append(f"relationship changed: {pair[0]}/{pair[1]} {rb[pair][0]}->{ra[pair][0]}")
    return "; ".join(parts) if parts else "no structural change"
