from core.critic.critic import _map_finding, _valid_targets
from core.critic.types import ProposedFinding
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata)


def _ul():
    return UbiquitousLanguage(entities=[], value_objects=[], domain_events=[])


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", ubiquitous_language=_ul()),
            BoundedContext(context_name="Inventory", ubiquitous_language=_ul()),
        ],
        global_rules=None)


def test_relationship_finding_with_valid_contexts_kept():
    m = _model()
    pf = ProposedFinding(finding_type="WRONG_RELATIONSHIP_TYPE", priority="high",
        target_ref="relationship:Ordering->Inventory", rationale="r", proposed_revision="ACL")
    out = _map_finding(pf, _valid_targets(m), {0}, model=m)
    assert out is not None and out.finding_type == "WRONG_RELATIONSHIP_TYPE"


def test_missing_relationship_on_absent_pair_survives():
    m = _model()
    pf = ProposedFinding(finding_type="MISSING_RELATIONSHIP", priority="medium",
        target_ref="relationship:Ordering->Inventory", rationale="should relate",
        proposed_revision="add Customer-Supplier")
    assert _map_finding(pf, _valid_targets(m), {0}, model=m) is not None


def test_relationship_with_unknown_context_dropped():
    m = _model()
    pf = ProposedFinding(finding_type="ILLEGAL_DEPENDENCY", priority="high",
        target_ref="relationship:Ordering->Ghost", rationale="r", proposed_revision="x")
    assert _map_finding(pf, _valid_targets(m), {0}, model=m) is None


def test_existing_context_finding_still_works():
    m = _model()
    pf = ProposedFinding(finding_type="ANEMIC_ENTITY", priority="low",
        target_ref="context:Ordering", rationale="r", proposed_revision="x")
    assert _map_finding(pf, _valid_targets(m), {0}, model=m) is not None
