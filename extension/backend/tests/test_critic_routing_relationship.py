from core.critic.routing import partition_findings, model_diff_summary
from core.schemas import (CritiqueFinding, DomainModel, BoundedContext,
                          UbiquitousLanguage, ProjectMetadata, ContextMap, ContextRelationship)


def _f(ft, tr, pr="high"):
    return CritiqueFinding(finding_type=ft, priority=pr, target_ref=tr,
                           rationale="r", proposed_revision="x")


def test_partition_returns_four_buckets():
    findings = [
        _f("CONTEXT_SHOULD_MERGE", "context:A"),
        _f("ANEMIC_ENTITY", "entity:A.E"),
        _f("WRONG_RELATIONSHIP_TYPE", "relationship:A->B"),
        _f("OTHER", "context:A", pr="low"),
    ]
    structural, content, relationship, advisory = partition_findings(findings)
    assert [x.finding_type for x in structural] == ["CONTEXT_SHOULD_MERGE"]
    assert [x.finding_type for x in content] == ["ANEMIC_ENTITY"]
    assert [x.finding_type for x in relationship] == ["WRONG_RELATIONSHIP_TYPE"]
    assert len(advisory) == 1


def _ul():
    return UbiquitousLanguage(entities=[], value_objects=[], domain_events=[])


def _model(rels=None):
    m = DomainModel(project_name="P",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[BoundedContext(context_name="A", ubiquitous_language=_ul()),
                          BoundedContext(context_name="B", ubiquitous_language=_ul())],
        global_rules=None)
    if rels is not None:
        m.context_map = ContextMap(model_id="m", relationships=rels)
    return m


def test_diff_summary_reports_relationship_change():
    before = _model(rels=[ContextRelationship(context_a="A", context_b="B",
        relationship_type="CONFORMIST", upstream="B", rationale="r")])
    after = _model(rels=[ContextRelationship(context_a="A", context_b="B",
        relationship_type="ANTI_CORRUPTION_LAYER", upstream="B", rationale="r")])
    summary = model_diff_summary(before, after)
    assert "A" in summary and "B" in summary
    assert "CONFORMIST" in summary or "ANTI_CORRUPTION_LAYER" in summary or "relationship" in summary.lower()
    assert summary != "no structural change"
