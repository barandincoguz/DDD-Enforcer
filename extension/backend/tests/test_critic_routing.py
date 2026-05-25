from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
    CritiqueFinding,
)
from core.critic.routing import (
    partition_findings, adapt_structural_to_issues, adapt_content_to_issues,
    model_diff_summary,
)


def _finding(ft, pri="high", target="context:Ctx"):
    return CritiqueFinding(
        finding_type=ft, priority=pri, target_ref=target,
        rationale="r", proposed_revision="p",
    )


def _model(ctx_names, entities_by_ctx):
    return DomainModel(
        project_name="P",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name=c,
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name=e, description="d", confidence=0.9,
                    justification="j", evidence_sentence_indices=[0],
                ) for e in entities_by_ctx.get(c, [])],
                value_objects=None, domain_events=None,
            ),
        ) for c in ctx_names],
        global_rules=None,
    )


def test_partition_splits_structural_content_advisory():
    findings = [
        _finding("CONTEXT_SHOULD_MERGE", "high"),
        _finding("ANEMIC_ENTITY", "medium", "entity:Ctx.Order"),
        _finding("NAMING_SMELL", "low", "entity:Ctx.Order"),
    ]
    structural, content, _relationship, advisory = partition_findings(findings)
    assert [f.finding_type for f in structural] == ["CONTEXT_SHOULD_MERGE"]
    assert [f.finding_type for f in content] == ["ANEMIC_ENTITY"]
    assert [f.finding_type for f in advisory] == ["NAMING_SMELL"]


def test_misplaced_entity_is_content_not_structural():
    structural, content, _rel, _ = partition_findings([_finding("MISPLACED_ENTITY", "high", "entity:A.X")])
    assert structural == []
    assert len(content) == 1


def test_adapt_structural_keeps_generic_target():
    issues = adapt_structural_to_issues([_finding("CONTEXT_SHOULD_MERGE", "high", "context:Ctx")])
    assert issues[0].target == "context:Ctx"
    assert issues[0].severity == "ERROR"
    assert "r" in issues[0].message and "p" in issues[0].suggestion


def test_adapt_content_emits_specialist_prefix_for_affected_context():
    # _specialist_with_feedback derives the affected context from a
    # "specialist:<ctx>" prefix (architect.py:_parse_target_ctx); the adapter
    # MUST emit that prefix or no context is re-extracted.
    issues = adapt_content_to_issues([_finding("ANEMIC_ENTITY", "high", "entity:Ctx.Order")])
    assert issues[0].location == "specialist:Ctx"
    assert issues[0].severity == "ERROR"


def test_model_diff_summary_reports_context_and_entity_deltas():
    before = _model(["A", "B"], {"A": ["X"]})
    after = _model(["A"], {"A": ["X", "Y"]})
    summary = model_diff_summary(before, after)
    assert "B" in summary
    assert "Y" in summary
