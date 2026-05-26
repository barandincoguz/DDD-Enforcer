from core.critic.loop import findings_signature
from core.schemas import CritiqueFinding


def _f(ft, tr):
    return CritiqueFinding(finding_type=ft, priority="high", target_ref=tr,
                           rationale="r", proposed_revision="x")


def test_signature_canonicalizes_reversed_relationship_pairs():
    s1 = findings_signature([_f("WRONG_RELATIONSHIP_TYPE", "relationship:Ordering->Inventory")])
    s2 = findings_signature([_f("WRONG_RELATIONSHIP_TYPE", "relationship:Inventory->Ordering")])
    assert s1 == s2


def test_signature_unchanged_for_context_targets():
    s = findings_signature([_f("ANEMIC_ENTITY", "entity:A.E")])
    assert s == (("ANEMIC_ENTITY", "entity:A.E"),)


import core.orchestration.pipeline as pipeline_mod
from core.orchestration.pipeline import PipelineDeps
from core.critic.loop import run_critique_loop
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata,
                          ContextMap, ContextRelationship, CriticReport, CriticLoopTrace)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


def _ul():
    return UbiquitousLanguage(entities=[], value_objects=[], domain_events=[])

def _model(rel_type="CONFORMIST"):
    m = DomainModel(project_name="P",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[BoundedContext(context_name="Ordering", ubiquitous_language=_ul()),
                          BoundedContext(context_name="Inventory", ubiquitous_language=_ul())],
        global_rules=None)
    m.context_map = ContextMap(model_id="m", relationships=[ContextRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type=rel_type,
        upstream="Inventory", rationale="r")])
    return m

def test_relationship_only_cycle_remaps_without_producer_rerun(monkeypatch):
    scout = ScoutOutput(sentences=[SectionedSentence(index=0, text="s")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=1, truncated_chunks=0))
    calls = {"generate": 0, "mapper": 0}

    def fake_generate_once(scout_, deps_, srs_path, *, architect_feedback=None):
        calls["generate"] += 1
        return _model(), object(), [object()]

    def fake_apply_map(model, deps, scout_, *, feedback=None):
        calls["mapper"] += 1
        out = model.model_copy(deep=True)
        out.context_map = ContextMap(model_id="m", relationships=[ContextRelationship(
            context_a="Ordering", context_b="Inventory",
            relationship_type="ANTI_CORRUPTION_LAYER", upstream="Inventory", rationale="fixed")])
        return out

    monkeypatch.setattr(pipeline_mod, "_generate_once", fake_generate_once)
    monkeypatch.setattr(pipeline_mod, "_apply_context_map", fake_apply_map)

    seq = [
        CriticReport(model_id="m", findings=[CritiqueFinding(
            finding_type="WRONG_RELATIONSHIP_TYPE", priority="high",
            target_ref="relationship:Ordering->Inventory", rationale="ACL not Conformist",
            proposed_revision="ACL")], loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged")),
        CriticReport(model_id="m", findings=[], loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged")),
    ]
    def fake_critic(model, scout_, history):
        return seq.pop(0)

    deps = PipelineDeps(scout=lambda t: scout, architect=lambda s: None,
        architect_with_feedback=lambda s, i: None, specialist=lambda a, s: None,
        synthesizer=lambda x: None, verifier=lambda s: None,
        specialist_with_feedback=lambda a, s, p, i: p, critic=fake_critic,
        context_mapper=lambda m, s, fb: ContextMap(model_id="m"))

    result = run_critique_loop(scout, deps, "srs")
    assert calls["generate"] == 1
    assert calls["mapper"] >= 1
    assert result.context_map.relationships[0].relationship_type == "ANTI_CORRUPTION_LAYER"
