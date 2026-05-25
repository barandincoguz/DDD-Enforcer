from core.critic.types import ProposedFinding, CriticResponse, CritiqueCycleMemory


def test_critic_response_parses_analysis_and_findings():
    resp = CriticResponse.model_validate({
        "analysis": "Ordering context is cohesive; Order entity is anemic.",
        "findings": [{
            "finding_type": "ANEMIC_ENTITY", "priority": "high",
            "target_ref": "entity:Ordering.Order", "rationale": "no behavior",
            "proposed_revision": "add place()/cancel()",
            "evidence_sentence_indices": [3],
        }],
    })
    assert resp.analysis.startswith("Ordering")
    assert resp.findings[0].finding_type == "ANEMIC_ENTITY"


def test_cycle_memory_holds_prior_findings_and_diff():
    mem = CritiqueCycleMemory(
        cycle=0,
        findings_summary=["high ANEMIC_ENTITY entity:Ordering.Order"],
        diff_summary="cycle 0: initial model",
    )
    assert mem.cycle == 0
