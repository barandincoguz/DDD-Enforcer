"""Phase C3: S1 semantic grounding check via LLM.

The check asks an LLM judge whether a claimed entity actually appears
in its cited Scout sentences. Tests use a mock LLM that returns canned
verdicts so unit tests stay fast and deterministic.
"""

from unittest.mock import MagicMock
from core.verifier.types import IssueSeverity
from core.verifier.checks_semantic import check_s1_entity_grounded_in_evidence


SCOUT_SENTENCES = {
    0: "The Order Management context handles all customer purchases.",
    1: "A Customer can place an Order.",
    2: "Payment is processed by the Billing service.",
}


def test_s1_passes_when_llm_confirms_grounding():
    fake_llm = MagicMock()
    fake_llm.judge.return_value = {"grounded": True, "reason": "Customer appears in sentence 1"}
    entity = {"name": "Customer", "evidence_sentence_indices": [1]}
    issues = check_s1_entity_grounded_in_evidence(
        entity, scout_sentences=SCOUT_SENTENCES, llm_judge=fake_llm
    )
    assert issues == []


def test_s1_flags_when_llm_says_not_grounded():
    fake_llm = MagicMock()
    fake_llm.judge.return_value = {"grounded": False, "reason": "PhantomEntity not present"}
    entity = {"name": "PhantomEntity", "evidence_sentence_indices": [0]}
    issues = check_s1_entity_grounded_in_evidence(
        entity, scout_sentences=SCOUT_SENTENCES, llm_judge=fake_llm
    )
    assert len(issues) == 1
    assert issues[0].issue_type == "semantic_ungrounded"
    assert issues[0].severity == IssueSeverity.ERROR


def test_s1_passes_with_no_indices_phase_c():
    """If the entity has no indices, D2 already handles it — S1 returns []."""
    fake_llm = MagicMock()
    entity = {"name": "Order", "evidence_sentence_indices": []}
    issues = check_s1_entity_grounded_in_evidence(
        entity, scout_sentences=SCOUT_SENTENCES, llm_judge=fake_llm
    )
    assert issues == []
    fake_llm.judge.assert_not_called()
