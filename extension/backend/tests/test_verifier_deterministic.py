"""Phase C2: D1-D5 deterministic checks."""

from core.verifier.types import IssueSeverity
from core.verifier.checks_deterministic import (
    check_d1_supporting_sentence_ids_subset,
    check_d2_entity_evidence_nonempty,
    check_d3_entity_names_unique_across_contexts,
    check_d4_aggregate_members_exist_in_context,
    check_d5_allowed_dependencies_reference_existing_contexts,
)


SCOUT_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


# ----- D1 -----

def test_d1_passes_when_all_supporting_ids_in_scout():
    contexts = [{"name": "OrderMgmt", "supporting_sentence_ids": [1, 2, 3]}]
    issues = check_d1_supporting_sentence_ids_subset(contexts, SCOUT_INDICES)
    assert issues == []


def test_d1_flags_id_not_in_scout():
    contexts = [{"name": "OrderMgmt", "supporting_sentence_ids": [1, 99]}]
    issues = check_d1_supporting_sentence_ids_subset(contexts, SCOUT_INDICES)
    assert len(issues) == 1
    assert issues[0].issue_type == "ungrounded_context"
    assert issues[0].severity == IssueSeverity.ERROR


# ----- D2 -----

def test_d2_passes_when_evidence_phase_a_optional():
    """In Phase A-C, evidence_sentence_indices is optional → emit only warn."""
    entity = {"name": "Order", "evidence_sentence_indices": []}
    issues = check_d2_entity_evidence_nonempty(
        context_name="OrderMgmt", entities=[entity], phase="C"
    )
    assert len(issues) == 1
    assert issues[0].severity == IssueSeverity.WARN


def test_d2_passes_when_evidence_present_phase_d():
    entity = {"name": "Order", "evidence_sentence_indices": [2, 5]}
    issues = check_d2_entity_evidence_nonempty(
        context_name="OrderMgmt", entities=[entity], phase="D"
    )
    assert issues == []


def test_d2_errors_when_evidence_empty_phase_d():
    entity = {"name": "Order", "evidence_sentence_indices": []}
    issues = check_d2_entity_evidence_nonempty(
        context_name="OrderMgmt", entities=[entity], phase="D"
    )
    assert len(issues) == 1
    assert issues[0].severity == IssueSeverity.ERROR


# ----- D3 -----

def test_d3_passes_when_names_unique():
    by_context = {
        "OrderMgmt": [{"name": "Order"}, {"name": "OrderLine"}],
        "Billing": [{"name": "Invoice"}],
    }
    issues = check_d3_entity_names_unique_across_contexts(by_context)
    assert issues == []


def test_d3_flags_duplicate_entity_in_two_contexts():
    by_context = {
        "OrderMgmt": [{"name": "Customer"}],
        "Billing": [{"name": "Customer"}],
    }
    issues = check_d3_entity_names_unique_across_contexts(by_context)
    assert len(issues) == 1
    assert issues[0].issue_type == "duplicate_entity_across_contexts"


# ----- D4 -----

def test_d4_passes_when_aggregate_members_are_entities():
    entities = [{"name": "Order"}, {"name": "OrderLine"}]
    aggregates = [{"name": "Order", "members": ["Order", "OrderLine"]}]
    issues = check_d4_aggregate_members_exist_in_context(
        context_name="OrderMgmt", entities=entities, aggregates=aggregates
    )
    assert issues == []


def test_d4_flags_phantom_member():
    entities = [{"name": "Order"}]
    aggregates = [{"name": "Order", "members": ["Order", "PhantomLine"]}]
    issues = check_d4_aggregate_members_exist_in_context(
        context_name="OrderMgmt", entities=entities, aggregates=aggregates
    )
    assert len(issues) == 1
    assert "PhantomLine" in issues[0].message


# ----- D5 -----

def test_d5_passes_when_dependencies_reference_existing():
    contexts = [
        {"name": "OrderMgmt", "allowed_dependencies": ["Billing"]},
        {"name": "Billing", "allowed_dependencies": []},
    ]
    issues = check_d5_allowed_dependencies_reference_existing_contexts(contexts)
    assert issues == []


def test_d5_flags_unknown_dependency():
    contexts = [
        {"name": "OrderMgmt", "allowed_dependencies": ["GhostContext"]},
    ]
    issues = check_d5_allowed_dependencies_reference_existing_contexts(contexts)
    assert len(issues) == 1
    assert "GhostContext" in issues[0].message
