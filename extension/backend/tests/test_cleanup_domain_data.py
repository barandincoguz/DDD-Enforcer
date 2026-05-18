"""Phase A6: _cleanup_domain_data must preserve LLM-emitted values rather
than fabricating defaults (FM-20). Test verifies that a snake_case
naming_convention emitted by the LLM survives, and that an absent
global_rules block does not get replaced with hardcoded content.
"""

from core.architect import DomainArchitect


def _make_arch():
    arch = DomainArchitect.__new__(DomainArchitect)
    return arch


def test_cleanup_preserves_llm_emitted_naming_convention():
    arch = _make_arch()
    raw = {
        "project_name": "X",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [],
        "global_rules": {
            "naming_convention": "snake_case",  # LLM intentionally chose snake_case
            "banned_global_terms": [],
        },
    }
    cleaned = arch._cleanup_domain_data(raw)
    assert cleaned["global_rules"]["naming_convention"] == "snake_case"


def test_cleanup_passes_through_missing_global_rules_unchanged():
    arch = _make_arch()
    raw = {
        "project_name": "X",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [],
        # No global_rules key at all
    }
    cleaned = arch._cleanup_domain_data(raw)
    # Cleanup is structural only; if the LLM omitted the block, leave it absent
    # so the orchestrator can decide whether to raise.
    assert "global_rules" not in cleaned or cleaned["global_rules"] is None


def test_cleanup_still_coerces_domain_events_objects_to_strings():
    """The only kept transformation: domain_events as objects → strings."""
    arch = _make_arch()
    raw = {
        "project_name": "X",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [{
            "context_name": "OrderMgmt",
            "description": "manages orders",
            "ubiquitous_language": {
                "entities": [],
                "value_objects": [],
                "domain_events": [{"name": "OrderPlaced"}, "OrderCancelled"],
            },
        }],
    }
    cleaned = arch._cleanup_domain_data(raw)
    events = cleaned["bounded_contexts"][0]["ubiquitous_language"]["domain_events"]
    assert events == ["OrderPlaced", "OrderCancelled"]
