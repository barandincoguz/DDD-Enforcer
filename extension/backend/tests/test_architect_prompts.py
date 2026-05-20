"""Prompt-level assertions on DomainArchitect's stage prompts.

These tests do NOT call any LLM. They assert that the prompt strings
sent to the LLM include the expected field examples so that all 6 D1
models (Gemini + 4 OSS) receive a faithful schema demonstration.

NOTE (WP-CORE-1 T8): The three former tests that grepped the legacy
omnibus Synthesizer prompt (test_synthesize_prompt_example_includes_*)
are DELETED. The Synthesizer is now a deterministic merge + narrow
per-context enrich; there is no single LLM prompt to grep.
"""


def test_specialist_prompt_emits_structured_aggregates():
    src = open("core/architect.py").read()
    # Anchor on the per-context helper — the canonical Specialist prompt location
    # post-P3 refactor (legacy omnibus extract_all_contexts_details deleted).
    specialist_section = src.split("def _build_specialist_prompt_per_context")[1].split("\n    def ")[0]
    assert '"aggregate_roots"' not in specialist_section, (
        "Specialist prompt must no longer emit flat aggregate_roots:[str]; "
        "use structured aggregates:[{name, members:[entity_name]}] instead"
    )
    assert '"aggregates"' in specialist_section
    assert '"members"' in specialist_section, (
        "Specialist prompt must require aggregate.members so V5 (aggregate boundary) "
        "violation detection has the data it needs."
    )


def test_specialist_prompt_lists_domain_events_as_extraction_target():
    src = open("core/architect.py").read()
    # Anchor on the per-context helper — the canonical Specialist prompt location
    # post-P3 refactor (legacy omnibus extract_all_contexts_details deleted).
    specialist_section = src.split("def _build_specialist_prompt_per_context")[1].split("\n    def ")[0]
    assert "Domain Events" in specialist_section, (
        "Specialist prompt must explicitly instruct extraction of Domain Events; "
        "Synthesizer must not be the first stage that mentions them."
    )
    assert '"domain_events"' in specialist_section


def test_specialist_prompt_demands_confidence_and_justification():
    src = open("core/architect.py").read()
    # The per-context Specialist prompt is now the definitive location for
    # confidence/justification requirements (the legacy synthesize() method
    # and its prompt no longer exist).
    specialist_section = src.split("def _build_specialist_prompt_per_context")[1].split("\n    def ")[0]
    assert '"confidence"' in specialist_section, (
        "Specialist per-context prompt must demonstrate the 'confidence' field "
        "so all 6 D1 models emit it."
    )
    assert '"justification"' in specialist_section, (
        "Specialist per-context prompt must demonstrate the 'justification' field."
    )


def test_specialist_per_context_prompt_requires_evidence_sentence_indices():
    src = open("core/architect.py").read()
    # The per-context prompt was added in C7b inside _build_specialist_prompt_per_context.
    # Split on the def-line to anchor on the body of the helper (the bare
    # function name also appears at its call site in extract_per_context_details).
    prompt_section = src.split("def _build_specialist_prompt_per_context")[1].split("\n    def ")[0]
    assert "evidence_sentence_indices" in prompt_section, (
        "Phase D1: Specialist per-context prompt must require "
        "evidence_sentence_indices per entity for grounding traceability"
    )


def test_specialist_per_context_prompt_emits_description_field():
    """Post-D1 Entity schema requires `description`. The per-context
    Specialist prompt example must include it so the LLM emits it."""
    from core.architect import DomainArchitect
    arch = object.__new__(DomainArchitect)  # bypass __init__
    prompt = DomainArchitect._build_specialist_prompt_per_context(
        arch,
        context_name="Sales",
        numbered_sentences_text="[0] An order is placed by a customer.\n[1] Each order contains line items.",
    )
    assert '"description"' in prompt, (
        "Specialist prompt must instruct LLM to emit entity.description; "
        "strict Entity schema (core/schemas.py:42-55) requires it post-D1."
    )
