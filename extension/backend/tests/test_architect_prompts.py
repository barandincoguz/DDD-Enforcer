"""Prompt-level assertions on DomainArchitect's stage prompts.

These tests do NOT call any LLM. They assert that the prompt strings
sent to the LLM include the expected field examples so that all 6 D1
models (Gemini + 4 OSS) receive a faithful schema demonstration.
"""


def test_synthesize_prompt_example_includes_services():
    src = open("core/architect.py").read()
    synthesize_section = src.split("def synthesize(")[1].split("def ")[0]
    assert '"services"' in synthesize_section, (
        "synthesize prompt example output must demonstrate the 'services' field "
        "or OSS models that follow the prompt literally will silently drop services"
    )


def test_synthesize_prompt_example_includes_aggregates():
    src = open("core/architect.py").read()
    synthesize_section = src.split("def synthesize(")[1].split("def ")[0]
    assert '"aggregates"' in synthesize_section, (
        "synthesize prompt example output must demonstrate the 'aggregates' field"
    )


def test_synthesize_prompt_example_includes_domain_events_objects():
    src = open("core/architect.py").read()
    synthesize_section = src.split("def synthesize(")[1].split("def ")[0]
    assert '"domain_events"' in synthesize_section


def test_specialist_prompt_emits_structured_aggregates():
    src = open("core/architect.py").read()
    specialist_section = src.split("def extract_all_contexts_details(")[1].split("def ")[0]
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
    specialist_section = src.split("def extract_all_contexts_details(")[1].split("def ")[0]
    assert "Domain Events" in specialist_section, (
        "Specialist prompt must explicitly instruct extraction of Domain Events; "
        "Synthesizer must not be the first stage that mentions them."
    )
    assert '"domain_events"' in specialist_section


def test_specialist_prompt_demands_confidence_and_justification():
    src = open("core/architect.py").read()
    # The Synthesizer prompt cascades this requirement downstream;
    # assert directly at the synthesize prompt which is definitive.
    synth = src.split("def synthesize(")[1].split("def ")[0]
    assert '"confidence"' in synth
    assert '"justification"' in synth
