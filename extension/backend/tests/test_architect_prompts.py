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
