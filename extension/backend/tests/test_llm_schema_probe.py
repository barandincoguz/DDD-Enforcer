"""Smoke tests for core.llm.schema_probe."""

import json
from unittest.mock import MagicMock, patch

import pytest

from core.llm.schema_probe import (
    BasicViolation,
    ComplexViolation,
    MediumViolation,
    PROMPTS,
    SCHEMAS,
    _parse_args,
    probe_cell,
    run_probe,
)
from core.llm.base import LLMResponse, TokenUsage
from core.llm.registry import MODELS


def test_three_schemas_registered():
    assert set(SCHEMAS.keys()) == {"basic", "medium", "complex"}
    assert SCHEMAS["basic"] is BasicViolation
    assert SCHEMAS["medium"] is MediumViolation
    assert SCHEMAS["complex"] is ComplexViolation


def test_each_schema_has_a_prompt():
    for name in SCHEMAS:
        assert name in PROMPTS
        assert isinstance(PROMPTS[name], str)
        assert PROMPTS[name].strip()


def test_basic_violation_validates_minimal_payload():
    inst = BasicViolation(name="Customer", description="A buyer")
    assert inst.name == "Customer"


def test_complex_violation_requires_entities():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ComplexViolation(
            context_name="X",
            description="missing entities field",
        )  # type: ignore[call-arg]


def _ok_response(model_id: str) -> LLMResponse:
    return LLMResponse(
        content='{"name": "Customer", "description": "A buyer"}',
        parsed=BasicViolation(name="Customer", description="A buyer"),
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model_id=model_id,
        provider="gemini",
        json_failed=False,
        latency_ms=42.0,
    )


def _failed_response(model_id: str) -> LLMResponse:
    return LLMResponse(
        content="garbage",
        parsed=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model_id=model_id,
        provider="gemini",
        json_failed=True,
        json_fail_reason="invalid_json: garbage",
        latency_ms=30.0,
    )


def test_probe_cell_aggregates_success_count():
    fake_client = MagicMock()
    fake_client.structured_output.side_effect = [
        _ok_response("gemini-3.1-pro-preview"),
        _ok_response("gemini-3.1-pro-preview"),
        _failed_response("gemini-3.1-pro-preview"),
    ]
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        result = probe_cell("gemini-3.1-pro-preview", "basic", trials=3)
    assert result.trials == 3
    assert result.success == 2
    assert result.json_failed == 1
    assert result.provider == "gemini"
    assert result.mean_latency_ms > 0
    assert result.total_tokens == 15 + 15 + 12
    assert any("invalid_json" in err for err in result.errors)


def test_run_probe_iterates_all_six_models_by_default():
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("placeholder")
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        report = run_probe(trials=1)
    # 6 models × 3 schemas = 18 cells
    assert report["trials_per_cell"] == 1
    assert len(report["results"]) == 18
    model_ids_in_report = {r["model_id"] for r in report["results"]}
    assert model_ids_in_report == set(MODELS.keys())


def test_run_probe_honors_models_filter():
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("placeholder")
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        report = run_probe(models=["gemini-3.1-pro-preview"], schemas=["basic"], trials=1)
    assert len(report["results"]) == 1
    assert report["results"][0]["model_id"] == "gemini-3.1-pro-preview"
    assert report["results"][0]["schema"] == "basic"


def test_parse_args_defaults():
    args = _parse_args([])
    assert args.out == "runs/probe.json"
    assert args.trials == 1
    assert args.models is None
    assert args.schemas is None


def test_parse_args_overrides():
    args = _parse_args([
        "--out", "/tmp/probe.json",
        "--trials", "3",
        "--models", "gemini-3.1-pro-preview",
        "--schemas", "complex",
    ])
    assert args.out == "/tmp/probe.json"
    assert args.trials == 3
    assert args.models == ["gemini-3.1-pro-preview"]
    assert args.schemas == ["complex"]


def test_report_serializes_to_json():
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("placeholder")
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        report = run_probe(models=["gemini-3.1-pro-preview"], schemas=["basic"], trials=1)
    # Should round-trip through json without error
    serialized = json.dumps(report)
    parsed = json.loads(serialized)
    assert parsed["results"][0]["success"] == 1
