"""Smoke tests for core.llm.registry — 6-model D1 lockdown."""

import pytest
from core.llm.registry import MODELS, ModelSpec, model_spec, models_by_provider


def test_registry_has_all_six_d1_models():
    expected = {
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite",
        "gpt-oss:120b-cloud",
        "qwen3-coder-next:cloud",
        "minimax-m2:cloud",
        "gemma4:31b-cloud",
    }
    assert set(MODELS.keys()) == expected


def test_registry_provider_split_is_2_gemini_4_ollama():
    assert len(models_by_provider("gemini")) == 2
    assert len(models_by_provider("ollama")) == 4


def test_model_spec_lookup_returns_spec():
    spec = model_spec("gemini-3.1-pro-preview")
    assert isinstance(spec, ModelSpec)
    assert spec.provider == "gemini"
    assert spec.context_window == 1_000_000


def test_model_spec_unknown_raises_keyerror():
    with pytest.raises(KeyError):
        model_spec("nonexistent-model")


def test_gemini_pro_cost_is_tiered():
    spec = model_spec("gemini-3.1-pro-preview")
    short_cost = spec.pricing.cost_for(prompt_tokens=10_000, completion_tokens=1_000)
    long_cost = spec.pricing.cost_for(prompt_tokens=500_000, completion_tokens=1_000)
    assert long_cost > short_cost, "long-context Pro tier should cost more per token"


def test_ollama_models_are_subscription_flat():
    for mid, spec in models_by_provider("ollama").items():
        assert spec.compute_mode == "subscription_flat", mid
        # Subscription bills flat; per-token reported cost is zero.
        assert spec.pricing.cost_for(prompt_tokens=1_000, completion_tokens=1_000) == 0.0


def test_all_six_models_support_json_schema():
    for spec in MODELS.values():
        assert spec.supports_json_schema, spec.model_id
