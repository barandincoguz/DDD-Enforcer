"""
API contract tests aligned to the current backend endpoint shapes.

These tests avoid live network calls by invoking endpoint functions directly
with controlled app state.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class FakeValidationService:
    def validate(self, **kwargs):
        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "BannedTermViolation",
                    "message": "Class name 'ClientManager' contains a banned term 'manager'.",
                    "suggestion": "Rename using approved domain terminology.",
                    "sources": [{"section": "Glossary", "document": "srs.txt"}],
                }
            ],
            "mode": kwargs.get("mode", "pipeline"),
            "metrics": {
                "validation_time_ms": 10.0,
                "file_size_chars": len(kwargs.get("content", "")),
                "file_loc": 2,
                "code_file_tokens": 10,
                "stage_latencies_ms": {
                    "ast_parse": 1.0,
                    "deterministic_rules": 1.0,
                    "advanced_llm": 5.0,
                    "rag": 2.0,
                    "total": 10.0,
                },
                "provider": "static-json",
                "model": "static-model",
                "llm_input_tokens": 20,
                "llm_output_tokens": 10,
                "llm_total_tokens": 30,
                "cached_tokens": 0,
                "cost_usd": 0.001,
                "api_calls": 1,
                "parseable_outputs": 1,
                "unparseable_outputs": 0,
            },
        }


class FakeGenerationService:
    def generate_from_files(self, **kwargs):
        return {
            "model": {
                "project_name": "Sample",
                "project_metadata": {"version": "1.0", "generated_at": "2026-03-12"},
                "bounded_contexts": [],
                "global_rules": {"banned_global_terms": []},
            },
            "project_name": "Sample",
            "bounded_contexts_count": 0,
            "stage_latencies_ms": {
                "Scout": 10.0,
                "Architect": 5.0,
                "Specialist": 7.0,
                "Synthesizer": 4.0,
                "total": 26.0,
            },
            "total_latency_ms": 26.0,
            "metrics": {
                "provider": "static-json",
                "model": "static-model",
                "llm_input_tokens": 100,
                "llm_output_tokens": 50,
                "llm_total_tokens": 150,
                "cached_tokens": 0,
                "cost_usd": 0.01,
                "api_calls": 4,
                "parseable_outputs": 4,
                "unparseable_outputs": 0,
            },
            "documents": [{"path": "srs.txt", "chars": 1200}],
        }

    def initialize_rag(self, file_paths):
        return FakeRag()


class FakeRag:
    def get_stats(self):
        return {"collection_name": "srs", "total_chunks": 5}

    def search(self, query: str, n_results: int = 5):
        return [{"text": query, "relevance": 0.9}] * n_results


@pytest.fixture
def api_state(monkeypatch):
    import main

    main.app_state.clear()
    main.app_state.update(
        {
            "domain_rules": {
                "project_name": "Sample",
                "bounded_contexts": [],
                "global_rules": {"banned_global_terms": []},
            },
            "domain_model_tokens": 42,
            "validation_service": FakeValidationService(),
            "generation_service": FakeGenerationService(),
            "rag": FakeRag(),
        }
    )
    monkeypatch.setattr(main, "_count_domain_model_tokens", lambda rules: 42)
    return main


def test_health_endpoint_contract(api_state):
    payload = api_state.health_check()
    assert payload["status"] == "healthy"
    assert payload["domain_model_loaded"] is True
    assert payload["rag_initialized"] is True


def test_generate_model_endpoint_contract(api_state, tmp_path):
    output_path = tmp_path / "model.json"
    payload = api_state.generate_model_endpoint(
        api_state.GenerateModelRequest(
            file_paths=["/tmp/srs.txt"],
            output_path=str(output_path),
        )
    )
    assert payload["success"] is True
    assert payload["model_path"] == str(output_path)
    assert "domain_model" in payload
    assert "stage_latencies_ms" in payload
    assert payload["metrics"]["api_calls"] == 4


def test_validate_endpoint_contract(api_state):
    payload = api_state.validate_code(
        api_state.CodeSubmission(
            filename="sample.py",
            content="class ClientManager:\n    pass\n",
        )
    )
    assert payload["is_violation"] is True
    assert payload["mode"] == "pipeline"
    assert payload["metrics"]["provider"] == "static-json"
    assert payload["metrics"]["stage_latencies_ms"]["total"] == 10.0


def test_rag_search_uses_n_results(api_state):
    payload = api_state.search_documents(query="Order", n_results=3)
    assert len(payload) == 3
    assert payload[0]["text"] == "Order"


def test_combined_metrics_contract(api_state):
    payload = api_state.get_combined_metrics()
    assert "domain_model" in payload
    assert "token_usage" in payload
    assert "validation_metrics" in payload
    assert "research_metrics" in payload
    assert "pricing_reference" in payload
