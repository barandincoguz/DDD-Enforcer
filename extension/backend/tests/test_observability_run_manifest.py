"""WP-CORE-20 — RunManifest Pydantic model tests (RED phase).

Covers spec §6.1 acceptance:
- Construction with sensible defaults
- model_dump_json / model_validate_json round-trip
- Every `outcome` literal accepted
- errors[] append behavior
- instrumentation_overhead_ms + monotonic_clock_source present
"""
from __future__ import annotations

import json

import pytest


def test_t_manifest_1_default_construction():
    """T-MANIFEST-1: RunManifest constructs with defaults; required fields auto-populated."""
    from core.observability import RunManifest

    m = RunManifest()
    assert m.schema_version == "1.0"
    assert m.min_supported_schema == "1.0"
    assert m.run_id  # uuid4 string
    assert m.started_at  # iso8601
    assert m.ended_at is None
    assert m.outcome == "in_progress"
    assert m.monotonic_clock_source == "time.monotonic_ns"
    assert m.stages == {}
    assert m.errors == []


def test_t_manifest_2_json_round_trip():
    """T-MANIFEST-2: dump→load round-trip preserves all fields."""
    from core.observability import RunManifest

    m = RunManifest(outcome="success")
    m.elapsed_ms = 1234.5
    m.instrumentation_overhead_ms = 12.3
    blob = m.model_dump_json()
    loaded = RunManifest.model_validate_json(blob)
    assert loaded.run_id == m.run_id
    assert loaded.outcome == "success"
    assert loaded.elapsed_ms == 1234.5
    assert loaded.instrumentation_overhead_ms == 12.3


@pytest.mark.parametrize("outcome", [
    "in_progress", "success",
    "no_input_files", "srs_parse_failed", "all_srs_empty",
    "architect_grounding_error", "refinement_exhausted",
    "synthesizer_empty_model", "pipeline_error",
    "output_write_failed", "unexpected_error",
])
def test_t_manifest_3_every_outcome_literal_accepted(outcome):
    """T-MANIFEST-3: every spec-declared outcome value validates (covers C-3 enum)."""
    from core.observability import RunManifest

    m = RunManifest(outcome=outcome)
    assert m.outcome == outcome


def test_t_manifest_4_invalid_outcome_rejected():
    """T-MANIFEST-4: outcome outside the literal set raises ValidationError."""
    from pydantic import ValidationError

    from core.observability import RunManifest

    with pytest.raises(ValidationError):
        RunManifest(outcome="invalid_made_up_outcome")


def test_t_manifest_5_errors_append():
    """T-MANIFEST-5: errors[] is a list of dicts; append works without re-validation."""
    from core.observability import RunManifest

    m = RunManifest()
    m.errors.append({
        "timestamp": "2026-05-23T00:00:00Z",
        "type": "ArchitectGroundingError",
        "stage": "architect",
        "message": "test",
        "srs_path": "/tmp/srs.txt",
        "context": {"residual_issues": 3},
    })
    blob = m.model_dump_json()
    assert "ArchitectGroundingError" in blob
    assert "residual_issues" in blob


def test_t_manifest_6_stage_record_round_trip():
    """T-MANIFEST-6: nested StageRecord survives dump/load with llm_calls list."""
    from core.observability import RunManifest
    from core.observability.run_manifest import LLMCallRecord, StageRecord

    m = RunManifest()
    rec = StageRecord(status="success")
    rec.llm_calls.append(LLMCallRecord(
        timestamp="2026-05-23T00:00:00Z",
        stage="architect",
        operation="identify_contexts attempt-1",
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        prompt_tokens=100,
        completion_tokens=50,
        cached_tokens=0,
        cost_usd=0.0012,
        latency_ms=523.0,
        json_failed=False,
        json_fail_reason=None,
        is_retry_exhausted=False,
    ))
    m.stages["architect"] = rec
    loaded = RunManifest.model_validate_json(m.model_dump_json())
    assert loaded.stages["architect"].llm_calls[0].model_id == "gemini-3.1-pro-preview"
    assert loaded.stages["architect"].llm_calls[0].latency_ms == 523.0


def test_t_manifest_7_llm_aggregate_default():
    """T-MANIFEST-7: top-level llm aggregate exists with zeroed counters."""
    from core.observability import RunManifest

    m = RunManifest()
    assert m.llm.total_calls == 0
    assert m.llm.json_failed_count == 0
    assert m.llm.json_parse_failure_count == 0
    assert m.llm.json_failed_total_count == 0
    assert m.llm.json_failed_rate == 0.0
    assert m.llm.retry_exhausted_count == 0
    assert m.llm.by_model == {}
    assert m.llm.by_stage == {}
