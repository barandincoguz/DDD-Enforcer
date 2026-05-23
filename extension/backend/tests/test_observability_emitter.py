"""WP-CORE-20 — StageEmitter tests (RED phase).

Covers spec §6.2 + §6.6 + Codex C-1 + C-4:
- Stage lifecycle context manager
- Exception path sets status=fail and appends to errors[]
- record_llm_call aggregation
- record_json_parse_failure (Codex C-4)
- ContextVar set/reset
- Parallel-Scout contextvars propagation (T-EMITTER-PARALLEL-1)
"""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest


def _make_llm_response(*, json_failed: bool = False, json_fail_reason=None,
                       latency_ms: float = 100.0):
    """Helper: build a real LLMResponse for emitter integration."""
    from core.llm.base import LLMResponse, TokenUsage
    return LLMResponse(
        content="{}",
        parsed=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model_id="gemini-3.1-flash-lite",
        provider="gemini",
        json_failed=json_failed,
        json_fail_reason=json_fail_reason,
        latency_ms=latency_ms,
    )


def test_t_emitter_1_stage_lifecycle_success():
    """T-EMITTER-1: with emitter.stage(name) records start, end, elapsed_ms; status=success."""
    from core.observability import RunManifest, StageEmitter

    m = RunManifest()
    em = StageEmitter(m)
    with em.stage("architect") as rec:
        rec.metrics["attempts"] = 1
    assert "architect" in m.stages
    assert m.stages["architect"].status == "success"
    assert m.stages["architect"].started_at
    assert m.stages["architect"].ended_at
    assert m.stages["architect"].elapsed_ms >= 0.0
    assert m.stages["architect"].metrics["attempts"] == 1


def test_t_emitter_2_stage_exception_path():
    """T-EMITTER-2: raising inside stage block sets status=fail and appends to errors."""
    from core.observability import RunManifest, StageEmitter

    m = RunManifest()
    em = StageEmitter(m)
    with pytest.raises(RuntimeError):
        with em.stage("architect"):
            raise RuntimeError("test boom")
    assert m.stages["architect"].status == "fail"
    assert len(m.errors) == 1
    assert m.errors[0]["type"] == "RuntimeError"
    assert m.errors[0]["stage"] == "architect"
    assert "test boom" in m.errors[0]["message"]


def test_t_emitter_3_record_llm_call_aggregates():
    """T-EMITTER-3: record_llm_call appends to stage.llm_calls and bumps llm.by_stage/by_model."""
    from core.observability import RunManifest, StageEmitter

    m = RunManifest()
    em = StageEmitter(m)
    with em.stage("scout"):
        em.record_llm_call(_make_llm_response(), operation="chunk_1")
        em.record_llm_call(_make_llm_response(latency_ms=200.0), operation="chunk_2")
    assert len(m.stages["scout"].llm_calls) == 2
    assert m.llm.total_calls == 2
    assert "scout" in m.llm.by_stage
    assert m.llm.by_stage["scout"]["calls"] == 2
    assert "gemini-3.1-flash-lite" in m.llm.by_model


def test_t_emitter_4_record_llm_call_outside_stage_silent_drop():
    """T-EMITTER-4: record_llm_call with no active stage does not raise (CLI / init path)."""
    from core.observability import RunManifest, StageEmitter

    m = RunManifest()
    em = StageEmitter(m)
    em.record_llm_call(_make_llm_response(), operation="orphan")
    assert m.llm.total_calls == 0  # silently dropped


def test_t_emitter_5_record_json_parse_failure_caller_side():
    """T-EMITTER-5: record_json_parse_failure bumps json_parse_failure_count (Codex C-4)."""
    from core.observability import RunManifest, StageEmitter

    m = RunManifest()
    em = StageEmitter(m)
    with em.stage("architect"):
        em.record_json_parse_failure(
            operation="identify_contexts retry-2",
            model_id="gemini-3.1-pro-preview",
            reason="json_parse_failed",
        )
    assert m.llm.json_parse_failure_count == 1
    assert m.llm.json_failed_total_count == 1
    assert m.llm.json_fail_reasons.get("caller_parse", 0) == 1
    assert len(m.stages["architect"].json_parse_failures) == 1


def test_t_emitter_6_provider_json_failed_counted():
    """T-EMITTER-6: LLMResponse.json_failed=True bumps llm.json_failed_count."""
    from core.observability import RunManifest, StageEmitter

    m = RunManifest()
    em = StageEmitter(m)
    with em.stage("architect"):
        em.record_llm_call(
            _make_llm_response(json_failed=True, json_fail_reason="schema_mismatch: ..."),
            operation="identify",
        )
    assert m.llm.json_failed_count == 1
    assert m.llm.json_failed_total_count == 1
    assert m.llm.json_fail_reasons["schema_mismatch"] == 1


def test_t_emitter_7_contextvar_set_and_reset():
    """T-EMITTER-7: get_current_emitter sees the emitter inside stage; None outside."""
    from core.observability import RunManifest, StageEmitter
    from core.observability.emitter import get_current_emitter

    m = RunManifest()
    em = StageEmitter(m)
    assert get_current_emitter() is None
    with em.stage("architect"):
        assert get_current_emitter() is em
    assert get_current_emitter() is None  # reset after stage exit


def test_t_emitter_parallel_1_thread_pool_executor_contextvar_propagation():
    """T-EMITTER-PARALLEL-1: parallel-Scout style ThreadPoolExecutor must propagate emitter
    via contextvars.copy_context().run(...). Codex C-1 regression guard."""
    from core.observability import RunManifest, StageEmitter
    from core.observability.emitter import get_current_emitter

    m = RunManifest()
    em = StageEmitter(m)

    results = []

    def worker(i):
        # The worker should see the same emitter when the parent uses copy_context.
        e = get_current_emitter()
        if e is not None:
            with e._lock if False else _noop():
                pass
            e.record_llm_call(_make_llm_response(latency_ms=10.0 + i), operation=f"chunk_{i}")
        results.append(e is em)
        return i

    def _noop():
        from contextlib import nullcontext
        return nullcontext()

    with em.stage("scout"):
        with ThreadPoolExecutor(max_workers=2) as ex:
            def run_in_ctx(i):
                ctx = contextvars.copy_context()
                return ctx.run(worker, i)
            list(ex.map(run_in_ctx, [0, 1, 2, 3]))

    # All workers saw the same emitter instance.
    assert all(results), f"expected all True, got {results}"
    # All 4 calls were recorded under stage=scout.
    assert len(m.stages["scout"].llm_calls) == 4
    assert m.llm.total_calls == 4
    assert m.llm.by_stage["scout"]["calls"] == 4


def test_t_emitter_8_finalize_safely_swallows_write_error(monkeypatch, tmp_path):
    """T-EMITTER-8 / T-MANIFEST-FINALIZE-1 (Codex W-2): finalize wrapper swallows write
    errors and never re-raises, even when an original exception is in flight."""
    from core.observability import RunManifest
    from core.observability.emitter import _finalize_manifest_safely

    m = RunManifest(outcome="architect_grounding_error")

    def _broken_write(_m, _path):
        raise OSError("disk full")

    # Replace the writer with a broken one; finalize must not raise.
    monkeypatch.setattr("core.observability.emitter.write_manifest_atomic", _broken_write)
    # No exception even though writer raises.
    _finalize_manifest_safely(m, original_exc=RuntimeError("pretend pipeline failed"))
