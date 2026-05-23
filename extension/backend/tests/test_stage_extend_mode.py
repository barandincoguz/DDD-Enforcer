"""WP-CORE-20c — StageEmitter.stage(extend=True) preserves rerun records.

Closes the documented follow-up from WP-CORE-20b: re-entering
`emitter.stage("architect")` during the WP-CORE-7 feedback rerun
previously overwrote `manifest.stages["architect"]`, dropping the
first-attempt llm_calls + json_parse_failures records. Paper-data
fidelity demands those records survive.

Design (extend mode):
* `stage(name, extend=False)` — default behavior unchanged: each enter
  creates a fresh StageRecord overwriting any prior entry under `name`.
* `stage(name, extend=True)` — if `manifest.stages[name]` already
  exists, REUSE it: prior llm_calls / json_parse_failures lists stay,
  new ones append. On exit:
  - `record.metrics["attempts"]` (List[Dict]) gains a per-attempt
    summary `{started_at, ended_at, elapsed_ms, status,
    llm_calls_added}`.
  - `record.elapsed_ms` becomes cumulative across all attempts.
  - `record.ended_at` reflects the final attempt's exit.
  - `record.started_at` is preserved from the FIRST attempt (never
    overwritten).
  - `record.status` reflects the FINAL attempt's outcome.
  - `record.p50_latency_ms` / `p95_latency_ms` recomputed over the
    UNION of llm_calls across attempts.
* No prior record + `extend=True` → behaves identically to
  `extend=False` (fresh record, but `record.metrics["attempts"]` is
  still populated with the single-attempt summary so consumers see a
  uniform shape after rerun-vs-no-rerun runs).

Run: pytest tests/test_stage_extend_mode.py -v
"""
from __future__ import annotations

import time

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_llm_response(*, latency_ms: float = 100.0, json_failed: bool = False):
    from core.llm.base import LLMResponse, TokenUsage
    return LLMResponse(
        content="{}",
        parsed=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model_id="gemini-3.1-flash-lite",
        provider="gemini",
        json_failed=json_failed,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# T-20c-1: extend=False default replaces prior record (regression guard)
# ---------------------------------------------------------------------------

def test_t_20c_1_extend_false_default_replaces():
    """extend=False (the default) overwrites prior llm_calls on re-enter.

    Locks in the pre-WP-CORE-20c behavior so the default mode does NOT
    regress to extend semantics by accident.
    """
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    with em.stage("architect"):
        em.record_llm_call(_make_llm_response(), operation="attempt-1-call")
    assert len(m.stages["architect"].llm_calls) == 1

    # Re-enter without extend → record replaced; prior llm_calls gone.
    with em.stage("architect"):
        em.record_llm_call(_make_llm_response(), operation="attempt-2-call")
    assert len(m.stages["architect"].llm_calls) == 1
    assert m.stages["architect"].llm_calls[0].operation == "attempt-2-call"


# ---------------------------------------------------------------------------
# T-20c-2: extend=True with NO prior record → fresh record, attempts seeded
# ---------------------------------------------------------------------------

def test_t_20c_2_extend_true_no_prior_record():
    """Single-attempt extend=True still records a 1-entry attempts list."""
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    with em.stage("architect", extend=True):
        em.record_llm_call(_make_llm_response(latency_ms=50.0), operation="solo")

    rec = m.stages["architect"]
    assert len(rec.llm_calls) == 1
    attempts = rec.metrics.get("attempts", [])
    assert isinstance(attempts, list)
    assert len(attempts) == 1
    assert attempts[0]["status"] == "success"
    assert attempts[0]["llm_calls_added"] == 1
    assert attempts[0]["started_at"]
    assert attempts[0]["ended_at"]


# ---------------------------------------------------------------------------
# T-20c-3: extend=True preserves prior llm_calls on re-enter
# ---------------------------------------------------------------------------

def test_t_20c_3_extend_true_preserves_prior_llm_calls():
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    with em.stage("architect"):
        em.record_llm_call(_make_llm_response(), operation="attempt-1-call")

    # Re-enter WITH extend=True → prior llm_calls survive.
    with em.stage("architect", extend=True):
        em.record_llm_call(_make_llm_response(), operation="attempt-2-call")

    rec = m.stages["architect"]
    assert len(rec.llm_calls) == 2
    ops = [c.operation for c in rec.llm_calls]
    assert ops == ["attempt-1-call", "attempt-2-call"]


# ---------------------------------------------------------------------------
# T-20c-4: extend=True appends per-attempt summary
# ---------------------------------------------------------------------------

def test_t_20c_4_extend_true_appends_attempt_summary():
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    # First attempt: NO extend (or extend=True with no prior — either seeds
    # the attempts list to length 1).
    with em.stage("architect", extend=True):
        em.record_llm_call(_make_llm_response(), operation="a1-c1")
        em.record_llm_call(_make_llm_response(), operation="a1-c2")

    # Second attempt: extend=True → appends to the attempts list.
    with em.stage("architect", extend=True):
        em.record_llm_call(_make_llm_response(), operation="a2-c1")

    rec = m.stages["architect"]
    attempts = rec.metrics["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["llm_calls_added"] == 2
    assert attempts[1]["llm_calls_added"] == 1
    assert all(a["started_at"] for a in attempts)
    assert all(a["ended_at"] for a in attempts)


# ---------------------------------------------------------------------------
# T-20c-5: extend=True cumulative elapsed_ms across attempts
# ---------------------------------------------------------------------------

def test_t_20c_5_extend_true_cumulative_elapsed_ms():
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    with em.stage("architect", extend=True):
        time.sleep(0.01)
    elapsed_after_first = m.stages["architect"].elapsed_ms
    assert elapsed_after_first > 0

    with em.stage("architect", extend=True):
        time.sleep(0.01)
    elapsed_after_second = m.stages["architect"].elapsed_ms
    # Cumulative: second total must exceed first.
    assert elapsed_after_second > elapsed_after_first


# ---------------------------------------------------------------------------
# T-20c-6: extend=True preserves first attempt's started_at
# ---------------------------------------------------------------------------

def test_t_20c_6_extend_true_preserves_first_started_at():
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    with em.stage("architect", extend=True):
        pass
    first_started = m.stages["architect"].started_at
    first_ended = m.stages["architect"].ended_at
    assert first_started is not None
    assert first_ended is not None

    # Force a measurable wall-clock gap before the next attempt.
    time.sleep(0.005)

    with em.stage("architect", extend=True):
        pass

    rec = m.stages["architect"]
    # started_at must NOT advance to the second attempt's start.
    assert rec.started_at == first_started
    # ended_at must reflect the LAST attempt.
    assert rec.ended_at is not None
    assert rec.ended_at >= first_ended


# ---------------------------------------------------------------------------
# T-20c-7: extend=True final attempt's exception sets status=fail
# ---------------------------------------------------------------------------

def test_t_20c_7_extend_true_final_status_wins():
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    # First attempt succeeds.
    with em.stage("architect", extend=True):
        pass
    assert m.stages["architect"].status == "success"

    # Second attempt raises → status flips to fail, but llm_calls + first
    # attempt's summary survive.
    with pytest.raises(RuntimeError):
        with em.stage("architect", extend=True):
            raise RuntimeError("rerun boom")
    rec = m.stages["architect"]
    assert rec.status == "fail"
    assert len(rec.metrics["attempts"]) == 2
    assert rec.metrics["attempts"][0]["status"] == "success"
    assert rec.metrics["attempts"][1]["status"] == "fail"


# ---------------------------------------------------------------------------
# T-20c-8: extend=True recomputes p50/p95 over union of llm_calls
# ---------------------------------------------------------------------------

def test_t_20c_8_extend_true_recomputes_percentiles_over_union():
    from core.observability import RunManifest, StageEmitter
    m = RunManifest()
    em = StageEmitter(m)

    with em.stage("architect", extend=True):
        em.record_llm_call(_make_llm_response(latency_ms=10.0), operation="a1-c1")

    with em.stage("architect", extend=True):
        em.record_llm_call(_make_llm_response(latency_ms=20.0), operation="a2-c1")
        em.record_llm_call(_make_llm_response(latency_ms=30.0), operation="a2-c2")

    rec = m.stages["architect"]
    # p50 of [10, 20, 30] under inclusive 100-quantile = 20.0.
    # p95 must exceed p50.
    assert rec.p50_latency_ms == pytest.approx(20.0, rel=0.05)
    assert rec.p95_latency_ms >= rec.p50_latency_ms


# ---------------------------------------------------------------------------
# T-20c-9: _optional_stage(name, extend=True) passes through to emitter
# ---------------------------------------------------------------------------

def test_t_20c_9_optional_stage_passes_extend_through():
    from core.observability import RunManifest, StageEmitter
    from core.observability.emitter import _emitter_var
    from core.orchestration.pipeline import _optional_stage

    m = RunManifest()
    em = StageEmitter(m)
    token = _emitter_var.set(em)
    try:
        with _optional_stage("architect"):
            em.record_llm_call(_make_llm_response(), operation="a1")
        with _optional_stage("architect", extend=True):
            em.record_llm_call(_make_llm_response(), operation="a2")
    finally:
        _emitter_var.reset(token)

    rec = m.stages["architect"]
    assert len(rec.llm_calls) == 2
    assert len(rec.metrics["attempts"]) == 2


# ---------------------------------------------------------------------------
# T-20c-10: pipeline architect rerun preserves first-attempt llm_calls
# ---------------------------------------------------------------------------

def test_t_20c_10_pipeline_architect_rerun_preserves_first_attempt():
    """Integration: run_pipeline's architect rerun branch must extend the
    prior architect StageRecord rather than overwriting it."""
    from unittest.mock import MagicMock
    from core.observability import RunManifest, StageEmitter
    from core.observability.emitter import _emitter_var
    from core.orchestration.pipeline import run_pipeline, PipelineDeps
    from core.pipeline_contracts import (
        ScoutOutput,
        ArchitectOutput,
        ContextHypothesis,
        SpecialistAnalysis,
        SectionedSentence,
        ChunkMetadata,
        VerifierResult,
        VerifierIssue,
    )
    from core.schemas import Entity
    from core.synthesizer import synthesize_domain_model

    arch_call_log: list[str] = []

    def scout_fn(_text):
        return ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order is placed.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=20),
        )

    def architect_fn(_scout):
        arch_call_log.append("architect_first")
        em = _emitter_var.get()
        if em is not None:
            em.record_llm_call(
                _make_llm_response(latency_ms=11.0),
                operation="architect_attempt_1",
            )
        return ArchitectOutput(
            contexts=[
                ContextHypothesis(context_name="OrdersCtx", description="orders bc")
            ]
        )

    def architect_with_feedback_fn(_scout, _issues):
        arch_call_log.append("architect_rerun")
        em = _emitter_var.get()
        if em is not None:
            em.record_llm_call(
                _make_llm_response(latency_ms=22.0),
                operation="architect_attempt_2",
            )
        return ArchitectOutput(
            contexts=[
                ContextHypothesis(context_name="OrdersCtx", description="orders bc")
            ]
        )

    def specialist_fn(arch, _scout):
        return [
            SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[
                    Entity(
                        name="Order",
                        description="A purchase order.",
                        confidence=0.9,
                        justification="from sentence 0",
                        evidence_sentence_indices=[0],
                    )
                ],
            )
        ]

    def synthesizer_fn(analyses):
        return synthesize_domain_model(
            analyses,
            llm_client=MagicMock(),
            project_name="Test",
            skip_enrich=True,
        )

    call_n = {"n": 0}

    def verifier_fn(_snapshot):
        call_n["n"] += 1
        if call_n["n"] == 1:
            # First verifier call: one architect-stage ERROR triggers
            # architect_with_feedback rerun.
            return VerifierResult(
                ok=False,
                issues=[
                    VerifierIssue(
                        severity="ERROR",
                        check_id="D1",
                        target="architect:OrdersCtx",
                        message="forced rerun",
                    )
                ],
            )
        return VerifierResult(ok=True, issues=[])

    deps = PipelineDeps(
        scout=scout_fn,
        architect=architect_fn,
        architect_with_feedback=architect_with_feedback_fn,
        specialist=specialist_fn,
        synthesizer=synthesizer_fn,
        verifier=verifier_fn,
    )

    m = RunManifest()
    em = StageEmitter(m)
    token = _emitter_var.set(em)
    try:
        model = run_pipeline(srs_text="x", deps=deps)
    finally:
        _emitter_var.reset(token)

    assert model.bounded_contexts
    assert arch_call_log == ["architect_first", "architect_rerun"]

    arch_rec = m.stages["architect"]
    ops = [c.operation for c in arch_rec.llm_calls]
    assert ops == ["architect_attempt_1", "architect_attempt_2"]
    attempts = arch_rec.metrics["attempts"]
    assert len(attempts) == 2
