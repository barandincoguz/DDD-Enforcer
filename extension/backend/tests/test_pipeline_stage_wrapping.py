"""WP-CORE-20b — Wrap architect-pipeline stage calls in emitter.stage scopes.

Closes the WP-CORE-20a follow-up gap (symmetric to WP-CORE-26's
caller-side `_record_json_parse_failure_if_emitter`): Scout / Architect
/ Specialist / Synthesizer dispatch points inside
`core.orchestration.pipeline.run_pipeline` previously executed OUTSIDE
any `with emitter.stage(...)` scope.  The downstream effects:

* `StageEmitter.record_llm_call` silently dropped every per-call record
  because `_stage_var.get()` returned None.
* `manifest.stages["scout"]` / "architect" / "specialist" /
  "synthesizer" were only ever populated indirectly (when one of the
  WP-CORE-26 callers manually toggled the stage var), so the per-stage
  manifest view was structurally incomplete.

This RED test suite locks the new wiring in:

1. Inside each stub stage function, `_stage_var.get()` returns the
   stage name the orchestrator is currently dispatching.
2. After `run_pipeline` returns, `manifest.stages` contains a
   `StageRecord` for each wrapped stage with `elapsed_ms > 0`.
3. Failure inside a wrapped call sets the stage record's status to
   "fail" AND appends a structured entry to `manifest.errors`.
4. The architect-feedback rerun path (WP-CORE-7) executes inside the
   "architect" scope, not outside it.
5. The wrapping is opt-in via `_emitter_var`; runs without an active
   emitter (most unit tests, CLI tooling) still complete without crash.

The wrapping is implemented as a single internal context manager
`_optional_stage(name)` that yields an `emitter.stage` scope when an
emitter is active and a no-op otherwise.  This keeps every existing
emitter-less call site working unchanged.
"""
from __future__ import annotations

import sys
from typing import List
from unittest.mock import MagicMock
import unittest


from core.observability.emitter import StageEmitter, _emitter_var, _stage_var
from core.observability.run_manifest import RunManifest
from core.orchestration.pipeline import PipelineDeps, run_pipeline
from core.pipeline_contracts import (
    ArchitectOutput,
    ChunkMetadata,
    ContextHypothesis,
    ScoutOutput,
    SectionedSentence,
    SpecialistAnalysis,
    VerifierResult as ContractVerifierResult,
    VerifierIssue as ContractVerifierIssue,
)
from core.schemas import DomainModel, Entity


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_emitter() -> StageEmitter:
    return StageEmitter(RunManifest(run_id="test-run", started_at="2026-05-23T00:00:00Z"))


def _make_deps(captured: List[tuple]) -> PipelineDeps:
    """PipelineDeps whose stubs append `(stage_label, _stage_var.get())` to
    `captured` on each invocation. This is the test contract: each stub
    must run inside the orchestrator's emitter.stage(...) scope and the
    ContextVar must read back the expected stage name.
    """

    def scout_fn(srs_text: str) -> ScoutOutput:
        captured.append(("scout", _stage_var.get()))
        return ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order is placed by a customer.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=45),
        )

    def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
        captured.append(("architect", _stage_var.get()))
        return ArchitectOutput(
            contexts=[ContextHypothesis(context_name="OrderMgmt", description="Manages orders")],
        )

    def architect_with_feedback_fn(scout: ScoutOutput, issues) -> ArchitectOutput:
        captured.append(("architect_feedback", _stage_var.get()))
        return architect_fn(scout)

    def specialist_fn(arch: ArchitectOutput, scout: ScoutOutput) -> List[SpecialistAnalysis]:
        captured.append(("specialist", _stage_var.get()))
        return [
            SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="A purchase order placed by a customer.",
                    confidence=0.9,
                    justification="Cited in sentence 0",
                    evidence_sentence_indices=[0],
                )],
            )
        ]

    def synthesizer_fn(analyses):
        captured.append(("synthesizer", _stage_var.get()))
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(
            analyses,
            llm_client=MagicMock(),
            project_name="Test",
            skip_enrich=True,
        )

    def verifier_fn(snapshot):
        return ContractVerifierResult(ok=True, issues=[])

    return PipelineDeps(
        scout=scout_fn,
        architect=architect_fn,
        architect_with_feedback=architect_with_feedback_fn,
        specialist=specialist_fn,
        synthesizer=synthesizer_fn,
        verifier=verifier_fn,
    )


# ---------------------------------------------------------------------------
# T-20b-1 .. T-20b-4: stage-var visibility inside stub calls
# ---------------------------------------------------------------------------


class TestStageVarVisibility(unittest.TestCase):
    def test_t_20b_1_scout_stage_var_set_during_call(self) -> None:
        emitter = _make_emitter()
        captured: List[tuple] = []
        token = _emitter_var.set(emitter)
        try:
            run_pipeline(srs_text="Sample SRS", deps=_make_deps(captured))
        finally:
            _emitter_var.reset(token)
        scout_entries = [entry for entry in captured if entry[0] == "scout"]
        self.assertEqual(len(scout_entries), 1)
        self.assertEqual(scout_entries[0][1], "scout")

    def test_t_20b_2_architect_stage_var_set_during_call(self) -> None:
        emitter = _make_emitter()
        captured: List[tuple] = []
        token = _emitter_var.set(emitter)
        try:
            run_pipeline(srs_text="Sample SRS", deps=_make_deps(captured))
        finally:
            _emitter_var.reset(token)
        architect_entries = [entry for entry in captured if entry[0] == "architect"]
        self.assertGreaterEqual(len(architect_entries), 1)
        self.assertEqual(architect_entries[0][1], "architect")

    def test_t_20b_3_specialist_stage_var_set_during_call(self) -> None:
        emitter = _make_emitter()
        captured: List[tuple] = []
        token = _emitter_var.set(emitter)
        try:
            run_pipeline(srs_text="Sample SRS", deps=_make_deps(captured))
        finally:
            _emitter_var.reset(token)
        specialist_entries = [entry for entry in captured if entry[0] == "specialist"]
        self.assertEqual(specialist_entries[0][1], "specialist")

    def test_t_20b_4_synthesizer_stage_var_set_during_call(self) -> None:
        emitter = _make_emitter()
        captured: List[tuple] = []
        token = _emitter_var.set(emitter)
        try:
            run_pipeline(srs_text="Sample SRS", deps=_make_deps(captured))
        finally:
            _emitter_var.reset(token)
        synthesizer_entries = [entry for entry in captured if entry[0] == "synthesizer"]
        self.assertEqual(synthesizer_entries[0][1], "synthesizer")


# ---------------------------------------------------------------------------
# T-20b-5: manifest.stages populated for all wrapped stages
# ---------------------------------------------------------------------------


class TestStageRecordsPopulated(unittest.TestCase):
    def test_t_20b_5_manifest_stages_contain_all_wrapped_stages(self) -> None:
        emitter = _make_emitter()
        captured: List[tuple] = []
        token = _emitter_var.set(emitter)
        try:
            run_pipeline(srs_text="Sample SRS", deps=_make_deps(captured))
        finally:
            _emitter_var.reset(token)
        for stage in ("scout", "architect", "specialist", "synthesizer"):
            self.assertIn(stage, emitter.manifest.stages, f"{stage} missing")
            record = emitter.manifest.stages[stage]
            self.assertEqual(record.status, "success")
            self.assertGreaterEqual(record.elapsed_ms, 0.0)


# ---------------------------------------------------------------------------
# T-20b-6: emitter-less context still runs without crash
# ---------------------------------------------------------------------------


class TestEmitterlessContext(unittest.TestCase):
    def test_t_20b_6_no_active_emitter_no_crash(self) -> None:
        """A run with no `_emitter_var` set completes; `_stage_var` reads
        None inside stubs and no manifest is mutated."""
        captured: List[tuple] = []
        result = run_pipeline(srs_text="Sample SRS", deps=_make_deps(captured))
        self.assertIsInstance(result, DomainModel)
        for entry in captured:
            self.assertIsNone(entry[1])  # _stage_var.get() returns None


# ---------------------------------------------------------------------------
# T-20b-7: failure inside a wrapped stage marks the record + records error
# ---------------------------------------------------------------------------


class TestStageFailureRecording(unittest.TestCase):
    def test_t_20b_7_specialist_exception_marks_stage_fail_and_records_error(self) -> None:
        emitter = _make_emitter()
        captured: List[tuple] = []
        deps = _make_deps(captured)

        def boom(arch, scout):
            captured.append(("specialist", _stage_var.get()))
            raise RuntimeError("specialist crash")

        deps.specialist = boom  # type: ignore[assignment]

        token = _emitter_var.set(emitter)
        try:
            with self.assertRaises(RuntimeError):
                run_pipeline(srs_text="Sample SRS", deps=deps)
        finally:
            _emitter_var.reset(token)

        self.assertIn("specialist", emitter.manifest.stages)
        self.assertEqual(emitter.manifest.stages["specialist"].status, "fail")
        # An error entry was appended for the failing stage.
        self.assertTrue(
            any(e.get("stage") == "specialist" for e in emitter.manifest.errors),
            f"no specialist error in {emitter.manifest.errors}",
        )


# ---------------------------------------------------------------------------
# T-20b-8: architect rerun via feedback path still wraps in "architect"
# ---------------------------------------------------------------------------


class TestArchitectFeedbackRerunWrapped(unittest.TestCase):
    def test_t_20b_8_architect_with_feedback_runs_inside_architect_scope(self) -> None:
        """When architect-stage verifier ERRORs trigger
        architect_with_feedback, that call must also execute inside the
        "architect" scope so its LLM records land in the right bucket."""
        emitter = _make_emitter()
        captured: List[tuple] = []
        deps = _make_deps(captured)

        # First verifier call returns one architect-stage ERROR (forces rerun);
        # second call returns ok.
        from core.verifier.types import VerifierIssue, IssueSeverity  # noqa
        rerun_state = {"calls": 0}

        def verifier_fn(snapshot):
            rerun_state["calls"] += 1
            if rerun_state["calls"] == 1:
                return ContractVerifierResult(
                    ok=False,
                    issues=[
                        ContractVerifierIssue(
                            severity="ERROR",
                            check_id="D1",
                            # Stage-prefix convention from
                            # core.orchestration.pipeline._KNOWN_STAGE_PREFIXES:
                            # the orchestrator routes by `target` prefix.
                            target="architect:OrderMgmt",
                            message="forced rerun",
                        )
                    ],
                )
            return ContractVerifierResult(ok=True, issues=[])

        deps.verifier = verifier_fn  # type: ignore[assignment]

        token = _emitter_var.set(emitter)
        try:
            run_pipeline(srs_text="Sample SRS", deps=deps)
        finally:
            _emitter_var.reset(token)

        feedback_entries = [e for e in captured if e[0] == "architect_feedback"]
        self.assertGreaterEqual(len(feedback_entries), 1)
        self.assertEqual(feedback_entries[0][1], "architect")


if __name__ == "__main__":
    unittest.main()
