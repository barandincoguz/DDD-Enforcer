"""WP-CORE-30b — specialist_with_feedback orchestration wiring.

Task 1: adds `SpecialistWithFeedbackFn` type alias and
`PipelineDeps.specialist_with_feedback` optional field; rewires
`_re_run_specialist` inside `run_pipeline` to forward
(arch, scout, prev_output, specialist_issues) to the dep when set.

TDD: RED suite written before any production-code change.
"""
from __future__ import annotations

import unittest
from typing import List

from core.orchestration.pipeline import PipelineDeps, run_pipeline
from core.pipeline_contracts import (
    ArchitectOutput,
    ChunkMetadata,
    ContextHypothesis,
    ScoutOutput,
    SectionedSentence,
    SpecialistAnalysis,
    VerifierIssue,
    VerifierResult,
)
from core.schemas import DomainModel, Entity


# ---------------------------------------------------------------------------
# Shared helpers (follow _make_deps style from test_pipeline_stage_wrapping.py)
# ---------------------------------------------------------------------------

def _minimal_scout() -> ScoutOutput:
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="An order is placed by a customer.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=45),
    )


def _minimal_arch() -> ArchitectOutput:
    return ArchitectOutput(
        contexts=[ContextHypothesis(context_name="OrderMgmt", description="Manages orders")],
    )


def _minimal_specialist_output(arch: ArchitectOutput) -> List[SpecialistAnalysis]:
    return [
        SpecialistAnalysis(
            context=arch.contexts[0],
            entities=[
                Entity(
                    name="Order",
                    description="A purchase order.",
                    confidence=0.9,
                    justification="Cited in sentence 0",
                    evidence_sentence_indices=[0],
                )
            ],
        )
    ]


def _make_base_deps(
    *,
    specialist_call_log: List[tuple],
    verifier_fn=None,
    specialist_with_feedback_fn=None,
) -> PipelineDeps:
    """Construct a minimal PipelineDeps whose specialist records calls.

    Args:
        specialist_call_log: mutable list; specialist appends ("specialist", arch, scout).
        verifier_fn: optional override; defaults to always-ok.
        specialist_with_feedback_fn: forwarded to PipelineDeps.specialist_with_feedback.
    """

    def scout_fn(srs_text: str) -> ScoutOutput:
        return _minimal_scout()

    def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
        return _minimal_arch()

    def architect_with_feedback_fn(scout: ScoutOutput, issues) -> ArchitectOutput:
        return _minimal_arch()

    def specialist_fn(arch: ArchitectOutput, scout: ScoutOutput) -> List[SpecialistAnalysis]:
        specialist_call_log.append(("specialist", arch, scout))
        return _minimal_specialist_output(arch)

    def synthesizer_fn(analyses: List[SpecialistAnalysis]) -> DomainModel:
        from unittest.mock import MagicMock
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(
            analyses,
            llm_client=MagicMock(),
            project_name="Test",
            skip_enrich=True,
        )

    default_verifier = verifier_fn or (lambda snapshot: VerifierResult(ok=True, issues=[]))

    kwargs = dict(
        scout=scout_fn,
        architect=architect_fn,
        architect_with_feedback=architect_with_feedback_fn,
        specialist=specialist_fn,
        synthesizer=synthesizer_fn,
        verifier=default_verifier,
    )
    if specialist_with_feedback_fn is not None:
        kwargs["specialist_with_feedback"] = specialist_with_feedback_fn  # type: ignore[arg-type]

    return PipelineDeps(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# T-30b-1: PipelineDeps instantiation with no specialist_with_feedback
# ---------------------------------------------------------------------------


class TestPipelineDepsDefaultNone(unittest.TestCase):
    def test_t_30b_1_default_specialist_with_feedback_is_none(self) -> None:
        """PipelineDeps instantiation without specialist_with_feedback succeeds
        and the field defaults to None (backward-compat)."""
        log: List[tuple] = []
        deps = _make_base_deps(specialist_call_log=log)
        self.assertIsNone(deps.specialist_with_feedback)


# ---------------------------------------------------------------------------
# T-30b-2: PipelineDeps instantiation with explicit specialist_with_feedback
# ---------------------------------------------------------------------------


class TestPipelineDepsExplicitFeedbackFn(unittest.TestCase):
    def test_t_30b_2_explicit_specialist_with_feedback_reads_back(self) -> None:
        """Passing an explicit specialist_with_feedback callable succeeds and
        the field reads back identically."""
        log: List[tuple] = []

        def my_feedback_fn(arch, scout, prev, issues):
            return _minimal_specialist_output(arch)

        deps = _make_base_deps(
            specialist_call_log=log,
            specialist_with_feedback_fn=my_feedback_fn,
        )
        self.assertIs(deps.specialist_with_feedback, my_feedback_fn)


# ---------------------------------------------------------------------------
# T-30b-3: backward-compat — specialist_with_feedback=None falls back to
#           deps.specialist on the refine cycle
# ---------------------------------------------------------------------------


class TestBackwardCompatNoFeedbackFn(unittest.TestCase):
    def test_t_30b_3_fallback_to_specialist_when_feedback_fn_is_none(self) -> None:
        """When specialist_with_feedback is None, the refiner rerun calls
        deps.specialist(arch, scout) and never invokes specialist_with_feedback.

        Setup: verifier returns one specialist-stage ERROR on first call, then
        ok on second.  Expect deps.specialist called EXACTLY TWICE; no feedback fn.
        """
        specialist_calls: List[tuple] = []
        feedback_calls: List[tuple] = []

        verifier_state = {"calls": 0}

        def verifier_fn(snapshot):
            verifier_state["calls"] += 1
            if verifier_state["calls"] == 1:
                return VerifierResult(
                    ok=False,
                    issues=[
                        VerifierIssue(
                            severity="ERROR",
                            check_id="S1",
                            target="specialist:OrderMgmt",
                            message="missing aggregate root",
                        )
                    ],
                )
            return VerifierResult(ok=True, issues=[])

        deps = _make_base_deps(
            specialist_call_log=specialist_calls,
            verifier_fn=verifier_fn,
            specialist_with_feedback_fn=None,
        )

        run_pipeline(srs_text="Sample SRS", deps=deps)

        # specialist must have been called twice: initial + 1 rerun
        self.assertEqual(len(specialist_calls), 2, specialist_calls)
        # feedback fn was never invoked (it is None; no calls accumulated)
        self.assertEqual(len(feedback_calls), 0)


# ---------------------------------------------------------------------------
# T-30b-4: specialist_with_feedback is invoked on refine cycle with correct args
# ---------------------------------------------------------------------------


class TestSpecialistWithFeedbackInvoked(unittest.TestCase):
    def test_t_30b_4_feedback_fn_called_with_correct_args(self) -> None:
        """When specialist_with_feedback is set and the verifier emits a
        specialist-stage ERROR, the rerun must invoke
        specialist_with_feedback(arch, scout, prev_output, specialist_issues_only).

        Setup: verifier emits ONE specialist ERROR + ONE synthesizer WARN on
        first call, then ok on second.

        Assertions:
          (a) specialist_with_feedback called exactly ONCE.
          (b) 4th arg (specialist_issues) contains the specialist ERROR but
              NOT the synthesizer WARN.
          (c) 3rd arg (prev_output) is the List[SpecialistAnalysis] from the
              first specialist call.
        """
        specialist_calls: List[tuple] = []
        feedback_call_log: List[tuple] = []  # (arch, scout, prev, issues)

        first_specialist_output: List[List[SpecialistAnalysis]] = []

        specialist_error = VerifierIssue(
            severity="ERROR",
            check_id="S1",
            target="specialist:OrderMgmt",
            message="missing aggregate root",
        )
        synthesizer_warn = VerifierIssue(
            severity="WARN",
            check_id="D9",
            target="synthesizer:GlobalRules",
            message="missing global rule",
        )

        verifier_state = {"calls": 0}

        def verifier_fn(snapshot):
            verifier_state["calls"] += 1
            if verifier_state["calls"] == 1:
                return VerifierResult(
                    ok=False,
                    issues=[specialist_error, synthesizer_warn],
                )
            return VerifierResult(ok=True, issues=[])

        def feedback_fn(
            arch: ArchitectOutput,
            scout: ScoutOutput,
            prev: List[SpecialistAnalysis],
            issues: List[VerifierIssue],
        ) -> List[SpecialistAnalysis]:
            feedback_call_log.append((arch, scout, prev, issues))
            return _minimal_specialist_output(arch)

        def tracking_specialist(arch, scout):
            out = _minimal_specialist_output(arch)
            specialist_calls.append(("specialist", arch, scout))
            if not first_specialist_output:
                first_specialist_output.append(out)
            return out

        # Build deps manually so we can inject tracking_specialist
        from unittest.mock import MagicMock
        from core.synthesizer import synthesize_domain_model

        deps = PipelineDeps(
            scout=lambda srs_text: _minimal_scout(),
            architect=lambda scout: _minimal_arch(),
            architect_with_feedback=lambda scout, issues: _minimal_arch(),
            specialist=tracking_specialist,
            synthesizer=lambda analyses: synthesize_domain_model(
                analyses, llm_client=MagicMock(), project_name="Test", skip_enrich=True
            ),
            verifier=verifier_fn,
            specialist_with_feedback=feedback_fn,
        )

        run_pipeline(srs_text="Sample SRS", deps=deps)

        # (a) feedback fn called exactly once (on the rerun cycle)
        self.assertEqual(len(feedback_call_log), 1, feedback_call_log)

        _arch_arg, _scout_arg, prev_arg, issues_arg = feedback_call_log[0]

        # (b) specialist_issues_only contains the specialist ERROR but not the
        #     synthesizer WARN. Note: the WARN is severity WARN, not ERROR,
        #     but the filter criterion is by stage prefix, not severity.
        #     _issue_stage(specialist_error) == "specialist"
        #     _issue_stage(synthesizer_warn) == "synthesizer"
        issue_targets = [i.target for i in issues_arg]
        self.assertIn(specialist_error.target, issue_targets, issues_arg)
        self.assertNotIn(synthesizer_warn.target, issue_targets, issues_arg)

        # (c) 3rd arg is the prev List[SpecialistAnalysis] from the first
        #     specialist call. Because tracking_specialist returns a fresh
        #     list each time, we compare by content (context_name).
        self.assertEqual(len(prev_arg), len(first_specialist_output[0]))
        self.assertEqual(
            prev_arg[0].context.context_name,
            first_specialist_output[0][0].context.context_name,
        )


if __name__ == "__main__":
    unittest.main()
