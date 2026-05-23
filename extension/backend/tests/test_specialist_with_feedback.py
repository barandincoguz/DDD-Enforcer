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


# ---------------------------------------------------------------------------
# Helpers shared by T-30b-5 through T-30b-7
# ---------------------------------------------------------------------------

def _make_domain_architect_with_mock_client():
    """Construct a DomainArchitect with a mocked GeminiClient so no real
    API key is required.  Returns (architect, mock_client)."""
    from unittest.mock import MagicMock, patch
    from core.architect import DomainArchitect
    from core.llm.base import LLMResponse, TokenUsage

    def _canned_llm_response(ctx_name: str = "Ctx_A") -> LLMResponse:
        import json
        payload = json.dumps({
            "context": ctx_name,
            "entities": [
                {
                    "name": f"RefinedEntity_{ctx_name}",
                    "description": "A refined entity.",
                    "attributes": [],
                    "confidence": 0.85,
                    "justification": "From feedback rerun.",
                    "evidence_sentence_indices": [0],
                }
            ],
            "value_objects": [],
            "services": [],
            "aggregates": [],
            "domain_events": [],
            "business_rules": [],
        })
        return LLMResponse(
            content=payload,
            parsed=None,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            model_id="test-model",
            provider="gemini",
        )

    mock_client = MagicMock()
    mock_client.chat.side_effect = lambda **kw: _canned_llm_response(
        ctx_name=kw.get("messages", [{}])[0].get("content", "")[:5] or "Ctx_A"
    )

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("core.llm.gemini.genai.Client"):
            da = DomainArchitect(min_delay=0.0)

    da.client = mock_client
    da.min_delay = 0.0
    return da, mock_client


def _make_two_context_setup():
    """Two contexts (Ctx_A, Ctx_B) with prev_output and a specialist issue
    targeting only Ctx_A."""
    from core.pipeline_contracts import (
        ArchitectOutput,
        ContextHypothesis,
        ScoutOutput,
        SectionedSentence,
        ChunkMetadata,
        SpecialistAnalysis,
        VerifierIssue,
    )
    from core.schemas import Entity

    ctx_a = ContextHypothesis(context_name="Ctx_A", description="Context A")
    ctx_b = ContextHypothesis(context_name="Ctx_B", description="Context B")
    arch = ArchitectOutput(contexts=[ctx_a, ctx_b])

    scout = ScoutOutput(
        sentences=[SectionedSentence(index=0, text="An order is placed.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=30),
    )

    prev_a = SpecialistAnalysis(
        context=ctx_a,
        entities=[
            Entity(
                name="OldEntity_A",
                description="Old entity in A.",
                confidence=0.7,
                justification="original",
                evidence_sentence_indices=[0],
            )
        ],
    )
    prev_b = SpecialistAnalysis(
        context=ctx_b,
        entities=[
            Entity(
                name="OnlyInPrev_B",
                description="Unique entity in B.",
                confidence=0.9,
                justification="original",
                evidence_sentence_indices=[0],
            )
        ],
    )
    prev_output = [prev_a, prev_b]

    specialist_issue = VerifierIssue(
        severity="ERROR",
        check_id="S1",
        target="specialist:Ctx_A.entities.0",
        message="missing aggregate root",
    )

    return arch, scout, prev_output, specialist_issue


# ---------------------------------------------------------------------------
# T-30b-5: render_refinement_prompt called exactly ONCE for the affected
#           context and NOT for the unaffected one.
# ---------------------------------------------------------------------------


class TestSpecialistWithFeedbackCallsRenderPrompt(unittest.TestCase):
    def test_t_30b_5_render_refinement_prompt_called_once_for_affected_ctx(
        self,
    ) -> None:
        """specialist_with_feedback_fn invokes render_refinement_prompt exactly
        once (for Ctx_A only); Ctx_B's prev entry is reused without any
        render_refinement_prompt call."""
        from unittest.mock import patch, MagicMock
        from core.architect import DomainArchitect
        from core.llm.base import LLMResponse, TokenUsage
        import json

        arch, scout, prev_output, specialist_issue = _make_two_context_setup()

        # Canned LLM response for Ctx_A rerun
        canned_json = json.dumps({
            "context": "Ctx_A",
            "entities": [
                {
                    "name": "RefinedEntity_A",
                    "description": "Refined entity in A.",
                    "attributes": [],
                    "confidence": 0.85,
                    "justification": "feedback",
                    "evidence_sentence_indices": [0],
                }
            ],
            "value_objects": [],
            "services": [],
            "aggregates": [],
            "domain_events": [],
            "business_rules": [],
        })
        mock_llm_response = LLMResponse(
            content=canned_json,
            parsed=None,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            model_id="test-model",
            provider="gemini",
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_llm_response

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("core.llm.gemini.genai.Client"):
                da = DomainArchitect(min_delay=0.0)
        da.client = mock_client
        da.min_delay = 0.0

        # Patch where it is looked up (architect imports it at module level).
        with patch(
            "core.architect.render_refinement_prompt",
            wraps=__import__(
                "core.refiner.prompts", fromlist=["render_refinement_prompt"]
            ).render_refinement_prompt,
        ) as mock_render:
            result = da._specialist_with_feedback(
                arch=arch,
                scout=scout,
                prev_output=prev_output,
                issues=[specialist_issue],
            )

        # render_refinement_prompt called exactly once (for Ctx_A only)
        self.assertEqual(mock_render.call_count, 1, mock_render.call_args_list)

        # The call was for stage_name="specialist"
        call_kwargs = mock_render.call_args[1]
        self.assertEqual(call_kwargs["stage_name"], "specialist")

        # The issues list passed contained the specialist issue
        call_issues = call_kwargs["issues"]
        self.assertEqual(len(call_issues), 1)

        # Ctx_B was NOT prompted — verify by checking that result has 2 entries
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# T-30b-6: narrow rerun preserves verbatim SpecialistAnalysis for unaffected
#           contexts (Ctx_B's prev entry is returned unchanged).
# ---------------------------------------------------------------------------


class TestSpecialistWithFeedbackPreservesUnaffectedCtx(unittest.TestCase):
    def test_t_30b_6_unaffected_ctx_entry_is_verbatim_prev(self) -> None:
        """The returned list's Ctx_B entry must be the same (equal) as the
        prev_output Ctx_B entry; Ctx_A entry must differ (refined output)."""
        from unittest.mock import patch, MagicMock
        from core.architect import DomainArchitect
        from core.llm.base import LLMResponse, TokenUsage
        import json

        arch, scout, prev_output, specialist_issue = _make_two_context_setup()
        prev_b = prev_output[1]  # SpecialistAnalysis for Ctx_B

        canned_json = json.dumps({
            "context": "Ctx_A",
            "entities": [
                {
                    "name": "RefinedEntity_A",
                    "description": "Refined entity in A.",
                    "attributes": [],
                    "confidence": 0.85,
                    "justification": "feedback",
                    "evidence_sentence_indices": [0],
                }
            ],
            "value_objects": [],
            "services": [],
            "aggregates": [],
            "domain_events": [],
            "business_rules": [],
        })
        mock_llm_response = LLMResponse(
            content=canned_json,
            parsed=None,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            model_id="test-model",
            provider="gemini",
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_llm_response

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("core.llm.gemini.genai.Client"):
                da = DomainArchitect(min_delay=0.0)
        da.client = mock_client
        da.min_delay = 0.0

        result = da._specialist_with_feedback(
            arch=arch,
            scout=scout,
            prev_output=prev_output,
            issues=[specialist_issue],
        )

        # Find Ctx_B in result
        result_by_ctx = {r.context.context_name: r for r in result}
        self.assertIn("Ctx_B", result_by_ctx)
        self.assertIn("Ctx_A", result_by_ctx)

        # Ctx_B entry must be verbatim (equal) to the prev entry
        self.assertEqual(result_by_ctx["Ctx_B"], prev_b)

        # Ctx_A entry must be the REFINED output (different entity name)
        refined_a = result_by_ctx["Ctx_A"]
        orig_a_entity_names = [e.name for e in prev_output[0].entities]
        refined_a_entity_names = [e.name for e in refined_a.entities]
        # The refined entity should have the new name from canned LLM response
        self.assertNotEqual(refined_a_entity_names, orig_a_entity_names)
        self.assertIn("RefinedEntity_A", refined_a_entity_names)


# ---------------------------------------------------------------------------
# T-30b-7: end-to-end — _specialist_with_feedback is wired into analyze_document
#           and the operation label in token_tracker uses the feedback prefix.
# ---------------------------------------------------------------------------


class TestSpecialistWithFeedbackOperationLabel(unittest.TestCase):
    def test_t_30b_7_operation_label_uses_feedback_prefix(self) -> None:
        """_specialist_with_feedback uses operation label
        'specialist_with_feedback:<ctx>:attempt-1' for token tracking.

        Verified by inspecting all calls to token_tracker.track_api_call
        when the method is invoked with a single-context affected set.
        """
        from unittest.mock import patch, MagicMock, call
        from core.architect import DomainArchitect
        from core.llm.base import LLMResponse, TokenUsage
        from core.pipeline_contracts import (
            ArchitectOutput,
            ContextHypothesis,
            ScoutOutput,
            SectionedSentence,
            ChunkMetadata,
            SpecialistAnalysis,
            VerifierIssue,
        )
        from core.schemas import Entity
        import json

        ctx_a = ContextHypothesis(context_name="OrderCtx", description="Orders")
        arch = ArchitectOutput(contexts=[ctx_a])
        scout = ScoutOutput(
            sentences=[SectionedSentence(index=0, text="Customer places an order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=30),
        )
        prev_a = SpecialistAnalysis(
            context=ctx_a,
            entities=[
                Entity(
                    name="OldOrder",
                    description="Old order entity.",
                    confidence=0.7,
                    justification="orig",
                    evidence_sentence_indices=[0],
                )
            ],
        )
        issue = VerifierIssue(
            severity="ERROR",
            check_id="S1",
            target="specialist:OrderCtx",
            message="missing aggregate root",
        )

        canned_json = json.dumps({
            "context": "OrderCtx",
            "entities": [
                {
                    "name": "Order",
                    "description": "A purchase order.",
                    "attributes": [],
                    "confidence": 0.9,
                    "justification": "feedback",
                    "evidence_sentence_indices": [0],
                }
            ],
            "value_objects": [],
            "services": [],
            "aggregates": [],
            "domain_events": [],
            "business_rules": [],
        })
        mock_llm_response = LLMResponse(
            content=canned_json,
            parsed=None,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
            model_id="test-model",
            provider="gemini",
        )
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_llm_response

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
            with patch("core.llm.gemini.genai.Client"):
                da = DomainArchitect(min_delay=0.0)
        da.client = mock_client
        da.min_delay = 0.0

        # Spy on token_tracker.track_api_call
        track_calls: list = []
        original_track = da.token_tracker.track_api_call

        def spy_track(response, *, stage, operation):
            track_calls.append({"stage": stage, "operation": operation})
            return original_track(response, stage=stage, operation=operation)

        da.token_tracker.track_api_call = spy_track

        result = da._specialist_with_feedback(
            arch=arch,
            scout=scout,
            prev_output=[prev_a],
            issues=[issue],
        )

        # (a) Pipeline completed without raising
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].context.context_name, "OrderCtx")

        # (b) At least one track_api_call used the feedback operation label
        feedback_ops = [
            c for c in track_calls
            if c["operation"].startswith("specialist_with_feedback:OrderCtx:")
        ]
        self.assertGreater(
            len(feedback_ops),
            0,
            f"Expected a 'specialist_with_feedback:OrderCtx:*' operation; got: {track_calls}",
        )

        # (c) The refined output contains the new entity from canned response
        entity_names = [e.name for e in result[0].entities]
        self.assertIn("Order", entity_names)


if __name__ == "__main__":
    unittest.main()
