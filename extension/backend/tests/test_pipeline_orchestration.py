"""5-stage pipeline driver tests. Mock LLM, mock verifier, mock refiner."""

import pytest
from unittest.mock import MagicMock
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.verifier.types import VerifierResult, VerifierIssue, IssueSeverity
from core.orchestration.errors import (
    ArchitectExtractionError,
    PipelineError,
    SpecialistFailureError,
    SynthesizerEmptyModelError,
)
from core.pipeline_contracts import (
    ScoutOutput,
    ArchitectOutput,
    ContextHypothesis,
    SpecialistAnalysis,
    SectionedSentence,
    ChunkMetadata,
)
from core.schemas import DomainModel, Entity


def _ok():
    return VerifierResult(ok=True, issues=[])


def _make_typed_deps():
    """Build PipelineDeps with typed envelope stubs."""

    def scout_fn(srs_text: str) -> ScoutOutput:
        return ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order is placed by a customer.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=45),
        )

    def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="Manages orders"),
        ])

    def specialist_fn(arch: ArchitectOutput, scout: ScoutOutput):
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
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(
            analyses,
            llm_client=MagicMock(),
            project_name="Test",
            skip_enrich=True,
        )

    def verifier_fn(snapshot):
        return _ok()

    return PipelineDeps(
        scout=scout_fn,
        architect=architect_fn,
        specialist=specialist_fn,
        synthesizer=synthesizer_fn,
        verifier=verifier_fn,
    )


def test_pipeline_happy_path_produces_domain_model():
    deps = _make_typed_deps()
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert isinstance(model, DomainModel)
    assert len(model.bounded_contexts) == 1
    assert model.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"


def test_pipeline_propagates_architect_extraction_error():
    deps = _make_typed_deps()
    deps.architect = MagicMock(side_effect=ArchitectExtractionError(srs_path="x"))
    with pytest.raises(ArchitectExtractionError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_propagates_specialist_failure():
    deps = _make_typed_deps()
    deps.specialist = MagicMock(side_effect=SpecialistFailureError(context_name="OrderMgmt"))
    with pytest.raises(SpecialistFailureError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_invokes_refiner_when_verifier_finds_issues():
    """Verifier returns an issue on first call; pipeline refines (re-runs Specialist)
    then verifier returns ok. specialist must be called twice."""
    specialist_mock = MagicMock()

    call_count = [0]

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="Manages orders"),
        ])

    def specialist_fn(arch, scout):
        call_count[0] += 1
        return [
            SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="A purchase order.",
                    confidence=0.9,
                    justification="Cited in sentence 0",
                    evidence_sentence_indices=[0],
                )],
            )
        ]

    verifier_calls = [0]

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="specialist",
                location="specialist:OrderMgmt.entities[0]",
                issue_type="missing_evidence",
                severity=IssueSeverity.ERROR,
                message="missing evidence",
            )])
        return VerifierResult(ok=True, issues=[])

    def synthesizer_fn(analyses):
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(
            analyses,
            llm_client=MagicMock(),
            project_name="Test",
            skip_enrich=True,
        )

    deps = PipelineDeps(
        scout=lambda text: ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
        ),
        architect=architect_fn,
        specialist=specialist_fn,
        synthesizer=synthesizer_fn,
        verifier=verifier_fn,
    )
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    # Refiner re-runs Specialist once → total 2 specialist calls
    assert call_count[0] == 2
    assert model is not None


# =============================================================================
# WP-CORE-5b — SynthesizerEmptyModelError guard placement + srs_path
# =============================================================================


def test_pipeline_raises_synthesizer_empty_model_error_when_specialist_returns_empty():
    """T-EMPTY-1: initial-empty Specialist DI path raises SynthesizerEmptyModelError,
    NOT pydantic.ValidationError. Verifies pre-call guard + PipelineError taxonomy.
    Closes Codex N-2 (merged v1's T-EMPTY-1 + T-EMPTY-2 into one assertion)."""
    deps = _make_typed_deps()
    deps.specialist = MagicMock(return_value=[])
    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert isinstance(exc_info.value, SynthesizerEmptyModelError)


def test_pipeline_synthesizer_not_invoked_when_specialist_empty():
    """T-EMPTY-2: pre-call guard short-circuits before deps.synthesizer is called."""
    deps = _make_typed_deps()
    deps.specialist = MagicMock(return_value=[])
    synth_mock = MagicMock()
    deps.synthesizer = synth_mock
    with pytest.raises(SynthesizerEmptyModelError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert synth_mock.call_count == 0, (
        "Pre-call guard must short-circuit before deps.synthesizer is invoked."
    )


def test_pipeline_raises_synthesizer_empty_model_error_when_refiner_rerun_returns_empty():
    """T-EMPTY-3 (Codex W-1): refiner-success-path edge — first Specialist call
    returns non-empty, verifier fails once, rerun returns [], verifier accepts.
    refined_specialist becomes []; pre-call guard raises SynthesizerEmptyModelError."""
    specialist_calls = [0]

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="x"),
        ])

    def specialist_fn(arch, scout):
        specialist_calls[0] += 1
        if specialist_calls[0] == 1:
            return [SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="An order.",
                    confidence=0.9,
                    justification="cited",
                    evidence_sentence_indices=[0],
                )],
            )]
        return []  # Rerun returns empty.

    verifier_calls = [0]

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="specialist",
                location="specialist:OrderMgmt.entities[0]",
                issue_type="missing_evidence",
                severity=IssueSeverity.ERROR,
                message="m",
            )])
        return VerifierResult(ok=True, issues=[])

    deps = PipelineDeps(
        scout=lambda text: ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
        ),
        architect=architect_fn,
        specialist=specialist_fn,
        synthesizer=MagicMock(),  # Should never be invoked (pre-call guard).
        verifier=verifier_fn,
    )

    with pytest.raises(SynthesizerEmptyModelError):
        run_pipeline(srs_text="x", deps=deps)
    assert specialist_calls[0] == 2, (
        "Refiner should have invoked Specialist twice (initial + one rerun)."
    )


def test_pipeline_post_call_check_catches_injected_synthesizer_returning_empty_model():
    """T-EMPTY-4 (Codex W-3): belt-and-suspenders for injected synthesizers that
    bypass Pydantic via DomainModel.model_construct (which skips validators)."""
    from core.schemas import ProjectMetadata

    deps = _make_typed_deps()

    def injected_synthesizer(analyses):
        # model_construct bypasses Pydantic validation, allowing empty bounded_contexts.
        return DomainModel.model_construct(
            project_name="Test",
            project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
            bounded_contexts=[],
            global_rules=None,
        )

    deps.synthesizer = injected_synthesizer

    with pytest.raises(SynthesizerEmptyModelError) as exc_info:
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert "bypassed Pydantic" in str(exc_info.value), (
        "Post-call check should emit a distinct message from the pre-call guard."
    )


# =============================================================================
# WP-CORE-6 / WP-CORE-7 — Degrade-log + architect-stage hard-fail
# =============================================================================


def _make_deps_with_feedback_arch(architect_fn, architect_with_feedback_fn,
                                  specialist_fn=None, verifier_fn=None,
                                  synthesizer_fn=None, scout_fn=None):
    """Build a PipelineDeps including the WP-CORE-7 architect_with_feedback callable.

    Pre-GREEN this will TypeError on construction; the GREEN commit adds the
    field to PipelineDeps. Used by every WP-CORE-7 dispatcher test.
    """
    from core.synthesizer import synthesize_domain_model

    def _default_scout(srs_text: str) -> ScoutOutput:
        return ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order is placed.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=20),
        )

    def _default_specialist(arch, scout):
        return [
            SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="An order placed.",
                    confidence=0.9,
                    justification="cited",
                    evidence_sentence_indices=[0],
                )],
            )
        ]

    def _default_synth(analyses):
        return synthesize_domain_model(
            analyses, llm_client=MagicMock(), project_name="Test", skip_enrich=True,
        )

    return PipelineDeps(
        scout=scout_fn or _default_scout,
        architect=architect_fn,
        architect_with_feedback=architect_with_feedback_fn,
        specialist=specialist_fn or _default_specialist,
        synthesizer=synthesizer_fn or _default_synth,
        verifier=verifier_fn or (lambda snapshot: _ok()),
    )


def test_refiner_exhaustion_on_architect_stage_raises_grounding_error(capsys):
    """T-DEGRADE-LOG-1 (WP-CORE-7 update, Codex C-1+C-2): architect-stage
    issues no longer degrade silently. After 1 architect re-run with persistent
    issues, pipeline raises ArchitectGroundingError. Log still includes issue
    location + message (preserves WP-CORE-6 C-4 visibility contract on the
    hard-fail path).

    Replaces the WP-CORE-6 T-DEGRADE-LOG-1 which expected best-effort degrade.
    """
    from core.orchestration.errors import ArchitectGroundingError

    architect_calls = [0]
    architect_with_feedback_calls = [0]

    def architect_fn(scout):
        architect_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="x"),
        ])

    def architect_with_feedback_fn(scout, issues):
        architect_with_feedback_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="x"),
        ])

    bad_issue = VerifierIssue(
        stage="architect",
        location="architect:contexts[OrderMgmt].supporting_sentence_ids",
        issue_type="ungrounded_context",
        severity=IssueSeverity.ERROR,
        message="Context 'OrderMgmt' has no supporting_sentence_ids — cannot verify SRS grounding",
    )

    deps = _make_deps_with_feedback_arch(
        architect_fn=architect_fn,
        architect_with_feedback_fn=architect_with_feedback_fn,
        verifier_fn=lambda snapshot: VerifierResult(ok=False, issues=[bad_issue]),
    )

    with pytest.raises(ArchitectGroundingError) as exc_info:
        run_pipeline(srs_text="x", deps=deps)

    assert exc_info.value.cycles_attempted == 1
    assert len(exc_info.value.issues) == 1
    assert len(exc_info.value.residual_issues) == 0
    assert architect_calls[0] == 1
    assert architect_with_feedback_calls[0] == 1

    captured = capsys.readouterr().out
    assert "architect:contexts[OrderMgmt].supporting_sentence_ids" in captured, (
        "Hard-fail log must include the failing issue's location"
    )
    assert "no supporting_sentence_ids" in captured, (
        "Hard-fail log must include the failing issue's message"
    )


# =============================================================================
# WP-CORE-7 — Refiner stage-aware dispatch (F-22 mode C hybrid)
# =============================================================================


def test_pipeline_re_runs_architect_on_initial_architect_stage_issue():
    """T-DISPATCH-1: verifier returns 1 architect-stage issue on first call,
    ok on second; architect_with_feedback invoked exactly once;
    plain architect_fn invoked once."""
    architect_calls = [0]
    architect_with_feedback_calls = [0]
    verifier_calls = [0]

    def architect_fn(scout):
        architect_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="x"),
        ])

    def architect_with_feedback_fn(scout, issues):
        architect_with_feedback_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(
                context_name="OrderMgmt", description="x",
                supporting_sentence_ids=[0],
            ),
        ])

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="architect",
                location="architect:contexts[OrderMgmt].supporting_sentence_ids",
                issue_type="ungrounded_context",
                severity=IssueSeverity.ERROR,
                message="missing IDs",
            )])
        return VerifierResult(ok=True, issues=[])

    deps = _make_deps_with_feedback_arch(
        architect_fn=architect_fn,
        architect_with_feedback_fn=architect_with_feedback_fn,
        verifier_fn=verifier_fn,
    )

    model = run_pipeline(srs_text="x", deps=deps)

    assert model is not None
    assert architect_calls[0] == 1
    assert architect_with_feedback_calls[0] == 1


def test_pipeline_raises_grounding_error_after_architect_rerun_exhaustion():
    """T-DISPATCH-2: verifier always returns architect-stage issue;
    ArchitectGroundingError raised; cycles_attempted=1, issues len 1."""
    from core.orchestration.errors import ArchitectGroundingError

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    def architect_with_feedback_fn(scout, issues):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    persistent_issue = VerifierIssue(
        stage="architect",
        location="architect:contexts[X].supporting_sentence_ids",
        issue_type="ungrounded_context",
        severity=IssueSeverity.ERROR,
        message="missing IDs",
    )

    deps = _make_deps_with_feedback_arch(
        architect_fn=architect_fn,
        architect_with_feedback_fn=architect_with_feedback_fn,
        verifier_fn=lambda s: VerifierResult(ok=False, issues=[persistent_issue]),
    )

    with pytest.raises(ArchitectGroundingError) as exc_info:
        run_pipeline(srs_text="x", deps=deps)

    assert exc_info.value.cycles_attempted == 1
    assert len(exc_info.value.issues) == 1
    assert len(exc_info.value.residual_issues) == 0
    assert exc_info.value.srs_path == "<unknown>"


def test_pipeline_specialist_stage_issue_does_not_re_run_architect():
    """T-DISPATCH-3: specialist-stage issue twice then ok; specialist called twice;
    architect_with_feedback never invoked. Regression contract for specialist path."""
    architect_calls = [0]
    architect_with_feedback_calls = [0]
    specialist_calls = [0]
    verifier_calls = [0]

    def architect_fn(scout):
        architect_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    def architect_with_feedback_fn(scout, issues):
        architect_with_feedback_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    def specialist_fn(arch, scout):
        specialist_calls[0] += 1
        return [SpecialistAnalysis(
            context=arch.contexts[0],
            entities=[Entity(
                name="Order", description="An order.", confidence=0.9,
                justification="cited", evidence_sentence_indices=[0],
            )],
        )]

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="specialist",
                location="specialist:X.entities[0]",
                issue_type="missing_evidence",
                severity=IssueSeverity.ERROR,
                message="m",
            )])
        return VerifierResult(ok=True, issues=[])

    deps = _make_deps_with_feedback_arch(
        architect_fn=architect_fn,
        architect_with_feedback_fn=architect_with_feedback_fn,
        specialist_fn=specialist_fn,
        verifier_fn=verifier_fn,
    )

    model = run_pipeline(srs_text="x", deps=deps)

    assert model is not None
    assert architect_calls[0] == 1
    assert architect_with_feedback_calls[0] == 0
    assert specialist_calls[0] == 2  # initial + 1 specialist refine rerun


def test_pipeline_mixed_stage_issues_persistent_architect_raises_with_residuals():
    """T-DISPATCH-4: persistent architect + specialist issues every verify;
    ArchitectGroundingError raised; exc.issues has 1 architect issue,
    exc.residual_issues has 1 specialist issue."""
    from core.orchestration.errors import ArchitectGroundingError

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    def architect_with_feedback_fn(scout, issues):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    arch_issue = VerifierIssue(
        stage="architect",
        location="architect:contexts[X].supporting_sentence_ids",
        issue_type="ungrounded_context",
        severity=IssueSeverity.ERROR,
        message="arch missing IDs",
    )
    spec_issue = VerifierIssue(
        stage="specialist",
        location="specialist:X.entities[0]",
        issue_type="missing_evidence",
        severity=IssueSeverity.ERROR,
        message="spec missing evidence",
    )

    deps = _make_deps_with_feedback_arch(
        architect_fn=architect_fn,
        architect_with_feedback_fn=architect_with_feedback_fn,
        verifier_fn=lambda s: VerifierResult(ok=False, issues=[arch_issue, spec_issue]),
    )

    with pytest.raises(ArchitectGroundingError) as exc_info:
        run_pipeline(srs_text="x", deps=deps)

    assert exc_info.value.cycles_attempted == 1
    assert len(exc_info.value.issues) == 1
    assert len(exc_info.value.residual_issues) == 1
    # Architect-stage issue is in .issues; specialist-stage in .residual_issues.
    arch_targets = [
        getattr(i, "target", None) or getattr(i, "location", "")
        for i in exc_info.value.issues
    ]
    resid_targets = [
        getattr(i, "target", None) or getattr(i, "location", "")
        for i in exc_info.value.residual_issues
    ]
    assert any(t.startswith("architect:") for t in arch_targets)
    assert any(t.startswith("specialist:") for t in resid_targets)


def test_pipeline_architect_rerun_succeeds_then_specialist_refine_runs():
    """T-DISPATCH-5 (Codex W-1): architect issue first; after architect rerun,
    specialist issue surfaces; specialist refine loop runs once; final ok.

    Verifies that a successful architect feedback rerun unblocks the specialist
    path — pipeline does NOT raise ArchitectGroundingError when feedback fixes
    the architect issue."""
    architect_calls = [0]
    architect_with_feedback_calls = [0]
    specialist_calls = [0]
    verifier_calls = [0]

    def architect_fn(scout):
        architect_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    def architect_with_feedback_fn(scout, issues):
        architect_with_feedback_calls[0] += 1
        return ArchitectOutput(contexts=[
            ContextHypothesis(
                context_name="X", description="x",
                supporting_sentence_ids=[0],
            ),
        ])

    def specialist_fn(arch, scout):
        specialist_calls[0] += 1
        return [SpecialistAnalysis(
            context=arch.contexts[0],
            entities=[Entity(
                name="Order", description="An order.", confidence=0.9,
                justification="cited", evidence_sentence_indices=[0],
            )],
        )]

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="architect",
                location="architect:contexts[X].supporting_sentence_ids",
                issue_type="ungrounded_context",
                severity=IssueSeverity.ERROR,
                message="arch missing IDs",
            )])
        if verifier_calls[0] == 2:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="specialist",
                location="specialist:X.entities[0]",
                issue_type="missing_evidence",
                severity=IssueSeverity.ERROR,
                message="spec missing evidence",
            )])
        return VerifierResult(ok=True, issues=[])

    deps = _make_deps_with_feedback_arch(
        architect_fn=architect_fn,
        architect_with_feedback_fn=architect_with_feedback_fn,
        specialist_fn=specialist_fn,
        verifier_fn=verifier_fn,
    )

    model = run_pipeline(srs_text="x", deps=deps)

    assert model is not None
    assert architect_calls[0] == 1
    assert architect_with_feedback_calls[0] == 1
    # specialist called: 1 initial (with fresh arch via architect_fn) +
    # 1 after architect rerun + 1 specialist-refine rerun.
    # Implementations may differ slightly on the second arch invocation's
    # specialist call: GREEN spec D-5 re-invokes deps.specialist after architect
    # feedback rerun → so total = 2 (post-rerun-architect specialist call + 1
    # refine-loop rerun). The first architect_fn produced 1 specialist call
    # but its output was discarded when architect_with_feedback fired.
    assert specialist_calls[0] >= 2


def test_pipeline_architect_fail_log_includes_full_issue_list(capsys):
    """T-LOG-1: ArchitectGroundingError raise path still emits stdout line
    containing each issue's target + message (WP-CORE-6 C-4 contract on
    hard-fail path).

    Distinct from T-DEGRADE-LOG-1: this tests the architect HARD-FAIL log
    independent of stage attribution detail."""
    from core.orchestration.errors import ArchitectGroundingError

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    def architect_with_feedback_fn(scout, issues):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="X", description="x"),
        ])

    issue = VerifierIssue(
        stage="architect",
        location="architect:contexts[OrderMgmt].supporting_sentence_ids",
        issue_type="ungrounded_context",
        severity=IssueSeverity.ERROR,
        message="UNIQUE_LOG_MARKER_xyz",
    )

    deps = _make_deps_with_feedback_arch(
        architect_fn=architect_fn,
        architect_with_feedback_fn=architect_with_feedback_fn,
        verifier_fn=lambda s: VerifierResult(ok=False, issues=[issue]),
    )

    with pytest.raises(ArchitectGroundingError):
        run_pipeline(srs_text="x", deps=deps)

    captured = capsys.readouterr().out
    assert "architect:contexts[OrderMgmt].supporting_sentence_ids" in captured
    assert "UNIQUE_LOG_MARKER_xyz" in captured


def test_pipeline_specialist_degrade_log_still_includes_issues_list(capsys):
    """T-LOG-2: Specialist-only exhaustion still degrades (preserves WP-CORE-6
    C-4 contract on the degrade path). Regression guard — passes from start;
    GREEN must preserve this behavior.

    Constructs deps WITHOUT architect_with_feedback (uses existing fixture)
    so it can run pre-GREEN as a sanity check that the specialist path is
    unchanged."""
    deps = _make_typed_deps()

    specialist_issue = VerifierIssue(
        stage="specialist",
        location="specialist:OrderMgmt.entities[0]",
        issue_type="missing_evidence",
        severity=IssueSeverity.ERROR,
        message="SPEC_LOG_MARKER_xyz",
    )
    deps.verifier = lambda snapshot: VerifierResult(ok=False, issues=[specialist_issue])

    model = run_pipeline(srs_text="x", deps=deps)
    assert model is not None  # degrades gracefully on specialist-only path

    captured = capsys.readouterr().out
    assert "specialist:OrderMgmt.entities[0]" in captured
    assert "SPEC_LOG_MARKER_xyz" in captured
