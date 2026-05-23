"""5-stage pipeline driver: Scout → Architect → Specialist → Verifier → Synthesizer.

Each stage is injected as a Callable so this module stays test-friendly.
Real wiring (Gemini-backed stages) happens in DomainArchitect.analyze_document.

WP-CORE-7 (F-22 mode C hybrid): the Refiner is stage-aware. Architect-stage
verifier ERRORs trigger ONE issue-aware re-prompt of the Architect via the
new `architect_with_feedback` dep; on persistent failure the pipeline raises
ArchitectGroundingError. Specialist-stage refine loop is unchanged.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from core.schemas import DomainModel
from core.pipeline_contracts import (
    ScoutOutput,
    ArchitectOutput,
    SpecialistAnalysis,
    VerifierResult,
)
from core.orchestration.errors import (
    ArchitectGroundingError,
    RefinementExhaustedError,
    SynthesizerEmptyModelError,
)
from core.refiner.loop import refine_until_clean


ScoutFn = Callable[[str], ScoutOutput]
ArchitectFn = Callable[[ScoutOutput], ArchitectOutput]
ArchitectWithFeedbackFn = Callable[[ScoutOutput, List[Any]], ArchitectOutput]
SpecialistFn = Callable[[ArchitectOutput, ScoutOutput], List[SpecialistAnalysis]]
SynthesizerFn = Callable[[List[SpecialistAnalysis]], DomainModel]
VerifierFn = Callable[[Dict[str, Any]], VerifierResult]


# WP-CORE-7 D-2 (Codex C-1): derive stage from VerifierIssue target/location
# prefix. Every Verifier check populates the prefix; the legacy/contract
# duality is handled by inspecting both attributes.
_KNOWN_STAGE_PREFIXES = ("scout:", "architect:", "specialist:", "synthesizer:")


def _issue_stage(issue: Any) -> Optional[str]:
    """Derive stage from an issue's target/location prefix or stage attribute.

    Supports both core.pipeline_contracts.VerifierIssue (Pydantic, `target`)
    and core.verifier.types.VerifierIssue (dataclass, `location` + `stage`).
    Returns None if no stage can be derived.

    Invariant: every VerifierIssue MUST populate target/location with a
    `{stage}:` prefix. Documented in handoff §"Non-negotiables carried
    forward" (WP-CORE-7).
    """
    target = getattr(issue, "target", None) or getattr(issue, "location", "") or ""
    if isinstance(target, str):
        for prefix in _KNOWN_STAGE_PREFIXES:
            if target.startswith(prefix):
                return prefix.rstrip(":")
    stage_attr = getattr(issue, "stage", None)
    if isinstance(stage_attr, str):
        return stage_attr
    return None


def _format_issue(issue: Any) -> str:
    """One-line issue summary for logs: severity@stage:target/location: message."""
    sev = getattr(issue, "severity", "")
    sev_str = getattr(sev, "value", None) if not isinstance(sev, str) else sev
    if sev_str is None:
        sev_str = str(sev)
    target = getattr(issue, "target", None) or getattr(issue, "location", "") or ""
    message = getattr(issue, "message", "")
    return f"{sev_str}@{target}: {message}"


def _log_architect_rerun(arch_issues: List[Any], attempt: int) -> None:
    summary = "; ".join(_format_issue(i) for i in arch_issues)
    print(
        f"  ↻ architect stage-aware rerun #{attempt}: "
        f"{len(arch_issues)} architect-stage issue(s) "
        f"feeding back to identify_contexts. Issues: {summary}"
    )


def _log_architect_fail(
    arch_issues: List[Any], attempts: int, srs_path: Optional[str],
) -> None:
    summary = "; ".join(_format_issue(i) for i in arch_issues)
    print(
        f"  ❌ architect grounding fail after {attempts} rerun(s) "
        f"(srs={srs_path or '<unknown>'}); raising ArchitectGroundingError. "
        f"Issues: {summary}"
    )


def _log_specialist_degrade(other_issues: List[Any]) -> None:
    summary = "; ".join(_format_issue(i) for i in other_issues)
    print(
        f"  ⚠️  refiner exhausted retries ({len(other_issues)} unresolved "
        f"issue(s)); continuing with last Specialist output. Issues: {summary}"
    )


@dataclass
class PipelineDeps:
    scout: ScoutFn
    architect: ArchitectFn
    architect_with_feedback: ArchitectWithFeedbackFn
    specialist: SpecialistFn
    synthesizer: SynthesizerFn
    verifier: VerifierFn


def run_pipeline(
    *,
    srs_text: str,
    deps: PipelineDeps,
    srs_path: Optional[str] = None,
) -> DomainModel:
    """Run the 5-stage pipeline with typed envelopes throughout.

    Raises PipelineError subclasses on failure; otherwise returns a
    validated DomainModel.

    Args:
        srs_text: SRS document text fed to the Scout stage.
        deps: Injected stage callables (Scout, Architect, Architect-with-
            feedback, Specialist, Synthesizer, Verifier).
        srs_path: Optional source path label for error messages. WP-CORE-5b
            threads this through `SynthesizerEmptyModelError.srs_path`;
            WP-CORE-7 also threads it through `ArchitectGroundingError`.
            Defaults to "<unknown>" inside the errors if omitted.

    WP-CORE-7 dispatch order (D-5):
        1. Scout once.
        2. Architect loop (outer): on architect-stage verifier ERRORs,
           re-invoke `architect_with_feedback` ONCE; on persistent failure
           raise ArchitectGroundingError.
        3. Inside each architect attempt: specialist + specialist refine loop
           (existing WP-CORE-1..6 behavior).
        4. Post-loop: pre-call guard, synthesizer, post-call check (unchanged).
    """
    scout: ScoutOutput = deps.scout(srs_text)

    architect_attempts = 0
    architect_max_cycles = 1
    architect_feedback: Optional[List[Any]] = None
    refined_specialist: Optional[List[SpecialistAnalysis]] = None

    while True:
        # Stage 2 (with optional issue-aware feedback on rerun).
        if architect_feedback is None:
            arch: ArchitectOutput = deps.architect(scout)
        else:
            arch = deps.architect_with_feedback(scout, architect_feedback)

        # Stage 3
        specialist_output: List[SpecialistAnalysis] = deps.specialist(arch, scout)

        snapshot: Dict[str, Any] = {
            "scout": scout,
            "architect": arch,
            "specialist": specialist_output,
        }

        # Pre-check (Codex C-2): inspect verifier output BEFORE entering the
        # specialist refine loop. Architect-stage issues dispatch to architect
        # rerun directly; otherwise the pre-check result is threaded into
        # refine_until_clean via `initial_result=` so the same verifier call
        # is not repeated.
        initial_result = deps.verifier(snapshot)
        arch_issues = [
            i for i in initial_result.issues if _issue_stage(i) == "architect"
        ]
        other_initial_issues = [
            i for i in initial_result.issues if _issue_stage(i) != "architect"
        ]

        if initial_result.error_count() > 0 and arch_issues:
            if architect_attempts < architect_max_cycles:
                architect_attempts += 1
                architect_feedback = arch_issues
                _log_architect_rerun(arch_issues, architect_attempts)
                continue
            _log_architect_fail(arch_issues, architect_attempts, srs_path)
            raise ArchitectGroundingError(
                srs_path=srs_path or "<unknown>",
                issues=arch_issues,
                residual_issues=other_initial_issues,
                cycles_attempted=architect_attempts,
            )

        # No architect-stage issues → enter specialist refine loop, threading
        # the already-evaluated initial_result so we don't re-verify
        # immediately.
        def _re_run_specialist(_prev, _result) -> List[SpecialistAnalysis]:
            return deps.specialist(arch, scout)

        try:
            refined_specialist, _cycles = refine_until_clean(
                stage_name="specialist",
                initial_output=specialist_output,
                stage_runner=_re_run_specialist,
                verifier=lambda s: deps.verifier({**snapshot, "specialist": s}),
                max_cycles=2,
                initial_result=initial_result,
            )
        except RefinementExhaustedError as exc:
            # Architect issues can surface AFTER specialist re-runs too
            # (e.g., re-evaluation reveals architect drift).
            late_arch_issues = [
                i for i in exc.issues if _issue_stage(i) == "architect"
            ]
            late_other_issues = [
                i for i in exc.issues if _issue_stage(i) != "architect"
            ]

            if late_arch_issues:
                if architect_attempts < architect_max_cycles:
                    architect_attempts += 1
                    architect_feedback = late_arch_issues
                    _log_architect_rerun(late_arch_issues, architect_attempts)
                    continue
                _log_architect_fail(late_arch_issues, architect_attempts, srs_path)
                raise ArchitectGroundingError(
                    srs_path=srs_path or "<unknown>",
                    issues=late_arch_issues,
                    residual_issues=late_other_issues,
                    cycles_attempted=architect_attempts,
                )

            # Specialist-only exhaustion: existing degrade-log path
            # (preserves WP-CORE-6 C-4 contract).
            _log_specialist_degrade(exc.issues)
            refined_specialist = specialist_output

        # WP-CORE-7 W-5 (Codex): the bare `except Exception` block from the
        # pre-WP-CORE-7 implementation has been removed. Unexpected exceptions
        # propagate to the caller per AGENTS.md "explicit failure".

        break  # exit outer while loop

    # Pre-call guard (primary): catches refined_specialist == [] for both
    # initial-empty and refiner-rerun-to-empty DI paths. Per Codex W-1 + W-2
    # of WP-CORE-5b, this is the only place where the empty case can be
    # observed taxonomically as a PipelineError; the in-tree synthesizer
    # would otherwise raise pydantic.ValidationError via DomainModel._non_empty
    # (core/schemas.py:207-215), escaping the PipelineError taxonomy.
    if not refined_specialist:
        raise SynthesizerEmptyModelError(
            input_summary="0 SpecialistAnalysis from upstream pipeline",
            srs_path=srs_path or "<unknown>",
        )

    model: DomainModel = deps.synthesizer(refined_specialist)

    # Post-call boundary check (belt-and-suspenders): retained per Codex W-3
    # of WP-CORE-5b because PipelineDeps.synthesizer is an injectable
    # SynthesizerFn; a future or test-injected synthesizer could construct
    # DomainModel via DomainModel.model_construct (which bypasses Pydantic
    # validation) and return an empty model.
    if not model.bounded_contexts:
        raise SynthesizerEmptyModelError(
            input_summary="synthesizer returned 0 bounded contexts (bypassed Pydantic)",
            srs_path=srs_path or "<unknown>",
        )
    return model
