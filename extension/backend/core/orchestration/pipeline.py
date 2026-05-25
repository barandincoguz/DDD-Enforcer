"""5-stage pipeline driver: Scout → Architect → Specialist → Verifier → Synthesizer.

Each stage is injected as a Callable so this module stays test-friendly.
Real wiring (Gemini-backed stages) happens in DomainArchitect.analyze_document.

WP-CORE-7 (F-22 mode C hybrid): the Refiner is stage-aware. Architect-stage
verifier ERRORs trigger ONE issue-aware re-prompt of the Architect via the
new `architect_with_feedback` dep; on persistent failure the pipeline raises
ArchitectGroundingError. Specialist-stage refine loop is unchanged.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    cast,
)
from core.schemas import DomainModel

if TYPE_CHECKING:
    # Type-only import for the CriticFn alias's return annotation. Kept under
    # TYPE_CHECKING so this module takes no runtime dependency on the critic
    # package (Task 7) — the alias references CriticReport as a string.
    from core.schemas import CriticReport, ContextMap
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


@contextmanager
def _optional_stage(name: str, *, extend: bool = False) -> Iterator[Any]:
    """WP-CORE-20b: wrap a stage call in `emitter.stage(name)` when an
    active emitter is present; no-op otherwise.

    Pipeline calls (Scout / Architect / Specialist / Synthesizer)
    previously executed OUTSIDE any emitter scope, so
    `StageEmitter.record_llm_call` silently dropped per-call records
    (`_stage_var.get()` returned None).  This helper closes that gap
    without touching StageEmitter itself.

    Behaviour:
    * Active emitter (`get_current_emitter()` non-None): enters
      `emitter.stage(name, extend=extend)` so `_stage_var` is set to
      `name`, `manifest.stages[name]` is registered, elapsed_ms is
      recorded on exit, and exceptions are logged into
      `manifest.errors`.
    * No active emitter (most unit tests, CLI tools, schema_probe):
      yields without side effects so existing emitter-less call sites
      keep working unchanged.

    WP-CORE-20c: `extend=True` reuses an existing `manifest.stages[name]`
    record across multiple invocations so the WP-CORE-7 architect
    feedback rerun preserves the first attempt's llm_calls /
    json_parse_failures.  Per-attempt bounds + status land in
    `record.metrics["attempts"]`.  Callers MUST pass `extend=True` only
    on the rerun (second+ enter); the first enter stays `extend=False`
    (default) so a fresh record is created.
    """
    from core.observability.emitter import get_current_emitter
    emitter = get_current_emitter()
    if emitter is None:
        yield None
        return
    with emitter.stage(name, extend=extend) as record:
        yield record


ScoutFn = Callable[[str], ScoutOutput]
ArchitectFn = Callable[[ScoutOutput], ArchitectOutput]
ArchitectWithFeedbackFn = Callable[[ScoutOutput, List[Any]], ArchitectOutput]
SpecialistFn = Callable[[ArchitectOutput, ScoutOutput], List[SpecialistAnalysis]]
SpecialistWithFeedbackFn = Callable[
    [ArchitectOutput, ScoutOutput, List[SpecialistAnalysis], List[Any]],
    List[SpecialistAnalysis],
]
SynthesizerFn = Callable[[List[SpecialistAnalysis]], DomainModel]
VerifierFn = Callable[[Dict[str, Any]], VerifierResult]
# Holistic Critic: per-cycle critique callable. CriticReport stays a string
# forward-ref so this module takes no new top-level import (Task 7).
CriticFn = Callable[[DomainModel, ScoutOutput, list], "CriticReport"]
# Context-Mapper: produce a strategic context map. feedback=None -> fresh map;
# a list of CritiqueFindings -> Critic-driven re-map.
ContextMapperFn = Callable[[DomainModel, ScoutOutput, Optional[list]], "ContextMap"]


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


def _flapping_signature(issues: List[Any]) -> tuple:
    """WP-CORE-30: canonical signature of an issue set for flap detection.

    Two cycles flap when their issue sets are identical (same set of
    (location, issue_type) tuples regardless of order). Sorting the
    tuple keeps the signature stable across reorderings.
    """
    sig = []
    for issue in issues:
        loc = getattr(issue, "location", "") or getattr(issue, "target", "")
        itype = (
            getattr(issue, "issue_type", None)
            or getattr(issue, "check_id", None)
            or ""
        )
        sig.append((str(loc), str(itype)))
    return tuple(sorted(sig))


def _detect_flapping(cycle_history: List[List[Any]]) -> bool:
    """WP-CORE-30: True iff the last two cycles emitted the same issue set.

    Empty / 1-cycle history → False (cannot flap with fewer than 2
    cycles). When `cycle_history >= 2`, compare last two signatures.
    """
    if len(cycle_history) < 2:
        return False
    return _flapping_signature(cycle_history[-1]) == _flapping_signature(
        cycle_history[-2]
    )


def _record_refiner_metrics_safely(
    *,
    cycles_used: int,
    exhausted: bool,
    residual_count: int,
    max_cycles: int = 2,
    flapped: bool = False,
    cycle_history_len: int = 0,
) -> None:
    """WP-CORE-24 + WP-CORE-30: append refiner stats to the active StageEmitter
    manifest.

    No-op when no emitter is in context (CLI runs, schema_probe, tests
    without a manifest). The active emitter is sourced via
    `get_current_emitter()` which reads the ContextVar in
    `core.observability.emitter`.

    Wrapped in a defensive try/except so observability bugs never bring
    down the orchestrator — refiner stats are best-effort.

    WP-CORE-30 fields: `flapped` (bool) + `cycle_history_len` (int).
    """
    try:
        from core.observability.emitter import get_current_emitter
        from core.observability.run_manifest import StageRecord
        emitter = get_current_emitter()
        if emitter is None:
            return
        record = StageRecord(
            status="exhausted" if exhausted else "clean",
        )
        record.metrics["cycles_used"] = cycles_used
        record.metrics["exhausted"] = exhausted
        record.metrics["exhausted_residual_count"] = residual_count
        record.metrics["max_cycles"] = max_cycles
        record.metrics["flapped"] = flapped
        record.metrics["cycle_history_len"] = cycle_history_len
        with emitter._lock:
            emitter.manifest.stages["refiner"] = record
    except Exception:
        # Never let observability bring down the orchestrator.
        pass


@dataclass
class PipelineDeps:
    scout: ScoutFn
    architect: ArchitectFn
    architect_with_feedback: ArchitectWithFeedbackFn
    specialist: SpecialistFn
    synthesizer: SynthesizerFn
    verifier: VerifierFn
    # WP-CORE-30b: optional issue-aware specialist rerun (symmetric to
    # architect_with_feedback).  When set, the refiner's _re_run_specialist
    # closure forwards (arch, scout, prev_output, specialist_issues_only) so
    # the concrete LLM caller can craft a targeted re-prompt.  Defaults to
    # None so every existing PipelineDeps construction site is unaffected.
    specialist_with_feedback: Optional[SpecialistWithFeedbackFn] = None
    # Holistic Critic: optional per-cycle critique callable. When set,
    # run_pipeline dispatches to run_critique_loop; None → single-pass behavior.
    critic: Optional["CriticFn"] = None
    context_mapper: Optional["ContextMapperFn"] = None


def _apply_context_map(
    model: DomainModel,
    deps: "PipelineDeps",
    scout: ScoutOutput,
    *,
    feedback: Optional[List[Any]] = None,
) -> DomainModel:
    """Attach a strategic context map + re-derive allowed_dependencies.

    PURE: returns a deep copy; never mutates `model` (the critique loop's
    best_model may alias it). No-op (returns `model` unchanged) when no
    context_mapper is wired. On ContextMapperError the text-scan baseline
    allowed_dependencies is kept and the failure is recorded on context_map.

    On SUCCESS, allowed_dependencies is a pure derived projection of the context
    map: every bounded context is overwritten from `derive_allowed_dependencies`
    (None when it has no outgoing dependency edges). The baseline is NEVER merged
    with the derived result — A is authoritative. This ensures SEPARATE_WAYS
    contexts (no edges → None) and upstream-only contexts in CUSTOMER_SUPPLIER
    relationships are not left with stale text-scan baselines."""
    if deps.context_mapper is None:
        return model
    from core.context_mapper import derive_allowed_dependencies
    from core.context_mapper.errors import ContextMapperError
    from core.schemas import ContextMap

    new_model = model.model_copy(deep=True)
    with _optional_stage("context_mapper"):
        try:
            cmap = deps.context_mapper(new_model, scout, feedback)
        except ContextMapperError as exc:
            print(f"  ⚠️  context-mapper failed: {exc}; keeping baseline allowed_dependencies")
            new_model.context_map = ContextMap(model_id="unknown", error=exc.reason)
            return new_model

    valid_names = {bc.context_name for bc in new_model.bounded_contexts}
    derived, warnings = derive_allowed_dependencies(cmap, valid_names)
    cmap.warnings.extend(warnings)
    new_model.context_map = cmap
    for bc in new_model.bounded_contexts:
        dep_list = derived.get(bc.context_name)
        bc.allowed_dependencies = sorted(dep_list) if dep_list else None
    return new_model


def run_pipeline(
    *,
    srs_text: str,
    deps: PipelineDeps,
    srs_path: Optional[str] = None,
) -> DomainModel:
    """Run the pipeline. With deps.critic set, drive the bounded critique loop;
    otherwise a single generation pass (historical behavior)."""
    with _optional_stage("scout"):
        scout: ScoutOutput = deps.scout(srs_text)

    if deps.critic is None:
        model, _arch, _specialist = _generate_once(scout, deps, srs_path)
        return model

    from core.critic.loop import run_critique_loop
    return run_critique_loop(scout, deps, srs_path)


def _generate_once(
    scout: ScoutOutput,
    deps: PipelineDeps,
    srs_path: Optional[str],
    *,
    architect_feedback: Optional[List[Any]] = None,
) -> tuple[DomainModel, ArchitectOutput, List[SpecialistAnalysis]]:
    """One full generation pass: Architect (rerun loop) → Specialist (refine)
    → Synthesizer. Returns the model plus the final ArchitectOutput and
    refined SpecialistAnalysis list (the critique loop reuses these). When
    `architect_feedback` is provided, it seeds the FIRST architect call.

    WP-CORE-7 dispatch order (D-5):
        1. Architect loop (outer): on architect-stage verifier ERRORs,
           re-invoke `architect_with_feedback` ONCE; on persistent failure
           raise ArchitectGroundingError.
        2. Inside each architect attempt: specialist + specialist refine loop
           (existing WP-CORE-1..6 behavior).
        3. Post-loop: pre-call guard, synthesizer, post-call check (unchanged).
    """
    architect_attempts = 0
    architect_max_cycles = 1
    architect_feedback_local: Optional[List[Any]] = architect_feedback
    refined_specialist: Optional[List[SpecialistAnalysis]] = None

    while True:
        # Stage 2 (with optional issue-aware feedback on rerun).
        # WP-CORE-20c: rerun (architect_feedback is not None) extends the
        # prior "architect" StageRecord so the first attempt's llm_calls
        # and json_parse_failures survive the re-enter (paper-data
        # fidelity).  First attempt stays extend=False so a fresh record
        # is created.
        with _optional_stage(
            "architect", extend=(architect_attempts > 0)
        ):
            if architect_feedback_local is None:
                arch: ArchitectOutput = deps.architect(scout)
            else:
                arch = deps.architect_with_feedback(scout, architect_feedback_local)

        # Stage 3
        with _optional_stage("specialist"):
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
                architect_feedback_local = arch_issues
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
        # WP-CORE-20b: the refiner's stage_runner invocation is ALSO a
        # specialist call; wrap it so its LLM records land in the
        # "specialist" bucket (rather than dropping silently outside any
        # stage scope).
        def _re_run_specialist(_prev, _result) -> List[SpecialistAnalysis]:
            with _optional_stage("specialist"):
                if deps.specialist_with_feedback is not None:
                    specialist_issues = [
                        i for i in _result.issues if _issue_stage(i) == "specialist"
                    ]
                    return deps.specialist_with_feedback(
                        arch, scout, _prev, specialist_issues
                    )
                return deps.specialist(arch, scout)

        # refine_until_clean expects `core.verifier.types.VerifierResult`
        # (dataclass), but `deps.verifier` returns
        # `core.pipeline_contracts.VerifierResult` (Pydantic). The two have
        # compatible duck-typed interfaces (`.ok`, `.error_count()`,
        # `.issues`); collapsing to a single type is a separate refactor
        # WP — for now bridge via cast at the boundary.
        try:
            refined_specialist, _cycles = refine_until_clean(
                stage_name="specialist",
                initial_output=specialist_output,
                stage_runner=_re_run_specialist,
                verifier=cast(
                    Callable[[Any], Any],
                    lambda s: deps.verifier({**snapshot, "specialist": s}),
                ),
                max_cycles=2,
                initial_result=cast(Any, initial_result),
            )
            # WP-CORE-24: refiner cleared. Record cycles_used for manifest.
            # WP-CORE-30: clean path has no exhaustion → no flapping.
            _record_refiner_metrics_safely(
                cycles_used=_cycles,
                exhausted=False,
                residual_count=0,
                max_cycles=2,
                flapped=False,
                cycle_history_len=_cycles,
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
                    architect_feedback_local = late_arch_issues
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
            # WP-CORE-24 + WP-CORE-30: refiner exhausted. Record
            # cycles_used + residual + flapping signal + history length.
            exc_history = getattr(exc, "cycle_history", []) or []
            _record_refiner_metrics_safely(
                cycles_used=getattr(exc, "cycles_attempted", 2),
                exhausted=True,
                residual_count=len(exc.issues),
                max_cycles=2,
                flapped=_detect_flapping(exc_history),
                cycle_history_len=len(exc_history),
            )

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

    # WP-CORE-20b: wrap synthesizer call too. Post-check (empty-model
    # guard) stays OUTSIDE the wrap so an empty-bounded-contexts model
    # produces a SynthesizerEmptyModelError rather than a stage-fail
    # log + a different exception type.
    with _optional_stage("synthesizer"):
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
    # `arch` (narrowed ArchitectOutput) and `refined_specialist` (narrowed
    # non-empty by the guard above) are both bound here; the critique loop
    # reuses them in Task 8.
    model = _apply_context_map(model, deps, scout)
    return model, arch, refined_specialist
