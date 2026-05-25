"""Bounded critique→revise loop (Topology A).

Cycle 0 generates + critiques. Each subsequent cycle routes the prior cycle's
high/medium findings back to the producers, regenerates, and re-critiques.
Returns the best (lowest-score) model with its CriticReport attached.

Stops on: converged (no high/med), flapped (repeated finding signature),
exhausted (cycle cap), or failed (CriticError / regeneration error → non-fatal,
returns best-so-far)."""
import os
from typing import Any, List, Optional, Tuple
from core.schemas import CritiqueFinding, CriticReport, CriticLoopTrace, DomainModel
from core.pipeline_contracts import ScoutOutput
from core.critic.errors import CriticError
from core.orchestration.errors import PipelineError
from core.critic.types import CritiqueCycleMemory
from core.critic.routing import (
    partition_findings, adapt_structural_to_issues, adapt_content_to_issues,
    model_diff_summary,
)

_PRIORITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}


def critique_score(findings: List[CritiqueFinding]) -> float:
    return sum(_PRIORITY_WEIGHT[f.priority] for f in findings)


def findings_signature(findings: List[CritiqueFinding]) -> Tuple:
    return tuple(sorted(
        (f.finding_type, f.target_ref)
        for f in findings if f.priority in ("high", "medium")
    ))


def _has_high_or_medium(report: CriticReport) -> bool:
    return any(f.priority in ("high", "medium") for f in report.findings)


def _max_cycles() -> int:
    try:
        return max(1, int(os.getenv("DDD_CRITIC_MAX_CYCLES", "3")))
    except ValueError:
        return 3


def _findings_summary(report: CriticReport) -> List[str]:
    return [f"{f.priority} {f.finding_type} {f.target_ref}" for f in report.findings]


def run_critique_loop(
    scout: ScoutOutput, deps: Any, srs_path: Optional[str],
) -> DomainModel:
    from core.orchestration.pipeline import _generate_once

    max_cycles = _max_cycles()
    history: List[CritiqueCycleMemory] = []
    score_trace: List[float] = []
    count_trace: List[int] = []

    # --- cycle 0 -----------------------------------------------------------
    model, arch, specialist = _generate_once(scout, deps, srs_path)
    try:
        report = deps.critic(model, scout, history)
    except CriticError as exc:
        return _finalize_failed(model, exc, cycles_used=1,
                                score_trace=[], count_trace=[])

    best_model, best_report, best_cycle = model, report, 0
    score_trace.append(critique_score(report.findings))
    count_trace.append(len(report.findings))
    history.append(CritiqueCycleMemory(
        cycle=0, findings_summary=_findings_summary(report),
        diff_summary="initial model",
    ))

    outcome = "converged"
    prev_signature = findings_signature(report.findings)

    # --- revision cycles ---------------------------------------------------
    for cycle in range(1, max_cycles):
        if not _has_high_or_medium(report):
            outcome = "converged"
            break
        structural, content, _advisory = partition_findings(report.findings)
        try:
            if structural:
                new_model, arch, specialist = _generate_once(
                    scout, deps, srs_path,
                    architect_feedback=adapt_structural_to_issues(structural),
                )
            else:  # content-only → reuse architecture, targeted specialist rerun
                specialist = deps.specialist_with_feedback(
                    arch, scout, specialist, adapt_content_to_issues(content),
                )
                new_model = deps.synthesizer(specialist)
            new_report = deps.critic(new_model, scout, history)
        except (CriticError, PipelineError) as exc:
            # Non-fatal: a revision-cycle critic failure (CriticError) OR a
            # regeneration failure (PipelineError subclasses such as
            # ArchitectGroundingError / SynthesizerEmptyModelError /
            # RefinementExhaustedError raised by _generate_once / synthesizer
            # / specialist_with_feedback) falls back to the best model so far.
            # Cycle-0 generation failure stays fatal (no prior good model).
            return _finalize_failed(best_model, exc, cycles_used=cycle + 1,
                                    score_trace=score_trace, count_trace=count_trace,
                                    best_report=best_report, best_cycle=best_cycle)

        score_trace.append(critique_score(new_report.findings))
        count_trace.append(len(new_report.findings))
        history.append(CritiqueCycleMemory(
            cycle=cycle, findings_summary=_findings_summary(new_report),
            diff_summary=model_diff_summary(model, new_model),
        ))

        if critique_score(new_report.findings) < critique_score(best_report.findings):
            best_model, best_report, best_cycle = new_model, new_report, cycle

        sig = findings_signature(new_report.findings)
        if sig == prev_signature:
            outcome = "flapped"
            model, report = new_model, new_report
            break
        prev_signature = sig
        model, report = new_model, new_report
    else:
        outcome = "exhausted" if _has_high_or_medium(report) else "converged"

    best_report.score = critique_score(best_report.findings)
    best_report.loop = CriticLoopTrace(
        cycles_used=len(score_trace), best_cycle=best_cycle, outcome=outcome,
        score_per_cycle=score_trace, findings_count_per_cycle=count_trace,
    )
    best_model.critic_report = best_report
    return best_model


def _finalize_failed(
    model: DomainModel, exc: Exception, *, cycles_used: int,
    score_trace: List[float], count_trace: List[int],
    best_report: Optional[CriticReport] = None, best_cycle: int = 0,
) -> DomainModel:
    loop = CriticLoopTrace(
        cycles_used=cycles_used, best_cycle=best_cycle, outcome="failed",
        score_per_cycle=score_trace, findings_count_per_cycle=count_trace,
    )
    report = best_report or CriticReport(model_id="unknown", findings=[], loop=loop)
    report.score = critique_score(report.findings)
    report.error = str(exc)
    report.loop = loop
    model.critic_report = report
    return model
