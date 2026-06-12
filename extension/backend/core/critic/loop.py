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


def _canonical_target(target_ref: str) -> str:
    """Canonicalize relationship pairs so A->B == B->A for flap detection."""
    if target_ref.startswith("relationship:") and "->" in target_ref:
        body = target_ref.split(":", 1)[1]
        a, b = (p.strip() for p in body.split("->", 1))
        return "relationship:" + "->".join(sorted((a, b)))
    return target_ref


def _relationship_pair_in(f: CritiqueFinding, names: set) -> bool:
    body = f.target_ref.split(":", 1)[-1]
    if "->" not in body:
        return False
    a, b = (p.strip() for p in body.split("->", 1))
    return a in names and b in names


def _active_priorities() -> set[str]:
    threshold = os.getenv("DDD_CRITIC_THRESHOLD", "HIGH_MED").upper()
    return {"high"} if threshold == "HIGH" else {"high", "medium"}


def findings_signature(findings: List[CritiqueFinding]) -> Tuple:
    active = _active_priorities()
    return tuple(sorted(
        (f.finding_type, _canonical_target(f.target_ref))
        for f in findings if f.priority in active
    ))


def _has_high_or_medium(report: CriticReport) -> bool:
    active = _active_priorities()
    return any(f.priority in active for f in report.findings)


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

    print("\n" + "=" * 70)
    print("🔍 [CRITIC LOOP] Initiating Holistic DDD Critique Loop")
    print(f"   Max Cycles configured: {max_cycles}")
    print("=" * 70 + "\n")

    # --- cycle 0 -----------------------------------------------------------
    print("🔄 [CRITIC CYCLE 0] Initial Generation...")
    model, arch, specialist = _generate_once(scout, deps, srs_path)
    
    print("\n👉 [CRITIC CYCLE 0] Evaluating generated domain model against SRS...")
    try:
        report = deps.critic(model, scout, history)
    except CriticError as exc:
        print(f"❌ [CRITIC CYCLE 0] Critic failed: {exc}")
        return _finalize_failed(model, exc, cycles_used=1,
                                score_trace=[], count_trace=[])

    best_model, best_report, best_cycle = model, report, 0
    score_trace.append(critique_score(report.findings))
    count_trace.append(len(report.findings))
    history.append(CritiqueCycleMemory(
        cycle=0, findings_summary=_findings_summary(report),
        diff_summary="initial model",
    ))

    print(f"\n📊 [CYCLE 0 SUMMARY] Score: {critique_score(report.findings)} | Findings count: {len(report.findings)}")
    high_med = [f for f in report.findings if f.priority in ("high", "medium")]
    if high_med:
        print("   ⚠️  Active Critiques:")
        for f in high_med:
            emoji = "❌" if f.priority == "high" else "⚠️"
            print(f"     {emoji} [{f.priority.upper()}] {f.finding_type} ({f.target_ref}): {f.rationale}")
    else:
        print("   ✅ No high/medium priority critiques found.")

    outcome = "converged"
    prev_signature = findings_signature(report.findings)

    # --- revision cycles ---------------------------------------------------
    for cycle in range(1, max_cycles):
        if not _has_high_or_medium(report):
            outcome = "converged"
            print("\n✅ [CRITIC LOOP] Conformance achieved! No high/medium issues remaining.")
            break
        structural, content, relationship, _advisory = partition_findings(report.findings)
        
        print(f"\n🔄 [CRITIC CYCLE {cycle}] Resolving issues and refining model...")
        if structural:
            print(f"   🛠️  Action: Structural issues detected. Triggering Bounded Context regeneration.")
        elif content:
            print(f"   🔧  Action: Content issues detected. Triggering targeted Specialist refinement.")
        elif relationship:
            print(f"   🔗  Action: Relationship issues detected. Applying Context Map updates.")
        
        try:
            from core.orchestration.pipeline import _generate_once, _apply_context_map
            if structural:
                new_model, arch, specialist = _generate_once(
                    scout, deps, srs_path,
                    architect_feedback=adapt_structural_to_issues(structural),
                )
            elif content:  # content-only → reuse architecture, targeted specialist rerun
                specialist = deps.specialist_with_feedback(
                    arch, scout, specialist, adapt_content_to_issues(content),
                )
                new_model = deps.synthesizer(specialist)
                new_model = _apply_context_map(new_model, deps, scout)
            else:  # relationship-only / advisory-only → no producer rerun
                new_model = model
            if relationship:  # every cycle: remap surviving-context relationships
                survivors = {bc.context_name for bc in new_model.bounded_contexts}
                rel_live = [f for f in relationship if _relationship_pair_in(f, survivors)]
                if rel_live:
                    print("   🔗 Applying Context Map relationship fixes...")
                    new_model = _apply_context_map(new_model, deps, scout, feedback=rel_live)
            
            print(f"\n👉 [CRITIC CYCLE {cycle}] Evaluating refined model...")
            new_report = deps.critic(new_model, scout, history)
        except (CriticError, PipelineError) as exc:
            print(f"⚠️  [CRITIC CYCLE {cycle}] Warning: Revision cycle failed with error: {exc}. Falling back to best model.")
            return _finalize_failed(best_model, exc, cycles_used=cycle + 1,
                                    score_trace=score_trace, count_trace=count_trace,
                                    best_report=best_report, best_cycle=best_cycle)

        score_trace.append(critique_score(new_report.findings))
        count_trace.append(len(new_report.findings))
        history.append(CritiqueCycleMemory(
            cycle=cycle, findings_summary=_findings_summary(new_report),
            diff_summary=model_diff_summary(model, new_model),
        ))

        print(f"\n📊 [CYCLE {cycle} SUMMARY] Score: {critique_score(new_report.findings)} | Findings count: {len(new_report.findings)}")
        high_med = [f for f in new_report.findings if f.priority in ("high", "medium")]
        if high_med:
            print("   ⚠️  Active Critiques:")
            for f in high_med:
                emoji = "❌" if f.priority == "high" else "⚠️"
                print(f"     {emoji} [{f.priority.upper()}] {f.finding_type} ({f.target_ref}): {f.rationale}")
        else:
            print("   ✅ No high/medium priority critiques found.")

        if critique_score(new_report.findings) < critique_score(best_report.findings):
            print(f"   🏆 New best model found in cycle {cycle}!")
            best_model, best_report, best_cycle = new_model, new_report, cycle

        sig = findings_signature(new_report.findings)
        if sig == prev_signature:
            outcome = "flapped"
            print(f"\n⚠️  [CRITIC LOOP] Flapping detected in cycle {cycle} (issues repeated). Terminating loop.")
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

    print("\n" + "=" * 70)
    print(f"🏁 [CRITIC LOOP COMPLETED] Outcome: {outcome.upper()}")
    print(f"   Total Cycles Used: {len(score_trace)} | Best Cycle: {best_cycle}")
    print(f"   Final Best Critique Score: {best_report.score}")
    print("=" * 70 + "\n")

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
