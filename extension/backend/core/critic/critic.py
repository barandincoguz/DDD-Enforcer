"""run_critic: one schema-enforced LLM call that judges DDD design quality.

Pure evaluator — never mutates the model. Maps the LLM's ProposedFindings to
persisted CritiqueFindings, dropping ones whose target_ref does not resolve and
trimming out-of-range evidence indices. Raises CriticError on json_failed."""
from typing import Any, List, Optional, Set
from core.schemas import CritiqueFinding, CriticReport, CriticLoopTrace, DomainModel
from core.pipeline_contracts import ScoutOutput
from core.critic.types import CriticResponse, ProposedFinding, CritiqueCycleMemory
from core.critic.prompt import build_critique_prompt
from core.critic.errors import CriticError


_RELATIONSHIP_TYPES = {"WRONG_RELATIONSHIP_TYPE", "ILLEGAL_DEPENDENCY", "MISSING_RELATIONSHIP"}


def _valid_targets(model: DomainModel) -> Set[str]:
    targets: Set[str] = set()
    for bc in model.bounded_contexts:
        targets.add(f"context:{bc.context_name}")
        for e in bc.ubiquitous_language.entities:
            targets.add(f"entity:{bc.context_name}.{e.name}")
    return targets


def _relationship_contexts_valid(target_ref: str, context_names: Set[str]) -> bool:
    """'relationship:A->B' -> True iff both A and B are existing context names."""
    body = target_ref.split(":", 1)[-1]
    if "->" not in body:
        return False
    a, b = (p.strip() for p in body.split("->", 1))
    return a in context_names and b in context_names


def _map_finding(
    pf: ProposedFinding, valid_targets: Set[str], scout_indices: Set[int],
    *, model: DomainModel,
) -> Optional[CritiqueFinding]:
    if pf.finding_type in _RELATIONSHIP_TYPES:
        context_names = {bc.context_name for bc in model.bounded_contexts}
        if not _relationship_contexts_valid(pf.target_ref, context_names):
            return None
    elif pf.target_ref not in valid_targets:
        return None
    evidence = [i for i in pf.evidence_sentence_indices if i in scout_indices]
    return CritiqueFinding(
        finding_type=pf.finding_type, priority=pf.priority,
        target_ref=pf.target_ref, rationale=pf.rationale,
        proposed_revision=pf.proposed_revision,
        evidence_sentence_indices=evidence,
    )


def run_critic(
    model: DomainModel,
    scout: ScoutOutput,
    history: List[CritiqueCycleMemory],
    *,
    client: Any,
    stage_cfg: Any,
) -> CriticReport:
    """Run one critique pass. `client` exposes structured_output(...);
    `stage_cfg` exposes .model_id/.temperature/.seed.

    Note: `loop` is a placeholder single-cycle trace; the loop driver
    (run_critique_loop) overwrites it with the real multi-cycle trace before
    persisting. score is left 0.0 here and computed by the loop driver."""
    prompt = build_critique_prompt(model, scout, history)
    response = client.structured_output(
        messages=[{"role": "user", "content": prompt}],
        schema=CriticResponse,
        model=stage_cfg.model_id,
        temperature=stage_cfg.temperature,
        seed=stage_cfg.seed,
    )
    if response.json_failed or not isinstance(response.parsed, CriticResponse):
        raise CriticError(reason=response.json_fail_reason or "empty_parse")

    parsed: CriticResponse = response.parsed
    valid_targets = _valid_targets(model)
    scout_indices = {s.index for s in scout.sentences}
    findings: List[CritiqueFinding] = []
    malformed = 0
    for pf in parsed.findings:
        mapped = _map_finding(pf, valid_targets, scout_indices, model=model)
        if mapped is None:
            malformed += 1
        else:
            findings.append(mapped)

    return CriticReport(
        model_id=stage_cfg.model_id,
        findings=findings,
        malformed_findings=malformed,
        loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"),
    )
