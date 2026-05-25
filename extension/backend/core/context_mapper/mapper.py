"""run_context_mapper: one schema-enforced LLM call producing a typed ContextMap.

Pure producer — maps ProposedRelationships to ContextRelationships, trimming
evidence to valid Scout indices (or the -1 inference sentinel) and dropping
relationships that fail schema validation (non-fatal). Raises ContextMapperError
on json_failed."""
from typing import Any, List, Optional
from pydantic import ValidationError
from core.schemas import ContextMap, ContextRelationship, CritiqueFinding
from core.pipeline_contracts import ScoutOutput
from core.context_mapper.types import ContextMapResponse, ProposedRelationship
from core.context_mapper.prompt import build_map_prompt, build_remap_prompt
from core.context_mapper.errors import ContextMapperError


def _map_one(pr: ProposedRelationship, scout_indices: set) -> Optional[ContextRelationship]:
    seen = set()
    evidence: List[int] = []
    for i in pr.evidence_sentence_indices:
        if (i in scout_indices or i == -1) and i not in seen:
            seen.add(i)
            evidence.append(i)
    try:
        return ContextRelationship(
            context_a=pr.context_a, context_b=pr.context_b,
            relationship_type=pr.relationship_type, upstream=pr.upstream,
            rationale=pr.rationale, evidence_sentence_indices=evidence,
        )
    except ValidationError:
        return None


def run_context_mapper(
    model: Any,
    scout: ScoutOutput,
    feedback: Optional[List[CritiqueFinding]],
    *,
    client: Any,
    stage_cfg: Any,
    _prev_map: Optional[ContextMap] = None,
) -> ContextMap:
    """`feedback` falsy → fresh map; else revise via build_remap_prompt using
    `_prev_map` (or the model's existing context_map; defaults to empty)."""
    if feedback:
        prev = _prev_map or getattr(model, "context_map", None) or ContextMap(model_id=stage_cfg.model_id)
        prompt = build_remap_prompt(model, scout, prev, list(feedback))
    else:
        prompt = build_map_prompt(model, scout)

    response = client.structured_output(
        messages=[{"role": "user", "content": prompt}],
        schema=ContextMapResponse,
        model=stage_cfg.model_id,
        temperature=stage_cfg.temperature,
        seed=stage_cfg.seed,
    )
    if response.json_failed or not isinstance(response.parsed, ContextMapResponse):
        raise ContextMapperError(reason=response.json_fail_reason or "empty_parse")

    parsed: ContextMapResponse = response.parsed
    scout_indices = {s.index for s in scout.sentences}
    relationships: List[ContextRelationship] = []
    for pr in parsed.relationships:
        mapped = _map_one(pr, scout_indices)
        if mapped is not None:
            relationships.append(mapped)

    return ContextMap(model_id=stage_cfg.model_id, relationships=relationships)
