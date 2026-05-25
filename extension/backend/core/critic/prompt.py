"""Build the holistic-critique prompt: model + Scout grounding + Reflexion
history, instructing CoT reasoning before structured findings."""
import json
from typing import List
from core.schemas import DomainModel
from core.pipeline_contracts import ScoutOutput
from core.critic.types import CritiqueCycleMemory

_INSTRUCTIONS = """You are a senior Domain-Driven Design reviewer. You are given a
domain model assembled from a requirements document, plus the numbered source
sentences it was derived from.

Think step by step (this is your `analysis` field), reviewing in order:
1. Bounded contexts: cohesion, over-splitting, god-contexts, boundary smells.
2. Entities: anemic (no behavior implied), misplaced (belong in another context).
3. Aggregates: missing consistency boundaries.
4. Naming: ubiquitous-language smells.
5. Context map: for each relationship, is the strategic pattern correct (e.g.
   is it really Conformist, or should it be ACL?), is the direction right, and
   are any required relationships MISSING or any dependency ILLEGAL?
Then emit `findings`. For each finding set:
- finding_type (one of the allowed values),
- priority: high (serious structural/design flaw), medium (worth fixing),
  low (cosmetic / advisory only),
- target_ref: "context:<Name>" or "entity:<Context>.<Entity>",
- rationale (why it is a problem), proposed_revision (what to change),
- evidence_sentence_indices: source sentence indices that justify the finding
  (use the numbered list; [] if none).
  Relationship findings use finding_type WRONG_RELATIONSHIP_TYPE /
  ILLEGAL_DEPENDENCY / MISSING_RELATIONSHIP with target_ref "relationship:<A>-><B>".
Do NOT rewrite the model. Only critique it. Be specific and grounded.
"""


def _serialize_model(model: DomainModel) -> str:
    compact = {
        "project_name": model.project_name,
        "bounded_contexts": [
            {
                "context_name": bc.context_name,
                "description": bc.description,
                "allowed_dependencies": bc.allowed_dependencies,
                "entities": [
                    {"name": e.name, "description": e.description,
                     "confidence": e.confidence}
                    for e in bc.ubiquitous_language.entities
                ],
                "aggregates": [
                    {"name": a.name, "members": a.members}
                    for a in (bc.ubiquitous_language.aggregates or [])
                ],
            }
            for bc in model.bounded_contexts
        ],
    }
    if model.context_map and model.context_map.relationships:
        compact["context_map"] = [
            {"context_a": r.context_a, "context_b": r.context_b,
             "relationship_type": r.relationship_type, "upstream": r.upstream}
            for r in model.context_map.relationships
        ]
    return json.dumps(compact, indent=2, ensure_ascii=False)


def _serialize_scout(scout: ScoutOutput) -> str:
    return "\n".join(f"[{s.index}] {s.text}" for s in scout.sentences)


def _serialize_history(history: List[CritiqueCycleMemory]) -> str:
    if not history:
        return ""
    lines = ["PREVIOUS CYCLES (do not re-report already-addressed issues):"]
    for mem in history:
        lines.append(f"- cycle {mem.cycle} findings: " + "; ".join(mem.findings_summary))
        lines.append(f"  producer changes since: {mem.diff_summary}")
    return "\n".join(lines) + "\n\n"


def build_critique_prompt(
    model: DomainModel, scout: ScoutOutput, history: List[CritiqueCycleMemory],
) -> str:
    return (
        _INSTRUCTIONS
        + "\n\n"
        + _serialize_history(history)
        + "DOMAIN MODEL UNDER REVIEW:\n"
        + _serialize_model(model)
        + "\n\nNUMBERED SOURCE SENTENCES:\n"
        + _serialize_scout(scout)
    )
