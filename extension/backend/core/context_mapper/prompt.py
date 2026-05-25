"""Build the context-mapping prompt: contexts + ubiquitous language + numbered
Scout sentences (grounding) + the 9-pattern taxonomy. Intent-level only — NO
AST import topology (it does not exist at generation time)."""
from typing import List
from core.schemas import DomainModel, ContextMap, CritiqueFinding
from core.pipeline_contracts import ScoutOutput

_TAXONOMY = """STRATEGIC DDD RELATIONSHIP PATTERNS (choose exactly one per related pair):
- CUSTOMER_SUPPLIER: downstream's needs drive an upstream supplier (set `upstream`).
- CONFORMIST: downstream conforms to upstream's model with no translation (set `upstream`).
- ANTI_CORRUPTION_LAYER: downstream translates upstream's model behind an ACL (set `upstream`).
- OPEN_HOST_SERVICE: upstream exposes a defined protocol for consumers (set `upstream`).
- PUBLISHED_LANGUAGE: integration via a shared well-documented language (set `upstream` = definer).
- PARTNERSHIP: two contexts succeed/fail together, co-evolved (upstream=null).
- SHARED_KERNEL: a shared subset of the model (upstream=null).
- SEPARATE_WAYS: no integration; they must NOT depend on each other (upstream=null).
- BIG_BALL_OF_MUD: tangled, unclear boundaries — a smell to flag (upstream=null).
Only emit a relationship for pairs that are genuinely related. Omit unrelated pairs."""

_INSTRUCTIONS = """You are a senior Domain-Driven Design strategist. Given the bounded
contexts of a system (with their ubiquitous language) and the numbered requirement
sentences they came from, identify the strategic relationships between context pairs.

Think step by step in `analysis`, then emit `relationships`. For each:
- context_a, context_b: the two context names (must be from the list below),
- relationship_type: one taxonomy value,
- upstream: context_a or context_b for directional types; null otherwise,
- rationale: why this pattern + direction, in DDD terms,
- evidence_sentence_indices: supporting numbered sentence indices ([] or [-1] if none).
"""


def _serialize_contexts(model: DomainModel) -> str:
    lines: List[str] = []
    for bc in model.bounded_contexts:
        ul = bc.ubiquitous_language
        ents = ", ".join(e.name for e in ul.entities) or "(none)"
        lines.append(f"- {bc.context_name}: {bc.description or '(no description)'} "
                     f"| entities: {ents}")
    return "\n".join(lines)


def _serialize_scout(scout: ScoutOutput) -> str:
    return "\n".join(f"[{s.index}] {s.text}" for s in scout.sentences)


def build_map_prompt(model: DomainModel, scout: ScoutOutput) -> str:
    return (
        _INSTRUCTIONS + "\n\n" + _TAXONOMY + "\n\nBOUNDED CONTEXTS:\n"
        + _serialize_contexts(model)
        + "\n\nNUMBERED SOURCE SENTENCES:\n" + _serialize_scout(scout)
    )


def _serialize_prev_map(prev: ContextMap) -> str:
    if not prev.relationships:
        return "(none)"
    return "\n".join(
        f"- {r.context_a}/{r.context_b}: {r.relationship_type} "
        f"(upstream={r.upstream})" for r in prev.relationships)


def _serialize_feedback(findings: List[CritiqueFinding]) -> str:
    return "\n".join(
        f"- {f.target_ref}: {f.rationale} | suggested: {f.proposed_revision}"
        for f in findings)


def build_remap_prompt(
    model: DomainModel, scout: ScoutOutput, prev: ContextMap,
    findings: List[CritiqueFinding],
) -> str:
    return (
        build_map_prompt(model, scout)
        + "\n\nYOUR PREVIOUS MAP:\n" + _serialize_prev_map(prev)
        + "\n\nCRITIC FEEDBACK (revise the affected relationships, keep the rest):\n"
        + _serialize_feedback(findings)
    )
