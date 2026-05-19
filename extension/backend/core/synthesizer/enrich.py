"""Narrow LLM enrichment: synonyms_to_avoid + allowed_dependencies.

ONE LLM call PER bounded context to enrich entity synonyms_to_avoid
within that context. ONE additional LLM call to disambiguate
allowed_dependencies across contexts.

Keeps payloads small; per-context retry granularity; total cost is
N+1 narrow calls vs the old 1 omnibus call (smaller per-call payloads,
lower truncation risk).
"""

import json
import re
from typing import List
from core.pipeline_contracts import SpecialistAnalysis
from core.schemas import DomainModel, BoundedContext
from configs.models import stage_config


def enrich_synonyms_and_dependencies(
    skeleton: DomainModel,
    analyses: List[SpecialistAnalysis],
    llm_client,
) -> DomainModel:
    """Per-context narrow enrichment of synonyms_to_avoid + cross-context
    allowed_dependencies."""
    for bc in skeleton.bounded_contexts:
        _enrich_context_synonyms(bc, llm_client)

    _infer_and_enrich_dependencies(skeleton)
    return skeleton


def _enrich_context_synonyms(bc: BoundedContext, llm_client) -> None:
    """For each entity in bc, get LLM-emitted synonyms_to_avoid."""
    entities = bc.ubiquitous_language.entities
    if not entities:
        return

    stage_model = stage_config("Synthesizer").model_id

    prompt = (
        f"You are filling in `synonyms_to_avoid` for entities in the "
        f"bounded context `{bc.context_name}`.\n\n"
        f"For each entity, list 2-4 common alternative names that "
        f"developers might use but should NOT in this context.\n\n"
        f"ENTITIES:\n"
    )
    for e in entities:
        prompt += f"- {e.name}: {e.description}\n"

    prompt += (
        "\nRespond with JSON: "
        "{\"entities\": [{\"name\": \"EntityName\", "
        "\"synonyms_to_avoid\": [\"Synonym1\", \"Synonym2\"]}, ...]}"
    )

    try:
        response = llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=stage_model,
            temperature=0.1,
            response_mime_type="application/json",
        )
        parsed = json.loads(response.content)
    except (json.JSONDecodeError, ValueError, RuntimeError, KeyError) as exc:
        # Enrichment failure is NOT fatal — entities still have all
        # required fields. Log and continue with synonyms_to_avoid=None.
        print(f"  ⚠️  Synonym enrichment failed for {bc.context_name}: {type(exc).__name__}: {exc}")
        return  # entities keep synonyms_to_avoid=None
    by_name = {item["name"]: item.get("synonyms_to_avoid", []) for item in parsed.get("entities", [])}
    for e in entities:
        if e.name in by_name:
            e.synonyms_to_avoid = by_name[e.name]


def _infer_and_enrich_dependencies(skeleton: DomainModel) -> None:
    """Infer cross-context dependencies by scanning entity-mention overlap.
    Currently pure text scanning; LLM disambiguation is a future enhancement."""
    context_names = {bc.context_name for bc in skeleton.bounded_contexts}
    for bc in skeleton.bounded_contexts:
        deps = set()
        for e in bc.ubiquitous_language.entities:
            # Scan description + justification for mentions of other contexts
            text = (e.description or "") + " " + (e.justification or "")
            for other in context_names - {bc.context_name}:
                pattern = re.compile(rf"\b{re.escape(other)}\b", re.IGNORECASE)
                if pattern.search(text):
                    deps.add(other)
        bc.allowed_dependencies = sorted(deps) if deps else None
