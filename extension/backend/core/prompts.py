"""
Prompt builders for domain model generation.

Prompts are intentionally centralized so the generation pipeline can be tuned
without burying production prompt text inside orchestration code.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def _json_block(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def build_scout_prompt(chunk_index: int, total_chunks: int, requirement_chunk: List[Dict[str, Any]]) -> str:
    example = {
        "requirements": [
            {
                "requirement_id": "REQ-001",
                "title": "Register to the website",
                "category": "functional",
                "description": "A non-registered user can create an account and verify email.",
                "actor": "NonRegisteredUser",
                "evidence_ids": ["E0001", "E0002"],
            }
        ],
        "actors": [
            {
                "name": "NonRegisteredUser",
                "description": "Visitor who can browse and register.",
                "evidence_ids": ["E0001"],
            }
        ],
        "entities": [
            {
                "name": "User",
                "description": "Account holder in the platform.",
                "evidence_ids": ["E0002"],
            }
        ],
        "constraints": [
            {
                "text": "Users must verify email before platform access.",
                "category": "security",
                "evidence_ids": ["E0002"],
            }
        ],
        "tables": [
            {
                "name": "Users",
                "description": "Stores user credentials and profile data.",
                "evidence_ids": ["E0003"],
            }
        ],
        "capabilities": [
            {
                "name": "RegisterAccount",
                "description": "Create an account and complete email verification.",
                "actor": "NonRegisteredUser",
                "evidence_ids": ["E0001", "E0002"],
            }
        ],
        "evidence_spans": [
            {
                "evidence_id": "E0001",
                "document": "SRS.docx",
                "section": "Functional Requirement",
                "excerpt": "TITLE: Register to the website",
                "line": 77,
                "requirement_id": "REQ-001",
            }
        ],
    }
    return f"""You are Scout, a DDD requirements normalizer.

Your task is to normalize structured SRS requirement records into evidence-backed domain signals.

Hard rules:
- Use ONLY the provided requirement records.
- Never invent actors, entities, tables, capabilities, or constraints.
- Every extracted item must carry at least one valid evidence_id from the input.
- If an item is uncertain, omit it instead of guessing.
- Keep names in PascalCase for actors/entities/capabilities when appropriate.

Input chunk {chunk_index}/{total_chunks}:
{_json_block(requirement_chunk)}

Output must match this example shape exactly:
{_json_block(example)}

Return valid JSON only."""


def build_architect_prompt(scout_summary: Dict[str, Any]) -> str:
    example = {
        "contexts": [
            {
                "context_name": "IdentityAndAccess",
                "description": "Authentication, registration, and profile ownership.",
                "ownership_rationale": "Owns account lifecycle and identity-related constraints.",
                "included_capabilities": ["RegisterAccount", "Login"],
                "excluded_capabilities": ["SearchProducts"],
                "primary_entities": ["User"],
                "allowed_dependencies": [],
                "evidence_ids": ["E0001", "E0002"],
            }
        ]
    }
    return f"""You are Architect, a senior Domain-Driven Design strategist.

Goal:
- Partition the domain into bounded contexts.
- Assign primary ownership of capabilities and entities.
- Keep the map minimal, non-overlapping, and evidence-backed.

Hard rules:
- Use ONLY actors, entities, capabilities, constraints, and tables found in the input.
- Each entity must belong to one primary context.
- Each capability should appear in at least one context, either included or excluded.
- Use business names, not technical layer names.
- Every context must include evidence_ids from the input.

Normalized domain signals:
{_json_block(scout_summary)}

Output shape example:
{_json_block(example)}

Return valid JSON only."""


def build_specialist_prompt(
    context_name: str,
    context_proposal: Dict[str, Any],
    relevant_requirements: List[Dict[str, Any]],
    scout_summary: Dict[str, Any],
) -> str:
    example = {
        "context": context_name,
        "description": "Context summary.",
        "actors": [
            {
                "name": "RegisteredUser",
                "description": "Actor who can perform authenticated actions.",
                "evidence_ids": ["E0001"],
            }
        ],
        "capabilities": [
            {
                "name": "AddProduct",
                "description": "List a product for sale.",
                "actor": "RegisteredUser",
                "evidence_ids": ["E0002"],
            }
        ],
        "aggregate_roots": [
            {
                "name": "ProductListing",
                "description": "Consistency boundary for a seller-managed listing.",
                "evidence_ids": ["E0002"],
            }
        ],
        "entities": [
            {
                "name": "Product",
                "description": "Sellable item exposed in the platform.",
                "evidence_ids": ["E0002"],
            }
        ],
        "value_objects": [
            {
                "name": "Price",
                "description": "Product monetary value.",
                "attributes": ["amount", "currency"],
                "evidence_ids": ["E0003"],
            }
        ],
        "business_rules": [
            {
                "text": "Only registered users can add products.",
                "category": "security",
                "evidence_ids": ["E0004"],
            }
        ],
        "domain_events": [
            {
                "name": "ProductListed",
                "description": "A product is published for discovery.",
                "evidence_ids": ["E0002"],
            }
        ],
        "domain_services": [
            {
                "name": "ListingPublicationService",
                "description": "Publishes validated listings.",
                "evidence_ids": ["E0002"],
            }
        ],
        "allowed_dependencies": [],
        "evidence_ids": ["E0002", "E0004"],
    }
    return f"""You are Specialist, a bounded-context analyst.

Focus on ONE context only: {context_name}

Hard rules:
- Use ONLY the provided evidence-backed requirement records.
- If information is absent, return an empty array instead of inferring from generic e-commerce knowledge.
- Every extracted item must reference evidence_ids from the input.
- Actors and capabilities must align with the context proposal.
- Aggregate roots must be stable business consistency boundaries, not UI or persistence concepts.
- Do not claim ownership of an entity unless it appears in primary_entities or the evidence clearly defines a context-specific projection/value object.
- If another context likely owns a concept, omit it here instead of duplicating ownership.

Context proposal:
{_json_block(context_proposal)}

Relevant requirement records:
{_json_block(relevant_requirements)}

Global scout signals:
{_json_block(scout_summary)}

Output shape example:
{_json_block(example)}

Return valid JSON only."""


def build_synthesizer_prompt(
    context_analyses: List[Dict[str, Any]],
    scout_summary: Dict[str, Any],
    glossary_aliases: Dict[str, List[str]],
    banned_terms: List[str],
) -> str:
    example = {
        "schema_version": "2.0.0",
        "project_name": "KinmailDomainModel",
        "project_metadata": {
            "version": "1.0.0",
            "generated_at": "SET_BY_CODE",
            "description": "Evidence-backed domain model synthesized from the SRS.",
        },
        "bounded_contexts": [
            {
                "context_name": "IdentityAndAccess",
                "description": "Registration, login, and user profile ownership.",
                "allowed_dependencies": [],
                "actors": [
                    {
                        "name": "NonRegisteredUser",
                        "description": "Visitor who can register and browse.",
                        "confidence": 0.8,
                        "evidence_ids": ["E0001"],
                        "sources": [],
                    }
                ],
                "capabilities": [],
                "ubiquitous_language": {
                    "entities": [
                        {
                            "name": "User",
                            "description": "Platform account holder.",
                            "confidence": 0.8,
                            "evidence_ids": ["E0001"],
                            "sources": [],
                            "synonyms_to_avoid": ["Account"],
                        }
                    ],
                    "value_objects": [],
                    "services": [],
                    "aggregates": [],
                    "domain_events": [],
                },
                "business_rules": [],
                "external_references": [],
                "evidence_ids": ["E0001"],
                "evidence": [],
            }
        ],
        "global_rules": {
            "naming_convention": "PascalCase",
            "banned_global_terms": ["Manager", "Util", "Helper", "Data", "Info"],
            "cross_cutting_constraints": [],
            "assumptions": [],
        },
    }
    return f"""You are Synthesizer, a chief architect producing the final domain model.

Goal:
- Merge context analyses into ONE complete model without dropping supported fields.
- Preserve evidence_ids on all extracted items.
- Keep empty arrays when information is missing.

Hard rules:
- Do not invent contexts, entities, actors, capabilities, rules, or events that are not supported by the analyses.
- Do not drop fields present in the analyses.
- Keep one primary owner per entity and aggregate; if a context only references a shared concept, keep it in external_references instead of duplicating ownership.
- project_metadata.generated_at will be set by code; use the literal string SET_BY_CODE.
- sources and evidence arrays must be empty lists in the response; code will resolve them from evidence_ids.
- synonyms_to_avoid should be conservative and derived from glossary aliases or banned generic terms only.

Context analyses:
{_json_block(context_analyses)}

Scout summary:
{_json_block(scout_summary)}

Glossary aliases:
{_json_block(glossary_aliases)}

Banned generic terms:
{_json_block(banned_terms)}

Output shape example:
{_json_block(example)}

Return valid JSON only."""


def build_verifier_prompt(requirements: List[Dict[str, Any]], final_model: Dict[str, Any]) -> str:
    return f"""You are Verifier, a strict reviewer of a generated domain model.

Identify:
- requirement ids not represented in the model
- fields lacking evidence
- duplicate entity ownership
- contradictions between requirements and model

Requirements:
{_json_block(requirements)}

Final model:
{_json_block(final_model)}

Return concise JSON only."""
