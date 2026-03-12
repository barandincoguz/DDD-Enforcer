"""
LLM client for DDD violation detection.

This module now depends on the provider abstraction rather than calling Gemini
directly, which makes validation logic reusable from the IDE backend and the
experiment runner.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from config import AnalyzerConfig
from core.llm_provider import GeminiLLMProvider, LLMProvider


class Violation(BaseModel):
    """Single DDD violation detected in code."""

    type: Literal[
        "SynonymViolation",
        "BannedTermViolation",
        "NamingConventionViolation",
        "ContextBoundaryViolation",
        "ValueObjectViolation",
        "DomainEventViolation",
        "SystemError",
    ] = Field(description="Type of violation")
    message: str = Field(description="Detailed explanation of the violation")
    suggestion: str = Field(description="Actionable suggestion to fix the code")


class ValidationResponse(BaseModel):
    """Structured response for validation output."""

    is_violation: bool = Field(description="True if any violation is detected")
    violations: List[Violation] = Field(description="List of detected violations")


class LLMClient:
    """Provider-backed client for DDD violation detection."""

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        provider: Optional[LLMProvider] = None,
        model_name: Optional[str] = None,
    ):
        self.config = config or AnalyzerConfig()
        self.provider = provider or GeminiLLMProvider()
        self.model_name = model_name or self.config.MODEL_NAME

    def analyze_violation(
        self,
        ast_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run deterministic and advanced checks together."""
        if not ast_data.get("classes") and not ast_data.get("functions") and not ast_data.get("imports"):
            return {"is_violation": False, "violations": []}

        deterministic = self.rule_based_name_violations(ast_data, domain_rules)
        advanced = self.analyze_advanced_violations(ast_data, domain_rules)
        violations = self._dedupe_violations(deterministic + advanced.get("violations", []))
        return {"is_violation": bool(violations), "violations": violations}

    def analyze_advanced_violations(
        self,
        ast_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run advanced context/value-object/event checks through the provider."""
        prompt = self._build_advanced_prompt(ast_data, domain_rules)
        retries = getattr(self.config, "VALIDATION_RETRIES", 2)
        backoff = float(getattr(self.config, "RETRY_BACKOFF_SECONDS", 1.0))

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                result = self.provider.generate_json(
                    model=self.model_name,
                    prompt=prompt,
                    stage="Validator",
                    operation="validate_advanced_checks",
                    response_schema=ValidationResponse,
                    response_mime_type=self.config.RESPONSE_MIME_TYPE,
                    temperature=self.config.TEMPERATURE,
                    retry_count=attempt,
                )
                if result.parse_success and result.parsed is not None:
                    parsed = result.parsed.model_dump(mode="json")
                else:
                    parsed = json.loads(result.text)
                allowed_types = {
                    "ContextBoundaryViolation",
                    "ValueObjectViolation",
                    "DomainEventViolation",
                    "SystemError",
                }
                advanced = [
                    violation
                    for violation in parsed.get("violations", [])
                    if violation.get("type") in allowed_types
                ]
                return {"is_violation": bool(advanced), "violations": advanced}
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(backoff * (2**attempt))

        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "SystemError",
                    "message": f"LLM Error after retries: {last_error}",
                    "suggestion": "Retry validation or check provider/API connectivity.",
                }
            ],
        }

    def analyze_naive_violations(self, filename: str, source_code: str) -> Dict[str, Any]:
        """Run the principled naive baseline without domain rules or AST guidance."""
        prompt = self._build_naive_prompt(filename, source_code)
        retries = getattr(self.config, "VALIDATION_RETRIES", 2)
        backoff = float(getattr(self.config, "RETRY_BACKOFF_SECONDS", 1.0))
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                result = self.provider.generate_json(
                    model=self.model_name,
                    prompt=prompt,
                    stage="Validator",
                    operation="validate_naive_baseline",
                    response_schema=ValidationResponse,
                    response_mime_type=self.config.RESPONSE_MIME_TYPE,
                    temperature=self.config.TEMPERATURE,
                    retry_count=attempt,
                )
                if result.parse_success and result.parsed is not None:
                    return result.parsed.model_dump(mode="json")
                return json.loads(result.text)
            except Exception as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(backoff * (2**attempt))

        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "SystemError",
                    "message": f"Naive baseline failed after retries: {last_error}",
                    "suggestion": "Retry validation or check provider/API connectivity.",
                }
            ],
        }

    def rule_based_name_violations(
        self,
        ast_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Deterministic checks for synonym/banned/naming violations."""
        global_rules = domain_rules.get("global_rules") or {}
        banned_terms = {
            str(term).lower()
            for term in global_rules.get("banned_global_terms", []) or []
            if str(term).strip()
        }

        synonyms_map: Dict[str, str] = {}
        for context in domain_rules.get("bounded_contexts", []) or []:
            ubiquitous_language = (context or {}).get("ubiquitous_language", {})
            for entity in ubiquitous_language.get("entities", []) or []:
                canonical = (entity or {}).get("name", "")
                for synonym in (entity or {}).get("synonyms_to_avoid", []) or []:
                    synonym_key = str(synonym).strip().lower()
                    if synonym_key:
                        synonyms_map[synonym_key] = canonical

        class_names = [cls.get("name", "") for cls in ast_data.get("classes", [])]
        function_names = [fn.get("name", "") for fn in ast_data.get("functions", [])]
        filename_raw = ast_data.get("filename", "")
        filename = Path(filename_raw).name if filename_raw else ""

        tagged_names: List[tuple[str, str]] = []
        if filename:
            tagged_names.append(("File", filename))
        tagged_names.extend(("Class", name) for name in class_names if name)
        tagged_names.extend(("Function", name) for name in function_names if name)

        violations: List[Dict[str, str]] = []
        for label, name in tagged_names:
            lowered_name = name.lower()
            matched_by_canonical: Dict[str, str] = {}
            for synonym in sorted(synonyms_map.keys(), key=len, reverse=True):
                canonical = synonyms_map[synonym]
                if synonym in lowered_name:
                    previous = matched_by_canonical.get(canonical)
                    if previous is None or len(synonym) > len(previous):
                        matched_by_canonical[canonical] = synonym

            for canonical, synonym in matched_by_canonical.items():
                violations.append(
                    {
                        "type": "SynonymViolation",
                        "message": f"{label} name '{name}' contains a synonym '{synonym}' for the term '{canonical}'.",
                        "suggestion": f"Use '{canonical}' terminology instead of '{synonym}'.",
                    }
                )

            for term in banned_terms:
                if term in lowered_name:
                    violations.append(
                        {
                            "type": "BannedTermViolation",
                            "message": f"{label} name '{name}' contains a banned term '{term}'.",
                            "suggestion": "Rename using approved domain terminology.",
                        }
                    )

        for class_name in class_names:
            if class_name and (class_name[0].islower() or "_" in class_name):
                violations.append(
                    {
                        "type": "NamingConventionViolation",
                        "message": f"Class '{class_name}' should be PascalCase.",
                        "suggestion": "Rename class to PascalCase.",
                    }
                )

        return self._dedupe_violations(violations)

    def _filter_hallucinations(
        self,
        result: Dict[str, Any],
        ast_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Filter simple hallucinations for name-based violations."""
        if not result.get("is_violation") or not result.get("violations"):
            return result

        banned_terms = {
            str(term).lower()
            for term in (domain_rules.get("global_rules", {}) or {}).get("banned_global_terms", [])
        }
        synonyms = set()
        for context in domain_rules.get("bounded_contexts", []) or []:
            ubiquitous_language = (context or {}).get("ubiquitous_language", {})
            for entity in ubiquitous_language.get("entities", []) or []:
                for synonym in (entity or {}).get("synonyms_to_avoid", []) or []:
                    synonyms.add(str(synonym).lower())

        all_names = {Path(ast_data.get("filename", "")).name.lower()}
        for cls in ast_data.get("classes", []):
            all_names.add(cls.get("name", "").lower())
        for fn in ast_data.get("functions", []):
            all_names.add(fn.get("name", "").lower())

        filtered = []
        for violation in result.get("violations", []):
            violation_type = violation.get("type", "")
            message = violation.get("message", "").lower()
            if violation_type == "BannedTermViolation":
                if any(term in message and any(term in name for name in all_names) for term in banned_terms):
                    filtered.append(violation)
            elif violation_type == "SynonymViolation":
                if any(syn in message and any(syn in name for name in all_names) for syn in synonyms):
                    filtered.append(violation)
            elif violation_type == "NamingConventionViolation":
                if any(
                    class_name and class_name.lower() in message and (class_name[0].islower() or "_" in class_name)
                    for class_name in [cls.get("name", "") for cls in ast_data.get("classes", [])]
                ):
                    filtered.append(violation)
            else:
                filtered.append(violation)

        result["violations"] = filtered
        result["is_violation"] = bool(filtered)
        return result

    def _build_advanced_prompt(
        self,
        ast_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> str:
        imports = ast_data.get("imports", [])
        assignments = ast_data.get("assignments", [])
        function_calls = ast_data.get("function_calls", [])

        compact_rules: Dict[str, Any] = {"bounded_contexts": []}
        for context in domain_rules.get("bounded_contexts", []) or []:
            ubiquitous_language = (context or {}).get("ubiquitous_language", {})
            compact_rules["bounded_contexts"].append(
                {
                    "context_name": context.get("context_name"),
                    "allowed_dependencies": context.get("allowed_dependencies", []),
                    "actors": [
                        actor.get("name", "")
                        for actor in context.get("actors", []) or []
                    ],
                    "capabilities": [
                        capability.get("name", "")
                        for capability in context.get("capabilities", []) or []
                    ],
                    "business_rules": [
                        rule.get("text", "")
                        for rule in context.get("business_rules", []) or []
                    ],
                    "aggregates": [
                        aggregate.get("name", "")
                        for aggregate in ubiquitous_language.get("aggregates", []) or []
                    ],
                    "services": [
                        service.get("name", "")
                        for service in ubiquitous_language.get("services", []) or []
                    ],
                    "entities": [
                        entity.get("name", "")
                        for entity in ubiquitous_language.get("entities", []) or []
                    ],
                    "value_objects": [
                        value_object.get("name", "")
                        for value_object in ubiquitous_language.get("value_objects", []) or []
                    ],
                    "domain_events": [
                        event.get("name", "") if isinstance(event, dict) else event
                        for event in ubiquitous_language.get("domain_events", []) or []
                    ],
                }
            )

        return f"""You are a DDD advanced-rule validator.
Only check these rule types:
1) ContextBoundaryViolation
2) ValueObjectViolation
3) DomainEventViolation

Do not output SynonymViolation, BannedTermViolation, or NamingConventionViolation.

AST IMPORTS:
{json.dumps(imports, indent=2)}

AST ASSIGNMENTS:
{json.dumps(assignments, indent=2)}

AST FUNCTION CALLS:
{json.dumps(function_calls, indent=2)}

RULES:
{json.dumps(compact_rules, indent=2)}

Output JSON:
{{"is_violation": true|false, "violations": [{{"type":"...","message":"...","suggestion":"..."}}]}}

If no advanced violation exists, return:
{{"is_violation": false, "violations": []}}"""

    def _build_naive_prompt(self, filename: str, source_code: str) -> str:
        return f"""You are performing a naive DDD review with no project-specific domain model.
Analyze the raw source code below and detect possible DDD violations using only general DDD knowledge.

Rules:
- You may output only these violation types:
  SynonymViolation, BannedTermViolation, NamingConventionViolation,
  ContextBoundaryViolation, ValueObjectViolation, DomainEventViolation
- If no violation exists, return {{"is_violation": false, "violations": []}}
- Output valid JSON only

Filename: {filename}
Source code:
```python
{source_code}
```"""

    def _dedupe_violations(self, violations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for violation in violations:
            key = (violation.get("type", ""), violation.get("message", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(violation)
        return deduped
