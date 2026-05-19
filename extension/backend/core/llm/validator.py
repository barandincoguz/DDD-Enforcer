"""
LLM Client

Interfaces with Google Gemini for DDD violation detection.
Uses structured outputs to ensure consistent response format.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AnalyzerConfig
from core.code_parser import build_advanced_validation_payload
from core.llm._response_adapter import LLMResponseAdapter
from core.llm.gemini import GeminiClient
from core.token_tracker import TokenTracker

load_dotenv()


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================


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
    """Response from violation analysis."""

    is_violation: bool = Field(description="True if any violation is detected")
    violations: List[Violation] = Field(description="List of detected violations")


# =============================================================================
# LLM CLIENT
# =============================================================================


class LLMClient:
    """Client for DDD violation detection using Google Gemini."""

    def __init__(self, config: Optional[AnalyzerConfig] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.config = config or AnalyzerConfig()
        self.client = GeminiClient(api_key=api_key)
        self.token_tracker = TokenTracker.get_instance()

    def analyze_violation(
        self, ast_data: Dict[str, Any], domain_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze code for DDD violations.

        Checks class and function names against synonym lists and
        banned global terms from the domain rules.
        """
        # Fast short-circuit: nothing analyzable in AST
        if not ast_data.get("classes") and not ast_data.get("functions") and not ast_data.get("imports"):
            return {"is_violation": False, "violations": []}

        prompt = self._build_prompt(ast_data, domain_rules)
        retries = getattr(self.config, "VALIDATION_RETRIES", 2)
        backoff = float(getattr(self.config, "RETRY_BACKOFF_SECONDS", 1.0))

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                llm_response = self.client.structured_output(
                    messages=[{"role": "user", "content": prompt}],
                    schema=ValidationResponse,
                    model=self.config.MODEL_NAME,
                    temperature=self.config.TEMPERATURE,
                )
                response = LLMResponseAdapter(llm_response)

                self.token_tracker.track_api_call(
                    response,
                    stage="Validator",
                    operation="validate_code"
                )

                result = json.loads(response.text)
                result = self._filter_hallucinations(result, ast_data, domain_rules)
                return result
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(backoff * (2 ** attempt))

        # Deterministic fallback after retries
        fallback_result = self._deterministic_fallback(ast_data, domain_rules)
        if fallback_result["is_violation"]:
            return fallback_result

        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "SystemError",
                    "message": f"LLM Error after retries: {str(last_error)}",
                    "suggestion": "Retry validation or check backend/API connectivity.",
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
            str(t).lower()
            for t in global_rules.get("banned_global_terms", []) or []
            if str(t).strip()
        }

        synonyms_map: Dict[str, str] = {}
        for ctx in domain_rules.get("bounded_contexts", []) or []:
            ul = (ctx or {}).get("ubiquitous_language", {})
            for entity in ul.get("entities", []) or []:
                canonical = (entity or {}).get("name", "")
                for syn in (entity or {}).get("synonyms_to_avoid", []) or []:
                    syn_name = str(syn).strip().lower()
                    if syn_name:
                        synonyms_map[syn_name] = canonical

        class_names = [cls.get("name", "") for cls in ast_data.get("classes", [])]
        function_names = [fn.get("name", "") for fn in ast_data.get("functions", [])]

        filename_raw = ast_data.get("filename", "")
        filename = Path(filename_raw).name if filename_raw else ""

        tagged_names: List[tuple[str, str]] = []
        if filename:
            tagged_names.append(("File", filename))
        tagged_names.extend(("Class", n) for n in class_names if n)
        tagged_names.extend(("Function", n) for n in function_names if n)

        violations: List[Dict[str, str]] = []
        for label, name in tagged_names:
            lowered_name = name.lower()

            # Prefer more specific synonym matches first to avoid noisy overlap
            # (e.g., 'item' and 'lineitem' both matching 'LineItem').
            matched_by_canonical: Dict[str, str] = {}
            sorted_synonyms = sorted(synonyms_map.keys(), key=len, reverse=True)
            for syn in sorted_synonyms:
                canonical = synonyms_map[syn]
                if syn in lowered_name:
                    prev = matched_by_canonical.get(canonical)
                    if prev is None or len(syn) > len(prev):
                        matched_by_canonical[canonical] = syn

            for canonical, syn in matched_by_canonical.items():
                violations.append(
                    {
                        "type": "SynonymViolation",
                        "message": f"{label} name '{name}' contains a synonym '{syn}' for the term '{canonical}'.",
                        "suggestion": f"Use '{canonical}' terminology instead of '{syn}'.",
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

        # Deduplicate by type/message
        seen = set()
        deduped: List[Dict[str, str]] = []
        for v in violations:
            key = (v["type"], v["message"])
            if key not in seen:
                seen.add(key)
                deduped.append(v)

        return deduped

    def analyze_advanced_violations(
        self,
        ast_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM analysis for advanced checks only (context/value-object/event)."""
        prompt = self._build_advanced_prompt(ast_data, domain_rules)
        retries = getattr(self.config, "VALIDATION_RETRIES", 2)
        backoff = float(getattr(self.config, "RETRY_BACKOFF_SECONDS", 1.0))

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                llm_response = self.client.structured_output(
                    messages=[{"role": "user", "content": prompt}],
                    schema=ValidationResponse,
                    model=self.config.MODEL_NAME,
                    temperature=self.config.TEMPERATURE,
                )
                response = LLMResponseAdapter(llm_response)

                self.token_tracker.track_api_call(
                    response,
                    stage="Validator",
                    operation="validate_advanced_checks",
                )

                result = json.loads(response.text)

                # Keep only advanced types
                allowed_types = {
                    "ContextBoundaryViolation",
                    "ValueObjectViolation",
                    "DomainEventViolation",
                    "SystemError",
                }
                advanced = [
                    v for v in result.get("violations", [])
                    if v.get("type") in allowed_types
                ]
                return {"is_violation": len(advanced) > 0, "violations": advanced}
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(backoff * (2 ** attempt))

        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "SystemError",
                    "message": f"LLM Error after retries: {str(last_error)}",
                    "suggestion": "Retry validation or check backend/API connectivity.",
                }
            ],
        }

    def _deterministic_fallback(
        self,
        ast_data: Dict[str, Any],
        domain_rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Lightweight fallback checks when LLM call fails."""
        violations: List[Dict[str, str]] = []

        banned_terms = [
            t.lower() for t in domain_rules.get("global_rules", {}).get("banned_global_terms", [])
        ]

        synonyms_map: Dict[str, str] = {}
        for ctx in domain_rules.get("bounded_contexts", []):
            ul = ctx.get("ubiquitous_language", {})
            for entity in ul.get("entities", []):
                canonical = entity.get("name", "")
                for syn in entity.get("synonyms_to_avoid", []) or []:
                    synonyms_map[syn.lower()] = canonical

        class_names = [cls.get("name", "") for cls in ast_data.get("classes", [])]
        func_names = [fn.get("name", "") for fn in ast_data.get("functions", [])]
        filename = ast_data.get("filename", "")
        all_names = [filename] + class_names + func_names

        for name in all_names:
            lower_name = name.lower()
            for term in banned_terms:
                if term and term in lower_name:
                    violations.append(
                        {
                            "type": "BannedTermViolation",
                            "message": f"Banned term '{term}' found in '{name}'.",
                            "suggestion": "Rename using domain terminology.",
                        }
                    )

            for syn, canonical in synonyms_map.items():
                if syn and syn in lower_name:
                    violations.append(
                        {
                            "type": "SynonymViolation",
                            "message": f"Synonym '{syn}' found in '{name}', prefer '{canonical}'.",
                            "suggestion": f"Use '{canonical}' in naming.",
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

        # Deduplicate by (type, message)
        seen = set()
        unique_violations: List[Dict[str, str]] = []
        for violation in violations:
            key = (violation["type"], violation["message"])
            if key not in seen:
                seen.add(key)
                unique_violations.append(violation)

        return {
            "is_violation": len(unique_violations) > 0,
            "violations": unique_violations,
        }
    
    def _filter_hallucinations(
        self, result: Dict[str, Any], ast_data: Dict[str, Any], domain_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter out hallucinated violations by verifying string matches.
        
        This catches cases where LLM claims a term exists in a name but it doesn't.
        """
        if not result.get("is_violation") or not result.get("violations"):
            return result
        
        # Get banned terms and synonyms for verification
        banned_terms = set()
        global_rules = domain_rules.get("global_rules", {})
        for term in global_rules.get("banned_global_terms", []):
            banned_terms.add(term.lower())
        
        synonyms = set()
        for ctx in domain_rules.get("bounded_contexts", []):
            ul = ctx.get("ubiquitous_language", {})
            for entity in ul.get("entities", []):
                for syn in entity.get("synonyms_to_avoid", []):
                    synonyms.add(syn.lower())
        
        # Get all names from AST
        all_names = set()
        filename = ast_data.get("filename", "")
        all_names.add(filename.lower())
        for cls in ast_data.get("classes", []):
            all_names.add(cls.get("name", "").lower())
        for func in ast_data.get("functions", []):
            all_names.add(func.get("name", "").lower())
        
        filtered_violations = []
        for violation in result.get("violations", []):
            v_type = violation.get("type", "")
            message = violation.get("message", "").lower()
            
            # Verify BannedTermViolation
            if v_type == "BannedTermViolation":
                # Extract the claimed banned term from message
                is_valid = False
                for term in banned_terms:
                    if term in message:
                        # Check if this term actually exists in any name
                        for name in all_names:
                            if term in name:
                                is_valid = True
                                break
                    if is_valid:
                        break
                
                if is_valid:
                    filtered_violations.append(violation)
                else:
                    print(f"      🚫 Filtered hallucination: {violation.get('message', '')[:60]}...")
            
            # Verify SynonymViolation
            elif v_type == "SynonymViolation":
                is_valid = False
                for syn in synonyms:
                    if syn in message:
                        for name in all_names:
                            if syn in name:
                                is_valid = True
                                break
                    if is_valid:
                        break
                
                if is_valid:
                    filtered_violations.append(violation)
                else:
                    print(f"      🚫 Filtered hallucination: {violation.get('message', '')[:60]}...")
            
            # Verify NamingConventionViolation  
            elif v_type == "NamingConventionViolation":
                # Check if the mentioned name actually exists in the AST
                is_valid = False
                mentioned_name = None
                
                # Try to find which name is being mentioned
                for cls in ast_data.get("classes", []):
                    name = cls.get("name", "")
                    if name and name.lower() in message:
                        mentioned_name = name
                        # Verify it actually violates the rule
                        if name[0].islower() or "_" in name:
                            is_valid = True
                        break
                
                if is_valid:
                    filtered_violations.append(violation)
                elif mentioned_name:
                    # Name exists but doesn't violate the rule
                    print(f"      🚫 Filtered false positive: '{mentioned_name}' is valid PascalCase")
                else:
                    # Name doesn't exist at all - hallucination
                    print(f"      🚫 Filtered hallucination: {violation.get('message', '')[:60]}...")
            
            else:
                # Other violation types - keep as-is
                filtered_violations.append(violation)
        
        result["violations"] = filtered_violations
        result["is_violation"] = len(filtered_violations) > 0
        
        return result

    def _build_prompt(
        self, ast_data: Dict[str, Any], domain_rules: Dict[str, Any]
    ) -> str:
        """Build the analysis prompt."""
        filename = ast_data.get("filename", "unknown.py")
        
        # Extract entity and value object names for whitelist
        whitelist_names = set()
        for ctx in domain_rules.get("bounded_contexts", []):
            ul = ctx.get("ubiquitous_language", {})
            for entity in ul.get("entities", []):
                whitelist_names.add(entity.get("name", ""))
            for vo in ul.get("value_objects", []):
                whitelist_names.add(vo.get("name", ""))
        whitelist_str = ", ".join(sorted(whitelist_names)) if whitelist_names else "None"

        # Extract actual names from AST for reference
        class_names = [cls.get("name", "") for cls in ast_data.get("classes", [])]
        function_names = [func.get("name", "") for func in ast_data.get("functions", [])]
        
        # Extract banned terms and synonyms
        banned_terms = domain_rules.get("global_rules", {}).get("banned_global_terms", [])
        
        synonyms_map = {}
        for ctx in domain_rules.get("bounded_contexts", []):
            ul = ctx.get("ubiquitous_language", {})
            for entity in ul.get("entities", []):
                entity_name = entity.get("name", "")
                for syn in entity.get("synonyms_to_avoid", []):
                    synonyms_map[syn] = entity_name

        return f"""You are a DDD violation detector. Analyze ONLY the code provided below.

⚠️ CRITICAL INSTRUCTION:
You must ONLY analyze the ACTUAL names listed below. Do NOT invent or imagine any names.
If a name is not in the lists below, it DOES NOT EXIST - do not report violations for non-existent names.

═══════════════════════════════════════════════════════════════════════════════
ACTUAL CODE TO ANALYZE (ONLY THESE NAMES EXIST)
═══════════════════════════════════════════════════════════════════════════════

Filename: {filename}
Class names in this file: {class_names if class_names else "NONE"}
Function names in this file: {function_names if function_names else "NONE"}

WHITELIST (domain entities - NEVER flag these): {whitelist_str}

═══════════════════════════════════════════════════════════════════════════════
VIOLATION RULES
═══════════════════════════════════════════════════════════════════════════════

1. SynonymViolation
   - Check if any CLASS/FUNCTION name contains these synonyms: {list(synonyms_map.keys()) if synonyms_map else "NONE"}
   - Synonym to correct term mapping: {synonyms_map if synonyms_map else "NONE"}
   - Only flag if synonym is SUBSTRING of an ACTUAL name from the lists above

2. BannedTermViolation
   - Check if filename or any CLASS/FUNCTION name contains: {banned_terms if banned_terms else "NONE"}
   - Only flag if banned term is SUBSTRING of an ACTUAL name from the lists above

3. NamingConventionViolation (FOR CLASSES ONLY)
   - Check each class name from: {class_names if class_names else "NONE"}
   - Flag ONLY if: starts with lowercase OR contains underscore "_"
   - DO NOT flag function names (snake_case is valid for functions)

4. ContextBoundaryViolation - Check imports against allowed_dependencies
5. ValueObjectViolation - Check if primitives used instead of Value Objects  
6. DomainEventViolation - Check event emissions

═══════════════════════════════════════════════════════════════════════════════
FULL AST DATA (for detailed analysis)
═══════════════════════════════════════════════════════════════════════════════

{json.dumps(ast_data, indent=2)}

═══════════════════════════════════════════════════════════════════════════════
DOMAIN RULES
═══════════════════════════════════════════════════════════════════════════════

{json.dumps(domain_rules, indent=2)}

═══════════════════════════════════════════════════════════════════════════════
OUTPUT RULES
═══════════════════════════════════════════════════════════════════════════════

- Return {{"is_violation": false, "violations": []}} if NO violations found
- ONLY report violations for names that ACTUALLY EXIST in the class/function lists above
- Each violation must reference a REAL name from the code
- Double-check: Is the name I'm reporting actually in {class_names + function_names}?"""

    def _build_advanced_prompt(
        self, ast_data: Dict[str, Any], domain_rules: Dict[str, Any]
    ) -> str:
        """Build compact prompt for advanced checks only."""
        parser_signals = build_advanced_validation_payload(ast_data)
        imports = parser_signals["imports"]
        assignments = parser_signals["assignments"]
        function_calls = parser_signals["function_calls"]

        compact_rules: Dict[str, Any] = {"bounded_contexts": []}
        for ctx in domain_rules.get("bounded_contexts", []) or []:
            ul = (ctx or {}).get("ubiquitous_language", {})
            compact_rules["bounded_contexts"].append(
                {
                    "context_name": ctx.get("context_name"),
                    "allowed_dependencies": ctx.get("allowed_dependencies", []),
                    "value_objects": [
                        vo.get("name", "") for vo in ul.get("value_objects", []) or []
                    ],
                    "domain_events": ul.get("domain_events", []) or [],
                }
            )

        return f"""You are a DDD advanced-rule validator.
Only check these 3 rule types:
1) ContextBoundaryViolation
2) ValueObjectViolation
3) DomainEventViolation

Do NOT output SynonymViolation, BannedTermViolation, NamingConventionViolation.

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


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    dummy_ast = {"classes": [{"name": "ClientManager"}], "imports": []}
    dummy_rules = {
        "bounded_contexts": [
            {
                "ubiquitous_language": {
                    "entities": [{"name": "Customer", "synonyms_to_avoid": ["Client"]}]
                }
            }
        ]
    }

    client = LLMClient()
    print("Analyzing...")
    result = client.analyze_violation(dummy_ast, dummy_rules)
    print(json.dumps(result, indent=2))
