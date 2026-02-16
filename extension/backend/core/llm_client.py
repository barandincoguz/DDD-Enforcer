"""
LLM Client

Interfaces with Google Gemini for DDD violation detection.
Uses structured outputs to ensure consistent response format.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AnalyzerConfig
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
        self.client = genai.Client(api_key=api_key)
        self.token_tracker = TokenTracker.get_instance()

    def analyze_violation(
        self, ast_data: Dict[str, Any], domain_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze code for DDD violations.

        Checks class and function names against synonym lists and
        banned global terms from the domain rules.
        """
        prompt = self._build_prompt(ast_data, domain_rules)

        try:
            response = self.client.models.generate_content(
                model=self.config.MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type=self.config.RESPONSE_MIME_TYPE,
                    response_schema=ValidationResponse,
                    temperature=self.config.TEMPERATURE,  # Use temperature from config
                ),
            )
            
            # Track token usage
            self.token_tracker.track_api_call(
                response,
                stage="Validator",
                operation="validate_code"
            )
            
            result = json.loads(response.text)
            
            # Post-process to filter out hallucinated violations
            result = self._filter_hallucinations(result, ast_data, domain_rules)
            
            return result

        except Exception as e:
            return {
                "is_violation": True,
                "violations": [
                    {
                        "type": "SystemError",
                        "message": f"LLM Error: {str(e)}",
                        "suggestion": "Check API logs or connectivity.",
                    }
                ],
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
