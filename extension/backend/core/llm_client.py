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
                ),
            )
            
            # Track token usage
            self.token_tracker.track_api_call(
                response,
                stage="Validator",
                operation="validate_code"
            )
            
            return json.loads(response.text)

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

        return f"""You are a Domain-Driven Design violation detector. Analyze code against domain rules and report violations.

WHITELIST (NEVER FLAG THESE NAMES): {whitelist_str}

VIOLATION TYPES:

1. SynonymViolation
   - Trigger: Class/function name contains a term from any entity's "synonyms_to_avoid" array
   - Example: If Customer has synonyms_to_avoid: ["Client"], then "ClientService" is a violation

2. BannedTermViolation  
   - Trigger: Filename, class, or function contains a term from "global_rules.banned_global_terms"
   - Check filename "{filename}" first

3. NamingConventionViolation
   - Trigger: Class name is NOT in PascalCase (first letter lowercase OR contains underscore)
   - SKIP if name is in WHITELIST above
   - SKIP if name already starts with uppercase and has no underscores
   - Valid: Customer, OrderService, PaymentProcessor
   - Invalid: customerService, order_manager, paymentUtil

4. ContextBoundaryViolation
   - Trigger: Import from a context not in current context's "allowed_dependencies"
   - Convert module path to context: inventory_management -> InventoryManagement

5. ValueObjectViolation
   - Trigger: Primitive type (float, str, int) used where a Value Object is defined
   - Check function parameters against value_objects definitions

6. DomainEventViolation
   - Trigger: Event emitted outside its defining context
   - Check emit_event/publish calls against domain_events in each context

ANALYSIS INPUT:
- Filename: {filename}
- Domain Rules: {json.dumps(domain_rules, indent=2)}
- Code Structure: {json.dumps(ast_data, indent=2)}

OUTPUT CONTRACT:
- Return is_violation: false with empty violations array if code is compliant
- Only report actual violations with specific class/function names
- Each violation needs: type, message (what is wrong), suggestion (how to fix)
- Do NOT flag names that are already correctly formatted
- Do NOT flag domain entity names from the whitelist"""


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
