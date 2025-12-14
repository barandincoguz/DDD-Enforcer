"""
LLM Client

Interfaces with Google Gemini for DDD violation detection.
Uses structured outputs to ensure consistent response format.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LLMConfig

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
        "SystemError"
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

    def __init__(self, config: Optional[LLMConfig] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.config = config or LLMConfig()
        self.client = genai.Client(api_key=api_key)

    def analyze_violation(self, ast_data: dict, domain_rules: dict) -> dict:
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
            return json.loads(response.text)

        except Exception as e:
            return {
                "is_violation": True,
                "violations": [{
                    "type": "SystemError",
                    "message": f"LLM Error: {str(e)}",
                    "suggestion": "Check API logs or connectivity.",
                }],
            }

    def _build_prompt(self, ast_data: dict, domain_rules: dict) -> str:
        """Build the analysis prompt."""
        filename = ast_data.get("filename", "unknown.py")
        
        return f"""You are a Domain-Driven Design (DDD) violation detector for enterprise software.

=== VIOLATION TYPES YOU MUST DETECT ===

📌 TYPE 1 - SYNONYM VIOLATIONS (SynonymViolation)
- Check 'synonyms_to_avoid' arrays in each entity
- Flag class/function names containing avoided synonyms
- Example: synonyms_to_avoid: ["Client", "User"] for Customer
  - "ClientManager" -> VIOLATION (use CustomerManager)
  - "get_user_info" -> VIOLATION (use get_customer_info)

📌 TYPE 2 - BANNED TERM VIOLATIONS (BannedTermViolation)
- Check 'global_rules.banned_global_terms'
- Flag any class/function/file name containing these terms
- **IMPORTANT: Check the filename first!**
- Example: banned_global_terms: ["Manager", "Util"]
  - File "payment_util.py" -> VIOLATION
  - Class "paymentUtil" -> VIOLATION
  - "OrderManager" -> VIOLATION

📌 TYPE 3 - NAMING CONVENTION VIOLATIONS (NamingConventionViolation)
- Check 'global_rules.naming_convention'
- Python standard: Classes use PascalCase, functions use snake_case
- Example: naming_convention: "PascalCase"
  - "paymentUtil" -> VIOLATION (should be PaymentUtil)
  - "CustomerService" -> CORRECT

📌 TYPE 4 - CONTEXT BOUNDARY VIOLATIONS (ContextBoundaryViolation)
- Check 'allowed_dependencies' for each bounded context
- Extract context name from import module path
- Module path format: context_name.entity (e.g., inventory_management.product)
- Convert snake_case to PascalCase: inventory_management -> InventoryManagement
- Flag imports/usage of entities from forbidden contexts
- Example: PaymentProcessing context with allowed_dependencies: ["CustomerManagement", "OrderManagement"]
  - Import "from inventory_management.product import Product" -> VIOLATION
  - Using Product entity in payment code -> VIOLATION
  - Payment can ONLY import from customer_management, order_management, payment_processing

📌 TYPE 5 - VALUE OBJECT VIOLATIONS (ValueObjectViolation)
- Check if primitive types are used instead of defined Value Objects
- Look for 'value_objects' in domain rules
- Check function parameters and assignments
- Example: Money Value Object with ["amount", "currency"]
  - Parameter "amount: float" -> VIOLATION (use Money Value Object)
  - Assignment "order.status = \\"CONFIRMED\\"" -> VIOLATION (use OrderStatus Value Object)
  - Using primitive 'str' for status -> VIOLATION

📌 TYPE 6 - DOMAIN EVENT VIOLATIONS (DomainEventViolation)
- Check if domain_events are used in wrong context
- Events should only be emitted in their defining context
- Look for function calls like emit_event, publish, trigger with event names
- Example: PaymentCompleted event defined in PaymentProcessing context
  - Emitting "PaymentCompleted" in OrderManagement -> VIOLATION
  - Emitting "OrderPlaced" in PaymentProcessing -> VIOLATION

=== ALLOWED DDD PATTERNS (NEVER FLAG THESE) ===
✅ Repository, Service, Factory, Gateway, Aggregate, Specification suffixes
✅ Entity names from domain model (Customer, Order, Product, Payment)
✅ Value Object names (Address, Money, OrderStatus, StockLevel)
✅ Domain events in their own context
✅ Proper dependency relationships per allowed_dependencies

=== FILE BEING ANALYZED ===
Filename: {filename}

=== DOMAIN RULES ===
{json.dumps(domain_rules, indent=2)}

=== CODE STRUCTURE ===
{json.dumps(ast_data, indent=2)}

=== ANALYSIS CHECKLIST ===
1. ✓ **Check filename**: Does "{filename}" contain any banned_global_terms?
2. ✓ **Check each class**:
   - Contains synonym from any entity's synonyms_to_avoid?
   - Contains any banned_global_terms?
   - Follows naming_convention (PascalCase)?
3. ✓ **Check each import**:
   - Extract context from module path (e.g., inventory_management -> InventoryManagement)
   - Is this context in allowed_dependencies?
   - Which entity is being imported? Is it used illegally?
4. ✓ **Check function parameters**:
   - Are primitive types (float, str, int) used instead of Value Objects?
   - Compare parameter types with value_objects definitions
   - Example: "amount: float" should be "amount: Money"
5. ✓ **Check assignments**:
   - Are strings assigned to fields that should use Value Objects?
   - Example: order.status = "CONFIRMED" should use OrderStatus Value Object
6. ✓ **Check function calls**:
   - Are domain events emitted in correct context?
   - Match event names with domain_events in each context

=== CONTEXT MATCHING RULES ===
- Convert module paths to context names:
  - customer_management -> CustomerManagement
  - order_management -> OrderManagement  
  - inventory_management -> InventoryManagement
  - payment_processing -> PaymentProcessing
  - product_catalog -> ProductCatalog

=== OUTPUT RULES ===
- Report ALL violations found, not just the first one
- Be specific: mention exact class/function/parameter names
- Provide actionable suggestions with correct names from domain model
- If NO violations: return is_violation: false with empty violations array
- Use correct violation type for each issue

=== EXAMPLES ===

Example 1: File "payment_util.py"
Violations:
  - BannedTermViolation: Filename contains banned term "Util"

Example 2: class paymentUtil:
Violations:
  - BannedTermViolation: Class name contains banned term "Util"
  - NamingConventionViolation: Should be "PaymentUtil" (PascalCase)

Example 3: from inventory_management.product import Product (in PaymentProcessing context)
Violation: ContextBoundaryViolation - PaymentProcessing context cannot depend on InventoryManagement. Allowed: ["CustomerManagement", "OrderManagement"]

Example 4: def process_payment(order: Order, amount: float):
Violation: ValueObjectViolation - Parameter 'amount' uses primitive type 'float'. Use Money Value Object instead (attributes: amount, currency)

Example 5: order.status = "CONFIRMED"
Violation: ValueObjectViolation - Assignment uses string literal instead of OrderStatus Value Object

Example 6: self.emit_event("PaymentCompleted") in OrderManagement context
Violation: DomainEventViolation - "PaymentCompleted" is defined in PaymentProcessing context, not OrderManagement

Now analyze the code thoroughly and report ALL violations."""


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    dummy_ast = {"classes": [{"name": "ClientManager"}], "imports": []}
    dummy_rules = {
        "bounded_contexts": [{
            "ubiquitous_language": {
                "entities": [{"name": "Customer", "synonyms_to_avoid": ["Client"]}]
            }
        }]
    }

    client = LLMClient()
    print("Analyzing...")
    result = client.analyze_violation(dummy_ast, dummy_rules)
    print(json.dumps(result, indent=2))
