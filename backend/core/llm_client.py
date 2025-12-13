import json
import os
from typing import List, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# .env dosyasını yükle
load_dotenv()


# --- 1. Pydantic Modelleri (Structured Output için Şema) ---
# Bu modeller, Gemini'nin yanıtı %100 bu formatta üretmesini garanti eder.
class Violation(BaseModel):
    type: Literal["NamingViolation", "ContextViolation", "SystemError"] = Field(
        description="The type of the violation."
    )
    message: str = Field(description="Detailed explanation of the violation.")
    suggestion: str = Field(description="Actionable suggestion to fix the code.")


class ValidationResponse(BaseModel):
    is_violation: bool = Field(description="True if any violation is detected.")
    violations: List[Violation] = Field(description="List of detected violations.")


class LLMClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found! Please check your .env file.")

        # Yeni SDK Client Başlatma
        self.client = genai.Client(api_key=api_key)

    def analyze_violation(self, ast_data: dict, domain_rules: dict) -> dict:
        """
        Sends the parsed code structure and domain rules to Gemini
        to detect DDD violations using Structured Outputs.
        """

        prompt = f"""
You are a Domain-Driven Design (DDD) violation detector. Your job is VERY specific and LIMITED.

=== YOU CAN ONLY FLAG TWO TYPES OF VIOLATIONS ===

TYPE 1 - SYNONYM VIOLATIONS:
- Look at 'synonyms_to_avoid' arrays in the domain rules
- If a class/function name contains a word from 'synonyms_to_avoid', flag it
- Example: synonyms_to_avoid: ["Client", "User", "Buyer"]
  - "ClientManager" contains "Client" → VIOLATION
  - "UserService" contains "User" → VIOLATION
  - "CustomerService" does NOT contain any synonym → NO VIOLATION

TYPE 2 - BANNED GLOBAL TERM VIOLATIONS:
- Look at 'global_rules.banned_global_terms' array
- If a class/function name contains a banned term, flag it
- Example: banned_global_terms: ["Manager", "Util", "Helper"]
  - "OrderManager" contains "Manager" → VIOLATION
  - "StringUtil" contains "Util" → VIOLATION
  - "OrderService" does NOT contain any banned term → NO VIOLATION

=== THINGS YOU MUST NOT FLAG (ALLOWED PATTERNS) ===

These are STANDARD DDD patterns and are ALWAYS allowed:
- Repository (CustomerRepository, OrderRepository, ProductRepository)
- Service (OrderService, PaymentService, CustomerService)
- Factory (OrderFactory, CustomerFactory)
- Gateway (PaymentGateway, EmailGateway)
- Aggregate (OrderAggregate)
- Specification (CustomerSpecification)
- Event (OrderPlaced, PaymentCompleted, CustomerCreated)
- ValueObject, Entity names from the domain model

=== CORRECT ENTITY NAMES ARE NEVER VIOLATIONS ===

The 'name' field in entities defines the CORRECT term:
- {{"name": "Customer", "synonyms_to_avoid": ["Client"]}}
- "Customer" is CORRECT → NEVER flag it
- "Client" is WRONG → flag it

=== DOMAIN RULES ===
{json.dumps(domain_rules, indent=2)}

=== CODE TO ANALYZE ===
{json.dumps(ast_data, indent=2)}

=== YOUR TASK ===
1. Check each CLASS name in "classes" array: Does it contain a word from ANY 'synonyms_to_avoid' list? If yes → violation
2. Check each CLASS name in "classes" array: Does it contain a word from 'banned_global_terms'? If yes → violation
3. Check each FUNCTION name in "functions" array: Does it contain a word from ANY 'synonyms_to_avoid' list? If yes → violation
4. Check each FUNCTION name in "functions" array: Does it contain a word from 'banned_global_terms'? If yes → violation
5. If NO violations found, return is_violation: false with empty violations array
6. DO NOT invent new rules. ONLY use the rules provided above.
7. When in doubt, DO NOT flag it as a violation.

=== FUNCTION NAME EXAMPLES ===
- synonyms_to_avoid for Customer: ["Client", "User", "Buyer"]
  - "get_buyer_info" contains "buyer" → VIOLATION (should be get_customer_info)
  - "add_client" contains "client" → VIOLATION (should be add_customer)
  - "get_customer_info" → NO VIOLATION (uses correct term)
"""

        try:
            # Yeni SDK Yapısı
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",  # En güncel hızlı model
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    # Pydantic modelini şema olarak veriyoruz:
                    response_schema=ValidationResponse,
                ),
            )

            # Yeni SDK'da response.text direkt şemaya uygun JSON döner.
            # Alternatif olarak response.parsed kullanılabilir ancak dict dönmesini istediğin için:
            return json.loads(response.text)  # type: ignore

        except Exception as e:
            # Hata durumunda manuel fallback
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


# --- TEST BLOCK ---
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
    print("AI thinking...")
    result = client.analyze_violation(dummy_ast, dummy_rules)
    print(json.dumps(result, indent=2))
