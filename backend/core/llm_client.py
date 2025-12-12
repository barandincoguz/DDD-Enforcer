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

        # Prompt artık daha sade çünkü JSON formatını config hallediyor
        prompt = f"""
        You are a strict Domain-Driven Design (DDD) Architect Enforcer.

        Your Task: Analyze the provided Python Code Structure (AST) against the strict Domain Rules.

        1. GROUND TRUTH (Domain Rules):
        {json.dumps(domain_rules, indent=2)}

        2. INPUT CODE STRUCTURE (AST):
        {json.dumps(ast_data, indent=2)}

        3. INSTRUCTIONS:
        - Analyze the AST specifically for 'ubiquitous_language' violations (synonyms defined in synonyms_to_avoid).
        - Check for 'allowed_dependencies' violations (Context Leakage).
        - If a violation is found, set is_violation to true and provide details.
        """

        try:
            # Yeni SDK Yapısı
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",  # En güncel hızlı model
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
