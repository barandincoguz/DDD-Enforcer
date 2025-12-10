import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

class LLMClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found! Please check your .env file.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_violation(self, ast_data: dict, domain_rules: dict) -> dict:
        """
        Sends the parsed code structure and domain rules to Gemini
        to detect DDD violations.
        """
        
        # PROMPT ENGINEERING
        prompt = f"""
        You are a strict Domain-Driven Design (DDD) Architect Enforcer.
        
        Your Task: Analyze the provided Python Code Structure (AST) against the strict Domain Rules.
        
        1. GROUND TRUTH (Domain Rules):
        {json.dumps(domain_rules, indent=2)}

        2. INPUT CODE STRUCTURE (AST):
        {json.dumps(ast_data, indent=2)}

        3. INSTRUCTIONS:
        - Check if any Class Name, Function Name, or Variable Name violates the 'ubiquitous_language' (synonyms_to_avoid).
        - Check if imports violate 'allowed_dependencies' (Context Leakage).
        - If 'Client' is used but Domain says 'Customer', flag it.
        - If 'Sales' imports 'Shipping' but dependencies say [], flag it.
        
        4. OUTPUT FORMAT (JSON ONLY):
        Return a valid JSON object. Do not add markdown like ```json ... ```.
        Format:
        {{
            "is_violation": true/false,
            "violations": [
                {{
                    "type": "NamingViolation" or "ContextViolation",
                    "message": "Detailed explanation of why this is wrong.",
                    "suggestion": "What should be used instead (e.g., Use 'Customer' not 'Client')."
                }}
            ]
        }}
        """

        try:
            response = self.model.generate_content(prompt)
            # Bazen model markdown code block içinde döndürür, temizleyelim
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception as e:
            return {
                "is_violation": True, 
                "violations": [{"type": "SystemError", "message": f"LLM Error: {str(e)}", "suggestion": "Check logs."}]
            }

# --- Test Bloğu ---
if __name__ == "__main__":
    # Sahte veriyle test edelim
    dummy_ast = {'classes': [{'name': 'ClientManager'}], 'imports': []}
    
    # Basit bir kural seti
    dummy_rules = {
        "bounded_contexts": [{
            "ubiquitous_language": {
                "entities": [{"name": "Customer", "synonyms_to_avoid": ["Client"]}]
            }
        }]
    }
    
    client = LLMClient()
    print("AI Düşünüyor...")
    result = client.analyze_violation(dummy_ast, dummy_rules)
    print(json.dumps(result, indent=2))