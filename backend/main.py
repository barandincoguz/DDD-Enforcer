import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Kendi modüllerimizi içe aktarıyoruz
from core.parser import CodeParser
from core.llm_client import LLMClient

# Global değişkenler (Sunucu açılınca bir kez yüklenecekler)
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- BAŞLANGIÇ (STARTUP) ---
    print("🚀 System initializing...")
    
    # 1. Domain Kurallarını Yükle
    try:
        with open("domain/model.json", "r") as f:
            app_state["domain_rules"] = json.load(f)
        print("✅ Domain Rules loaded.")
    except Exception as e:
        print(f"❌ Error loading domain rules: {e}")
        app_state["domain_rules"] = {}

    # 2. Araçları Hazırla
    try:
        app_state["parser"] = CodeParser()
        app_state["llm"] = LLMClient()
        print("✅ AI & Parser ready.")
    except Exception as e:
        print(f"❌ Error initializing core modules: {e}")
    
    yield # Sunucu burada çalışır...
    
    # --- KAPANIŞ (SHUTDOWN) ---
    print("👋 System shutting down...")

app = FastAPI(lifespan=lifespan)

# --- Veri Modelleri (Request Body) ---
class CodeSubmission(BaseModel):
    filename: str
    content: str

# --- API Endpointleri ---

@app.get("/")
def read_root():
    return {"status": "active", "system": "DDD Enforcer AI"}

@app.post("/validate")
def validate_code(submission: CodeSubmission):
    """
    VS Code Extension buraya kod gönderecek.
    Biz de analiz edip sonucu döneceğiz.
    """
    
    # 1. Önce araçları alalım
    parser = app_state.get("parser")
    llm = app_state.get("llm")
    rules = app_state.get("domain_rules")

    if not parser or not llm:
        raise HTTPException(status_code=500, detail="System not initialized properly.")

    # 2. Kodu Parse Et (AST) - Ali'nin Modülü
    print(f"🔍 Parsing file: {submission.filename}")
    ast_data = parser.parse_code(submission.content)
    
    # Eğer Syntax hatası varsa AI'a sormaya gerek yok, direkt dön
    if "error" in ast_data:
        return {
            "is_violation": True,
            "violations": [{
                "type": "SyntaxError", 
                "message": ast_data["error"], 
                "suggestion": "Fix Python syntax errors first."
            }]
        }

    # 3. AI ile Analiz Et - Ahmet'in Modülü
    print("🤖 Asking Gemini...")
    analysis_result = llm.analyze_violation(ast_data, rules)
    
    return analysis_result
# CI Test: Triggering Backend Workflow