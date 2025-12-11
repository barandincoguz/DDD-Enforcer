import glob
import json
import os
from contextlib import asynccontextmanager

from core.architect import DomainArchitect

# --- Yeni importlar ---
from core.document_parser import SRSDocumentParser
from core.llm_client import LLMClient
from core.parser import CodeParser
from core.schemas import DomainModel
from fastapi import FastAPI
from pydantic import BaseModel

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 System initializing...")

    model_path = "domain/model.json"

    # Desteklenen SRS formatları
    possible_srs_files = []
    possible_srs_files.extend(glob.glob("inputs/*.pdf"))
    possible_srs_files.extend(glob.glob("inputs/*.docx"))
    possible_srs_files.extend(glob.glob("inputs/*.txt"))

    # ------------------------------------------------------
    # 📌 1) Eğer domain modeli yoksa: SRS'den üret
    # ------------------------------------------------------
    if not os.path.exists(model_path):
        print("⚠️ Domain Model not found. Searching for SRS files...")

        if possible_srs_files:
            srs_file_path = possible_srs_files[0]
            print(f"📄 Found SRS file: {srs_file_path}")

            try:
                # 1) Parse & Chunk
                doc_parser = SRSDocumentParser()
                raw_text = doc_parser.parse_file(srs_file_path)
                print(f"   -> Parsed document with {len(raw_text)} characters.")

                print(f"   -> Parsed into {len(raw_text)} character.")

                if not raw_text.strip():
                    raise ValueError("Document is empty or could not be parsed.")

                # 2) DomainArchitect pipeline (Scout → Architect → Specialist)
                architect = DomainArchitect()

                analysis_results = architect.analyze_document(raw_text=raw_text)

                # 3) Merge → JSON domain modeli oluştur
                print("🧠 Synthesizing final Domain Model JSON...")
                final_model: DomainModel = architect.synthesize_final_model(
                    analysis_results
                )

                print("   -> Domain Model synthesis complete.")

                # JSON olarak kaydet
                with open(model_path, "w") as f:
                    f.write(final_model.model_dump_json(indent=2))

                app_state["domain_rules"] = final_model.model_dump(mode="json")
                print("✅ Domain Model generated and saved successfully!")

            except Exception as e:
                print(f"❌ Critical Error during generation: {e}")
                app_state["domain_rules"] = {}

        else:
            print("❌ No SRS file found in 'inputs/' folder.")
            app_state["domain_rules"] = {}

    # ------------------------------------------------------
    # 📌 2) Domain modeli zaten varsa: Yükle
    # ------------------------------------------------------
    else:
        print("📂 Loading existing domain model...")
        with open(model_path, "r") as f:
            app_state["domain_rules"] = json.load(f)

    # Araçlar
    app_state["parser"] = CodeParser()
    app_state["llm"] = LLMClient()

    yield
    print("👋 System shutting down...")


app = FastAPI(lifespan=lifespan)


# =============================================================================
# 🚀 VALIDATE ENDPOINT
# =============================================================================
class CodeSubmission(BaseModel):
    filename: str
    content: str


@app.post("/validate")
def validate_code(submission: CodeSubmission):
    parser = app_state.get("parser")
    llm = app_state.get("llm")
    rules = app_state.get("domain_rules")

    if not rules:
        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "ConfigError",
                    "message": "Domain Model is empty. Please check backend logs.",
                    "suggestion": "Add inputs/srs.pdf and restart backend.",
                }
            ],
        }

    # Python kodunu AST’e parse et
    ast_data = parser.parse_code(submission.content)  # type: ignore

    if "error" in ast_data:
        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "SyntaxError",
                    "message": ast_data["error"],
                    "suggestion": "Fix Python syntax.",
                }
            ],
        }

    # Kurallara göre violation kontrolü
    return llm.analyze_violation(ast_data, rules)  # type: ignore
