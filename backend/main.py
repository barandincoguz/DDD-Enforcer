import glob
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from core.architect import DomainArchitect

# --- Yeni importlar ---
from core.document_parser import SRSDocumentParser
from core.llm_client import LLMClient
from core.parser import CodeParser
from core.rag_pipeline import RAGPipeline
from core.schemas import DomainModel
from fastapi import FastAPI
from pydantic import BaseModel

app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] System initializing...")

    # Get the directory where this script is located (backend directory)
    backend_dir = Path(__file__).parent.absolute()
    model_path = backend_dir / "domain" / "model.json"
    inputs_dir = backend_dir / "inputs"

    print(f"[DIR] Backend directory: {backend_dir}")
    print(f"[FILE] Model path: {model_path}")
    print(f"[DIR] Inputs directory: {inputs_dir}")

    # Desteklenen SRS formatları
    possible_srs_files = []
    possible_srs_files.extend(glob.glob(str(inputs_dir / "*.pdf")))
    possible_srs_files.extend(glob.glob(str(inputs_dir / "*.docx")))
    possible_srs_files.extend(glob.glob(str(inputs_dir / "*.txt")))

    # ------------------------------------------------------
    # 1) If domain model doesn't exist: Generate from SRS
    # ------------------------------------------------------
    if not model_path.exists():
        print("[WARN] Domain Model not found. Searching for SRS files...")

        if possible_srs_files:
            srs_file_path = possible_srs_files[0]
            print(f"[FILE] Found SRS file: {srs_file_path}")

            try:
                # 1) Parse & Chunk
                doc_parser = SRSDocumentParser()
                raw_text = doc_parser.parse_file(srs_file_path)
                print(f"   -> Parsed document with {len(raw_text)} characters.")

                print(f"   -> Parsed into {len(raw_text)} character.")

                if not raw_text.strip():
                    raise ValueError("Document is empty or could not be parsed.")

                # 2) DomainArchitect pipeline (Scout -> Architect -> Specialist)
                architect = DomainArchitect()

                analysis_results = architect.analyze_document(raw_text=raw_text)

                # 3) Merge -> Create JSON domain model
                print("[AI] Synthesizing final Domain Model JSON...")
                final_model: DomainModel = architect.synthesize_final_model(
                    analysis_results
                )

                print("   -> Domain Model synthesis complete.")

                # JSON olarak kaydet - ensure directory exists first
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(model_path, "w") as f:
                    f.write(final_model.model_dump_json(indent=2))

                app_state["domain_rules"] = final_model.model_dump(mode="json")
                print("[OK] Domain Model generated and saved successfully!")

            except Exception as e:
                print(f"[ERROR] Critical Error during generation: {e}")
                print(f"   [DEBUG] Working directory: {os.getcwd()}")
                print(f"   [DEBUG] Backend directory: {backend_dir}")
                print(f"   [DEBUG] Model path: {model_path}")
                print(f"   [DEBUG] Model directory exists: {model_path.parent.exists()}")
                import traceback

                print("   [DEBUG] Full traceback:")
                traceback.print_exc()
                app_state["domain_rules"] = {}

        else:
            print("[ERROR] No SRS file found in 'inputs/' folder.")
            app_state["domain_rules"] = {}

    # ------------------------------------------------------
    # 2) If domain model exists: Load it
    # ------------------------------------------------------
    else:
        print("[LOAD] Loading existing domain model...")
        try:
            with open(model_path, "r") as f:
                app_state["domain_rules"] = json.load(f)
            print("[OK] Domain model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Error loading existing model: {e}")
            print(f"   [DEBUG] Working directory: {os.getcwd()}")
            print(f"   [DEBUG] Backend directory: {backend_dir}")
            print(f"   [DEBUG] Model path: {model_path}")
            print(f"   [DEBUG] Model file exists: {model_path.exists()}")
            app_state["domain_rules"] = {}

    # Araçlar
    app_state["parser"] = CodeParser()
    app_state["llm"] = LLMClient()

    # ------------------------------------------------------
    # 3) Initialize RAG Pipeline and index SRS document
    # ------------------------------------------------------
    print("[RAG] Initializing RAG pipeline...")
    try:
        rag = RAGPipeline()

        # Index SRS document if available
        if possible_srs_files:
            srs_file_path = possible_srs_files[0]
            doc_parser = SRSDocumentParser()
            raw_text = doc_parser.parse_file(srs_file_path)

            if raw_text.strip():
                srs_filename = Path(srs_file_path).name
                srs_ext = Path(srs_file_path).suffix[1:]  # Remove leading dot

                chunk_count = rag.index_document(
                    raw_text=raw_text,
                    doc_id="srs_main",
                    doc_name=srs_filename,
                    doc_type=srs_ext
                )
                print(f"[RAG] Indexed {chunk_count} chunks from {srs_filename}")

        app_state["rag"] = rag
        print("[RAG] RAG pipeline initialized successfully!")

    except Exception as e:
        print(f"[RAG] Warning: RAG pipeline initialization failed: {e}")
        app_state["rag"] = None

    yield
    print("[SHUTDOWN] System shutting down...")


app = FastAPI(lifespan=lifespan)


# =============================================================================
# VALIDATE ENDPOINT
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
    result = llm.analyze_violation(ast_data, rules)  # type: ignore

    # Enhance violations with source references from RAG
    rag = app_state.get("rag")
    if result.get("is_violation") and rag:
        for violation in result.get("violations", []):
            try:
                sources = rag.retrieve_source(
                    violation_type=violation.get("type", ""),
                    violation_message=violation.get("message", ""),
                    n_results=2
                )
                violation["sources"] = sources
            except Exception:
                violation["sources"] = []

    return result


# =============================================================================
# RAG DIAGNOSTIC ENDPOINTS
# =============================================================================
@app.get("/rag/stats")
def get_rag_stats():
    """Return RAG index statistics."""
    rag = app_state.get("rag")
    if not rag:
        return {"status": "not_initialized", "message": "RAG pipeline not available"}
    return rag.get_stats()


@app.get("/rag/search")
def search_documents(query: str, n_results: int = 5):
    """Search indexed documents (for debugging)."""
    rag = app_state.get("rag")
    if not rag:
        return {"error": "RAG pipeline not initialized"}
    return {"query": query, "results": rag.search(query, n_results)}
