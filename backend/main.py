"""
DDD-Enforcer Backend Server

FastAPI server that provides code validation against Domain-Driven Design rules.
Integrates with LLM for intelligent violation detection and RAG for source tracking.
"""

import glob
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from config import (
    BASE_DIR,
    INPUTS_DIR,
    DOMAIN_DIR,
    DOMAIN_MODEL_PATH,
    RAGConfig,
    ParserConfig,
)
from core.architect import DomainArchitect
from core.document_parser import SRSDocumentParser
from core.llm_client import LLMClient
from core.parser import CodeParser
from core.rag_pipeline import RAGPipeline
from core.schemas import DomainModel

app_state = {}
rag_config = RAGConfig()


def find_srs_files() -> list:
    """Find all supported SRS files in the inputs directory."""
    files = []
    for ext in ParserConfig.SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(str(INPUTS_DIR / f"*{ext}")))
    return files


def generate_domain_model(srs_path: str) -> dict:
    """Generate domain model from SRS document using AI pipeline."""
    doc_parser = SRSDocumentParser()
    raw_text = doc_parser.parse_file(srs_path)
    print(f"   -> Parsed document: {len(raw_text)} characters")

    if not raw_text.strip():
        raise ValueError("Document is empty or could not be parsed.")

    architect = DomainArchitect()
    analysis_results = architect.analyze_document(raw_text=raw_text)

    print("[AI] Synthesizing final Domain Model JSON...")
    final_model: DomainModel = architect.synthesize_final_model(analysis_results)
    print("   -> Domain Model synthesis complete.")

    DOMAIN_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DOMAIN_MODEL_PATH, "w") as f:
        f.write(final_model.model_dump_json(indent=2))

    return final_model.model_dump(mode="json")


def load_existing_model() -> dict:
    """Load existing domain model from file."""
    with open(DOMAIN_MODEL_PATH, "r") as f:
        return json.load(f)


def initialize_rag(srs_files: list) -> RAGPipeline:
    """Initialize RAG pipeline and index SRS documents."""
    rag = RAGPipeline()

    if srs_files:
        srs_path = srs_files[0]
        doc_parser = SRSDocumentParser()
        raw_text = doc_parser.parse_file(srs_path)

        if raw_text.strip():
            filename = Path(srs_path).name
            ext = Path(srs_path).suffix[1:]
            chunk_count = rag.index_document(
                raw_text=raw_text,
                doc_id="srs_main",
                doc_name=filename,
                doc_type=ext
            )
            print(f"[RAG] Indexed {chunk_count} chunks from {filename}")

    return rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    print("[STARTUP] System initializing...")
    print(f"[DIR] Backend: {BASE_DIR}")
    print(f"[FILE] Model path: {DOMAIN_MODEL_PATH}")
    print(f"[DIR] Inputs: {INPUTS_DIR}")

    srs_files = find_srs_files()

    # Generate or load domain model
    if not DOMAIN_MODEL_PATH.exists():
        print("[WARN] Domain Model not found. Searching for SRS files...")
        if srs_files:
            print(f"[FILE] Found SRS file: {srs_files[0]}")
            try:
                app_state["domain_rules"] = generate_domain_model(srs_files[0])
                print("[OK] Domain Model generated and saved!")
            except Exception as e:
                print(f"[ERROR] Generation failed: {e}")
                import traceback
                traceback.print_exc()
                app_state["domain_rules"] = {}
        else:
            print("[ERROR] No SRS file found in 'inputs/' folder.")
            app_state["domain_rules"] = {}
    else:
        print("[LOAD] Loading existing domain model...")
        try:
            app_state["domain_rules"] = load_existing_model()
            print("[OK] Domain model loaded!")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            app_state["domain_rules"] = {}

    # Initialize tools
    app_state["parser"] = CodeParser()
    app_state["llm"] = LLMClient()

    # Initialize RAG pipeline
    print("[RAG] Initializing RAG pipeline...")
    try:
        app_state["rag"] = initialize_rag(srs_files)
        print("[RAG] RAG pipeline ready!")
    except Exception as e:
        print(f"[RAG] Warning: RAG initialization failed: {e}")
        app_state["rag"] = None

    yield

    print("[SHUTDOWN] System shutting down...")


app = FastAPI(lifespan=lifespan)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CodeSubmission(BaseModel):
    """Request model for code validation."""
    filename: str
    content: str


# =============================================================================
# VALIDATION ENDPOINT
# =============================================================================

@app.post("/validate")
def validate_code(submission: CodeSubmission):
    """
    Validate Python code against DDD rules.

    Returns violation report with source references from RAG.
    """
    parser = app_state.get("parser")
    llm = app_state.get("llm")
    rules = app_state.get("domain_rules")

    if not rules:
        return {
            "is_violation": True,
            "violations": [{
                "type": "ConfigError",
                "message": "Domain Model is empty. Check backend logs.",
                "suggestion": "Add inputs/srs.pdf and restart backend.",
            }],
        }

    # Parse code to AST
    ast_data = parser.parse_code(submission.content)

    if "error" in ast_data:
        return {
            "is_violation": True,
            "violations": [{
                "type": "SyntaxError",
                "message": ast_data["error"],
                "suggestion": "Fix Python syntax.",
            }],
        }

    # Check for violations
    result = llm.analyze_violation(ast_data, rules)

    # Add source references from RAG
    rag = app_state.get("rag")
    if result.get("is_violation") and rag:
        for violation in result.get("violations", []):
            try:
                sources = rag.retrieve_source(
                    violation_type=violation.get("type", ""),
                    violation_message=violation.get("message", "")
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
    """Get RAG index statistics."""
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
