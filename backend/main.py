"""
DDD-Enforcer Backend Server

FastAPI server that provides code validation against Domain-Driven Design rules.
Integrates with LLM for intelligent violation detection and RAG for source tracking.
"""

import glob
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from typing import Any, Dict, List

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
from core.token_tracker import TokenTracker
from core.validation_metrics import ValidationMetricsTracker

app_state: Dict[str, Any] = {}
rag_config = RAGConfig()


def find_srs_files() -> List[str]:
    """Find all supported SRS files in the inputs directory."""
    files = []
    for ext in ParserConfig.SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(str(INPUTS_DIR / f"*{ext}")))
    return files


def generate_domain_model(srs_path: str) -> Dict[str, Any]:
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


def load_existing_model() -> Dict[str, Any]:
    """Load existing domain model from file."""
    with open(DOMAIN_MODEL_PATH, "r") as f:
        return json.load(f)


def initialize_rag(srs_files: List[str]) -> RAGPipeline:
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
                raw_text=raw_text, doc_id="srs_main", doc_name=filename, doc_type=ext
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

    # Find all supported SRS files
    possible_srs_files = find_srs_files()

    # Generate or load domain model
    if not DOMAIN_MODEL_PATH.exists():
        print("[WARN] Domain Model not found. Searching for SRS files...")
        if possible_srs_files:
            print(f"[FILE] Found SRS file: {possible_srs_files[0]}")
            try:
                app_state["domain_rules"] = generate_domain_model(possible_srs_files[0])
                print("[OK] Domain Model generated and saved!")
                
                # Print token usage report
                tracker = TokenTracker.get_instance()
                tracker.print_summary()
                tracker.export_to_json(str(BASE_DIR / "token_usage_report.json"), detailed=True)
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
    
    # Calculate domain model token size for metrics
    if app_state.get("domain_rules"):
        try:
            from google import genai
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            domain_model_text = json.dumps(app_state["domain_rules"])
            token_count_response = client.models.count_tokens(
                model="gemini-2.5-flash-lite",
                contents=domain_model_text
            )
            app_state["domain_model_tokens"] = token_count_response.total_tokens
            print(f"[METRICS] Domain model size: {app_state['domain_model_tokens']:,} tokens")
        except Exception as e:
            print(f"[METRICS] Could not calculate domain model tokens: {e}")
            app_state["domain_model_tokens"] = 0
    else:
        app_state["domain_model_tokens"] = 0

    # Initialize RAG pipeline
    print("[RAG] Initializing RAG pipeline...")
    try:
        if possible_srs_files:
            app_state["rag"] = initialize_rag(possible_srs_files)
            print("[RAG] RAG pipeline ready!")
        else:
            print("[RAG] No SRS files found. Skipping RAG initialization.")
            app_state["rag"] = None
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
    Tracks metrics for UBMK presentation.
    """
    import time
    
    print(f"\n{'='*70}")
    print(f"🔍 VALIDATION REQUEST")
    print(f"{'='*70}")
    print(f"  File: {submission.filename}")
    print(f"  Size: {len(submission.content)} chars")
    
    # Count code file tokens for metrics (BEFORE latency measurement)
    code_file_tokens = 0
    try:
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        token_count_response = client.models.count_tokens(
            model="gemini-2.0-flash-exp",
            contents=submission.content
        )
        code_file_tokens = token_count_response.total_tokens
        print(f"  📊 Code tokens: {code_file_tokens:,}")
    except Exception as e:
        print(f"  ⚠️  Token counting failed: {e}")
    
    # START LATENCY MEASUREMENT - only actual validation logic
    start_time = time.time()
    
    parser = app_state.get("parser")
    llm = app_state.get("llm")
    rules = app_state.get("domain_rules")
    validation_tracker = ValidationMetricsTracker.get_instance()

    if not rules:
        print("  ❌ ERROR: Domain model not loaded")
        return {
            "is_violation": True,
            "violations": [
                {
                    "type": "ConfigError",
                    "message": "Domain Model is empty. Check backend logs.",
                    "suggestion": "Add inputs/srs.pdf and restart backend.",
                }
            ],
        }

    # Parse code to AST with filename
    ast_data = parser.parse_code(submission.content, submission.filename)

    if "error" in ast_data:
        print(f"  ❌ SYNTAX ERROR: {ast_data['error']}")
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

    # Check for violations
    print("  🤖 Analyzing with LLM...")
    result = llm.analyze_violation(ast_data, rules)

    # Add source references from RAG
    rag = app_state.get("rag")
    has_sources = False
    if result.get("is_violation") and rag:
        violations_list = result.get("violations", [])
        print(f"  📚 Fetching RAG sources for {len(violations_list)} violation(s)...")
        
        for idx, violation in enumerate(violations_list):
            print(f"     [{idx+1}/{len(violations_list)}] Processing: {violation.get('type', 'Unknown')}")
            try:
                print(f"     🔍 Retrieving sources for: {violation.get('type', 'Unknown')}")
                sources = rag.retrieve_source(
                    violation_type=violation.get("type", ""),
                    violation_message=violation.get("message", ""),
                )
                print(f"     ✅ Found {len(sources)} sources")
                violation["sources"] = sources
                if sources:
                    has_sources = True
            except Exception as e:
                print(f"     ❌ RAG ERROR: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                violation["sources"] = []
            print(f"     ✔️  Violation [{idx+1}] completed")
        
        print(f"  ✅ All RAG sources fetched!")
    
    
    # Track validation metrics
    validation_time_ms = (time.time() - start_time) * 1000
    violations_count = len(result.get("violations", []))
    
    validation_tracker.track_validation(
        filename=submission.filename,
        file_size_chars=len(submission.content),
        code_file_tokens=code_file_tokens,
        validation_time_ms=validation_time_ms,
        violations=result.get("violations", []),
        has_sources=has_sources
    )
    
    # Print result
    if result.get("is_violation"):
        print(f"  ⚠️  VIOLATIONS FOUND: {violations_count}")
        for v in result.get("violations", []):
            print(f"     - {v.get('type')}: {v.get('message', '')[:60]}...")
            sources_count = len(v.get("sources", []))
            print(f"       📚 Sources attached: {sources_count}")
    else:
        print("  ✅ NO VIOLATIONS - Code is clean!")
    
    print(f"  ⏱️  Time: {validation_time_ms:.2f}ms")
    print(f"  📤 Returning response: is_violation={result.get('is_violation')}, violations={violations_count}")
    print(f"{'='*70}\n")

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
    """Search indexed documents by query."""
    rag = app_state.get("rag")
    if not rag:
        return {"status": "not_initialized", "message": "RAG pipeline not available"}
    return rag.search(query, top_k=n_results)


# =============================================================================
# TOKEN USAGE & COST TRACKING ENDPOINTS
# =============================================================================


@app.get("/tokens/stats")
def get_token_stats():
    """
    Get comprehensive token usage statistics and cost estimation.
    
    Returns:
        - Total tokens used (prompt + completion)
        - Per-stage breakdown
        - Cost estimation based on Gemini 2.5 Flash pricing
        - Detailed call history
    """
    tracker = TokenTracker.get_instance()
    return tracker.get_report(detailed=True)


@app.get("/tokens/summary")
def get_token_summary():
    """Get concise token usage summary without detailed call history."""
    tracker = TokenTracker.get_instance()
    return tracker.get_report(detailed=False)


@app.post("/tokens/reset")
def reset_token_tracker():
    """Reset token tracker (for testing purposes)."""
    TokenTracker.reset()
    return {"status": "success", "message": "Token tracker has been reset"}


@app.get("/tokens/export")
def export_token_report():
    """Export detailed token report to JSON file."""
    tracker = TokenTracker.get_instance()
    export_path = str(BASE_DIR / "token_usage_export.json")
    tracker.export_to_json(export_path, detailed=True)
    return {
        "status": "success", 
        "message": f"Report exported to {export_path}",
        "file_path": export_path
    }


# =============================================================================
# VALIDATION METRICS ENDPOINTS
# =============================================================================


@app.get("/metrics/validation")
def get_validation_metrics():
    """
    Get comprehensive validation metrics.
    
    Returns:
        - Total validations performed
        - Violations found (total and breakdown by type)
        - Performance metrics (avg latency)
        - RAG usage statistics
    """
    tracker = ValidationMetricsTracker.get_instance()
    return tracker.get_report(detailed=True)


@app.get("/metrics/validation/summary")
def get_validation_summary():
    """Get concise validation summary without detailed history."""
    tracker = ValidationMetricsTracker.get_instance()
    return tracker.get_report(detailed=False)


@app.get("/metrics/combined")
def get_combined_metrics():
    """
    Get combined metrics for UBMK presentation.
    
    Returns both token usage and validation metrics with per-validation averages
    and monthly cost projections.
    """
    token_tracker = TokenTracker.get_instance()
    validation_tracker = ValidationMetricsTracker.get_instance()
    
    token_report = token_tracker.get_report(detailed=False)
    validation_report = validation_tracker.get_report(detailed=False)
    
    # Calculate per-validation averages
    total_validations = validation_report.get("summary", {}).get("total_validations", 0)
    per_validation_metrics = {}
    monthly_projections = {}
    
    if total_validations > 0:
        total_cost = token_report.get("cost_estimation", {}).get("total_cost", 0)
        avg_latency = validation_report.get("performance", {}).get("avg_validation_time_ms", 0)
        total_tokens = token_report.get("summary", {}).get("total_tokens", 0)
        avg_input_tokens = token_report.get("summary", {}).get("total_prompt_tokens", 0) / total_validations
        avg_output_tokens = token_report.get("summary", {}).get("total_completion_tokens", 0) / total_validations
        
        per_validation_metrics = {
            "avg_cost_per_validation": round(total_cost / total_validations, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "avg_tokens_per_validation": round(total_tokens / total_validations, 2),
            "avg_input_tokens": round(avg_input_tokens, 2),
            "avg_output_tokens": round(avg_output_tokens, 2)
        }
        
        # Monthly projections (1000 validations/day * 30 days)
        validations_per_month = 1000 * 30  # 30,000 validations
        monthly_cost = (total_cost / total_validations) * validations_per_month
        
        monthly_projections = {
            "validations_per_day": 1000,
            "validations_per_month": validations_per_month,
            "estimated_monthly_cost": round(monthly_cost, 2),
            "estimated_monthly_input_tokens": round(avg_input_tokens * validations_per_month, 0),
            "estimated_monthly_output_tokens": round(avg_output_tokens * validations_per_month, 0),
            "currency": "USD"
        }
    
    # Domain model information
    domain_info = {
        "domain_model_tokens": app_state.get("domain_model_tokens", 0),
        "domain_model_path": str(DOMAIN_MODEL_PATH) if DOMAIN_MODEL_PATH.exists() else None
    }
    
    return {
        "domain_model": domain_info,
        "token_usage": token_report,
        "validation_metrics": validation_report,
        "per_validation_averages": per_validation_metrics,
        "monthly_projections": monthly_projections
    }
    
    return {
        "token_usage": token_report,
        "validation_metrics": validation_report,
        "per_validation_averages": per_validation_metrics
    }


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
    """Search indexed documents by query."""
    rag = app_state.get("rag")
    if not rag:
        return {"status": "not_initialized", "message": "RAG pipeline not available"}
    return rag.search(query, top_k=n_results)

