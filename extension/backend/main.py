"""
DDD-Enforcer Backend Server

FastAPI server that provides code validation against Domain-Driven Design rules.
Integrates with LLM for intelligent violation detection and RAG for source tracking.
"""

import glob
import json
import os
import asyncio
import re
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from typing import Any, Dict, List, Generator, Optional, Tuple

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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
from core.code_parser import has_advanced_validation_signals
from core.document_parser import EmptySRSDocumentError, SRSDocumentParser
from core.llm.validator import LLMClient
from core.parser import CodeParser
from core.rag_pipeline import RAGPipeline
from core.schemas import DomainModel
from configs.models import model_for_stage
from core.token_tracker import TokenTracker
from core.validation_metrics import ValidationMetricsTracker
from core.AST.ast_model_signals import ASTModelSignalExtractor

app_state: Dict[str, Any] = {}
rag_config = RAGConfig()


def find_srs_files() -> List[str]:
    """Find all supported SRS files in the inputs directory."""
    files = []
    for ext in ParserConfig.SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(str(INPUTS_DIR / f"*{ext}")))
    return files


def _parse_srs_batch(
    parser: SRSDocumentParser,
    file_paths: List[str],
) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse all SRS files for a batch endpoint.

    Returns (combined_text, srs_docs, error).
    - error is None on success (at least one non-empty file parsed).
    - error is a {"success": False, "error": ...} dict on:
        * per-file non-empty exception (FileNotFoundError, unsupported, etc.)
        * all files parsed to empty content (post-loop aggregate check).
    Per-file EmptySRSDocumentError is logged and the file is skipped;
    partial batches succeed.
    """
    combined_text = ""
    srs_docs: List[Dict[str, Any]] = []
    for file_path in file_paths:
        try:
            raw_text = parser.parse_file(file_path)
        except EmptySRSDocumentError as exc:
            print(f"  ⚠️  Skipping empty document: {exc}")
            continue
        except Exception as exc:
            print(f"  ❌ Failed to parse: {exc}")
            return (
                "",
                [],
                {
                    "success": False,
                    "error": f"Failed to parse {Path(file_path).name}: {exc}",
                },
            )
        combined_text += f"\n\n--- Document: {Path(file_path).name} ---\n\n"
        combined_text += raw_text
        srs_docs.append({"path": file_path, "content": raw_text})

    if not srs_docs:
        return (
            "",
            [],
            {"success": False, "error": "All documents were empty after parsing"},
        )
    return combined_text, srs_docs, None


def generate_domain_model(srs_path: str) -> Dict[str, Any]:
    """Generate domain model from SRS document using AI pipeline."""
    doc_parser = SRSDocumentParser()
    raw_text = doc_parser.parse_file(srs_path)
    print(f"   -> Parsed document: {len(raw_text)} characters")

    architect = DomainArchitect()
    final_model: DomainModel = architect.analyze_document(text=raw_text)

    # AST enrichment for higher precision/traceability
    workspace_path = os.getenv("WORKSPACE_PATH", "")
    if workspace_path:
        extractor = ASTModelSignalExtractor()
        final_model = extractor.enrich_domain_model(
            final_model,
            workspace_path,
            srs_docs=[{"path": srs_path, "content": raw_text}],
        )

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
        try:
            raw_text = doc_parser.parse_file(srs_path)
        except EmptySRSDocumentError as exc:
            print(f"[RAG] skip empty SRS: {exc}")
            return rag
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
            from config import AnalyzerConfig
            domain_model_text = json.dumps(app_state["domain_rules"])
            app_state["domain_model_tokens"] = app_state["llm"].client.count_tokens(
                domain_model_text, AnalyzerConfig.MODEL_NAME
            )
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
# HEALTH & STATUS ENDPOINTS
# =============================================================================


@app.get("/health")
def health_check():
    """Health check endpoint for extension to verify backend is running."""
    return {
        "status": "healthy",
        "domain_model_loaded": bool(app_state.get("domain_rules")),
        "rag_initialized": app_state.get("rag") is not None,
    }


@app.get("/status")
def get_status():
    """Get detailed backend status."""
    return {
        "status": "running",
        "domain_model": {
            "loaded": bool(app_state.get("domain_rules")),
            "path": str(DOMAIN_MODEL_PATH) if DOMAIN_MODEL_PATH.exists() else None,
            "tokens": app_state.get("domain_model_tokens", 0),
        },
        "rag": {
            "initialized": app_state.get("rag") is not None,
        },
    }


# =============================================================================
# DOMAIN MODEL GENERATION ENDPOINT
# =============================================================================


class GenerateModelRequest(BaseModel):
    """Request model for domain model generation."""
    file_paths: List[str]  # Paths to SRS documents
    output_path: str  # Where to save model.json


def _estimate_tokens_from_text(text: str) -> int:
    """Cheap local token estimate to avoid an extra API call per validation."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def _is_semantically_empty_python(content: str) -> bool:
    """Return True if content is empty or comment-only (semantic no-op)."""
    lines = content.splitlines()
    has_code = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        has_code = True
        break
    return not has_code


def _needs_llm_advanced_checks(ast_data: Dict[str, Any]) -> bool:
    """Run LLM only when advanced signals exist in AST."""
    return has_advanced_validation_signals(ast_data)


def _merge_violations(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge and deduplicate violations by (type, message)."""
    merged: List[Dict[str, Any]] = []
    seen = set()
    for group in groups:
        for violation in group:
            key = (violation.get("type", ""), violation.get("message", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(violation)
    return merged


def _rag_cache_key(violation: Dict[str, Any]) -> str:
    """Build cache key for source retrieval deduplication."""
    v_type = violation.get("type", "")
    message = violation.get("message", "")
    match = re.search(r"'([^']+)'", message)
    focus = (match.group(1).lower() if match else message[:80].lower())
    return f"{v_type}:{focus}"


@app.post("/generate-model")
def generate_model_endpoint(request: GenerateModelRequest):
    """
    Generate domain model from uploaded SRS documents.
    
    This endpoint is called by the extension when user runs "Initialize Domain Model".
    """
    print(f"\n{'='*70}")
    print(f"🏗️  DOMAIN MODEL GENERATION REQUEST")
    print(f"{'='*70}")
    print(f"  Input files: {request.file_paths}")
    print(f"  Output path: {request.output_path}")
    
    if not request.file_paths:
        return {
            "success": False,
            "error": "No input files provided",
        }
    
    try:
        doc_parser = SRSDocumentParser()
        combined_text, srs_docs, error = _parse_srs_batch(
            doc_parser, request.file_paths
        )
        if error is not None:
            return error

        print(f"  📊 Total combined text: {len(combined_text)} characters")
        
        # Generate domain model using AI
        print("  🤖 Generating domain model with AI...")
        architect = DomainArchitect()
        final_model: DomainModel = architect.analyze_document(text=combined_text)

        # AST enrichment from workspace Python files
        workspace_path = os.getenv("WORKSPACE_PATH", "")
        if workspace_path:
            extractor = ASTModelSignalExtractor()
            final_model = extractor.enrich_domain_model(
                final_model,
                workspace_path,
                srs_docs=srs_docs,
            )
        
        # Save to specified output path
        output_path = Path(request.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(final_model.model_dump_json(indent=2))
        
        print(f"  ✅ Model saved to: {output_path}")
        
        # Update app state
        app_state["domain_rules"] = final_model.model_dump(mode="json")
        
        # Initialize RAG with the new documents
        print("  📚 Initializing RAG with documents...")
        try:
            rag = RAGPipeline()
            for file_path in request.file_paths:
                try:
                    raw_text = doc_parser.parse_file(file_path)
                except EmptySRSDocumentError as exc:
                    print(f"     [RAG] skip empty SRS: {exc}")
                    continue
                filename = Path(file_path).name
                ext = Path(file_path).suffix[1:]
                chunk_count = rag.index_document(
                    raw_text=raw_text,
                    doc_id=f"srs_{filename}",
                    doc_name=filename,
                    doc_type=ext,
                )
                print(f"     -> Indexed {chunk_count} chunks from {filename}")
            app_state["rag"] = rag
            print("  ✅ RAG initialized!")
        except Exception as e:
            print(f"  ⚠️  RAG initialization failed: {e}")
            app_state["rag"] = None
        
        # Print token usage
        tracker = TokenTracker.get_instance()
        tracker.print_summary()
        
        print(f"{'='*70}\n")
        
        return {
            "success": True,
            "model_path": str(output_path),
            "project_name": final_model.project_name,
            "bounded_contexts_count": len(final_model.bounded_contexts),
        }
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


@app.post("/generate-model-stream")
def generate_model_stream_endpoint(request: GenerateModelRequest):
    """
    Generate domain model with Server-Sent Events for real-time progress.
    
    This endpoint streams progress updates as the pipeline runs through:
    1. Scout - Extracting domain sentences
    2. Architect - Identifying bounded contexts
    3. Specialist - Analyzing context details
    4. Synthesizer - Creating final model
    """
    import queue
    import threading
    
    progress_queue = queue.Queue()
    result_holder = {"result": None, "error": None}
    
    def progress_callback(progress_data: Dict[str, Any]):
        """Callback to receive progress updates from DomainArchitect."""
        progress_queue.put({"type": "progress", "data": progress_data})
    
    def run_pipeline():
        """Run the model generation pipeline in a separate thread."""
        try:
            if not request.file_paths:
                result_holder["error"] = "No input files provided"
                return
            
            # Parse and combine all documents
            doc_parser = SRSDocumentParser()
            combined_text, srs_docs, error = _parse_srs_batch(
                doc_parser, request.file_paths
            )
            if error is not None:
                result_holder["error"] = error["error"]
                return
            
            # Generate domain model with progress callback
            architect = DomainArchitect(progress_callback=progress_callback)
            final_model: DomainModel = architect.analyze_document(text=combined_text)

            workspace_path = os.getenv("WORKSPACE_PATH", "")
            if workspace_path:
                extractor = ASTModelSignalExtractor()
                final_model = extractor.enrich_domain_model(
                    final_model,
                    workspace_path,
                    srs_docs=srs_docs,
                )
            
            # Save to specified output path
            output_path = Path(request.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                f.write(final_model.model_dump_json(indent=2))
            
            # Update app state
            app_state["domain_rules"] = final_model.model_dump(mode="json")
            
            # Initialize RAG
            try:
                rag = RAGPipeline()
                for file_path in request.file_paths:
                    try:
                        raw_text = doc_parser.parse_file(file_path)
                    except EmptySRSDocumentError as exc:
                        print(f"     [RAG] skip empty SRS: {exc}")
                        continue
                    filename = Path(file_path).name
                    ext = Path(file_path).suffix[1:]
                    rag.index_document(
                        raw_text=raw_text,
                        doc_id=f"srs_{filename}",
                        doc_name=filename,
                        doc_type=ext,
                    )
                app_state["rag"] = rag
            except Exception:
                app_state["rag"] = None
            
            # Get final metrics
            tracker = TokenTracker.get_instance()
            metrics = tracker.get_combined_metrics()
            
            result_holder["result"] = {
                "success": True,
                "model_path": str(output_path),
                "project_name": final_model.project_name,
                "bounded_contexts_count": len(final_model.bounded_contexts),
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"[STREAM] Domain model pipeline error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            result_holder["error"] = str(e)
        finally:
            progress_queue.put(None)  # Signal completion
    
    def event_generator() -> Generator[str, None, None]:
        """Generate SSE events from progress queue."""
        # Start pipeline in background thread
        thread = threading.Thread(target=run_pipeline)
        thread.start()
        
        # Stream progress events
        while True:
            try:
                item = progress_queue.get(timeout=120)  # 2 minute timeout
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        
        # Wait for thread to finish
        thread.join(timeout=5)
        
        # Send final result
        if result_holder["error"]:
            yield f"data: {json.dumps({'type': 'error', 'error': result_holder['error']})}\n\n"
        elif result_holder["result"]:
            yield f"data: {json.dumps({'type': 'complete', 'data': result_holder['result']})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
    
    # Local estimation avoids extra count_tokens API call.
    code_file_tokens = _estimate_tokens_from_text(submission.content)
    print(f"  📊 Code tokens (estimated): {code_file_tokens:,}")

    if _is_semantically_empty_python(submission.content):
        print("  ⏭️  Skip LLM: comment/blank-only content")
        return {
            "is_violation": False,
            "violations": [],
            "metrics": {
                "validation_time_ms": 0,
                "code_file_tokens": code_file_tokens,
                "llm_input_tokens": 0,
                "llm_output_tokens": 0,
                "llm_total_tokens": 0,
                "cost_usd": 0,
                "api_calls": 0,
            },
        }
    
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

    token_tracker = TokenTracker.get_instance()
    _pre_validator = token_tracker.tokens_for_stage("Validator")
    pre_input = _pre_validator.prompt_tokens
    pre_output = _pre_validator.completion_tokens
    pre_calls = token_tracker.stats.total_api_calls

    # Run deterministic checks first (no LLM token cost)
    deterministic_violations = llm.rule_based_name_violations(ast_data, rules)
    if deterministic_violations:
        print(f"  ⚙️  Deterministic checks found {len(deterministic_violations)} violation(s)")

    # Run LLM only for advanced checks if required by AST signals
    advanced_result = {"is_violation": False, "violations": []}
    if _needs_llm_advanced_checks(ast_data):
        print("  🤖 Analyzing advanced rules with LLM...")
        advanced_result = llm.analyze_advanced_violations(ast_data, rules)
    else:
        print("  ⏭️  Skip LLM: no advanced AST signals")

    merged_violations = _merge_violations(
        deterministic_violations,
        advanced_result.get("violations", []),
    )
    result = {
        "is_violation": len(merged_violations) > 0,
        "violations": merged_violations,
    }

    # Add source references from RAG
    rag = app_state.get("rag")
    has_sources = False
    if result.get("is_violation") and rag:
        violations_list = result.get("violations", [])
        print(f"  📚 Fetching RAG sources for {len(violations_list)} violation(s)...")
        rag_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        for idx, violation in enumerate(violations_list):
            print(f"     [{idx+1}/{len(violations_list)}] Processing: {violation.get('type', 'Unknown')}")
            try:
                cache_key = _rag_cache_key(violation)
                if cache_key in rag_cache:
                    sources = rag_cache[cache_key]
                    print("     ♻️  Reusing cached sources")
                else:
                    print(f"     🔍 Retrieving sources for: {violation.get('type', 'Unknown')}")
                    sources = rag.retrieve_source(
                        violation_type=violation.get("type", ""),
                        violation_message=violation.get("message", ""),
                    )
                    rag_cache[cache_key] = sources
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
    
    # Get token usage delta for this validation call
    _post_validator = token_tracker.tokens_for_stage("Validator")
    post_input = _post_validator.prompt_tokens
    post_output = _post_validator.completion_tokens
    post_calls = token_tracker.stats.total_api_calls
    llm_input_delta = max(0, post_input - pre_input)
    llm_output_delta = max(0, post_output - pre_output)
    llm_calls_delta = max(0, post_calls - pre_calls)

    call_cost = model_for_stage("Validator").pricing.cost_for(
        prompt_tokens=llm_input_delta,
        completion_tokens=llm_output_delta,
    )

    validation_metrics = {
        "validation_time_ms": round(validation_time_ms, 2),
        "code_file_tokens": code_file_tokens,
        "llm_input_tokens": llm_input_delta,
        "llm_output_tokens": llm_output_delta,
        "llm_total_tokens": llm_input_delta + llm_output_delta,
        "cost_usd": round(call_cost, 8),
        "api_calls": llm_calls_delta,
    }
    
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
    print(f"  📊 Tokens: {validation_metrics['llm_total_tokens']:,} (in: {validation_metrics['llm_input_tokens']:,}, out: {validation_metrics['llm_output_tokens']:,})")
    print(f"  💰 Cost: ${validation_metrics['cost_usd']:.6f}")
    print(f"  📤 Returning response: is_violation={result.get('is_violation')}, violations={violations_count}")
    print(f"{'='*70}\n")

    # Add metrics to result
    result["metrics"] = validation_metrics
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
    and monthly cost projections. Now supports multi-model pricing:
    - Models read from configs/models.py registry; current defaults below.
    """
    token_tracker = TokenTracker.get_instance()
    validation_tracker = ValidationMetricsTracker.get_instance()
    
    token_report = token_tracker.get_report(detailed=False)
    validation_report = validation_tracker.get_report(detailed=False)
    
    # Calculate per-validation averages (using the Validator stage's registry-pinned model).
    total_validations = validation_report.get("summary", {}).get("total_validations", 0)
    per_validation_metrics = {}
    monthly_projections = {}
    
    if total_validations > 0 and total_validations is not None:
        # Get validator-specific costs using registry model id
        validator_id = model_for_stage("Validator").model_id
        by_model_costs = token_report.get("cost_estimation", {}).get("by_model", {})
        validator_cost = by_model_costs.get(validator_id, {}).get("total_cost", 0)
        validator_model_usage = token_report.get("model_usage", {}).get(validator_id, {})
        validator_input = validator_model_usage.get("prompt_tokens", 0)
        validator_output = validator_model_usage.get("completion_tokens", 0)

        avg_latency = validation_report.get("performance", {}).get("avg_validation_time_ms", 0)

        per_validation_metrics = {
            "model": validator_id,
            "avg_cost_per_validation": round(validator_cost / total_validations, 8) if total_validations > 0 else 0,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_input_tokens": round(validator_input / total_validations, 2) if total_validations > 0 else 0,
            "avg_output_tokens": round(validator_output / total_validations, 2) if total_validations > 0 else 0,
            "avg_total_tokens": round((validator_input + validator_output) / total_validations, 2) if total_validations > 0 else 0
        }

        # Monthly projections (1000 validations/day * 30 days) - using validator model pricing
        validations_per_month = 1000 * 30  # 30,000 validations
        cost_per_validation = validator_cost / total_validations if total_validations > 0 else 0
        monthly_cost = cost_per_validation * validations_per_month

        monthly_projections = {
            "validations_per_day": 1000,
            "validations_per_month": validations_per_month,
            "estimated_monthly_cost": round(monthly_cost, 4),
            "estimated_monthly_input_tokens": round((validator_input / total_validations) * validations_per_month, 0) if total_validations > 0 else 0,
            "estimated_monthly_output_tokens": round((validator_output / total_validations) * validations_per_month, 0) if total_validations > 0 else 0,
            "currency": "USD",
            "model": validator_id,
            "pricing": {
                "input_per_1m": model_for_stage("Validator").pricing.cost_for(1_000_000, 0),
                "output_per_1m": model_for_stage("Validator").pricing.cost_for(0, 1_000_000),
            },
        }
    
    # Domain model information (generated with flash model)
    domain_info = {
        "domain_model_tokens": app_state.get("domain_model_tokens", 0),
        "domain_model_path": str(DOMAIN_MODEL_PATH) if DOMAIN_MODEL_PATH.exists() else None,
        "generation_model": model_for_stage("Architect").model_id
    }
    
    # Model pricing reference (derived from configs/models.py registry)
    architect_info = model_for_stage("Architect")
    validator_info = model_for_stage("Validator")
    pricing_reference = {
        architect_info.model_id: {
            "use_case": "Domain Model Generation",
            "stages": ["Scout", "Architect", "Specialist", "Synthesizer"],
            "input_per_1m_tokens": architect_info.pricing.cost_for(1_000_000, 0) * 1_000_000,
            "output_per_1m_tokens": architect_info.pricing.cost_for(0, 1_000_000) * 1_000_000,
            "note": "Output price includes thinking tokens"
        },
        validator_info.model_id: {
            "use_case": "Code Validation",
            "stages": ["Validator"],
            "input_per_1m_tokens": validator_info.pricing.cost_for(1_000_000, 0) * 1_000_000,
            "output_per_1m_tokens": validator_info.pricing.cost_for(0, 1_000_000) * 1_000_000,
            "note": "Output price includes thinking tokens"
        }
    }
    
    return {
        "domain_model": domain_info,
        "token_usage": token_report,
        "validation_metrics": validation_report,
        "per_validation_averages": per_validation_metrics,
        "monthly_projections": monthly_projections,
        "pricing_reference": pricing_reference
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
