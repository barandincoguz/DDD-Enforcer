"""
DDD-Enforcer backend server.

This backend powers the VS Code extension and now also exposes research-facing
metrics exports suitable for experiment automation.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
import queue
import threading
from typing import Any, Dict, Generator, List, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    AnalyzerConfig,
    ArchitectConfig,
    BASE_DIR,
    DOMAIN_MODEL_PATH,
    INPUTS_DIR,
    ParserConfig,
    PricingConfig,
    ResearchArtifactsConfig,
    WORKSPACE_PATH,
)
from core.llm_client import LLMClient
from core.model_generation_service import ModelGenerationService
from core.research_metrics import ResearchMetricsStore
from core.rag_pipeline import RAGPipeline
from core.token_tracker import TokenTracker
from core.validation_metrics import ValidationMetricsTracker
from core.validation_service import ValidationService

app_state: Dict[str, Any] = {}


def find_srs_files() -> List[str]:
    """Find supported SRS files in the inputs directory."""
    return sorted(
        str(path)
        for extension in ParserConfig.SUPPORTED_EXTENSIONS
        for path in INPUTS_DIR.glob(f"*{extension}")
    )


def load_existing_model() -> Dict[str, Any]:
    with DOMAIN_MODEL_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_domain_model_tokens(domain_rules: Dict[str, Any]) -> int:
    if not domain_rules:
        return 0
    llm_client: Optional[LLMClient] = app_state.get("llm")
    if not llm_client:
        return 0
    try:
        return llm_client.provider.count_tokens(
            model=ArchitectConfig.MODEL_NAME,
            text=json.dumps(domain_rules),
        )
    except Exception:
        return 0


def _get_validation_service() -> ValidationService:
    return app_state["validation_service"]


def _get_generation_service() -> ModelGenerationService:
    return app_state["generation_service"]


def _get_rag() -> Optional[RAGPipeline]:
    return app_state.get("rag")


def _update_domain_runtime_state(domain_rules: Dict[str, Any], file_paths: List[str]) -> None:
    app_state["domain_rules"] = domain_rules
    app_state["domain_model_tokens"] = _count_domain_model_tokens(domain_rules)
    app_state["rag"] = _get_generation_service().initialize_rag(file_paths)


def _build_generation_response(
    generation: Dict[str, Any],
    model_path: str,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = {
        "success": True,
        "model_path": model_path,
        "project_name": generation["project_name"],
        "bounded_contexts_count": generation["bounded_contexts_count"],
        "domain_model": generation["model"],
        "metrics": metrics if metrics is not None else generation["metrics"],
        "stage_latencies_ms": generation["stage_latencies_ms"],
    }
    if "verification_report" in generation:
        response["verification_report"] = generation["verification_report"]
    return response


def _build_rag_unavailable_response() -> Dict[str, str]:
    return {"status": "not_initialized", "message": "RAG pipeline not available"}


def _build_missing_domain_model_response() -> Dict[str, Any]:
    return {
        "is_violation": True,
        "violations": [
            {
                "type": "ConfigError",
                "message": "Domain Model is empty. Check backend logs.",
                "suggestion": "Generate a domain model before validation.",
            }
        ],
    }


def _stream_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _build_runtime_services() -> None:
    llm_client = LLMClient()
    validation_service = ValidationService(llm_client=llm_client)
    generation_service = ModelGenerationService()

    app_state["llm"] = llm_client
    app_state["validation_service"] = validation_service
    app_state["generation_service"] = generation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize runtime services and best-effort model/RAG state."""
    print("[STARTUP] System initializing...")
    print(f"[DIR] Backend: {BASE_DIR}")
    print(f"[FILE] Model path: {DOMAIN_MODEL_PATH}")
    print(f"[DIR] Inputs: {INPUTS_DIR}")

    _build_runtime_services()
    possible_srs_files = find_srs_files()
    generation_service = _get_generation_service()

    if DOMAIN_MODEL_PATH.exists():
        try:
            app_state["domain_rules"] = load_existing_model()
            print("[LOAD] Existing domain model loaded.")
        except Exception as exc:
            print(f"[ERROR] Failed to load domain model: {exc}")
            app_state["domain_rules"] = {}
    elif possible_srs_files:
        try:
            generation = generation_service.generate_from_files(
                file_paths=[possible_srs_files[0]],
                output_path=str(DOMAIN_MODEL_PATH),
                workspace_path=WORKSPACE_PATH,
            )
            app_state["domain_rules"] = generation["model"]
            print("[LOAD] Domain model generated at startup.")
        except Exception as exc:
            print(f"[ERROR] Failed to generate startup model: {exc}")
            app_state["domain_rules"] = {}
    else:
        app_state["domain_rules"] = {}

    app_state["domain_model_tokens"] = _count_domain_model_tokens(app_state["domain_rules"])

    try:
        app_state["rag"] = generation_service.initialize_rag(possible_srs_files)
    except Exception as exc:
        print(f"[RAG] Initialization failed: {exc}")
        app_state["rag"] = None

    yield
    print("[SHUTDOWN] System shutting down...")


app = FastAPI(lifespan=lifespan)


class GenerateModelRequest(BaseModel):
    """Request payload for domain-model generation."""

    file_paths: List[str]
    output_path: str


class CodeSubmission(BaseModel):
    """Request payload for validation."""

    filename: str
    content: str


@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "domain_model_loaded": bool(app_state.get("domain_rules")),
        "rag_initialized": app_state.get("rag") is not None,
    }


@app.get("/status")
def get_status() -> Dict[str, Any]:
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


@app.post("/generate-model")
def generate_model_endpoint(request: GenerateModelRequest) -> Dict[str, Any]:
    generation = _get_generation_service().generate_from_files(
        file_paths=request.file_paths,
        output_path=request.output_path,
        workspace_path=WORKSPACE_PATH,
    )
    _update_domain_runtime_state(generation["model"], request.file_paths)
    return _build_generation_response(generation, request.output_path)


@app.post("/generate-model-stream")
def generate_model_stream_endpoint(request: GenerateModelRequest):
    progress_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    result_holder: Dict[str, Any] = {"result": None, "error": None}

    def progress_callback(progress_data: Dict[str, Any]) -> None:
        progress_queue.put({"type": "progress", "data": progress_data})

    def run_pipeline() -> None:
        try:
            result_holder["result"] = _get_generation_service().generate_from_files(
                file_paths=request.file_paths,
                output_path=request.output_path,
                workspace_path=WORKSPACE_PATH,
                progress_callback=progress_callback,
            )
            _update_domain_runtime_state(result_holder["result"]["model"], request.file_paths)
        except Exception as exc:
            result_holder["error"] = str(exc)
        finally:
            progress_queue.put(None)

    def event_generator() -> Generator[str, None, None]:
        thread = threading.Thread(target=run_pipeline)
        thread.start()
        while True:
            try:
                item = progress_queue.get(timeout=120)
                if item is None:
                    break
                yield _stream_event(item)
            except queue.Empty:
                yield _stream_event({"type": "heartbeat"})

        thread.join(timeout=5)
        if result_holder["error"]:
            yield _stream_event({"type": "error", "error": result_holder["error"]})
        else:
            data = result_holder["result"]
            yield _stream_event(
                {
                    "type": "complete",
                    "data": _build_generation_response(
                        data,
                        request.output_path,
                        metrics=TokenTracker.get_instance().get_combined_metrics(),
                    ),
                }
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/validate")
def validate_code(submission: CodeSubmission) -> Dict[str, Any]:
    rules = app_state.get("domain_rules")
    if not rules:
        return _build_missing_domain_model_response()
    return _get_validation_service().validate(
        filename=submission.filename,
        content=submission.content,
        domain_rules=rules,
        rag=_get_rag(),
        mode="pipeline",
    )


@app.get("/rag/stats")
def get_rag_stats():
    rag = _get_rag()
    if rag is None:
        return _build_rag_unavailable_response()
    return rag.get_stats()


@app.get("/rag/search")
def search_documents(query: str, n_results: int = 5):
    rag = _get_rag()
    if rag is None:
        return _build_rag_unavailable_response()
    return rag.search(query=query, n_results=n_results)


@app.get("/tokens/stats")
def get_token_stats():
    return TokenTracker.get_instance().get_report(detailed=True)


@app.get("/tokens/summary")
def get_token_summary():
    return TokenTracker.get_instance().get_report(detailed=False)


@app.post("/tokens/reset")
def reset_token_tracker():
    TokenTracker.reset()
    ValidationMetricsTracker.reset()
    ResearchMetricsStore.reset()
    _build_runtime_services()
    return {"status": "success", "message": "Token and validation trackers have been reset"}


@app.get("/tokens/export")
def export_token_report():
    export_path = str(BASE_DIR / "token_usage_export.json")
    TokenTracker.get_instance().export_to_json(export_path, detailed=True)
    return {"status": "success", "file_path": export_path}


@app.get("/metrics/validation")
def get_validation_metrics():
    return ValidationMetricsTracker.get_instance().get_report(detailed=True)


@app.get("/metrics/validation/summary")
def get_validation_summary():
    return ValidationMetricsTracker.get_instance().get_report(detailed=False)


@app.get("/metrics/research")
def get_research_metrics():
    return ResearchMetricsStore.get_instance().get_report(detailed=True)


@app.get("/metrics/export")
def export_research_metrics():
    exported = ResearchMetricsStore.get_instance().export_csvs(ResearchArtifactsConfig.DEFAULT_EXPORT_DIR)
    return {"status": "success", "exports": exported}


@app.get("/metrics/combined")
def get_combined_metrics():
    token_report = TokenTracker.get_instance().get_report(detailed=False)
    validation_report = ValidationMetricsTracker.get_instance().get_report(detailed=False)
    research_report = ResearchMetricsStore.get_instance().get_report(detailed=False)
    total_validations = validation_report.get("summary", {}).get("total_validations", 0)
    avg_cost_per_validation = (
        validation_report.get("llm_usage", {}).get("avg_cost_usd", 0.0) if total_validations else 0.0
    )

    monthly_validations = 30_000
    monthly_cost = round(avg_cost_per_validation * monthly_validations, 8)
    return {
        "domain_model": {
            "domain_model_tokens": app_state.get("domain_model_tokens", 0),
            "domain_model_path": str(DOMAIN_MODEL_PATH) if DOMAIN_MODEL_PATH.exists() else None,
            "generation_model": ArchitectConfig.MODEL_NAME,
        },
        "token_usage": token_report,
        "validation_metrics": validation_report,
        "research_metrics": research_report,
        "per_validation_averages": {
            "model": AnalyzerConfig.MODEL_NAME,
            "avg_latency_ms": validation_report.get("performance", {}).get("avg_validation_time_ms", 0.0),
            "avg_cost_per_validation": avg_cost_per_validation,
            "avg_input_tokens": validation_report.get("llm_usage", {}).get("avg_input_tokens", 0.0),
            "avg_output_tokens": validation_report.get("llm_usage", {}).get("avg_output_tokens", 0.0),
            "avg_total_tokens": validation_report.get("llm_usage", {}).get("avg_total_tokens", 0.0),
        },
        "monthly_projections": {
            "validations_per_day": 1000,
            "validations_per_month": monthly_validations,
            "estimated_monthly_cost": monthly_cost,
            "currency": PricingConfig.CURRENCY,
            "model": AnalyzerConfig.MODEL_NAME,
        },
        "pricing_reference": PricingConfig.MODEL_PRICING,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
