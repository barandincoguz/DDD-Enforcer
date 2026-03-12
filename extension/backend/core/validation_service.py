"""
Shared validation service used by the FastAPI backend and experiment runner.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Literal, Optional

from core.llm_client import LLMClient
from core.parser import CodeParser
from core.rag_pipeline import RAGPipeline
from core.research_metrics import ResearchMetricsStore
from core.token_tracker import TokenTracker
from core.validation_metrics import ValidationMetricsTracker

ValidationMode = Literal["pipeline", "naive"]


def estimate_tokens_from_text(text: str) -> int:
    """Cheap local token estimate to avoid a count_tokens API call per validation."""
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def is_semantically_empty_python(content: str) -> bool:
    """Return True if content is blank or comment-only."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return False
    return True


def needs_llm_advanced_checks(ast_data: Dict[str, Any]) -> bool:
    """Run advanced LLM checks only when there are non-trivial AST signals."""
    return bool(
        ast_data.get("imports")
        or ast_data.get("assignments")
        or ast_data.get("function_calls")
    )


def merge_violations(*groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge and deduplicate violations by type/message."""
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


def rag_cache_key(violation: Dict[str, Any]) -> str:
    """Build cache key for RAG source retrieval deduplication."""
    violation_type = violation.get("type", "")
    message = violation.get("message", "")
    match = re.search(r"'([^']+)'", message)
    focus = match.group(1).lower() if match else message[:80].lower()
    return f"{violation_type}:{focus}"


class ValidationService:
    """Validation orchestrator with metrics, baseline modes, and traceability."""

    def __init__(
        self,
        *,
        parser: Optional[CodeParser] = None,
        llm_client: Optional[LLMClient] = None,
        validation_tracker: Optional[ValidationMetricsTracker] = None,
        token_tracker: Optional[TokenTracker] = None,
        research_metrics: Optional[ResearchMetricsStore] = None,
    ):
        self.parser = parser or CodeParser()
        self.llm_client = llm_client or LLMClient()
        self.validation_tracker = validation_tracker or ValidationMetricsTracker.get_instance()
        self.token_tracker = token_tracker or TokenTracker.get_instance()
        self.research_metrics = research_metrics or ResearchMetricsStore.get_instance()

    def validate(
        self,
        *,
        filename: str,
        content: str,
        domain_rules: Dict[str, Any],
        rag: Optional[RAGPipeline] = None,
        mode: ValidationMode = "pipeline",
        run_metadata: Optional[Dict[str, Any]] = None,
        expected_retrieval_sections: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Validate a single file using the selected execution mode."""
        file_size_chars = len(content)
        file_loc = len(content.splitlines())
        code_file_tokens = estimate_tokens_from_text(content)

        if is_semantically_empty_python(content):
            result = {
                "is_violation": False,
                "violations": [],
                "mode": mode,
                "metrics": {
                    "validation_time_ms": 0.0,
                    "file_size_chars": file_size_chars,
                    "file_loc": file_loc,
                    "code_file_tokens": code_file_tokens,
                    "stage_latencies_ms": {
                        "ast_parse": 0.0,
                        "deterministic_rules": 0.0,
                        "advanced_llm": 0.0,
                        "naive_llm": 0.0,
                        "rag": 0.0,
                        "total": 0.0,
                    },
                    "provider": getattr(self.llm_client.provider, "provider_name", None),
                    "model": self.llm_client.model_name,
                    "llm_input_tokens": 0,
                    "llm_output_tokens": 0,
                    "llm_total_tokens": 0,
                    "cached_tokens": 0,
                    "cost_usd": 0.0,
                    "api_calls": 0,
                    "parseable_outputs": 0,
                    "unparseable_outputs": 0,
                },
            }
            self._record_validation(
                filename=filename,
                file_size_chars=file_size_chars,
                code_file_tokens=code_file_tokens,
                result=result,
                mode=mode,
                run_metadata=run_metadata,
            )
            return result

        total_start = time.perf_counter()
        token_snapshot = self.token_tracker.snapshot()
        stage_latencies = {
            "ast_parse": 0.0,
            "deterministic_rules": 0.0,
            "advanced_llm": 0.0,
            "naive_llm": 0.0,
            "rag": 0.0,
        }
        ast_data: Dict[str, Any] = {}

        if mode == "pipeline":
            parse_start = time.perf_counter()
            ast_data = self.parser.parse_code(content, filename)
            stage_latencies["ast_parse"] = round((time.perf_counter() - parse_start) * 1000, 4)

            if "error" in ast_data:
                validation_time_ms = round((time.perf_counter() - total_start) * 1000, 4)
                result = {
                    "is_violation": True,
                    "violations": [
                        {
                            "type": "SyntaxError",
                            "message": ast_data["error"],
                            "suggestion": "Fix Python syntax.",
                        }
                    ],
                    "mode": mode,
                    "metrics": {
                        "validation_time_ms": validation_time_ms,
                        "file_size_chars": file_size_chars,
                        "file_loc": file_loc,
                        "code_file_tokens": code_file_tokens,
                        "stage_latencies_ms": {**stage_latencies, "total": validation_time_ms},
                        "provider": getattr(self.llm_client.provider, "provider_name", None),
                        "model": self.llm_client.model_name,
                        "llm_input_tokens": 0,
                        "llm_output_tokens": 0,
                        "llm_total_tokens": 0,
                        "cached_tokens": 0,
                        "cost_usd": 0.0,
                        "api_calls": 0,
                        "parseable_outputs": 0,
                        "unparseable_outputs": 0,
                    },
                }
                self._record_validation(
                    filename=filename,
                    file_size_chars=file_size_chars,
                    code_file_tokens=code_file_tokens,
                    result=result,
                    mode=mode,
                    run_metadata=run_metadata,
                )
                return result

            deterministic_start = time.perf_counter()
            deterministic = self.llm_client.rule_based_name_violations(ast_data, domain_rules)
            stage_latencies["deterministic_rules"] = round(
                (time.perf_counter() - deterministic_start) * 1000,
                4,
            )

            advanced = {"is_violation": False, "violations": []}
            if needs_llm_advanced_checks(ast_data):
                advanced_start = time.perf_counter()
                advanced = self.llm_client.analyze_advanced_violations(ast_data, domain_rules)
                stage_latencies["advanced_llm"] = round(
                    (time.perf_counter() - advanced_start) * 1000,
                    4,
                )

            merged_violations = merge_violations(
                deterministic,
                advanced.get("violations", []),
            )
            result = {
                "is_violation": bool(merged_violations),
                "violations": merged_violations,
                "mode": mode,
            }
        else:
            naive_start = time.perf_counter()
            result = self.llm_client.analyze_naive_violations(filename, content)
            stage_latencies["naive_llm"] = round((time.perf_counter() - naive_start) * 1000, 4)
            result["mode"] = mode

        retrieval_top1_hit = None
        retrieval_top3_hit = None
        has_sources = False
        if mode == "pipeline" and result.get("is_violation") and rag is not None:
            rag_start = time.perf_counter()
            rag_cache: Dict[str, List[Dict[str, Any]]] = {}
            for violation in result.get("violations", []):
                key = rag_cache_key(violation)
                if key not in rag_cache:
                    sources = rag.retrieve_source(
                        violation_type=violation.get("type", ""),
                        violation_message=violation.get("message", ""),
                    )
                    rag_cache[key] = sources
                else:
                    sources = rag_cache[key]
                violation["sources"] = sources
                if sources:
                    has_sources = True

                expected_sections = (expected_retrieval_sections or {}).get(key, [])
                observed_sections = [source.get("section", "") for source in sources]
                if expected_sections:
                    retrieval_top1_hit = bool(observed_sections[:1] and observed_sections[0] in expected_sections)
                    retrieval_top3_hit = any(section in expected_sections for section in observed_sections[:3])

                self.research_metrics.record_retrieval_event(
                    {
                        "filename": filename,
                        "violation_type": violation.get("type", ""),
                        "violation_message": violation.get("message", ""),
                        "latency_ms": 0.0,
                        "expected_sections": expected_sections,
                        "observed_sections": observed_sections[:3],
                        "top1_hit": retrieval_top1_hit,
                        "top3_hit": retrieval_top3_hit,
                    }
                )

            stage_latencies["rag"] = round((time.perf_counter() - rag_start) * 1000, 4)

        token_delta = self.token_tracker.delta(token_snapshot)
        validation_time_ms = round((time.perf_counter() - total_start) * 1000, 4)
        stage_latencies["total"] = validation_time_ms
        result["metrics"] = {
            "validation_time_ms": validation_time_ms,
            "file_size_chars": file_size_chars,
            "file_loc": file_loc,
            "code_file_tokens": code_file_tokens,
            "stage_latencies_ms": stage_latencies,
            "provider": getattr(self.llm_client.provider, "provider_name", None),
            "model": self.llm_client.model_name,
            **token_delta,
        }

        self._record_validation(
            filename=filename,
            file_size_chars=file_size_chars,
            code_file_tokens=code_file_tokens,
            result=result,
            mode=mode,
            has_sources=has_sources,
            retrieval_top1_hit=retrieval_top1_hit,
            retrieval_top3_hit=retrieval_top3_hit,
            run_metadata=run_metadata,
        )
        return result

    def _record_validation(
        self,
        *,
        filename: str,
        file_size_chars: int,
        code_file_tokens: int,
        result: Dict[str, Any],
        mode: ValidationMode,
        has_sources: bool = False,
        retrieval_top1_hit: Optional[bool] = None,
        retrieval_top3_hit: Optional[bool] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        metrics = result.get("metrics", {})
        self.validation_tracker.track_validation(
            filename=filename,
            file_size_chars=file_size_chars,
            code_file_tokens=code_file_tokens,
            validation_time_ms=float(metrics.get("validation_time_ms", 0.0)),
            violations=result.get("violations", []),
            has_sources=has_sources,
            mode=mode,
            provider=metrics.get("provider"),
            model=metrics.get("model"),
            stage_latencies_ms=metrics.get("stage_latencies_ms", {}),
            llm_input_tokens=int(metrics.get("llm_input_tokens", 0)),
            llm_output_tokens=int(metrics.get("llm_output_tokens", 0)),
            llm_total_tokens=int(metrics.get("llm_total_tokens", 0)),
            cached_tokens=int(metrics.get("cached_tokens", 0)),
            cost_usd=float(metrics.get("cost_usd", 0.0)),
            api_calls=int(metrics.get("api_calls", 0)),
            parseable_outputs=int(metrics.get("parseable_outputs", 0)),
            unparseable_outputs=int(metrics.get("unparseable_outputs", 0)),
            retrieval_top1_hit=retrieval_top1_hit,
            retrieval_top3_hit=retrieval_top3_hit,
        )
        self.research_metrics.record_validation_run(
            {
                "filename": filename,
                "mode": mode,
                "file_size_chars": file_size_chars,
                "code_file_tokens": code_file_tokens,
                "is_violation": result.get("is_violation", False),
                "violations": result.get("violations", []),
                "metrics": metrics,
                "run_metadata": run_metadata or {},
            }
        )
