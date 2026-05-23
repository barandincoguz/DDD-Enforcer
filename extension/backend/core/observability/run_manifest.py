"""RunManifest Pydantic v2 model + atomic JSON writer.

Single source of truth for the per-pipeline-run observability record. The same
model is used for write (`model_dump_json`) and read (`model_validate_json`) so
schema drift between writer and reader is impossible (Codex W-6).

Atomic write uses tmp + fsync + os.replace (Codex W-7).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


OutcomeLiteral = Literal[
    "in_progress",
    "success",
    "no_input_files",
    "srs_parse_failed",
    "all_srs_empty",
    "architect_grounding_error",
    "refinement_exhausted",
    "synthesizer_empty_model",
    "pipeline_error",
    "output_write_failed",
    "unexpected_error",
]


StageStatusLiteral = Literal[
    "in_progress",
    "success",
    "fail",
    "skipped",
    "partial_degrade",
    "clean",
    "issues_found",
    "exhausted",
    "empty_model",
]


class LLMCallRecord(BaseModel):
    timestamp: str
    stage: str
    operation: str
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    json_failed: bool = False
    json_fail_reason: Optional[str] = None
    is_retry_exhausted: bool = False


class JSONParseFailureRecord(BaseModel):
    timestamp: str
    stage: str
    operation: str
    model_id: str
    reason: str


class StageRecord(BaseModel):
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_ms: float = 0.0
    status: StageStatusLiteral = "in_progress"
    llm_calls: List[LLMCallRecord] = Field(default_factory=list)
    json_parse_failures: List[JSONParseFailureRecord] = Field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)


class LLMAggregate(BaseModel):
    total_calls: int = 0
    total_tokens: Dict[str, int] = Field(default_factory=lambda: {
        "prompt": 0, "completion": 0, "cached": 0, "billable_prompt": 0, "total": 0,
    })
    total_cost_usd: float = 0.0
    json_failed_count: int = 0          # structured_output provider-side failures
    json_parse_failure_count: int = 0   # caller-side _parse_json_response failures
    json_failed_total_count: int = 0    # sum of both
    json_failed_rate: float = 0.0       # json_failed_total_count / total_calls
    json_fail_reasons: Dict[str, int] = Field(default_factory=dict)
    retry_exhausted_count: int = 0
    by_model: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    by_stage: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class RunManifest(BaseModel):
    schema_version: str = "1.0"
    min_supported_schema: str = "1.0"
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: Optional[str] = None
    elapsed_ms: float = 0.0
    instrumentation_overhead_ms: float = 0.0
    monotonic_clock_source: str = "time.monotonic_ns"
    outcome: OutcomeLiteral = "in_progress"
    environment: Dict[str, Any] = Field(default_factory=dict)
    request: Dict[str, Any] = Field(default_factory=dict)
    stages: Dict[str, StageRecord] = Field(default_factory=dict)
    llm: LLMAggregate = Field(default_factory=LLMAggregate)
    domain_model_summary: Dict[str, int] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


def write_manifest_atomic(manifest: RunManifest, path: Path) -> None:
    """Atomic write: tmp file + fsync + os.replace (POSIX-atomic rename).

    Codex W-7 fix. Aggregator ignores *.tmp; consumers only see fully-written
    files. Crash-safe: a partial write leaves the .tmp behind, never a corrupt
    target file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = manifest.model_dump_json(indent=2, exclude_none=False)
    with open(tmp, "w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
