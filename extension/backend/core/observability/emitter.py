"""StageEmitter — per-run observability emitter.

Public API:
- StageEmitter(manifest): NOT a singleton, one per pipeline invocation.
- with emitter.stage(name) as record: context-manager records start/end/elapsed.
- emitter.record_llm_call(response, operation): bumps stage llm_calls + global aggregates.
- emitter.record_json_parse_failure(operation, model_id, reason): caller-side parse failure (Codex C-4).
- emitter.record_verifier_result(result): buckets issues by canonical check_id (Codex C-5).
- get_current_emitter(): ContextVar-backed accessor for instrumentation points
  outside the orchestrator (LLMClient.chat / structured_output).
- _finalize_manifest_safely(manifest, original_exc): write-with-swallow (Codex W-2).

Thread-safety: per-run instance lock guards every mutation. For
ThreadPoolExecutor-style fan-out (parallel-Scout), callers MUST wrap workers
with `contextvars.copy_context().run(...)` so the emitter ContextVar
propagates into the worker thread (Codex C-1).
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, TYPE_CHECKING

from core.observability._verifier_mapping import (
    CANONICAL_CHECK_IDS,
    canonical_check_id,
    empty_issue_counts,
)
from core.observability.run_manifest import (
    JSONParseFailureRecord,
    LLMCallRecord,
    RunManifest,
    StageRecord,
    write_manifest_atomic,
)


if TYPE_CHECKING:
    from core.llm.base import LLMResponse


_emitter_var: ContextVar[Optional["StageEmitter"]] = ContextVar("emitter", default=None)
_stage_var: ContextVar[Optional[str]] = ContextVar("stage", default=None)


def get_current_emitter() -> Optional["StageEmitter"]:
    """Return the active StageEmitter for the current ContextVar scope, or None."""
    return _emitter_var.get()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_context(exc: BaseException) -> dict[str, Any]:
    """Best-effort serializable context dict for the error.

    Codex R-8: Pydantic model_dump_json may choke on non-serializable values
    nested in error context. Coerce to repr-strings if necessary so the
    manifest write does not raise.
    """
    out: dict[str, Any] = {}
    for attr in ("issues", "residual_issues", "cycles_attempted", "srs_path"):
        if hasattr(exc, attr):
            val = getattr(exc, attr)
            try:
                if isinstance(val, (list, dict, str, int, float, bool, type(None))):
                    out[attr] = val
                else:
                    out[attr] = repr(val)
            except Exception:
                out[attr] = "<unrepresentable>"
    return out


def _percentiles(values: list[float]) -> tuple[float, float]:
    """Return (p50, p95) with stdlib statistics.quantiles. Empty list → (0, 0)."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        v = values[0]
        return v, v
    # 100-quantile gives us indices [49]→p50, [94]→p95.
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return qs[49], qs[94]


class StageEmitter:
    """Per-pipeline-run emitter. One instance per RunManifest.

    NOT a singleton. NEVER pass between invocations.
    """

    def __init__(self, manifest: RunManifest) -> None:
        self.manifest = manifest
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        """Context manager that opens/closes a stage record.

        On normal exit: status stays at whatever the caller set (default
        'success'). On exception: status='fail' and the exception is appended
        to manifest.errors[] (with srs_path + safe context). The exception
        is always re-raised.
        """
        record = StageRecord(started_at=_now_iso(), status="success")
        # Register upfront so concurrent record_llm_call / record_json_parse_failure
        # mutate the SAME record object (not a setdefault-created stand-in that
        # we then overwrite in finally — that bug lost all per-call records).
        with self._lock:
            self.manifest.stages[name] = record
        start_ns = time.monotonic_ns()
        token_emitter = _emitter_var.set(self)
        token_stage = _stage_var.set(name)
        try:
            yield record
        except Exception as exc:
            record.status = "fail"
            with self._lock:
                self.manifest.errors.append({
                    "timestamp": _now_iso(),
                    "type": type(exc).__name__,
                    "stage": name,
                    "message": str(exc),
                    "srs_path": getattr(exc, "srs_path", None),
                    "context": _safe_context(exc),
                })
            raise
        finally:
            record.ended_at = _now_iso()
            record.elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
            latencies = [c.latency_ms for c in record.llm_calls]
            record.p50_latency_ms, record.p95_latency_ms = _percentiles(latencies)
            _stage_var.reset(token_stage)
            _emitter_var.reset(token_emitter)

    def record_llm_call(self, response: "LLMResponse", operation: str) -> None:
        """Append a per-call record + bump global llm aggregates."""
        stage_name = _stage_var.get()
        if stage_name is None:
            # Outside any stage (init / CLI tooling). Silently drop.
            return

        record = LLMCallRecord(
            timestamp=_now_iso(),
            stage=stage_name,
            operation=operation,
            model_id=response.model_id,
            provider=response.provider,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cached_tokens=response.usage.cached_tokens,
            cost_usd=0.0,  # spec §5: cost computed at manifest finalize via registry
            latency_ms=response.latency_ms,
            json_failed=response.json_failed,
            json_fail_reason=response.json_fail_reason,
            is_retry_exhausted=False,
        )

        with self._lock:
            stage_rec = self.manifest.stages.setdefault(stage_name, StageRecord())
            stage_rec.llm_calls.append(record)
            self._bump_llm_aggregates(record)

    def record_json_parse_failure(
        self, operation: str, model_id: str, reason: str
    ) -> None:
        """Caller-side _parse_json_response failure (Codex C-4).

        Architect / Scout / Specialist call `chat()` then manually parse the
        content. Provider-side `json_failed` is always False on that path;
        this method captures the caller-side parse failure so the EMSE paper's
        json_failed_rate is correct.
        """
        stage_name = _stage_var.get()
        if stage_name is None:
            return

        record = JSONParseFailureRecord(
            timestamp=_now_iso(),
            stage=stage_name,
            operation=operation,
            model_id=model_id,
            reason=reason,
        )

        with self._lock:
            stage_rec = self.manifest.stages.setdefault(stage_name, StageRecord())
            stage_rec.json_parse_failures.append(record)

            self.manifest.llm.json_parse_failure_count += 1
            self.manifest.llm.json_failed_total_count += 1
            self.manifest.llm.json_fail_reasons["caller_parse"] = (
                self.manifest.llm.json_fail_reasons.get("caller_parse", 0) + 1
            )
            self._recompute_json_failed_rate()

    def record_verifier_result(self, result: Any) -> None:
        """Bucket verifier issues by canonical check_id (Codex C-5).

        Writes counts to `manifest.stages.<current-stage>.metrics["issue_counts_by_check"]`.
        Active stage is whatever the surrounding `with emitter.stage(name)` is —
        typically "verifier".
        """
        stage_name = _stage_var.get()
        if stage_name is None:
            return

        counts = empty_issue_counts()
        unknown = 0
        for issue in getattr(result, "issues", []) or []:
            cid = canonical_check_id(issue)
            if cid in counts:
                counts[cid] += 1
            else:
                unknown += 1

        with self._lock:
            stage_rec = self.manifest.stages.setdefault(stage_name, StageRecord())
            stage_rec.metrics.setdefault("issue_counts_by_check", empty_issue_counts())
            for cid in CANONICAL_CHECK_IDS:
                stage_rec.metrics["issue_counts_by_check"][cid] += counts[cid]
            if unknown:
                stage_rec.metrics["issue_counts_by_check"]["unknown"] = (
                    stage_rec.metrics["issue_counts_by_check"].get("unknown", 0) + unknown
                )

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _bump_llm_aggregates(self, record: LLMCallRecord) -> None:
        """Lock is already held by caller."""
        agg = self.manifest.llm
        agg.total_calls += 1
        agg.total_tokens["prompt"] += record.prompt_tokens
        agg.total_tokens["completion"] += record.completion_tokens
        agg.total_tokens["cached"] += record.cached_tokens
        agg.total_tokens["billable_prompt"] += max(
            record.prompt_tokens - record.cached_tokens, 0
        )
        agg.total_tokens["total"] += record.prompt_tokens + record.completion_tokens
        agg.total_cost_usd += record.cost_usd

        if record.json_failed:
            agg.json_failed_count += 1
            agg.json_failed_total_count += 1
            reason_bucket = (record.json_fail_reason or "unknown").split(":", 1)[0]
            agg.json_fail_reasons[reason_bucket] = (
                agg.json_fail_reasons.get(reason_bucket, 0) + 1
            )

        self._recompute_json_failed_rate()

        m = agg.by_model.setdefault(record.model_id, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "billable_prompt": 0,
            "cost_usd": 0.0,
            "json_failed_count": 0,
            "json_failed_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        })
        m["calls"] += 1
        m["prompt_tokens"] += record.prompt_tokens
        m["completion_tokens"] += record.completion_tokens
        m["cached_tokens"] += record.cached_tokens
        m["billable_prompt"] += max(record.prompt_tokens - record.cached_tokens, 0)
        m["cost_usd"] += record.cost_usd
        if record.json_failed:
            m["json_failed_count"] += 1
        m["json_failed_rate"] = (
            m["json_failed_count"] / m["calls"] if m["calls"] else 0.0
        )

        s = agg.by_stage.setdefault(record.stage, {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "json_failed_count": 0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        })
        s["calls"] += 1
        s["prompt_tokens"] += record.prompt_tokens
        s["completion_tokens"] += record.completion_tokens
        s["cost_usd"] += record.cost_usd
        if record.json_failed:
            s["json_failed_count"] += 1

    def _recompute_json_failed_rate(self) -> None:
        agg = self.manifest.llm
        if agg.total_calls > 0:
            agg.json_failed_rate = agg.json_failed_total_count / agg.total_calls
        else:
            agg.json_failed_rate = 0.0


def _finalize_manifest_safely(
    manifest: RunManifest, original_exc: Optional[BaseException] = None
) -> None:
    """Write manifest to disk; swallow ALL write errors (Codex W-2).

    NEVER raise from this function. The caller may have an `original_exc` they
    plan to re-raise — masking it with a manifest-write OSError would lose the
    typed PipelineError that FastAPI's response-builder relies on
    (WP-CORE-8 contract).

    Manifest-write failure is logged to stderr as a side-channel event.
    """
    try:
        path = _resolve_manifest_path(manifest.run_id)
        write_manifest_atomic(manifest, path)
    except Exception as write_exc:
        try:
            sys.stderr.write(
                f"[observability] manifest_write_failed: "
                f"{type(write_exc).__name__}: {write_exc}\n"
            )
            sys.stderr.flush()
        except Exception:
            # Even stderr is gone. Silently give up rather than masking original_exc.
            pass


def _resolve_manifest_path(run_id: str) -> Path:
    """Build `runs/manifests/run-{iso8601}-{uuidprefix}.json` path.

    Env override `DDD_MANIFEST_DIR` lets tests redirect output.
    """
    import os
    base = os.environ.get("DDD_MANIFEST_DIR")
    if base:
        dir_ = Path(base)
    else:
        backend_dir = Path(__file__).resolve().parent.parent.parent
        dir_ = backend_dir / "runs" / "manifests"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = run_id[:8]
    return dir_ / f"run-{ts}-{short}.json"
