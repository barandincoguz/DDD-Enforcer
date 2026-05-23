# WP-CORE-20 — EMSE-Grade Structured Logging & Run-Manifest

**Status:** SPEC v2 (post-Codex xhigh)
**Date:** 2026-05-23
**Author:** Claude Opus 4.7 (1M context) on behalf of Baran Dincoguz
**Finding closed:** F-9 (MINOR-OPEN: Zero structured logging anywhere)
**Scope label:** L (Large) — multi-file emitter rollout + new module + aggregation script
**Codex xhigh review:** APPLIED (5C + 8W + 1OQ inline)

---

## 0. Codex review applied — change log spec v1 → v2

| Codex tag | Inline fix in spec v2 |
|---|---|
| **C-1** ContextVar doesn't propagate to ThreadPoolExecutor | §6.5 + §9: explicit `contextvars.copy_context()` wrapping for parallel-Scout workers. New test T-EMITTER-PARALLEL-1. |
| **C-2** TokenTracker delta-snapshot fails under concurrency | §6.4 + §9 + R-2: **delta approach dropped**. Manifest LLM aggregates built **exclusively** from per-run emitter records. TokenTracker singleton untouched (continues legacy auto-export). |
| **C-3** Pre-pipeline failures missed | §5 outcome enum + §6.4: manifest **created at endpoint entry** (`main.py:_run_generate_pipeline`). New outcomes: `no_input_files`, `srs_parse_failed`, `all_srs_empty`, `output_write_failed`. |
| **C-4** `json_failed` not captured for chat-then-manual-parse paths | §6.5 + §8: emitter exposes `record_json_parse_failure(stage, reason)` for callsites that manually parse `chat()` content. Architect/Scout/Specialist callsites updated. |
| **C-5** Verifier check_id mismatch (`issue_type` vs `check_id`) | §8: new `_ISSUE_TYPE_TO_CHECK_ID` mapping table (single source of truth) used by the emitter; counts canonicalized to `D1..D8`/`S1` regardless of which check class produced them. |
| **W-1** Wrong instrumentation boundary | §6.5: instrumentation moves from `with_retry_and_rotation` to `LLMClient.chat`/`structured_output` **after** `LLMResponse` is constructed. Synthetic failed-call record on `RetryExhausted`. |
| **W-2** Finally write masks original exception | §6.4 + R-1: new `finalize_manifest_safely(manifest, original_exc)` helper with nested try/except. Acceptance T-MANIFEST-FINALIZE-1 tests both paths. |
| **W-3** Averaging averages | §6.3 + §10: aggregator computes **pooled** rates and percentiles from raw call records, not averages of per-run stats. Per-run distributions also kept separately for box plots. |
| **W-4** Aggregator mutates inputs | §6.3: aggregator **never modifies input manifests**; emits a separate redacted aggregate copy + records SHA-256 of each input manifest. |
| **W-5** No schema upgrade path | §5 + new §16 versioning policy: `schema_version` is SemVer; `min_supported_schema` constant in aggregator; migration functions live in `core/observability/migrations/`. |
| **W-6** Dataclass writer + Pydantic reader duplication | §6.1: **single Pydantic model** for write and read. `RunManifest` is a Pydantic `BaseModel`; writer calls `manifest.model_dump_json(indent=2)`. Drift impossible. |
| **W-7** Non-atomic write | §6.1: write via `path.with_suffix(".tmp")` + `os.fsync` + `os.replace` (POSIX-atomic rename). Aggregator ignores `*.tmp`. |
| **W-8** Self-measurement transparency | §10 + new fields: `instrumentation_overhead_ms`, `monotonic_clock_source`. Latency uses `time.monotonic_ns()` exclusively. Manifest write time excluded from `elapsed_ms`. |
| **OQ-1** `/validate` scope | §3: **explicitly out of scope**. WP-CORE-20 covers `/generate-model` and `/generate-model-stream` only. WP-CORE-22 handles `/validate`. |

---

## 1. Motivation

The DDD-Enforcer EMSE submission needs empirical claims backed by reproducible per-run
metrics. The current observability surface is split between two singletons
(`core/token_tracker.py` and `core/validation_metrics.py`), each auto-exporting its own
JSON file with different keys, different aggregation conventions, and no shared run-id.
Neither tracker captures:

1. **Latency percentiles** (only avg is reported; the paper claims p50/p95 in the methods section).
2. **`json_failed` rate** — already surfaced by `LLMResponse.json_failed` at the provider
   layer (`core/llm/base.py:37`) for `structured_output`, but **never** for `chat()` calls
   that the architect / scout / specialist manually parse via `_parse_json_response`
   (`core/architect.py:390+`). These manual parses are the production hot path.
3. **Stage failure events** — Architect retries / Specialist shape-errors / Refiner
   exhaustions / `ArchitectGroundingError` raises. These are logged via `print(...)` and
   `_log_*` helpers but are not machine-readable and not aggregated.
4. **D1–D8 verifier check counts per run** — checks emit `VerifierIssue` lists; deterministic
   checks use `issue_type` strings (`ungrounded_context`, `missing_evidence`, …) while
   semantic D6–D8 checks use `check_id` strings (`D6`, `D7`, `D8`). No unified counter
   bucket exists.
5. **Per-run correlation** — N=10 runs per D4 spec need a shared `run_id` so a downstream
   aggregator can compute mean / std / quartiles across runs.

Without a unified manifest, paper claims like "Refiner exhaustion rate is X% across N=10
runs" or "json_failed rate for `gemini-3.1-flash-lite` is Y%" cannot be defended from
git artifacts; each manifest must be self-describing and ship in the replication package.

## 2. In-scope outcomes

A single new module `core/observability/` exports:

- **`RunManifest`** Pydantic model — one per `/generate-model` (or `/generate-model-stream`)
  invocation, started at the **endpoint entry**, finalized in a `try/finally` block,
  written atomically to `runs/manifests/run-{ISO8601-ts}-{uuid4-prefix}.json`.
- **`StageEmitter` API** — context manager `with emitter.stage("architect") as s: ...`
  that records start/end monotonic timestamps, status, success/fail, and accepts
  child LLM call records and JSON parse failures.
- **`Aggregator` script** at `scripts/aggregate_runs.py` that consumes N run-manifests
  (glob `runs/manifests/run-*.json`) and emits a CSV + summary JSON with **pooled**
  rates and percentiles for the EMSE paper Methods section, plus per-run distributions
  for box plots.
- **No migration churn** — existing `TokenTracker`/`ValidationMetricsTracker` keep
  running side-by-side; manifest is built exclusively from emitter records, not from
  shared singleton state.

## 3. Out-of-scope (deferred)

- **Real-time log streaming** (e.g., to an OTLP collector). EMSE artifact is post-hoc.
- **Removal of `print(...)` calls.** Print lines remain as user-facing progress.
- **`/validate` endpoint manifest** — explicitly excluded (OQ-1). The existing
  `ValidationMetricsTracker` continues unchanged. WP-CORE-22 covers `/validate`.
- **VS Code extension surfacing of the manifest.** Backend-only.
- **Consolidation of legacy trackers.** Side-by-side operation is the v1 contract;
  WP-CORE-21 consolidates if desired.

## 4. Locked decisions

| ID | Decision | Rationale |
|---|---|---|
| WP-CORE-20.D1 | **stdlib `logging` + Pydantic v2 model**, not `structlog`/`loguru`. | Zero new dependencies; Pydantic v2 is already in `requirements.lock`. Custom JSONFormatter unnecessary because manifest export is via `model_dump_json`. |
| WP-CORE-20.D2 | **Run-manifest format = one JSON file per pipeline run**, atomically written to `runs/manifests/run-{ISO8601-ts}-{uuid4-prefix}.json`. | Filename convention mirrors `runs/probe-{ts}.json` + `domain_run-{ts}.json`. UUID4 prefix breaks ties on sub-millisecond runs. |
| WP-CORE-20.D3 | **Aggregator output = CSV (one row per run) + pooled-summary JSON + per-run distributions JSON**. | CSV is paper-table-ready. Pooled summary defends against averaging-averages (W-3). Per-run distributions enable box plots / outlier inspection. |
| WP-CORE-20.D4 | **`run_id` = `uuid4()` generated at pipeline start**, passed as an orchestration kwarg. | UUIDs prevent collision under parallel runs. |
| WP-CORE-20.D5 | **Latency raw values kept per call**; percentiles computed at manifest-finalize via `statistics.quantiles(method="inclusive")`. Per-call records also retained for cross-run pooling. | Storage cost is small; reviewer reproducibility requires raw values. |
| WP-CORE-20.D6 | **Existing `TokenTracker`/`ValidationMetricsTracker` are NOT modified.** Manifest sources data **exclusively** from per-run emitter records (Codex C-2 fix). | Removes concurrency-coupling risk. Singletons keep their legacy auto-exports for backward compat. |
| WP-CORE-20.D7 | **No 3rd-party tracing library.** | EMSE replication package must be runnable offline. |
| WP-CORE-20.D8 | **Manifest schema is SemVer**: append-only within MAJOR. Aggregator declares `min_supported_schema` and rejects below. Migration functions in `core/observability/migrations/` only for MAJOR. (Codex W-5 fix.) | Allows iterative refinement without breaking the N=10 replication corpus. |
| WP-CORE-20.D9 | **Manifest is a Pydantic v2 `BaseModel`**; export via `model_dump_json(indent=2, exclude_none=False)`. Single source of truth, no dataclass-Pydantic drift. (Codex W-6 fix.) | Pydantic v2 is already a transitive dep through FastAPI. Validates at write. |
| WP-CORE-20.D10 | **Atomic write**: write to `path.with_suffix(".tmp")`, `os.fsync(fd)`, `os.replace(tmp, final)`. Aggregator skips `*.tmp`. (Codex W-7 fix.) | POSIX-atomic rename guarantee. macOS + Linux + FastAPI worker dev environments all support. |
| WP-CORE-20.D11 | **Latency uses `time.monotonic_ns()` exclusively.** Manifest carries `monotonic_clock_source` field. Wall-clock timestamps (`datetime.now(timezone.utc)`) only used for `started_at`/`ended_at` reporting. (Codex W-8 fix.) | EMSE methodological clarity. |

## 5. RunManifest schema (v1.0)

```jsonc
{
  "schema_version": "1.0",
  "min_supported_schema": "1.0",
  "run_id": "<uuid4>",
  "started_at": "<ISO8601 UTC>",
  "ended_at": "<ISO8601 UTC>",
  "elapsed_ms": 0.0,                   // monotonic; excludes manifest write
  "instrumentation_overhead_ms": 0.0,  // self-measured; for transparency
  "monotonic_clock_source": "time.monotonic_ns",
  "outcome": "success"
            | "no_input_files"
            | "srs_parse_failed"
            | "all_srs_empty"
            | "architect_grounding_error"
            | "refinement_exhausted"
            | "synthesizer_empty_model"
            | "pipeline_error"
            | "output_write_failed"
            | "unexpected_error",

  "environment": { ... },              // same as v1, unchanged shape
  "request": { ... },                  // same as v1, unchanged shape

  "stages": {
    "scout": {
      "started_at": "...", "ended_at": "...", "elapsed_ms": 0.0,
      "status": "success" | "fail" | "skipped",
      "chunks_processed": 0,
      "sentences_extracted": 0,
      "llm_calls": [...],              // per-call records (raw, see §5.x)
      "json_parse_failures": [...],    // NEW v2: caller-side parse failures
      "p50_latency_ms": 0.0, "p95_latency_ms": 0.0
    },
    "architect": { ... },              // same shape; same fields
    "specialist": { ... },
    "verifier": { ... },               // see §8 for issue_counts_by_check shape
    "refiner": { ... },
    "synthesizer": { ... }
  },

  "llm": {
    "total_calls": 0,
    "total_tokens": { "prompt": 0, "completion": 0, "cached": 0, "billable_prompt": 0, "total": 0 },
    "total_cost_usd": 0.0,
    "json_failed_count": 0,            // sum of structured_output failures
    "json_parse_failure_count": 0,     // NEW v2: caller-side _parse_json_response failures
    "json_failed_total_count": 0,      // sum of both above; the EMSE paper metric
    "json_failed_rate": 0.0,           // json_failed_total_count / total_calls
    "json_fail_reasons": { "invalid_json": 0, "schema_mismatch": 0, "empty_response": 0, "caller_parse": 0 },
    "retry_exhausted_count": 0,        // synthetic record for RetryExhausted

    "by_model": { "<model_id>": { ... } },
    "by_stage": { "<stage>": { ... } }
  },

  "domain_model_summary": { ... },     // populated on success only

  "errors": [
    {
      "timestamp": "<ISO8601 UTC>",
      "type": "ArchitectGroundingError" | ...,
      "stage": "architect" | ...,
      "message": "...",
      "srs_path": "<optional>",
      "context": { ... }
    }
  ]
}
```

### Per-call LLM record (`stages.<stage>.llm_calls[]`)

```jsonc
{
  "timestamp": "<ISO8601 UTC>",
  "stage": "architect",
  "operation": "identify_contexts attempt-1",
  "model_id": "gemini-3.1-pro-preview",
  "provider": "gemini",
  "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
  "cost_usd": 0.0,
  "latency_ms": 0.0,                   // time.monotonic_ns based
  "json_failed": false,                // from LLMResponse.json_failed (structured_output)
  "json_fail_reason": null,            // from LLMResponse.json_fail_reason
  "is_retry_exhausted": false          // synthetic record marker
}
```

### Per-stage JSON parse failure (`stages.<stage>.json_parse_failures[]`)

NEW v2 — catches Codex C-4 (chat-then-manual-parse hot path).

```jsonc
{
  "timestamp": "<ISO8601 UTC>",
  "stage": "scout",
  "operation": "extract_sentences_chunk_3 retry-2",
  "model_id": "gemini-3.1-flash-lite",
  "reason": "json_parse_failed" | "schema_mismatch" | "truncated_response"
}
```

## 6. API surface (post-Codex)

### 6.1 `core/observability/run_manifest.py`

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional
import uuid
from datetime import datetime, timezone

OutcomeLiteral = Literal[
    "in_progress", "success",
    "no_input_files", "srs_parse_failed", "all_srs_empty",
    "architect_grounding_error", "refinement_exhausted",
    "synthesizer_empty_model", "pipeline_error",
    "output_write_failed", "unexpected_error",
]


class StageRecord(BaseModel):
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_ms: float = 0.0
    status: Literal["success", "fail", "skipped", "partial_degrade", "clean", "issues_found", "exhausted", "empty_model"] = "success"
    llm_calls: List["LLMCallRecord"] = Field(default_factory=list)
    json_parse_failures: List["JSONParseFailureRecord"] = Field(default_factory=list)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)  # extension dict


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
    llm: "LLMAggregate" = Field(default_factory=lambda: LLMAggregate())
    domain_model_summary: Dict[str, int] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


def write_manifest_atomic(manifest: RunManifest, path: Path) -> None:
    """W-7 fix: atomic write via tmp + fsync + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    payload = manifest.model_dump_json(indent=2, exclude_none=False)
    with open(tmp, "w") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
```

### 6.2 `core/observability/emitter.py`

```python
from contextlib import contextmanager
from contextvars import ContextVar
import time

_emitter_var: ContextVar[Optional["StageEmitter"]] = ContextVar("emitter", default=None)
_stage_var: ContextVar[Optional[str]] = ContextVar("stage", default=None)


class StageEmitter:
    """Per-pipeline-run emitter. NOT a singleton — one instance per run."""

    def __init__(self, manifest: RunManifest) -> None:
        self.manifest = manifest
        self._lock = threading.Lock()

    @contextmanager
    def stage(self, name: str) -> Iterator[StageRecord]:
        record = StageRecord()
        record.started_at = datetime.now(timezone.utc).isoformat()
        start_ns = time.monotonic_ns()
        token_emitter = _emitter_var.set(self)
        token_stage = _stage_var.set(name)
        try:
            yield record
            if record.status == "success":  # caller may override (e.g. partial_degrade)
                record.status = "success"
        except Exception as exc:
            record.status = "fail"
            with self._lock:
                self.manifest.errors.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "type": type(exc).__name__,
                    "stage": name,
                    "message": str(exc),
                    "srs_path": getattr(exc, "srs_path", None),
                    "context": _safe_context(exc),
                })
            raise
        finally:
            record.ended_at = datetime.now(timezone.utc).isoformat()
            record.elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
            record.p50_latency_ms, record.p95_latency_ms = _percentiles(
                [c.latency_ms for c in record.llm_calls]
            )
            with self._lock:
                self.manifest.stages[name] = record
            _stage_var.reset(token_stage)
            _emitter_var.reset(token_emitter)

    def record_llm_call(self, response: "LLMResponse", operation: str) -> None:
        """Append per-call record to current stage. Bumps llm.* aggregates."""
        stage_name = _stage_var.get()
        if stage_name is None:
            return  # call outside any stage; silently drop (e.g., during init)
        record = LLMCallRecord.from_response(response, stage=stage_name, operation=operation)
        stage_rec = self.manifest.stages.setdefault(stage_name, StageRecord())
        with self._lock:
            stage_rec.llm_calls.append(record)
            self._bump_llm_aggregates(record)

    def record_json_parse_failure(self, operation: str, model_id: str, reason: str) -> None:
        """C-4 fix: caller-side parse failures (Architect/Scout/Specialist manual parse path)."""
        stage_name = _stage_var.get()
        if stage_name is None:
            return
        rec = JSONParseFailureRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage_name, operation=operation, model_id=model_id, reason=reason,
        )
        stage_rec = self.manifest.stages.setdefault(stage_name, StageRecord())
        with self._lock:
            stage_rec.json_parse_failures.append(rec)
            self.manifest.llm.json_parse_failure_count += 1
            self.manifest.llm.json_fail_reasons["caller_parse"] = \
                self.manifest.llm.json_fail_reasons.get("caller_parse", 0) + 1


def get_current_emitter() -> Optional[StageEmitter]:
    return _emitter_var.get()
```

### 6.3 Aggregator `scripts/aggregate_runs.py` (W-3 + W-4 applied)

```
$ python scripts/aggregate_runs.py runs/manifests/run-*.json \
    --out-csv runs/aggregates/n10-per-run.csv \
    --out-pooled runs/aggregates/n10-pooled.json \
    --out-distributions runs/aggregates/n10-distributions.json
```

Outputs:

1. **`n10-per-run.csv`** — one row per run with raw per-run scalars + run-hash. No aggregation.
2. **`n10-pooled.json`** — pooled rates from raw counts:
   - `pooled_json_failed_rate = sum(json_failed_total_count) / sum(total_calls)`
   - `pooled_latency_p50_ms` per stage = `statistics.quantiles(all_call_latencies_across_runs, n=100)[49]`
   - `pooled_latency_p95_ms` per stage = same with index 94
   - `pooled_total_cost_usd_per_run = mean/std/median/p25/p75 of total_cost_usd across runs`
3. **`n10-distributions.json`** — per-metric arrays (one array per metric, one element per run) for box plots.

**No input mutation.** Aggregator reads manifests, computes outputs, writes to separate aggregate dir. Each output references input manifests by SHA-256.

### 6.4 Wiring into pipeline (C-3 applied)

Manifest is created **at endpoint entry**, not inside `DomainArchitect.analyze_document`:

```python
# main.py: /generate-model and /generate-model-stream both call this
def _run_generate_pipeline(file_paths: List[str], srs_dir_resolved: str) -> Dict[str, Any]:
    manifest = RunManifest()
    _populate_environment(manifest)
    _populate_request_pre_parse(manifest, file_paths)
    original_exc: Optional[BaseException] = None
    emitter = StageEmitter(manifest)
    final_model: Optional[DomainModel] = None
    start_ns = time.monotonic_ns()

    try:
        # NEW v2: gate before parsing (Codex C-3)
        if not file_paths:
            manifest.outcome = "no_input_files"
            return {"success": False, "error": "no input files"}

        with emitter.stage("ingestion") as ing:
            parser = SRSDocumentParser()
            combined_text, srs_docs, err = _parse_srs_batch(file_paths, parser)
            if err is not None:
                if "empty after parsing" in err["error"]:
                    manifest.outcome = "all_srs_empty"
                else:
                    manifest.outcome = "srs_parse_failed"
                return err
            ing.metrics.update({
                "documents_parsed": len(srs_docs),
                "total_chars": len(combined_text),
            })

        _populate_request_post_parse(manifest, combined_text)
        architect = DomainArchitect(emitter=emitter)
        final_model = architect.analyze_document(combined_text, srs_path=file_paths[0])
        manifest.outcome = "success"
        _populate_domain_summary(manifest, final_model)
        return {"success": True, "model": final_model.model_dump()}

    except ArchitectGroundingError as exc:
        manifest.outcome = "architect_grounding_error"; original_exc = exc; raise
    except RefinementExhaustedError as exc:
        manifest.outcome = "refinement_exhausted"; original_exc = exc; raise
    except SynthesizerEmptyModelError as exc:
        manifest.outcome = "synthesizer_empty_model"; original_exc = exc; raise
    except PipelineError as exc:
        manifest.outcome = "pipeline_error"; original_exc = exc; raise
    except Exception as exc:
        manifest.outcome = "unexpected_error"; original_exc = exc; raise
    finally:
        manifest.ended_at = datetime.now(timezone.utc).isoformat()
        manifest.elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
        _finalize_manifest_safely(manifest, original_exc)


def _finalize_manifest_safely(manifest: RunManifest, original_exc: Optional[BaseException]) -> None:
    """W-2 fix: never mask original exception. Manifest write failure is a logged
    side-channel event, not a raise."""
    try:
        write_manifest_atomic(manifest, _manifest_path(manifest.run_id))
    except Exception as write_exc:
        # Log to stderr; never re-raise. If we masked original_exc here,
        # FastAPI would lose the typed PipelineError that the response layer
        # uses to construct typed JSON responses (WP-CORE-8 contract).
        sys.stderr.write(
            f"[observability] manifest_write_failed: {type(write_exc).__name__}: {write_exc}\n"
        )
```

### 6.5 LLM call interception (W-1 + C-4 applied)

**Instrumentation point: `LLMClient.chat`/`structured_output`, AFTER `LLMResponse` is constructed.** NOT the retry decorator (the decorator doesn't see `LLMResponse`).

Add to `core/llm/gemini.py:GeminiClient.chat` and `.structured_output` (and symmetrically to `ollama.py`):

```python
# at the end of each method, just before `return LLMResponse(...)`:
emitter = get_current_emitter()
if emitter is not None:
    operation = kwargs.get("operation", "<unknown>")  # NEW kwarg, optional
    emitter.record_llm_call(llm_response, operation=operation)
return llm_response
```

**Manual-parse callsites** (`core/architect.py`, `core/scout/` if exists, etc.) gain:

```python
result = self._parse_json_response(self._safe_response_text(response))
if isinstance(result, dict) and result.get("error") == "json_parse_failed":
    emitter = get_current_emitter()
    if emitter is not None:
        emitter.record_json_parse_failure(
            operation=f"extract_sentences_chunk_{chunk_num} retry-{retry+1}",
            model_id=resolved_model_id,
            reason="json_parse_failed",
        )
    ...
```

**RetryExhausted synthetic record** — `core/llm/retry.py:with_retry_and_rotation` gets a final `except RetryExhaustedError` that calls a helper:

```python
emitter = get_current_emitter()
if emitter is not None:
    emitter._record_retry_exhausted(
        operation=operation, model_id=model_id, last_response=last_response,
    )
```

### 6.6 Parallel-Scout contextvars (C-1 applied)

`core/architect.py:analyze_document` parallel branch:

```python
# BEFORE (broken: ContextVar doesn't propagate to ThreadPoolExecutor workers)
with ThreadPoolExecutor(max_workers=self.scout_max_workers) as ex:
    chunk_results = list(ex.map(
        lambda a: self._extract_sentences_from_chunk(*a), args,
    ))

# AFTER (C-1 fix: copy_context + run)
import contextvars
def _run_in_context(args):
    ctx = contextvars.copy_context()
    return ctx.run(self._extract_sentences_from_chunk, *args)

with ThreadPoolExecutor(max_workers=self.scout_max_workers) as ex:
    chunk_results = list(ex.map(_run_in_context, args))
```

New test `T-EMITTER-PARALLEL-1` runs Scout with `DDD_SCOUT_MAX_WORKERS=2` and asserts
that `manifest.stages["scout"].llm_calls` has records from both threads with correct
stage attribution.

## 7. Existing artifacts: migration policy

| Existing | New behavior |
|---|---|
| `extension/backend/validation_metrics_report.json` | UNCHANGED. `ValidationMetricsTracker` auto-export persists. |
| `extension/backend/token_usage.json` (if exported) | UNCHANGED. |
| `extension/backend/core/intermediate/*.json` | UNCHANGED. Per-stage debug dumps. |
| `extension/backend/runs/probe-*.json` + `.manifest.json` | UNCHANGED. `schema_probe` predates this WP. |
| `extension/backend/runs/manifests/run-*.json` | **NEW**. One per `/generate-model`(-stream) invocation. |

## 8. Verifier check count integration (C-5 applied)

`core/observability/_verifier_mapping.py`:

```python
# Single source of truth for issue_type → D-code canonical mapping.
# Built from grepping core/verifier/checks_*.py at WP-CORE-20 implementation time.
_ISSUE_TYPE_TO_CHECK_ID = {
    "ungrounded_context": "D1",
    "missing_evidence": "D2",
    "duplicate_entity_across_contexts": "D3",
    "invalid_aggregate_member": "D4",
    "unknown_dependency": "D5",
    # D6/D7/D8 already use contract.VerifierIssue.check_id directly:
    "D6": "D6", "D7": "D7", "D8": "D8",
    "evidence_indices_out_of_range": "S1",
    "semantic_ungrounded": "S1",
}

def canonical_check_id(issue: Any) -> str:
    """Resolve a VerifierIssue (legacy dataclass OR contract Pydantic) to D1..D8/S1."""
    if hasattr(issue, "check_id") and issue.check_id in ("D6", "D7", "D8"):
        return issue.check_id
    issue_type = getattr(issue, "issue_type", None) or getattr(issue, "check_id", None) or ""
    return _ISSUE_TYPE_TO_CHECK_ID.get(issue_type, "unknown")
```

`StageEmitter.record_verifier_result(result: VerifierResult)` walks the issues and
buckets by `canonical_check_id`, writing to `manifest.stages.verifier.metrics["issue_counts_by_check"]`.

**Acceptance test T-VERIFIER-COUNTS-1**: invents one `VerifierIssue` per known
`issue_type` + per `D6/D7/D8` `check_id`, asserts each lands in the correct bucket.

## 9. Threading & concurrency (C-1 + C-2 applied)

- `RunManifest` is **per-invocation**, never singleton. No cross-run contamination.
- `StageEmitter` instance is passed via `ContextVar`. For threads spawned by
  `ThreadPoolExecutor`, callers MUST use `contextvars.copy_context().run(...)` —
  this is wired into `architect.py` parallel-Scout branch (§6.6) and documented at the
  emitter API. **Test T-EMITTER-PARALLEL-1 enforces it.**
- All mutable state on `StageEmitter` is guarded by `self._lock` (a `threading.Lock`).
  Per-run-local, so contention is bounded by the parallel-Scout worker count.
- `TokenTracker` singleton state is **never consulted** by the manifest. The manifest
  is built exclusively from emitter records, eliminating C-2's concurrent-pipeline race.
- For overlapping pipeline invocations (a future possibility under FastAPI worker concurrency),
  each invocation has its own `RunManifest` + `StageEmitter`, written to its own
  `run-{ts}-{uuid}.json`. No shared state.

## 10. EMSE paper alignment (W-8 applied)

This WP supports the following Methods-section claims:

| Claim | Manifest field |
|---|---|
| "Per-stage p50/p95 latency across N=10 runs (pooled)" | Aggregator pooled percentiles from `stages.<stage>.llm_calls[*].latency_ms` |
| "JSON conformance rate per D1 6-model registry" | `llm.json_failed_rate` per-model from `llm.by_model.<m>.json_failed_total_count / .calls` |
| "Refinement cycles needed before clean verification" | `stages.refiner.metrics.cycles_used` + aggregator mean/std |
| "Architect grounding hard-fail rate" | `outcome == 'architect_grounding_error'` count / N |
| "Cost per run / per stage / per model" | `llm.total_cost_usd` + `llm.by_stage` + `llm.by_model` |
| "Verifier check fire rate per rule" | `stages.verifier.metrics.issue_counts_by_check.D1..D8.S1` |
| "Specialist degrade rate" | `stages.specialist.metrics.degraded_context_count / contexts_processed` |
| "Instrumentation overhead" (transparency) | `instrumentation_overhead_ms` per run |
| "Clock source" (transparency) | `monotonic_clock_source` field |

## 11. Acceptance criteria (post-Codex)

1. **`core/observability/`** module exists with `run_manifest.py`, `emitter.py`, `_verifier_mapping.py`, `__init__.py`.
2. **One JSON manifest per `/generate-model` invocation** lands in `runs/manifests/run-{ts}-{uuid}.json` regardless of success or failure (try/finally guarantees write; W-7 atomic).
3. **Manifest passes Pydantic v2 round-trip** — `RunManifest.model_validate_json(manifest.model_dump_json()) == manifest`.
4. **All 6 stages + ingestion** appear in `manifest.stages` after a successful run.
5. **`json_failed` aggregation works** for both `structured_output` (provider-side) and `chat`-then-manual-parse (caller-side via `record_json_parse_failure`). C-4 closed.
6. **D1–D8/S1 check counters** populated via `canonical_check_id` mapping. C-5 closed.
7. **Aggregator script** at `scripts/aggregate_runs.py` produces three outputs (per-run CSV, pooled JSON, distributions JSON) without mutating input manifests. W-3 + W-4 closed.
8. **Pre-pipeline failures** (no_input_files, srs_parse_failed, all_srs_empty) produce a manifest with the matching `outcome`. C-3 closed.
9. **Parallel-Scout test** (`T-EMITTER-PARALLEL-1`) asserts emitter visibility under `DDD_SCOUT_MAX_WORKERS=2`. C-1 closed.
10. **Finalize-safely test** (`T-MANIFEST-FINALIZE-1`) asserts that a forced write failure in `finally` does NOT mask the original raised exception. W-2 closed.
11. **Pyright strict** passes on the new module.
12. **`pytest -m "not integration"`** baseline must reach ≥ **412** (current 404 + 8 new tests minimum).
13. **No new dependency** in `requirements.lock`. stdlib + Pydantic v2 (already pinned).
14. **`monotonic_clock_source` field** populated, `instrumentation_overhead_ms` measurable. W-8 closed.
15. **Atomic write verified** by `T-ATOMIC-WRITE-1` — assert no `*.tmp` left behind on success; aggregator skips `*.tmp` deliberately.

## 12. Risks (post-Codex)

| ID | Risk | Mitigation |
|---|---|---|
| R-1 | Manifest write fails inside `finally` block during error path → masks original exception. | **Fixed by W-2**: `_finalize_manifest_safely(manifest, original_exc)` logs write failure to stderr; never raises from finally. |
| R-2 | ~~`TokenTracker` singleton state pollutes manifest with prior-run calls.~~ | **Removed by C-2**: manifest no longer reads from `TokenTracker`. Singleton untouched; manifest sources data exclusively from per-run emitter records. |
| R-3 | Aggregator chokes on a partial/corrupt manifest. | Aggregator skips `*.tmp` (atomic-write guarantee) and any manifest whose `schema_version < min_supported_schema`; emits a `corrupt_manifest` row in CSV with input SHA-256. |
| R-4 | UUID4 collision over N=10 runs is essentially zero. Filename collision via timestamp can happen for sub-millisecond runs. | Filename = `run-{iso8601_ts}-{run_id[:8]}.json`. UUID prefix breaks ties. |
| R-5 | Existing `DomainArchitect.analyze_document` called outside HTTP context (CLI, tests). Manifest may not be created. | `analyze_document` accepts an `emitter: Optional[StageEmitter] = None` kwarg. When None, no manifest is written; existing behavior preserved. Callers wanting a manifest construct one explicitly. |
| R-6 | Threadlocal `_emitter_var` leaks between requests on a FastAPI worker. | Always set/reset via `ContextVar.set(...)` + `ContextVar.reset(token)` inside `emitter.stage()` `finally`. ContextVars are request-scoped under FastAPI by default. |
| R-7 | EMSE reviewers may flag "self-reported metrics" as a threat to validity. | Out of scope for code; `docs/threats_to_validity_notes.md` already discusses self-measurement. WP-CORE-20 contributes `instrumentation_overhead_ms` + `monotonic_clock_source` for transparency. |
| R-8 | `model_dump_json` may fail on non-serializable nested objects in `errors[*].context`. | `_safe_context()` helper coerces to `{k: repr(v)}` if necessary. Test T-ERROR-CONTEXT-1 covers a non-serializable `Exception` payload. |
| R-9 | Parallel-Scout context propagation forgotten in a future refactor. | `T-EMITTER-PARALLEL-1` is a regression guard; lint rule unnecessary. |

## 13. Open questions resolved by Codex

All v1 OQs resolved:
- **OQ-1 (TokenTracker delta)** — REPLACED by Codex C-2 decision: emitter-only sourcing.
- **OQ-2 (Pydantic reader + dataclass writer)** — REPLACED by Codex W-6: single Pydantic model.
- **OQ-3 (skipped stage)** — RESOLVED: `status: "skipped"` with zeroed metrics preserves aggregator stability.
- **OQ-4 (strip per-call records)** — REPLACED by Codex W-4: never strip, emit separate aggregate copy.
- **OQ-5 (errors context typing)** — RESOLVED: `Dict[str, Any]` + `_safe_context()` coercion; not strict-mode hostile.
- **OQ-6 (missing EMSE metrics)** — Tracked: instrumentation overhead + monotonic clock added (W-8). Other metrics (e.g., rate-limit backoff total duration) accepted as follow-up if reviewers request.

## 14. Implementation order (TDD)

1. **RED** — `tests/test_observability_run_manifest.py`: T-MANIFEST-1..N (construct, round-trip, every `outcome` literal, errors append, finalize idempotency, instrumentation_overhead_ms positive).
2. **RED** — `tests/test_observability_emitter.py`: T-EMITTER-1..N (stage lifecycle, exception path → `status=fail`, `record_llm_call` aggregation, `record_json_parse_failure`, contextvar set/reset, `T-EMITTER-PARALLEL-1`).
3. **RED** — `tests/test_observability_verifier_counts.py`: T-VERIFIER-COUNTS-1 (`_ISSUE_TYPE_TO_CHECK_ID` covers every check; D6/D7/D8 path via `check_id`).
4. **RED** — `tests/test_observability_aggregator.py`: T-AGG-1..N (3-manifest fixture → CSV + pooled + distributions; no input mutation; SHA-256 fingerprints; pooled rate ≠ averaged rate).
5. **RED** — `tests/test_observability_atomic_write.py`: T-ATOMIC-WRITE-1 (no `*.tmp` left; fsync called; aggregator skips `*.tmp`).
6. **RED** — `tests/test_pipeline_observability_e2e.py`: full pipeline run lands a valid manifest on disk; success outcome.
7. **RED** — `tests/test_pipeline_observability_failures.py`: `no_input_files`, `srs_parse_failed`, `all_srs_empty`, `architect_grounding_error`, `refinement_exhausted` paths each produce a manifest with matching `outcome`. T-MANIFEST-FINALIZE-1 in this file.
8. **GREEN** — `core/observability/run_manifest.py`, `emitter.py`, `_verifier_mapping.py`, `__init__.py`.
9. **GREEN** — wire `core/llm/gemini.py` + `core/llm/ollama.py` (instrumentation at LLMResponse-construction point).
10. **GREEN** — wire `core/architect.py:analyze_document` parallel-Scout fix (§6.6) + manual-parse callsites (§6.5).
11. **GREEN** — wire `main.py:/generate-model{,-stream}` entry point (§6.4).
12. **GREEN** — `scripts/aggregate_runs.py`.
13. **DOC** — `development_docs/WP-CORE-20-emse-grade-logging.md` + `INDEX.md` row.
14. **PLANNING** — audit state + improvements backlog + decision log.

Atomic commits (target 6):
- `feat(observability): WP-CORE-20 RunManifest + StageEmitter Pydantic core`
- `feat(llm): WP-CORE-20 instrument LLMClient at LLMResponse construction`
- `feat(architect, scout): WP-CORE-20 parallel-Scout contextvar propagation + manual-parse failure record`
- `feat(main, orchestration): WP-CORE-20 endpoint-entry manifest creation + finalize-safely`
- `feat(scripts): WP-CORE-20 aggregate_runs.py with pooled + distributions outputs`
- `chore(artifacts): WP-CORE-20 dev_doc + audit state + INDEX update`

## 15. Out of scope but recommended follow-ups

- **WP-CORE-21**: Consolidate `TokenTracker` + `ValidationMetricsTracker` + `RunManifest` into a single source of truth (remove the legacy auto-exports).
- **WP-CORE-22**: Per-`/validate`-request manifest (one per code-file validation) with linkage to the upstream `/generate-model` manifest via shared `domain_model_id` field.
- **WP-CORE-23**: Real-time SSE manifest streaming during `/generate-model-stream`.
- **WP-CORE-24**: `scripts/manifest_diff.py` for A/B comparisons under the D1 6-model registry.

## 16. Schema versioning policy (W-5)

- **`schema_version` is SemVer** (`MAJOR.MINOR.PATCH`).
- **MINOR / PATCH** bumps are **additive only**: new optional fields, new outcomes appended to the literal. The aggregator MUST tolerate unknown new fields (Pydantic v2 default behavior with `extra="ignore"`).
- **MAJOR** bumps may rename, remove, or retype existing fields. Each MAJOR bump ships a migration function in `core/observability/migrations/v{N}_to_v{N+1}.py`.
- **`min_supported_schema`** in the aggregator is updated only when a migration is added.
- **Manifests pre-`min_supported_schema`** are skipped with a logged warning, not silently consumed.
- **Schema-version negotiation example**: paper replication N=10 corpus is locked at `schema_version=1.0`. If a reviewer re-runs the corpus on a future version that has reached `schema_version=2.0`, the aggregator runs `migrate_v1_to_v2()` on disk-loaded manifests before pooled computation.
