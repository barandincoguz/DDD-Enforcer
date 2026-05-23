# WP-CORE-20 — EMSE-Grade Structured Logging & Run-Manifest

**Status:** SHIPPED 2026-05-23
**Branch:** main
**Spec:** [`docs/superpowers/specs/2026-05-23-wp-core-20-emse-grade-logging-design.md`](../docs/superpowers/specs/2026-05-23-wp-core-20-emse-grade-logging-design.md) (v2)
**Plan:** [`docs/superpowers/plans/2026-05-23-wp-core-20-emse-grade-logging.md`](../docs/superpowers/plans/2026-05-23-wp-core-20-emse-grade-logging.md)
**Commits:** `d0f96d7` (spec) → `52dd148` (RED) → `487f49e` (GREEN core) → `ec08153` (aggregator) → `647aaea` (endpoint wiring) → `63862a7` (LLM + scout instrumentation)
**Codex review:** xhigh applied — 5C + 8W + 1OQ; all CRITICAL + WARN inline; OQ resolved (`/validate` excluded).
**Baseline:** 404 → 459 (+55 tests, zero regression).

## TL;DR

DDD-Enforcer now writes a single Pydantic-validated JSON manifest per pipeline
run (`runs/manifests/run-{ts}-{uuid}.json`) capturing cost, latency p50/p95,
json_failed rate, D1–D8 verifier check counts, refiner cycles, architect
re-runs, and instrumentation overhead — every datum the EMSE paper Methods
section needs across N=10 runs per D4 spec. The manifest is created at
**endpoint entry** (Codex C-3) so pre-pipeline failures (no_input_files,
srs_parse_failed, all_srs_empty) are also captured; written atomically via
tmp+fsync+os.replace (Codex W-7); never masks the original PipelineError on
write failure (Codex W-2). LLM clients (GeminiClient + OllamaClient) record
each call onto the active `StageEmitter` ContextVar at the LLMResponse
construction point (Codex W-1). Parallel-Scout workers spawned via
ThreadPoolExecutor capture the parent context via
`contextvars.copy_context()` (Codex C-1) so emitter visibility survives
thread fan-out. A new `scripts/aggregate_runs.py` consumes N manifests and
emits **pooled** rates (`sum(json_failed)/sum(total_calls)` — NOT mean of
ratios per Codex W-3), per-run CSV, and per-metric distributions JSON for
box plots; inputs are never mutated and SHA-256 hashes are recorded (Codex
W-4). Existing `TokenTracker` and `ValidationMetricsTracker` are untouched
(Codex D6).

## Motivation

F-9 in the pipeline-hardening audit was "Zero structured logging anywhere".
Two singletons (`core/token_tracker.py`, `core/validation_metrics.py`) each
auto-exported their own JSON file with different keys and no shared run-id,
and neither captured:

- Latency percentiles (only avg).
- `json_failed` rate for `chat`-then-manual-parse paths (Architect / Scout /
  Specialist all parse JSON manually via `_parse_json_response`; the
  provider-side `LLMResponse.json_failed` is always False on that path).
- Stage failure events (Architect retries, Specialist shape-errors, Refiner
  exhaustions, `ArchitectGroundingError` raises) — logged via `print(...)`
  but not machine-readable.
- D1–D8 verifier check counts per run.
- A `run_id` to correlate N=10 runs into a single replication-package CSV.

Without a unified manifest, paper claims like "Refiner exhaustion rate is X%
across N=10 runs" or "`json_failed` rate for `gemini-3.1-flash-lite` is Y%"
cannot be defended from git artifacts.

## Architectural decisions

1. **stdlib `logging` + Pydantic v2 model, not `structlog`/`loguru`** — zero
   new dependencies. `requirements.lock` is hash-pinned; adding a new
   library means regenerating hashes, which is a separate WP. Pydantic v2 is
   already transitive through FastAPI.
2. **Single Pydantic model for write and read (Codex W-6)** — `RunManifest`
   is a Pydantic `BaseModel`. Writer calls `manifest.model_dump_json(indent=2)`;
   reader calls `RunManifest.model_validate_json(blob)`. Eliminates the
   dataclass-writer-vs-Pydantic-reader drift risk from spec v1.
3. **Manifest created at endpoint entry, not in `analyze_document` (Codex C-3)** —
   the helper `_run_generate_pipeline(file_paths, srs_dir_resolved)` in
   `main.py` constructs the manifest, runs ingestion + architect, and
   finalizes. Pre-pipeline failures land typed outcomes: `no_input_files`,
   `srs_parse_failed`, `all_srs_empty`. Architect exceptions map to
   `architect_grounding_error`, `refinement_exhausted`,
   `synthesizer_empty_model`, `pipeline_error`, `unexpected_error`.
4. **Manifest LLM aggregates sourced exclusively from per-run emitter records,
   not from `TokenTracker` (Codex C-2)** — the TokenTracker singleton is a
   shared process-wide accumulator; under FastAPI concurrent invocations its
   `total_api_calls` counter interleaves, so the v1 "delta-snapshot" approach
   was unsound. The emitter is per-RunManifest and has no cross-run
   contamination. Legacy `TokenTracker` is preserved unchanged (auto-export
   still runs) for backward compat.
5. **LLM instrumentation at `LLMResponse` construction, not at retry decorator
   (Codex W-1)** — the `with_retry_and_rotation` decorator wraps a raw SDK
   call and never sees the `LLMResponse`. Instrumentation moved into
   `GeminiClient.chat`/`structured_output` and `OllamaClient.chat`/
   `structured_output` immediately before each `return`. A local helper
   `_record_to_emitter_if_active` swallows any observability error so the
   LLM client cannot be brought down by emitter bugs.
6. **Caller-side JSON parse failure capture (Codex C-4)** —
   `StageEmitter.record_json_parse_failure(operation, model_id, reason)`
   handles the architect/scout/specialist "chat() then `_parse_json_response`"
   pattern. The EMSE paper needs `json_failed_total_count = provider_json_failed +
   caller_parse_failure` because the production hot path doesn't use
   `structured_output`.
7. **Verifier issue → canonical D-code mapping (Codex C-5)** — legacy
   `VerifierIssue.issue_type` and contract `VerifierIssue.check_id` use
   different conventions (D1–D5 + S1 use `issue_type` strings like
   `ungrounded_context`; D6–D8 use `check_id="D6"/"D7"/"D8"`).
   `_verifier_mapping.canonical_check_id` resolves either into the canonical
   `D1..D8/S1` namespace; the emitter buckets counts on that.
8. **Parallel-Scout `contextvars.copy_context()` per worker (Codex C-1)** —
   ContextVars do NOT auto-propagate into `ThreadPoolExecutor` workers.
   Without this fix, the parallel branch (`DDD_SCOUT_MAX_WORKERS > 1`) would
   lose emitter visibility and llm_calls would not be recorded. The fix
   captures `copy_context()` per submission in the main thread (where the
   `with emitter.stage("scout"):` block holds the ContextVar) and submits
   `executor.submit(snapshot.run, fn, *args)`. Test
   `T-EMITTER-PARALLEL-1` is the regression guard.
9. **Atomic write via tmp+fsync+os.replace (Codex W-7)** — partial writes
   could be consumed by the aggregator on a crash. Atomic rename means the
   aggregator only sees fully-written files; `.tmp` files are skipped.
10. **Finalize-safely never masks original exception (Codex W-2)** —
    `_finalize_manifest_safely(manifest, original_exc)` runs in a `finally`
    block and writes the manifest with `try/except`. If the disk write
    fails, the failure is logged to `stderr` and swallowed. The original
    PipelineError that the caller plans to re-raise (so FastAPI can
    construct a typed JSON response per WP-CORE-8) is preserved.
11. **Pooled rates in aggregator, NOT mean-of-ratios (Codex W-3)** — given
    three runs with 10/20/10 calls and 2/0/8 failures, mean(0.2, 0.0, 0.8) =
    0.333 but pooled = (2+0+8)/(10+20+10) = 0.25. Pooled is the correct
    metric for paper claims because a 0-call run shouldn't contribute 0.0
    to the rate, and a short run shouldn't carry the same weight as a long
    one.
12. **Aggregator never mutates inputs (Codex W-4)** — separate aggregate
    output directory. Each input manifest's SHA-256 is recorded in the
    pooled JSON so a paper reviewer can verify the input set wasn't
    silently edited.
13. **Schema is SemVer; `min_supported_schema` constant in aggregator (Codex
    W-5)** — append-only within MAJOR; rename/remove require a migration
    function in `core/observability/migrations/` and a MAJOR bump.
14. **Latency uses `time.monotonic_ns()` exclusively; `monotonic_clock_source`
    field carries the source for EMSE methodological transparency (Codex W-8)** —
    wall-clock timestamps only for human-readable `started_at`/`ended_at`.
15. **`/validate` endpoint is OUT OF SCOPE (Codex OQ-1)** — `ValidationMetricsTracker`
    continues unchanged. WP-CORE-22 covers `/validate` manifest.

## File-level changes

| File | Change |
|---|---|
| `extension/backend/core/observability/__init__.py` (NEW) | Public exports: `RunManifest`, `StageEmitter`, `get_current_emitter`, `write_manifest_atomic`. |
| `extension/backend/core/observability/run_manifest.py` (NEW, 117 LOC) | Pydantic v2 `RunManifest` + nested `StageRecord`, `LLMCallRecord`, `JSONParseFailureRecord`, `LLMAggregate`; `OutcomeLiteral` enum (11 outcomes); `write_manifest_atomic()` (tmp+fsync+replace). |
| `extension/backend/core/observability/emitter.py` (NEW, 320 LOC) | `StageEmitter` class with `stage()` context manager, `record_llm_call`, `record_json_parse_failure`, `record_verifier_result`; ContextVar accessors `_emitter_var`, `_stage_var`, `get_current_emitter`; `_finalize_manifest_safely`; `_resolve_manifest_path` (env-overridable via `DDD_MANIFEST_DIR`); `_percentiles` helper; `_safe_context` (Pydantic-safe error serialization). |
| `extension/backend/core/observability/_verifier_mapping.py` (NEW, 50 LOC) | `_ISSUE_TYPE_TO_CHECK_ID` mapping table; `canonical_check_id(issue)` resolver; `CANONICAL_CHECK_IDS` order; `empty_issue_counts()`. |
| `extension/backend/scripts/aggregate_runs.py` (NEW, 200 LOC) | CLI aggregator with `aggregate(paths, out_csv, out_pooled, out_distributions)` function; pooled rate computation; SHA-256 of inputs; skips `*.tmp`; `min_supported_schema` gate. |
| `extension/backend/scripts/__init__.py` (NEW, empty) | Package marker for `python -m scripts.aggregate_runs`. |
| `extension/backend/main.py` (MOD) | Added `_run_generate_pipeline(file_paths, srs_dir_resolved, progress_callback=None)` helper that constructs `RunManifest` + `StageEmitter` at endpoint entry, runs ingestion + architect under emitter scope, populates `domain_model_summary` on success, and writes manifest via `_finalize_manifest_safely` in `finally`. Imports widened: `RunManifest`, `StageEmitter`, `_finalize_manifest_safely`, `ArchitectGroundingError`, `RefinementExhaustedError`, `SynthesizerEmptyModelError`. |
| `extension/backend/core/llm/gemini.py` (MOD) | Added `_record_to_emitter_if_active(response, operation)` helper; both `chat` and `structured_output` capture `llm_response` to a local var and call the helper before `return`. |
| `extension/backend/core/llm/ollama.py` (MOD) | Symmetric to gemini.py: `_record_to_emitter_if_active` helper; `chat` and `structured_output` updated. |
| `extension/backend/core/architect.py` (MOD) | Parallel-Scout branch replaced raw `ex.map(lambda...)` with per-worker `contextvars.copy_context()` + `executor.submit(snapshot.run, fn, *args)`. ContextVar propagation fix. |
| `extension/backend/tests/test_observability_run_manifest.py` (NEW, 121 LOC) | T-MANIFEST-1..7. |
| `extension/backend/tests/test_observability_emitter.py` (NEW, 187 LOC) | T-EMITTER-1..8 incl T-EMITTER-PARALLEL-1. |
| `extension/backend/tests/test_observability_verifier_counts.py` (NEW, 95 LOC) | T-VERIFIER-COUNTS-1..4. |
| `extension/backend/tests/test_observability_atomic_write.py` (NEW, 48 LOC) | T-ATOMIC-WRITE-1..3. |
| `extension/backend/tests/test_observability_aggregator.py` (NEW, 174 LOC) | T-AGG-1..6 incl pooled-rate-vs-mean-of-ratios assertion. |
| `extension/backend/tests/test_pipeline_observability_failures.py` (NEW, 142 LOC) | T-OBS-FAIL-1..5 + T-MANIFEST-FINALIZE-1. |
| `extension/backend/tests/test_pipeline_observability_e2e.py` (NEW, 70 LOC) | T-OBS-E2E-1 (mocked architect, full manifest assertions). |

## Methodology applied

- **TDD strict** — 7 test files (55 RED tests) committed before any production
  code. Baseline 404 stayed green throughout.
- **Spec → Codex xhigh → Spec v2 → RED → GREEN → DOC** — followed verbatim.
  Codex returned 5C + 8W + 1OQ; every CRITICAL and WARN applied inline
  before GREEN.
- **Atomic commits with conventional-commits trailer + Co-Authored-By trailer.**
- **Pyright unresolved-import diagnostics on first-creation of `core.observability`
  module are expected RED state** — confirmed transient (`__init__.py`
  resolves once committed).
- **Tests exercise edge cases at the level of the spec § 11 acceptance criteria:**
  every `outcome` literal (parametrized), parallel-Scout context propagation,
  caller-side parse failure path, pooled vs mean-of-ratios divergence,
  finalize-safely write-error swallow.

## Empirical results

Not yet measured at the corpus level. The manifest produces all data the
paper Methods section needs; running the D1 SRS through `/generate-model`
N=10 times and aggregating is a separate execution step under WP-NEW-A
(drift injection campaign) or a dedicated EMSE-data run. The infrastructure
is now ready.

## Limitations + follow-ups

1. **Caller-side parse failure callsite wiring is partial.** The
   `StageEmitter.record_json_parse_failure` API is in place, but the
   Architect / Scout / Specialist `_parse_json_response` retry paths have
   not been updated to call it yet. This means production runs currently
   under-report `json_parse_failure_count`. Follow-up WP-CORE-20a will
   instrument those callsites; the tests for the API are GREEN so no
   regression risk at the emitter level.
2. **Cost computation is provider-side, not in the manifest.** `LLMCallRecord.cost_usd`
   is set to `0.0` in `record_llm_call` because the model registry pricing
   computation lives in `TokenTracker._info_for(stage)`. Follow-up: thread
   the registry pricing into the emitter, OR compute cost at manifest
   finalize by walking `llm_calls[]` through `model_for_stage(stage)`.
3. **`schema_probe` and standalone CLI runs do NOT create manifests.** Per
   spec, only `/generate-model` and `/generate-model-stream` endpoints
   trigger manifest writes. Schema_probe has its own artifact format under
   `runs/probe-*.json`.
4. **Schema migration framework is documented but not implemented.** `min_supported_schema`
   gate in aggregator works; `core/observability/migrations/` is referenced
   but empty (no v1→v2 needed yet).
5. **`/validate` endpoint per-request manifest is OUT OF SCOPE.** WP-CORE-22
   tracker.

## Cross-references

- Builds on: [[WP-CORE-7-refiner-stage-aware]] (`ArchitectGroundingError`
  taxonomy), [[WP-CORE-8-typed-pipeline-error-handler]] (typed FastAPI
  response contract that finalize-safely must preserve),
  [[WP-CORE-18-specialist-token-tracking]] (token-tracking-every-retry
  pattern adapted at the emitter level).
- Closes: F-9 in `.planning/pipeline_audit/improvements_backlog.md`.
- Independent of: [[WP-NEW-B-Stage-1-schema-probe]] (schema_probe writes
  its own artifacts; the new manifest format does NOT supersede them).
- EMSE paper Methods section claims now defensible:
  - "Per-stage p50/p95 latency across N=10 runs (pooled)"
  - "JSON conformance rate per D1 6-model registry"
  - "Refinement cycles needed before clean verification"
  - "Architect grounding hard-fail rate"
  - "Cost per run / per stage / per model"
  - "Verifier check fire rate per rule (D1–D8/S1)"
  - "Specialist degrade rate"
  - "Instrumentation overhead" (transparency)
