# WP-CORE-20 — Implementation Plan

**Spec:** `2026-05-23-wp-core-20-emse-grade-logging-design.md` (v2)
**Date:** 2026-05-23
**Owner:** Baran (autonomous via Claude Opus 4.7)
**Status:** READY FOR RED

---

## Atomic commit sequence (target 6 commits)

### Commit 1 — `test(observability): WP-CORE-20 RED tests for RunManifest + Emitter + Aggregator`

**Files NEW:**
- `extension/backend/tests/test_observability_run_manifest.py`
- `extension/backend/tests/test_observability_emitter.py`
- `extension/backend/tests/test_observability_verifier_counts.py`
- `extension/backend/tests/test_observability_aggregator.py`
- `extension/backend/tests/test_observability_atomic_write.py`
- `extension/backend/tests/test_pipeline_observability_e2e.py`
- `extension/backend/tests/test_pipeline_observability_failures.py`

**Expected state:** RED — all tests fail at import (no `core.observability` module).

### Commit 2 — `feat(observability): WP-CORE-20 RunManifest + StageEmitter Pydantic core (F-9)`

**Files NEW:**
- `extension/backend/core/observability/__init__.py`
- `extension/backend/core/observability/run_manifest.py` — Pydantic `RunManifest`, `StageRecord`, `LLMCallRecord`, `JSONParseFailureRecord`, `LLMAggregate`. `write_manifest_atomic`.
- `extension/backend/core/observability/emitter.py` — `StageEmitter`, `_emitter_var`, `_stage_var`, `get_current_emitter`, `_finalize_manifest_safely`.
- `extension/backend/core/observability/_verifier_mapping.py` — `_ISSUE_TYPE_TO_CHECK_ID` + `canonical_check_id`.

**Expected state:** `tests/test_observability_run_manifest.py`, `tests/test_observability_emitter.py`, `tests/test_observability_verifier_counts.py`, `tests/test_observability_atomic_write.py` GREEN. Pipeline E2E tests still RED (no wiring yet).

### Commit 3 — `feat(llm): WP-CORE-20 instrument GeminiClient + OllamaClient at LLMResponse construction (F-9)`

**Files MODIFIED:**
- `extension/backend/core/llm/gemini.py` — `chat` + `structured_output` final return: call `emitter.record_llm_call(llm_response, operation=kwargs.get("operation","<unknown>"))` if emitter present.
- `extension/backend/core/llm/ollama.py` — symmetric.
- `extension/backend/core/llm/retry.py` — final `except RetryExhaustedError` → emitter `_record_retry_exhausted`.

**Expected state:** No new RED→GREEN flip in this commit (emitter still not wired into pipeline). Existing 404 tests + new observability tests still GREEN.

### Commit 4 — `feat(architect, scout): WP-CORE-20 parallel-Scout contextvar propagation + manual-parse failure record (F-9)`

**Files MODIFIED:**
- `extension/backend/core/architect.py`:
  - `DomainArchitect.__init__` accepts `emitter: Optional[StageEmitter] = None`.
  - `analyze_document` wraps each stage with `with emitter.stage(name)` (no-op if emitter is None).
  - Parallel-Scout branch uses `contextvars.copy_context().run(...)` wrapper.
  - Manual-parse callsites (Scout, Architect, Specialist) call `emitter.record_json_parse_failure(...)` on `json_parse_failed` retry path.

**Expected state:** `T-EMITTER-PARALLEL-1` + manual-parse tests GREEN. Pipeline E2E test still RED (endpoint wiring missing).

### Commit 5 — `feat(main): WP-CORE-20 endpoint-entry manifest creation + finalize-safely (F-9)`

**Files MODIFIED:**
- `extension/backend/main.py` — `/generate-model` and `/generate-model-stream` thread call `_run_generate_pipeline(file_paths, srs_dir_resolved)` which constructs the manifest, runs the pipeline under `try/except/finally`, and calls `_finalize_manifest_safely`.

**Expected state:** All RED tests now GREEN, including E2E + failure-path manifest tests. Baseline ≥ 412.

### Commit 6 — `feat(scripts): WP-CORE-20 aggregate_runs.py for N=10 pooled + distributions output (F-9)`

**Files NEW:**
- `extension/backend/scripts/aggregate_runs.py` — CLI: `python -m scripts.aggregate_runs runs/manifests/run-*.json --out-csv ... --out-pooled ... --out-distributions ...`

**Files MODIFIED:**
- `extension/backend/tests/test_observability_aggregator.py` may need updating once the script signature is finalized.

**Expected state:** All tests GREEN. `runs/aggregates/` is the canonical paper-artifact directory.

### Commit 7 — `chore(artifacts): WP-CORE-20 dev_doc + audit state + INDEX update`

**Files NEW:**
- `development_docs/WP-CORE-20-emse-grade-logging.md`

**Files MODIFIED:**
- `development_docs/INDEX.md` — new ACTIVE row.
- `.planning/pipeline_audit/CURRENT.md`
- `.planning/pipeline_audit/improvements_backlog.md` — F-9 MINOR-OPEN → SHIPPED.
- `.planning/pipeline_audit/decision_log.md` — D-PICK-WP-CORE-20 + D-CODEX-REVIEW-WP-CORE-20.

## Test count delta projection

- T-MANIFEST-1..7 (run_manifest tests): +7
- T-EMITTER-1..8 (emitter tests) including T-EMITTER-PARALLEL-1: +8
- T-VERIFIER-COUNTS-1..3: +3
- T-AGG-1..5 (aggregator): +5
- T-ATOMIC-WRITE-1, T-ATOMIC-WRITE-2: +2
- T-OBS-E2E-1 (pipeline E2E): +1
- T-OBS-FAIL-1..6 (failure paths, finalize-safely): +6

**Total minimum: +32 tests.** Baseline goes from 404 → ~436.

## Risk register (carried from spec §12)

| ID | Status | Owner |
|---|---|---|
| R-1 manifest write failure masks original exc | Fixed (W-2) | implementation |
| R-2 TokenTracker singleton pollution | Fixed (C-2) | implementation |
| R-3 corrupt manifest | Aggregator-skip + SHA-256 | implementation |
| R-4 UUID collision | UUID prefix in filename | implementation |
| R-5 CLI/test invocation without emitter | Optional kwarg, default None | implementation |
| R-6 ContextVar leak FastAPI | Set/reset token in finally | implementation |
| R-7 self-measurement validity | Out of scope; threats doc note | docs |
| R-8 non-serializable error context | `_safe_context()` coerce | implementation |
| R-9 future parallel-Scout regress | T-EMITTER-PARALLEL-1 guard | tests |

## Rollback plan

If GREEN reveals a structural issue not caught in spec v2, revert commits in reverse order via `git revert HEAD~1..HEAD` (atomic commits make this safe). No DB migrations or external side effects to clean up.
