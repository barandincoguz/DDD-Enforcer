# WP-CORE-8 — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-8-typed-pipeline-error-handler-design.md` (v2, post-Codex)
**Status:** EXECUTED (RED `72898af`, GREEN `a2bca34`, DOC `5d229b6`, PLANNING `{this-sha}`)
**Baseline:** 358 passed at HEAD `cecfee1` (post-WP-CORE-7)
**Target:** 365 passed (+7 tests, zero regression)

## Task breakdown

### Task 1 — RED commit (`72898af`)

`test(main): WP-CORE-8 red-phase tests for typed PipelineError response`

Files:
- `tests/test_main_pipeline_error_response.py` (NEW) — T-HELPER-1..5 (Codex W-1 + W-4)
- `tests/test_main_pipeline_error_endpoint.py` (NEW) — T-ENDPOINT-1 + T-SSE-1 (Codex W-2)

RED pytest: 358 passed, 7 failed, 31 deselected (365 collected). Failures: 5 ImportError (`_build_pipeline_error_response` not defined) + 2 KeyError (`error_type` missing in response).

### Task 2 — GREEN commit (`a2bca34`)

`fix(main): WP-CORE-8 typed PipelineError response handler in /generate-model + /generate-model-stream`

Files modified:
- `main.py` — 
  - `from core.orchestration.errors import PipelineError`
  - `_issue_to_dict`, `_scalarize`, `_build_pipeline_error_response` helpers (60 LOC)
  - `except PipelineError` block in `generate_model_endpoint` BEFORE bare-Exception (line 427)
  - `except PipelineError` block in `run_pipeline()` thread BEFORE bare-Exception (line 533)
  - `event_generator` SSE emitter dict-spread adapter for typed payload (line 561)

GREEN pytest: 365 passed, 0 failed, 31 deselected. +7 vs baseline; zero regression.

### Task 3 — DOC commit (`5d229b6`)

`chore(artifacts): WP-CORE-8 dev_doc + audit state update + F-23 SHIPPED`

Files:
- `development_docs/WP-CORE-8-typed-pipeline-error-handler.md` (created)
- `development_docs/INDEX.md` (ACTIVE row #9 added)
- `.planning/pipeline_audit/CURRENT.md`
- `.planning/pipeline_audit/improvements_backlog.md` (F-23 → SHIPPED; status summary refreshed)
- `.planning/pipeline_audit/decision_log.md` (D-PICK + D-CODEX-REVIEW entries)
- `.planning/pipeline_audit/handoff-2026-05-23-1330.md` (iteration 8 handoff)

### Task 4 — PLANNING commit (this one)

`chore(planning): WP-CORE-8 spec v2 + plan into git history`

Files:
- `docs/superpowers/specs/2026-05-23-wp-core-8-typed-pipeline-error-handler-design.md` (v2)
- `docs/superpowers/plans/2026-05-23-wp-core-8-typed-pipeline-error-handler.md` (this file)

## Codex review summary

**0 CRITICAL + 4 WARN + 3 NIT + 1 OQ.** All WARN adopted inline:
- W-1: SpecialistShapeError `validation_errors` + `raw_excerpt` added to helper attr lists; T-HELPER-5 added.
- W-2: T-SSE-1 added; drains `body_iterator`, parses final `data:` line, asserts wire-compat + typed siblings.
- W-3: TS wire-compat claim reframed (extension currently swallows SSE errors as parse warnings; future work).
- W-4: T-HELPER-4 strengthened with `json.loads(json.dumps(payload))` round-trip + severity-string normalization + no-repr-fallback assertions.

3 NIT confirmed. 1 OQ-5 (lifespan path) deferred with concrete revisit trigger ("only if startup auto-generation becomes EMSE run evidence").

## Post-execution status

- Pytest: 358 → 365 (+7 tests, zero regression).
- Commits: 4 atomic with Claude trailer.
- No `git push`. Loop discipline maintained.
- F-23 → SHIPPED. Orchestrator MAJOR-live count is now ZERO.
- Iteration 8 handoff at `.planning/pipeline_audit/handoff-2026-05-23-1330.md` recommends F-2 (cp1254) — ingestion pivot.
