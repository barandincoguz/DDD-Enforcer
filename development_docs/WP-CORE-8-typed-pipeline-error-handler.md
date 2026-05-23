# WP-CORE-8 — Typed PipelineError handler in main.py

**Status:** SHIPPED 2026-05-23
**Branch / commits:**
- RED `72898af` — test(main): WP-CORE-8 red-phase tests for typed PipelineError response
- GREEN `a2bca34` — fix(main): WP-CORE-8 typed PipelineError response handler in /generate-model + /generate-model-stream
- DOC `{this commit}` — chore(artifacts): WP-CORE-8 dev_doc + audit state update + F-23 SHIPPED
- PLANNING `{pending}` — chore(planning): WP-CORE-8 spec v2 + plan into git history

**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-8-typed-pipeline-error-handler-design.md` (v2; revised after Codex xhigh review)
**Plan:** `docs/superpowers/plans/2026-05-23-wp-core-8-typed-pipeline-error-handler.md` (to be added in PLANNING commit)
**Parent finding:** `.planning/pipeline_audit/improvements_backlog.md` finding **F-23** (MAJOR, NEW from WP-CORE-7 Codex W-6) — now SHIPPED.

## TL;DR

F-23 closes the FastAPI-response-level typed-taxonomy gap. Pre-WP-CORE-8 main.py had 10 bare-Exception handlers and zero `from core.orchestration` imports; ArchitectGroundingError (WP-CORE-7), SynthesizerEmptyModelError (WP-CORE-5b), IntermediateSaveError (WP-CORE-4), SpecialistShapeError (P3-era), and ArchitectExtractionError were all reduced to `{"success": False, "error": str(e)}`. Post-WP-CORE-8: typed `except PipelineError` BEFORE bare-Exception fallback in `/generate-model` (sync) + `/generate-model-stream` (thread); new `_build_pipeline_error_response` helper preserves payload attributes (srs_path, issues, residual_issues, cycles_attempted, etc.); SSE wire format additive — `event.error` stays a string for VSCode extension compat, typed siblings (`error_type`, `srs_path`, etc.) ride along.

Baseline: 358 → 365 passing (+7 tests, zero regression).

## Motivation

WP-CORE-7 Codex W-6 surfaced this gap: spec v1 of WP-CORE-7 had claimed "existing `try/except PipelineError` blocks catch it transparently" in main.py. Codex W-6 corrected: all 10 main.py exception handlers at lines 77, 180, 194, 211, 226, 410, 427, 518, 533, 721 are bare `except Exception` with no `PipelineError` import. F-23 was opened explicitly to close this gap as the WP-CORE-7 follow-up.

Empirical impact pre-WP-CORE-8: when `ArchitectGroundingError` raised from production paths, the response collapsed to `{"success": False, "error": "Architect re-run exhausted (1 cycle(s)) with 1 unresolved grounding issue(s) (0 non-architect residual) for srs=inputs/D1.docx"}`. Clients lost the ability to programmatically branch on failure mode — `error_type`, `srs_path`, and the issues list were all string-interpolated away.

Production reachability check (loop discipline): F-23 is LIVE. Verified by tracing the call chain `/generate-model` (line 332) → `DomainArchitect.analyze_document` (line 362) → `run_pipeline` (which raises typed `PipelineError` subclasses post-WP-CORE-7) → bare `except Exception` at line 427 → response shape lost type info.

## Architectural decisions

### D-1 — Smallest correct change: 2 endpoints touched, helper in main.py

`/validate` doesn't call `analyze_document`; lifespan-startup (`main.py:173`) has no HTTP response body. F-23 scope is strictly the two `/generate-model*` endpoints. The helper `_build_pipeline_error_response` lives in main.py per OQ-2 ("smallest correct change"; no other callers exist).

### D-2 — SSE additive wire format (VSCode extension compat)

`extension/src/extension.ts:683` reads `event.error` and calls `new Error(event.error || "Unknown error")`. If `event.error` becomes a dict, JavaScript coerces to `"[object Object]"` — bad UX. To preserve compatibility:
- `result_holder["error"]` becomes `Union[str, Dict[str, Any]]`.
- `event_generator` dict-spreads the dict payload into the SSE event so `error` (string) stays as a top-level field, typed siblings (`error_type`, `srs_path`, `issues`, ...) live alongside it.
- Legacy generic-Exception path emits `{"type": "error", "error": "<string>"}` unchanged.

Codex W-3 reframed this further: the current TS handler at `extension.ts:680-687` wraps the throw in a parse-warning catch, so the SSE error events are currently swallowed by the extension. WP-CORE-8 emits the typed payload correctly; the TS handler fix is future work (no backlog entry yet — file one if user-visible error UX becomes a milestone).

### D-3 — `_issue_to_dict` duck-typing for both legacy + contract VerifierIssue

Two `VerifierIssue` classes coexist:
- `core.verifier.types.VerifierIssue` (frozen dataclass, fields: stage / location / issue_type / severity (IssueSeverity enum) / message / suggestion).
- `core.pipeline_contracts.VerifierIssue` (Pydantic BaseModel, fields: severity (Literal str) / check_id / target / message).

Helper duck-typing order: `.model_dump()` (Pydantic) → `.__dict__` scalarized (dataclass) → `repr()` fallback. `_scalarize` normalizes enums to `.value` so `IssueSeverity.ERROR` round-trips as `"error"` not `"IssueSeverity.ERROR"`. Codex W-4 strengthened test T-HELPER-4 to assert this explicitly.

### D-4 — SpecialistShapeError extra payload preserved (Codex W-1)

`SpecialistShapeError(SpecialistFailureError)` carries `validation_errors` (list of pydantic error dicts) + `raw_excerpt` (LLM payload prefix). v1 spec missed these; v2 helper attr list extended with `raw_excerpt` (scalar) + `validation_errors` (list, serialized via `_issue_to_dict`). T-HELPER-5 added.

### D-5 — Bare-Exception fallback retained, not removed (AGENTS.md "explicit failure")

The typed catch is BEFORE the bare-Exception fallback. Non-PipelineError exceptions (filesystem I/O, RAG init failures, etc.) skip the typed branch entirely and hit the legacy fallback. This is intentional: we don't want to suppress non-pipeline bugs by routing them through PipelineError shape (which would mislabel them as `error_type: "Exception"`).

## File-level changes

| File | Change | LOC delta |
|---|---|---|
| `core/orchestration/__init__.py` | (unchanged — `PipelineError` already re-exported from WP-CORE-7) | 0 |
| `extension/backend/main.py` | + `from core.orchestration.errors import PipelineError`; + `_issue_to_dict`, `_scalarize`, `_build_pipeline_error_response` helpers (~60 LOC); + `except PipelineError` block in `generate_model_endpoint` (line 427); + `except PipelineError` block in `run_pipeline()` thread inside `generate_model_stream_endpoint` (line 533); + dict-spread adapter in `event_generator` for SSE wire format (line 561-572). | +96 / -1 |
| `tests/test_main_pipeline_error_response.py` (NEW, RED) | T-HELPER-1..5 (5 tests covering ArchitectGroundingError + SynthesizerEmptyModelError + IntermediateSaveError + round-trip serialization + SpecialistShapeError). | +200 |
| `tests/test_main_pipeline_error_endpoint.py` (NEW, RED) | T-ENDPOINT-1 (sync endpoint typed payload) + T-SSE-1 (Codex W-2 NEW; drains body_iterator + parses final SSE data line + asserts wire compat). | +150 |

## Methodology applied

- **TDD with genuine RED → GREEN.** RED commit `72898af` accepted 7 known-failing tests (ImportError × 5, KeyError × 2 — all at body execution, no collection errors). GREEN commit `a2bca34` flipped all 7 green + preserved baseline 358.
- **Spec → Codex xhigh review → spec v2 → atomic commits.** Codex returned 0 CRITICAL + 4 WARN + 3 NIT + 1 OQ. All WARN handled inline; 3 NIT confirmed; 1 OQ (lifespan path) deferred with concrete trigger.
- **Production reachability subsection in spec §Motivation.** F-23 confirmed LIVE.

## Empirical results

- **Test baseline**: 358 (pre-RED) → 365 passing (GREEN; +7 new tests, zero regression).
- **LOC delta vs WP-CORE-7 HEAD `cecfee1`**: +96 / -1 (main.py only in GREEN; +350 in RED test files).
- **Response shape on failure**: pre-WP-CORE-8 was `{success: false, error: "<string>"}`; post-WP-CORE-8 PipelineError responses additively carry `error_type`, `srs_path`, `issues`, `residual_issues`, `cycles_attempted`, `input_summary`, `stage`, `filepath`, `context_name`, `raw_excerpt`, `validation_errors` (as applicable per subclass).
- **SSE wire compat**: `event.error` field remains a string; new typed fields are siblings. VSCode extension at `extension.ts:683` reads same field as before.

## Limitations + follow-ups

- **OQ-5 (DEFERRED, lifespan path)**: `main.py:173` lifespan-startup also has bare-Exception catch erasing typed PipelineError. Out of F-23 response-boundary scope. Concrete revisit trigger: open a future WP only if startup auto-generation becomes EMSE run-manifest evidence.
- **TS handler future fix**: `extension.ts:680-687` currently swallows SSE error events as parse warnings. Python payload is correct; TS handler upgrade is future work. No backlog entry yet — file one if user-visible error UX becomes a milestone.
- **`_build_pipeline_error_response` location**: `main.py` for v1 per AGENTS.md smallest-correct-change. If a 2nd endpoint surface (CLI, MCP) appears, extract to `core/orchestration/serialization.py`.

## Cross-references

- **Predecessor**: `[[WP-CORE-7-refiner-stage-aware]]` — Codex W-6 directly opened F-23 as the follow-up.
- **Invariant chain**:
  - WP-CORE-4: any future stage retry wrapper using `_save_intermediate` MUST include `except IntermediateSaveError: raise`.
  - WP-CORE-5b: any future pipeline-orchestration code constructing `SynthesizerEmptyModelError` MUST pass `srs_path`.
  - WP-CORE-6: any future Architect stage that produces context proposals MUST include `supporting_sentence_ids`; `extract_per_context_details` signature is `List[ContextHypothesis]`.
  - WP-CORE-7: every `VerifierIssue.target` MUST be `'{stage}:'`-prefixed; every `PipelineDeps` constructor MUST supply `architect_with_feedback`; architect-stage D1 ERRORs raise `ArchitectGroundingError` after 1 feedback rerun.
  - **WP-CORE-8 NEW invariant**: every new `PipelineError` subclass added under `core/orchestration/errors.py` SHOULD have its payload attributes added to the `_build_pipeline_error_response` helper attr-list (scalar attrs to the scalar-attrs tuple; list attrs to the list-attrs tuple). The helper's duck-typing fallback covers unknown attrs gracefully but loses typed structure.
- **EMSE paper**: pre-WP-CORE-8 the Methods-section claim "typed pipeline errors are surfaced through the API" was empirically false — bare-Exception erased the taxonomy. Post-WP-CORE-8 it is true on both endpoints. Flag for advisor at next paper revision.
