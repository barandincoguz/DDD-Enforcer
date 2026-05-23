# WP-CORE-8 — Typed PipelineError handler in main.py (F-23)

**Date:** 2026-05-23
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 7)
**Status:** REVISED v2 — addressed Codex xhigh adversarial review (0 CRITICAL + 4 WARN + 3 NIT + 1 OQ; all WARN handled inline; 1 OQ tracked as explicit out-of-scope deferral with trigger)
**Parent finding:** `.planning/pipeline_audit/improvements_backlog.md` finding **F-23** (MAJOR, NEW from WP-CORE-7 Codex W-6)
**Loop:** Domain Pipeline Hardening Loop (seventh WP; baseline 358 confirmed at HEAD `cecfee1`)
**Sibling iterations:**
- Iteration 1 — WP-CORE-2 shipped at `25e6880` (reference-heading truncation)
- Iteration 2 — WP-CORE-3 shipped at `daefeb0` (empty-input contract)
- Iteration 3 — WP-CORE-4 shipped at `02e0fe9` (`IntermediateSaveError` + `srs_path` propagation)
- Iteration 4 — WP-CORE-5 ABANDONED + WP-CORE-5b shipped at `27a5d98` (`SynthesizerEmptyModelError` taxonomy preservation)
- Iteration 5 — WP-CORE-6 shipped at `4c8580c` (D1 verifier non-vacuous, Architect populates `supporting_sentence_ids` end-to-end)
- Iteration 6 — WP-CORE-7 shipped at `cecfee1` (Refiner stage-aware re-runs + `ArchitectGroundingError`)
**Codex review:** `decision_log.md` D-CODEX-REVIEW-WP-CORE-8 (to be appended at DOC commit).

---

## Revision history

- **v1 (draft, 2026-05-23 ~12:55 GMT+3)** — initial spec; sent to Codex xhigh for adversarial review.
- **v2 (this version, 2026-05-23 ~13:10 GMT+3)** — Codex xhigh review verdict: **0 CRITICAL + 4 WARN + 3 NIT + 1 OQ**. Dispositions:

  | # | finding | category | disposition |
  |---|---|---|---|
  | **W-1 (F-1)** | `specialist-shape-error-attrs-dropped`: `SpecialistShapeError(SpecialistFailureError)` (`errors.py:78-95`) carries `validation_errors` + `raw_excerpt` populated at `architect.py:679-683` and `:696-700`. v1 helper attr list omits both. Typed taxonomy partially erased for an existing subclass. | scope gap | **ADOPTED.** Spec v2 §D-3 adds `raw_excerpt` to scalar attrs and `validation_errors` to list attrs (serialized via `_issue_to_dict` since elements are pydantic-error dicts). T-HELPER-5 added: `_build_pipeline_error_response(SpecialistShapeError(...))` preserves both fields. |
  | **W-2 (F-2)** | `missing-sse-wire-format-test`: D-6 changes observable SSE shape but no test parses emitted SSE JSON. v1 risk table's R-1 mitigation rests on manual verification. | test gap | **ADOPTED.** New T-SSE-1 added: monkeypatch `DomainArchitect.analyze_document` to raise `ArchitectGroundingError`; drain `event_generator` output; parse final `data:` line; assert `type == "error"`, `isinstance(event["error"], str)`, `event["error_type"] == "ArchitectGroundingError"`, and `event["error"]` is not a nested object. |
  | **W-3 (F-3)** | `ts-wire-compat-claim-overstated`: extension at `extension.ts:680-687` wraps the SSE error throw in a catch-and-log fallback — currently swallows stream error events as parse warnings. The wire-compat claim "VSCode extension reads `event.error` as string" is true at the surface but the extension doesn't act on it today. | documentation accuracy | **ADOPTED with reframe (no TS change).** Spec v2 §D-6 + §Downstream-impact reworded: "Python payload preserves `event.error` as a string for compatible consumers; the current VSCode extension SSE-error handler at `extension.ts:680-687` is wrapped in a parse-warning fallback and does not surface this field directly. WP-CORE-8 does not modify the TS handler — a future TS fix can opt into the typed fields without Python changes." No code change. |
  | **W-4 (F-4)** | `t-helper-4-too-weak`: v1 wording "json.dumps()-able" only catches exception-level failures, not silent semantic corruption (e.g., a collapsed `{"repr": "..."}` fallback would pass). | test correctness | **ADOPTED.** T-HELPER-4 strengthened: assert `json.loads(json.dumps(payload))` round-trip preserves expected keys + values for both legacy `core.verifier.types.VerifierIssue` (dataclass with `IssueSeverity` enum) and contract `core.pipeline_contracts.VerifierIssue` (Pydantic). Specifically: severity normalizes to a string, `message` field preserved verbatim, no `repr` fallback used. |
  | **OQ-1 (F-5)** | `lifespan-path-also-erases`: `main.py:173` lifespan calls `generate_domain_model` → `analyze_document`; exception caught at `:180-185` as generic `Exception`, app_state set to `{}`. Lifespan has no HTTP response so the helper doesn't fit directly. Spec v1 mentions `/validate` exclusion but not lifespan. | scope hygiene | **ADOPTED as explicit out-of-scope OQ-5 (NEW).** Lifespan typed logging is out of F-23 response-boundary scope. Documented with concrete revisit trigger: "Open a future WP only if startup auto-generation becomes EMSE run evidence (e.g., reproducibility manifest depends on lifespan failure-mode signaling)." No code change for v2. |
  | **N-1 (F-6)** | `t-endpoint-1-direct-call-safe`: existing wiring tests at `test_main_wiring.py:177-182` call `main.generate_model_endpoint(request)` directly. | confirmation | **ACCEPT-AS-IS.** T-ENDPOINT-1 uses the same direct pattern. |
  | **N-2 (F-7)** | `r-3-oq-1-defensible`: bare-Exception fallback retention + `/validate` exclusion both correctly scoped. | confirmation | **ACCEPT-AS-IS.** OQ-1 v2 adds line-number citations for audit trace. |
  | **N-3 (F-8)** | `helper-over-inline-x2`: helper avoids JSON-shape drift between HTTP and SSE paths. | confirmation | **ACCEPT-AS-IS.** |

  **Codex disposition summary**: 0 CRITICAL (clean spec architecture); 4 WARN all ADOPTED inline (3 spec/test changes + 1 documentation reframe); 3 NIT confirmed; 1 OQ recorded as explicit out-of-scope deferral with concrete revisit trigger.

---

## Motivation

### The gap (typed taxonomy lost at FastAPI response boundary)

`extension/backend/main.py` has 10 bare `except Exception` handlers (lines 77, 180, 194, 211, 226, 410, 427, 518, 533, 721). None catch `PipelineError`. The post-WP-CORE-7 pipeline produces typed `PipelineError` subclasses with rich payload (`ArchitectGroundingError.srs_path` + `.issues` + `.residual_issues` + `.cycles_attempted`; `SynthesizerEmptyModelError.srs_path` + `.input_summary`; `IntermediateSaveError.stage` + `.filepath` + `.srs_path`), but the bare-Exception catch at `main.py:427` (`/generate-model`) and `main.py:533` (`/generate-model-stream` thread) reduce all of it to `str(e)`.

Empirical impact:
- **Run manifest**: failure responses lose typed structure. Clients can't programmatically distinguish "Architect couldn't ground a context" from "Synthesizer returned empty model" from "Intermediate save failed" without parsing the raw string.
- **EMSE methodology paper**: the Methods-section reproducibility claim (post-WP-CORE-4/5b/6/7) builds on typed pipeline errors. The response boundary erases the taxonomy.
- **WP-CORE-7 Codex W-6 lineage**: spec v1 of WP-CORE-7 incorrectly claimed `main.py` had typed handlers; Codex W-6 corrected that the only catches are bare `Exception`. F-23 was opened as the explicit follow-up to close that gap.

### Production reachability (loop discipline — mandatory subsection)

Per loop discipline: every WP spec verifies the bug is LIVE before drafting the fix.

**F-23 status: LIVE in production.** Path verified:
1. `/generate-model` endpoint (`main.py:330-434`) calls `DomainArchitect.analyze_document` (line 362).
2. `analyze_document` invokes `run_pipeline` (`core/orchestration/pipeline.py`).
3. `run_pipeline` raises `PipelineError` subclasses on failure (`ArchitectGroundingError` post-WP-CORE-7, `SynthesizerEmptyModelError` post-WP-CORE-5b, `ArchitectExtractionError` from existing code, etc.).
4. The bare `except Exception as e:` at line 427 catches it as a generic `Exception`. Response: `{"success": False, "error": str(e)}` — type information lost.

Comparable path for `/generate-model-stream` (lines 437-574): exception inside the background `run_pipeline()` thread is caught at line 533 with the same bare-Exception pattern; `result_holder["error"] = str(e)`; SSE event emits `{'type': 'error', 'error': '<string>'}`.

**Verifying with the live exception chain**: WP-CORE-7 GREEN's T-INT-1 + WP-CORE-7 DOC's empirical observation confirm `ArchitectGroundingError` raises from production paths. The bare-Exception catch reduces it to a string at the FastAPI response level.

The `/validate` endpoint does NOT call `analyze_document` — it only loads pre-computed `domain_rules`. F-23 scope is therefore JUST the two `/generate-model*` endpoints.

---

## Discovery (audit-text-vs-code-reality)

### D-1. Backlog claim verified

**Claim** (backlog F-23): "All exception handlers in main.py are bare `except Exception` with no typed `PipelineError` catch."

**Code reality (HEAD `cecfee1`):**

```bash
$ grep -n "except\|PipelineError\|from core.orchestration" main.py
# (no PipelineError import; no typed catches found)
```

Confirmed: zero `from core.orchestration` imports in `main.py`; zero typed `except PipelineError` blocks. All 10 catches at lines 77, 180, 194, 211, 226, 410, 427, 518, 533, 721 are bare `except Exception` (some `except Exception as e:`, some `except Exception:`).

### D-2. SSE wire format contract with VSCode extension

`extension/src/extension.ts:683` consumes the SSE error event:

```typescript
} else if (event.type === "error") {
  throw new Error(event.error || "Unknown error");
}
```

Wire payload (`main.py:562`):

```python
yield f"data: {json.dumps({'type': 'error', 'error': result_holder['error']})}\n\n"
```

`event.error` MUST remain a JSON-serializable string for the TypeScript `new Error(...)` constructor to render a meaningful message. If `event.error` becomes an object, JavaScript `new Error(object)` coerces to `"[object Object]"` — bad UX.

**Constraint:** any new fields added to the SSE error payload MUST be additive (sibling fields alongside `error`). The `error` string field stays as the human-readable summary; new fields like `error_type`, `srs_path`, etc., live at the same top level of the SSE payload.

---

## Design

### D-3. New helper `_build_pipeline_error_response`

`main.py` gains a private helper that converts a `PipelineError` instance into a JSON-serializable response dict:

```python
from core.orchestration.errors import PipelineError

def _build_pipeline_error_response(exc: PipelineError) -> Dict[str, Any]:
    """Build a structured response dict from a PipelineError.

    Always includes:
        success: False
        error: str(exc)                 # human-readable summary (compat with
                                         # legacy SSE consumers that expect
                                         # event.error to be a string)
        error_type: type(exc).__name__   # e.g. "ArchitectGroundingError"
    
    Conditionally includes (when the exception carries them):
        srs_path, input_summary, stage, filepath,
        issues, residual_issues, cycles_attempted

    Lists of typed VerifierIssue or other dataclasses/Pydantic objects are
    converted to JSON-serializable shapes (lists of dicts) via best-effort
    duck-typing: tries `.model_dump()` first (Pydantic), then `__dict__`
    (dataclasses), then `str()` (everything else).
    """
    payload: Dict[str, Any] = {
        "success": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
    }
    # Scalar attributes — preserve as-is if JSON-safe.
    # Codex W-1 (F-1): `raw_excerpt` added for SpecialistShapeError.
    for attr in ("srs_path", "input_summary", "stage", "filepath",
                 "cycles_attempted", "context_name", "chunk_id",
                 "attempts", "entity_name", "raw_excerpt"):
        val = getattr(exc, attr, None)
        if val is not None and isinstance(val, (str, int, float, bool)):
            payload[attr] = val
    # List attributes — serialize each item to dict-or-string.
    # Codex W-1 (F-1): `validation_errors` added for SpecialistShapeError.
    for attr in ("issues", "residual_issues", "validation_errors"):
        val = getattr(exc, attr, None)
        if isinstance(val, list):
            payload[attr] = [_issue_to_dict(i) for i in val]
    return payload


def _issue_to_dict(issue: Any) -> Dict[str, Any]:
    """Convert a VerifierIssue (legacy dataclass or contract Pydantic) into
    a JSON-serializable dict."""
    if hasattr(issue, "model_dump"):
        return issue.model_dump()  # type: ignore[no-any-return]
    if hasattr(issue, "__dict__"):
        out: Dict[str, Any] = {}
        for k, v in issue.__dict__.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            elif hasattr(v, "value"):  # Enum
                out[k] = v.value
            else:
                out[k] = str(v)
        return out
    return {"repr": str(issue)}
```

Two helper functions. ~50 LOC total. Located near top of `main.py` after imports (before endpoint handlers).

### D-4. `/generate-model` endpoint typed catch

`main.py:427` becomes:

```python
except PipelineError as exc:
    print(f"  ❌ PIPELINE ERROR ({type(exc).__name__}): {exc}")
    import traceback
    traceback.print_exc()
    return _build_pipeline_error_response(exc)
except Exception as e:
    print(f"  ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    return {
        "success": False,
        "error": str(e),
    }
```

Typed catch BEFORE the bare-Exception fallback. The bare-Exception catch is preserved for non-PipelineError failures (filesystem I/O errors, RAG init errors that escape the inner try, etc.) per AGENTS.md "explicit failure" — we don't want to suppress non-pipeline bugs by routing them through PipelineError shape.

### D-5. `/generate-model-stream` thread typed catch

`main.py:533` becomes:

```python
except PipelineError as exc:
    print(f"[STREAM] Pipeline error ({type(exc).__name__}): {exc}")
    import traceback
    traceback.print_exc()
    result_holder["error"] = _build_pipeline_error_response(exc)
except Exception as e:
    print(f"[STREAM] Domain model pipeline error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    result_holder["error"] = str(e)
```

`result_holder["error"]` is now a `Union[str, Dict[str, Any]]`. The `event_generator` SSE emitter adapts.

### D-6. SSE `event_generator` adapts to dict-or-string error payload

`main.py:561-562` becomes:

```python
if result_holder["error"]:
    err = result_holder["error"]
    if isinstance(err, dict):
        # PipelineError path: flatten dict into top-level SSE event so
        # `event.error` (string) stays compatible with VSCode extension
        # at extension.ts:683 while new typed fields are siblings.
        yield f"data: {json.dumps({'type': 'error', **err})}\n\n"
    else:
        # Generic Exception path: legacy string-only payload.
        yield f"data: {json.dumps({'type': 'error', 'error': err})}\n\n"
```

The dict spread (`**err`) puts every key from `_build_pipeline_error_response` (`success`, `error`, `error_type`, etc.) at the top level of the SSE payload. The VSCode extension's `event.error` field still resolves to the human-readable string. New consumers can read `event.error_type`, `event.srs_path`, etc.

### D-7. Import `PipelineError` at top of main.py

```python
from core.orchestration.errors import PipelineError
```

Single import at the imports block. No subclass imports needed — the `PipelineError` superclass catches all subclasses.

---

## Test plan

**RED commit expected pytest result:** 358 + 7 new RED-by-design = 365 collected; 358 passed, 7 failed, 31 deselected.

| # | name | file | what it asserts | RED expectation |
|---|---|---|---|---|
| T-HELPER-1 | `test_build_pipeline_error_response_handles_architect_grounding_error` | `tests/test_main_pipeline_error_response.py` (NEW) | `_build_pipeline_error_response(ArchitectGroundingError(...))` returns dict with `success=False`, `error_type="ArchitectGroundingError"`, `srs_path`, `cycles_attempted`, `issues` (list of dicts), `residual_issues` (list of dicts) | FAIL — helper doesn't exist |
| T-HELPER-2 | `test_build_pipeline_error_response_handles_synthesizer_empty_model_error` | same | helper returns `error_type="SynthesizerEmptyModelError"`, `srs_path`, `input_summary` | FAIL — helper doesn't exist |
| T-HELPER-3 | `test_build_pipeline_error_response_handles_intermediate_save_error` | same | helper returns `error_type="IntermediateSaveError"`, `stage`, `filepath`, `srs_path` | FAIL — helper doesn't exist |
| T-HELPER-4 | `test_build_pipeline_error_response_round_trip_preserves_issue_fields` (Codex W-4 strengthened) | same | `payload = _build_pipeline_error_response(exc_with_issues)` where `exc.issues` contains BOTH a legacy `core.verifier.types.VerifierIssue` AND a contract `core.pipeline_contracts.VerifierIssue`. `roundtripped = json.loads(json.dumps(payload))`. Assert: (a) `roundtripped["issues"]` is a list of dicts; (b) each dict has `message` matching the original; (c) severity is a string (not the `IssueSeverity` enum repr); (d) no dict equals `{"repr": "..."}` (the `_issue_to_dict` fallback path) | FAIL — helper doesn't exist |
| T-HELPER-5 | `test_build_pipeline_error_response_handles_specialist_shape_error` (Codex W-1 NEW) | same | `_build_pipeline_error_response(SpecialistShapeError(context_name="X", errors=[{"loc":["k"], "msg":"missing"}], raw_excerpt="..."))` preserves `error_type="SpecialistShapeError"`, `context_name`, `raw_excerpt`, AND `validation_errors` (list of dicts in roundtrip-safe shape) | FAIL — helper doesn't exist |
| T-ENDPOINT-1 | `test_generate_model_endpoint_returns_typed_error_on_pipeline_error` | `tests/test_main_pipeline_error_endpoint.py` (NEW) | Mock `DomainArchitect.analyze_document` to raise `ArchitectGroundingError`; call `generate_model_endpoint` directly (Codex N-1 disposition — matches existing pattern at `test_main_wiring.py:177-182`); response dict contains `success=False`, `error_type="ArchitectGroundingError"`, `srs_path` populated, `issues` populated | FAIL — endpoint catches as generic Exception |
| T-SSE-1 | `test_generate_model_stream_emits_typed_error_in_sse_payload` (Codex W-2 NEW) | same | Mock `DomainArchitect.analyze_document` to raise `ArchitectGroundingError`; call `generate_model_stream_endpoint(request)`; drain the returned `StreamingResponse.body_iterator`; parse final `data: ...` line as JSON; assert `event["type"] == "error"`, `isinstance(event["error"], str)` (NOT a dict — keeps VSCode extension wire-compat), `event["error_type"] == "ArchitectGroundingError"`, `event["srs_path"]` present | FAIL — emitter doesn't include typed fields |

**Total**: 7 fail. GREEN turns all 7 green.

---

## Risks

| # | risk | mitigation |
|---|---|---|
| R-1 | SSE wire format change breaks VSCode extension. | Spec D-6 keeps `error` string field; dict-spread emits additive fields. `extension.ts:683` `event.error` resolves to string as before. Manually verified pre-spec. |
| R-2 | Helper duck-typing for issue serialization is fragile (Pydantic vs dataclass vs other). | T-HELPER-4 explicitly tests `json.dumps` round-trip on both legacy `core.verifier.types.VerifierIssue` (dataclass with IssueSeverity enum) and contract `core.pipeline_contracts.VerifierIssue` (Pydantic). |
| R-3 | Bare-Exception fallback retained — any latent bug now goes through the typed catch instead. | NO — `except PipelineError` is BEFORE `except Exception`. Non-PipelineError exceptions skip the typed branch entirely and hit the legacy fallback. |
| R-4 | `_build_pipeline_error_response` adds JSON-fields that contain potentially-large issue lists; SSE payload size can grow. | `ArchitectGroundingError.issues` is bounded by D1 verifier check output (one issue per under-grounded context, typically ≤ 8 per pipeline run). Acceptable size. Document in dev_doc as known characteristic. |
| R-5 | Backwards-compat: legacy clients keyed on `error` string field. | Compat preserved — `error` field always present in both endpoints' responses. Helper invariant: `payload["error"] = str(exc)` is unconditional. |

---

## Open questions

| # | question | disposition |
|---|---|---|
| **OQ-1** | Should `/validate` endpoint also get typed catch (e.g., for `_needs_llm_advanced_checks` or `llm.analyze_advanced_violations` failures)? | **NO for v1.** `/validate` (`main.py:594-595`) doesn't invoke `analyze_document`, so it doesn't surface `PipelineError`. The `except Exception as e:` at `:721` lives inside the RAG-per-violation loop (`:705-725`) and only catches RAG / chroma failures. If a future WP adds Verifier-stage calls to `/validate`, that WP extends the typed catch. (Codex N-2 confirmed.) |
| **OQ-2** | Should `_build_pipeline_error_response` live in `main.py` or `core/orchestration/serialization.py`? | **`main.py` for v1.** AGENTS.md "smallest correct change" — the helper is FastAPI-response-shape specific and has no other callers. If a 2nd endpoint surface (CLI, MCP) appears, extract then. (Codex N-3 confirmed.) |
| **OQ-3** | Should the SSE payload include a Pydantic schema reference for clients to validate against? | **NO.** Premature schema standardization. The current additive-fields contract is enough for the VSCode extension + future EMSE reproducibility tooling. |
| **OQ-4** | Should `traceback` strings be added to the response for debugging? | **NO.** Security/info-disclosure risk in production deployments. The server-side log still prints `traceback.print_exc()`. |
| **OQ-5 (NEW, Codex F-5)** | Lifespan path (`main.py:173`) also erases typed `PipelineError` via `except Exception` at `:180-185`. Should it get typed handling too? | **DEFERRED, out-of-scope for F-23.** Lifespan has no HTTP response body so `_build_pipeline_error_response` doesn't fit directly — it would need a logging-only or app_state-marker variant. Concrete revisit trigger: open a future WP **only if** startup auto-generation becomes EMSE run evidence (e.g., reproducibility manifest depends on lifespan failure-mode signaling). For current scope F-23 is response-boundary only. |

---

## Atomic commit sequence

1. **RED commit** — `test(main): WP-CORE-8 red-phase tests for typed PipelineError response`
   - `tests/test_main_pipeline_error_response.py` (NEW) — T-HELPER-1..5 (Codex W-1 + W-4)
   - `tests/test_main_pipeline_error_endpoint.py` (NEW) — T-ENDPOINT-1 + T-SSE-1 (Codex W-2)
   - RED pytest: 365 collected, 358 passed, 7 failed, 31 deselected
   - LOC: +~200

2. **GREEN commit** — `fix(main): WP-CORE-8 typed PipelineError response handler in /generate-model + /generate-model-stream`
   - `main.py` — add `_build_pipeline_error_response` + `_issue_to_dict` helpers + `from core.orchestration.errors import PipelineError` + typed catches in both endpoints + SSE dict-or-string adapter
   - RED tests turn green; baseline 358 → 363
   - LOC: +~80 production

3. **DOC commit** — `chore(artifacts): WP-CORE-8 dev_doc + audit state update + F-23 SHIPPED`
   - `development_docs/WP-CORE-8-typed-pipeline-error-handler.md` (created)
   - `development_docs/INDEX.md` (ACTIVE row #9 added)
   - `.planning/pipeline_audit/CURRENT.md` (iteration 7 SHIPPED status)
   - `.planning/pipeline_audit/improvements_backlog.md` (F-23 → SHIPPED)
   - `.planning/pipeline_audit/decision_log.md` (D-PICK-WP-CORE-8 + D-CODEX-REVIEW-WP-CORE-8)
   - `.planning/pipeline_audit/handoff-2026-05-23-<time>.md` (iteration 8 handoff)

4. **PLANNING commit** — `chore(planning): WP-CORE-8 spec v2 + plan into git history`
   - `docs/superpowers/specs/2026-05-23-wp-core-8-typed-pipeline-error-handler-design.md` (v2)
   - `docs/superpowers/plans/2026-05-23-wp-core-8-typed-pipeline-error-handler.md`

---

## Downstream impact

| concern | impact | action |
|---|---|---|
| VSCode extension SSE consumer (`extension.ts:680-687`) | **Codex W-3 disposition:** the current TS SSE-error handler is wrapped in a parse-warning fallback (`extension.ts:685-687`) and currently swallows stream error events. Python payload preserves `event.error` as a string for compatible consumers; additive sibling fields (`error_type`, `srs_path`, etc.) are emitted but not surfaced by today's TS code. WP-CORE-8 does NOT modify the TS handler — a future TS fix can opt into the typed fields without Python changes. | None for WP-CORE-8 (Python-only). TS handler fix tracked as out-of-scope follow-up (no backlog entry yet — file one if user-visible error UX becomes a milestone). |
| `/generate-model` non-stream response shape | Adds optional `error_type`, `srs_path`, `issues`, etc. fields on failure. `error` string preserved. | Document in dev_doc as additive contract. |
| Run manifest / log artifacts | No change in artifact format; only console log emits typed prefix `❌ PIPELINE ERROR (TypeName)`. | None. |
| EMSE paper Methods section | Pre-WP-CORE-8: typed pipeline taxonomy lost at FastAPI boundary. Post-WP-CORE-8: typed taxonomy preserved end-to-end through HTTP response. | Flag for advisor at next paper revision. |

---

## Goal-backward verification (spec-level)

| Iteration-7 goal | Evidence at spec-time |
|---|---|
| Pick F-23 per WP-CORE-7 handoff Codex W-6 lineage | F-23 picked; LIVE-in-production verified by Discovery D-1. |
| Spec → Codex xhigh review → plan → SDD → dev_doc → state update | Spec v1 drafted (this file). Codex review pending. |
| Each commit gated on pytest ≥ baseline | Atomic commit sequence specifies RED-pytest delta (358 + 7 fail) and GREEN-pytest delta (→ 365 pass). |
| Production reachability subsection in spec §Motivation | YES — §Motivation includes "Production reachability (loop discipline)" subsection. |
| Smallest correct change (AGENTS.md) | YES — only 2 endpoints touched; bare-Exception catches retained as fallback; helper kept in main.py per OQ-2. |
| No breakage of VSCode extension wire format | YES — D-6 dict-spread keeps `error` field as string; new fields are additive siblings. |

Spec v1 ready for Codex xhigh review.
