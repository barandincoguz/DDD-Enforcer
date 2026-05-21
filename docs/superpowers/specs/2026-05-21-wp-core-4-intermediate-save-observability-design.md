# WP-CORE-4 — Intermediate-save observability + error context propagation

**Date:** 2026-05-21
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 3)
**Status:** REVISED v2 — addressed Codex xhigh adversarial review (2 CRITICAL + 5 WARN + 3 NITS + 7 OQ all handled inline)
**Parent:** `.planning/pipeline_audit/findings/architect.md` finding **F-13** (MAJOR) + anomaly **`_current_srs_path` never assigned**
**Loop:** Domain Pipeline Hardening Loop (third WP; baseline 321 confirmed at HEAD `d7dc188`)
**Sibling iterations:**
- Iteration 1: WP-CORE-2 shipped at `25e6880` (reference-heading truncation; `core/document_parser.py`)
- Iteration 2: WP-CORE-3 shipped at `daefeb0` (empty-input contract; `core/document_parser.py` + `main.py`)
**Codex consult:** review at runtime 2026-05-21 ~14:30; raw output preserved in `decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-4`.

---

## Revision history

- **v1 (draft)** — initial spec; sent to Codex xhigh for adversarial review.
- **v2 (this version)** — 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ disposed:

  | # | finding | disposition |
  |---|---|---|
  | **C-1** | Architect-stage saves at lines 449/462 are inside `identify_contexts`' broad `except Exception` retry wrapper (line 485); a raised `IntermediateSaveError` is silently caught, retried 5×, and rewrapped as `ArchitectExtractionError`. User never sees the real save failure. | **Add `except IntermediateSaveError: raise` immediately before `except Exception` in `identify_contexts`.** The Specialist save at line 650 is already OUTSIDE the per-context retry loop (verified via re-read), so no equivalent guard is needed there. Scout save at line 236 is in dead `extract_domain_sentences` (per W-4), so analyze_document never reaches it. |
  | **C-2** | RED-phase tests stub `_save_intermediate` directly but do not test the real production propagation path. False-green risk. | **Add T-SAVE-4: `_save_intermediate` failure inside `identify_contexts` → assert `IntermediateSaveError` propagates (NOT `ArchitectExtractionError`).** Plus T-SAVE-5: failure inside `extract_per_context_details` propagates cleanly (Specialist path). |
  | **W-1** | `IntermediateSaveError(OSError)` violates the orchestration error taxonomy in `core/orchestration/errors.py` (which has `PipelineError` as base for P3 pipeline failures). | **Move `IntermediateSaveError` into `core/orchestration/errors.py` and subclass `PipelineError`.** Carry `stage`, `filepath`, `cause`, `srs_path` (the latter per W-5). |
  | **W-2** | Conditional assignment `if srs_path is not None: self._current_srs_path = srs_path` leaks stale path on instance reuse. | **Unconditional assignment** at start of every `analyze_document`: `self._current_srs_path = srs_path or "<unknown>"`. Plus T-SRS-4: sequential reuse test (call `analyze_document(srs_path="A")` then `analyze_document(srs_path=None)` and assert second-call `_current_srs_path == "<unknown>"`). |
  | **W-3** | "; "-joined batch label claim "bounded by request validation" is unsupported — `GenerateModelRequest.file_paths` is `List[str]` with no visible max length. | **Drop the bounded claim.** Treat the joined label as display-only human-readable. No truncation introduced (KISS); a 100-file batch would produce a long error message — acceptable, no clipping side-effect. |
  | **W-4** | Scout's `_save_intermediate` call at line 236 lives in legacy `extract_domain_sentences`, which `analyze_document`'s nested `scout_fn` (lines 735-752) does NOT call. Scout-save during real `analyze_document` is currently unreachable. | **Adjusted Motivation §F-13:** the production-reachable saves are Architect (lines 449/462 via `identify_contexts` invoked by `architect_fn`) and Specialist (line 650 via `extract_per_context_details` invoked by `specialist_fn`). Scout save at line 236 is dead from production (only `tests/test_architect_helpers.py:119` exercises it); we still fix the method for test consistency and to avoid surprises if the legacy path ever gets re-wired. The "typed Scout dump missing" gap is documented as a NEW anomaly in `findings/architect.md` (deferred). |
  | **W-5** | `IntermediateSaveError` carries `stage`, `filepath`, `cause` but not `srs_path`. Endpoint users won't know which SRS was being processed. | **Add `srs_path` to `IntermediateSaveError.__init__`** and read it from `self._current_srs_path` at raise time. Both observability concerns (F-13 + anomaly fold-in) now bind into the same error message. |
  | **N-1** | "6 new tests" vs "Total new tests: 7" inconsistency. | **Resolved to 11 total** (post-Codex additions): 5 for F-13, 4 for `_current_srs_path`, 3 for wiring. See updated §"Red-phase tests". |
  | **N-2** | `JSONDecodeError` cited under `json.dump` rationale — wrong direction (decode-side). | **Removed.** Encoder-side errors are `TypeError` (non-serializable object) and `ValueError` (circular references). Catch list rewritten. |
  | **N-3** | Speculative env-var no-op patch escape hatch — not actually implemented. | **Removed** from Open Questions #3 disposition. |
  | **OQ1** | Codex DISAGREE with `OSError` base. | **Adopted PipelineError base.** |
  | **OQ2** | Codex PARTIALLY: catch list narrow-correct but Architect retry must re-raise. | **Adopted both:** narrow catch `(OSError, TypeError, ValueError)` + Architect re-raise (C-1). |
  | **OQ3** | Codex PARTIALLY: label OK but no max-length bound. | **Dropped max-length claim per W-3.** |
  | **OQ4** | Codex PARTIALLY: cleaning getattr defensible if tests updated. | **Kept getattr fallbacks** — smallest correct change still applies; the 4 reads are pure-defensive, removing them is cosmetic-only. |
  | **OQ5** | Codex DISAGREE with deferring endpoint wiring tests. | **Adopted:** T-WIRE-MAIN-2 (`/generate-model`) + T-WIRE-MAIN-3 (`/generate-model-stream`) added. Lightweight monkeypatch — no FastAPI TestClient required. |
  | **OQ6** | Codex PARTIALLY: production-only callers vs all callers. | **Spec text adjusted** to read "no other production callers" (tests do call `analyze_document` directly — they pass no srs_path → "<unknown>", which is the intended fallback). |
  | **OQ7** | Codex AGREE: F-21 deferral is correct. | **Confirmed.** F-21 (vacuous D1 pass) added to `findings/architect.md` and deferred to WP-CORE-5+. |

---

## Motivation

Iteration-3 close-lookup of `core/architect.py` (923 LOC) + `core/orchestration/{pipeline,errors}.py` surfaced 10 findings (4 MAJOR + 5 MINOR + 1 TRIVIAL — F-20 downgraded to MINOR after verification; F-21 added as NEW MAJOR deferred) plus 6 anomalies. Two of these — F-13 (`_save_intermediate` silently swallows I/O exceptions) and the anomaly that `DomainArchitect._current_srs_path` is read 4 times but **never assigned** — both degrade pipeline observability. Both fixes belong together because:

1. They share the same `DomainArchitect` surface.
2. They both implement AGENTS.md "Error handling: explicit failure. No silent degradation."
3. Fixing F-13 in isolation reveals a sub-question — *which* SRS path was being processed when the save failed? — which is exactly what `_current_srs_path` is supposed to surface. They are observability twins. Per W-5 disposition, `IntermediateSaveError` now carries the `srs_path` field populated from `self._current_srs_path` at raise time, so the two fixes interlock at the error-message level.
4. Smallest-correct-change + atomic-commit cadence (matches WP-CORE-3's pattern of folding a latent bug into the main fix).

### F-13 — `_save_intermediate` silently swallows I/O exceptions

`core/architect.py:880-891`:

```python
def _save_intermediate(self, stage: str, data: Dict[str, Any]):
    """Save intermediate pipeline output to JSON file."""
    try:
        filename = f"{self.run_timestamp}_{stage}.json"
        filepath = os.path.join(INTERMEDIATE_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  💾 Saved intermediate output: {filename}")
    except Exception as e:
        print(f"  ⚠️  Failed to save intermediate output: {e}")
```

**4 call sites** (Explore subagent miscounted as 3; verified by grep):
- `core/architect.py:236` — Scout, inside `extract_domain_sentences` (`1_scout` stage). **DEAD from production** per Codex W-4 — `analyze_document` uses an inline `scout_fn` (lines 735-752) that does NOT call `_save_intermediate`. Only `tests/test_architect_helpers.py:119` invokes `extract_domain_sentences` directly. We still fix this call site because the method might be re-wired later and the fix is uniform with the other 3.
- `core/architect.py:449` — Architect, inside `identify_contexts` (`2_architect`, dict-shaped result). **PRODUCTION-REACHABLE.**
- `core/architect.py:462` — Architect, inside `identify_contexts` (`2_architect`, list-shaped result). **PRODUCTION-REACHABLE.**
- `core/architect.py:650` — Specialist, end of `extract_per_context_details` (`3_specialist`). **PRODUCTION-REACHABLE — outside the per-context retry loop, so propagates cleanly.**

**Failure modes silently swallowed today:**
- `OSError` / `PermissionError` — filesystem failures (read-only, full disk, broken `INTERMEDIATE_DIR` symlink). Dominant case.
- `TypeError` — non-serializable object reaches `json.dump`. Defensive: `data` is built from typed Pydantic dumps + native types, but a future refactor could break this.
- `ValueError` — `json.dump` raises this on circular references or invalid `indent`/`separators`.

**Codex N-2:** `JSONDecodeError` is a *decode* exception, not an *encode* exception — removed from catch rationale.

**Why this matters (blast radius = PIPELINE):**

CLAUDE.md §"Persistent Development Memory" and §"Active Submission Context" both depend on these intermediate artifacts:

> `extension/backend/core/intermediate/` — Scout/Architect/Specialist/Synthesizer per-stage JSON dumps (timestamped); useful for debugging the pipeline without re-running

For the EMSE submission, intermediate JSON dumps are the **methodology reproducibility evidence** for any RQ1-RQ4 result. A silent save failure means a run completes successfully (returns a `DomainModel`) but its lineage is unrecoverable. Worse: subsequent stages re-call `_save_intermediate` with no signal that the previous save failed, so a transient `OSError` at Architect corrupts the entire run's diagnostic chain.

The current code violates AGENTS.md explicitly:

> Error handling: explicit failure. No empty `try/except`, no silent degradation, no permissive fallbacks during development. Convert exceptions, add context, or rethrow.

The `try: ... except Exception as e: print(...)` here is a textbook silent fallback.

### Anomaly fold-in — `_current_srs_path` never assigned

`core/architect.py` reads `self._current_srs_path` at four sites via `getattr` fallback:

```
:434 — srs_path=getattr(self, "_current_srs_path", "<unknown>"),  # JSON-parse exhaustion
:479 — srs_path=getattr(self, "_current_srs_path", "<unknown>"),  # empty-contexts exhaustion
:491 — srs_path=getattr(self, "_current_srs_path", "<unknown>"),  # generic exception in identify_contexts
:496 — srs_path=getattr(self, "_current_srs_path", "<unknown>"),  # loop fallthrough
```

`grep -n "_current_srs_path" core/architect.py` shows **zero assignment sites**. Every `ArchitectExtractionError` produced by `identify_contexts` carries `srs_path="<unknown>"`. Result: when `/generate-model` fails because the LLM returned malformed JSON for a specific SRS, the error tells the user **nothing** about which SRS was being processed.

### Two problems with the status quo

1. **Silent I/O failure on a paper-reproducibility artifact.** F-13 directly violates AGENTS.md and CLAUDE.md §"Persistent Development Memory".
2. **All `ArchitectExtractionError` messages say `<unknown>` for srs_path** AND **post-fix any `IntermediateSaveError` message would say `<unknown>` too** (per W-5 binding). The two fixes interlock.

---

## Goal

A single atomic behavior change to `core/architect.py` + `core/orchestration/errors.py` + 3 main.py callsite updates such that:

**(a)** Any I/O / serialization failure in `_save_intermediate` raises a new `IntermediateSaveError(PipelineError)` (located in `core/orchestration/errors.py` per W-1) instead of swallowing the exception. The error carries `stage`, `filepath`, `cause`, and `srs_path` (per W-5).

**(b)** `DomainArchitect.identify_contexts`'s retry handler explicitly re-raises `IntermediateSaveError` before the generic `except Exception` (per C-1) so the save failure is not silently rewrapped as `ArchitectExtractionError`.

**(c)** `DomainArchitect.analyze_document(text: str, srs_path: Optional[str] = None)` accepts an optional `srs_path` keyword argument; at the **start of every call** (per W-2), unconditionally assigns `self._current_srs_path = srs_path or "<unknown>"` so reused instances cannot leak a stale previous path.

**(d)** Three main.py call sites updated to pass `srs_path` from their callers:
- `main.py:107` (`generate_domain_model`) — single `srs_path`.
- `main.py:362` (`/generate-model`) — `"; ".join(str(p) for p in request.file_paths)` batch label (display-only, no max-length claim — per W-3).
- `main.py:473` (`/generate-model-stream`) — same as `:362`.

**(e)** New regression tests cover both behaviors AND the real production propagation paths (per C-2 + OQ5). Existing tests that mock `_save_intermediate` via `patch.object` are not affected (still mocked).

---

## Scope and preconditions

**In scope (per AGENTS.md "Smallest correct change"):**
- New `IntermediateSaveError(PipelineError)` exception class in `core/orchestration/errors.py` (per W-1, moved out of `architect.py`).
- `_save_intermediate` rewritten to raise on failure with explicit narrow catch `(OSError, TypeError, ValueError)` + `from e` chain (per N-2).
- `identify_contexts` retry handler updated with `except IntermediateSaveError: raise` before `except Exception` (per C-1).
- `__init__` initializes `self._current_srs_path = "<unknown>"` so the attribute always exists.
- `analyze_document` signature widened by one optional kwarg; unconditional assignment at function start (per W-2).
- 3 main.py callsites updated (per W-3, no truncation).
- 11 new tests (per N-1 + C-2 + W-2 + OQ5):
  - 5 for F-13 (T-SAVE-1..5)
  - 4 for `_current_srs_path` (T-SRS-1..4)
  - 3 for wiring (T-WIRE-MAIN-1..3)

**Out of scope (deferred to later WPs):**
- F-11 parallel Scout rate-limit race (M-L effort, concurrency redesign).
- F-14 SynthesizerEmptyModelError pipeline policy (M effort, contract decision).
- F-21 (vacuous D1 pass via empty `supporting_sentence_ids`) — Codex OQ7 AGREE: deferral correct. Requires Architect prompt + parsing change. WP-CORE-5+.
- All MINOR + TRIVIAL findings (F-12, F-15, F-16, F-17, F-18, F-19, F-20-downgraded).
- All 6 anomalies except the `_current_srs_path` fold-in (`SpecialistShapeError` inheritance, `max_cycles=2` hardcoded, late `section_aware_chunks` import, verifier dict reconstruction from typed Pydantic, hardcoded model names — all observed but no fix this round; "typed Scout dump missing in `scout_fn`" NEW anomaly added to findings doc).
- Cleaning up the 4 `getattr` defensive reads to direct attribute access — Codex OQ4 PARTIALLY: defensible either way; keep them for belt-and-suspenders.

**Preconditions (verified):**
- `_save_intermediate` is private (single-underscore) and called from exactly 4 sites, all within `core/architect.py` (verified by `grep -n "_save_intermediate" core/architect.py`).
- `analyze_document` has **3 production callers** (per Codex OQ6 phrasing): `main.py:107`, `main.py:362`, `main.py:473`. Tests call it too (test_architect_facade.py et al.) but they pass no srs_path → "<unknown>", which is the intended fallback.
- Architect's `identify_contexts` retry handler at lines 483-493 catches `ArchitectExtractionError` first (re-raise), then `Exception`. We insert `except IntermediateSaveError: raise` between them.
- Specialist's `extract_per_context_details` already has the `_save_intermediate` call at line 650 OUTSIDE the per-context retry `try/except` block — clean propagation. Verified.
- Scout's `_save_intermediate` call at line 236 is in DEAD legacy `extract_domain_sentences` (only `tests/test_architect_helpers.py:119` exercises it). Fix is cosmetic for that path.
- `IntermediateSaveError` subclasses `PipelineError` (per W-1) — chosen for taxonomy consistency with `ArchitectExtractionError`, `SpecialistFailureError`, `SynthesizerEmptyModelError` already in `core/orchestration/errors.py`.

---

## Chosen approach

### Single atomic behavior commit

Following WP-CORE-3's atomic-GREEN pattern (parser + helper + callsite migration in one commit), WP-CORE-4 ships one GREEN-atomic commit that touches `core/architect.py` + `core/orchestration/errors.py` + `main.py` together. Test-first discipline preserved by RED having all 11 tests — all red until GREEN lands all production-code changes.

### F-13 + W-1 + W-5 implementation sketch

`core/orchestration/errors.py` (NEW class added alongside existing P3 pipeline errors):

```python
class IntermediateSaveError(PipelineError):
    """Raised when stage diagnostic JSON cannot be persisted.

    Subclasses PipelineError per orchestration error-taxonomy convention.
    Carries the stage label, the intended filepath, the wrapped cause, and
    the SRS path (or batch label) being processed. Per AGENTS.md
    "no silent degradation": stage diagnostic artifacts are EMSE
    reproducibility evidence; silent loss is a methodology gap.
    """

    def __init__(
        self,
        stage: str,
        filepath: str,
        cause: Exception,
        srs_path: str = "<unknown>",
    ):
        self.stage = stage
        self.filepath = filepath
        self.cause = cause
        self.srs_path = srs_path
        super().__init__(
            f"Failed to save intermediate output for stage='{stage}' "
            f"at '{filepath}' (srs={srs_path}): {type(cause).__name__}: {cause}"
        )
```

`core/architect.py` (`_save_intermediate` rewritten):

```python
def _save_intermediate(self, stage: str, data: Dict[str, Any]):
    """Save intermediate pipeline output to JSON file.

    Raises IntermediateSaveError on any I/O or serialization failure.
    Per AGENTS.md "no silent degradation": stage diagnostic artifacts are
    EMSE reproducibility evidence; silent loss is a methodology gap.
    """
    filename = f"{self.run_timestamp}_{stage}.json"
    filepath = os.path.join(INTERMEDIATE_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as e:
        raise IntermediateSaveError(
            stage=stage,
            filepath=filepath,
            cause=e,
            srs_path=getattr(self, "_current_srs_path", "<unknown>"),
        ) from e
    print(f"  💾 Saved intermediate output: {filename}")
```

The `print` after `json.dump` moves outside the try block so a successful save is logged exactly once and a failed save doesn't print success before raising.

### C-1 — `identify_contexts` retry handler must re-raise `IntermediateSaveError`

`core/architect.py:483-493` (current):

```python
except ArchitectExtractionError:
    raise
except Exception as e:
    if not self._is_quota_error_and_backoff(e, retry):
        print(f"   [WARN] Context identification error: {e}")
        if retry >= 4:
            self._report_progress("Architect", "error", str(e), 100)
            raise ArchitectExtractionError(
                srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                message=f"Architect failed with {type(e).__name__}: {e}",
            ) from e
```

Becomes:

```python
except ArchitectExtractionError:
    raise
except IntermediateSaveError:
    raise  # WP-CORE-4: never silently rewrap save failures as ArchitectExtractionError
except Exception as e:
    if not self._is_quota_error_and_backoff(e, retry):
        print(f"   [WARN] Context identification error: {e}")
        if retry >= 4:
            self._report_progress("Architect", "error", str(e), 100)
            raise ArchitectExtractionError(
                srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                message=f"Architect failed with {type(e).__name__}: {e}",
            ) from e
```

Per Codex C-1: without this re-raise, an `IntermediateSaveError` raised at lines 449/462 (during a successful LLM call's save step) would be caught by `except Exception` at line 485, classified as a non-quota-error, printed as a warning, retried 4 more times (each failing the save again), and finally rewrapped as `ArchitectExtractionError("Architect failed with IntermediateSaveError: ...")`. The user never sees the real cause. **Without this fix, the entire WP-CORE-4 acceptance criterion fails silently for the Architect stage.**

Specialist's `extract_per_context_details` does NOT need an equivalent guard — its `_save_intermediate` call at line 650 is OUTSIDE the per-context retry try-block (verified by reading lines 595-660).

### W-2 + Anomaly fold-in implementation sketch

```python
# In DomainArchitect.__init__ (after line 117 os.makedirs)
self._current_srs_path: str = "<unknown>"  # set by analyze_document(srs_path=...)

# Widened analyze_document signature
def analyze_document(
    self,
    text: str,
    srs_path: Optional[str] = None,
) -> DomainModel:
    """Run the 5-stage pipeline on raw SRS text and return a typed DomainModel.

    Args:
        text: SRS document text (already parsed by SRSDocumentParser).
        srs_path: Optional source path label for error messages. Single file
            path for lifespan boot, "; "-joined for batch endpoints. Defaults
            to "<unknown>" when not supplied (e.g., direct internal calls
            from tests).
    """
    # Per W-2: unconditional assignment guards against stale path on instance reuse.
    self._current_srs_path = srs_path or "<unknown>"
    # ... rest unchanged
```

The 4 existing `getattr(self, "_current_srs_path", "<unknown>")` reads are **left as-is** (Codex OQ4 PARTIALLY — keeping them is defensible). The `__init__` assignment + W-2 unconditional re-assignment make them redundant, but removing the `getattr` would mean trusting the assignment in all places it's reached from; preserving them costs nothing and is defensive against future code paths.

### W-3 — Per-callsite policy in main.py (anomaly fold-in)

| line | callsite | new code | rationale |
|---|---|---|---|
| `main.py:107` | `generate_domain_model(srs_path)` lifespan single-file | `architect.analyze_document(text=raw_text, srs_path=srs_path)` | Single file in scope; pass through directly. |
| `main.py:362` | `/generate-model` batch endpoint | `architect.analyze_document(text=combined_text, srs_path="; ".join(str(p) for p in request.file_paths))` | Multi-file batch; join paths with `; ` separator for human-readable error messages. **No max-length truncation per W-3** — the joined label is display-only; a 100-file batch produces a long error string and that's acceptable. |
| `main.py:473` | `/generate-model-stream` batch endpoint | Same as `:362` | Same batch shape. |

---

## Red-phase tests (TDD; written before any production change)

**11 tests total**, organized by concern. All written against existing test conventions (`tests/test_architect_*.py`, `tests/test_main_wiring.py` shape, or new `tests/test_intermediate_save.py` — TBD per plan-phase preference).

### F-13 — `_save_intermediate` raise behavior (5 tests)

**T-SAVE-1 — happy path:** `_save_intermediate("test_stage", {"x": 1})` writes valid JSON file to `INTERMEDIATE_DIR`, content round-trips via `json.load`. Asserts file exists, parsed content matches input. Cleanup via `tmp_path` fixture or explicit `os.remove`.

**T-SAVE-2 — filesystem failure raises IntermediateSaveError:** Patch `builtins.open` (within the `core.architect` module namespace) to raise `PermissionError("read-only")`. Assert `pytest.raises(IntermediateSaveError) as exc_info`. Verify `exc_info.value.stage == "test_stage"`, `exc_info.value.cause.__class__ is PermissionError`, `isinstance(exc_info.value, PipelineError) is True`, and the error message includes `srs_path="<unknown>"` (since architect was constructed without analyze_document call).

**T-SAVE-3 — non-serializable data raises IntermediateSaveError:** Build a `data` dict containing a non-serializable object (e.g., `{"obj": object()}`). Assert `pytest.raises(IntermediateSaveError)` and `exc_info.value.cause.__class__ is TypeError`.

**T-SAVE-4 — failure inside `identify_contexts` propagates IntermediateSaveError, NOT ArchitectExtractionError (per C-2):** Construct `DomainArchitect()`. Patch `_save_intermediate` to raise `IntermediateSaveError(...)` directly (bypassing the actual open/dump). Patch the LLM client to return a valid JSON response so `identify_contexts` reaches the save step. Assert `pytest.raises(IntermediateSaveError)` — explicitly NOT `ArchitectExtractionError`. This is the critical regression that prevents the false-green from Codex C-1.

**T-SAVE-5 — failure inside `extract_per_context_details` propagates cleanly (Specialist path):** Same shape as T-SAVE-4 but exercise Specialist. Patch `_save_intermediate` to raise on the Specialist stage; patch LLM client to return valid per-context payloads. Assert `pytest.raises(IntermediateSaveError)`.

### `_current_srs_path` fold-in — error message carries real path (4 tests)

**T-SRS-1 — `analyze_document(srs_path=...)` assigns attribute:** Construct `DomainArchitect()` (verify `_current_srs_path == "<unknown>"` from `__init__`); call `arch.analyze_document(text="...", srs_path="/path/to/srs.docx")` with mocked stage methods so the call completes synchronously; assert `arch._current_srs_path == "/path/to/srs.docx"`.

**T-SRS-2 — `analyze_document(srs_path=None)` resets to `<unknown>`:** Construct `DomainArchitect()`; call `arch.analyze_document(text="...")` (no srs_path); assert `arch._current_srs_path == "<unknown>"`.

**T-SRS-3 — `ArchitectExtractionError` carries assigned path:** Construct `DomainArchitect()`, manually set `arch._current_srs_path = "/p/foo.docx"`, force `identify_contexts` to exhaust retries (mock client to return JSON parse failure 5 times), assert resulting `ArchitectExtractionError.srs_path == "/p/foo.docx"` (NOT `"<unknown>"`).

**T-SRS-4 — instance reuse resets path (per W-2):** Construct one `DomainArchitect()`. Call `arch.analyze_document(text="...", srs_path="/p/A.docx")` (mocked stages). Then call `arch.analyze_document(text="...")` again (no srs_path). Assert second-call `arch._current_srs_path == "<unknown>"`, NOT `"/p/A.docx"`. Prevents the stale-path leak Codex W-2 flagged.

### Wiring (smoke) — main.py callsites pass srs_path through (3 tests)

**T-WIRE-MAIN-1 — `generate_domain_model` forwards srs_path:** Monkeypatch `DomainArchitect.analyze_document` to a stub that records the `srs_path` kwarg; call `main.generate_domain_model("/tmp/test.docx")` with a stubbed parser returning a non-empty string; assert the recorded `srs_path == "/tmp/test.docx"`.

**T-WIRE-MAIN-2 — `/generate-model` forwards joined srs_path (per OQ5):** Monkeypatch `DomainArchitect.analyze_document` to record `srs_path`. Construct a `GenerateModelRequest` with `file_paths=["/tmp/a.docx", "/tmp/b.docx"]`. Call the endpoint function directly (or via FastAPI TestClient — pick the lightest approach). Assert recorded `srs_path == "/tmp/a.docx; /tmp/b.docx"`.

**T-WIRE-MAIN-3 — `/generate-model-stream` forwards joined srs_path (per OQ5):** Same shape as T-WIRE-MAIN-2 but exercise the streaming endpoint. Since the endpoint spawns a worker thread, the monkeypatch must capture from the thread; use a `threading.Event` or `queue.Queue` to synchronize the assertion. Assert recorded `srs_path == "/tmp/a.docx; /tmp/b.docx"`.

---

## Acceptance criteria

**Behavior (primary):**
- `pytest -m "not integration" --tb=no -q` returns `≥ 332 passed, 31 deselected` (baseline 321 + 11 new).
- All 11 new tests pass.
- Baseline 321 tests still pass (no regression).

**Behavior (negative — verified via test):**
- A failing save during real `analyze_document` (Architect or Specialist stage) now aborts with `IntermediateSaveError` (NOT `ArchitectExtractionError` per C-1 + C-2).
- An `IntermediateSaveError` raised during `/generate-model` carries `srs_path="; "-joined file_paths` (verified via T-WIRE-MAIN-2 + T-SAVE-2 message format).
- An `ArchitectExtractionError` raised inside `analyze_document(srs_path="X")` carries `srs_path="X"`.
- Instance reuse: `analyze_document(srs_path="A")` then `analyze_document()` second call has `_current_srs_path == "<unknown>"` (per W-2 + T-SRS-4).

**Grep cleanup (secondary verification, not primary acceptance):**
- `grep -n "except Exception" core/architect.py` returns at most the original count (no new bare-Exception catches introduced).
- `grep -n "self._current_srs_path =" core/architect.py` shows ≥ 2 assignment sites (was 0 — one in `__init__`, one at top of `analyze_document`).
- `grep -n "srs_path=" main.py` shows ≥ 3 sites passing srs_path to architect (was 0).
- `grep -n "IntermediateSaveError" core/orchestration/errors.py` finds the class definition (was 0).
- `grep -n "IntermediateSaveError" core/architect.py` shows ≥ 2 references: the raise in `_save_intermediate` + the re-raise in `identify_contexts` (per C-1).

**Reviewability:**
- One RED commit (tests only, failing).
- One GREEN-atomic commit (production + tests now passing).
- Optional DOC commit (dev_doc + audit state).
- Optional planning commit (spec + plan into git history).
- Each commit gated on `pytest -m "not integration"` ≥ baseline after that commit (RED: failing tests are expected red signal; GREEN: 332).
- No `--no-verify`. No silent fallbacks. Conventional Commits + Claude trailer.

---

## Implementation order (RED → GREEN → DOC)

1. **RED commit** (`test(architect): WP-CORE-4 red-phase tests for save observability + srs_path propagation`)
   - Add 11 new tests across (a) new `tests/test_intermediate_save.py` for T-SAVE-1..5, (b) `tests/test_architect_helpers.py` or new file for T-SRS-1..4, (c) `tests/test_main_wiring.py` for T-WIRE-MAIN-1..3.
   - Run: 11 expected failures (`IntermediateSaveError` not defined, `analyze_document` doesn't accept `srs_path`, main.py callsites don't pass srs_path, etc.). Baseline-other tests still pass.
   - Acceptance: pytest reports 11 failed, 321 passed.

2. **GREEN-atomic commit** (`fix(architect, main, orchestration): WP-CORE-4 IntermediateSaveError + srs_path propagation`)
   - Add `IntermediateSaveError(PipelineError)` to `core/orchestration/errors.py` (per W-1).
   - Rewrite `_save_intermediate` in `core/architect.py` to raise on failure with narrow catch + srs_path field (per W-5).
   - Insert `except IntermediateSaveError: raise` in `identify_contexts` retry handler (per C-1).
   - Initialize `self._current_srs_path = "<unknown>"` in `__init__`.
   - Widen `analyze_document(text, srs_path=None)` signature; unconditionally assign at function start (per W-2).
   - Update 3 main.py callsites to pass `srs_path` (per W-3, no truncation).
   - Run: 332 passed, 31 deselected.

3. **DOC commit** (`chore(artifacts): WP-CORE-4 dev_doc + audit state update`)
   - `development_docs/WP-CORE-4-intermediate-save-observability.md`.
   - `development_docs/INDEX.md` ACTIVE row update.
   - `.planning/pipeline_audit/{CURRENT,improvements_backlog,decision_log}.md` updates.
   - Iteration-3 handoff doc.
   - No code touched.

4. **Planning commit** (`chore(planning): WP-CORE-4 spec + plan into git history`)
   - Commit this spec + plan file into git history (matches WP-CORE-3 pattern).

---

## Risk register

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| **R-1:** Existing test `patch.object(arch, "_save_intermediate")` mocks bypass the new raise; they continue to work. | low | — | Verified by manual review of 4 patch sites; mocks replace the method entirely. |
| **R-2:** A `_save_intermediate` failure in production now aborts the run instead of completing with corrupt diagnostics. This is the desired behavior change per AGENTS.md — but it is a **user-facing behavior change** and should be documented in the handoff. | high (by design) | medium (user-visible) | Dev doc records the behavior delta. Per AGENTS.md "Error handling: explicit failure", silent degradation was wrong; the fix surfaces the failure where it always belonged. |
| **R-3:** `srs_path` joined with `"; "` could collide with a legitimate semicolon in a filename. | very low | low | Path joining is for human-readable error messages, not parsing. Worst case: a confusing error message — never a runtime failure. |
| **R-4:** T-SAVE-4 / T-SAVE-5 mock the LLM client; setting up the mock LLM response shape (Pydantic `LLMResponseAdapter`) might be fragile and require deep stubbing. | medium | low | Reuse the mock-LLM patterns from `test_architect_extraction_error.py` (which already mock the same client). If too fragile, replace with a higher-level monkeypatch of `_extract_sentences_from_chunk` / `identify_contexts` internals to short-circuit to the save step. |
| **R-5:** T-WIRE-MAIN-3 (`/generate-model-stream`) needs thread synchronization to capture the `srs_path` kwarg from the worker thread. | medium | low | Use `threading.Event` or `queue.Queue` to block the main test thread until the worker records the kwarg, with a 5-second timeout. If still too fragile, mark as expected-skip and document. |
| **R-6:** A FastAPI TestClient call to `/generate-model` triggers the lifespan handler, which would try to load a real SRS — this would slow tests and require mocking the lifespan. | medium | medium | Call the endpoint function directly (not via TestClient). FastAPI endpoint functions are plain Python coroutines/functions; we can construct a `GenerateModelRequest` and call the endpoint directly with mocks. |
| **R-7:** Test count drift: spec says 11, RED commit might end up with 10 or 12 if a test is split or merged. | low | low | Acceptance criterion is `≥ 332 passed`, not `==` — drift in either direction (smaller is bad, larger is fine) is detected immediately. |

---

## Open design questions (now resolved by Codex review)

1. **`IntermediateSaveError` location:** ✅ `core/orchestration/errors.py` (Codex W-1).
2. **Exception base class:** ✅ `PipelineError` (Codex OQ1 DISAGREE on `OSError`).
3. **Failure propagation policy:** ✅ Hard-raise on every save failure. Architect retry handler explicitly re-raises (Codex C-1).
4. **`srs_path` in error message:** ✅ Include in `IntermediateSaveError` via `self._current_srs_path` lookup at raise time (Codex W-5).
5. **Endpoint wiring tests:** ✅ Add T-WIRE-MAIN-2 + T-WIRE-MAIN-3 (Codex OQ5 DISAGREE with deferral).
6. **Cleaning up `getattr` reads:** ✅ Kept as belt-and-suspenders (Codex OQ4 PARTIALLY — defensible either way).
7. **F-21 deferral:** ✅ Defer to WP-CORE-5+ (Codex OQ7 AGREE).

---

## Cross-references

- Parent finding: `.planning/pipeline_audit/findings/architect.md` §F-13 + Anomalies §1 (`_current_srs_path` never set)
- New finding added during spec drafting: `.planning/pipeline_audit/findings/architect.md` §F-21 (MAJOR, deferred)
- Charter rules cited: AGENTS.md "Error handling: explicit failure"; CLAUDE.md §"Persistent Development Memory" + §"intermediate JSON dumps"
- Sibling iterations: WP-CORE-2 (`25e6880`), WP-CORE-3 (`daefeb0`)
- Deferred siblings: F-14 (synthesizer empty model), F-11 (parallel Scout race), F-20-downgraded (token tracker docs gap), F-21 (vacuous D1 pass)
- Codex review: raw output recorded in `decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-4`.

---

**v2 disposition complete. Zero deferred WARNs — all 7 inline. Ready for plan-phase.**
