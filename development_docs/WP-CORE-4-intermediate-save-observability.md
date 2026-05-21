# WP-CORE-4 — Intermediate-save observability + error context propagation

**Status:** SHIPPED 2026-05-21 (iteration 3 of Domain Pipeline Hardening Loop)
**Branch:** `main` (not pushed; per loop rule, only push on explicit user instruction)
**Commit SHAs:**
- RED phase: `0023fa2` — 9 expected failures + 2 invariant tests (T-SAVE-1 happy path, T-SRS-3 getattr fallback)
- GREEN-atomic phase: `02e0fe9` — `IntermediateSaveError(PipelineError)` + `_save_intermediate` raise rewrite + `identify_contexts` re-raise guard + `_current_srs_path` propagation through `analyze_document` signature + 3 main.py callsite updates
- DOC phase (this entry's commit): recorded after this doc is staged

**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md` (v2 — Codex xhigh reviewed: 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ all handled inline, zero deferred)
**Plan:** `docs/superpowers/plans/2026-05-21-wp-core-4-intermediate-save-observability.md`
**Audit finding:** `.planning/pipeline_audit/findings/architect.md` F-13 (MAJOR) + Anomalies §1 (`_current_srs_path` never assigned)
**Sibling WPs:** [[WP-CORE-2-reference-truncate-fix]] (iteration 1, ingestion); [[WP-CORE-3-empty-input-contract]] (iteration 2, ingestion). WP-CORE-4 is the first orchestrator-layer iteration after two ingestion-layer wins.

---

## TL;DR

`DomainArchitect._save_intermediate` previously swallowed every I/O and serialization failure with a `print` + `continue`, silently corrupting the EMSE-reproducibility intermediate JSON artifact chain. WP-CORE-4 introduces `IntermediateSaveError(PipelineError)` (in `core/orchestration/errors.py`) and rewrites `_save_intermediate` to raise it. `identify_contexts`'s broad `except Exception` retry handler is augmented with `except IntermediateSaveError: raise` (per Codex CRITICAL-1) so save failures aren't silently rewrapped as `ArchitectExtractionError`. The companion anomaly fold-in fixes `_current_srs_path` (4 reads via `getattr` fallback, 0 assignment sites): `__init__` initializes the attribute to `"<unknown>"`; `analyze_document` gains an `Optional[str]` `srs_path` kwarg and unconditionally reassigns at function start (per Codex WARN-2 — guards against stale path on instance reuse). Three `main.py` callsites pass the SRS path (single path for lifespan, `"; "`-joined for batch endpoints). The two observability fixes interlock: `IntermediateSaveError` carries `srs_path` populated from `self._current_srs_path` at raise time, so endpoint users now see which SRS was being processed when a save fails. Pytest baseline: **321 → 332 passed, 31 deselected** (+11 tests, 0 regressions).

---

## Motivation

Iteration-3 close-lookup of `core/architect.py` (923 LOC) + `core/orchestration/{pipeline,errors}.py` produced `.planning/pipeline_audit/findings/architect.md` with 11 findings: 4 MAJOR (F-11 parallel Scout race, F-13 silent save swallow, F-14 SynthesizerEmptyModelError pipeline escape, F-21 NEW vacuous D1 pass via empty `supporting_sentence_ids`), 6 MINOR (F-12, F-15, F-17, F-18, F-19, F-20-downgraded after verification), 1 TRIVIAL (F-16 dead `_split_text_into_chunks`), plus 6 anomalies including the `_current_srs_path` never-assigned latent bug.

F-13 was selected for WP-CORE-4 over F-11, F-14, F-21 for three reasons:

1. **Smallest correct change** (AGENTS.md). F-13 is a single method (`_save_intermediate`, 4 callsites all inside `core/architect.py`) plus a new exception class. F-11 (parallel Scout race) requires concurrency redesign; F-14 (synthesizer empty model) requires a contract policy decision; F-21 (vacuous D1) requires Architect-prompt + parsing changes — all M-L effort.
2. **EMSE-reproducibility-blocking**. Per CLAUDE.md §"Persistent Development Memory" the intermediate JSON dumps under `core/intermediate/` are the methodology evidence chain for any RQ1-RQ4 result. A silent save failure means a run completes successfully (returns a `DomainModel`) but the lineage is unrecoverable. Worse, transient `OSError` at Architect-stage corrupts the whole diagnostic chain without the user noticing.
3. **Anomaly bundling synergy**. The `_current_srs_path` anomaly is a 1-line bug (attribute never assigned, all error messages say `"<unknown>"`) with the same observability theme as F-13. Codex review WARN-5 surfaced that `IntermediateSaveError` would itself need to know which SRS failed — bundling lets the same code change satisfy both fixes via a single error-message format.

The status quo for `_save_intermediate`:

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

That `except Exception` clause is a textbook silent fallback (AGENTS.md "Error handling: explicit failure. No empty try/except, no silent degradation, no permissive fallbacks during development. Convert exceptions, add context, or rethrow"). The 4 callsites — Scout (line 236, dead from production per Codex WARN-4), Architect dict-shaped (line 449), Architect list-shaped (line 462), Specialist (line 650) — could not tell a failed save from a successful one. The post-WP-01a P3 architecture has a typed exception class `PipelineError` (in `core/orchestration/errors.py`) that this case should subclass for taxonomy consistency.

For the `_current_srs_path` anomaly, `getattr(self, "_current_srs_path", "<unknown>")` reads occur at lines 434, 479, 491, 496 of `core/architect.py`. `grep -nE "self\._current_srs_path[[:space:]]*=" core/architect.py` returns zero assignment sites. The fallback was clearly *intended* to be a defensive guard against assignment failure; in practice the assignment is missing entirely. Every `ArchitectExtractionError` produced by `identify_contexts` carries `srs_path="<unknown>"`, which is useless context for the user trying to debug which document failed.

---

## Architectural decisions

### D1 — `IntermediateSaveError(PipelineError)` in `core/orchestration/errors.py`

Per Codex OQ1 + W-1. Original spec draft had `IntermediateSaveError(OSError)` colocated in `core/architect.py`. Codex correctly flagged that `core/orchestration/errors.py` already establishes `PipelineError` as the base for P3 pipeline failures; sibling classes include `ArchitectExtractionError`, `SpecialistFailureError`, `SynthesizerEmptyModelError`, `RefinementExhaustedError`, `InsufficientGroundingError`. Subclassing `OSError` would violate this taxonomy. Subclassing `PipelineError` preserves consistency: any handler that catches `PipelineError` (e.g., a future top-level orchestrator failure_log writer) catches save failures too. The exception carries `stage`, `filepath`, `cause`, and `srs_path` (the latter per W-5 — see D5).

### D2 — Narrow catch `(OSError, TypeError, ValueError)` excluding `JSONDecodeError`

Per Codex N-2. Spec v1 listed `JSONDecodeError` in the catch rationale; Codex pointed out `JSONDecodeError` is a *decode* exception (`json.loads`), not an *encode* one (`json.dump`). Encoder-side failures are `TypeError` (non-serializable object) and `ValueError` (circular reference, invalid `indent`/`separators`). `OSError` covers filesystem (`PermissionError`, `FileNotFoundError`, disk-full).

The deliberately narrow catch excludes `KeyboardInterrupt` / `SystemExit` (which are `BaseException`, not `Exception` — so the prior `except Exception` already excluded them, but Codex preferred the narrow form for AGENTS.md "explicit failure" alignment).

### D3 — `except IntermediateSaveError: raise` in `identify_contexts` retry handler

Per Codex CRITICAL-1. The retry handler at `architect.py:483-498` catches `ArchitectExtractionError` explicitly (`raise`) then catches `Exception` broadly (classifies as quota-retry or rewraps as `ArchitectExtractionError`). Without an explicit re-raise for `IntermediateSaveError`, a save failure during a successful Architect LLM call would be:

1. Caught by `except Exception` at line 485.
2. Classified as non-quota-error.
3. Printed as a warning.
4. Retried 4 more times (each retry's save fails identically).
5. Rewrapped as `ArchitectExtractionError("Architect failed with IntermediateSaveError: ...")`.

The user would see the broad endpoint error from `main.py:424-431` — never the real `IntermediateSaveError`. **Without this fix, WP-CORE-4's primary acceptance criterion would silently fail for the Architect stage.** T-SAVE-4 was added to RED to lock this regression: it asserts `pytest.raises(IntermediateSaveError)` specifically and not `ArchitectExtractionError`.

Specialist's `extract_per_context_details` does NOT need an equivalent guard — its `_save_intermediate` call at line 650 is OUTSIDE the per-context retry try-block. Verified via re-reading lines 595-660 during spec drafting.

### D4 — Unconditional `self._current_srs_path = srs_path or "<unknown>"`

Per Codex WARN-2. Spec v1's sketch was `if srs_path is not None: self._current_srs_path = srs_path`. Codex pointed out that production currently constructs a fresh `DomainArchitect` per call (`main.py:106`, `:361`, `:472`), so the stale-path-on-reuse bug isn't reachable today — but the API contract should not depend on that, and any future refactor that reuses an instance (e.g., a connection-pooled architect for amortized rate-limit cost) would leak the previous run's path into the new run's error messages.

The fix: at the start of every `analyze_document`, **unconditionally** assign `self._current_srs_path = srs_path or "<unknown>"`. A second call with `srs_path=None` resets to the default. T-SRS-4 was added to RED to lock this: it calls `analyze_document(srs_path="/p/A.docx")` then `analyze_document()` and asserts the second call's `_current_srs_path == "<unknown>"`.

### D5 — `IntermediateSaveError` carries `srs_path` via `getattr` at raise time

Per Codex WARN-5. Without this, endpoint users would see `IntermediateSaveError("Failed to save intermediate output for stage='2_architect' at '...'")` — but no indication of which document was being processed. With it, the message reads `"... (srs=/path/to/srs.docx)"`. The two observability fixes (F-13 + `_current_srs_path` anomaly) interlock at the error-message level: D4's unconditional assignment guarantees `self._current_srs_path` is set when D2's narrow catch fires; the `IntermediateSaveError` constructor reads it via `getattr(self, "_current_srs_path", "<unknown>")` (defensive — the attribute SHOULD be present from `__init__`, but the `getattr` guards against an edge case where `_save_intermediate` is called on an architect built via `__new__` without `__init__`, e.g., in tests).

### D6 — "; "-joined batch label, display-only, no truncation

Per Codex WARN-3. Spec v1 claimed the joined label was "bounded by request validation"; Codex correctly noted `GenerateModelRequest.file_paths` is `List[str]` with no visible max length. WP-CORE-4 drops the bounded claim. The label is display-only — the user sees a long error message for a 100-file batch, which is acceptable for an error path. KISS over premature truncation (AGENTS.md "no speculative generalization"). A future WP could add an env-var-controlled truncation if EMSE reviewers flag it.

### D7 — Kept the 4 `getattr` defensive reads as belt-and-suspenders

Per Codex OQ4 PARTIALLY. The `__init__` initialization (D4 prep) makes `self._current_srs_path` always exist, so the 4 `getattr(self, "_current_srs_path", "<unknown>")` reads in `identify_contexts` (lines 434, 479, 491, 496) are technically redundant. Codex noted cleaning them up to direct attribute access is defensible under "explicit failure" rules. Kept them as-is under "smallest correct change": removing 4 read sites for cosmetic purity is out of WP-CORE-4 scope, and the `getattr` form is defensive against any future code path that bypasses `analyze_document` (which would skip the unconditional assignment).

### D8 — `_save_intermediate` print moved out of try block

Subtle cleanup folded in: previously the `print(f"  💾 Saved intermediate output: {filename}")` line lived inside the try block, *before* the `except`. After the rewrite, the print fires only on the success path (after json.dump returns cleanly, before any exception could be raised). A failed save now never prints a misleading "Saved …" message before raising. Codex did not flag this but the fix is consistent with the "explicit failure" theme.

---

## File-level changes

| file | change | LOC delta |
|---|---|---|
| `extension/backend/core/orchestration/errors.py` | NEW class `IntermediateSaveError(PipelineError)` — 4-arg constructor, formatted message including srs_path. | +30 |
| `extension/backend/core/architect.py` | (a) Import `IntermediateSaveError` from `core.orchestration.errors`. (b) `__init__`: 1 new line `self._current_srs_path: str = "<unknown>"`. (c) `_save_intermediate` rewrite: explicit narrow catch + raise; print moved out of try. (d) `identify_contexts` retry handler: `except IntermediateSaveError: raise` inserted before `except Exception`. (e) `analyze_document` signature widens by `srs_path: Optional[str] = None`; new docstring; 2 new lines `self._current_srs_path = srs_path or "<unknown>"` at function start. | +48 / −12 |
| `extension/backend/main.py` | 3 callsite updates: `generate_domain_model` lifespan (single path), `/generate-model` batch (`"; ".join(...)`), `/generate-model-stream` batch (same join). | +12 / −3 |
| `extension/backend/tests/test_intermediate_save.py` (NEW) | T-SAVE-1..5: happy path, filesystem-error raise, non-serializable-data raise, save-failure-inside-identify_contexts propagation (Codex C-2), save-failure-inside-extract_per_context_details propagation. | +175 |
| `extension/backend/tests/test_architect_srs_path.py` (NEW) | T-SRS-1..4: assign-on-srs-path, reset-on-no-srs-path, ArchitectExtractionError carries assigned path, instance reuse resets (Codex W-2). | +135 |
| `extension/backend/tests/test_main_wiring.py` | T-WIRE-MAIN-2 + T-WIRE-MAIN-3 appended: `/generate-model` joined-path forwarding, `/generate-model-stream` joined-path forwarding (drained via `asyncio.run` since `StreamingResponse.body_iterator` is an async iterator). | +131 |

---

## Methodology applied

**TDD with explicit RED-phase commit.** 11 new tests written before any production change. RED commit `0023fa2` recorded a 9-failed / 323-passed state (the 2 invariant tests — T-SAVE-1 file-round-trip and T-SRS-3 getattr-fallback — pass on the pre-GREEN code because they exercise paths that already work; the 9 genuine failures all reference symbols/signatures that don't exist until GREEN).

**Codex xhigh adversarial review.** Spec v1 → Codex review at runtime 2026-05-21 ~10:15 → 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ → spec v2 with every finding disposed inline (zero deferred). Raw review preserved in `.planning/pipeline_audit/decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-4`. Comparison with iteration history:

| iteration | WP | Codex findings | deferred WARNs |
|---|---|---|---|
| 1 | WP-CORE-2 | 0 CRITICAL + 6 WARN | 4 (4/6 WARNs accepted-with-rationale as out-of-scope) |
| 2 | WP-CORE-3 | 2 CRITICAL + 5 WARN | 0 |
| 3 | WP-CORE-4 | 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ | 0 |

WP-CORE-3 and WP-CORE-4 both ship with zero deferred WARNs — the iteration discipline of folding all findings inline at spec time, rather than accepting some as "future work," is now the loop default.

**Single atomic GREEN commit.** Matches WP-CORE-3's pattern: production code + test stub corrections + 3 callsite migrations all in one commit. Test-first preserved because all 11 tests existed in the prior RED commit; GREEN is "turn the lights on" plus minor stub corrections (raising_save signature parity, valid Pydantic stub, async drain) that surfaced when GREEN landed and exposed pre-existing stub bugs.

**Each commit gated on pytest baseline.** RED: 321 baseline + 11 new = 332 tests, 9 fail. GREEN: 332 tests, 0 fail. No `--no-verify`. No silent fallbacks introduced.

---

## Empirical results

- **Pytest delta:** 321 → 332 passed, 31 deselected (`pytest -m "not integration"`). 0 regressions across the existing 321-test baseline.
- **Test-distribution:** 5 tests for F-13 raise path (T-SAVE-1..5), 4 for `_current_srs_path` propagation (T-SRS-1..4), 3 for endpoint wiring (T-WIRE-MAIN-1 pre-existing + T-WIRE-MAIN-2/3 new this WP — note T-WIRE-MAIN-1 was added in WP-CORE-3 for the unrelated `_parse_srs_batch` helper, not WP-CORE-4).
- **LOC delta (production):** +90 net (+78 in `architect.py`+`errors.py`+`main.py` after deletions). `core/architect.py` 923 → 949 LOC (+26 net). `_save_intermediate` grew from 12 to 21 LOC (5 LOC of comment+raise + 2 LOC of error-context binding).
- **Behavior change observable from outside:** a `_save_intermediate` failure under `/generate-model` now produces an HTTP 200 with `{"success": false, "error": "Failed to save intermediate output for stage='2_architect' at '...' (srs=...): PermissionError: ...", ...}` (caught by the broad `except Exception` at `main.py:424-431` and returned as the error body). Previously the request succeeded with corrupt diagnostics on disk and no signal in the response.
- **Cumulative loop progress:** pre-loop baseline 272 (HEAD `3d13f26`'s predecessor) → iteration-1 close 305 (+33) → iteration-2 close 321 (+16) → iteration-3 close 332 (+11). Net loop output: +60 tests across 3 iterations + 3 shipped MAJOR fixes + 11 catalogued findings, 0 regressions.

---

## Limitations + follow-ups

### Limitations

- **`identify_contexts` is the only stage with the re-raise guard.** Scout's legacy `extract_domain_sentences` is dead from `analyze_document` (per Codex WARN-4 verification), and Specialist's save call is outside its retry loop, so no guards are needed there. If a future refactor moves Scout save inside an analogous retry wrapper, the same guard pattern must be added — flagged as a maintenance invariant in the architect.py code review checklist (see "Cross-references" below).
- **`IntermediateSaveError.srs_path` is read at raise time, not stage-execution time.** If a future code path bypasses `analyze_document` and calls `_save_intermediate` directly with a stale `_current_srs_path`, the error message could be misleading. The `getattr(self, "_current_srs_path", "<unknown>")` defensive default in the constructor falls back to `"<unknown>"` if the attribute is missing entirely — but not if it's stale. Acceptable for current scope; documented as an invariant in the `IntermediateSaveError` docstring.
- **Endpoint wiring tests cover the joined-path label but NOT the failure-propagation path through the endpoint.** T-WIRE-MAIN-2 + T-WIRE-MAIN-3 assert that the kwarg is forwarded correctly when the pipeline succeeds. They do not assert what happens when `_save_intermediate` raises inside the endpoint (the broad `except Exception` at `main.py:424-431` swallows it into the success=false response). A future WP could tighten endpoint error-shape testing.

### Follow-ups (deferred)

- **F-11 — parallel Scout rate-limit window race.** Effort M-L; concurrency redesign (semaphore or condition variable). Highest-priority remaining MAJOR.
- **F-14 — `SynthesizerEmptyModelError` escapes pipeline without explicit handler.** Effort M; policy decision (hard-fail vs degrade-best-effort, analogous to `RefinementExhaustedError`).
- **F-21 (NEW from spec drafting) — `ContextHypothesis.supporting_sentence_ids` always empty; D1 verifier passes vacuously.** Effort M; requires Architect prompt + parsing change. Important for EMSE methodology validity (D1 deterministic check is effectively dead code in current state).
- **F-20-downgraded — token tracker thread-safety comment gap.** Effort S; one-line comment at Scout parallel-extract call site documenting that thread-safety is delegated to `TokenTracker._lock`.
- **F-12, F-15, F-17, F-18, F-19** — minor observability and code-clarity gaps; bundle into a single "architect.py polish" WP when the major findings are cleared.
- **F-16 — dead `_split_text_into_chunks`.** TRIVIAL; bundle with similar cleanup tasks.
- **New anomaly discovered:** Scout-stage `_save_intermediate` call in legacy `extract_domain_sentences` is unreachable from `analyze_document` (which uses an inline `scout_fn` that doesn't dump). "Typed Scout dump missing in scout_fn" — observability gap, not behavior bug. Defer.

---

## Cross-references

- **Spec v2:** `docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md`
- **Plan:** `docs/superpowers/plans/2026-05-21-wp-core-4-intermediate-save-observability.md`
- **Audit findings:** `.planning/pipeline_audit/findings/architect.md` §F-13 + Anomalies §1 + (NEW) §F-21
- **Codex review record:** `.planning/pipeline_audit/decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-4` (2026-05-21 14:30)
- **Sibling iterations:** [[WP-CORE-2-reference-truncate-fix]] (iteration 1, ingestion-layer F-5), [[WP-CORE-3-empty-input-contract]] (iteration 2, ingestion-layer F-3 + post-loop-guard latent bug)
- **Charter rules cited:** AGENTS.md "Error handling: explicit failure" + "Smallest correct change"; CLAUDE.md §"Persistent Development Memory" + §"intermediate JSON dumps"
- **Code-review invariant:** Any future retry wrapper introduced for Scout's or any other stage's `_save_intermediate` calls MUST include `except IntermediateSaveError: raise` before any generic `except Exception` — same pattern as `identify_contexts`. Flag in `core/architect.py` review checklist.
- **WP-NEW-B Stage-1 schema_probe** ([[WP-NEW-B-Stage-1-schema-probe.md]]) — sibling EMSE methodology artifact (run manifest format); the run manifest depends on the intermediate JSONs that WP-CORE-4 now reliably saves.

---

**End of WP-CORE-4 doc.**
