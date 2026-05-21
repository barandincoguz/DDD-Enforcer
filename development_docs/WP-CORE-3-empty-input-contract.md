# WP-CORE-3 — Empty-input contract for `SRSDocumentParser.parse_file`

**Status:** SHIPPED 2026-05-21 (iteration 2 of Domain Pipeline Hardening Loop)
**Branch:** `main` (not pushed; per loop rule, only push on explicit user instruction)
**Commit SHAs:**
- RED phase: `91dbeb4` — 15 failing tests (11 parser + 4 wiring)
- GREEN phase (atomic): `daefeb0` — `EmptySRSDocumentError` + parser raise + `_parse_srs_batch` helper + 6 call-site migrations
- DOC phase (this entry's commit): recorded after this doc is staged

**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md` (v2 — Codex xhigh reviewed: 2 CRITICAL + 5 WARN handled inline)
**Plan:** `docs/superpowers/plans/2026-05-21-wp-core-3-empty-input-contract.md`
**Audit finding:** `.planning/pipeline_audit/findings/document_parser.md` F-3 (MAJOR)
**Sibling WP:** [[WP-CORE-2-reference-truncate-fix]] (iteration 1; same file context; reference-heading truncation)

---

## TL;DR

`SRSDocumentParser.parse_file` now raises a new `EmptySRSDocumentError(ValueError)` when the post-processed text is empty, replacing six duplicated empty-string guards across `main.py` (three behavioral shapes, two of them broken). All six call sites migrated to explicit per-path policy: HARD propagate (sync generate flow), SOFT skip+log (RAG init + RAG re-index loops), MIXED batch via new `_parse_srs_batch` helper (skip-and-continue on empty + early-return on read failure + aggregate srs_docs-emptiness check for all-empty). Latent bug folded in: previous post-loop `combined_text.strip()` guards at `/generate-model{,-stream}` were dead code (separator headers made the check always non-empty). Pytest baseline: **305 → 321 passed, 31 deselected** (+16 tests).

---

## Motivation

`parse_file` returned `""` (no exception) for every flavor of empty input — 0-byte TXT, DOCX with zero blocks, PDF with all-empty pages, whitespace-only TXT. Every consumer in `main.py` had to re-implement the empty-string check:

| line | shape | behavior on empty |
|---|---|---|
| `main.py:61` | `if not raw_text.strip(): raise ValueError("Document is empty…")` | HARD |
| `main.py:101` | `if raw_text.strip():` (positive guard, silent skip) | SOFT |
| `main.py:326-330` | post-loop `if not combined_text.strip(): return {"success": False, ...}` | HARD (BROKEN) |
| `main.py:367` | `if raw_text.strip():` | SOFT |
| `main.py:449-451` | same as `:326-330` but inside thread | HARD (BROKEN) |
| `main.py:481` | `if raw_text.strip():` | SOFT |

Three problems:

1. **Leaky contract.** Six duplicated guards across three behavioral shapes — the invariant belongs inside the parser. AGENTS.md "Stable entrypoints; isolate change-prone logic" applies directly.
2. **Silent data swallowing.** SOFT branches dropped empty files from RAG indexing with no log, no metric — unrecorded behavior for EMSE reproducibility manifests.
3. **Indistinguishable failure modes.** The HARD branches used `"Document is empty or could not be parsed"` for two genuinely different failures: read-fail vs read-OK-but-empty.

**Latent bug uncovered during spec drafting:** the post-loop `combined_text.strip()` guards at `:326-330` and `:449-451` were dead code. The batch loop prepends `f"\n\n--- Document: {Path(file_path).name} ---\n\n"` (non-whitespace separator) before each raw_text, so even when every file parsed to `""`, `combined_text.strip()` returned separator-only text — never `""`. The "all documents empty" branch was unreachable. Folded into WP-CORE-3 as a bonus fix.

---

## Architectural decisions

### D1 — Custom exception subclasses `ValueError`

`EmptySRSDocumentError(ValueError)`. Three reasons over a sibling `DomainParserError` hierarchy:

- **Backwards-compat preservation.** Any future caller using `except ValueError` (e.g., lifespan currently has `except Exception`, but if a future caller narrows to `ValueError` for explicit-failure discipline) still catches the new error — no surprise regression.
- **Smallest correct change.** A new hierarchy is speculative generalization (AGENTS.md "no speculative generalization"). The single exception class is sufficient for WP-CORE-3's scope; a broader `DomainParserError` taxonomy can be retro-introduced if a future WP needs it.
- **Type-distinct when explicit handlers want it.** Callers wanting precise control can `except EmptySRSDocumentError` and ignore the broader `ValueError`.

### D2 — Empty check after `_post_process` (not after raw read)

`if not self._post_process(raw):` covers three cases for free:
- 0-byte / whitespace-only inputs (raw read empty).
- Cross-WP integration with [[WP-CORE-2-reference-truncate-fix]]: documents that truncate to empty after `_truncate_at_references` strips everything (e.g., whitespace-padded `References` heading past the 0.5 position guard).
- Documents that other `_post_process` steps reduce to empty (rare but possible — e.g., header/footer pattern matching every line).

Single check, three failure modes. T-EMPTY-9 locks the cross-WP integration.

### D3 — Per-path policy at callers (HARD propagate / SOFT skip+log / MIXED batch)

The spec's per-path policy table makes the contract explicit:

| call site | path class | policy |
|---|---|---|
| `generate_domain_model` (sync) | HARD | Exception propagates; lifespan `except Exception` logs. |
| `initialize_rag` | SOFT | `try/except EmptySRSDocumentError → print + return rag`. |
| `/generate-model` batch | MIXED | Via helper: skip+log empty, return on non-empty exception, aggregate `srs_docs` check. |
| `/generate-model-stream` batch | MIXED | Same helper. |
| RAG re-index loops (both endpoints) | SOFT | Per-file `try/except → print + continue`. |

Rationale for SOFT-skip on RAG: RAG init is best-effort by existing convention (`app_state["rag"] = None` on init failure at `:381` in pre-WP code). Hard-failing the whole `/generate-model` endpoint on a single empty file in the RAG re-index step would be a behavior regression. SOFT-skip with `print` log preserves observability without coupling RAG availability to per-file content.

### D4 — Extracted `_parse_srs_batch` helper

The two batch endpoints (`/generate-model` and `/generate-model-stream`) had near-identical inline loops with separate per-file error variables (`return {dict}` vs `result_holder["error"]`). Extracting a `_parse_srs_batch(parser, file_paths) -> (combined_text, srs_docs, error_dict_or_none)` helper does three things:

- **Centralizes the empty-skip + read-error + aggregate-empty logic** so future changes happen in one place.
- **Provides a testable seam** (response to Codex CRITICAL-2). The helper accepts any `SRSDocumentParser`-shaped argument; tests stub via subclass.
- **Lets the two endpoint call sites differ only in how they surface the error dict** (return vs `result_holder["error"] = error["error"]`), which is the genuine difference between the sync and threaded endpoints.

Helper is module-private (`_` prefix). Not a public API.

### D5 — Atomic GREEN commit (parser raise + helper + all 6 call-site updates in one commit)

Response to Codex WARN-3. Splitting parser raise and call-site migration into separate commits leaves an intermediate commit where the parser raises but the call sites are stale — codebase in an inconsistent state for a non-zero window. Atomic GREEN commit avoids this; test-first discipline is preserved by the RED commit landing all 15 tests first (failing at collection on `ImportError`).

### D6 — Behavior change documented, not silently absorbed

Spec v1 incorrectly claimed "no change to batch atomicity." Codex CRITICAL-1 caught it: pre-WP mixed batches (one empty + one good) silently included the empty's separator-only content in `combined_text`; post-WP mixed batches now succeed cleanly (empty skipped + logged). Spec v2 §"R-5" explicitly documents this as an **intentional behavior change**, not a regression: the new behavior is strictly better observable (per-file empty-skip emits a `⚠️` log line) and the previous behavior was masked by the broken post-loop guard anyway.

---

## File-level changes

| file | change | LOC delta |
|---|---|---|
| `extension/backend/core/document_parser.py` | Add `EmptySRSDocumentError(ValueError)` class; add `if not processed: raise EmptySRSDocumentError(...)` at end of `parse_file`. | +14 LOC |
| `extension/backend/main.py` | Add `Optional, Tuple` to typing imports; add `EmptySRSDocumentError` to `core.document_parser` import; add `_parse_srs_batch` helper (50 LOC); delete sync HARD guard; replace `initialize_rag` SOFT guard with try/except; replace `/generate-model` and `/generate-model-stream` batch loops with helper call (saves ~22 LOC × 2 = ~44 LOC); replace RAG re-index loops in both endpoints with per-file try/except (3 LOC each). | +44 LOC, -89 LOC (net -45) |
| `extension/backend/tests/test_document_parser.py` | Append 11 T-EMPTY-* tests (12 after parametrize expansion). | +109 LOC |
| `extension/backend/tests/test_main_wiring.py` | NEW file: 4 T-WIRE-* tests via `_StubParser(SRSDocumentParser)` subclass. | +86 LOC |

Net source LOC: `+14 + (-45)` = **-31 LOC of production code** despite adding the helper, because the inline batch-loop duplication was retired. Net test LOC: **+195 LOC**.

---

## Methodology applied

- **TDD (RED → GREEN atomic).** RED commit `91dbeb4` adds 15 failing tests; GREEN commit `daefeb0` lands all production code; baseline 305 → 321 in one atomic step. T-EMPTY-9 needed a rewrite mid-GREEN because the WP-CORE-2 position guard intentionally protects single-line `"References\n"` files; the test now uses whitespace-padded content to push References past the 0.5 fraction guard.
- **Codex xhigh adversarial review pre-implementation.** Spec v1 was reviewed by Codex xhigh and returned 2 CRITICAL + 5 WARN findings; spec v2 handled all 7 inline (no WARNs accepted-with-rationale this time, unlike WP-CORE-2 where 4 of 6 WARNs were deferred). Review verbatim is captured in `decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-3`.
- **Per-path policy table.** Codex WARN-1 caught an ambiguity between SOFT-skip and HARD-propagate intent; spec v2 added an explicit policy table that the GREEN commit followed literally.
- **Sole-ingress precondition verified by grep.** Codex WARN-2 raised the question of LLM-layer callers bypassing `parse_file`. Spec §"Scope and preconditions" includes the grep evidence (`git grep -n "analyze_document" extension/backend/` returns only the three known call sites, each fed by `parse_file` upstream); the empty-input invariant is therefore enforceable at the parser layer alone.
- **Sacred baseline preserved.** Each commit gated on `pytest -m "not integration"` ≥ baseline.

---

## Empirical results

- **Test count:** 305 passed → 321 passed (31 deselected). Net +16 tests (T-EMPTY-1..11 with parametrize expansion = 12 + T-WIRE-1..4 = 4).
- **Cleanup checks (all 0 results, confirming complete migration):**
  - `git grep "raw_text.strip()" extension/backend/main.py` → 0
  - `git grep "combined_text.strip()" extension/backend/main.py` → 0
  - `git grep "All documents are empty or could not be parsed" extension/backend/main.py` → 0
  - `git grep "Document is empty or could not be parsed" extension/backend/main.py` → 0
- **Behavior locks (T-WIRE-*):**
  - T-WIRE-1: mixed batch (empty + good) → success, only good file in `srs_docs`.
  - T-WIRE-2: all-empty batch → aggregate error `"All documents were empty after parsing"`.
  - T-WIRE-3: per-file `FileNotFoundError` → batch error `"Failed to parse missing.txt"` (disambiguation locked).
  - T-WIRE-4: `initialize_rag` returns without `index_document` being called.
- **Cross-WP integration (T-EMPTY-9):** whitespace-padded `References` heading at latter half → truncated to empty → `EmptySRSDocumentError` raised. WP-CORE-2 position guard + WP-CORE-3 raise work together as designed.
- **Pyright:** unchanged net diagnostic count on `core/document_parser.py` and `main.py` (pre-existing import-resolution noise from `fastapi`/`pydantic`/`uvicorn`/`pytest`/`docx` resolution + `result_holder["error"]` Literal-vs-None Optional widening — none introduced by this WP).

---

## Limitations and follow-ups

- **F-1, F-2, F-7 still open.** Reader-layer defensive handling (encrypted PDFs, cp1254 binary garbage, DOCX try/except) is independent and out of scope.
- **F-9 (logging module) still open.** WP-CORE-3 uses `print` per existing `main.py` convention; introducing the `logging` module is a separate WP.
- **F-10 (duplicate parse) still open.** Helper extraction did not solve the duplicate I/O (RAG re-index loops still call `parse_file` a second time per file in the same request).
- **Batch atomicity policy for read failures preserved (one read failure kills the batch).** If a future feature wants batch-tolerant read-failure behavior, that's a different scope.
- **No endpoint-level integration test.** T-WIRE-* tests cover the helper + `initialize_rag` via stubs; the actual FastAPI request-shape behavior is unchanged (request/response Pydantic models untouched) but not unit-tested. Integration tests (`pytest -m integration`) remain the canonical place for endpoint-shape coverage.
- **Lifespan error-message refinement (mention "empty SRS" in the lifespan log) deferred to F-9 / observability work.**

---

## Cross-references

- Spec v2: `docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md`
- Plan: `docs/superpowers/plans/2026-05-21-wp-core-3-empty-input-contract.md`
- Audit finding: `.planning/pipeline_audit/findings/document_parser.md` §F-3
- Decision log: `.planning/pipeline_audit/decision_log.md` (entries `D-CODEX-REVIEW-WP-CORE-3` + `D-EMPTY-INPUT-CONTRACT-2026-05-21` + `D-SHIP-WP-CORE-3`)
- Backlog: `.planning/pipeline_audit/improvements_backlog.md` (F-3 → SHIPPED)
- Sibling iteration: [[WP-CORE-2-reference-truncate-fix]]
- AGENTS.md rules cited: "Stable entrypoints, isolate change-prone logic", "no permissive fallbacks during development", "Error handling: explicit failure", "no speculative generalization", "smallest correct change"
- Handoff for iteration 3: `.planning/pipeline_audit/handoff-2026-05-21-<HHMM>.md` (this iteration's closure)
