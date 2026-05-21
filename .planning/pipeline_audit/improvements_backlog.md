# Improvements Backlog — Domain Pipeline Hardening

Each row: `id | component | finding | severity | effort | blast | status`.

- `severity` ∈ {BLOCKER, MAJOR, MINOR, TRIVIAL}
- `effort` ∈ {S (≤2h), M (≤1d), L (>1d)}
- `blast` ∈ {LOCAL, MODULE, PIPELINE, REPO}
- `status` ∈ {OPEN, IN-PROGRESS, SHIPPED, REJECTED, DEFERRED}

## Open

### Ingestion layer (`core/document_parser*.py`)

| id | component | finding | severity | effort | blast | status |
|---|---|---|---|---|---|---|
| F-1 | document_parser_readers.py | `read_pdf` has no defensive handling for encrypted/image-only/malformed PDFs — propagates raw pypdf errors + emits empty string on image-only PDFs (pypdf2:16-19, 22-27). | MAJOR | S | PIPELINE | OPEN |
| F-2 | document_parser_readers.py | `read_txt` silently emits binary garbage when `_looks_like_text` passes a near-binary file under cp1254 (`:92-109`, `:120-126`). | MAJOR | S | PIPELINE | OPEN |
| F-4 | document_parser.py | TOC heuristic anchored to first 120 lines + `cluster < 2` drop; layout-mode PDFs leak TOC entries into Scout (`:81-101`, `:103-117`). | MAJOR (uncertain) | M | PIPELINE | OPEN |
| F-6 | document_parser.py | `_should_merge` only checks `[.!?;:]$`; quote-terminated / bracket-terminated / Unicode-ellipsis lines collapse silently; soft-hyphen vs compound-word hyphen indistinct (`:154-165`). | MINOR | S | LOCAL | OPEN |
| F-7 | document_parser_readers.py | DOCX reader has zero try/except around `docx.Document(file_path)`; `PackageNotFoundError` propagates raw (`:30-31`). | MINOR | S | LOCAL | OPEN |
| F-8 | document_parser_readers.py | No explicit XXE / external-entity hardening on lxml XML parsing (defense-in-depth gap visible to EMSE reviewers) (`:31`). | MINOR (uncertain) | S | REPO | OPEN |
| F-9 | document_parser.py + readers | Zero logging anywhere; PDF layout→plain downgrade invisible to WP-NEW-B run manifest. | MINOR | S | PIPELINE | OPEN |
| F-10 | document_parser.py + `main.py:366,480` | Same SRS re-parsed twice per `/generate-model` request (Architect input + RAG indexing) — duplicate I/O, no memoization. | TRIVIAL | M | LOCAL | OPEN |

### Orchestrator layer (`core/architect.py` + `core/orchestration/*.py`)

| id | component | finding | severity | effort | blast | status |
|---|---|---|---|---|---|---|
| F-11 | core/architect.py | Parallel Scout (`scout_max_workers > 1`) rate-limit reentrancy: lock window in `_wait_for_rate_limit` (`:141-150`) allows microslip on consecutive workers — cumulative quota loss under N workers can be `(N-1) * (min_delay / N)`. Needs condition-variable or semaphore. **DORMANT (2026-05-21 11:08)**: parallel Scout path (`extract_domain_sentences` `ThreadPoolExecutor` at line 228) is dead from production — `analyze_document.scout_fn` at lines 757-774 calls only `section_aware_chunks()` (no LLM, no `_wait_for_rate_limit`). Codex xhigh review on WP-CORE-5 spec also surfaced that the proposed reservation fix paces returns not sends, so a correct fix needs send-gating (out of "smallest correct change" envelope). Sequential primitive slip (~1-5 ms per gap) within 6 s buffer in practice. Reopen when Scout is rewired or `section_aware_chunks` gains an LLM call. See `decision_log.md` D-CODEX-REVIEW-WP-CORE-5. | MAJOR | M-L | PIPELINE | OPEN (DORMANT) |
| F-12 | core/architect.py | Specialist shape-error retry token tracking gap (`:604-635`): on retry-N→retry-N+1 success, previous retry tokens are not tracked because `token_tracker.track_api_call` runs after validation; run manifest understates token cost. | MINOR | S | LOCAL | OPEN |
| F-15 | core/orchestration/pipeline.py | Refiner exhaustion fallback (`:75-78`) logs only exception type, not residual verifier issues; observability gap. | MINOR | S | LOCAL | OPEN |
| F-16 | core/architect.py | Dead code: `_split_text_into_chunks` (`:249-265`) never called — Scout uses `section_aware_chunks`. | TRIVIAL | S | NONE | OPEN |
| F-17 | core/architect.py | Stage config validation deferred to first LLM call (`:99,303,403,586`); should validate at `__init__` for early error surfacing. | MINOR | S | LOCAL | OPEN |
| F-18 | core/architect.py | Synthetic `f"{n} context"` descriptions written to intermediate JSON (`:757-759`) mismatch the LLM-enriched descriptions stored downstream by synthesizer; debugging confusion. | MINOR | S | LOCAL | OPEN |
| F-19 | core/architect.py | Exponential backoff comment (`:171`) says "15s, 30s, 60s, 120s" but actual progression goes to 300s; comment-code mismatch. | MINOR | S | NONE | OPEN |
| F-20 | core/architect.py + core/token_tracker.py | Token tracker thread-safety undocumented at architect call site (`:337-341`); verified safe via TokenTracker's `_lock` at `:44`+`:97`. DOWNGRADED to documentation gap. | MINOR | S | LOCAL | OPEN |
| F-21 | core/architect.py + core/verifier/checks_deterministic.py | `ContextHypothesis.supporting_sentence_ids` defaulted to `[]` because Architect never populates it (`:757-759`); D1 verifier check (`:794`) iterates empty list, reports no violations. **D1 has passed vacuously for every run in project history.** Fixing requires Architect prompt + parsing change. | MAJOR | M | PIPELINE | OPEN |

## Shipped

| id | component | finding | severity | effort | blast | status |
|---|---|---|---|---|---|---|
| F-5 | document_parser.py | `_truncate_at_references` matched Turkish `kaynakça` but NOT `Kaynaklar`; also false-positive on numbered `3.4 References` mid-document. Fixed via regex alternation expansion + optional trailing colon + position guard (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`). | MAJOR | S | PIPELINE | **SHIPPED (25e6880)** |
| F-3 | document_parser.py + main.py | `parse_file` returned empty string for empty inputs; 6 duplicated guards in `main.py` (3 HARD, 3 SOFT, 2 of the HARD broken because separator-headers made `combined_text.strip()` always non-empty). Fixed: `EmptySRSDocumentError(ValueError)` raised by `parse_file`; per-path policy at callers (HARD propagate / SOFT skip+log / MIXED batch via new `_parse_srs_batch` helper); aggregate check switched to `srs_docs` emptiness. Latent post-loop-guard bug folded in. | MAJOR | S | MODULE+PIPELINE | **SHIPPED (daefeb0)** |
| F-13 | core/architect.py + core/orchestration/errors.py + main.py | `_save_intermediate` (`:880-891`) caught `Exception` and printed + continued — silent loss of EMSE-reproducibility intermediate JSON artifacts. Fixed: new `IntermediateSaveError(PipelineError)` raises on `(OSError, TypeError, ValueError)`; `identify_contexts` retry handler explicitly re-raises (Codex C-1); anomaly fold-in: `_current_srs_path` initialized in `__init__` + unconditional reassignment in `analyze_document(srs_path=…)` (Codex W-2) + 3 main.py callsites pass joined path; error message binds both fixes (Codex W-5). | MAJOR | S | PIPELINE | **SHIPPED (02e0fe9)** |
| F-14 | core/orchestration/errors.py + core/orchestration/pipeline.py + core/architect.py | `SynthesizerEmptyModelError` unreachable (Pydantic `_non_empty` validator fires first → `ValidationError` escapes `PipelineError` taxonomy). Audit text reframed by Codex: hard-fail policy already explicit (`test_create_fallback_model_is_gone`); real gap is taxonomy preservation, not policy choice. Fixed: pre-call guard at `pipeline.py` raises `SynthesizerEmptyModelError` when `refined_specialist == []` (closes initial-empty + refiner-shrink-success edges per Codex W-1); post-call boundary check retained as belt-and-suspenders for injected synthesizers that bypass Pydantic via `model_construct` (Codex W-3); `srs_path` added per WP-CORE-4 symmetry (Codex OQ-2). Production-dormant (Architect upstream guard intact); fix is contract cleanup for paper-methodology integrity. | MAJOR | M | PIPELINE | **SHIPPED (27a5d98)** |

## Rejected / Deferred

_(empty)_

---

**Decision priority:** production bug fix > test-coverage critical gap > measurable perf regression > evidence-backed clarity smell > cosmetic.

**Status summary (post-iteration-4):**
- 10 ingestion findings: 2 SHIPPED, 4 MAJOR-OPEN, 3 MINOR-OPEN, 1 TRIVIAL-OPEN.
- 11 orchestrator findings: 2 SHIPPED (F-13, F-14), 1 MAJOR-OPEN-DORMANT (F-11), 1 MAJOR-OPEN (F-21), 6 MINOR-OPEN (F-12, F-15, F-17, F-18, F-19, F-20-downgraded), 1 TRIVIAL-OPEN (F-16).
- **Total OPEN MAJOR (live): 5** (F-1, F-2, F-4-uncertain, F-21) + 1 DORMANT (F-11).
- **Iteration-5 recommendation:** F-21 (vacuous D1 verifier pass) per Codex W-8 priority bump — affects every project run methodologically.

**Last refresh:** 2026-05-21 10:33 GMT+3 (post-WP-CORE-4)
