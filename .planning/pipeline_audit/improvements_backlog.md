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
| F-23 | main.py (`:77, 180, 194, 211, 226, 410, 427, 518, 533, 721`) | All exception handlers in main.py are bare `except Exception` with no typed `PipelineError` catch. `ArchitectGroundingError` (WP-CORE-7 NEW) is caught generically and serialized via `str(e)`; run-manifest still surfaces failure but loses typed taxonomy at the FastAPI response level. Fix: add typed `except PipelineError as e: ...` handler before the bare-Exception catch in `/generate-model` (`:533`) and `/validate` (`:427`) endpoints; emit structured `{type: type(e).__name__, details: ...}` JSON so clients can branch on failure mode. | MAJOR | S | LOCAL | OPEN (NEW, WP-CORE-7 Codex W-6) |
| F-24 | core/pipeline_contracts.py + 13 callsites | Contract `VerifierIssue` (Pydantic) has no `srs_path` field. WP-CORE-6 deferred A6-srs-path OQ with concrete revisit trigger "post-F-22". F-22 now SHIPPED → trigger fires. Adding `srs_path` to `VerifierIssue` enables run-manifest issue-level provenance symmetry with WP-CORE-4's `IntermediateSaveError.srs_path` and WP-CORE-5b's `SynthesizerEmptyModelError.srs_path`. Migration: schema widen + 13 construction sites (5 in `checks_semantic_d6_d7_d8.py`, 2 in `checks_semantic.py`, 2 in `checks_deterministic.py` adapter, 4 test files) + verifier_fn closure threads srs_path. | MINOR | M | LOCAL | OPEN (NEW, WP-CORE-7 OQ-6) |

## Shipped

| id | component | finding | severity | effort | blast | status |
|---|---|---|---|---|---|---|
| F-5 | document_parser.py | `_truncate_at_references` matched Turkish `kaynakça` but NOT `Kaynaklar`; also false-positive on numbered `3.4 References` mid-document. Fixed via regex alternation expansion + optional trailing colon + position guard (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`). | MAJOR | S | PIPELINE | **SHIPPED (25e6880)** |
| F-3 | document_parser.py + main.py | `parse_file` returned empty string for empty inputs; 6 duplicated guards in `main.py` (3 HARD, 3 SOFT, 2 of the HARD broken because separator-headers made `combined_text.strip()` always non-empty). Fixed: `EmptySRSDocumentError(ValueError)` raised by `parse_file`; per-path policy at callers (HARD propagate / SOFT skip+log / MIXED batch via new `_parse_srs_batch` helper); aggregate check switched to `srs_docs` emptiness. Latent post-loop-guard bug folded in. | MAJOR | S | MODULE+PIPELINE | **SHIPPED (daefeb0)** |
| F-13 | core/architect.py + core/orchestration/errors.py + main.py | `_save_intermediate` (`:880-891`) caught `Exception` and printed + continued — silent loss of EMSE-reproducibility intermediate JSON artifacts. Fixed: new `IntermediateSaveError(PipelineError)` raises on `(OSError, TypeError, ValueError)`; `identify_contexts` retry handler explicitly re-raises (Codex C-1); anomaly fold-in: `_current_srs_path` initialized in `__init__` + unconditional reassignment in `analyze_document(srs_path=…)` (Codex W-2) + 3 main.py callsites pass joined path; error message binds both fixes (Codex W-5). | MAJOR | S | PIPELINE | **SHIPPED (02e0fe9)** |
| F-14 | core/orchestration/errors.py + core/orchestration/pipeline.py + core/architect.py | `SynthesizerEmptyModelError` unreachable (Pydantic `_non_empty` validator fires first → `ValidationError` escapes `PipelineError` taxonomy). Audit text reframed by Codex: hard-fail policy already explicit (`test_create_fallback_model_is_gone`); real gap is taxonomy preservation, not policy choice. Fixed: pre-call guard at `pipeline.py` raises `SynthesizerEmptyModelError` when `refined_specialist == []` (closes initial-empty + refiner-shrink-success edges per Codex W-1); post-call boundary check retained as belt-and-suspenders for injected synthesizers that bypass Pydantic via `model_construct` (Codex W-3); `srs_path` added per WP-CORE-4 symmetry (Codex OQ-2). Production-dormant (Architect upstream guard intact); fix is contract cleanup for paper-methodology integrity. | MAJOR | M | PIPELINE | **SHIPPED (27a5d98)** |
| F-21 | core/architect.py + core/verifier/checks_deterministic.py + core/orchestration/pipeline.py | `ContextHypothesis.supporting_sentence_ids` defaulted to `[]` because Architect never populated it; D1 verifier check passed vacuously for every project run in history. Audit reframed by Codex: Specialist also rebuilt `ContextHypothesis` fresh (architect.py:707), dropping any IDs upstream. **Production-LIVE** (unlike F-11/F-14 dormant). Fixed end-to-end: Architect prompt rewrite (numbered sentences + object array shape); strict-shape parser (rejects bare-string + top-level list per Codex W-2); line-pair-aware truncation `_truncate_numbered_pairs` (Codex W-1); `extract_per_context_details` signature widened from `List[str]` to `List[ContextHypothesis]` (Codex C-1 — preserves IDs into SpecialistAnalysis.context); D1 also flags empty IDs as ungrounded_context ERROR (honest signal, not enforcement — F-22 tracks full enforcement); degrade-log emits full `exc.issues` list (Codex C-4). | MAJOR | M | PIPELINE | **SHIPPED (a86bbbb)** |
| F-22 | core/orchestration/pipeline.py + core/refiner/loop.py + core/architect.py + core/orchestration/errors.py | Refiner orchestration only re-ran the Specialist stage; architect-stage D1 ERRORs degraded silently per WP-CORE-6 D-3 "honest signal, not enforcement". WP-CORE-7 closes mode C hybrid: `_issue_stage(issue)` helper derives stage from `target` prefix (Codex C-1, avoids `VerifierIssue` schema widen across 13 callsites); `run_pipeline` pre-checks verifier ONCE and dispatches architect-stage issues directly to `architect_with_feedback` rerun (Codex C-2); `refine_until_clean(initial_result=...)` threads pre-check result so the verifier is not double-called on the common path; on persistent architect-stage failure raises new `ArchitectGroundingError(PipelineError)` with srs_path + issues + residual_issues + cycles_attempted (Codex W-4); bare `except Exception` narrowed to `except RefinementExhaustedError` (Codex W-5 — explicit-failure mandate); `identify_contexts(feedback_issues=...)` prepends a structured `PREVIOUS ATTEMPT FAILED VERIFICATION:` block once per outer architect attempt (Codex W-3, N-1). | MAJOR | M-L | PIPELINE | **SHIPPED (ce56d99)** |

## Rejected / Deferred

_(empty)_

---

**Decision priority:** production bug fix > test-coverage critical gap > measurable perf regression > evidence-backed clarity smell > cosmetic.

**Status summary (post-iteration-6):**
- 10 ingestion findings: 2 SHIPPED, 4 MAJOR-OPEN, 3 MINOR-OPEN, 1 TRIVIAL-OPEN.
- 14 orchestrator findings: 4 SHIPPED (F-13, F-14, F-21, F-22), 1 MAJOR-OPEN-DORMANT (F-11), 1 MAJOR-OPEN-NEW (F-23), 7 MINOR-OPEN (F-12, F-15, F-17, F-18, F-19, F-20-downgraded, F-24-NEW), 1 TRIVIAL-OPEN (F-16).
- **Total OPEN MAJOR (live): 4** (F-1, F-2, F-4-uncertain, F-23) + 1 DORMANT (F-11).
- **Iteration-7 recommendation:** F-23 (typed PipelineError handler in main.py) — small scope, completes the WP-CORE-7 enforcement story by surfacing typed run-manifest signal at the FastAPI response boundary. Alternative: pivot to ingestion-layer (F-1 PDF defensive handling / F-2 cp1254 silent garbage) after 5 consecutive orchestrator-layer iterations.

**Last refresh:** 2026-05-23 12:43 GMT+3 (post-WP-CORE-7)
