# Decision Log — Append-only

Format per entry: `## YYYY-MM-DD HH:MM <decision-id>` → 2-3 line rationale + Codex consult summary (if any) + outcome (SHA / dev_doc path).

---

## 2026-05-21 01:34 D-INIT
First fire of domain-pipeline hardening loop. Baseline pytest = 272 passed, 31 deselected; HEAD = 029f187. State directory `.planning/pipeline_audit/` initialized with CURRENT/catalog/backlog/decision_log + findings/ subdir. Explore subagent dispatched (very thorough) to populate component catalog. No Codex consult required at init.

## 2026-05-21 01:42 D-CL1
Close-lookup #1 completed on `core/document_parser.py` + `core/document_parser_readers.py`. Findings card: `.planning/pipeline_audit/findings/document_parser.md`. 10 findings catalogued: 0 BLOCKER, 5 MAJOR (2 uncertain), 4 MINOR, 1 TRIVIAL. Component marked DONE in catalog. No Codex consult — TL;DR was actionable.

## 2026-05-21 01:42 D-PICK-WP-CORE-2
Selected **F-5** standalone for WP-CORE-2:
- Severity: MAJOR. Bug: `_truncate_at_references` regex (`document_parser.py:9-12`, `:60-65`) matches Turkish `kaynakça` but NOT the more common plural `Kaynaklar`. Locale-asymmetric silent data loss for Turkish-authored SRS (D2 banking domain risk). Same regex also can false-positive on a legitimate numbered section like `3.4 References` mid-document.
- Why pick alone (vs. group F-1/F-2/F-3 silent-fallback cluster): smallest correct change per AGENTS.md; single file; single regex + supporting helper; ≤30 LOC diff; pure unit-testable (no I/O); zero downstream coupling change required; clear positive + negative test cases for TDD.
- Why pick over F-3 (empty-input contract): F-5 is a *content-loss* bug (silent), F-3 is a *contract-leak* bug (downstream catches). Silent data corruption > leaky contract in EMSE-paper-impact terms — D2 + D3 SRS may be Turkish-authored and currently silently truncate.
- Why pick over F-4 (TOC heuristic): F-4 marked MAJOR-uncertain; needs deeper Codex consult on downstream Scout compensation. Defer until evidence stronger.
- Codex consult: skipped at decision (value/risk dengesi clear). Adversarial spec review will gate at step 7.

**Outcome:** spec draft → step 5.

## 2026-05-21 01:54 D-CODEX-REVIEW-WP-CORE-2
Codex xhigh adversarial review verdict: **REVISE**. 0 BLOCKER/CRITICAL, 6 WARN.

| # | WARN | category | disposition |
|---|---|---|---|
| W-1 | Proposed `\d+` prefix regresses real `5.1 References` bibliography subsections. | coverage gap | **HANDLED** in spec v2 — replace regex-narrowing with a position guard (truncate only past `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`). Keep nested-number prefix; let `5.1 References` truncate when it sits in latter half; let `3.4 References` *survive* when it sits in earlier half. |
| W-2a | `References:` / `Kaynaklar:` trailing-colon variant untested + unmatched. | coverage gap | **HANDLED** in spec v2 — add optional trailing `:` (or fullwidth `：`) to regex. |
| W-2b | Multiword Turkish like `Yararlanılan Kaynaklar` untested + unmatched. | coverage gap | **ACCEPTED — out of scope for this WP.** Position guard does NOT help (still no regex match). If observed in D2/D3 live runs, follow-up WP. |
| W-3 | No code-fence / preformatted-text guard — inline `References` still truncates. | hidden assumption | **ACCEPTED — out of scope for this WP.** SRS documents rarely embed code fences. Position guard mitigates the most common case (mid-doc fences). Tracked as future hardening. |
| W-4 | Unicode lookalikes / BiDi marks / control marks beyond NBSP+ZWSP not handled. | hidden assumption | **ACCEPTED — out of scope for this WP.** Existing `_normalize_text` covers the realistic SRS cases. Turkish dotted-I problem analyzed: not applicable to `kaynakça`/`kaynaklar` alternation (no I/İ in either word). |
| W-5 | No direct parametrized unit test on the regex object itself. | testability hole | **HANDLED** in spec v2 — add `pytest.mark.parametrize` test against `SRSDocumentParser().reference_heading_pattern.match`. |
| W-6 | ASCII-folded `kaynakca` (no diacritic) excluded without recorded rationale. | minimalism | **ACCEPTED — out of scope for this WP.** Without corpus evidence of `Kaynakca` heading variant in real SRS, do not add speculatively (AGENTS.md). If observed in D2/D3, follow-up WP. |

3 WARNs handled by spec revision (W-1, W-2a, W-5). 4 WARNs accepted with rationale (W-2b, W-3, W-4, W-6). Spec proceeds to v2.


## 2026-05-21 02:17 D-SHIP-WP-CORE-2

WP-CORE-2 SHIPPED. SHAs:
- RED commit (test-first): `4f932d2` — `test(parser): WP-CORE-2 red-phase tests for reference-truncation correctness`
- GREEN commit (production): `25e6880` — `fix(parser): WP-CORE-2 reference truncation — locale parity + position guard`
- DOC commit (this entry's commit): to be recorded in the next loop tick.

Post-WP pytest baseline: 305 passed, 31 deselected (net +33 tests vs pre-WP 272). Live D1 E2E re-run skipped — regex change strictly reduces false-positive matches; English `References` (D1 corpus) unaffected.

Dev doc: `development_docs/WP-CORE-2-reference-truncate-fix.md`.
INDEX row: appended to ACTIVE table.


## 2026-05-21 07:52 D-CODEX-REVIEW-WP-CORE-3
Codex xhigh adversarial review of WP-CORE-3 spec v1 returned **2 CRITICAL + 5 WARN**.

| # | severity | finding | disposition |
|---|---|---|---|
| C-1 | CRITICAL | Mixed-batch behavior mislabeled as unchanged. v1 said "no change to batch atomicity" but proposed making each empty file kill the batch. | **HANDLED** in spec v2 — batch loops now skip-and-continue on `EmptySRSDocumentError`; aggregate check switched from broken `combined_text.strip()` to `srs_docs` emptiness. Pre-WP post-loop guard was already dead code (separator headers made strip always non-empty); folded fix into WP. Behavior change documented explicitly in R-5. |
| C-2 | CRITICAL | Main call-site migration declared untested; greps only prove string removal, not control-flow correctness. | **HANDLED** in spec v2 — extracted `_parse_srs_batch(parser, file_paths) -> (combined_text, srs_docs, error_or_none)` helper as testable seam; added T-WIRE-1..4 (mixed batch, all-empty aggregate, per-file read failure, `initialize_rag` skip) via `_StubParser(SRSDocumentParser)` subclass + monkeypatching. |
| W-1 | WARN | SOFT-vs-uncaught policy ambiguity. | **HANDLED** in spec v2 — new §"Per-path empty-input policy" table with HARD/SOFT/MIXED rows. |
| W-2 | WARN | LLM-layer precondition (parse_file is sole ingress) unstated. | **HANDLED** in spec v2 — new §"Scope and preconditions" with grep evidence proving sole-ingress for production `DomainArchitect.analyze_document(text=...)`. |
| W-3 | WARN | Intermediate GREEN commit half-migrates the contract. | **HANDLED** in spec v2 — implementation order collapses GREEN + REFACTOR into one atomic behavior commit. RED commit lands all tests first to preserve TDD discipline. |
| W-4 | WARN | Logging policy cites nonexistent `AGENTS.md "Logging policy: silent OR print"` rule. | **HANDLED** in spec v2 — citation removed; rationale rewritten to cite the actual `main.py` `print`-everywhere convention with `logging`-module introduction deferred to F-9. |
| W-5 | WARN | Acceptance grep overbroad and behavior-blind. | **HANDLED** in spec v2 — behavior acceptance criteria added (T-WIRE-* outcomes + `except EmptySRSDocumentError` count ≥ 4); greps retained only as secondary cleanup verification. |

All 7 findings handled inline. Zero WARNs accepted-with-rationale (unlike WP-CORE-2 where 4 of 6 WARNs were deferred). Spec v2 is implementation-ready.


## 2026-05-21 09:32 D-EMPTY-INPUT-CONTRACT-2026-05-21
**Decision:** Chose **Alt C** (parser raises uniformly + per-path policy at callers) over Alt B (two `parse_file` entry points, one raising one not) and Alt E (helper-extraction only without parser raise).

**Rationale:**
- AGENTS.md "Stable entrypoints, isolate change-prone logic" — the empty-input invariant belongs inside the parser, not duplicated across six call sites.
- Alt B codifies the leaky contract into the API (two entry points = policy in shape).
- Alt E papers over only the batch case; single-file `generate_domain_model` and SOFT RAG sites would still leak.
- `EmptySRSDocumentError IS-A ValueError` chosen over new `DomainParserError` hierarchy: smallest-correct-change; preserves `except ValueError` callers; broader hierarchy is speculative generalization.
- Empty check **after** `_post_process` (not after raw read): captures three failure modes for free (0-byte input, whitespace-only, cross-WP-CORE-2 post-truncation-empty).

**Outcome:** GREEN commit `daefeb0`. Pytest 305 → 321 (+16 tests). Dev doc: `development_docs/WP-CORE-3-empty-input-contract.md`.


## 2026-05-21 09:37 D-SHIP-WP-CORE-3

WP-CORE-3 SHIPPED. SHAs:
- RED commit (test-first): `91dbeb4` — `test(parser, main): WP-CORE-3 red-phase tests for empty-input contract`
- GREEN commit (atomic): `daefeb0` — `fix(parser, main): WP-CORE-3 EmptySRSDocumentError — explicit empty-input contract`
- DOC commit (this entry's commit): to be recorded in the next loop tick.

Post-WP pytest baseline: 321 passed, 31 deselected (net +16 tests vs pre-WP 305). Live D1 E2E re-run skipped — empty-input contract strictly tightens behavior; D1 SRS has non-empty content and triggers no new code paths.

Dev doc: `development_docs/WP-CORE-3-empty-input-contract.md`.
INDEX row: appended to ACTIVE table.

Latent bug folded in: post-loop `combined_text.strip()` guards at `/generate-model{,-stream}` were dead code (broken by separator-header inclusion); fixed via `srs_docs`-emptiness aggregate check in the new helper.

Behavior change documented (R-5): mixed batches (one empty + one good file) now succeed cleanly (empty skipped + logged); previously the empty file silently degraded combined input with separator-only content.

---

## 2026-05-21 10:00 D-CL2
Close-lookup #2 completed on `core/architect.py` (LOC=923) + `core/orchestration/pipeline.py` (LOC=84) + `core/orchestration/errors.py` (LOC=70). Findings card: `.planning/pipeline_audit/findings/architect.md`. 11 findings catalogued (after F-21 added during spec drafting + F-20 downgrade to MINOR after verification): 0 BLOCKER, 4 MAJOR (F-11, F-13, F-14, F-21), 6 MINOR (F-12, F-15, F-17, F-18, F-19, F-20-downgraded), 1 TRIVIAL (F-16). 6 anomalies recorded (3 latent observability bugs + 3 design smells). Component-catalog row updated PENDING → DONE. No Codex consult at audit time (TL;DR actionable).

## 2026-05-21 10:10 D-PICK-WP-CORE-4
Selected **F-13** (`_save_intermediate` silently swallows I/O exceptions) bundled with anomaly **`_current_srs_path` never assigned** for WP-CORE-4:
- F-13 severity: MAJOR. Bug: `core/architect.py:880-891` catches `Exception` and prints + continues. 4 callsites silently lose intermediate JSON diagnostics. EMSE-reproducibility-blocking per CLAUDE.md §"Persistent Development Memory".
- Anomaly severity: latent MINOR alone; MAJOR when bundled because save-failure error messages need an SRS path. The two observability fixes interlock at the error-message level.
- Why bundle (vs. ship separately): both touch `DomainArchitect`; both implement AGENTS.md "no silent degradation"; pattern matches WP-CORE-3's fold-in of post-loop guard bug into the empty-input fix.
- Why not F-11 (parallel Scout race): M-L effort, concurrency redesign; loop cadence stays at S-effort MAJOR per iteration.
- Why not F-14 (SynthesizerEmptyModelError escape): M effort, policy decision (hard-fail vs degrade) requires extra Codex round-trip.
- Why not F-21 (vacuous D1 pass): Architect prompt + parsing change; out of "smallest correct change" envelope.
- Codex consult: spec adversarial review at step 7 (mandatory per loop ritual).

**Outcome:** spec v1 → Codex xhigh review → spec v2.

## 2026-05-21 10:15 D-CODEX-REVIEW-WP-CORE-4
Codex xhigh adversarial review verdict: **REVISE**. 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ.

| # | finding | category | disposition |
|---|---|---|---|
| C-1 | Architect-stage `_save_intermediate` calls at lines 449/462 are inside `identify_contexts`' broad `except Exception` retry wrapper (line 485); raised `IntermediateSaveError` silently caught, retried 5×, rewrapped as `ArchitectExtractionError`. | propagation hole | **HANDLED** in spec v2 — add `except IntermediateSaveError: raise` between line 483 (`except ArchitectExtractionError: raise`) and line 485 (`except Exception as e:`) in `identify_contexts`. Specialist save at line 650 is already outside its retry loop; no equivalent guard needed there. |
| C-2 | RED tests stub `_save_intermediate` directly but don't test real production propagation; false-green risk for the C-1 path. | test gap | **HANDLED** in spec v2 — added T-SAVE-4 (`_save_intermediate` failure inside `identify_contexts` → assert `IntermediateSaveError` propagates, NOT `ArchitectExtractionError`) + T-SAVE-5 (same for Specialist). |
| W-1 | `IntermediateSaveError(OSError)` violates the orchestration error taxonomy in `errors.py` (which has `PipelineError` as base). | taxonomy consistency | **HANDLED** — moved to `core/orchestration/errors.py`, base = `PipelineError`. |
| W-2 | Conditional assignment `if srs_path is not None: ...` leaks stale path on instance reuse. | correctness | **HANDLED** — unconditional `self._current_srs_path = srs_path or "<unknown>"` at start of every `analyze_document`. Added T-SRS-4 reuse test. |
| W-3 | "; "-joined batch label "bounded by validation" claim unsupported (no max-length on `GenerateModelRequest.file_paths`). | unsupported claim | **HANDLED** — dropped bounded claim; label is display-only; no truncation introduced (KISS). |
| W-4 | Scout's `_save_intermediate` call at line 236 is in DEAD `extract_domain_sentences` — only `tests/test_architect_helpers.py:119` exercises it. `analyze_document`'s nested `scout_fn` does NOT call it. | stale spec text | **HANDLED** — adjusted Motivation §F-13 to note Scout save is dead from production; production-reachable saves are Architect (lines 449/462) + Specialist (line 650). Fix still applied uniformly for legacy-path consistency. New anomaly "typed Scout dump missing in scout_fn" added to findings doc. |
| W-5 | `IntermediateSaveError` message lacks `srs_path`. Endpoint users won't know which SRS failed. | observability binding | **HANDLED** — added `srs_path` field to error constructor; populated from `self._current_srs_path` at raise time. The two observability fixes interlock at error-message level. |
| N-1 | Test count inconsistent (6 vs 7). | doc bug | **HANDLED** — resolved to 11 total: T-SAVE-1..5 + T-SRS-1..4 + T-WIRE-MAIN-1..3. |
| N-2 | `JSONDecodeError` cited under `json.dump` rationale (wrong direction). | doc bug | **HANDLED** — removed; encoder-side errors are `TypeError` + `ValueError` (circular ref). |
| N-3 | Env-var no-op patch escape hatch speculative. | scope creep | **HANDLED** — removed from OQ#3 disposition. |
| OQ1 | DISAGREE: `OSError` base wrong. | base class | **ADOPTED** — `PipelineError` base. |
| OQ2 | PARTIALLY: catch list correct but Architect retry must re-raise. | composite | **ADOPTED both** — narrow `(OSError, TypeError, ValueError)` + C-1 re-raise. |
| OQ3 | PARTIALLY: label OK but no max-length. | composite | **HANDLED via W-3.** |
| OQ4 | PARTIALLY: getattr cleanup defensible either way. | judgment | **KEPT getattr** as belt-and-suspenders (smallest correct change). |
| OQ5 | DISAGREE: defer endpoint wiring tests. | test coverage | **ADOPTED Codex** — added T-WIRE-MAIN-2 + T-WIRE-MAIN-3. |
| OQ6 | PARTIALLY: "no other callers" vs "no other production callers". | phrasing | **HANDLED** — spec adjusted to "no other production callers". |
| OQ7 | AGREE: defer F-21. | scope | **CONFIRMED** — F-21 deferred to WP-CORE-5+. |

**Outcome:** zero deferred WARNs (contrast WP-CORE-2 which deferred 4/6 WARNs as out-of-scope). v2 ready for plan-phase.

## 2026-05-21 10:55 D-PICK-WP-CORE-5
Selected **F-11** (parallel Scout rate-limit reentrancy race) for iteration 4 per handoff-2026-05-21-1033 §"Recommended next iteration" Option A:
- Severity: MAJOR. Concurrency primitive — `_wait_for_rate_limit` at `core/architect.py:145-155` timestamps `last_request_time` at lock-release, not API-send-time. Jitter between unlock and `client.chat()` invocation accumulates over N parallel workers.
- Effort estimate: M-L (concurrency redesign).
- Why now: highest-impact remaining OPEN MAJOR; foundational primitive; F-14/F-21 fixes will coexist with parallel Scout.
- Codex consult: spec adversarial review at step 7 (mandatory per loop ritual).

**Outcome:** spec v1 drafted at `docs/superpowers/specs/2026-05-21-wp-core-5-parallel-scout-rate-limit-design.md` → Codex xhigh review (next entry).

## 2026-05-21 11:08 D-CODEX-REVIEW-WP-CORE-5
Codex xhigh adversarial review verdict: **NOT READY FOR RED.** 3 CRITICAL + 5 WARN + 3 NITS + 3 OQ. Findings invalidate spec framing + RED design.

| # | finding | category | disposition |
|---|---|---|---|
| C-1 | `extract_domain_sentences` (`ThreadPoolExecutor`-based parallel Scout the spec targeted) is **dead from production**. `analyze_document.scout_fn` at lines 757-774 calls only `section_aware_chunks()` — no LLM, no `_wait_for_rate_limit`. F-11 symptom dormant in current production code. | scope misframing | **ABANDON WP** — confirmed by observation 3618 (2026-05-21 10:11) from WP-CORE-4 sweep. Rewiring `extract_domain_sentences` into `analyze_document.scout_fn` is its own larger WP; not in scope. |
| C-2 | Reservation pattern paces `_wait_for_rate_limit()` returns, not `client.chat()` send-time. Actual wire-level microslip lives in post-return / pre-send window. Correct primitive must gate `client.chat()` entry. | design wrong | **ABANDON WP** — fix would require a rate-limited send helper, not a reservation rewrite. Out of "smallest correct change" envelope without first establishing send-time as the invariant. |
| C-3 | T-RATE-3/T-RATE-4 (parallel multi-worker wall-clock gaps) pass against current implementation. Current lock-held `time.sleep` already serializes returns + totals. RED tests don't red-signal F-11. | test design wrong | **ABANDON WP** — would need send-level deterministic test with injected post-wait jitter; rewriting RED is half the work. |
| W-1..W-5, N-1..N-3, OQ-1..OQ-3 | Various tightening + EMSE-evidence asks. | composite | **MOOT given abandon.** Preserved in Codex output (saved to this entry below) for the record. |

Raw Codex output:
- **CRITICAL-1**: scope — parallel Scout path dead in `analyze_document` (`scout_fn` builds `section_aware_chunks` only).
- **CRITICAL-2**: invariant — reservation paces returns, not sends; lock-release-to-`client.chat()` gap remains.
- **CRITICAL-3**: RED tests pass against current code; need send-level deterministic test.
- **WARN-1**: thundering-herd if host pauses past multiple reserved slots (sleep outside lock).
- **WARN-2**: F-11 description overstates pre-send Python work + cumulative slip ("30 × j" claim unsupported).
- **WARN-3**: CI flake risk from `<= 0.5s` upper bound + 10 ms tolerance on `time.sleep()` jitter.
- **WARN-4**: T-RATE-5 conflicts with stated "black-box only" — would need internal-attribute access.
- **WARN-5**: EMSE-paper claim of "empirically defensible 6 s contract" under-evidenced; need before/after measurements + N=10 baseline.
- **NIT-1**: test paths omit `extension/backend/` prefix.
- **NIT-2**: OQ numbering skips from 2 to 4.
- **NIT-3**: monotonic_start rationale unnecessary; `max(now, 0.0)` already correct.
- **OQ-1**: invariant choice — return spacing vs `client.chat` entry spacing vs wire-send spacing.
- **OQ-2**: "hard 6 s minimum delay" quota model assumption — verify vs Gemini docs (sliding RPM window?).
- **OQ-3**: starvation analysis shouldn't rely on CPython `threading.Lock` fairness (no FIFO contract).

**User decision (2026-05-21 ~11:08):** drop WP-CORE-5; pivot iteration 4 to F-14 (`SynthesizerEmptyModelError` pipeline escape). F-11 stays OPEN with status note "DORMANT — parallel Scout path dead from production; reopen when `extract_domain_sentences` is rewired into `analyze_document.scout_fn` or `section_aware_chunks` gains an LLM call." Sequential primitive slip (~1-5 ms per gap) within 6 s buffer in practice; defense-in-depth fix can wait.

**Outcome:** spec v1 marked ABANDONED at top (banner preserved for audit trail); WP-CORE-5b spawned for F-14. No commits this WP.

## 2026-05-21 11:09 D-PICK-WP-CORE-5b
Selected **F-14** (`SynthesizerEmptyModelError` pipeline escape) for iteration 4 pivot:
- Severity: MAJOR. `core/orchestration/pipeline.py:81-83` raises `SynthesizerEmptyModelError` with no analogous degrade-handler (unlike `RefinementExhaustedError` at lines 68-74 which falls back to best-effort). Hard-fail on degenerate Synthesizer output silently propagates to `main.py:107` → caught generically as `Exception` at lifespan handler line 180 → `app_state["domain_rules"] = {}`. EMSE-reproducibility gap.
- Effort estimate: M (policy decision + 1-2 file changes).
- Why over F-21 (vacuous D1): F-14 is a clear policy choice with concrete options (hard-fail-with-context / degrade-best-effort / retry-relaxed); F-21 requires Architect prompt change which has uncertain LLM-behavior impact. F-14 = test-first-friendly.
- Codex consult: spec adversarial review at step 7 (mandatory).

**Outcome:** spec v1 to be drafted at `docs/superpowers/specs/2026-05-21-wp-core-5b-synthesizer-empty-model-policy-design.md`.

## 2026-05-21 11:30 D-CODEX-REVIEW-WP-CORE-5b
Codex xhigh adversarial review verdict: **GO with 3 conditions.** 0 CRITICAL + 6 WARN + 3 NITS + 3 OQ. All findings handled inline in spec v2 (zero deferred — third consecutive zero-deferred iteration after WP-CORE-3 + WP-CORE-4).

| # | finding | category | disposition |
|---|---|---|---|
| W-1 | `refiner-empty-success-path`: `refine_until_clean` non-exception path returns whatever `stage_runner` last produced; if rerun returns `[]` AND verifier accepts, `refined_specialist` becomes `[]` even though initial `specialist_output` was non-empty. Spec v1 missed this edge. | edge gap | **HANDLED via T-EMPTY-3** — new test: first Specialist returns non-empty, verifier fails once, rerun returns `[]`, verifier accepts → pipeline raises `SynthesizerEmptyModelError`. |
| W-2 | `specialist-empty-is-only-blocked-upstream`: `extract_per_context_details([])` returns `[]` cleanly (no raise); the "Specialist raises" claim in spec v1 Discovery 3 was overstated. | accuracy | **HANDLED** — Discovery 3 reworded: production chain is protected by **Architect's** upstream raise; Specialist itself has no empty-input guard. |
| W-3 | `keep-post-boundary-check`: deleting the post-call check loses a cheap boundary invariant for injected/future synthesizers. `PipelineDeps.synthesizer` is freely injectable. | safety net | **HANDLED** — post-call check retained as belt-and-suspenders for injected synthesizers that bypass Pydantic via `DomainModel.model_construct(...)`. T-EMPTY-4 added to cover this layer. |
| W-4 | `missing-refiner-empty-test`: covered by W-1 disposition. | test gap | **HANDLED via T-EMPTY-3.** |
| W-5 | `t-empty-4-duplicates-existing-layer`: Pydantic-validator coverage already at `test_synthesizer_deterministic_merge.py:97-103`. | redundancy | **HANDLED** — v1's "T-EMPTY-4 direct ValidationError" test dropped; coverage exists. |
| W-6 | `red-commit-ci-risk`: RED commit with pytest exit 1 ok locally but bad pushed-history hygiene. | discipline | **CONFIRMED EXISTING** — loop discipline keeps commits local until user-driven `git push`. Documented in spec v2 §Atomic commit sequence. |
| W-7 | `taxonomy-not-consumed`: `main.py` catches generic `Exception`; PipelineError taxonomy is future-facing not operational today. | framing | **HANDLED** — §Motivation reframed as "contract cleanup for paper-methodology integrity" not "production hardening". |
| W-8 | `f21-priority`: F-21 has clearer paper impact (every project run vacuously passes D1 check). | scope advice | **ACKNOWLEDGED** — WP-CORE-5b ships as quick cleanup per user decision; F-21 queued as iteration-5 target. |
| N-1 | `dead-code-scope`: post-call check is dead only for in-tree synthesizer, not globally. | precision | **HANDLED** — Discovery 2 tightened. |
| N-2 | `taxonomy-test-overlap`: T-EMPTY-1 + T-EMPTY-2 overlap. | redundancy | **HANDLED** — merged into single `pytest.raises(PipelineError) as exc; isinstance(exc.value, SynthesizerEmptyModelError)`. |
| N-3 | `emse-framing-holds`: hard-fail grounded; do not add recovery modes. | confirmation | **CONFIRMED** — §Non-goals unchanged. |
| OQ-1 | Pre-call vs catch-and-rewrap. | design | **PRE-CALL ADOPTED** (Codex agreed; KISS). |
| OQ-2 | `srs-path` symmetry with `IntermediateSaveError`. | symmetry | **ADOPTED** — `srs_path: str = "<unknown>"` added; widened `run_pipeline` signature; threaded from `analyze_document`. T-EMPTY-5 verifies. |
| OQ-3 | `main-handler-test`: lifespan/endpoints catch generic `Exception`; no handler-selection change. | scope | **NO ACTION** (Codex agreed). |

**Outcome:** zero deferred WARNs (third consecutive iteration matching WP-CORE-3 + WP-CORE-4 standard). v2 ready for plan-phase. WP-CORE-5b shipped: RED `cc82e64`, GREEN `27a5d98`, DOC commit pending (this one), PLANNING pending. Baseline 332 → 338 (+6 tests; zero regression). Iteration-5 target: F-21 per W-8 advice.
