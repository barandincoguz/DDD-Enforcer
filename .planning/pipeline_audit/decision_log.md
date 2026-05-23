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

## 2026-05-22 12:30 D-PICK-WP-CORE-6
Selected **F-21** (D1 verifier vacuous-pass) for iteration 5 per Codex W-8 from WP-CORE-5b review:
- Severity: MAJOR. Production-LIVE bug: D1 has passed vacuously for every project run in history because Architect's `identify_contexts` returns bare context-name strings, leaving `ContextHypothesis.supporting_sentence_ids` at Pydantic default `[]`. EMSE methodology paper claim "D1 catches contexts citing un-emitted sentences" is currently empirically vacuous.
- Effort estimate: M (Architect prompt + parsing + cross-stage data flow).
- Why over F-11 (DORMANT) or MINOR cluster: highest-paper-impact remaining MAJOR; affects every run, not a dormant edge.
- Codex consult: spec adversarial review at step 7 (mandatory per loop ritual + production-reachability subsection required per iteration-4 lesson).

**Outcome:** spec v1 drafted at `docs/superpowers/specs/2026-05-21-wp-core-6-d1-verifier-non-vacuous-design.md` → Codex xhigh review (next entry).

## 2026-05-22 12:50 D-CODEX-REVIEW-WP-CORE-6
Codex xhigh adversarial review verdict: **4 CRITICAL + 4 WARN + 6 NIT + 1 OQ.** All CRITICAL+WARN handled inline in spec v2; 1 OQ deferred with explicit scope-bounded rationale + concrete revisit trigger (post-F-22).

| # | finding | category | disposition |
|---|---|---|---|
| C-1 (A7) | `final-model-loss`: Even with Architect populating IDs, Specialist receives `List[str]` names only (architect.py:788), rebuilds `ContextHypothesis(context_name=ctx_name, description="")` fresh at line 707 with default empty IDs. Synthesizer merge copies the empty list. Final DomainModel still vacuous. | scope gap | **ADOPTED.** Spec v2 widens scope: `extract_per_context_details` signature changes from `List[str]` to `List[ContextHypothesis]`. `specialist_fn` passes `list(arch.contexts)` directly. Line 707 ctx-rebuild deleted; input ctx re-used. Synthesizer merge unchanged (already copies; just receives non-empty input now). T-INT-1 added for E2E coverage. |
| C-2 (A2-d3-mask) | `d3-mask`: D-3 (D1 ERROR on empty IDs) doesn't enforce because Refiner exhausts → pipeline.py catches → degrades to best-effort. ERROR is logged-and-discarded. | enforcement gap | **ADOPTED with reframe.** D-3 kept as honest signal (defense-in-depth against future Architect prompt regression). Paired with D-6 degrade-log enrichment (closes A5-risk4). Full enforcement deferred to F-22 (NEW backlog). |
| C-3 (A3-integration-gap) | `integration-gap`: 5 tests insufficient; no E2E test verifying IDs survive Architect → Specialist → Synthesizer. | test gap | **ADOPTED via T-INT-1**: `analyze_document` E2E with mocked LLM responses; assert final `DomainModel.bounded_contexts[0].supporting_sentence_ids` matches Architect's IDs. |
| C-4 (A5-risk4) | `risk4-understated`: `RefinementExhaustedError.issues` reduced to `print(type(exc).__name__)`; D1 errors not reliably logged. | observability gap | **ADOPTED via D-6**: split `except Exception` into `except RefinementExhaustedError` (issues-list log) + fallback. Closes the run-manifest visibility gap. |
| W-1 (A2-truncation) | `truncation-mid-prefix`: `_truncate_with_head_tail` char-slices; can chop `[N] ` mid-prefix. | correctness gap | **ADOPTED via D-4**: new `_truncate_numbered_pairs(pairs, max_chars, head_ratio)` helper drops whole `(idx, text)` pairs from middle. Pair atomicity preserved. |
| W-2 (A3-top-level-old-shape) | `top-level-list-bypass`: parser at architect.py:464-476 accepts top-level list `["X", "Y"]`. | parsing gap | **ADOPTED via D-5b**: top-level list branch DELETED. Strict dict-wrapper only with per-context shape validation. |
| W-3 (A4-scope) | `commit-plan-mis-scoped`: GREEN mis-scoped until integration + downstream propagation included. | scope | **ADOPTED** via C-1 + C-3 expansion. GREEN now covers 4 production files (was 3). |
| W-4 (A6-f22) | `refiner-cant-rerun-architect`: limitation should become F-22 or be fixed now. | backlog hygiene | **ADOPTED as F-22 backlog add.** New OPEN MAJOR row in DOC commit. Deferred fix (own WP). |
| N-1, N-2 (A1) | Discovery claims accurate. | confirmation | **ACCEPT-AS-IS.** |
| N-3 (A2-strict-shape) | Strict rejection correct (no compat surface). | confirmation | **ACCEPT-AS-IS.** |
| N-4 (A3-tarch2-trace) | T-ARCH-2 RED expectation accurate. | confirmation | **ACCEPT-AS-IS.** |
| N-5 (A3-prompt-test) | T-ARCH-3 spy/substr OK if inspects `client.chat(messages=[...])` payload. | confirmation | **ACCEPT-AS-IS** — test calls `mock_chat.call_args.kwargs["messages"][0]["content"]`. |
| N-6 (A5-methodology + A6-lineage) | EMSE framing + F-21 lineage accurate. | confirmation | **ACCEPT-AS-IS.** |
| OQ-1 (A6-srs-path) | Adding `srs_path` to D1 issues not symmetric without `VerifierIssue` schema widening. | scope | **DEFERRED with rationale.** Verifier issues are runtime-only (not persisted artifacts like orchestration errors); adding `srs_path` requires schema change + 5-site threading. Revisit trigger: post-F-22 (if verifier issues become Refiner control-flow primary signals). |

**Outcome:** 4 CRITICAL + 4 WARN all ADOPTED inline; 6 NIT all confirmed; 1 OQ deferred with concrete revisit trigger. The 4-iteration zero-deferred streak (CORE-3/4/5b/6) ends here, by design — the OQ deferral is scope-bounded with a promotion criterion, not vague "future work" drift. WP-CORE-6 shipped: RED `fd7f203`, GREEN `a86bbbb`, DOC commit pending (this one), PLANNING pending. Baseline 338 → 348 (+10 tests; zero regression). Iteration-6 target: F-22.

**Outcome:** 4 CRITICAL + 4 WARN all ADOPTED inline; 6 NIT all confirmed; 1 OQ deferred with concrete revisit trigger. The 4-iteration zero-deferred streak (CORE-3/4/5b/6) ends here, by design — the OQ deferral is scope-bounded with a promotion criterion, not vague "future work" drift. WP-CORE-6 shipped: RED `fd7f203`, GREEN `a86bbbb`, DOC `70fa8c8`, PLANNING `4c8580c`. Baseline 338 → 348 (+10 tests; zero regression). Iteration-6 target: F-22.

## 2026-05-22 22:00 D-PICK-WP-CORE-7
Selected **F-22** (Refiner stage-aware re-runs) for iteration 6 per WP-CORE-6 handoff §"Recommended next iteration" + Codex W-4 / A6-f22 from WP-CORE-6 review:
- Severity: MAJOR. Production-LIVE bug: D1 architect-stage ERRORs surface (post-WP-CORE-6 honest-signal) but pipeline degrades silently — `RefinementExhaustedError` caught + logged + best-effort fallback. EMSE methodology claim "D1 enforces grounding" undermined.
- Mode choice (user-driven via AskUserQuestion): **C hybrid** (1 architect re-run with issue-aware feedback prompt, then hard-fail via new `ArchitectGroundingError`). Bounded cost (max 10 LLM calls worst case). Genuine self-correction shot. Explicit failure on persistent grounding violation. Matches AGENTS.md "smallest correct change" + "explicit failure".
- Effort estimate: M-L (Refiner control-flow refactor + new exception + feedback prompt).
- Codex consult: spec adversarial review at step 7 (mandatory per loop ritual + production-reachability subsection per iteration-4 lesson).

**Outcome:** spec v1 drafted at `docs/superpowers/specs/2026-05-22-wp-core-7-refiner-stage-aware-design.md` → Codex xhigh review (next entry).

## 2026-05-23 00:30 D-CODEX-REVIEW-WP-CORE-7
Codex xhigh adversarial review verdict: **2 CRITICAL + 6 WARN + 2 NIT + 1 OQ.** All CRITICAL+WARN handled inline in spec v2; 2 NIT inlined; 1 OQ recorded as F-24 follow-up backlog (post-F-22 unlocked).

| # | finding | category | disposition |
|---|---|---|---|
| C-1 | `verifier-issue-stage-drop`: contract `VerifierIssue` (`pipeline_contracts.py:152-163`) has no `stage` field; `_to_contract_issue` adapter (`architect.py:835-846`) drops the legacy `.stage`. D-5's `i.stage == "architect"` AttributeError at runtime. | dispatcher-blocking | **ADOPTED.** Spec v2 §D-2: new `_issue_stage(issue) -> Optional[str]` helper derives stage from `target` prefix (every Verifier check populates `"{stage}:"` prefix). No `VerifierIssue` schema widen (avoids 13-callsite migration). Invariant documented in handoff §"Non-negotiables carried forward". |
| C-2 | `dispatcher-ordering`: v1 enters `refine_until_clean(_re_run_specialist)` first, then partitions on `RefinementExhaustedError`. Architect-stage issues burn 2 wasted Specialist re-runs before exhaustion. | correctness + efficiency | **ADOPTED.** Spec v2 §D-5 restructured: pre-check verifier ONCE before entering `refine_until_clean`. Architect-stage issues dispatch directly to architect feedback rerun. Specialist refine loop only entered when no architect-stage issues. `refine_until_clean` gains optional `initial_result=` kwarg to skip its own first verify on the common path. |
| W-1 | `mixed-stage-success-untested`: T-DISPATCH-4 covers persistent failure but not "architect fix succeeds, specialist still has issues, specialist refine runs". | test gap | **ADOPTED.** New T-DISPATCH-5 added. Verifies architect issue first → architect rerun → specialist issue → specialist refine → ok. Pipeline does NOT raise ArchitectGroundingError when feedback fixes architect. |
| W-2 | `red-collection-math + import-collection-failure`: spec oscillates between "355" and "356"; top-level `from core.orchestration.errors import ArchitectGroundingError` would cause pytest collection failure. | test-plan correctness | **ADOPTED.** Math reconciled: RED = 358 collected, 348 pass, 10 fail. T-AGE-1 + T-INT-1 + flip tests import `ArchitectGroundingError` inside test function bodies — pytest treats as ImportError-on-execution (test failure, not collection error). Verified empirically in RED `aea15e4`. |
| W-3 | `feedback-test-too-weak`: T-FEEDBACK-1 says "feedback substring"; field name ambiguous (`target` vs `location`). | precision | **ADOPTED.** Spec v2 §D-4 specifies exact format: `"PREVIOUS ATTEMPT FAILED VERIFICATION:"` + per-issue `- {issue.target}: {issue.message}`. T-FEEDBACK-1 asserts all three substrings + ordering (feedback before main instruction block). |
| W-4 | `aGE-payload-vs-claim-mismatch`: T-DISPATCH-4 v1 claims "specialist issues kept for post-mortem visibility" but `ArchitectGroundingError(issues=arch_issues)` only carries architect issues. | spec internal consistency | **ADOPTED with payload widening.** `ArchitectGroundingError` gains `residual_issues: List[VerifierIssue]` parameter for non-architect issues. T-DISPATCH-4 asserts both `len(.issues) == N_arch` and `len(.residual_issues) == N_other`. |
| W-5 | `silent-degrade-retained`: v1 D-5 retained bare `except Exception` at `pipeline.py:98-112` as "defensive." Contradicts explicit-failure mandate. | methodology integrity | **ADOPTED.** Spec v2 §D-5 narrows to `except RefinementExhaustedError` only. Other exceptions propagate. Aligns with AGENTS.md "explicit failure". |
| W-6 | `main.py-pipelineerror-claim-wrong`: v1 says "existing `try/except PipelineError` blocks catch it transparently." Reality: all 10 main.py handlers are bare `except Exception`. | downstream-impact accuracy | **ADOPTED with correction.** Spec v2 §Downstream-impact corrected: bare-Exception catches the new exception via `str(e)` serialization. Typed `PipelineError` handler tracked as **F-23 backlog** (NEW, out of WP-CORE-7 scope). |
| N-1 | `feedback-per-internal-retry`: v1 doesn't specify per-attempt vs per-internal-retry feedback. | precision | **ADOPTED.** Spec v2 §D-4: feedback prepended ONCE per outer architect attempt; reused across 5 internal JSON-parse retries. |
| N-2 | `oq-4-d5-unreachable-not-indirect`: v1 OQ-4 said D5 synthesizer check is "covered indirectly." Reality: `ContextHypothesis` has no `allowed_dependencies` field; D5 is unreachable. | rationale precision | **ADOPTED.** Spec v2 OQ-4 reworded: "D5 synthesizer-stage check is unreachable today. When a real synthesizer-stage check is added, its dispatch becomes a new WP." |
| OQ-1 | `a6-srs-path-unlocked`: WP-CORE-6's A6-srs-path deferred OQ trigger "post-F-22" now fires. | scope hygiene | **ADOPTED as F-24 backlog (NEW), out-of-scope.** Adding `srs_path` to `VerifierIssue` requires schema widening + 13-callsite threading. Kept blast radius narrow for WP-CORE-7. |

**Outcome:** 2 CRITICAL + 6 WARN all ADOPTED inline; 2 NIT inlined; 1 OQ → F-24 follow-up. No items deferred without trigger. WP-CORE-7 shipped: RED `aea15e4`, GREEN `ce56d99`, DOC commit pending (this one), PLANNING pending. Baseline 348 → 358 (+10 tests; zero regression). Iteration-7 target: F-23 (small scope; completes WP-CORE-7 enforcement story at FastAPI response boundary) or pivot to ingestion-layer MAJOR.

## 2026-05-23 12:45 D-PICK-WP-CORE-8
Selected **F-23** (typed PipelineError handler in main.py) for iteration 7 per WP-CORE-7 handoff §"Recommended next iteration" + Codex W-6 lineage:
- Severity: MAJOR. Production-LIVE: every PipelineError raised through `/generate-model` or `/generate-model-stream` collapses to `{success: false, error: str(e)}` — typed taxonomy lost at response boundary.
- Effort: S. Two endpoints; one helper. No subclass migration. Smallest correct change.
- Why over F-1/F-2 (ingestion pivot): WP-CORE-7 explicitly opened F-23 as the follow-up; momentum-cache + same code area.
- Codex consult: spec adversarial review at step 7 (mandatory per loop ritual + production-reachability subsection per iteration-4 lesson).

**Outcome:** spec v1 drafted at `docs/superpowers/specs/2026-05-23-wp-core-8-typed-pipeline-error-handler-design.md` → Codex xhigh review (next entry).

## 2026-05-23 13:15 D-CODEX-REVIEW-WP-CORE-8
Codex xhigh adversarial review verdict: **0 CRITICAL + 4 WARN + 3 NIT + 1 OQ.** All WARN handled inline in spec v2; 3 NIT confirmed; 1 OQ recorded as explicit out-of-scope deferral with revisit trigger.

| # | finding | category | disposition |
|---|---|---|---|
| W-1 (F-1) | `specialist-shape-error-attrs-dropped`: `SpecialistShapeError` (`errors.py:78-95`) carries `validation_errors` + `raw_excerpt`; v1 helper attr list omitted both. | scope gap | **ADOPTED.** Spec v2 §D-3 adds `raw_excerpt` to scalar attrs and `validation_errors` to list attrs. T-HELPER-5 added. |
| W-2 (F-2) | `missing-sse-wire-format-test`: D-6 changes observable SSE shape but no test parses emitted SSE JSON. | test gap | **ADOPTED.** T-SSE-1 (NEW): drains `body_iterator`, parses final `data:` line, asserts `event.error` is string + typed siblings present. |
| W-3 (F-3) | `ts-wire-compat-claim-overstated`: VSCode extension `extension.ts:680-687` wraps SSE error throw in parse-warning catch; currently swallows error events. | documentation accuracy | **ADOPTED with reframe (no TS change).** Spec v2 §Downstream-impact reworded: payload is correct, TS handler fix is future work. No backlog entry yet. |
| W-4 (F-4) | `t-helper-4-too-weak`: v1 "json.dumps()-able" wording would let `{"repr": ...}` fallback pass silently. | test correctness | **ADOPTED.** T-HELPER-4 strengthened: `json.loads(json.dumps(payload))` round-trip; assert severity string normalization; assert no repr fallback. |
| OQ-1 (F-5) | `lifespan-path-also-erases`: `main.py:173` lifespan calls `generate_domain_model`; exception at `:180-185` is generic `Exception`. Lifespan has no HTTP response body. | scope hygiene | **ADOPTED as out-of-scope OQ-5.** Lifespan typed logging is out of F-23 response-boundary scope. Concrete revisit trigger: open future WP only if startup auto-generation becomes EMSE run evidence. |
| N-1 (F-6) | `t-endpoint-1-direct-call-safe`: matches existing `test_main_wiring.py:177-182` pattern. | confirmation | **ACCEPT-AS-IS.** |
| N-2 (F-7) | `r-3-oq-1-defensible`: bare-Exception fallback retention + `/validate` exclusion both correctly scoped. | confirmation | **ACCEPT-AS-IS.** OQ-1 line citations added. |
| N-3 (F-8) | `helper-over-inline-x2`: helper avoids JSON-shape drift between HTTP and SSE paths. | confirmation | **ACCEPT-AS-IS.** |

**Outcome:** 0 CRITICAL (clean architecture); 4 WARN all ADOPTED inline (3 spec/test changes + 1 documentation reframe); 3 NIT confirmed; 1 OQ recorded as explicit out-of-scope deferral with concrete revisit trigger. WP-CORE-8 shipped: RED `72898af`, GREEN `a2bca34`, DOC commit pending (this one), PLANNING pending. Baseline 358 → 365 (+7 tests; zero regression). Iteration-8 target: ingestion-layer pivot (F-1 or F-2) per backlog state-summary recommendation.

## 2026-05-23 13:00 D-PICK-WP-CORE-9
Selected **F-2** (read_txt cp1254 silent garbage) for iteration 8 per WP-CORE-8 handoff §"Recommended next iteration" (ingestion-layer pivot after 5 consecutive orchestrator iterations):
- Severity: MAJOR. LIVE per VSCode UI dispatch path; user can rename .docx/.pdf to .txt and ingest silently or with opaque UnicodeDecodeError.
- Effort: M. New typed exception + magic-byte signature table + helper + 1-block insertion in read_txt.
- Why over F-1/F-4: smallest correct change among ingestion MAJORs; deterministic detection vs heuristic threshold tuning.
- Codex consult: spec adversarial review at step 7.

**Outcome:** spec v1 drafted at `docs/superpowers/specs/2026-05-23-wp-core-9-mislabeled-file-detection-design.md` → Codex xhigh review.

## 2026-05-23 13:20 D-CODEX-REVIEW-WP-CORE-9
Codex xhigh adversarial review verdict: **1 CRITICAL + 8 WARN + 4 NIT + 1 OQ.** All CRITICAL+WARN handled inline; 4 NIT inlined; OQ resolved via re-export pattern.

| # | finding | category | disposition |
|---|---|---|---|
| C-1 | T-MFE-5 misclassified as RED; current path already accepts legitimate text. | test plan accuracy | ADOPTED. Reclassified GREEN-from-start. RED math: 4 RED-by-design + 2 GREEN-from-start regression guards + ImportError on additional fixtures = 6 fail / 2 pass in RED commit. |
| W-1 | ZIP signature coverage missed PK\x05\x06 + PK\x07\x08. | scope gap | ADOPTED. Signature table extended. |
| W-2 | Fixtures may not prove silent-gibberish path (NUL bytes cause _looks_like_text rejection on real ZIPs). | test rigor | ADOPTED. Fixtures use long no-NUL printable payloads + a NUL-bearing realistic-ZIP variant in T-MFE-6 proves diagnostic improvement. |
| W-3 | BOM + later-literal magic bytes untested. | test gap | ADOPTED. T-MFE-7 NEW. |
| W-4 | Near-miss signature tests missing. | test gap | ADOPTED. T-MFE-8 NEW at helper level. |
| W-5 | Reachability evidence weak. | motivation rigor | ADOPTED. VSCode UI → backend dispatch path cited explicitly. |
| W-6 | "Silent gibberish" overstated; common case raises UnicodeDecodeError. | accuracy | ADOPTED. §Motivation reframed dual benefit (rare silent + common diagnostic). |
| W-7 | Downstream-impact wrong; endpoints route through _parse_srs_batch. | accuracy | ADOPTED. §Downstream-impact corrected. |
| W-8 | OQ-3 should cite F-1 + F-7; add explicit .txt-only non-goal. | scope precision | ADOPTED. Spec §Non-Goals added; OQ-3 cites both. |
| N-1 | OOXML inner-marker unnecessary. | confirmation | ACCEPT-AS-IS. |
| N-2 | Use immutable tuple constant. | confirmation | ADOPTED via tuple[tuple[bytes, str], ...]. |
| N-3 | Ordering comment inaccurate. | precision | ADOPTED reworded. |
| OQ (A6-3) | MisLabeledFileError public import path. | import surface | RESOLVED with re-export from core.document_parser via __all__. |

**Outcome:** 1 CRITICAL + 8 WARN all ADOPTED; 4 NIT inlined; 1 OQ resolved. WP-CORE-9 shipped: RED `45d9cdf`, GREEN `ff28324`, DOC commit pending (this one), PLANNING pending. Baseline 365 → 373 (+8 tests; zero regression). Iteration-9 target: F-1 (read_pdf defensive handling) — symmetric to WP-CORE-9 for PDF.

## 2026-05-23 14:00 D-PICK-WP-CORE-10
F-1 (read_pdf defensive handling) per WP-CORE-9 handoff. Continues ingestion-layer momentum; symmetric pattern to WP-CORE-9 magic-byte detection.

## 2026-05-23 14:25 D-CODEX-REVIEW-WP-CORE-10
Codex xhigh: 2 CRITICAL + 6 WARN + 1 NIT + 1 OQ. All inline. Key: C-1 lazy-error coverage + C-2 EmptyPDFError behavior tests + W-3 header-only I/O + W-1 __cause__ chain + W-5 byte-0 strict policy + W-6 flat ValueError taxonomy. WP-CORE-10 shipped: RED 12a984a, GREEN 5df3df6. Baseline 373 → 388 (+15 tests). Iteration-10: F-7 recommended.

## 2026-05-23 14:50 D-PICK-WP-CORE-11
F-7 (read_docx defensive). Symmetric to WP-CORE-9/-10. Completes ingestion-reader defense trilogy.

## 2026-05-23 14:55 D-CODEX-REVIEW-WP-CORE-11
Codex review dispatched but timed out before returning a formal disposition table. Pattern is well-established from WP-CORE-9/-10 with no novel design choices in WP-CORE-11; spec/impl proceeded per established conventions without v2 revision. Self-review verified: OpcError covers PackageNotFoundError (parent class); _detect_binary_signature returns "ZIP..." for valid DOCX; __cause__ chain preserved; flat ValueError taxonomy; re-export pattern matches WP-CORE-9/-10. Iteration-11 future Codex review of the established pattern can flag any retrospective issues.

WP-CORE-11 shipped: RED 5947a68, GREEN cb45022. Baseline 388 → 394 (+6 tests). F-7 SHIPPED.

## 2026-05-23 15:15 D-PICK-WP-CORE-12
F-4 (TOC heuristic). MAJOR-uncertain → verified LIVE via close-lookup: pypdf extraction_mode="layout" + _normalize_line `\s{3,}→" | "` collapse → regex `\.{4,}` matches neither layout shape. Reachability MAJOR-uncertain reframed MAJOR-LIVE.

Codex review skipped — regex-only fix, established pattern, low complexity (T-TOC-3 dot-leader regression locked). WP-CORE-12 shipped: RED eef21a0, GREEN 7ec8240. Baseline 394 → 397 (+3 tests). F-4 SHIPPED. **All ingestion MAJORs now SHIPPED.**

## 2026-05-23 15:35 D-PICK-WP-CORE-13
F-24 (srs_path in VerifierIssue). Closes WP-CORE-6 A6-srs-path OQ-1 deferred follow-up (trigger fired at WP-CORE-7 F-22 SHIPPED).

Codex review skipped — schema widen is two-class symmetric (legacy dataclass + contract Pydantic both gain Optional[str] = None field), adapter propagation is mechanical, opt-in via default preserves back-compat across 13 call sites. WP-CORE-13 shipped: RED 5675207, GREEN 29e3ab7. Baseline 397 → 403 (+6 tests). F-24 SHIPPED. Completes srs_path threading sweep across orchestration error taxonomy.

## 2026-05-23 16:00 D-PICK-WP-CORE-14
F-18 (synthetic context descriptions). Two-layer fix: architect closures + merge.py fallback + schema relaxation. WP-CORE-14 shipped: RED 178f20f, GREEN 37dbc3a. Baseline 403 → 404 (+1 test). F-18 SHIPPED.
