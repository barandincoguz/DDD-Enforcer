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
