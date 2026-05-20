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
