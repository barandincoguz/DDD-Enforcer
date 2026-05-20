# WP-CORE-2 — Reference-heading truncation correctness

**Status:** SHIPPED
**Branch:** main
**Pre-WP SHA:** 029f187
**RED commit SHA:** 4f932d2
**GREEN commit SHA:** 25e6880
**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-2-reference-truncate-fix-design.md` (v2 — Codex-reviewed)
**Plan:** `docs/superpowers/plans/2026-05-21-wp-core-2-reference-truncate-fix.md`
**Audit finding:** `.planning/pipeline_audit/findings/document_parser.md` F-5
**Codex review summary:** `.planning/pipeline_audit/decision_log.md` D-CODEX-REVIEW-WP-CORE-2

## TL;DR

`SRSDocumentParser._truncate_at_references` had two real silent-content-loss defects: it matched Turkish `kaynakça` but not the more common plural `Kaynaklar`, and the existing regex permitted mid-document false-positives on numbered subsections like `3.4 References`. The fix adds `kaynaklar` and an optional trailing-colon variant to the regex, and bounds the truncation scan to the latter half of the document via a single named constant (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`). All seven new integration tests and 26 parametrized grammar cases pass, and the prior 272 baseline became 305 (272 + 33 new tests, all green).

## Motivation

Finding F-5 (`.planning/pipeline_audit/findings/document_parser.md`) flagged two failure modes in the post-normalization truncation pass at `extension/backend/core/document_parser.py:9-12` and `:60-65`:

- **Locale gap.** The keyword alternation listed `kaynakça` but not the more common Turkish plural `Kaynaklar`. A Turkish-authored SRS using `Kaynaklar` as the bibliography heading was ingested *with* the bibliography section — silently polluting the Scout corpus.
- **Mid-document false-positive.** The regex's optional section-number prefix `(?:\d+(?:\.\d+)*\.?\s+)?` permitted any depth of dotted nesting, so a legitimate subsection titled exactly `3.4 References` (with no following text on the line) would silently truncate everything below it.

Both were within a single regex string and a single helper. Codex xhigh adversarial review escalated the v1 plan (regex-narrow only) because narrowing `\d+(?:\.\d+)*` → `\d+` would silently regress legitimate `5.1 References` bibliography subsections in appendix-style layouts. v2 instead bounds the *scan window* to the latter half of the document.

## Architectural decisions

1. **Position guard over regex narrowing.** Codex WARN-1 surfaced that narrowing the nested-number prefix would block a legitimate bibliography subsection layout. The position guard keeps the original regex permissiveness while structurally enforcing "bibliography lives in the latter half of an SRS" — a content-domain truth, not a syntactic accident.
2. **Single named class constant.** `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5` carries the magic number with a 2-line rationale comment. Per AGENTS.md, "explicit structure over hidden complexity": the constant is named and lives at the class top-of-body, not embedded in the loop.
3. **Multi-line regex with adjacent string literals.** The new regex was assembled from four adjacent string literals (Python compile-time concatenation, no `+`) to keep each clause readable in isolation. No semantic change vs. a single-line form.
4. **Direct parametrized regex test.** Codex WARN-5: integration tests via `parse_file` did not provide auditable grammar coverage. A 26-case `pytest.mark.parametrize` against `SRSDocumentParser().reference_heading_pattern.match(line.strip())` was added to catch any future drift in the grammar.
5. **Out-of-scope Codex WARNs intentionally retained.** Multiword Turkish (`Yararlanılan Kaynaklar`), code-fence guards, Unicode lookalike resilience, and ASCII-folded `kaynakca` are all tracked as future hardening in `decision_log.md` D-CODEX-REVIEW-WP-CORE-2. Speculative coverage rejected per AGENTS.md "no speculative generalization."

## File-level changes

| file | change |
|---|---|
| `extension/backend/core/document_parser.py:7-8` | New class constant `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5` + 2-line rationale comment, immediately before `__init__`. |
| `extension/backend/core/document_parser.py:9-12` | Regex pattern expanded across 4 adjacent string literals; `|kaynaklar` added to alternation; `(?:\s*[:：])?` (ASCII or U+FF1A fullwidth colon) added before `\s*$`. Nested-number prefix preserved unchanged. |
| `extension/backend/core/document_parser.py:60-65` | `_truncate_at_references` body replaced with a position-guarded loop: `if not lines: return text`; `earliest_match_index = int(len(lines) * self.REFERENCE_HEADING_MIN_DOCUMENT_FRACTION)`; loop scans `range(earliest_match_index, len(lines))` only. |
| `extension/backend/tests/test_document_parser.py` | Appended T1–T7 integration tests + `test_reference_heading_pattern_direct_grammar` parametrize with 26 cases (+177 LOC). |

## Methodology

- Spec written, then Codex xhigh adversarial review on v1. Spec revised to v2 incorporating WARN-1 (position guard), WARN-2a (trailing colon), WARN-5 (parametrize block). Other 4 WARNs accepted with rationale.
- TDD red-green per task. RED commit established failing tests against unchanged production. GREEN commit applied the three coordinated production edits. Atomic commits with conventional-commit prefixes (`test(parser)`, `fix(parser)`).
- Pytest baseline checked at every transition: pre-WP 272 / 31 / 0; after RED 296 / 31 / 9; after GREEN 305 / 31 / 0. Full suite is the canonical gate.
- No new dependency. No imports changed. No facade or call-site changes elsewhere in the codebase. Verified by `git diff 029f187..HEAD --stat`.

## Empirical results

- Pre-WP pytest baseline: **272 passed, 31 deselected**.
- Post-WP pytest baseline: **305 passed, 31 deselected**. Net +33 (7 integration + 26 parametrize). Zero regression.
- Live D1 SRS E2E re-run skipped — regex change strictly *reduces* false-positive matches and adds an alternation that does not affect English `References` (D1 corpus). The new position guard cannot regress any existing passing test because every existing test where the regex was expected to fire already has the heading in the latter half (verified per-test in the spec).

## Limitations & follow-ups

- Multiword Turkish headings (e.g. `Yararlanılan Kaynaklar`) still not matched. Tracked as follow-up if a D2/D3 corpus exhibits the pattern.
- Code-fence / preformatted-text guard not implemented; position guard mitigates the common case but a heading-style line embedded in a code fence in the latter half can still false-positive. Future hardening.
- Unicode lookalike defenses (BiDi marks, Cyrillic homoglyphs) not implemented. Future hardening.
- ASCII-folded `kaynakca` (no diacritic) intentionally excluded pending corpus evidence.
- `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5` is a single, named magic number. If a D2/D3 corpus uses bibliography placement past mid-doc but within the last quarter, the value can be tuned; tests will catch the change.

## Cross-references

- Parent loop: `.planning/pipeline_audit/CURRENT.md`
- Audit catalog: `.planning/pipeline_audit/component_catalog.md`
- Finding F-5: `.planning/pipeline_audit/findings/document_parser.md`
- Decision log entries: `.planning/pipeline_audit/decision_log.md` D-CL1, D-PICK-WP-CORE-2, D-CODEX-REVIEW-WP-CORE-2, D-SHIP-WP-CORE-2
- Related future WPs: F-1 (PDF defensive handling), F-2 (.txt printability), F-3 (empty-input contract), F-4 (TOC heuristic), F-6 (`_should_merge` regex completeness), F-7 (DOCX try/except), F-8 (XXE), F-9 (logging), F-10 (double-parse)
- Engineering charter: `AGENTS.md`
- Project conventions: `CLAUDE.md`

## Backlinks

- `[[WP-CORE-1-typed-pipeline-deterministic-synthesizer]]` — parent typed-pipeline refactor (placeholder backfill if doc not yet written).
- `[[WP-NEW-B-Stage-1-schema-probe]]` — parallel schema-conformance probe work.
