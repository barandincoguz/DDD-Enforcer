# WP-CORE-10 — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-10-pdf-defensive-handling-design.md` (v2)
**Status:** EXECUTED (RED `12a984a`, GREEN `5df3df6`, DOC `{prev}`, PLANNING `{this}`)
**Baseline:** 373 → 388 (+15 tests, zero regression).

## Tasks

1. RED `12a984a`: 15 new tests in `tests/test_document_parser_pdf_defensive.py`. 12 RED-by-design + 3 GREEN-from-start (T-PDF-MIXED, T-PDF-INHERIT, T-PDF-HAPPY).
2. GREEN `5df3df6`: `core/document_parser_readers.py` adds 3 typed exceptions + `_MAGIC_HEADER_BYTES` + defensive `read_pdf`. `core/document_parser.py` re-exports.
3. DOC: dev_doc + INDEX + CURRENT + backlog + decision_log + handoff.
4. PLANNING (this): spec v2 + plan.

## Codex review

2 CRITICAL (lazy-error coverage + EmptyPDFError behavior tests) + 6 WARN (chain, taxonomy guard, header-only I/O, single-pass dispatch, strict byte-0 policy, flat ValueError taxonomy) + 1 NIT + 1 OQ — all inline.

## Post-execution

- 388 passed; +15 vs baseline 373.
- 4 atomic commits.
- F-1 SHIPPED.
- Iteration 10 handoff at `.planning/pipeline_audit/handoff-2026-05-23-1430.md`.
