# WP-CORE-11 — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-11-docx-defensive-handling-design.md`
**Status:** EXECUTED (RED `5947a68`, GREEN `cb45022`, DOC + PLANNING in flight).
**Baseline:** 388 → 394 (+6 tests, zero regression).

## Tasks

1. RED `5947a68`: 6 new tests in `tests/test_document_parser_docx_defensive.py`. 5 RED-by-design + 1 GREEN-from-start happy-path.
2. GREEN `cb45022`: 2 typed exceptions + magic-byte check + OpcError wrap + EmptyDOCXError gate in `read_docx`. Re-exported via `__all__`.
3. DOC: dev_doc + INDEX + CURRENT + backlog + decision_log.
4. PLANNING (this): spec + plan into git.

## Codex review

Dispatched but timed out before returning a formal disposition. Pattern was well-established from WP-CORE-9/-10 with no novel design questions in WP-CORE-11 (symmetric reader-defense, single OpcError parent catch, established re-export pattern). Self-review verified the established conventions hold. Future iterations can apply retrospective fixes if Codex flags issues.

## Post-execution

- 394 passed; +6 vs baseline 388.
- 4 atomic commits.
- F-7 SHIPPED. Ingestion-reader defense trilogy COMPLETE.
- Iteration 11: F-4 (TOC heuristic refactor) recommended — last ingestion-layer MAJOR.
