# WP-CORE-9 — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-9-mislabeled-file-detection-design.md` (v2)
**Status:** EXECUTED (RED `45d9cdf`, GREEN `ff28324`, DOC `{prev}`, PLANNING `{this}`)
**Baseline:** 365 at HEAD `0e43812` (post-WP-CORE-8)
**Target:** 373 (+8 tests, zero regression)

## Tasks

1. RED `45d9cdf`: 8 new tests in `tests/test_document_parser_mislabeled_file.py` (6 RED-by-design + 2 GREEN-from-start regression guards). Pytest: 365 passed + 6 failed.
2. GREEN `ff28324`: `core/document_parser_readers.py` adds `MisLabeledFileError(ValueError)` + `_BINARY_MAGIC_SIGNATURES` tuple (10 entries) + `_detect_binary_signature` helper + pre-decode check in `read_txt`. `core/document_parser.py` re-exports via `__all__`. Pytest: 373 passed.
3. DOC: dev_doc + INDEX + CURRENT + backlog + decision_log + findings + handoff.
4. PLANNING (this): spec v2 + plan.

## Codex review summary

1 CRITICAL (T-MFE-5 reclassification) + 8 WARN (all spec/test additions) + 4 NIT (inlined) + 1 OQ (re-export resolution). All inline.

## Post-execution

- Pytest 365 → 373 (+8, zero regression).
- 4 atomic commits with Claude trailer.
- No `git push`.
- F-2 SHIPPED.
