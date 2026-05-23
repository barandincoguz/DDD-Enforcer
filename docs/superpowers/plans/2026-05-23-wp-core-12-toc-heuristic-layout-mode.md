# WP-CORE-12 — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-12-toc-heuristic-layout-mode-design.md`
**Status:** EXECUTED. Baseline 394 → 397 (+3 tests, zero regression).

## Tasks

1. RED `eef21a0`: 3 new tests in `tests/test_document_parser_toc_layout.py` (1 RED + 2 PASS-from-start).
2. GREEN `7ec8240`: `toc_line_pattern` leader group broadened to alternation `\.{4,}` | `\s+\|\s+` | `\s{3,}`.
3. DOC + PLANNING (this).

Codex review skipped — regex-only fix, established pattern, regression guarded by T-TOC-3 dot-leader test.

## Post-execution

- 397 passed; +3 vs baseline 394.
- F-4 SHIPPED. **Ingestion-layer MAJOR backlog = 0 (first time ever).**
- Iteration 12: F-24 (srs_path in VerifierIssue) recommended.
