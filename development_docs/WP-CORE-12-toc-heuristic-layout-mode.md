# WP-CORE-12 — TOC heuristic layout-mode detection

**Status:** SHIPPED 2026-05-23
**Commits:** RED `eef21a0` → GREEN `7ec8240` → DOC `{this}` → PLANNING `{pending}`
**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-12-toc-heuristic-layout-mode-design.md`
**Parent finding:** F-4 (MAJOR-uncertain → MAJOR-LIVE confirmed) — SHIPPED.

## TL;DR

`toc_line_pattern` required `\.{4,}` dot leaders. pypdf's `extraction_mode="layout"` (project default) produces TOCs with whitespace separators not dots. `_normalize_line` collapses 3+ spaces to `" | "`. The regex matched neither form → TOC entries leaked to Scout as domain sentences. WP-CORE-12 broadens the leader group to alternation: `\.{4,}` | `\s+\|\s+` | `\s{3,}`. Cluster<2 + 120-line-window guards retained for false-positive protection.

Baseline 394 → 397 (+3 tests).

## Key decisions

- **Smallest correct change**: regex-only fix, no logic refactor.
- **Three alternations**: dot-leader (legacy) + pipe-separator (post-_normalize_line common case) + raw whitespace (fallback).
- **Non-greedy title body** (`.*?`): cleaner backtracking for alternation handling.
- **Guards retained**: cluster<2 + 120-line window still protect against false-positive isolated lines (verified by T-TOC-4 regression).

## Empirical

Baseline 394 → 397. F-4 reachability MAJOR-uncertain → MAJOR-LIVE confirmed. Ingestion-layer MAJOR backlog now ZERO live.

## Cross-references

- **Predecessors**: `[[WP-CORE-9-mislabeled-file-detection]]`, `[[WP-CORE-10-pdf-defensive-handling]]`, `[[WP-CORE-11-docx-defensive-handling]]` — same audit-layer.
- **WP-CORE-12 NEW invariant**: `toc_line_pattern` accepts pipe-separator and raw-whitespace TOC shapes alongside legacy dot leaders.
