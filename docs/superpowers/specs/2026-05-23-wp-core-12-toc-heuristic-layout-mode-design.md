# WP-CORE-12 — TOC heuristic layout-mode reachability fix (F-4)

**Date:** 2026-05-23
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 11)
**Status:** DRAFT v1
**Parent finding:** `findings/document_parser.md` **F-4** (MAJOR-uncertain — reachability NOW VERIFIED).
**Loop:** baseline 394 at HEAD `a5cccc3`.

## Motivation

`SRSDocumentParser._find_toc_line_indexes` (`document_parser.py:133-153`) filters out TOC lines so they don't leak into Scout input. The detection relies on `self.toc_line_pattern` at line 59-62:

```python
r"^(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\.?\s+)?[A-Za-z0-9ÇĞİÖŞÜçğıöşü].*\.{4,}\s*\d+\s*$"
```

This requires `\.{4,}` (4+ literal dots) before the trailing page number. Traditional TOCs use dot leaders (`"Section 3 ........ 14"`).

**The bug**: `read_pdf` extracts via `extraction_mode="layout"` (`document_parser_readers.py:270`). Layout mode preserves visual structure by replacing dot-leader patterns with multiple spaces — TOC lines become `"Section 3       14"` (whitespace-separator). `_normalize_line` at line 171-175 collapses 3+ spaces to `" | "` for table preservation, producing `"Section 3 | 14"`. The current pattern requires literal `\.{4,}` and matches NEITHER form. TOC entries leak through to Scout as "domain sentences" → pollute the bounded-context graph.

**Severity reframe**: F-4 was MAJOR-uncertain because reachability needed confirmation. WP-CORE-12 confirms LIVE: every PDF SRS using layout-mode extraction (the project default) has this leak.

### Production reachability

LIVE: `extension/backend/core/document_parser_readers.py:270` always uses `extraction_mode="layout"` for PDF text extraction. Any SRS PDF with a TOC will leak TOC entries to Scout.

## Discovery (audit-text-vs-code-reality)

Audit F-4 cited: "PDFs rendered in pypdf layout mode often replace dot leaders with multiple spaces or tabs". Verified at HEAD: `_normalize_line` regex `r"(?<=\S)(?: {3,}|\t+)(?=\S)"` produces `" | "` for 3+ spaces or tabs; `toc_line_pattern` requires `\.{4,}` — no `|` or whitespace alternative. **Audit text accurate.**

## Design

### D-1 — Broaden `toc_line_pattern` to match BOTH dot-leader and layout-mode shapes

```python
self.toc_line_pattern = re.compile(
    r"^(?:#{1,6}\s*)?"                                                  # optional MD prefix
    r"(?:\d+(?:\.\d+)*\.?\s+)?"                                         # optional section number
    r"[A-Za-z0-9ÇĞİÖŞÜçğıöşü]"                                          # alpha-numeric first char
    r".*?"                                                              # title body (non-greedy)
    r"(?:\.{4,}\s*|\s+\|\s+|\s{3,})"                                    # leader: dots OR pipe-separator OR whitespace
    r"\d+\s*$",                                                         # trailing page number
    re.IGNORECASE,
)
```

The three alternatives in the leader group:
- `\.{4,}\s*` — traditional dot leader (legacy).
- `\s+\|\s+` — post-`_normalize_line` table-pipe separator (the current layout-mode case).
- `\s{3,}` — raw multi-space separator (in case the line bypasses `_normalize_line` for some path).

**Risk**: this broader pattern can false-positive on legitimate text lines ending with `"...    42"` patterns (e.g., requirement text ending in a numeric value). Mitigation: the `cluster < 2` and 120-line-window guards still apply; a single false-positive line gets dropped from the cluster (per `_flush_toc_cluster`).

### D-2 — Verify cluster guard still holds

`_flush_toc_cluster` (line 155-169) requires `len(cluster) >= 2` and that the previous-line "Contents" heading is matched. The broadened regex doesn't change this gate — false-positives at isolated lines still get rejected.

### D-3 — `_normalize_line` interaction

`_normalize_line` converts 3+ consecutive spaces/tabs to `" | "`. After normalization, layout-mode TOC lines look like `"Section 3 | 14"`. The new `\s+\|\s+` alternation catches this. No changes to `_normalize_line` itself.

## Test plan

| # | name | what | RED expectation |
|---|---|---|---|
| T-TOC-1 | `test_toc_layout_mode_pipe_separator_detected` | text containing `"1.1 Introduction | 1\n1.2 Scope | 2"` (post-normalize_line form) is filtered out | FAIL — current regex doesn't match `|` separator |
| T-TOC-2 | `test_toc_layout_mode_raw_whitespace_separator_detected` | text containing `"1.1 Introduction       1\n1.2 Scope       2"` (raw 3+ spaces) is filtered | FAIL — regex doesn't match `\s{3,}` |
| T-TOC-3 | `test_toc_dot_leader_still_detected` | regression — traditional `"1.1 Introduction ........ 1"` still filtered | PASS-from-start |
| T-TOC-4 | `test_toc_single_layout_line_not_dropped_via_cluster_guard` | a single line `"My requirement spec | 42"` (potential false-positive) is NOT filtered because cluster<2 | PASS-from-start (cluster guard intact) |

RED: 2 fail + 2 PASS-from-start regression. GREEN: 4 pass.

## Risks

- **R-1**: False-positive on legitimate "Section name | 42" lines that aren't TOC. Mitigation: `cluster<2` rejection.
- **R-2**: Existing TOC tests may rely on the dot-leader-only behavior. Audit `test_document_parser.py` — `test_parse_pdf_merges_wrapped_lines_and_stops_at_references` covers happy path; no explicit TOC test exists.

## Atomic commit sequence

1. RED tests
2. GREEN regex broadening
3. DOC + PLANNING (combined)

Spec v1 ready.
