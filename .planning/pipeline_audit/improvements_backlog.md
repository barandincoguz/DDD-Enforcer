# Improvements Backlog — Domain Pipeline Hardening

Each row: `id | component | finding | severity | effort | blast | status`.

- `severity` ∈ {BLOCKER, MAJOR, MINOR, TRIVIAL}
- `effort` ∈ {S (≤2h), M (≤1d), L (>1d)}
- `blast` ∈ {LOCAL, MODULE, PIPELINE, REPO}
- `status` ∈ {OPEN, IN-PROGRESS, SHIPPED, REJECTED, DEFERRED}

## Open

| id | component | finding | severity | effort | blast | status |
|---|---|---|---|---|---|---|
| F-1 | document_parser_readers.py | `read_pdf` has no defensive handling for encrypted/image-only/malformed PDFs — propagates raw pypdf errors + emits empty string on image-only PDFs (pypdf2:16-19, 22-27). | MAJOR | S | PIPELINE | OPEN |
| F-2 | document_parser_readers.py | `read_txt` silently emits binary garbage when `_looks_like_text` passes a near-binary file under cp1254 (`:92-109`, `:120-126`). | MAJOR | S | PIPELINE | OPEN |
| F-4 | document_parser.py | TOC heuristic anchored to first 120 lines + `cluster < 2` drop; layout-mode PDFs leak TOC entries into Scout (`:81-101`, `:103-117`). | MAJOR (uncertain) | M | PIPELINE | OPEN |
| F-6 | document_parser.py | `_should_merge` only checks `[.!?;:]$`; quote-terminated / bracket-terminated / Unicode-ellipsis lines collapse silently; soft-hyphen vs compound-word hyphen indistinct (`:154-165`). | MINOR | S | LOCAL | OPEN |
| F-7 | document_parser_readers.py | DOCX reader has zero try/except around `docx.Document(file_path)`; `PackageNotFoundError` propagates raw (`:30-31`). | MINOR | S | LOCAL | OPEN |
| F-8 | document_parser_readers.py | No explicit XXE / external-entity hardening on lxml XML parsing (defense-in-depth gap visible to EMSE reviewers) (`:31`). | MINOR (uncertain) | S | REPO | OPEN |
| F-9 | document_parser.py + readers | Zero logging anywhere; PDF layout→plain downgrade invisible to WP-NEW-B run manifest. | MINOR | S | PIPELINE | OPEN |
| F-10 | document_parser.py + `main.py:366,480` | Same SRS re-parsed twice per `/generate-model` request (Architect input + RAG indexing) — duplicate I/O, no memoization. | TRIVIAL | M | LOCAL | OPEN |

## Shipped

| id | component | finding | severity | effort | blast | status |
|---|---|---|---|---|---|---|
| F-5 | document_parser.py | `_truncate_at_references` matched Turkish `kaynakça` but NOT `Kaynaklar`; also false-positive on numbered `3.4 References` mid-document. Fixed via regex alternation expansion + optional trailing colon + position guard (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`). | MAJOR | S | PIPELINE | **SHIPPED (25e6880)** |
| F-3 | document_parser.py + main.py | `parse_file` returned empty string for empty inputs; 6 duplicated guards in `main.py` (3 HARD, 3 SOFT, 2 of the HARD broken because separator-headers made `combined_text.strip()` always non-empty). Fixed: `EmptySRSDocumentError(ValueError)` raised by `parse_file`; per-path policy at callers (HARD propagate / SOFT skip+log / MIXED batch via new `_parse_srs_batch` helper); aggregate check switched to `srs_docs` emptiness. Latent post-loop-guard bug folded in. | MAJOR | S | MODULE+PIPELINE | **SHIPPED (daefeb0)** |

## Rejected / Deferred

_(empty)_

---

**Decision priority:** production bug fix > test-coverage critical gap > measurable perf regression > evidence-backed clarity smell > cosmetic.

**Last refresh:** 2026-05-21 09:37 GMT+3
