# WP-CORE-10 — `read_pdf` defensive handling

**Status:** SHIPPED 2026-05-23
**Commits:** RED `12a984a` → GREEN `5df3df6` → DOC `{this}` → PLANNING `{pending}`
**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-10-pdf-defensive-handling-design.md` (v2)
**Parent finding:** F-1 (MAJOR) — now SHIPPED.

## TL;DR

`read_pdf` was 3 LOC with zero defensive handling — opaque `PdfReadError` on corruption, silent empty pages on encryption, misleading "empty document" downstream for mislabeled/image-only PDFs. WP-CORE-10 adds three typed exceptions (`EncryptedPDFError`, `CorruptedPDFError`, `EmptyPDFError`), header-only magic-byte check re-using WP-CORE-9's `_detect_binary_signature` for symmetric `MisLabeledFileError` on non-PDF byte-0 content, and wraps both constructor + lazy page/extract failures under one `except PdfReadError` block.

Baseline: 373 → 388 (+15 tests, zero regression).

## Architectural decisions

### D-1 — Symmetric pattern with WP-CORE-9
Same magic-byte detection helper reused. Single-pass dispatch: `_detect_binary_signature(header)` returns "PDF" for legitimate PDFs (signature `b"%PDF-"`); anything else → `MisLabeledFileError`.

### D-2 — Header-only I/O (Codex W-3)
`with open(path, "rb") as f: header = f.read(_MAGIC_HEADER_BYTES)` reads only the first 8 bytes instead of `read_bytes()` reading the entire (potentially multi-GB) PDF. `_MAGIC_HEADER_BYTES = max(len(prefix) for prefix, _ in _BINARY_MAGIC_SIGNATURES)` is computed once at module load.

### D-3 — Lazy-error coverage (Codex C-1)
The `except PdfReadError` block wraps `PdfReader.__init__`, `is_encrypted` check, `len(reader.pages)`, AND per-page `extract_text`. T-PDF-INHERIT (Codex W-2) verifies `PdfStreamError` is a `PdfReadError` subclass so one except catches both. Without lazy coverage, page-stream errors would escape as opaque pypdf exceptions.

### D-4 — Strict byte-0 policy (Codex W-5)
Leading whitespace before `%PDF-` is rejected — stricter than pypdf's tolerant parsing. Consistent with WP-CORE-9's `startswith` policy. T-PDF-STRICT locks this.

### D-5 — Flat ValueError taxonomy (Codex W-6)
`EmptyPDFError(ValueError)` not `EmptyPDFError(EmptySRSDocumentError)`. Trade-off: callers catching `EmptySRSDocumentError` for PDFs won't auto-catch this; by design, since PDF-specific failure modes (encryption, corruption, image-only) deserve typed distinction.

### D-6 — Re-export for clean import path
`from core.document_parser import EncryptedPDFError, CorruptedPDFError, EmptyPDFError` (mirror WP-CORE-9 `MisLabeledFileError`).

## File-level changes

| File | LOC |
|---|---|
| `core/document_parser_readers.py` | +110 (3 exceptions + `_MAGIC_HEADER_BYTES` + defensive read_pdf) |
| `core/document_parser.py` | +5 (re-export 3 exceptions) |
| `tests/test_document_parser_pdf_defensive.py` (NEW) | +341 (15 tests) |

## Empirical results

Baseline 373 → 388 (+15 tests, zero regression).

## Limitations + follow-ups

- **F-7 (DOCX zero try/except)** uncovered by WP-CORE-10; same pattern re-applicable.
- **Strict byte-0 policy** may reject some legitimate PDFs with leading whitespace/comments — accepted trade-off; relax via separate WP if real-world data demands.
- **No OCR fallback** for image-only PDFs — `EmptyPDFError(reason="image-only")` surfaces; users must run OCR offline.

## Cross-references

- **Predecessor**: `[[WP-CORE-9-mislabeled-file-detection]]` — magic-byte helper reused.
- **WP-CORE-10 NEW invariant**: `read_pdf` rejects non-`%PDF-` byte-0 content via `MisLabeledFileError` and raises typed exceptions for encryption/corruption/empty.
- **EMSE paper**: PDF ingestion-layer error taxonomy now fully typed end-to-end.
