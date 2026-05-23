# WP-CORE-11 — `read_docx` defensive handling

**Status:** SHIPPED 2026-05-23
**Commits:** RED `5947a68` → GREEN `cb45022` → DOC `{this}` → PLANNING `{pending}`
**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-11-docx-defensive-handling-design.md`
**Parent finding:** F-7 (MINOR) — SHIPPED.

## TL;DR

`read_docx` was 14 LOC zero-defense. Renamed binaries / corrupted DOCX / empty DOCX surfaced as opaque `docx.opc.exceptions.PackageNotFoundError` or downstream misleading "empty document". WP-CORE-11 adds `CorruptedDOCXError` + `EmptyDOCXError` typed exceptions; re-uses WP-CORE-9 `_detect_binary_signature` for symmetric `MisLabeledFileError` on non-ZIP byte-0; completes the ingestion-reader defense trilogy with WP-CORE-9 (TXT) + WP-CORE-10 (PDF).

Baseline 388 → 394 (+6 tests).

## Key decisions

- DOCX = ZIP archive (`PK\x03\x04`); magic-byte check matches `detected.startswith("ZIP")`.
- `OpcError` is the parent of `PackageNotFoundError` and other docx OPC failures; single `except OpcError` covers.
- `__cause__` chain preserved via `raise ... from`.
- Flat `ValueError` taxonomy (consistent with WP-CORE-10).
- Re-export via `core.document_parser.__all__`.

## File-level changes

| File | LOC |
|---|---|
| `core/document_parser_readers.py` | +60 (2 exceptions + defensive read_docx) |
| `core/document_parser.py` | +4 (re-export) |
| `tests/test_document_parser_docx_defensive.py` (NEW) | +135 (6 tests) |

## Limitations + follow-ups

- Encrypted DOCX (password-protected via OLE compound or DOCX encryption) → magic-byte catches non-ZIP OLE compound case; in-ZIP encrypted DOCX raises corruption-flavored `OpcError`. Real-world coverage acceptable.
- `read_docx` does NOT add OCR fallback; image-only DOCX has no extractable text → `EmptyDOCXError`.

## Cross-references

- Predecessor: `[[WP-CORE-10-pdf-defensive-handling]]` — same pattern.
- **WP-CORE-11 NEW invariant**: `read_docx` rejects non-ZIP byte-0 content; raises typed exceptions for corrupt/empty DOCX.
- Ingestion-reader defense trilogy complete (`.txt` / `.pdf` / `.docx`).
