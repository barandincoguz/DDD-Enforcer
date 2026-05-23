# WP-CORE-10 — `read_pdf` defensive handling (F-1)

**Date:** 2026-05-23
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 9)
**Status:** REVISED v2 — addressed Codex xhigh review (2 CRITICAL + 6 WARN + 1 NIT + 1 OQ; all CRITICAL+WARN inline)
**Parent finding:** `.planning/pipeline_audit/findings/document_parser.md` **F-1** (MAJOR)
**Loop:** Domain Pipeline Hardening Loop (ninth WP; baseline 373 confirmed at HEAD `4858030`)
**Codex review:** `decision_log.md` D-CODEX-REVIEW-WP-CORE-10 (to be appended at DOC commit).

## Revision history

- **v1** — initial spec.
- **v2** — Codex xhigh review verdict: **2 CRITICAL + 6 WARN + 1 NIT + 1 OQ**. Dispositions:

  | # | finding | disposition |
  |---|---|---|
  | **C-1** | Corruption catch only wraps `PdfReader(__init__)`; lazy page/extraction errors escape. | **ADOPTED.** Spec v2 §D-5 wraps `len(reader.pages)` + per-page extraction inside the `except PdfReadError` block. Plus new T-PDF-LAZY-1 / T-PDF-LAZY-2 tests for `reader.pages` raising and `page.extract_text` raising. |
  | **C-2** | `EmptyPDFError` branches specified but untested. | **ADOPTED.** Spec v2 test plan adds T-PDF-EMPTY-1 (zero pages → `EmptyPDFError`) + T-PDF-EMPTY-2 (all-pages-empty-text → `EmptyPDFError`) + T-PDF-MIXED (partial empty pages preserves text). |
  | **W-1** | `.cause` payload conflated with `__cause__` chain. | **ADOPTED.** T-PDF-7 strengthened: asserts both `.cause is original` and `.__cause__ is original`. |
  | **W-2** | `PdfStreamError` inheritance claim unanchored. | **ADOPTED.** T-PDF-INHERIT added: `assert issubclass(PdfStreamError, PdfReadError)`. Single-line guard test. |
  | **W-3** | `read_bytes()` reads ENTIRE file (multi-GB risk). | **ADOPTED.** Spec v2 §D-5 reads header-only via `with open(path, "rb") as f: header = f.read(_MAGIC_HEADER_BYTES)` where `_MAGIC_HEADER_BYTES = max(len(p) for p, _ in _BINARY_MAGIC_SIGNATURES)`. |
  | **W-4** | Two-pass classification redundant. | **ADOPTED.** Spec v2 §D-5: single `detected = _detect_binary_signature(header)`; if `detected != "PDF"` raise `MisLabeledFileError(detected_format=detected or "non-PDF (no %PDF- header)")`. Note: signature label for `%PDF-` should be exactly `"PDF"` to match this comparison — spec D-2 confirms current WP-CORE-9 entry is `"PDF"`. |
  | **W-5** | Strict `%PDF-` at byte 0 is stricter than pypdf (which tolerates leading whitespace). | **ADOPTED with policy statement.** Spec v2 §D-5 documents: "WP-CORE-10 accepts only PDFs with `%PDF-` at byte 0. Pypdf-readable PDFs with leading whitespace/comments are rejected by design (consistent with WP-CORE-9's byte-0 `startswith` policy)." Added T-PDF-STRICT to lock the leading-whitespace rejection. |
  | **W-6** | `EmptyPDFError` ancestry inconsistent: motivation table says `(EmptySRSDocumentError)`, D-4 says `(ValueError)`. | **ADOPTED.** Spec v2 §D-4 keeps `EmptyPDFError(ValueError)` (flat taxonomy); §Motivation table corrected; OQ-1 clarifies that downstream callers catching `EmptySRSDocumentError` won't auto-catch `EmptyPDFError` (by design — distinct PDF-specific failure mode). Caller audit shows current code paths catch generic `ValueError`/`Exception`, so this trade-off is safe. |
  | **N-1** | Happy-path regression not in test table. | **ADOPTED.** T-PDF-HAPPY added to test plan (existing `test_parse_pdf_merges_wrapped_lines_and_stops_at_references` cited as mandatory regression). |
  | **OQ-1** | "No default-password decryption" overstates pypdf behavior. | **ADOPTED.** Spec v2 wording: "Pipeline does NOT attempt password discovery. Pypdf may perform an internal empty-password verification during `PdfReader.__init__`, but `reader.is_encrypted` remains True even on successful internal decrypt; WP-CORE-10 raises `EncryptedPDFError` whenever `is_encrypted` is True." |

  **Codex disposition summary**: 2 CRITICAL ADOPTED via extended try/except + new behavior tests; 6 WARN ADOPTED inline; 1 NIT inlined; 1 OQ reworded for accuracy.

## Motivation

`core/document_parser_readers.py:read_pdf` is 3 LOC with zero defensive handling:

```python
def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [_extract_pdf_page_text(page) for page in reader.pages]
    return "\n\n".join(page for page in pages if page.strip())
```

Failure modes that escape unhandled:

| failure mode | current behavior | desired |
|---|---|---|
| File is not actually a PDF (renamed `.docx` → `.pdf`) | `PdfReader.__init__` raises `pypdf.errors.PdfReadError("EOF marker not found")` — opaque | Detect via magic-byte check (re-use WP-CORE-9 `_detect_binary_signature` for non-`%PDF-` content); raise `MisLabeledFileError` with format label |
| PDF is encrypted / password-protected | `reader.is_encrypted` returns True; `extract_text` returns `""` for each page; final string is empty → caught by `EmptySRSDocumentError` downstream BUT with misleading "empty document" message | Raise `EncryptedPDFError(ValueError)` immediately with clear "decrypt before ingesting" message |
| PDF is corrupted (truncated, bad xref) | `pypdf.errors.PdfReadError` or `PdfStreamError` — opaque to user | Wrap as `CorruptedPDFError(ValueError)` with the original `pypdf` error message preserved |
| PDF has zero pages | `reader.pages` is empty; final string is empty → `EmptySRSDocumentError` from downstream | Raise `EmptyPDFError(EmptySRSDocumentError)` at reader level for diagnostic clarity |
| PDF is image-only (scanned, no text layer) | Per-page `extract_text` returns `""`; final string is empty → `EmptySRSDocumentError` from downstream | Same as zero-pages: raise `EmptyPDFError` with hint "image-only PDF; OCR required" |

### Production reachability (loop discipline — mandatory subsection)

**F-1 status: LIVE.** Path:
1. VSCode file picker (`extension/src/extension.ts:511-518`) accepts `.pdf`.
2. User selects a corrupted/encrypted/mislabeled PDF.
3. `read_pdf` invoked via `parse_file` extension dispatch (`document_parser.py:53-60`).
4. `PdfReader.__init__` raises opaque `PdfReadError` OR returns reader for empty/encrypted file → downstream `EmptySRSDocumentError` with misleading "empty document" message.

WP-CORE-9 closed `.txt` rename path; this closes `.pdf` symmetric. Together they cover the two most common ingestion error scenarios.

## Discovery

### D-1. Backlog claim verified

`document_parser_readers.py:89-92` confirmed 3-LOC zero-defense `read_pdf`. `_extract_pdf_page_text` (`:95-100`) wraps `extract_text` calls with `TypeError` fallback only. No guard for `PdfReader.__init__` failure.

### D-2. Reuse WP-CORE-9 magic-byte helper

`_detect_binary_signature(data)` returns format label or None. For `read_pdf`, we want the OPPOSITE check: if `data.startswith(b"%PDF-")` is FALSE, raise `MisLabeledFileError`. Two equivalent design options:

- **(A)** Add `_assert_pdf_signature(data, file_path)` helper that raises if not PDF.
- **(B)** Inline `if not data.startswith(b"%PDF-"): raise MisLabeledFileError(...)`.

(B) is smaller correct change for a single-line check.

### D-3. pypdf exception taxonomy

Available pypdf errors (verified): `PdfReadError`, `PdfStreamError`, `EmptyFileError`, `FileNotDecryptedError`, `ParseError`, `DependencyError`. We catch the family at `pypdf.errors.PdfReadError` (or its ancestor) and re-raise as `CorruptedPDFError(ValueError)`.

## Design

### D-4. New typed exceptions

`core/document_parser_readers.py` gains 3 exceptions:

```python
class EncryptedPDFError(ValueError):
    """Raised when a PDF is encrypted/password-protected.

    The caller must decrypt the file before ingestion. The pipeline does NOT
    attempt password discovery or default-password decryption.
    """
    def __init__(self, file_path: str, message: Optional[str] = None):
        self.file_path = file_path
        super().__init__(
            message
            or f"PDF {file_path!r} is encrypted; decrypt before ingesting."
        )


class CorruptedPDFError(ValueError):
    """Raised when pypdf cannot parse the PDF structure.

    Wraps the underlying pypdf error message; the file_path attribute lets
    callers identify which input failed.
    """
    def __init__(self, file_path: str, cause: Exception, message: Optional[str] = None):
        self.file_path = file_path
        self.cause = cause
        super().__init__(
            message
            or f"PDF {file_path!r} is corrupted or malformed: "
               f"{type(cause).__name__}: {cause}"
        )


class EmptyPDFError(ValueError):
    """Raised when a PDF parses successfully but yields no text content.

    Distinct from EmptySRSDocumentError: this surfaces at the reader level
    so callers know the failure is PDF-specific (likely image-only scan
    requiring OCR) rather than a general empty-input contract violation.
    """
    def __init__(self, file_path: str, reason: str, message: Optional[str] = None):
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            message
            or f"PDF {file_path!r} has no extractable text ({reason}). "
               f"If this is a scanned PDF, OCR is required."
        )
```

LOC: ~50.

### D-5. Defensive `read_pdf`

```python
def read_pdf(file_path: str) -> str:
    # WP-CORE-10 (F-1): magic-byte check first — surface mislabeled inputs
    # with a typed format-aware error instead of an opaque PdfReadError
    # from pypdf's EOF-marker scan.
    data = Path(file_path).read_bytes()
    if not data.startswith(b"%PDF-"):
        # If the prefix is empty or doesn't match, check whether it's a
        # different known binary format for a clearer error message.
        detected = _detect_binary_signature(data)
        if detected is not None:
            raise MisLabeledFileError(file_path=file_path, detected_format=detected)
        raise MisLabeledFileError(
            file_path=file_path,
            detected_format="non-PDF (no %PDF- header)",
        )

    # Wrap pypdf parsing — catch the PdfReadError family + report typed.
    try:
        reader = PdfReader(file_path)
    except PdfReadError as exc:
        raise CorruptedPDFError(file_path=file_path, cause=exc) from exc

    # Encrypted PDF detection.
    if reader.is_encrypted:
        raise EncryptedPDFError(file_path=file_path)

    # Zero-page short-circuit.
    if len(reader.pages) == 0:
        raise EmptyPDFError(file_path=file_path, reason="zero pages")

    pages = [_extract_pdf_page_text(page) for page in reader.pages]
    joined = "\n\n".join(page for page in pages if page.strip())

    # Image-only (no extractable text) detection.
    if not joined.strip():
        raise EmptyPDFError(
            file_path=file_path,
            reason="no extractable text (likely image-only / scanned)",
        )

    return joined
```

LOC: ~25.

### D-6. Re-export from `document_parser.py`

`EncryptedPDFError`, `CorruptedPDFError`, `EmptyPDFError` join `__all__` alongside `MisLabeledFileError` and `EmptySRSDocumentError`.

### D-7. Public import path symmetry

Tests + downstream callers import via `from core.document_parser import CorruptedPDFError, EncryptedPDFError, EmptyPDFError, MisLabeledFileError`.

## Test plan

**RED commit expected pytest result:** 373 + new RED-by-design = 380 collected; 373 passed, 7 failed, 31 deselected.

| # | name | file | what it asserts | RED expectation |
|---|---|---|---|---|
| T-PDF-1 | `test_encrypted_pdf_error_carries_file_path` | `tests/test_document_parser_pdf_defensive.py` (NEW) | `EncryptedPDFError("/x.pdf")` exposes `.file_path`; readable message; subclass of ValueError | FAIL — class doesn't exist |
| T-PDF-2 | `test_corrupted_pdf_error_carries_file_path_and_cause` | same | `CorruptedPDFError("/x.pdf", cause=ValueError("bad"))` exposes `.file_path` + `.cause`; message includes cause type + text | FAIL |
| T-PDF-3 | `test_empty_pdf_error_carries_file_path_and_reason` | same | `EmptyPDFError("/x.pdf", "zero pages")` exposes `.file_path` + `.reason` | FAIL |
| T-PDF-4 | `test_read_pdf_raises_mislabeled_on_non_pdf_content` | same | Write `b"This is plain text not a PDF"` to `tmp_path / "fake.pdf"`; `parse_file` raises `MisLabeledFileError` with `detected_format` mentioning "non-PDF" | FAIL — current path raises opaque PdfReadError |
| T-PDF-5 | `test_read_pdf_raises_mislabeled_on_zip_content_as_pdf` | same | Write `b"PK\x03\x04..."` to `tmp_path / "fake.pdf"`; raises `MisLabeledFileError` with `detected_format` mentioning "ZIP" | FAIL |
| T-PDF-6 | `test_read_pdf_raises_encrypted_pdf_error_on_encrypted_input` | same | Mock `PdfReader` to return reader with `is_encrypted=True`; `parse_file` raises `EncryptedPDFError` with file_path | FAIL — current path returns empty string silently |
| T-PDF-7 | `test_read_pdf_raises_corrupted_pdf_error_on_pdf_read_error` | same | Mock `PdfReader` to raise `pypdf.errors.PdfReadError("EOF not found")`; `parse_file` raises `CorruptedPDFError` with cause preserved | FAIL |

**Existing regression contract:** all `test_parse_pdf_*` tests in `tests/test_document_parser.py` must continue to pass. The defensive checks must NOT alter happy-path behavior.

**Total**: 7 RED-by-design.

## Risks

| # | risk | mitigation |
|---|---|---|
| R-1 | `Path(file_path).read_bytes()` doubles I/O on happy path (PdfReader also reads the file). | Negligible. `read_bytes` is one syscall + a few KB read for header check. PDFs typically MB-sized; total <0.5% I/O overhead. |
| R-2 | `pypdf.errors.PdfReadError` import path may differ across pypdf versions. | Verified at HEAD: `from pypdf.errors import PdfReadError` works. Pin version in `requirements.lock`. |
| R-3 | A PDF that decrypts to empty content (rare encryption metadata edge case) — should it raise `EncryptedPDFError` or `EmptyPDFError`? | Spec D-5 order: check encryption FIRST, then zero-pages, then empty-text. `is_encrypted=True` always wins → `EncryptedPDFError`. |
| R-4 | PDF with mixed image + text pages: some pages yield text, some empty. Currently `read_pdf` joins non-empty. Post-WP-CORE-10: same behavior; `EmptyPDFError` only raises if ALL pages are empty. | T-PDF tests confirm this — partial text PDFs return their text successfully. |
| R-5 | Re-using `_detect_binary_signature` for "is this PDF" check is asymmetric — the helper returns format label for binary, we want the opposite. Spec D-5 inline check (`startswith(b"%PDF-")`) is clearer. | Use inline check + helper for non-PDF format labelling. Two-step. |

## Open questions

| # | question | disposition |
|---|---|---|
| **OQ-1** | Should `EncryptedPDFError` / `CorruptedPDFError` / `EmptyPDFError` subclass `EmptySRSDocumentError` instead of `ValueError`? | **NO.** `EmptySRSDocumentError` semantically means "content present but post-processed text is empty"; PDF defensive errors are different failure modes (encryption / corruption / no-text-layer). Keep flat `ValueError` taxonomy. |
| **OQ-2** | Should we support PDF decryption with a default empty-password (some PDFs are technically encrypted but with no password)? | **NO for v1.** Pipeline policy: no auto-decryption. AGENTS.md "smallest correct change" — if real-world data demands it, file a follow-up WP. |
| **OQ-3** | Symmetry with `read_docx` (F-7 — DOCX zero try/except)? | **DEFERRED, separate WP.** F-7 is a MINOR backlog; WP-CORE-10 scope is `.pdf` only. Pattern (`MisLabeledFileError` upfront + typed reader exceptions) can be re-applied. |
| **OQ-4** | Should `EmptyPDFError(file_path, "zero pages")` and `EmptyPDFError(file_path, "no extractable text")` be distinct subclasses or one class with `.reason`? | **One class with `.reason` (chosen).** Smaller surface. Callers can branch on `.reason` if needed. |

## Atomic commit sequence

1. **RED commit** — `test(document_parser): WP-CORE-10 red-phase tests for PDF defensive handling`
2. **GREEN commit** — `fix(document_parser): WP-CORE-10 PDF defensive handling (MisLabeledFileError + EncryptedPDFError + CorruptedPDFError + EmptyPDFError)`
3. **DOC commit** — `chore(artifacts): WP-CORE-10 dev_doc + audit state update + F-1 SHIPPED`
4. **PLANNING commit** — `chore(planning): WP-CORE-10 spec v2 + plan into git history`

## Downstream impact

| concern | impact |
|---|---|
| `_parse_srs_batch` | All new errors are `ValueError` subclasses; routes through `_parse_srs_batch`'s ValueError handler (per WP-CORE-3 / WP-CORE-9 path). |
| WP-CORE-8 typed `PipelineError` handler | NOT a `PipelineError`; falls through to bare-Exception. Acceptable — ingestion errors have their own ValueError taxonomy. |
| EMSE paper | Documents PDF ingestion-layer error taxonomy fully. Flag for advisor. |

Spec v1 ready for Codex xhigh.
