# WP-CORE-11 — `read_docx` defensive handling (F-7)

**Date:** 2026-05-23
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 10)
**Status:** DRAFT v1 — pending Codex xhigh adversarial review
**Parent finding:** `.planning/pipeline_audit/findings/document_parser.md` **F-7** (MINOR)
**Loop:** baseline 388 at HEAD `97bc4c4`
**Codex review:** pending.

## Motivation

`core/document_parser_readers.py:read_docx` (14 LOC) has zero defensive handling. Mirrors WP-CORE-9 (read_txt) + WP-CORE-10 (read_pdf) pattern. Completes the ingestion-reader trilogy.

```python
def read_docx(file_path: str) -> str:
    document = docx.Document(file_path)  # raises PackageNotFoundError on non-DOCX
    blocks = []
    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            paragraph_text = _extract_docx_paragraph(block)
            if paragraph_text:
                blocks.append(paragraph_text)
            continue
        table_text = _extract_docx_table(block)
        if table_text:
            blocks.append(table_text)
    return "\n\n".join(blocks)
```

**Failure modes**:
1. File extension `.docx` but content is not a DOCX (renamed binary, plain text, etc.) → `docx.opc.exceptions.PackageNotFoundError` — opaque.
2. Corrupted DOCX (truncated ZIP, missing required parts) → `docx.opc.exceptions.OpcError` / variants — opaque.
3. Empty DOCX (no paragraphs, no tables) → returns `""` → downstream `EmptySRSDocumentError` with misleading "empty document" message.

### Production reachability

LIVE: VSCode picker accepts `.docx`; user renames or supplies corrupted file; current path raises opaque pypdf-style errors.

## Discovery

`docx.opc.exceptions` exposes `OpcError` (base) + `PackageNotFoundError` (subclass). `PackageNotFoundError` raised for both "file not found" and "file is not a valid OPC package" — the second is our F-7 trigger.

The DOCX file IS a ZIP archive (`PK\x03\x04` signature). So magic-byte detection re-uses WP-CORE-9 helper: if byte-0 is NOT `PK`, it's mislabeled. If byte-0 IS `PK\x03\x04` but `docx.Document()` raises, it's a corrupted DOCX (or non-OOXML ZIP).

## Design

### D-1 — New typed exceptions

```python
class CorruptedDOCXError(ValueError):
    def __init__(self, file_path: str, cause: Exception, message=None):
        self.file_path = file_path
        self.cause = cause
        super().__init__(
            message
            or f"DOCX {file_path!r} could not be opened: {type(cause).__name__}: {cause}"
        )


class EmptyDOCXError(ValueError):
    def __init__(self, file_path: str, message=None):
        self.file_path = file_path
        super().__init__(
            message
            or f"DOCX {file_path!r} parsed but contains no extractable paragraphs or tables."
        )
```

LOC: ~30.

### D-2 — Refactored `read_docx`

```python
def read_docx(file_path: str) -> str:
    # Magic-byte check: DOCX is a ZIP archive. Non-ZIP byte-0 → MisLabeledFileError.
    with open(file_path, "rb") as f:
        header = f.read(_MAGIC_HEADER_BYTES)
    detected = _detect_binary_signature(header)
    # ZIP-family signatures: "ZIP archive (likely .docx/.xlsx/.zip)" or empty/spanned
    if detected is None or not detected.startswith("ZIP"):
        raise MisLabeledFileError(
            file_path=file_path,
            detected_format=detected or "non-ZIP/DOCX (no PK signature at byte 0)",
        )

    try:
        document = docx.Document(file_path)
        blocks = []
        for block in _iter_docx_blocks(document):
            if isinstance(block, Paragraph):
                paragraph_text = _extract_docx_paragraph(block)
                if paragraph_text:
                    blocks.append(paragraph_text)
                continue
            table_text = _extract_docx_table(block)
            if table_text:
                blocks.append(table_text)
    except OpcError as exc:
        raise CorruptedDOCXError(file_path=file_path, cause=exc) from exc

    joined = "\n\n".join(blocks)
    if not joined.strip():
        raise EmptyDOCXError(file_path=file_path)
    return joined
```

LOC: +25.

### D-3 — Re-export from `core.document_parser`

Add to `__all__`.

## Test plan

| # | name | what | expectation |
|---|---|---|---|
| T-DOCX-1 | exception class shape (CorruptedDOCXError + EmptyDOCXError) | shapes + `.cause` | FAIL ImportError |
| T-DOCX-2 | non-ZIP byte-0 in `.docx` raises `MisLabeledFileError` | plain text payload as fake.docx | FAIL |
| T-DOCX-3 | ZIP byte-0 but invalid DOCX raises `CorruptedDOCXError` with `__cause__` chain | mock `docx.Document` raising `PackageNotFoundError` | FAIL |
| T-DOCX-4 | Empty DOCX (no blocks) raises `EmptyDOCXError` | mock `docx.Document` returning document with empty body | FAIL |
| T-DOCX-5 | Happy regression: legit DOCX parses unchanged | use existing test_document_parser.py fixture | PASS-from-start |

RED: 4 fail + 1 PASS-from-start regression = 393 collected, 389 pass, 4 fail.

## Risks

| # | risk | mitigation |
|---|---|---|
| R-1 | `OpcError` covers all DOCX exceptions? Check inheritance. | `PackageNotFoundError` is `OpcError` subclass (verified `python-docx` source). One except covers. |
| R-2 | Legitimate DOCX with truly empty body — should it raise `EmptyDOCXError` or fall back to `EmptySRSDocumentError`? | T-DOCX-4 confirms raise at reader level for diagnostic clarity. Generic ValueError still caught downstream. |

## Atomic commit sequence

1. RED `test(document_parser): WP-CORE-11 red-phase tests for DOCX defensive handling`
2. GREEN `fix(document_parser): WP-CORE-11 CorruptedDOCXError + EmptyDOCXError + MisLabeledFileError in read_docx`
3. DOC `chore(artifacts): WP-CORE-11 dev_doc + audit state update + F-7 SHIPPED`
4. PLANNING `chore(planning): WP-CORE-11 spec v2 + plan`

## Open questions

- **OQ-1**: encrypted DOCX (password-protected office docs use OLE compound + encryption-specific container)? **Out of scope**; magic-byte check catches non-ZIP OLE compound; if encrypted DOCX wraps in standard ZIP, `docx.Document()` raises corruption-flavored error. Real-world risk acceptable.

Spec v1 ready for Codex.
