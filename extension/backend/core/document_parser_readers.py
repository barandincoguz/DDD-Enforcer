import re
from pathlib import Path
from typing import Iterator, Optional, Union

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.opc.exceptions import OpcError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

LIST_ITEM_PATTERN = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)")


# =============================================================================
# WP-CORE-9 — Mislabeled-file detection (F-2)
# =============================================================================


class MisLabeledFileError(ValueError):
    """Raised when a file extension does not match its magic-byte signature.

    Example: a `.docx` (ZIP archive) saved with `.txt` extension; a `.pdf`
    saved as `.txt`; etc. The file's first bytes are checked against known
    binary signatures BEFORE encoding-decode attempts; on match the file is
    rejected with a clear message naming the detected real format.

    Distinct from `EmptySRSDocumentError`: this is a content-format mismatch,
    not a content-emptiness issue. Both inherit from `ValueError`.

    Dual benefit:
      (1) Catches the rare silent-accept case where cp1254 decodes a no-NUL
          mostly-printable binary file and `_looks_like_text` passes.
      (2) Improves diagnostics for the common case: instead of
          `UnicodeDecodeError("Unable to decode text file")`, the caller
          receives the actual detected format label.
    """

    def __init__(
        self,
        file_path: str,
        detected_format: str,
        message: Optional[str] = None,
    ):
        self.file_path = file_path
        self.detected_format = detected_format
        super().__init__(
            message
            or (
                f"File {file_path!r} appears to be a {detected_format} file, "
                f"not text. Rename to the correct extension or convert to text."
            )
        )


# Magic-byte signatures for common binary formats commonly renamed to .txt.
# Non-overlapping by `startswith`; order has no behavioral effect today.
# If overlapping prefixes are added later, longer-specific signatures must
# come first.
_BINARY_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "ZIP archive (likely .docx/.xlsx/.zip)"),
    (b"PK\x05\x06", "ZIP archive (empty)"),
    (b"PK\x07\x08", "ZIP archive (split/spanned)"),
    (b"%PDF-",      "PDF"),
    (b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1", "Microsoft compound document (likely legacy .doc/.xls)"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xFF\xD8\xFF", "JPEG image"),
    (b"GIF87a",     "GIF image"),
    (b"GIF89a",     "GIF image"),
    (b"\x1f\x8b\x08", "gzip archive"),
)


def _detect_binary_signature(data: bytes) -> Optional[str]:
    """Return a human-readable format label if `data` starts with a known
    binary magic-byte signature; otherwise None.

    Used BEFORE the encoding-decode loop in `read_txt` so a renamed
    binary file is rejected with a typed `MisLabeledFileError` instead
    of being silently decoded via single-byte fallback encodings.

    Also used by WP-CORE-10 `read_pdf` for symmetric magic-byte detection
    at the PDF-reader boundary (rejects non-PDF byte-0 content with format
    label).
    """
    for prefix, label in _BINARY_MAGIC_SIGNATURES:
        if data.startswith(prefix):
            return label
    return None


# WP-CORE-10 W-3 (Codex): max bytes needed for magic-byte detection. Used
# by `read_pdf` to read only the file header (not the full file).
_MAGIC_HEADER_BYTES = max(len(prefix) for prefix, _ in _BINARY_MAGIC_SIGNATURES)


# =============================================================================
# WP-CORE-10 — PDF defensive handling (F-1)
# =============================================================================


class EncryptedPDFError(ValueError):
    """Raised when a PDF is encrypted/password-protected.

    The pipeline does NOT attempt password discovery. Pypdf may perform an
    internal empty-password verification during `PdfReader.__init__`, but
    `reader.is_encrypted` remains True even on successful internal decrypt;
    WP-CORE-10 raises EncryptedPDFError whenever `is_encrypted` is True
    (Codex OQ-1 disposition).
    """

    def __init__(self, file_path: str, message: Optional[str] = None):
        self.file_path = file_path
        super().__init__(
            message
            or f"PDF {file_path!r} is encrypted; decrypt before ingesting."
        )


class CorruptedPDFError(ValueError):
    """Raised when pypdf cannot parse the PDF structure (constructor or
    lazy page/extract failures).

    Codex C-1 + W-1: covers both `PdfReader.__init__` raises AND
    `len(reader.pages)` / `page.extract_text` raises during downstream
    iteration. `.cause` (custom payload) AND `__cause__` (Python exception
    chain via `raise ... from`) both preserve the original pypdf error.
    """

    def __init__(
        self,
        file_path: str,
        cause: Exception,
        message: Optional[str] = None,
    ):
        self.file_path = file_path
        self.cause = cause
        super().__init__(
            message
            or (
                f"PDF {file_path!r} is corrupted or malformed: "
                f"{type(cause).__name__}: {cause}"
            )
        )


class CorruptedDOCXError(ValueError):
    """Raised when python-docx cannot open the file (e.g., truncated archive,
    missing OPC parts, mislabeled non-OOXML ZIP). Wraps the underlying
    `docx.opc.exceptions.OpcError`.

    Pattern symmetric to `CorruptedPDFError`: `.cause` (custom payload)
    AND `__cause__` (Python exception chain via `raise ... from`) both
    preserve the original docx error.
    """

    def __init__(
        self,
        file_path: str,
        cause: Exception,
        message: Optional[str] = None,
    ):
        self.file_path = file_path
        self.cause = cause
        super().__init__(
            message
            or (
                f"DOCX {file_path!r} could not be opened: "
                f"{type(cause).__name__}: {cause}"
            )
        )


class EmptyDOCXError(ValueError):
    """Raised when a DOCX parses successfully but yields no paragraphs or
    tables. Distinct from `EmptySRSDocumentError` (generic post-processed
    empty content): this surfaces at the reader level so callers know
    the failure is DOCX-specific.
    """

    def __init__(self, file_path: str, message: Optional[str] = None):
        self.file_path = file_path
        super().__init__(
            message
            or (
                f"DOCX {file_path!r} parsed but contains no extractable "
                f"paragraphs or tables."
            )
        )


class EmptyPDFError(ValueError):
    """Raised when a PDF parses successfully but yields no text content.

    Distinct from `EmptySRSDocumentError`: surfaces at the reader level
    so callers know the failure is PDF-specific (likely image-only scan
    requiring OCR, or zero-page PDF) rather than a general empty-input
    contract violation.

    Codex W-6 disposition: flat `ValueError` taxonomy (not
    `EmptySRSDocumentError` subclass) — callers needing PDF-specific
    handling catch `EmptyPDFError` directly; generic catch handlers
    still match via `ValueError`.
    """

    def __init__(
        self,
        file_path: str,
        reason: str,
        message: Optional[str] = None,
    ):
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            message
            or (
                f"PDF {file_path!r} has no extractable text ({reason}). "
                f"If this is a scanned PDF, OCR is required."
            )
        )


def read_pdf(file_path: str) -> str:
    # WP-CORE-10 (F-1) W-3 + W-4: header-only read for magic-byte check
    # (avoids reading multi-GB PDFs twice); single-pass classification via
    # _detect_binary_signature. WP-CORE-9 helper labels the format; if it's
    # not "PDF", raise MisLabeledFileError with the actual format.
    with open(file_path, "rb") as f:
        header = f.read(_MAGIC_HEADER_BYTES)
    detected = _detect_binary_signature(header)
    if detected != "PDF":
        raise MisLabeledFileError(
            file_path=file_path,
            detected_format=detected or "non-PDF (no %PDF- header at byte 0)",
        )

    # WP-CORE-10 C-1: wrap constructor AND lazy page/extract failures in
    # one PdfReadError block. PdfStreamError is a PdfReadError subclass
    # (T-PDF-INHERIT) so a single except covers both.
    try:
        reader = PdfReader(file_path)

        if reader.is_encrypted:
            raise EncryptedPDFError(file_path=file_path)

        page_count = len(reader.pages)
        if page_count == 0:
            raise EmptyPDFError(file_path=file_path, reason="zero pages")

        pages = [_extract_pdf_page_text(page) for page in reader.pages]
    except PdfReadError as exc:
        raise CorruptedPDFError(file_path=file_path, cause=exc) from exc

    joined = "\n\n".join(page for page in pages if page.strip())

    if not joined.strip():
        raise EmptyPDFError(
            file_path=file_path,
            reason="no extractable text (likely image-only / scanned PDF)",
        )

    return joined


def _extract_pdf_page_text(page) -> str:
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        text = page.extract_text() or ""
    return text if text.strip() else (page.extract_text() or "")


def read_docx(file_path: str) -> str:
    # WP-CORE-11 (F-7): DOCX is a ZIP archive (PK\x03\x04). Magic-byte check
    # at byte 0 rejects mislabeled non-ZIP content upfront with
    # MisLabeledFileError instead of opaque PackageNotFoundError.
    with open(file_path, "rb") as f:
        header = f.read(_MAGIC_HEADER_BYTES)
    detected = _detect_binary_signature(header)
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


def _iter_docx_blocks(document: DocxDocument) -> Iterator[Union[Paragraph, Table]]:
    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _extract_docx_paragraph(paragraph: Paragraph) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""
    if _is_list_paragraph(paragraph) and not LIST_ITEM_PATTERN.match(text):
        return f"- {text}"
    return text


def _is_list_paragraph(paragraph: Paragraph) -> bool:
    style_name = getattr(getattr(paragraph, "style", None), "name", "") or ""
    properties = paragraph._p.pPr
    has_numbering = bool(properties is not None and properties.numPr is not None)
    return has_numbering or style_name.lower().startswith("list")


def _extract_docx_table(table: Table) -> str:
    rows = []

    for row in table.rows:
        cells = []
        seen_cells = set()

        for cell in row.cells:
            cell_id = id(cell._tc)
            if cell_id in seen_cells:
                continue
            seen_cells.add(cell_id)
            cells.append(re.sub(r"\s*\n\s*", " / ", cell.text.strip()))

        if any(cells):
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def read_txt(file_path: str) -> str:
    data = Path(file_path).read_bytes()

    # WP-CORE-9 (F-2): detect mislabeled binary files BEFORE the encoding-
    # decode loop. Single-byte fallback encodings (cp1254/cp1252) decode any
    # byte sequence without raising; the printable-ratio heuristic at
    # _looks_like_text can silently accept gibberish from a renamed .docx/.pdf
    # in the rare case of no-NUL printable content. Detecting common
    # magic-byte signatures upfront surfaces the actual file format with a
    # typed MisLabeledFileError instead of opaque UnicodeDecodeError.
    detected = _detect_binary_signature(data)
    if detected is not None:
        raise MisLabeledFileError(file_path=file_path, detected_format=detected)

    for encoding in _candidate_text_encodings(data):
        try:
            decoded = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_like_text(decoded):
            return decoded

    raise UnicodeDecodeError(
        "document_parser",
        data,
        0,
        max(len(data) - 1, 0),
        f"Unable to decode text file: {file_path}",
    )


def _candidate_text_encodings(data: bytes) -> list[str]:
    if data.startswith(b"\xef\xbb\xbf"):
        return ["utf-8-sig", "utf-8", "cp1254", "cp1252"]
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return ["utf-16", "utf-16-le", "utf-16-be"]
    return ["utf-8", "utf-8-sig", "utf-16", "cp1254", "cp1252"]


def _looks_like_text(text: str) -> bool:
    if not text:
        return True
    if "\x00" in text:
        return False
    meaningful = sum(1 for char in text if char.isprintable() or char.isspace())
    return meaningful / len(text) >= 0.95
