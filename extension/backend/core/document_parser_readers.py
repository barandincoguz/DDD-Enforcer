import re
from pathlib import Path
from typing import Iterator, Optional, Union

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

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
    """
    for prefix, label in _BINARY_MAGIC_SIGNATURES:
        if data.startswith(prefix):
            return label
    return None


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = [_extract_pdf_page_text(page) for page in reader.pages]
    return "\n\n".join(page for page in pages if page.strip())


def _extract_pdf_page_text(page) -> str:
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        text = page.extract_text() or ""
    return text if text.strip() else (page.extract_text() or "")


def read_docx(file_path: str) -> str:
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

    return "\n\n".join(blocks)


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
