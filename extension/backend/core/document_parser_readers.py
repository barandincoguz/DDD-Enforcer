import re
from pathlib import Path
from typing import Iterator, Union

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

LIST_ITEM_PATTERN = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)")


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
