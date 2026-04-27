from pathlib import Path

import pytest
from docx import Document

from core.document_parser import SRSDocumentParser


def _write_simple_pdf(path: Path, pages: list[list[str]]) -> None:
    objects = []
    page_object_ids = []
    content_object_ids = []

    font_object_id = 3
    next_object_id = 4

    for _ in pages:
        page_object_ids.append(next_object_id)
        content_object_ids.append(next_object_id + 1)
        next_object_id += 2

    kids = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_index, lines in enumerate(pages):
        page_id = page_object_ids[page_index]
        content_id = content_object_ids[page_index]

        stream_lines = ["BT", "/F1 12 Tf", "72 720 Td"]
        for line_index, line in enumerate(lines):
            escaped = (
                line.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            if line_index > 0:
                stream_lines.append("0 -18 Td")
            stream_lines.append(f"({escaped}) Tj")
        stream_lines.append("ET")

        stream = "\n".join(stream_lines).encode("latin-1")
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for object_id, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n".encode("latin-1")
    )
    pdf.extend(f"startxref\n{xref_offset}\n%%EOF".encode("latin-1"))
    path.write_bytes(bytes(pdf))


def test_parse_pdf_merges_wrapped_lines_and_stops_at_references(tmp_path):
    pdf_file = tmp_path / "requirements.pdf"
    _write_simple_pdf(
        pdf_file,
        [[
            "1. Requirements",
            "The system shall track",
            "orders for each customer.",
            "References",
            "This appendix should not be indexed.",
        ]],
    )

    content = SRSDocumentParser().parse_file(str(pdf_file))

    assert "The system shall track orders for each customer." in content
    assert "This appendix should not be indexed." not in content


def test_parse_docx_preserves_lists_and_tables(tmp_path):
    docx_file = tmp_path / "requirements.docx"
    document = Document()
    document.add_heading("Order Management", level=1)
    document.add_paragraph("The system shall store orders.")
    document.add_paragraph("Customer can cancel an order.", style="List Bullet")

    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "Field"
    table.rows[0].cells[1].text = "Rule"
    table.rows[1].cells[0].text = "Status"
    table.rows[1].cells[1].text = "Required"

    document.save(docx_file)

    content = SRSDocumentParser().parse_file(str(docx_file))

    assert "Order Management" in content
    assert "- Customer can cancel an order." in content
    assert "Field | Rule" in content
    assert "Status | Required" in content


def test_parse_txt_supports_utf16_input(tmp_path):
    txt_file = tmp_path / "requirements.txt"
    txt_file.write_bytes("Odeme kaydi zorunludur.".encode("utf-16"))

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "Odeme kaydi zorunludur." in content


def test_parse_txt_does_not_truncate_regular_requirement_lines(tmp_path):
    txt_file = tmp_path / "references-prefix.txt"
    txt_file.write_text(
        "References to external systems shall be preserved.\n"
        "This rule remains important.",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "References to external systems shall be preserved." in content
    assert "This rule remains important." in content


def test_parse_nonexistent_file_raises_file_not_found():
    parser = SRSDocumentParser()

    with pytest.raises(FileNotFoundError):
        parser.parse_file("/nonexistent/file.txt")
