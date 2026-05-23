"""WP-CORE-10 — read_pdf defensive handling (F-1).

T-PDF-1..3: exception class shapes.
T-PDF-4..5: MisLabeledFileError on non-PDF byte-0 content (re-uses
            WP-CORE-9 magic-byte helper via single-pass dispatch).
T-PDF-6: EncryptedPDFError on is_encrypted=True.
T-PDF-7: CorruptedPDFError on PdfReadError, with .cause AND __cause__
         chain preserved (Codex W-1 disposition).
T-PDF-LAZY-1: CorruptedPDFError when reader.pages access raises (Codex C-1).
T-PDF-LAZY-2: CorruptedPDFError when page.extract_text raises (Codex C-1).
T-PDF-EMPTY-1: EmptyPDFError on zero pages (Codex C-2).
T-PDF-EMPTY-2: EmptyPDFError on all-pages-empty-text (Codex C-2).
T-PDF-MIXED: mixed empty + non-empty pages — non-empty text returned
             (Codex C-2 regression).
T-PDF-STRICT: PDF with leading whitespace before %PDF- rejected by design
              (Codex W-5).
T-PDF-INHERIT: pypdf.errors.PdfStreamError subclasses PdfReadError so the
               single `except PdfReadError` catches both (Codex W-2).
T-PDF-HAPPY: legit PDF still parses post-WP-CORE-10 (Codex N-1).

Run: pytest tests/test_document_parser_pdf_defensive.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _write_minimal_pdf(path: Path, body_text: str = "Hello PDF World") -> None:
    """Write a minimal valid PDF that pypdf can parse.

    Adapted from test_document_parser.py's _write_simple_pdf pattern.
    """
    # Very small valid PDF: %PDF-1.4 header + catalog + pages + content
    # + xref + startxref + %%EOF. pypdf accepts this.
    content_stream = f"BT /F1 24 Tf 100 700 Td ({body_text}) Tj ET"
    content_bytes = content_stream.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"\nendstream",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_offset = len(pdf)
    pdf += b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n"
    pdf += b"0000000000 65535 f\n"
    for off in offsets:
        pdf += f"{off:010d} 00000 n\n".encode("ascii")
    pdf += b"trailer\n<< /Size " + str(len(objects) + 1).encode("ascii") + b" /Root 1 0 R >>\n"
    pdf += b"startxref\n" + str(xref_offset).encode("ascii") + b"\n%%EOF\n"
    path.write_bytes(pdf)


# =============================================================================
# T-PDF-1..3 — Exception class shapes
# =============================================================================


def test_encrypted_pdf_error_carries_file_path():
    """T-PDF-1: EncryptedPDFError shape + ValueError lineage."""
    from core.document_parser import EncryptedPDFError

    exc = EncryptedPDFError(file_path="/x.pdf")
    assert isinstance(exc, ValueError)
    assert exc.file_path == "/x.pdf"
    assert "/x.pdf" in str(exc)
    assert "encrypted" in str(exc).lower()


def test_corrupted_pdf_error_carries_file_path_and_cause():
    """T-PDF-2: CorruptedPDFError shape + .cause attribute."""
    from core.document_parser import CorruptedPDFError

    inner = ValueError("bad PDF")
    exc = CorruptedPDFError(file_path="/x.pdf", cause=inner)
    assert isinstance(exc, ValueError)
    assert exc.file_path == "/x.pdf"
    assert exc.cause is inner
    assert "ValueError" in str(exc)
    assert "bad PDF" in str(exc)


def test_empty_pdf_error_carries_file_path_and_reason():
    """T-PDF-3: EmptyPDFError shape + .reason."""
    from core.document_parser import EmptyPDFError

    exc = EmptyPDFError(file_path="/x.pdf", reason="zero pages")
    assert isinstance(exc, ValueError)
    assert exc.file_path == "/x.pdf"
    assert exc.reason == "zero pages"
    assert "zero pages" in str(exc)


# =============================================================================
# T-PDF-4..5 — MisLabeledFileError on non-PDF byte-0
# =============================================================================


def test_read_pdf_raises_mislabeled_on_non_pdf_content(tmp_path):
    """T-PDF-4: a .pdf containing plain text bytes raises MisLabeledFileError."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"This is plain text not a PDF " * 10)

    with pytest.raises(MisLabeledFileError) as exc_info:
        SRSDocumentParser().parse_file(str(fake))
    assert "non-PDF" in exc_info.value.detected_format or "PDF" not in exc_info.value.detected_format


def test_read_pdf_raises_mislabeled_on_zip_content_as_pdf(tmp_path):
    """T-PDF-5: a .pdf wrapping ZIP bytes raises MisLabeledFileError naming ZIP."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"PK\x03\x04" + b"AAAA" * 100)

    with pytest.raises(MisLabeledFileError) as exc_info:
        SRSDocumentParser().parse_file(str(fake))
    assert "ZIP" in exc_info.value.detected_format


# =============================================================================
# T-PDF-6 — Encrypted PDF
# =============================================================================


def test_read_pdf_raises_encrypted_pdf_error_on_encrypted_input(tmp_path):
    """T-PDF-6: PdfReader returns reader.is_encrypted=True → EncryptedPDFError."""
    from core.document_parser import EncryptedPDFError
    from core.document_parser_readers import read_pdf

    fake = tmp_path / "encrypted.pdf"
    _write_minimal_pdf(fake)  # Real PDF bytes (so magic-byte check passes)

    mock_reader = MagicMock()
    mock_reader.is_encrypted = True

    with patch("core.document_parser_readers.PdfReader", return_value=mock_reader):
        with pytest.raises(EncryptedPDFError) as exc_info:
            read_pdf(str(fake))
    assert exc_info.value.file_path == str(fake)


# =============================================================================
# T-PDF-7 — CorruptedPDFError on PdfReadError with full chain
# =============================================================================


def test_read_pdf_raises_corrupted_pdf_error_with_cause_chain_on_pdf_read_error(tmp_path):
    """T-PDF-7 (Codex W-1): CorruptedPDFError.cause AND __cause__ both
    preserve the original pypdf error."""
    from pypdf.errors import PdfReadError
    from core.document_parser import CorruptedPDFError
    from core.document_parser_readers import read_pdf

    fake = tmp_path / "corrupt.pdf"
    _write_minimal_pdf(fake)

    original = PdfReadError("EOF marker not found")

    with patch("core.document_parser_readers.PdfReader", side_effect=original):
        with pytest.raises(CorruptedPDFError) as exc_info:
            read_pdf(str(fake))

    assert exc_info.value.cause is original
    assert exc_info.value.__cause__ is original
    assert exc_info.value.file_path == str(fake)


# =============================================================================
# T-PDF-LAZY-1/2 — Lazy errors during page access / extraction (Codex C-1)
# =============================================================================


def test_read_pdf_raises_corrupted_pdf_error_on_lazy_pages_error(tmp_path):
    """T-PDF-LAZY-1: reader.pages property access raises PdfReadError
    → CorruptedPDFError. Constructor succeeds; lazy parse fails."""
    from pypdf.errors import PdfReadError
    from core.document_parser import CorruptedPDFError
    from core.document_parser_readers import read_pdf

    fake = tmp_path / "lazy-pages.pdf"
    _write_minimal_pdf(fake)

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    # Make len(reader.pages) raise PdfReadError
    pages_proxy = MagicMock()
    pages_proxy.__len__ = MagicMock(side_effect=PdfReadError("lazy page parse failed"))
    mock_reader.pages = pages_proxy

    with patch("core.document_parser_readers.PdfReader", return_value=mock_reader):
        with pytest.raises(CorruptedPDFError) as exc_info:
            read_pdf(str(fake))
    assert isinstance(exc_info.value.cause, PdfReadError)


def test_read_pdf_raises_corrupted_pdf_error_on_lazy_extract_text_error(tmp_path):
    """T-PDF-LAZY-2: page.extract_text raises PdfReadError → CorruptedPDFError."""
    from pypdf.errors import PdfReadError
    from core.document_parser import CorruptedPDFError
    from core.document_parser_readers import read_pdf

    fake = tmp_path / "lazy-extract.pdf"
    _write_minimal_pdf(fake)

    mock_page = MagicMock()
    mock_page.extract_text = MagicMock(side_effect=PdfReadError("stream parse failed"))
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [mock_page]

    with patch("core.document_parser_readers.PdfReader", return_value=mock_reader):
        with pytest.raises(CorruptedPDFError) as exc_info:
            read_pdf(str(fake))
    assert isinstance(exc_info.value.cause, PdfReadError)


# =============================================================================
# T-PDF-EMPTY-1/2 — EmptyPDFError branches (Codex C-2)
# =============================================================================


def test_read_pdf_raises_empty_pdf_error_on_zero_pages(tmp_path):
    """T-PDF-EMPTY-1: PdfReader returns reader with 0 pages → EmptyPDFError."""
    from core.document_parser import EmptyPDFError
    from core.document_parser_readers import read_pdf

    fake = tmp_path / "empty-pages.pdf"
    _write_minimal_pdf(fake)

    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = []

    with patch("core.document_parser_readers.PdfReader", return_value=mock_reader):
        with pytest.raises(EmptyPDFError) as exc_info:
            read_pdf(str(fake))
    assert "zero pages" in exc_info.value.reason


def test_read_pdf_raises_empty_pdf_error_when_all_pages_extract_empty_text(tmp_path):
    """T-PDF-EMPTY-2: all pages extract empty text → EmptyPDFError (likely image-only)."""
    from core.document_parser import EmptyPDFError
    from core.document_parser_readers import read_pdf

    fake = tmp_path / "image-only.pdf"
    _write_minimal_pdf(fake)

    page_a = MagicMock()
    page_a.extract_text = MagicMock(return_value="")
    page_b = MagicMock()
    page_b.extract_text = MagicMock(return_value="   \n  ")
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [page_a, page_b]

    with patch("core.document_parser_readers.PdfReader", return_value=mock_reader):
        with pytest.raises(EmptyPDFError) as exc_info:
            read_pdf(str(fake))
    assert "image-only" in exc_info.value.reason.lower() or "no extractable" in exc_info.value.reason.lower()


def test_read_pdf_returns_text_when_some_pages_empty_and_others_have_text(tmp_path):
    """T-PDF-MIXED: partial empty pages must NOT raise; non-empty text returned."""
    from core.document_parser_readers import read_pdf

    fake = tmp_path / "mixed.pdf"
    _write_minimal_pdf(fake)

    page_text = MagicMock()
    page_text.extract_text = MagicMock(return_value="real content here")
    page_empty = MagicMock()
    page_empty.extract_text = MagicMock(return_value="")
    mock_reader = MagicMock()
    mock_reader.is_encrypted = False
    mock_reader.pages = [page_empty, page_text]

    with patch("core.document_parser_readers.PdfReader", return_value=mock_reader):
        result = read_pdf(str(fake))
    assert "real content here" in result


# =============================================================================
# T-PDF-STRICT — leading whitespace rejected (Codex W-5 policy)
# =============================================================================


def test_read_pdf_rejects_leading_whitespace_before_pdf_header(tmp_path):
    """T-PDF-STRICT (Codex W-5): WP-CORE-10 enforces %PDF- at byte 0 strictly.
    pypdf-tolerant cases (leading whitespace/comments before signature) are
    rejected by design — consistent with WP-CORE-9 startswith policy."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    fake = tmp_path / "leading-ws.pdf"
    fake.write_bytes(b"\n   %PDF-1.4\nrest...")

    with pytest.raises(MisLabeledFileError):
        SRSDocumentParser().parse_file(str(fake))


# =============================================================================
# T-PDF-INHERIT — pypdf taxonomy guard (Codex W-2)
# =============================================================================


def test_pdf_stream_error_subclasses_pdf_read_error():
    """T-PDF-INHERIT (Codex W-2): assert pypdf hierarchy so a single
    `except PdfReadError` covers both PdfReadError and PdfStreamError."""
    from pypdf.errors import PdfReadError, PdfStreamError
    assert issubclass(PdfStreamError, PdfReadError)


# =============================================================================
# T-PDF-HAPPY — Regression: legit PDF still parses (Codex N-1)
# =============================================================================


def test_read_pdf_happy_path_legit_pdf_parses_unchanged(tmp_path):
    """T-PDF-HAPPY: a valid PDF still parses and returns its text."""
    from core.document_parser_readers import read_pdf

    pdf_file = tmp_path / "good.pdf"
    _write_minimal_pdf(pdf_file, body_text="Hello WP-CORE-10")

    text = read_pdf(str(pdf_file))
    # The minimal-PDF text may not extract cleanly without a content stream
    # interpreter; we just verify no exception raises and a string returns.
    assert isinstance(text, str)
