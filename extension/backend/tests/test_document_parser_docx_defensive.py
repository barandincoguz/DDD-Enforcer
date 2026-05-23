"""WP-CORE-11 — read_docx defensive handling (F-7).

T-DOCX-1: exception class shapes.
T-DOCX-2: MisLabeledFileError on non-ZIP byte-0 .docx.
T-DOCX-3: CorruptedDOCXError on docx.Document raising (with __cause__ chain).
T-DOCX-4: EmptyDOCXError when no extracted blocks.
T-DOCX-5: happy regression — legit DOCX still parses.

Run: pytest tests/test_document_parser_docx_defensive.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _write_minimal_docx(path: Path, body_text: str = "Hello DOCX") -> None:
    """Write a minimal valid DOCX via python-docx for happy-path tests."""
    from docx import Document
    doc = Document()
    doc.add_paragraph(body_text)
    doc.save(str(path))


# =============================================================================
# T-DOCX-1 — Exception class shapes
# =============================================================================


def test_corrupted_docx_error_carries_file_path_and_cause():
    """T-DOCX-1a: CorruptedDOCXError shape."""
    from core.document_parser import CorruptedDOCXError

    inner = ValueError("bad DOCX")
    exc = CorruptedDOCXError(file_path="/x.docx", cause=inner)
    assert isinstance(exc, ValueError)
    assert exc.file_path == "/x.docx"
    assert exc.cause is inner


def test_empty_docx_error_carries_file_path():
    """T-DOCX-1b: EmptyDOCXError shape."""
    from core.document_parser import EmptyDOCXError

    exc = EmptyDOCXError(file_path="/x.docx")
    assert isinstance(exc, ValueError)
    assert exc.file_path == "/x.docx"
    assert "/x.docx" in str(exc)


# =============================================================================
# T-DOCX-2 — MisLabeledFileError on non-ZIP byte-0
# =============================================================================


def test_read_docx_raises_mislabeled_on_non_zip_content(tmp_path):
    """T-DOCX-2: a .docx containing plain text raises MisLabeledFileError."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    fake = tmp_path / "fake.docx"
    fake.write_bytes(b"This is plain text, not a DOCX " * 10)

    with pytest.raises(MisLabeledFileError):
        SRSDocumentParser().parse_file(str(fake))


# =============================================================================
# T-DOCX-3 — CorruptedDOCXError on docx.Document raising
# =============================================================================


def test_read_docx_raises_corrupted_docx_error_with_cause_chain(tmp_path):
    """T-DOCX-3: docx.Document raises OpcError → CorruptedDOCXError with
    .cause AND __cause__ preserved."""
    from docx.opc.exceptions import PackageNotFoundError
    from core.document_parser import CorruptedDOCXError
    from core.document_parser_readers import read_docx

    fake = tmp_path / "corrupt.docx"
    # Real DOCX magic byte (ZIP) so magic check passes, then docx.Document fails
    _write_minimal_docx(fake)  # Then override via patch

    original = PackageNotFoundError("Package not found at /corrupt.docx")

    with patch("core.document_parser_readers.docx.Document", side_effect=original):
        with pytest.raises(CorruptedDOCXError) as exc_info:
            read_docx(str(fake))

    assert exc_info.value.cause is original
    assert exc_info.value.__cause__ is original


# =============================================================================
# T-DOCX-4 — EmptyDOCXError when no blocks extracted
# =============================================================================


def test_read_docx_raises_empty_docx_error_when_no_blocks(tmp_path):
    """T-DOCX-4: docx.Document parses successfully but yields no paragraphs
    or tables → EmptyDOCXError."""
    from core.document_parser import EmptyDOCXError
    from core.document_parser_readers import read_docx

    fake = tmp_path / "empty.docx"
    _write_minimal_docx(fake)

    # Mock docx.Document to return a document with empty body element.
    mock_document = MagicMock()
    mock_body = MagicMock()
    mock_body.iterchildren = MagicMock(return_value=iter([]))
    mock_document.element.body = mock_body

    with patch("core.document_parser_readers.docx.Document", return_value=mock_document):
        with pytest.raises(EmptyDOCXError):
            read_docx(str(fake))


# =============================================================================
# T-DOCX-5 — Happy regression
# =============================================================================


def test_read_docx_happy_path_legit_docx_parses(tmp_path):
    """T-DOCX-5: a valid DOCX still parses post-WP-CORE-11."""
    from core.document_parser_readers import read_docx

    docx_file = tmp_path / "good.docx"
    _write_minimal_docx(docx_file, body_text="Hello WP-CORE-11")

    text = read_docx(str(docx_file))
    assert "Hello WP-CORE-11" in text
