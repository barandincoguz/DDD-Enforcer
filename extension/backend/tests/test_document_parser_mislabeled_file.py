"""WP-CORE-9 — MisLabeledFileError magic-byte detection in read_txt.

T-MFE-1: exception class shape (Codex v2 disposition).
T-MFE-2/3/4: realistic mislabeled binary detection (Codex W-2 — fixtures
             expanded to long no-NUL printable payloads + a NUL-bearing
             realistic-ZIP variant in T-MFE-6).
T-MFE-5: false-positive resistance for legitimate text with literal magic
         bytes mid-content (Codex C-1 reclassified as GREEN regression).
T-MFE-6: realistic ZIP header with embedded NUL bytes — diagnostic
         improvement (Codex W-6: current path raises UnicodeDecodeError;
         post-WP-CORE-9 raises MisLabeledFileError naming the format).
T-MFE-7: BOM + later-literal-magic-bytes (Codex W-3).
T-MFE-8: helper-level near-miss prefixes (Codex W-4).

Run: pytest tests/test_document_parser_mislabeled_file.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# T-MFE-1 — MisLabeledFileError shape
# =============================================================================


def test_mislabeled_file_error_carries_file_path_and_detected_format():
    """T-MFE-1: MisLabeledFileError exposes .file_path + .detected_format;
    subclass of ValueError; readable message."""
    from core.document_parser import MisLabeledFileError

    exc = MisLabeledFileError(
        file_path="/x/renamed.txt",
        detected_format="ZIP archive (likely .docx)",
    )
    assert isinstance(exc, ValueError)
    assert exc.file_path == "/x/renamed.txt"
    assert exc.detected_format == "ZIP archive (likely .docx)"
    msg = str(exc)
    assert "renamed.txt" in msg
    assert "ZIP" in msg


# =============================================================================
# T-MFE-2 — ZIP magic bytes
# =============================================================================


def test_read_txt_raises_on_zip_magic_bytes(tmp_path):
    """T-MFE-2 (Codex W-2): a long no-NUL printable payload prefixed with
    PK\\x03\\x04 — proves the rare silent-accept case (current heuristic
    accepts cp1254 decode of pure-printable content). Post-WP-CORE-9
    raises MisLabeledFileError naming "ZIP archive"."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    # 4-byte ZIP local-file-header signature, then printable bytes that
    # cp1254 would happily decode (no NUL → _looks_like_text passes today).
    fake_payload = b"PK\x03\x04" + (b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 " * 8)
    txt_file = tmp_path / "renamed.txt"
    txt_file.write_bytes(fake_payload)

    with pytest.raises(MisLabeledFileError) as exc_info:
        SRSDocumentParser().parse_file(str(txt_file))
    assert "ZIP" in exc_info.value.detected_format


# =============================================================================
# T-MFE-3 — PDF magic bytes
# =============================================================================


def test_read_txt_raises_on_pdf_magic_bytes(tmp_path):
    """T-MFE-3 (Codex W-2): %PDF- header + long printable suffix."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    fake_payload = b"%PDF-1.4\n" + (b"This is fake PDF body content without NUL bytes. " * 16)
    txt_file = tmp_path / "renamed.txt"
    txt_file.write_bytes(fake_payload)

    with pytest.raises(MisLabeledFileError) as exc_info:
        SRSDocumentParser().parse_file(str(txt_file))
    assert "PDF" in exc_info.value.detected_format


# =============================================================================
# T-MFE-4 — Microsoft compound document (legacy .doc)
# =============================================================================


def test_read_txt_raises_on_microsoft_compound_doc_magic_bytes(tmp_path):
    """T-MFE-4: OLE compound-file magic bytes \\xD0\\xCF\\x11\\xE0\\xA1\\xB1\\x1A\\xE1."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    fake_payload = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1" + (b"OLE compound body content. " * 16)
    txt_file = tmp_path / "renamed.txt"
    txt_file.write_bytes(fake_payload)

    with pytest.raises(MisLabeledFileError) as exc_info:
        SRSDocumentParser().parse_file(str(txt_file))
    assert "Microsoft" in exc_info.value.detected_format or "compound" in exc_info.value.detected_format.lower()


# =============================================================================
# T-MFE-5 — GREEN regression: legitimate text with literal "PK" substring
# =============================================================================


def test_read_txt_does_not_raise_on_legitimate_text_containing_zip_signature_substring(tmp_path):
    """T-MFE-5 (Codex C-1 reclassified as GREEN regression): legitimate
    text containing "PK\\x03\\x04" as a literal substring MID-CONTENT
    (not at offset 0) must pass through unchanged. The signature check
    uses data.startswith() — only the first bytes matter.

    Pre-GREEN: this test passes because read_txt accepts text.
    Post-GREEN: this test STILL passes (signature only matches at byte 0)."""
    from core.document_parser import SRSDocumentParser

    # Note: this writes the PK bytes literally inside text — but at byte
    # offset > 0 — so startswith() does NOT match. Encoding loop decodes
    # via UTF-8 (NUL-free single-byte content matches cp1254 too) and
    # _looks_like_text passes.
    txt_file = tmp_path / "legit.txt"
    txt_file.write_text(
        "The ZIP signature is PK\\x03\\x04 in hex notation. "
        "This is a legitimate text file.",
        encoding="utf-8",
    )
    content = SRSDocumentParser().parse_file(str(txt_file))
    assert "ZIP signature" in content
    assert "legitimate text file" in content


# =============================================================================
# T-MFE-6 — Realistic ZIP with embedded NUL bytes (diagnostic improvement)
# =============================================================================


def test_read_txt_raises_mislabeled_on_realistic_zip_with_nul_bytes(tmp_path):
    """T-MFE-6 (Codex W-6 diagnostic improvement): a realistic ZIP header
    contains NUL bytes early in the structure (version, flags, time fields).
    Pre-WP-CORE-9 path: cp1254 decodes; _looks_like_text rejects on NUL;
    falls through; raises UnicodeDecodeError with no format detail.
    Post-WP-CORE-9: MisLabeledFileError detected at byte-0 ZIP signature
    BEFORE the decode loop; better error message."""
    from core.document_parser import SRSDocumentParser, MisLabeledFileError

    # Realistic ZIP local-file-header bytes: signature + version + flags + ...
    # Plenty of NUL bytes throughout the structure.
    zip_header = (
        b"PK\x03\x04"           # signature
        b"\x14\x00"               # version needed
        b"\x00\x00"               # general flags (NUL)
        b"\x00\x00"               # compression method (NUL)
        b"\x00\x00"               # last mod time (NUL)
        b"\x00\x00"               # last mod date (NUL)
        b"\x00\x00\x00\x00"       # CRC32 (NUL)
        b"\x00\x00\x00\x00"       # compressed size (NUL)
        b"\x00\x00\x00\x00"       # uncompressed size (NUL)
        b"\x08\x00"               # filename length
        b"\x00\x00"               # extra field length (NUL)
        b"test.txt"               # filename
    )
    txt_file = tmp_path / "renamed.txt"
    txt_file.write_bytes(zip_header)

    with pytest.raises(MisLabeledFileError) as exc_info:
        SRSDocumentParser().parse_file(str(txt_file))
    assert "ZIP" in exc_info.value.detected_format


# =============================================================================
# T-MFE-7 — UTF-8 BOM + later-literal magic bytes
# =============================================================================


def test_read_txt_does_not_raise_on_utf8_bom_with_later_literal_pdf_substring(tmp_path):
    """T-MFE-7 (Codex W-3): UTF-8 BOM (\\xEF\\xBB\\xBF) followed by text
    that contains literal "%PDF-" as a substring (not at offset 0).
    Signature detection uses startswith() — BOM is at offset 0, "%PDF-"
    is mid-content, NEITHER triggers the magic-byte check."""
    from core.document_parser import SRSDocumentParser

    bom = b"\xEF\xBB\xBF"
    body = "This file mentions %PDF-1.4 as a string but is real text.".encode("utf-8")
    txt_file = tmp_path / "bom-text.txt"
    txt_file.write_bytes(bom + body)

    content = SRSDocumentParser().parse_file(str(txt_file))
    assert "%PDF-1.4" in content


# =============================================================================
# T-MFE-8 — Helper-level near-miss prefixes
# =============================================================================


def test_detect_binary_signature_returns_none_for_near_miss_prefixes():
    """T-MFE-8 (Codex W-4): near-miss prefixes (PK\\x03 truncated, %PDX-
    typo, \\x89P truncated PNG) MUST return None from the helper."""
    from core.document_parser_readers import _detect_binary_signature

    near_misses = [
        b"PK\x03",                  # truncated ZIP signature (3 bytes)
        b"PK\x03\x05",              # one byte wrong
        b"%PDX-",                   # typo for %PDF-
        b"\x89P",                   # truncated PNG header
        b"GIF8",                    # truncated GIF
        b"",                        # empty
        b"normal text",             # ASCII text
    ]
    for prefix in near_misses:
        result = _detect_binary_signature(prefix)
        assert result is None, f"near-miss {prefix!r} unexpectedly matched: {result}"

    # Sanity: full signatures DO match.
    assert _detect_binary_signature(b"PK\x03\x04") is not None
    assert _detect_binary_signature(b"%PDF-1.4") is not None
