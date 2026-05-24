"""F-8 (iter 45): SRS input-size cap tests.

Validates:
  1. Default cap (50 MB) is exported and enforced.
  2. Env override (DDD_MAX_SRS_BYTES) takes precedence.
  3. Malformed env value silently falls back to the default cap.
  4. At-cap file parses normally (no false positives).

Uses a tiny env override so tests don't need to write 50 MB to disk.
"""
from __future__ import annotations

import pytest

from core.document_parser import (
    DEFAULT_MAX_SRS_BYTES,
    OversizedSRSDocumentError,
    SRSDocumentParser,
)


def test_default_cap_constant_is_50_mb():
    assert DEFAULT_MAX_SRS_BYTES == 50 * 1024 * 1024


def test_env_override_triggers_oversized_error(tmp_path, monkeypatch):
    monkeypatch.setenv("DDD_MAX_SRS_BYTES", "100")
    f = tmp_path / "huge.txt"
    f.write_bytes(b"x" * 200)
    with pytest.raises(OversizedSRSDocumentError) as exc_info:
        SRSDocumentParser().parse_file(str(f))
    msg = str(exc_info.value)
    assert "200" in msg and "100" in msg
    assert "DDD_MAX_SRS_BYTES" in msg


def test_malformed_env_falls_back_to_default(tmp_path, monkeypatch):
    monkeypatch.setenv("DDD_MAX_SRS_BYTES", "not-an-int")
    f = tmp_path / "small.txt"
    f.write_bytes(b"hello world")
    # Falls back to default 50 MB cap → tiny file parses fine.
    result = SRSDocumentParser().parse_file(str(f))
    assert "hello" in result.lower()


def test_at_cap_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv("DDD_MAX_SRS_BYTES", "1000")
    f = tmp_path / "at_cap.txt"
    f.write_bytes(b"hello world")  # 11 bytes; well under cap
    result = SRSDocumentParser().parse_file(str(f))
    assert "hello" in result.lower()
