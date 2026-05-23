"""Tests for core.run_manifest — PaperRunManifest schema + writer.

TDD order: all tests written RED (before production code), then GREEN.
WP-01b Task A acceptance criteria: schema definition + atomic writer.

Test IDs correspond to T-01b-A-{N} in the WP-01b spec.
"""

from __future__ import annotations

import hashlib
import os
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.run_manifest import (
    PaperRunManifest,
    Violation,
    compose_run_id,
    default_runs_root,
    sha256_of_code_tree,
    sha256_of_file,
    write_paper_run_manifest,
)

# ---------------------------------------------------------------------------
# T-01b-A-1  Instantiation with required fields + defaults
# ---------------------------------------------------------------------------


def test_T01b_A1_instantiation_defaults():
    """PaperRunManifest instantiates with required fields; defaults applied."""
    manifest = PaperRunManifest(
        run_id="P1_gemini-3_1-pro-preview_D1_SRS_2026-05-23T18:30:00Z_seed42",
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        srs_path="/workspace/inputs/SRS.docx",
        srs_sha256="a" * 64,
    )

    assert manifest.pipeline is None
    assert manifest.violations == []
    assert manifest.schema_version == "1.0"

    # timestamp_utc should parse as ISO-8601 UTC
    ts = manifest.timestamp_utc
    assert ts.endswith("Z"), f"Expected Z suffix, got: {ts!r}"
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


# ---------------------------------------------------------------------------
# T-01b-A-2  compose_run_id returns path-safe string
# ---------------------------------------------------------------------------


def test_T01b_A2_compose_run_id_path_safe():
    """compose_run_id returns a string with no '/', '.', ' ', or ':' characters.

    The ISO-8601 timestamp ``"2026-05-23T18:30:00Z"`` contains colons; they
    must be sanitised because colons are illegal in Windows directory names and
    ``run_id`` is used as a directory name under ``runs/``.
    """
    result = compose_run_id(
        "P1",
        "gemini-3.1-pro-preview",
        "D1_SRS",
        "2026-05-23T18:30:00Z",
        "seed-42",
    )
    assert "/" not in result, f"'/' found in run_id: {result!r}"
    assert "." not in result, f"'.' found in run_id: {result!r}"
    assert " " not in result, f"space found in run_id: {result!r}"
    assert ":" not in result, f"':' found in run_id: {result!r}"


# ---------------------------------------------------------------------------
# T-01b-A-3  compose_run_id is deterministic
# ---------------------------------------------------------------------------


def test_T01b_A3_compose_run_id_deterministic():
    """Same arguments → same output, always."""
    args = ("P1", "gemini-3.1-pro-preview", "D1_SRS", "2026-05-23T18:30:00Z", "seed-42")
    assert compose_run_id(*args) == compose_run_id(*args)
    assert compose_run_id(*args) == compose_run_id(*args)


# ---------------------------------------------------------------------------
# T-01b-A-4  pipeline field accepts P1/P2/P3/None; rejects P4
# ---------------------------------------------------------------------------


def test_T01b_A4_pipeline_literal_accepts():
    """pipeline field accepts 'P1', 'P2', 'P3', and None."""
    base = dict(
        run_id="r",
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        srs_path="/x",
        srs_sha256="a" * 64,
    )
    for val in ("P1", "P2", "P3", None):
        m = PaperRunManifest(**base, pipeline=val)
        assert m.pipeline == val


def test_T01b_A4_pipeline_literal_rejects_P4():
    """pipeline='P4' must raise Pydantic ValidationError."""
    with pytest.raises(ValidationError):
        PaperRunManifest(
            run_id="r",
            model_id="gemini-3.1-pro-preview",
            provider="gemini",
            srs_path="/x",
            srs_sha256="a" * 64,
            pipeline="P4",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# T-01b-A-5  provider rejects unknown providers (D1 lock)
# ---------------------------------------------------------------------------


def test_T01b_A5_provider_rejects_unknown():
    """provider='openai' must raise ValidationError (D1 lock)."""
    with pytest.raises(ValidationError):
        PaperRunManifest(
            run_id="r",
            model_id="gpt-4o",
            provider="openai",  # type: ignore[arg-type]
            srs_path="/x",
            srs_sha256="a" * 64,
        )


# ---------------------------------------------------------------------------
# T-01b-A-6  Violation.severity rejects "INFO"
# ---------------------------------------------------------------------------


def test_T01b_A6_violation_severity():
    """Violation.severity accepts ERROR/WARN; rejects INFO."""
    v_error = Violation(
        violation_type="ubiquitous_language_drift",
        location="src/foo.py",
        severity="ERROR",
        message="drift detected",
    )
    assert v_error.severity == "ERROR"

    v_warn = Violation(
        violation_type="entity_field_missing",
        location="src/bar.py",
        severity="WARN",
        message="missing field",
    )
    assert v_warn.severity == "WARN"

    with pytest.raises(ValidationError):
        Violation(
            violation_type="some_type",
            location="src/baz.py",
            severity="INFO",  # type: ignore[arg-type]
            message="should fail",
        )


# ---------------------------------------------------------------------------
# T-01b-A-7  sha256_of_file matches known digest
# ---------------------------------------------------------------------------


def test_T01b_A7_sha256_of_file(tmp_path):
    """sha256_of_file produces the correct SHA-256 for known content."""
    content = b"Hello, DDD-Enforcer paper data!\n"
    f = tmp_path / "sample.txt"
    f.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    result = sha256_of_file(f)

    assert result == expected
    assert len(result) == 64
    assert result == result.lower()


# ---------------------------------------------------------------------------
# T-01b-A-8  sha256_of_code_tree canonical sort + change detection
# ---------------------------------------------------------------------------


def test_T01b_A8_sha256_of_code_tree(tmp_path):
    """sha256_of_code_tree is canonical: order-independent; changes with content."""
    dir_a = tmp_path / "proj_a"
    dir_a.mkdir()
    (dir_a / "alpha.py").write_bytes(b"x = 1\n")
    (dir_a / "beta.py").write_bytes(b"y = 2\n")

    # Same tree written in reverse creation order under a second directory
    dir_b = tmp_path / "proj_b"
    dir_b.mkdir()
    (dir_b / "beta.py").write_bytes(b"y = 2\n")
    (dir_b / "alpha.py").write_bytes(b"x = 1\n")

    digest_a = sha256_of_code_tree(dir_a)
    digest_b = sha256_of_code_tree(dir_b)

    assert len(digest_a) == 64
    assert digest_a == digest_b, "digest should be canonical (sorted), not creation-order dependent"

    # Adding a third file must change the digest
    (dir_a / "gamma.py").write_bytes(b"z = 3\n")
    digest_a2 = sha256_of_code_tree(dir_a)
    assert digest_a2 != digest_a, "adding a file must change the digest"


# ---------------------------------------------------------------------------
# T-01b-A-9  sha256_of_code_tree returns empty-hash sentinel for empty dir
# ---------------------------------------------------------------------------

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_T01b_A9_sha256_of_code_tree_empty(tmp_path):
    """sha256_of_code_tree returns the empty-hash sentinel for an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert sha256_of_code_tree(empty_dir) == EMPTY_SHA256


def test_T01b_A9_sha256_of_code_tree_nonexistent(tmp_path):
    """sha256_of_code_tree returns the sentinel for a non-existent root (no raise)."""
    missing = tmp_path / "does_not_exist"
    assert sha256_of_code_tree(missing) == EMPTY_SHA256


# ---------------------------------------------------------------------------
# T-01b-A-10  write_paper_run_manifest round-trip
# ---------------------------------------------------------------------------


def test_T01b_A10_write_round_trip(tmp_path):
    """write_paper_run_manifest creates manifest.json; round-trip via model_validate_json."""
    manifest = PaperRunManifest(
        run_id="P1_gemini_D1SRS_2026-05-23T00:00:00Z_seed1",
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        srs_path="/workspace/inputs/SRS.docx",
        srs_sha256="b" * 64,
        violations=[
            Violation(
                violation_type="ubiquitous_language_drift",
                location="src/order.py",
                severity="ERROR",
                message="drift",
            )
        ],
    )
    runs_root = tmp_path / "runs"

    written_path = write_paper_run_manifest(manifest, runs_root)

    expected_path = runs_root / manifest.run_id / "manifest.json"
    assert written_path == expected_path
    assert written_path.exists()

    recovered = PaperRunManifest.model_validate_json(written_path.read_text())
    assert recovered == manifest


# ---------------------------------------------------------------------------
# T-01b-A-11  Atomic write: partial write must NOT reach target
# ---------------------------------------------------------------------------


def test_T01b_A11_atomic_write_no_partial(tmp_path):
    """On os.replace raising, the target file must NOT exist (only .tmp on disk)."""
    manifest = PaperRunManifest(
        run_id="P1_gemini_D1SRS_2026-05-23T00:00:00Z_seed2",
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        srs_path="/workspace/inputs/SRS.docx",
        srs_sha256="c" * 64,
    )
    runs_root = tmp_path / "runs"
    target = runs_root / manifest.run_id / "manifest.json"

    with unittest.mock.patch("os.replace", side_effect=OSError("simulated crash")):
        with pytest.raises(OSError, match="simulated crash"):
            write_paper_run_manifest(manifest, runs_root)

    # Target must NOT exist — only the .tmp file should be on disk
    assert not target.exists(), "Target file must not exist after a failed atomic write"


# ---------------------------------------------------------------------------
# T-01b-A-12  default_runs_root respects DDD_RUNS_DIR; falls back to <backend>/runs
# ---------------------------------------------------------------------------


def test_T01b_A12_default_runs_root_env(tmp_path, monkeypatch):
    """default_runs_root returns DDD_RUNS_DIR path when env var is set."""
    custom = str(tmp_path / "custom_runs")
    monkeypatch.setenv("DDD_RUNS_DIR", custom)
    result = default_runs_root()
    assert result == Path(custom)


def test_T01b_A12_default_runs_root_fallback(monkeypatch):
    """default_runs_root falls back to <backend>/runs/ when DDD_RUNS_DIR is unset."""
    monkeypatch.delenv("DDD_RUNS_DIR", raising=False)
    result = default_runs_root()
    # Should end with 'runs' and be inside the backend directory
    assert result.name == "runs"
    assert "extension" in str(result) or "backend" in str(result)
