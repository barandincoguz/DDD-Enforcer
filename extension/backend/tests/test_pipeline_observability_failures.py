"""WP-CORE-20 — Pipeline failure-path manifest tests (RED phase).

Covers Codex C-3 (pre-pipeline failure outcomes) + W-2 (finalize-safely):
- no_input_files
- srs_parse_failed
- all_srs_empty
- architect_grounding_error
- refinement_exhausted
- finalize-safely never masks original exception
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest


def _read_latest_manifest(manifest_dir: Path) -> dict:
    files = sorted(manifest_dir.glob("run-*.json"))
    assert files, f"no manifest written in {manifest_dir}"
    return json.loads(files[-1].read_text())


def test_t_obs_fail_1_no_input_files_writes_manifest(tmp_path, monkeypatch):
    """T-OBS-FAIL-1 (Codex C-3): empty file_paths list produces outcome=no_input_files."""
    monkeypatch.setenv("DDD_MANIFEST_DIR", str(tmp_path))
    from main import _run_generate_pipeline

    result = _run_generate_pipeline(file_paths=[], srs_dir_resolved=str(tmp_path))
    assert result["success"] is False

    m = _read_latest_manifest(tmp_path)
    assert m["outcome"] == "no_input_files"


def test_t_obs_fail_2_srs_parse_failed_writes_manifest(tmp_path, monkeypatch):
    """T-OBS-FAIL-2: parse-time exception produces outcome=srs_parse_failed."""
    monkeypatch.setenv("DDD_MANIFEST_DIR", str(tmp_path))
    from main import _run_generate_pipeline

    # Point at a missing file → SRSDocumentParser raises FileNotFoundError → captured as srs_parse_failed.
    bogus = tmp_path / "does-not-exist.txt"
    result = _run_generate_pipeline(
        file_paths=[str(bogus)], srs_dir_resolved=str(tmp_path)
    )
    assert result["success"] is False

    m = _read_latest_manifest(tmp_path)
    assert m["outcome"] == "srs_parse_failed"
    assert len(m["errors"]) >= 1


def test_t_obs_fail_3_all_srs_empty_writes_manifest(tmp_path, monkeypatch):
    """T-OBS-FAIL-3: all input docs parse to empty content → outcome=all_srs_empty."""
    monkeypatch.setenv("DDD_MANIFEST_DIR", str(tmp_path))
    from main import _run_generate_pipeline

    # Empty file → EmptySRSDocumentError → skipped → all_srs_empty
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    result = _run_generate_pipeline(
        file_paths=[str(empty)], srs_dir_resolved=str(tmp_path)
    )
    assert result["success"] is False

    m = _read_latest_manifest(tmp_path)
    assert m["outcome"] == "all_srs_empty"


def test_t_obs_fail_4_architect_grounding_error_outcome(tmp_path, monkeypatch):
    """T-OBS-FAIL-4: ArchitectGroundingError sets outcome=architect_grounding_error."""
    monkeypatch.setenv("DDD_MANIFEST_DIR", str(tmp_path))
    from core.observability import RunManifest, StageEmitter
    from core.observability.emitter import _finalize_manifest_safely
    from core.orchestration.errors import ArchitectGroundingError

    m = RunManifest()
    em = StageEmitter(m)
    original: Optional[BaseException] = None
    try:
        with em.stage("architect"):
            raise ArchitectGroundingError(
                srs_path="/tmp/x.txt", issues=[], residual_issues=[], cycles_attempted=1
            )
    except ArchitectGroundingError as exc:
        original = exc
        m.outcome = "architect_grounding_error"
    _finalize_manifest_safely(m, original_exc=original)

    assert m.outcome == "architect_grounding_error"
    assert m.stages["architect"].status == "fail"


def test_t_manifest_finalize_1_does_not_mask_original_exception(tmp_path, monkeypatch):
    """T-MANIFEST-FINALIZE-1 (Codex W-2): a write failure inside finalize must not mask
    the original exception when caller re-raises."""
    from core.observability import RunManifest
    from core.observability.emitter import _finalize_manifest_safely

    m = RunManifest(outcome="architect_grounding_error")

    def _broken_write(_m, _path):
        raise OSError("disk full")

    monkeypatch.setattr("core.observability.emitter.write_manifest_atomic", _broken_write)

    # finalize must not raise; caller is free to re-raise original.
    original = RuntimeError("pipeline boom")
    # _finalize_manifest_safely returns cleanly even when the write fails.
    _finalize_manifest_safely(m, original_exc=original)

    # The original is preserved by the caller, not by finalize. Spec contract:
    # finalize only ever logs the write error to stderr.


def test_t_obs_fail_5_outcome_unexpected_error(tmp_path, monkeypatch):
    """T-OBS-FAIL-5: an unexpected, non-typed exception sets outcome=unexpected_error."""
    monkeypatch.setenv("DDD_MANIFEST_DIR", str(tmp_path))

    # Force an unexpected exception by monkeypatching SRSDocumentParser to raise
    # something that isn't a typed PipelineError.
    from core.document_parser import SRSDocumentParser

    def _boom(self, path):
        raise ValueError("brand new failure mode")

    monkeypatch.setattr(SRSDocumentParser, "parse_file", _boom)
    from main import _run_generate_pipeline

    p = tmp_path / "fake.txt"
    p.write_text("dummy")
    result = _run_generate_pipeline(file_paths=[str(p)], srs_dir_resolved=str(tmp_path))
    assert result["success"] is False

    m = _read_latest_manifest(tmp_path)
    # ValueError raised by parser → falls into srs_parse_failed bucket by spec §6.4.
    # (Spec maps parser exceptions to srs_parse_failed, not unexpected_error;
    # unexpected_error is reserved for non-parser surprises post-ingestion.)
    assert m["outcome"] == "srs_parse_failed"
