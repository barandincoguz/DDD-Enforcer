"""WP-CORE-20 — Atomic-write tests (RED phase).

Covers Codex W-7: tmp + fsync + os.replace + no .tmp left behind.
"""
from __future__ import annotations

import json

import pytest


def test_t_atomic_write_1_no_tmp_left_on_success(tmp_path):
    """T-ATOMIC-WRITE-1: successful write leaves only the final .json, not .tmp."""
    from core.observability import RunManifest
    from core.observability.run_manifest import write_manifest_atomic

    m = RunManifest(outcome="success")
    target = tmp_path / "manifests" / "run-test.json"
    write_manifest_atomic(m, target)

    assert target.exists()
    assert target.read_text().startswith("{")
    # No .tmp left behind.
    tmps = list((tmp_path / "manifests").glob("*.tmp"))
    assert tmps == [], f"unexpected tmp files: {tmps}"


def test_t_atomic_write_2_payload_round_trips(tmp_path):
    """T-ATOMIC-WRITE-2: written file parses back to a manifest with same run_id."""
    from core.observability import RunManifest
    from core.observability.run_manifest import write_manifest_atomic

    m = RunManifest(outcome="success")
    m.elapsed_ms = 999.9
    target = tmp_path / "run-x.json"
    write_manifest_atomic(m, target)

    loaded = RunManifest.model_validate_json(target.read_text())
    assert loaded.run_id == m.run_id
    assert loaded.elapsed_ms == 999.9
    assert loaded.outcome == "success"


def test_t_atomic_write_3_parent_dir_created(tmp_path):
    """T-ATOMIC-WRITE-3: nested parent dirs are created on demand."""
    from core.observability import RunManifest
    from core.observability.run_manifest import write_manifest_atomic

    m = RunManifest()
    target = tmp_path / "deep" / "nested" / "path" / "run.json"
    write_manifest_atomic(m, target)
    assert target.exists()
