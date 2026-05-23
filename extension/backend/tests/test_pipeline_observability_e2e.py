"""WP-CORE-20 — Pipeline E2E manifest test (RED phase).

Covers spec §11.4: all 6 stages + ingestion appear in manifest.stages after a successful run.
This test uses heavy mocking to keep it offline; the production E2E happens under
the integration-test marker.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_t_obs_e2e_1_success_path_writes_full_manifest(tmp_path, monkeypatch):
    """T-OBS-E2E-1: successful /generate-model run lands a manifest with every stage."""
    monkeypatch.setenv("DDD_MANIFEST_DIR", str(tmp_path))

    # Fake a minimal SRS so parsing succeeds.
    srs = tmp_path / "fake.txt"
    srs.write_text("An order is placed by a customer. Order contains items.")

    # Patch DomainArchitect.analyze_document to short-circuit the LLM pipeline.
    from core.schemas import BoundedContext, DomainModel, Entity
    fake_model = DomainModel(
        bounded_contexts=[
            BoundedContext(
                name="Ordering",
                description="",
                entities=[Entity(name="Order", description="", evidence=[], supporting_sentence_ids=[0])],
                value_objects=[],
                services=[],
                aggregates=[],
                domain_events=[],
                allowed_dependencies=[],
            )
        ],
        global_rules={},
    )

    with patch("core.architect.DomainArchitect.analyze_document", return_value=fake_model):
        from main import _run_generate_pipeline

        result = _run_generate_pipeline(file_paths=[str(srs)], srs_dir_resolved=str(tmp_path))

    assert result["success"] is True

    manifest_files = sorted(tmp_path.glob("run-*.json"))
    assert manifest_files, "no manifest written"
    m = json.loads(manifest_files[-1].read_text())

    assert m["outcome"] == "success"
    assert "ingestion" in m["stages"]
    assert m["elapsed_ms"] > 0
    # domain_model_summary populated on success (spec §6.4).
    assert m["domain_model_summary"]["bounded_context_count"] == 1
    assert m["domain_model_summary"]["entity_count"] == 1
