"""Phase D4: integration regression. After a full pipeline run, every
persisted entity must have a non-empty evidence_sentence_indices, and
no InferenceSource may have rule='LLM_SYNTHESIS' or file='generated'.
"""

import json
import os
import pytest
from pathlib import Path


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("DDD_INTEGRATION_TEST") != "1",
    reason="integration test gated by DDD_INTEGRATION_TEST=1"
)
def test_persisted_model_has_real_evidence():
    """Assumes the integration test from Task C8 already ran and
    persisted domain/model.json. Loads it and asserts.
    """
    candidates = [
        Path("domain/model.json"),
        Path("extension/backend/domain/model.json"),
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        pytest.skip("No domain/model.json found; run the C8 integration test first")

    model = json.loads(model_path.read_text())
    for bc in model.get("bounded_contexts", []):
        for entity in bc.get("ubiquitous_language", {}).get("entities", []):
            assert entity.get("evidence_sentence_indices"), (
                f"Entity {entity.get('name')!r} in context "
                f"{bc.get('context_name')!r} has no evidence_sentence_indices"
            )
            for src in entity.get("sources", []):
                assert src.get("rule") != "LLM_SYNTHESIS", (
                    f"Entity {entity.get('name')!r} carries a forbidden "
                    f"InferenceSource rule='LLM_SYNTHESIS'"
                )
                assert src.get("file") != "generated"
