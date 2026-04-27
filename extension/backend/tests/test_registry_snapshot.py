"""Snapshot drift guards for the model registry.

Failing one of these tests means a default model has changed. If the change
is intentional, update the asserted strings here in the same commit. The
point of these guards is to make accidental drift impossible.

Run: pytest tests/test_registry_snapshot.py -v
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSnapshotDriftGuard:
    """Failing one of these tests means a default has changed.

    If you intentionally upgraded a default, update the asserted string
    here in the same commit. These guards exist to make accidental drift
    impossible.
    """

    def test_domain_extraction_default(self):
        from configs.models import STAGE_GROUPS

        assert STAGE_GROUPS["domain_extraction"].model_id == "gemini-3.1-pro-preview", (
            "Domain extraction default changed. If intentional, update this assertion in the same commit."
        )

    def test_validation_default(self):
        from configs.models import STAGE_GROUPS

        assert STAGE_GROUPS["validation"].model_id == "gemini-3-flash-preview", (
            "Validation default changed. If intentional, update this assertion in the same commit."
        )

    def test_no_lite_models_anywhere(self):
        """Project policy: no flash-lite tier in defaults or registry."""
        from configs.models import MODELS, STAGE_GROUPS

        for model_id in MODELS:
            assert "lite" not in model_id.lower(), (
                f"Lite tier policy violation: {model_id!r} has 'lite' in name"
            )
        for group in STAGE_GROUPS.values():
            assert "lite" not in group.model_id.lower(), (
                f"Lite tier policy violation in stage group: {group.model_id!r}"
            )
