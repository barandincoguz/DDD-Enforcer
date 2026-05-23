"""WP-CORE-7 — analyze_document end-to-end ArchitectGroundingError propagation.

T-INT-1: Full pipeline with mocked LLM. Architect produces empty
supporting_sentence_ids on initial AND feedback-rerun call (LLM doesn't fix
the grounding violation). Pipeline must raise ArchitectGroundingError with
srs_path populated.

Run: pytest tests/test_analyze_document_e2e.py -v
"""

import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_architect():
    from core.architect import DomainArchitect
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_for_test"}):
        with patch("core.llm.gemini.genai.Client"):
            a = DomainArchitect()
    a.min_delay = 0
    a.last_request_time = 0
    a._rate_limit_lock = threading.Lock()
    a.scout_max_workers = 1
    return a


def _llm_response_with_text(text_payload: str):
    from core.llm.base import LLMResponse, TokenUsage
    return LLMResponse(
        content=text_payload,
        parsed=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        json_failed=False,
    )


# =============================================================================
# T-INT-1 — analyze_document raises ArchitectGroundingError on persistent D1
# =============================================================================


class TestAnalyzeDocumentE2EArchitectGroundingError:
    """WP-CORE-7 mode C hybrid: when the Architect persistently produces
    contexts with empty supporting_sentence_ids, analyze_document raises
    ArchitectGroundingError (not best-effort degrade).

    Import of ArchitectGroundingError is inside the test body per Codex W-2
    so RED commit collection succeeds (the test fails by ImportError at
    body execution, counted as a test failure not a collection error)."""

    def test_analyze_document_e2e_architect_grounding_error_surfaces(self):
        from core.orchestration.errors import ArchitectGroundingError

        arch = _make_architect()

        # Architect returns contexts with EMPTY supporting_sentence_ids
        # twice — initial call + feedback rerun. D1 verifier flags both.
        architect_payload_empty_ids = json.dumps({
            "contexts": [
                {"name": "OrderMgmt", "supporting_sentence_ids": []},
            ],
        })
        specialist_payload = json.dumps({
            "context": "OrderMgmt",
            "entities": [{
                "name": "Order",
                "description": "An order.",
                "attributes": [],
                "confidence": 0.9,
                "justification": "cited in sentence 0",
                "evidence_sentence_indices": [0],
            }],
            "value_objects": [],
            "services": [],
            "aggregates": [],
            "domain_events": [],
            "business_rules": [],
        })

        # Order of LLM calls under WP-CORE-7 mode C:
        # 1) identify_contexts (initial) → empty IDs
        # 2) extract_per_context_details (initial) → OK
        # 3) D1 fails → architect_with_feedback rerun
        # 4) identify_contexts (rerun with feedback) → still empty IDs
        # 5) extract_per_context_details (rerun) → OK
        # 6) D1 fails again → exhaust → raise ArchitectGroundingError
        responses_in_order = [
            _llm_response_with_text(architect_payload_empty_ids),  # initial identify
            _llm_response_with_text(specialist_payload),           # initial spec
            _llm_response_with_text(architect_payload_empty_ids),  # rerun identify
            _llm_response_with_text(specialist_payload),           # rerun spec
        ]
        arch.client = MagicMock()
        arch.client.chat = MagicMock(side_effect=responses_in_order)

        with patch.object(arch, "_save_intermediate"), \
             patch.object(arch, "_report_progress"), \
             patch.object(arch, "_wait_for_rate_limit"), \
             patch.object(arch.token_tracker, "track_api_call"), \
             patch("core.synthesizer.enrich_synonyms_and_dependencies",
                   side_effect=lambda skel, *a, **kw: skel):
            with pytest.raises(ArchitectGroundingError) as exc_info:
                arch.analyze_document(
                    text="An order is placed by a customer.\nOrder contains items.",
                    srs_path="test_e2e.srs",
                )

        assert exc_info.value.srs_path == "test_e2e.srs"
        assert exc_info.value.cycles_attempted == 1
        assert len(exc_info.value.issues) >= 1
