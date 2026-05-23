"""WP-CORE-6 — supporting_sentence_ids propagation through the pipeline.

T-PROP-1: Specialist preserves Architect's ContextHypothesis.supporting_sentence_ids (Codex C-1).
T-PROP-2: Synthesizer merge carries IDs into final BoundedContext (regression-lock).
T-INT-1:  analyze_document end-to-end IDs survive Architect → Specialist → Synthesizer (Codex C-3).

Run: pytest tests/test_architect_id_propagation.py -v
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


def _llm_response_with_text(text_payload: str) -> MagicMock:
    """Build an LLMResponse-shaped object for client.chat return."""
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
# T-PROP-1 — Specialist preserves Architect's ContextHypothesis IDs
# =============================================================================


class TestSpecialistPreservesContextHypothesisIds:
    """Codex C-1: extract_per_context_details must accept ContextHypothesis
    inputs and preserve supporting_sentence_ids into SpecialistAnalysis.context."""

    def test_extract_per_context_details_preserves_context_hypothesis_ids(self):
        from core.pipeline_contracts import ContextHypothesis

        arch = _make_architect()
        ctx = ContextHypothesis(
            context_name="OrderMgmt",
            description="",
            supporting_sentence_ids=[0, 3],
        )
        specialist_json = (
            '{"context": "OrderMgmt", "entities": [{"name": "Order", '
            '"description": "An order.", "attributes": [], "confidence": 0.9, '
            '"justification": "cited", "evidence_sentence_indices": [0]}], '
            '"value_objects": [], "services": [], "aggregates": [], '
            '"domain_events": [], "business_rules": []}'
        )
        arch.client = MagicMock()
        arch.client.chat = MagicMock(return_value=_llm_response_with_text(specialist_json))

        with patch.object(arch, "_save_intermediate"), \
             patch.object(arch, "_report_progress"), \
             patch.object(arch, "_wait_for_rate_limit"), \
             patch.object(arch.token_tracker, "track_api_call"):
            # POST-WP-CORE-6 signature: List[ContextHypothesis] (not List[str])
            results = arch.extract_per_context_details([ctx], ["s0", "s1", "s2", "s3"])

        assert len(results) == 1
        assert results[0].context.supporting_sentence_ids == [0, 3], (
            "Specialist must preserve Architect's supporting_sentence_ids "
            "into SpecialistAnalysis.context (Codex C-1)"
        )


# =============================================================================
# T-PROP-2 — Synthesizer merge regression-lock
# =============================================================================


class TestSynthesizerMergePreservesSupportingSentenceIds:
    """Regression-lock: synthesizer/merge.py already copies
    analysis.context.supporting_sentence_ids into BoundedContext.
    Guard against accidental future regression."""

    def test_synthesizer_merge_carries_supporting_sentence_ids_into_bounded_context(self):
        from core.pipeline_contracts import ContextHypothesis, SpecialistAnalysis
        from core.schemas import Entity
        from core.synthesizer.merge import build_deterministic_skeleton

        analysis = SpecialistAnalysis(
            context=ContextHypothesis(
                context_name="OrderMgmt",
                description="",
                supporting_sentence_ids=[5, 9],
            ),
            entities=[Entity(
                name="Order",
                description="An order.",
                confidence=0.9,
                justification="cited",
                evidence_sentence_indices=[5],
            )],
        )

        skeleton = build_deterministic_skeleton([analysis], project_name="T")

        assert len(skeleton.bounded_contexts) == 1
        assert skeleton.bounded_contexts[0].supporting_sentence_ids == [5, 9]


# =============================================================================
# T-INT-1 — analyze_document E2E IDs survive to final DomainModel
# =============================================================================


class TestAnalyzeDocumentE2EPreservesIds:
    """Codex C-3: full pipeline integration test. Mocks LLM responses for
    Scout/Architect/Specialist; verifies final DomainModel has populated IDs."""

    def test_analyze_document_e2e_preserves_supporting_sentence_ids_to_final_domain_model(self):
        arch = _make_architect()

        # 1) Architect identify_contexts → object array with IDs.
        # WP-CORE-7: D1 verifier now enforces non-empty + subset-of-Scout
        # AND raises ArchitectGroundingError on persistent failure (pre-
        # WP-CORE-7 degraded silently). For this test the real Scout chunker
        # emits only index 0 for the two-sentence text, so we cite [0] only.
        architect_payload = json.dumps({
            "contexts": [
                {"name": "OrderMgmt", "supporting_sentence_ids": [0]},
            ],
        })
        # 2) Specialist per-context → valid entity JSON
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

        responses_in_order = [
            _llm_response_with_text(architect_payload),  # identify_contexts
            _llm_response_with_text(specialist_payload),  # extract_per_context_details
        ]
        arch.client = MagicMock()
        arch.client.chat = MagicMock(side_effect=responses_in_order)

        with patch.object(arch, "_save_intermediate"), \
             patch.object(arch, "_report_progress"), \
             patch.object(arch, "_wait_for_rate_limit"), \
             patch.object(arch.token_tracker, "track_api_call"), \
             patch("core.synthesizer.enrich_synonyms_and_dependencies",
                   side_effect=lambda skel, *a, **kw: skel):
            final_model = arch.analyze_document(
                text="An order is placed by a customer.\nOrder contains items.",
                srs_path="test.srs",
            )

        assert len(final_model.bounded_contexts) == 1
        bc = final_model.bounded_contexts[0]
        assert bc.context_name == "OrderMgmt"
        assert bc.supporting_sentence_ids == [0], (
            "End-to-end propagation: Architect IDs must survive through "
            "Specialist + Synthesizer to final DomainModel (Codex C-3)"
        )
