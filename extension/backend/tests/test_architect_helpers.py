"""Unit tests for DomainArchitect helper methods.

Run: pytest tests/test_architect_helpers.py -v
"""

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSafeResponseText:
    """DomainArchitect._safe_response_text — None-safe wrapper around response.text."""

    def _make_architect(self):
        from core.architect import DomainArchitect
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_for_test"}):
            with patch("core.llm.gemini.genai.Client"):
                return DomainArchitect()

    def test_returns_text_when_present(self):
        a = self._make_architect()
        resp = Mock()
        resp.text = "hello world"
        assert a._safe_response_text(resp) == "hello world"

    def test_returns_empty_when_text_is_none(self):
        a = self._make_architect()
        resp = Mock()
        resp.text = None
        assert a._safe_response_text(resp) == ""

    def test_returns_empty_when_text_attr_missing(self):
        a = self._make_architect()
        resp = Mock(spec=[])  # mock with no attributes
        assert a._safe_response_text(resp) == ""


class TestQuotaErrorBackoff:
    """DomainArchitect._is_quota_error_and_backoff — explicit boolean-return semantics."""

    def _make_architect(self):
        from core.architect import DomainArchitect
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_for_test"}):
            with patch("core.llm.gemini.genai.Client"):
                return DomainArchitect()

    def test_returns_false_for_non_quota_error(self):
        a = self._make_architect()
        # Note: error message must NOT contain "quota", "429", or "ResourceExhausted"
        # — those substrings trigger the heuristic match.
        result = a._is_quota_error_and_backoff(ValueError("connection timeout"), retry_count=0)
        assert result is False

    def test_returns_true_for_429(self):
        a = self._make_architect()
        with patch("core.architect.time.sleep") as mock_sleep:
            result = a._is_quota_error_and_backoff(
                Exception("429: Too Many Requests"),
                retry_count=0,
            )
        assert result is True
        mock_sleep.assert_called_once()  # backoff happened


class TestTruncateWithHeadTail:
    """Module-level helper that preserves head + tail when truncating."""

    def test_no_truncation_when_under_budget(self):
        from core.architect import _truncate_with_head_tail
        assert _truncate_with_head_tail("short text", 100) == "short text"

    def test_keeps_head_and_tail(self):
        from core.architect import _truncate_with_head_tail
        text = "A" * 1000 + "B" * 1000
        result = _truncate_with_head_tail(text, max_chars=400, head_ratio=0.6)
        assert "A" in result[:200]   # head present
        assert "B" in result[-200:]  # tail present
        assert "[middle truncated" in result

    def test_total_length_within_budget(self):
        from core.architect import _truncate_with_head_tail
        text = "X" * 10_000
        result = _truncate_with_head_tail(text, max_chars=500)
        assert len(result) <= 500


class TestScoutParallel:
    """Parallel Scout chunk smoke — opt-in via scout_max_workers > 1."""

    def _make_architect(self, scout_max_workers=None):
        from core.architect import DomainArchitect
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_for_test"}):
            with patch("core.llm.gemini.genai.Client"):
                return DomainArchitect(scout_max_workers=scout_max_workers)

    def test_default_is_sequential(self):
        a = self._make_architect()
        assert a.scout_max_workers == 1

    def test_kwarg_overrides_default(self):
        a = self._make_architect(scout_max_workers=4)
        assert a.scout_max_workers == 4


class TestTruncateNumberedPairs:
    """Tests for _truncate_numbered_pairs helper."""

    def test_no_truncation_when_under_budget(self):
        from core.architect import _truncate_numbered_pairs
        pairs = [(0, "hello"), (1, "world")]
        assert _truncate_numbered_pairs(pairs, 100) == pairs

    def test_truncation_drops_middle_pairs(self):
        from core.architect import _truncate_numbered_pairs
        # total chars of 5 pairs of 100 characters ≈ 500 characters
        pairs = [(i, "A" * 100) for i in range(5)]
        # max_chars=300 with head_ratio=0.5: head_budget=150 (takes index 0), tail_budget=150 (takes index 4)
        result = _truncate_numbered_pairs(pairs, 300, head_ratio=0.5)
        assert len(result) == 2
        assert result[0][0] == 0
        assert result[1][0] == 4

    def test_fallback_when_single_pair_exceeds_budget(self):
        from core.architect import _truncate_numbered_pairs
        pairs = [(0, "A" * 1000)]
        result = _truncate_numbered_pairs(pairs, 100)
        assert len(result) == 1
        assert result[0][0] == 0
        assert len(result[0][1]) < 100
        assert len(result[0][1]) > 50


class TestValidateSpecialistPayload:
    """Tests for DomainArchitect._validate_specialist_payload helper."""

    def test_coerces_dict_business_rules_to_strings(self):
        from core.architect import DomainArchitect
        from core.pipeline_contracts import ContextHypothesis

        ctx = ContextHypothesis(context_name="JobPosting", supporting_sentence_ids=[0])
        payload = {
            "context": "JobPosting",
            "business_rules": [
                {
                    "name": "Schema.org Compliance",
                    "description": "All input data for job postings must comply...",
                },
                "Plain string rule",
            ],
            "entities": [],
            "value_objects": [],
            "services": [],
            "aggregates": [],
            "domain_events": [],
            "ambiguities": [],
        }
        # Run validation
        analysis = DomainArchitect._validate_specialist_payload(payload, ctx)
        assert len(analysis.business_rules) == 2
        assert analysis.business_rules[0] == "Schema.org Compliance: All input data for job postings must comply..."
        assert analysis.business_rules[1] == "Plain string rule"
