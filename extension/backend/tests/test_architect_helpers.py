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
            with patch("core.architect.genai"):
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
            with patch("core.architect.genai"):
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
