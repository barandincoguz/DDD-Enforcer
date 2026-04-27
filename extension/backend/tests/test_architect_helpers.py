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
        result = a._is_quota_error_and_backoff(ValueError("not a quota error"), retry_count=0)
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
