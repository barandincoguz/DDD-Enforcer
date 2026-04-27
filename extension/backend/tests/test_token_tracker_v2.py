"""Tests for the registry-driven TokenTracker (post-refactor contract).

Run: pytest tests/test_token_tracker_v2.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_gemini_response(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0):
    """Build a Mock that mimics google.genai response.usage_metadata."""
    response = MagicMock()
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = completion_tokens
    response.usage_metadata.cached_content_token_count = cached_tokens
    return response


class TestTokenTrackerV2:
    """Post-refactor TokenTracker contract."""

    def setup_method(self):
        from core.token_tracker import TokenTracker
        TokenTracker.reset()

    def test_track_validator_call_uses_validator_pricing(self):
        """Tracking a Validator call applies gemini-3-flash-preview pricing."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        # 100k input + 1k output via Validator stage (gemini-3-flash-preview, flat $0.50/$3)
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100_000, completion_tokens=1_000),
            stage="Validator",
            operation="validate_code",
        )
        accum = tracker.tokens_for_stage("Validator")
        assert accum.prompt_tokens == 100_000
        assert accum.completion_tokens == 1_000
        assert accum.cost_usd == pytest.approx(100_000 * 0.50 / 1_000_000 + 1_000 * 3.0 / 1_000_000)
        assert accum.call_count == 1

    def test_track_architect_call_uses_pro_pricing_under_200k(self):
        """Architect at <200k prompt tokens applies gemini-3.1-pro-preview cheap tier."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=50_000, completion_tokens=2_000),
            stage="Architect",
            operation="identify_contexts",
        )
        accum = tracker.tokens_for_stage("Architect")
        # 50k * 2.0 / 1M + 2k * 12.0 / 1M
        expected = 50_000 * 2.0 / 1_000_000 + 2_000 * 12.0 / 1_000_000
        assert accum.cost_usd == pytest.approx(expected)

    def test_track_architect_call_uses_pro_pricing_over_200k(self):
        """Architect at >200k prompt tokens applies gemini-3.1-pro-preview expensive tier."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=300_000, completion_tokens=2_000),
            stage="Architect",
            operation="identify_contexts",
        )
        accum = tracker.tokens_for_stage("Architect")
        # 300k * 4.0 / 1M + 2k * 18.0 / 1M
        expected = 300_000 * 4.0 / 1_000_000 + 2_000 * 18.0 / 1_000_000
        assert accum.cost_usd == pytest.approx(expected)

    def test_cached_tokens_excluded_from_billable(self):
        """cached_content_token_count is subtracted from billable prompt tokens."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=10_000, completion_tokens=500, cached_tokens=4_000),
            stage="Validator",
            operation="validate_code",
        )
        accum = tracker.tokens_for_stage("Validator")
        # Billable input = 10_000 - 4_000 = 6_000
        assert accum.prompt_tokens == 6_000

    def test_tokens_for_stage_unknown_stage_returns_zero(self):
        """Querying a stage that has not been tracked returns a zero accumulator."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        accum = tracker.tokens_for_stage("Validator")
        assert accum.prompt_tokens == 0
        assert accum.completion_tokens == 0
        assert accum.call_count == 0

    def test_get_report_uses_model_id_keys(self):
        """get_report()['model_usage'] is keyed by full model_id, not flash/flash-lite."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        report = tracker.get_report()
        assert "gemini-3-flash-preview" in report["model_usage"]
        # No flash-lite anywhere in the report.
        assert all("lite" not in k.lower() for k in report["model_usage"].keys())
        assert all("lite" not in k.lower() for k in report.get("cost_estimation", {}).get("by_model", {}).keys())

    def test_get_report_cost_estimation_has_by_model(self):
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=200, completion_tokens=20),
            stage="Architect",
            operation="identify_contexts",
        )
        report = tracker.get_report()
        ce = report["cost_estimation"]
        assert "by_model" in ce
        assert "gemini-3-flash-preview" in ce["by_model"]
        assert "gemini-3.1-pro-preview" in ce["by_model"]
        assert ce["currency"] == "USD"
        assert ce["total_cost"] >= 0.0

    def test_no_legacy_keys_in_report(self):
        """flash_model, flash_lite_model, gemini-2.5-* keys are gone."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        report = tracker.get_report()
        ce = report.get("cost_estimation", {})
        for legacy in ("flash_model", "flash_lite_model"):
            assert legacy not in ce, f"Legacy key {legacy!r} still present in cost_estimation"
        for legacy in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
            assert legacy not in report["model_usage"], f"Legacy model_usage key {legacy!r} still present"

    def test_get_combined_metrics_per_stage(self):
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        m = tracker.get_combined_metrics()
        assert m["api_calls"] == 1
        assert m["total_input_tokens"] == 100
        assert m["total_output_tokens"] == 10
        assert "Validator" in m["by_stage"]
        assert m["by_stage"]["Validator"]["api_calls"] == 1
