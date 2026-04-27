"""Concurrency smoke for TokenTracker — required before parallel Scout.

Run: pytest tests/test_token_tracker_concurrency.py -v
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_response(prompt_tokens: int, completion_tokens: int):
    response = MagicMock()
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = completion_tokens
    response.usage_metadata.cached_content_token_count = 0
    return response


class TestTokenTrackerConcurrency:
    """track_api_call must be thread-safe; no lost updates under contention."""

    def setup_method(self):
        from core.token_tracker import TokenTracker
        TokenTracker.reset()

    def test_no_lost_updates_under_8_threads_x_100_calls(self):
        """8 threads × 100 calls each = 800 expected total_api_calls."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        n_threads = 8
        n_calls_per_thread = 100

        def worker():
            for _ in range(n_calls_per_thread):
                tracker.track_api_call(
                    _fake_response(prompt_tokens=10, completion_tokens=5),
                    stage="Validator",
                    operation="concurrent_smoke",
                )

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(worker) for _ in range(n_threads)]
            for f in futures:
                f.result()  # raise any exception

        expected_total = n_threads * n_calls_per_thread
        assert tracker.stats.total_api_calls == expected_total, (
            f"Lost updates: expected {expected_total} calls, got "
            f"{tracker.stats.total_api_calls}"
        )
        assert tracker.stats.total_prompt_tokens == expected_total * 10
        assert tracker.stats.total_completion_tokens == expected_total * 5
        validator = tracker.tokens_for_stage("Validator")
        assert validator.call_count == expected_total
