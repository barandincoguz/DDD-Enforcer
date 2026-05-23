"""WP-CORE-6 — Architect identify_contexts returns dict shape with supporting_sentence_ids.

T-ARCH-1: returns {"name": str, "supporting_sentence_ids": List[int]} dicts.
T-ARCH-2: strict-shape rejection of old dict {"contexts": ["X"]} → retry.
T-ARCH-2b: strict-shape rejection of top-level list ["X", "Y"] → retry.
T-ARCH-3: prompt includes [N] sentence numbering.

Run: pytest tests/test_architect_identify_contexts.py -v
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


def _llm_response_json(payload):
    """Build an LLMResponse with JSON content matching client.chat return shape."""
    from core.llm.base import LLMResponse, TokenUsage
    return LLMResponse(
        content=json.dumps(payload),
        parsed=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        json_failed=False,
    )


class TestIdentifyContextsReturnShape:
    """T-ARCH-1: identify_contexts returns object array with supporting_sentence_ids."""

    def test_identify_contexts_returns_dict_shape_with_supporting_sentence_ids(self):
        arch = _make_architect()
        payload = {"contexts": [
            {"name": "OrderMgmt", "supporting_sentence_ids": [0, 2]},
        ]}
        arch.client = MagicMock()
        arch.client.chat = MagicMock(return_value=_llm_response_json(payload))

        result = arch.identify_contexts(["s0", "s1", "s2"])

        assert result == [
            {"name": "OrderMgmt", "supporting_sentence_ids": [0, 2]},
        ]


class TestIdentifyContextsStrictShape:
    """T-ARCH-2 + T-ARCH-2b: strict-shape rejection of legacy responses."""

    def test_identify_contexts_retries_on_old_dict_shape(self):
        """T-ARCH-2: old dict shape {"contexts": ["X"]} (bare strings) must
        be rejected and retried."""
        arch = _make_architect()
        old_shape = {"contexts": ["OrderMgmt"]}
        new_shape = {"contexts": [{"name": "OrderMgmt", "supporting_sentence_ids": [0]}]}
        responses = [_llm_response_json(old_shape)] * 4 + [_llm_response_json(new_shape)]
        arch.client = MagicMock()
        arch.client.chat = MagicMock(side_effect=responses)

        result = arch.identify_contexts(["s0", "s1"])

        assert result == [{"name": "OrderMgmt", "supporting_sentence_ids": [0]}]
        assert arch.client.chat.call_count == 5

    def test_identify_contexts_retries_on_top_level_list_shape(self):
        """T-ARCH-2b (Codex W-2): top-level list ["X", "Y"] (no dict wrapper)
        must be rejected and retried."""
        arch = _make_architect()
        bare_list = ["OrderMgmt", "Inventory"]
        new_shape = {"contexts": [
            {"name": "OrderMgmt", "supporting_sentence_ids": [0]},
            {"name": "Inventory", "supporting_sentence_ids": [1]},
        ]}
        responses = [_llm_response_json(bare_list)] * 4 + [_llm_response_json(new_shape)]
        arch.client = MagicMock()
        arch.client.chat = MagicMock(side_effect=responses)

        result = arch.identify_contexts(["s0", "s1"])

        assert len(result) == 2
        assert result[0]["name"] == "OrderMgmt"
        assert arch.client.chat.call_count == 5


class TestIdentifyContextsPromptNumbering:
    """T-ARCH-3: prompt embeds numbered sentences via [N] prefix."""

    def test_identify_contexts_prompt_includes_numbered_sentences(self):
        arch = _make_architect()
        payload = {"contexts": [
            {"name": "OrderMgmt", "supporting_sentence_ids": [0]},
        ]}
        arch.client = MagicMock()
        arch.client.chat = MagicMock(return_value=_llm_response_json(payload))

        arch.identify_contexts(["sent zero text", "sent one text"])

        # Inspect prompt passed to client.chat
        prompt = arch.client.chat.call_args.kwargs["messages"][0]["content"]
        assert "[0] sent zero text" in prompt
        assert "[1] sent one text" in prompt


# =============================================================================
# WP-CORE-7 — Architect feedback re-prompt (F-22 mode C hybrid)
# =============================================================================


class TestIdentifyContextsFeedbackBlock:
    """T-FEEDBACK-1: identify_contexts accepts feedback_issues kwarg and
    prepends a structured feedback block to the LLM prompt."""

    def test_identify_contexts_prepends_feedback_block_when_issues_provided(self):
        """T-FEEDBACK-1 (Codex W-3): the feedback block contains exactly:
        (a) "PREVIOUS ATTEMPT FAILED VERIFICATION:" header,
        (b) the issue.target field value,
        (c) the issue.message text.

        Feedback is built ONCE per outer architect attempt (Codex N-1) and
        reused across all 5 internal JSON-parse retries.
        """
        from core.pipeline_contracts import VerifierIssue as ContractVerifierIssue

        arch = _make_architect()
        payload = {"contexts": [
            {"name": "OrderMgmt", "supporting_sentence_ids": [0]},
        ]}
        arch.client = MagicMock()
        arch.client.chat = MagicMock(return_value=_llm_response_json(payload))

        feedback_issue = ContractVerifierIssue(
            severity="ERROR",
            check_id="D1",
            target="architect:contexts[Inventory].supporting_sentence_ids",
            message="Context 'Inventory' has no supporting_sentence_ids",
        )

        arch.identify_contexts(
            ["sent zero text", "sent one text"],
            feedback_issues=[feedback_issue],
        )

        prompt = arch.client.chat.call_args.kwargs["messages"][0]["content"]
        assert "PREVIOUS ATTEMPT FAILED VERIFICATION:" in prompt
        assert "architect:contexts[Inventory].supporting_sentence_ids" in prompt
        assert "Context 'Inventory' has no supporting_sentence_ids" in prompt
        # Sanity: feedback appears BEFORE the main instruction block.
        feedback_pos = prompt.find("PREVIOUS ATTEMPT FAILED VERIFICATION:")
        main_pos = prompt.find("Identify distinct Bounded Contexts")
        assert feedback_pos >= 0 and main_pos >= 0
        assert feedback_pos < main_pos, "feedback block must come before main instructions"
