"""WP-CORE-4 — RED-phase tests for F-13 (silent I/O swallow in _save_intermediate).

These tests fail until the GREEN commit lands:
- IntermediateSaveError(PipelineError) in core/orchestration/errors.py
- _save_intermediate raises IntermediateSaveError on (OSError, TypeError, ValueError)
- identify_contexts retry handler re-raises IntermediateSaveError (Codex C-1)

T-SAVE-1 — happy path (file round-trips)
T-SAVE-2 — filesystem failure raises IntermediateSaveError
T-SAVE-3 — non-serializable data raises IntermediateSaveError
T-SAVE-4 — failure inside identify_contexts propagates IntermediateSaveError
T-SAVE-5 — failure inside extract_per_context_details propagates IntermediateSaveError
"""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from core.architect import DomainArchitect
from core.token_tracker import TokenTracker

# NOTE: IntermediateSaveError is intentionally imported inside each test function
# below. This file is part of the WP-CORE-4 RED phase: the symbol does not yet
# exist in core.orchestration.errors until the GREEN commit lands. Importing at
# module level would cause a collection error; per-test import keeps collection
# passing and each test fails individually with ImportError until GREEN.


def _bare_arch():
    """Bare DomainArchitect for tests — bypasses __init__ env var checks.

    Sets up only the attributes needed by methods under test, including
    `_current_srs_path = "<unknown>"` so the new IntermediateSaveError
    constructor has a value to surface.
    """
    a = DomainArchitect.__new__(DomainArchitect)
    a.model_name = "gemini-3.1-pro-preview"
    a.last_request_time = 0
    a.min_delay = 0
    a.request_count = 0
    a._rate_limit_lock = threading.Lock()
    a.scout_max_workers = 1
    a.token_tracker = TokenTracker.get_instance()
    a.progress_callback = None
    a.run_timestamp = "20260521_140000"
    a.client = MagicMock()
    a._current_srs_path = "<unknown>"
    return a


def test_save_intermediate_happy_path_round_trips_json(tmp_path, monkeypatch):
    """T-SAVE-1: happy path writes JSON, content round-trips."""
    monkeypatch.setattr("core.architect.INTERMEDIATE_DIR", str(tmp_path))
    arch = _bare_arch()

    payload = {"x": 1, "list": [1, 2, 3], "nested": {"k": "v"}}
    arch._save_intermediate("test_stage_happy", payload)

    files = list(tmp_path.glob("*_test_stage_happy.json"))
    assert len(files) == 1, f"expected exactly 1 file, found {files}"
    assert json.loads(files[0].read_text(encoding="utf-8")) == payload


def test_save_intermediate_raises_on_filesystem_error(tmp_path, monkeypatch):
    """T-SAVE-2: PermissionError → IntermediateSaveError(PipelineError)."""
    from core.orchestration.errors import IntermediateSaveError, PipelineError
    monkeypatch.setattr("core.architect.INTERMEDIATE_DIR", str(tmp_path))
    arch = _bare_arch()

    # Patch open() at core.architect's namespace to raise PermissionError.
    # `create=True` lets us patch a name that core.architect doesn't explicitly
    # import (it uses the builtin open).
    with patch("core.architect.open", create=True, side_effect=PermissionError("read-only")):
        with pytest.raises(IntermediateSaveError) as exc_info:
            arch._save_intermediate("test_stage_fail", {"x": 1})

    assert exc_info.value.stage == "test_stage_fail"
    assert isinstance(exc_info.value.cause, PermissionError)
    assert isinstance(exc_info.value, PipelineError), \
        "IntermediateSaveError must subclass PipelineError per Codex W-1"
    assert exc_info.value.srs_path == "<unknown>", \
        "srs_path should default to <unknown> when analyze_document not called"


def test_save_intermediate_raises_on_non_serializable_data(tmp_path, monkeypatch):
    """T-SAVE-3: TypeError from json.dump → IntermediateSaveError."""
    from core.orchestration.errors import IntermediateSaveError
    monkeypatch.setattr("core.architect.INTERMEDIATE_DIR", str(tmp_path))
    arch = _bare_arch()

    non_serializable = {"obj": object()}

    with pytest.raises(IntermediateSaveError) as exc_info:
        arch._save_intermediate("test_stage_typeerror", non_serializable)

    assert isinstance(exc_info.value.cause, TypeError)
    assert exc_info.value.stage == "test_stage_typeerror"


def test_save_failure_in_identify_contexts_propagates_intermediate_save_error():
    """T-SAVE-4 (Codex C-2): save failure inside identify_contexts must propagate
    IntermediateSaveError, NOT be silently rewrapped as ArchitectExtractionError.
    This is the critical regression for Codex CRITICAL-1.
    """
    from core.orchestration.errors import IntermediateSaveError
    arch = _bare_arch()

    # Mock LLM client to return a valid response containing a contexts list,
    # so identify_contexts reaches the _save_intermediate step.
    from core.llm.base import LLMResponse, TokenUsage
    ok_response = LLMResponse(
        content='{"contexts": ["CoreDomain"]}',
        parsed=None,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        json_failed=False,
    )
    arch.client.chat.return_value = ok_response

    # Force _save_intermediate to raise IntermediateSaveError directly.
    def raising_save(stage, data):  # noqa: ARG001 — `data` kept for signature parity with _save_intermediate
        raise IntermediateSaveError(
            stage=stage,
            filepath="/fake/path.json",
            cause=PermissionError("test"),
            srs_path="<unknown>",
        )

    with patch.object(arch, "_save_intermediate", side_effect=raising_save), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch.token_tracker, "track_api_call"), \
         patch.object(
             arch, "_parse_json_response",
             return_value={"contexts": ["CoreDomain"]},
         ):
        with pytest.raises(IntermediateSaveError):
            arch.identify_contexts(domain_sentences=["one.", "two."])

    # Explicit anti-assertion: must NOT be caught + rewrapped as ArchitectExtractionError.
    # This is enforced by the pytest.raises above; if Codex C-1's fix is missing,
    # the test catches ArchitectExtractionError instead and the pytest.raises fails.


def test_save_failure_in_extract_per_context_details_propagates_intermediate_save_error():
    """T-SAVE-5: save failure at end of Specialist propagates cleanly.

    The Specialist save at line 650 is OUTSIDE the per-context retry loop,
    so no re-raise guard is needed there — this test just confirms the
    clean propagation path.
    """
    from core.orchestration.errors import IntermediateSaveError
    arch = _bare_arch()

    from core.llm.base import LLMResponse, TokenUsage
    ok_response = LLMResponse(
        content=(
            '{"context": "X", "entities": [{"name": "E", "description": "An entity.", '
            '"attributes": [], "confidence": 0.9, "justification": "t", '
            '"evidence_sentence_indices": [0]}], "value_objects": [], "services": [], '
            '"aggregates": [], "domain_events": [], "business_rules": []}'
        ),
        parsed=None,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        json_failed=False,
    )
    arch.client.chat.return_value = ok_response

    def raising_save(stage, data):  # noqa: ARG001 — `data` kept for signature parity with _save_intermediate
        raise IntermediateSaveError(
            stage=stage,
            filepath="/fake/path.json",
            cause=OSError("disk full"),
            srs_path="<unknown>",
        )

    with patch.object(arch, "_save_intermediate", side_effect=raising_save), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch.token_tracker, "track_api_call"), \
         patch.object(
             arch, "_parse_json_response",
             return_value={
                 "context": "X",
                 "entities": [{
                     "name": "E", "description": "An entity.",
                     "attributes": [], "confidence": 0.9,
                     "justification": "t", "evidence_sentence_indices": [0],
                 }],
                 "value_objects": [], "services": [], "aggregates": [],
                 "domain_events": [], "business_rules": [],
             },
         ):
        with pytest.raises(IntermediateSaveError):
            arch.extract_per_context_details(["X"], ["s0", "s1"])
