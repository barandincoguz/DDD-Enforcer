"""WP-CORE-4 — RED-phase tests for `_current_srs_path` propagation (anomaly fold-in).

These tests fail until the GREEN commit lands:
- __init__ initializes self._current_srs_path = "<unknown>"
- analyze_document(text, srs_path: Optional[str] = None) signature
- Unconditional `self._current_srs_path = srs_path or "<unknown>"` at function start (per W-2)

T-SRS-1 — analyze_document(srs_path=...) assigns attribute
T-SRS-2 — analyze_document(srs_path=None) resets to "<unknown>"
T-SRS-3 — ArchitectExtractionError carries assigned path
T-SRS-4 — instance reuse resets path (per Codex W-2)
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from core.architect import DomainArchitect
from core.orchestration.errors import ArchitectExtractionError
from core.token_tracker import TokenTracker


def _bare_arch():
    """Bare DomainArchitect — pre-populates `_current_srs_path` only via the
    new __init__ behavior we're testing. Other attrs set manually."""
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
    # Initialize _current_srs_path to the expected default. The new __init__
    # behavior also does this; we set here because we bypassed __init__.
    a._current_srs_path = "<unknown>"
    a.project_name = "DomainModel"
    return a


def _stub_run_pipeline(*args, **kwargs):
    """Stub for core.orchestration.pipeline.run_pipeline that returns a minimal
    valid DomainModel (Pydantic requires project_metadata + non-empty
    bounded_contexts) so analyze_document can complete without real LLM calls."""
    from core.schemas import (
        BoundedContext, DomainModel, Entity, ProjectMetadata, UbiquitousLanguage,
    )
    return DomainModel(
        project_name="Test",
        project_metadata=ProjectMetadata(version="1.0", generated_at="2026-05-21"),
        bounded_contexts=[
            BoundedContext(
                context_name="Ctx",
                description="d",
                ubiquitous_language=UbiquitousLanguage(
                    entities=[Entity(
                        name="E", description="d", confidence=0.9,
                        justification="t", evidence_sentence_indices=[0],
                    )],
                    value_objects=[], domain_events=[],
                ),
            )
        ],
        global_rules=None,
    )


def test_analyze_document_with_srs_path_assigns_attribute(monkeypatch):
    """T-SRS-1: analyze_document(srs_path="/p/x.docx") sets _current_srs_path."""
    arch = _bare_arch()
    assert arch._current_srs_path == "<unknown>"

    monkeypatch.setattr("core.architect.run_pipeline", _stub_run_pipeline)

    arch.analyze_document(text="some srs text", srs_path="/path/to/srs.docx")

    assert arch._current_srs_path == "/path/to/srs.docx"


def test_analyze_document_without_srs_path_resets_to_unknown(monkeypatch):
    """T-SRS-2: analyze_document() with no srs_path resets to '<unknown>'."""
    arch = _bare_arch()
    # Pre-set a stale value so the test actually verifies reset behavior.
    arch._current_srs_path = "stale/path.docx"

    monkeypatch.setattr("core.architect.run_pipeline", _stub_run_pipeline)

    arch.analyze_document(text="some srs text")

    assert arch._current_srs_path == "<unknown>"


def test_architect_extraction_error_carries_assigned_path():
    """T-SRS-3: ArchitectExtractionError raised inside identify_contexts uses
    the assigned _current_srs_path, not '<unknown>'."""
    arch = _bare_arch()
    arch._current_srs_path = "/p/foo.docx"

    # Force JSON parse to fail 5 times so identify_contexts exhausts retries.
    bad_response = MagicMock()
    bad_response.candidates = [MagicMock()]
    bad_response.candidates[0].finish_reason = "STOP"
    bad_response.text = "not valid json"
    arch.client.chat.return_value = bad_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(
             arch, "_parse_json_response",
             return_value={"error": "json_parse_failed", "raw_response": "..."},
         ):
        with pytest.raises(ArchitectExtractionError) as exc_info:
            arch.identify_contexts(domain_sentences=["one.", "two."])

    assert exc_info.value.srs_path == "/p/foo.docx", \
        f"expected '/p/foo.docx', got {exc_info.value.srs_path!r}"


def test_analyze_document_reuse_resets_path_per_w2(monkeypatch):
    """T-SRS-4 (Codex W-2): instance reuse must reset stale path.

    Without W-2's unconditional assignment, a second call with no srs_path
    would leak the previous run's path. This is the regression the unconditional
    assignment guards against.
    """
    arch = _bare_arch()
    monkeypatch.setattr("core.architect.run_pipeline", _stub_run_pipeline)

    # First call: assigns "/p/A.docx"
    arch.analyze_document(text="...", srs_path="/p/A.docx")
    assert arch._current_srs_path == "/p/A.docx"

    # Second call: no srs_path → must reset to "<unknown>" (not leak "/p/A.docx")
    arch.analyze_document(text="...")
    assert arch._current_srs_path == "<unknown>", \
        f"stale path leak: expected '<unknown>', got {arch._current_srs_path!r}"
