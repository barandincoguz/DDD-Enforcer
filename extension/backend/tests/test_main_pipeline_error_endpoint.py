"""WP-CORE-8 — /generate-model + /generate-model-stream typed PipelineError tests.

T-ENDPOINT-1: /generate-model returns dict with error_type + typed fields
              when DomainArchitect.analyze_document raises ArchitectGroundingError.
T-SSE-1 (Codex W-2 NEW): /generate-model-stream emits SSE payload with
        error (string, VSCode-compat) + error_type + srs_path as siblings.

Pattern: direct endpoint-function call per existing test_main_wiring.py:177-182
convention; no FastAPI TestClient required (Codex N-1 confirmed).
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _raising_architect_factory(exc):
    """Build a DomainArchitect factory whose analyze_document raises exc."""

    class _RaiseArchitect:
        def __init__(self, progress_callback=None):
            pass

        def analyze_document(self, text, srs_path=None):
            raise exc

    def factory(progress_callback=None):
        return _RaiseArchitect(progress_callback=progress_callback)

    return factory


# =============================================================================
# T-ENDPOINT-1 — /generate-model returns typed payload
# =============================================================================


def test_generate_model_endpoint_returns_typed_error_on_pipeline_error(monkeypatch, tmp_path):
    """T-ENDPOINT-1: mocks DomainArchitect to raise ArchitectGroundingError;
    call /generate-model endpoint directly; response contains error_type +
    srs_path + issues (typed signal preserved beyond generic str(e))."""
    import main
    from core.orchestration.errors import ArchitectGroundingError

    arch_issue = {
        "check_id": "D1",
        "target": "architect:contexts[OrderMgmt].supporting_sentence_ids",
        "message": "Context 'OrderMgmt' has no supporting_sentence_ids",
        "severity": "ERROR",
    }
    exc = ArchitectGroundingError(
        srs_path="inputs/D1.docx",
        issues=[arch_issue],
        residual_issues=[],
        cycles_attempted=1,
    )
    monkeypatch.setattr("main.DomainArchitect", _raising_architect_factory(exc))

    def stub_parse(self, path):
        return f"content of {path}"
    monkeypatch.setattr(
        "core.document_parser.SRSDocumentParser.parse_file", stub_parse
    )
    monkeypatch.setenv("WORKSPACE_PATH", "")

    class FakeRAG:
        def index_document(self, **kwargs):
            return 0
    monkeypatch.setattr("main.RAGPipeline", FakeRAG)

    request = main.GenerateModelRequest(
        file_paths=["/tmp/D1.docx"],
        output_path=str(tmp_path / "model.json"),
    )
    response = main.generate_model_endpoint(request)

    assert response["success"] is False
    assert response["error_type"] == "ArchitectGroundingError"
    assert response["srs_path"] == "inputs/D1.docx"
    assert response["cycles_attempted"] == 1
    assert isinstance(response["issues"], list) and len(response["issues"]) == 1
    # Legacy `error` field still present (VSCode compat)
    assert isinstance(response["error"], str)
    assert "ArchitectGroundingError" in response.get("error", "") or \
           "D1.docx" in response.get("error", "")


# =============================================================================
# T-SSE-1 (Codex W-2 NEW) — /generate-model-stream SSE typed payload
# =============================================================================


def test_generate_model_stream_emits_typed_error_in_sse_payload(monkeypatch, tmp_path):
    """T-SSE-1 (Codex W-2): mock DomainArchitect to raise; drain SSE body;
    parse final `data:` line; verify event.error is string (NOT dict, keeps
    VSCode extension compat at extension.ts:683); error_type + srs_path
    are sibling top-level fields."""
    import main
    from core.orchestration.errors import ArchitectGroundingError

    exc = ArchitectGroundingError(
        srs_path="inputs/D1.docx",
        issues=[{"check_id": "D1", "target": "architect:contexts[X]", "message": "no IDs"}],
        residual_issues=[],
        cycles_attempted=1,
    )
    monkeypatch.setattr("main.DomainArchitect", _raising_architect_factory(exc))

    def stub_parse(self, path):
        return f"content of {path}"
    monkeypatch.setattr(
        "core.document_parser.SRSDocumentParser.parse_file", stub_parse
    )
    monkeypatch.setenv("WORKSPACE_PATH", "")

    class FakeRAG:
        def index_document(self, **kwargs):
            return 0
    monkeypatch.setattr("main.RAGPipeline", FakeRAG)

    request = main.GenerateModelRequest(
        file_paths=["/tmp/D1.docx"],
        output_path=str(tmp_path / "model.json"),
    )
    response = main.generate_model_stream_endpoint(request)

    # Drain the StreamingResponse body_iterator (async).
    async def _drain(it):
        items = []
        async for item in it:
            items.append(item)
        return items

    events_raw = asyncio.run(_drain(response.body_iterator))

    # Find the final SSE line that contains an 'error' event payload.
    error_payload = None
    for raw in events_raw:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        for line in text.split("\n"):
            if line.startswith("data: "):
                try:
                    parsed = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue
                if parsed.get("type") == "error":
                    error_payload = parsed

    assert error_payload is not None, f"no SSE error event found in: {events_raw!r}"
    # event.error must remain a STRING for VSCode extension compat.
    assert isinstance(error_payload["error"], str), (
        f"event.error must stay string for VSCode extension compat; "
        f"got {type(error_payload['error'])}"
    )
    # Typed sibling fields populated.
    assert error_payload["error_type"] == "ArchitectGroundingError"
    assert error_payload["srs_path"] == "inputs/D1.docx"
    assert error_payload["cycles_attempted"] == 1
