"""Wiring tests for WP-CORE-3 empty-input contract.

Verifies the `_parse_srs_batch` helper and `initialize_rag` SOFT path
without touching real FastAPI, real LLMs, or real ChromaDB.
"""

from core.document_parser import EmptySRSDocumentError, SRSDocumentParser
from main import _parse_srs_batch


class _StubParser(SRSDocumentParser):
    """SRSDocumentParser-shaped stub for testing batch helper.

    Subclasses SRSDocumentParser so the helper's typed `parser:
    SRSDocumentParser` signature accepts the stub (substitutability).
    """

    def __init__(self, behaviors):
        super().__init__()
        self.behaviors = behaviors
        self.calls = []

    def parse_file(self, file_path: str) -> str:
        self.calls.append(file_path)
        behavior = self.behaviors[file_path]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def test_parse_srs_batch_skips_empty_file_and_continues_with_non_empty():
    parser = _StubParser(
        {
            "empty.txt": EmptySRSDocumentError("empty.txt parsed to empty content"),
            "good.txt": "good content here",
        }
    )
    combined, docs, err = _parse_srs_batch(parser, ["empty.txt", "good.txt"])
    assert err is None
    assert len(docs) == 1
    assert docs[0]["path"] == "good.txt"
    assert docs[0]["content"] == "good content here"
    assert "good content here" in combined
    assert parser.calls == ["empty.txt", "good.txt"]


def test_parse_srs_batch_returns_aggregate_error_when_all_files_empty():
    parser = _StubParser(
        {
            "a.txt": EmptySRSDocumentError("a.txt empty"),
            "b.txt": EmptySRSDocumentError("b.txt empty"),
        }
    )
    combined, docs, err = _parse_srs_batch(parser, ["a.txt", "b.txt"])
    assert err is not None
    assert err["success"] is False
    assert "All documents were empty" in err["error"]
    assert docs == []
    assert combined == ""


def test_parse_srs_batch_returns_per_file_error_for_non_empty_parse_failure():
    parser = _StubParser({"missing.txt": FileNotFoundError("missing.txt")})
    combined, docs, err = _parse_srs_batch(parser, ["missing.txt"])
    assert err is not None
    assert err["success"] is False
    assert "Failed to parse missing.txt" in err["error"]
    assert docs == []


def test_initialize_rag_silently_returns_empty_rag_on_empty_srs(monkeypatch):
    import main

    def fake_parse_file(self, path):
        raise EmptySRSDocumentError(path)

    class FakeRAG:
        def __init__(self):
            self.indexed_calls = []

        def index_document(self, **kwargs):
            self.indexed_calls.append(kwargs)
            return 0

    monkeypatch.setattr(
        "core.document_parser.SRSDocumentParser.parse_file", fake_parse_file
    )
    monkeypatch.setattr("main.RAGPipeline", FakeRAG)

    rag = main.initialize_rag(["/tmp/empty.txt"])
    assert isinstance(rag, FakeRAG)
    assert rag.indexed_calls == []


# =============================================================================
# WP-CORE-4 — endpoint wiring tests (T-WIRE-MAIN-2, T-WIRE-MAIN-3)
#
# Per Codex OQ5: both /generate-model and /generate-model-stream callsites
# must be tested for srs_path forwarding. Lightweight monkeypatch — no FastAPI
# TestClient required.
# =============================================================================


def _stub_domain_model():
    """Minimal DomainModel for test stubs to return."""
    from core.schemas import DomainModel, GlobalRules
    return DomainModel(
        project_name="TestProject",
        bounded_contexts=[],
        global_rules=GlobalRules(),
    )


class _CapturingArchitectFactory:
    """Factory that, when called, produces an architect stub whose
    `analyze_document(text, srs_path)` records both kwargs into the
    `captured` dict for assertion. Pattern adapted from existing
    _StubParser above (Liskov-substitutable for pyright)."""

    def __init__(self, captured):
        self.captured = captured

    def __call__(self, progress_callback=None):
        captured = self.captured
        stub = _CapturingArchitectFactory._Stub(captured)
        return stub

    class _Stub:
        def __init__(self, captured):
            self._captured = captured

        def analyze_document(self, text, srs_path=None):
            self._captured["text"] = text
            self._captured["srs_path"] = srs_path
            return _stub_domain_model()


def test_generate_model_endpoint_forwards_joined_srs_path(monkeypatch, tmp_path):
    """T-WIRE-MAIN-2 (Codex OQ5): /generate-model passes "; "-joined paths."""
    import main

    captured = {}
    monkeypatch.setattr("main.DomainArchitect", _CapturingArchitectFactory(captured))

    # Stub parser returns non-empty text per path
    def stub_parse(self, path):
        return f"content of {path}"
    monkeypatch.setattr(
        "core.document_parser.SRSDocumentParser.parse_file", stub_parse
    )

    # No workspace → skip AST enrichment
    monkeypatch.setenv("WORKSPACE_PATH", "")

    # No-op RAG
    class FakeRAG:
        def index_document(self, **kwargs):
            return 0
    monkeypatch.setattr("main.RAGPipeline", FakeRAG)

    request = main.GenerateModelRequest(
        file_paths=["/tmp/a.docx", "/tmp/b.docx"],
        output_path=str(tmp_path / "model.json"),
    )
    response = main.generate_model_endpoint(request)

    assert response.get("success") is True, f"endpoint returned: {response}"
    assert captured.get("srs_path") == "/tmp/a.docx; /tmp/b.docx", \
        f"expected joined path, got {captured.get('srs_path')!r}"


def test_generate_model_stream_endpoint_forwards_joined_srs_path(monkeypatch, tmp_path):
    """T-WIRE-MAIN-3 (Codex OQ5): /generate-model-stream passes "; "-joined paths.

    The endpoint spawns a worker thread. The StreamingResponse generator starts
    the thread only when iterated; we drain the generator to drive it to
    completion, then check the captured kwargs.
    """
    import main

    captured = {}
    monkeypatch.setattr("main.DomainArchitect", _CapturingArchitectFactory(captured))

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
        file_paths=["/tmp/a.docx", "/tmp/b.docx"],
        output_path=str(tmp_path / "model.json"),
    )
    response = main.generate_model_stream_endpoint(request)

    # Drain the streaming generator — this also starts the worker thread.
    events = list(response.body_iterator)
    assert events, "stream should have yielded at least one event"

    # By the time the generator finishes, the worker thread has joined.
    assert captured.get("srs_path") == "/tmp/a.docx; /tmp/b.docx", \
        f"expected joined path, got {captured.get('srs_path')!r}"
