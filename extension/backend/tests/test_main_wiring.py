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
