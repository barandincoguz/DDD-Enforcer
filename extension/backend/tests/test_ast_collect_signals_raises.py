"""Phase B5 / G01: _collect_signals must raise on a malformed source file
instead of silently logging and dropping the file from the candidate set.

The test creates a file with bytes that are not valid UTF-8 / not valid
Python, and asserts that the public surface of ASTModelSignalExtractor
which internally calls _collect_signals raises (SyntaxError, UnicodeDecodeError,
ValueError, or OSError) rather than swallowing.
"""

import pytest
from core.AST.ast_model_signals import ASTModelSignalExtractor
from core.schemas import BoundedContext, DomainModel, ProjectMetadata, UbiquitousLanguage


def _build_model():
    """Create a minimal valid DomainModel for testing."""
    return DomainModel(
        project_name="Test",
        project_metadata=ProjectMetadata(
            version="1.0.0",
            generated_at="2026-03-13",
            description="test",
        ),
        bounded_contexts=[
            BoundedContext(
                context_name="CoreDomain",
                description="CoreDomain responsibilities",
                allowed_dependencies=[],
                ubiquitous_language=UbiquitousLanguage(
                    entities=[],
                    value_objects=[],
                    services=[],
                    aggregates=[],
                    domain_events=[],
                ),
            )
        ],
        global_rules=None,
    )


def test_collect_signals_raises_on_malformed_utf8_file(tmp_path):
    """Test that enrich_domain_model raises when encountering invalid UTF-8 in a Python file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bad = workspace / "broken.py"
    # Write invalid UTF-8 bytes
    bad.write_bytes(b"\xff\xfe\x00\x00not valid utf-8\x00")

    extractor = ASTModelSignalExtractor()
    model = _build_model()

    # Should raise, not silently return an empty model
    with pytest.raises((SyntaxError, UnicodeDecodeError, ValueError, OSError)):
        extractor.enrich_domain_model(
            model=model,
            workspace_path=str(workspace),
        )


def test_collect_signals_raises_on_syntax_error_file(tmp_path):
    """Test that enrich_domain_model raises when encountering invalid Python syntax."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bad = workspace / "syntax_error.py"
    # Write valid UTF-8 but invalid Python syntax
    bad.write_text("class Broken\n  pass\n", encoding="utf-8")

    extractor = ASTModelSignalExtractor()
    model = _build_model()

    # Should raise, not silently return an empty model
    with pytest.raises((SyntaxError, UnicodeDecodeError, ValueError, OSError)):
        extractor.enrich_domain_model(
            model=model,
            workspace_path=str(workspace),
        )
