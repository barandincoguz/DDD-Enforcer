"""C7: analyze_document delegates to core.orchestration.pipeline.run_pipeline."""

from unittest.mock import patch, MagicMock
from core.architect import DomainArchitect
from core.schemas import (
    DomainModel, BoundedContext, UbiquitousLanguage,
    Entity, ProjectMetadata,
)


def test_analyze_document_calls_run_pipeline():
    arch = DomainArchitect.__new__(DomainArchitect)
    arch.model_name = "gemini-3.1-pro-preview"
    arch.last_request_time = 0
    arch.min_delay = 0
    arch.request_count = 0
    import threading
    arch._rate_limit_lock = threading.Lock()
    from core.token_tracker import TokenTracker
    arch.token_tracker = TokenTracker.get_instance()
    arch.progress_callback = None
    arch.run_timestamp = "20260518_000000"
    arch.client = MagicMock()
    arch.scout_max_workers = 1

    fake_model = DomainModel(
        project_name="X",
        project_metadata=ProjectMetadata(version="1.0", generated_at="2026-05-18"),
        bounded_contexts=[BoundedContext(
            context_name="OrderMgmt",
            description="manages",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="d", confidence=0.9,
                    justification="t", evidence_sentence_indices=[0],
                )],
                value_objects=[], domain_events=[]
            ),
        )],
        global_rules=None,
    )

    with patch("core.architect.run_pipeline", return_value=fake_model) as mock_run:
        result = arch.analyze_document(text="sample SRS text")
        assert result.project_name == "X"
        mock_run.assert_called_once()
