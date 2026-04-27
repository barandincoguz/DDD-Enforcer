"""
Unit Tests for Backend Components

These tests don't require running server or API key.
They test individual components in isolation.

Run with: pytest tests/test_unit.py -v
"""

import pytest
import json
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TOKEN TRACKER TESTS
# =============================================================================

class TestTokenTrackerLegacyMigration:
    """Smoke tests guarding the post-refactor TokenTracker contract.
    Detailed behavior is in tests/test_token_tracker_v2.py.
    """

    def test_no_flash_pricing_module_constants(self):
        """Old module-level pricing dicts must be gone."""
        from core import token_tracker

        for name in ("FLASH_PRICING", "FLASH_LITE_PRICING", "STAGE_MODEL_MAP"):
            assert not hasattr(token_tracker, name), f"Legacy symbol {name!r} still exported"

    def test_tracker_singleton(self):
        from core.token_tracker import TokenTracker

        TokenTracker.reset()
        a = TokenTracker.get_instance()
        b = TokenTracker.get_instance()
        assert a is b
        TokenTracker.reset()
        c = TokenTracker.get_instance()
        assert c is not a

    def test_no_legacy_stats_fields(self):
        """TokenUsageStats no longer carries flash_*_tokens fields."""
        from core.token_tracker import TokenTracker

        TokenTracker.reset()
        tracker = TokenTracker.get_instance()
        for legacy in (
            "flash_prompt_tokens",
            "flash_completion_tokens",
            "flash_lite_prompt_tokens",
            "flash_lite_completion_tokens",
        ):
            assert not hasattr(tracker.stats, legacy), f"Legacy field {legacy!r} still present"


# =============================================================================
# VALIDATION METRICS TRACKER TESTS
# =============================================================================

class TestValidationMetricsTracker:
    """Test ValidationMetricsTracker functionality."""
    
    def test_tracker_singleton(self):
        """Test ValidationMetricsTracker singleton pattern."""
        from core.validation_metrics import ValidationMetricsTracker
        
        tracker1 = ValidationMetricsTracker.get_instance()
        tracker2 = ValidationMetricsTracker.get_instance()
        
        assert tracker1 is tracker2
    
    def test_track_validation(self):
        """Test tracking a validation."""
        from core.validation_metrics import ValidationMetricsTracker
        
        tracker = ValidationMetricsTracker.get_instance()
        initial_count = tracker.stats.total_validations
        
        tracker.track_validation(
            filename="test.py",
            file_size_chars=100,
            code_file_tokens=50,
            validation_time_ms=50.0,
            violations=[
                {"type": "SynonymViolation", "message": "test"},
                {"type": "BannedTermViolation", "message": "test"}
            ],
            has_sources=True
        )
        
        assert tracker.stats.total_validations == initial_count + 1


# =============================================================================
# CODE PARSER TESTS
# =============================================================================

class TestCodeParser:
    """Test CodeParser functionality."""
    
    def test_parse_valid_code(self):
        """Test parsing valid Python code."""
        from core.parser import CodeParser
        
        parser = CodeParser()
        code = '''
class Order:
    def __init__(self, order_id):
        self.order_id = order_id
'''
        
        result = parser.parse_code(code, "order.py")
        
        assert "error" not in result
        assert "classes" in result
        assert len(result["classes"]) == 1
        assert result["classes"][0]["name"] == "Order"
    
    def test_parse_syntax_error(self):
        """Test parsing code with syntax error."""
        from core.parser import CodeParser
        
        parser = CodeParser()
        code = '''
def broken(
    print("missing parenthesis"
'''
        
        result = parser.parse_code(code, "broken.py")
        
        assert "error" in result
    
    def test_parse_functions(self):
        """Test parsing functions."""
        from core.parser import CodeParser
        
        parser = CodeParser()
        code = '''
def process_order(order_id: str) -> dict:
    """Process an order."""
    return {"id": order_id}
'''
        
        result = parser.parse_code(code, "service.py")
        
        assert "functions" in result
        assert len(result["functions"]) == 1
        assert result["functions"][0]["name"] == "process_order"


# =============================================================================
# DETERMINISTIC VALIDATION TESTS
# =============================================================================

class TestDeterministicValidation:
    """Test non-LLM deterministic violation detection."""

    def test_rule_based_name_violations_detects_synonyms_and_banned_terms(self):
        from core.llm_client import LLMClient

        llm = object.__new__(LLMClient)

        ast_data = {
            "filename": "sample.py",
            "classes": [
                {"name": "ClientManager"},
                {"name": "PaymentRecord"},
                {"name": "DataHelper"},
            ],
            "functions": [],
        }
        domain_rules = {
            "bounded_contexts": [
                {
                    "ubiquitous_language": {
                        "entities": [
                            {
                                "name": "Customer",
                                "synonyms_to_avoid": ["Client", "User"],
                            },
                            {
                                "name": "Payment",
                                "synonyms_to_avoid": ["PaymentRecord"],
                            },
                        ]
                    }
                }
            ],
            "global_rules": {
                "banned_global_terms": ["Manager", "Data", "Helper"]
            },
        }

        violations = llm.rule_based_name_violations(ast_data, domain_rules)
        types = [v["type"] for v in violations]

        assert "SynonymViolation" in types
        assert "BannedTermViolation" in types
        assert any("ClientManager" in v["message"] for v in violations)
        assert any("PaymentRecord" in v["message"] for v in violations)
        assert any("DataHelper" in v["message"] for v in violations)


# =============================================================================
# AST MODEL SIGNAL TESTS
# =============================================================================

class TestASTModelSignals:
    """Test AST candidate extraction and model enrichment."""

    def test_extract_candidates_with_traceability(self, tmp_path):
        from core.AST.ast_model_signals import ASTModelSignalExtractor

        sample = tmp_path / "domain_model.py"
        sample.write_text(
            """
from dataclasses import dataclass

class Order:
    def __init__(self, order_id, total):
        self.order_id = order_id
        self.total = total

    def add_item(self, item):
        return item

@dataclass
class Money:
    amount: float
    currency: str

class OrderService:
    def place(self):
        return True

class OrderAggregate:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)
""",
            encoding="utf-8",
        )

        extractor = ASTModelSignalExtractor()
        candidates = extractor.extract_candidates([str(sample)], grounding_docs=[])

        assert any(c["name"] == "Order" for c in candidates["entities"])
        assert any(c["name"] == "Money" for c in candidates["value_objects"])
        assert any(c["name"] == "OrderService" for c in candidates["services"])
        assert any(c["name"] == "OrderAggregate" for c in candidates["aggregates"])

        order_entity = next(c for c in candidates["entities"] if c["name"] == "Order")
        assert 0.0 <= order_entity["confidence"] <= 1.0
        assert len(order_entity["sources"]) >= 1
        assert "file" in order_entity["sources"][0]
        assert "line" in order_entity["sources"][0]
        assert "rule" in order_entity["sources"][0]

    def test_enrich_domain_model_adds_confidence_and_sources(self, tmp_path):
        from core.AST.ast_model_signals import ASTModelSignalExtractor
        from core.schemas import DomainModel, ProjectMetadata

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "billing.py").write_text(
            """
class Invoice:
    def __init__(self, invoice_id, amount):
        self.invoice_id = invoice_id
        self.amount = amount
""",
            encoding="utf-8",
        )

        model = DomainModel(
            project_name="Test",
            project_metadata=ProjectMetadata(
                version="1.0.0",
                generated_at="2026-02-16",
                description="test"
            ),
            bounded_contexts=[],
            global_rules=None,
        )

        extractor = ASTModelSignalExtractor()
        enriched = extractor.enrich_domain_model(
            model,
            str(workspace),
            srs_docs=[{"path": "srs.txt", "content": "Invoice must include amount."}],
        )

        assert len(enriched.bounded_contexts) >= 1
        ul = enriched.bounded_contexts[0].ubiquitous_language
        assert ul.entities is not None
        assert any(e.name == "Invoice" for e in ul.entities)
        invoice = next(e for e in ul.entities if e.name == "Invoice")
        assert invoice.confidence >= 0.5
        assert len(invoice.sources) >= 1


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestSchemas:
    """Test domain model schemas."""
    
    def test_domain_model_structure(self):
        """Test DomainModel schema structure."""
        from core.schemas import DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage
        
        # Test with complete required data
        model = DomainModel(
            project_name="Test Project",
            project_metadata=ProjectMetadata(
                version="1.0.0",
                generated_at="2026-01-21T12:00:00"
            ),
            bounded_contexts=[],
            global_rules=None
        )
        
        model_dict = model.model_dump()
        
        assert "project_name" in model_dict
        assert "project_metadata" in model_dict
        assert "bounded_contexts" in model_dict
    
    def test_bounded_context_schema(self):
        """Test BoundedContext schema."""
        from core.schemas import BoundedContext, UbiquitousLanguage, Entity
        
        context = BoundedContext(
            context_name="Order Processing",
            description="Handles order lifecycle",
            ubiquitous_language=UbiquitousLanguage(
                entities=[
                    Entity(name="Order", description="A purchase request"),
                    Entity(name="OrderItem", description="A line item in an order")
                ],
                value_objects=[],
                domain_events=[]
            )
        )
        
        assert context.context_name == "Order Processing"
        assert len(context.ubiquitous_language.entities) == 2


# =============================================================================
# DOCUMENT PARSER TESTS
# =============================================================================

class TestDocumentParser:
    """Test SRSDocumentParser functionality."""
    
    def test_parse_txt_file(self, tmp_path):
        """Test parsing a .txt file."""
        from core.document_parser import SRSDocumentParser
        
        # Create temp file
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("# Test SRS\n\nThis is a test document.")
        
        parser = SRSDocumentParser()
        content = parser.parse_file(str(txt_file))
        
        assert "Test SRS" in content
        assert "test document" in content
    
    def test_parse_nonexistent_file(self):
        """Test parsing a non-existent file raises error."""
        from core.document_parser import SRSDocumentParser
        
        parser = SRSDocumentParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.txt")


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Test configuration values."""
    
    def test_model_names(self):
        """Test model name configurations."""
        from config import AnalyzerConfig, ArchitectConfig
        
        # Validation uses flash-lite
        assert AnalyzerConfig.MODEL_NAME == "gemini-2.5-flash-lite"
        
        # Domain model generation uses flash
        assert ArchitectConfig.MODEL_NAME == "gemini-2.5-flash"
    
    def test_server_config(self):
        """Test server configuration."""
        from config import ServerConfig
        
        assert ServerConfig.HOST == "127.0.0.1"
        assert ServerConfig.PORT == 8000
    
    def test_rag_config(self):
        """Test RAG configuration."""
        from config import RAGConfig
        
        config = RAGConfig()
        
        assert config.CHUNK_SIZE > 0
        assert config.TOP_K > 0
        assert 0 <= config.MIN_RELEVANCE_SCORE <= 1


# =============================================================================
# FASTAPI APP TESTS (No server required)
# =============================================================================

class TestFastAPIRoutes:
    """Test FastAPI route definitions."""
    
    def test_app_has_health_endpoint(self):
        """Test /health endpoint exists."""
        # Mock the genai import to avoid API key check
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            # Import with mocked client
            with patch("core.llm_client.genai"):
                with patch("core.architect.genai"):
                    from main import app
                    
                    routes = [route.path for route in app.routes]
                    assert "/health" in routes
    
    def test_app_has_validate_endpoint(self):
        """Test /validate endpoint exists."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            with patch("core.llm_client.genai"):
                with patch("core.architect.genai"):
                    from main import app
                    
                    routes = [route.path for route in app.routes]
                    assert "/validate" in routes
    
    def test_app_has_token_endpoints(self):
        """Test token tracking endpoints exist."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            with patch("core.llm_client.genai"):
                with patch("core.architect.genai"):
                    from main import app
                    
                    routes = [route.path for route in app.routes]
                    assert "/tokens/stats" in routes
                    assert "/tokens/summary" in routes
                    assert "/tokens/reset" in routes
    
    def test_app_has_metrics_endpoints(self):
        """Test metrics endpoints exist."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):
            with patch("core.llm_client.genai"):
                with patch("core.architect.genai"):
                    from main import app
                    
                    routes = [route.path for route in app.routes]
                    assert "/metrics/validation" in routes
                    assert "/metrics/combined" in routes


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
