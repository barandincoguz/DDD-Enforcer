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

class TestTokenTracker:
    """Test TokenTracker functionality."""
    
    def test_pricing_constants(self):
        """Test pricing constants are correctly set."""
        from core.token_tracker import FLASH_PRICING, FLASH_LITE_PRICING
        
        # Flash pricing (gemini-2.5-flash)
        assert FLASH_PRICING["input"] == 0.30 / 1_000_000
        assert FLASH_PRICING["output"] == 2.50 / 1_000_000
        
        # Flash-Lite pricing (gemini-2.5-flash-lite)
        assert FLASH_LITE_PRICING["input"] == 0.10 / 1_000_000
        assert FLASH_LITE_PRICING["output"] == 0.40 / 1_000_000
    
    def test_stage_model_mapping(self):
        """Test stage to model mapping."""
        from core.token_tracker import STAGE_MODEL_MAP
        
        # Domain model stages use flash
        assert STAGE_MODEL_MAP["Scout"] == "flash"
        assert STAGE_MODEL_MAP["Architect"] == "flash"
        assert STAGE_MODEL_MAP["Specialist"] == "flash"
        assert STAGE_MODEL_MAP["Synthesizer"] == "flash"
        
        # Validation stage uses flash-lite
        assert STAGE_MODEL_MAP["Validator"] == "flash-lite"
    
    def test_tracker_singleton(self):
        """Test TokenTracker singleton pattern."""
        from core.token_tracker import TokenTracker
        
        # Reset first
        TokenTracker.reset()
        
        tracker1 = TokenTracker.get_instance()
        tracker2 = TokenTracker.get_instance()
        
        assert tracker1 is tracker2
    
    def test_calculate_call_cost_flash(self):
        """Test cost calculation for flash model."""
        from core.token_tracker import TokenTracker, FLASH_PRICING
        
        TokenTracker.reset()
        tracker = TokenTracker.get_instance()
        
        prompt_tokens = 1000
        completion_tokens = 500
        
        cost = tracker._calculate_call_cost("flash", prompt_tokens, completion_tokens)
        
        expected = (prompt_tokens * FLASH_PRICING["input"] + 
                   completion_tokens * FLASH_PRICING["output"])
        
        assert abs(cost - expected) < 0.000001
    
    def test_calculate_call_cost_flash_lite(self):
        """Test cost calculation for flash-lite model."""
        from core.token_tracker import TokenTracker, FLASH_LITE_PRICING
        
        TokenTracker.reset()
        tracker = TokenTracker.get_instance()
        
        prompt_tokens = 1000
        completion_tokens = 500
        
        cost = tracker._calculate_call_cost("flash-lite", prompt_tokens, completion_tokens)
        
        expected = (prompt_tokens * FLASH_LITE_PRICING["input"] + 
                   completion_tokens * FLASH_LITE_PRICING["output"])
        
        assert abs(cost - expected) < 0.000001
    
    def test_get_model_for_stage(self):
        """Test model selection based on stage."""
        from core.token_tracker import TokenTracker
        
        TokenTracker.reset()
        tracker = TokenTracker.get_instance()
        
        assert tracker._get_model_for_stage("Scout") == "flash"
        assert tracker._get_model_for_stage("Validator") == "flash-lite"
        assert tracker._get_model_for_stage("Unknown") == "flash-lite"  # default


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
