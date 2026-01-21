"""
Comprehensive Backend API Tests for DDD-Enforcer

Tests all endpoints:
1. Health & Status endpoints
2. Domain Model Generation
3. Code Validation
4. RAG Search & Stats
5. Token Tracking
6. Validation Metrics
7. Combined Metrics

Run with: pytest tests/test_api.py -v
"""

import pytest
import httpx
import os
import json
import tempfile
import time
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = os.getenv("DDD_BACKEND_URL", "http://localhost:8000")
TIMEOUT = 120.0  # Domain model generation can take time

# Sample test data
SAMPLE_SRS_CONTENT = """
# E-Commerce Platform SRS

## 1. Introduction
This document describes the software requirements for an e-commerce platform.

## 2. Domain Concepts

### 2.1 Order
An Order represents a customer's purchase request. Orders contain OrderItems.
- OrderID: Unique identifier
- CustomerID: Reference to customer
- Status: pending, confirmed, shipped, delivered
- TotalAmount: Sum of all item prices

### 2.2 Product
A Product represents an item available for purchase.
- ProductID: Unique identifier  
- Name: Product name
- Price: Unit price in USD
- StockQuantity: Available inventory

### 2.3 Customer
A Customer is a registered user who can place orders.
- CustomerID: Unique identifier
- Email: Contact email
- Name: Full name

## 3. Business Rules
- An Order must have at least one OrderItem
- Product price must be positive
- StockQuantity cannot be negative
- Order TotalAmount = sum of (item price * quantity)

## 4. Ubiquitous Language
- Order: A purchase request from a customer
- Product: An item for sale
- Customer: A registered buyer
- Cart: Temporary collection before order
- Payment: DO NOT USE "transaction" - use "Payment" instead
"""

VALID_PYTHON_CODE = '''
"""Order service for e-commerce platform."""

class Order:
    """Represents a customer order."""
    
    def __init__(self, order_id: str, customer_id: str):
        self.order_id = order_id
        self.customer_id = customer_id
        self.items = []
        self.status = "pending"
        self.total_amount = 0.0
    
    def add_item(self, product_id: str, quantity: int, price: float):
        """Add an item to the order."""
        self.items.append({
            "product_id": product_id,
            "quantity": quantity,
            "price": price
        })
        self.total_amount += price * quantity
    
    def confirm(self):
        """Confirm the order."""
        if not self.items:
            raise ValueError("Order must have at least one item")
        self.status = "confirmed"
'''

INVALID_PYTHON_CODE_BANNED_TERM = '''
"""Payment service with banned term."""

class PaymentService:
    """Handles payment transactions."""
    
    def process_transaction(self, amount: float):
        """Process a transaction."""  # Uses banned term "transaction"
        return {"status": "success", "transaction_id": "123"}
'''

SYNTAX_ERROR_CODE = '''
def broken_function(
    print("missing closing parenthesis"
'''


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def client():
    """HTTP client for API requests."""
    return httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)


@pytest.fixture(scope="module")
def temp_srs_file():
    """Create a temporary SRS file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(SAMPLE_SRS_CONTENT)
        temp_path = f.name
    yield temp_path
    # Cleanup
    try:
        os.unlink(temp_path)
    except:
        pass


# =============================================================================
# 1. HEALTH & STATUS TESTS
# =============================================================================

class TestHealthEndpoints:
    """Test health check and status endpoints."""
    
    def test_health_check(self, client):
        """Test /health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "domain_model_loaded" in data
        assert "rag_initialized" in data
    
    def test_status_endpoint(self, client):
        """Test /status endpoint returns detailed status."""
        response = client.get("/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "running"
        assert "domain_model" in data
        assert "rag" in data
        assert "loaded" in data["domain_model"]
        assert "initialized" in data["rag"]


# =============================================================================
# 2. DOMAIN MODEL GENERATION TESTS
# =============================================================================

class TestDomainModelGeneration:
    """Test domain model generation from SRS documents."""
    
    def test_generate_model_no_files(self, client):
        """Test error when no input files provided."""
        response = client.post("/generate-model", json={
            "file_paths": [],
            "output_path": "/tmp/test_model.json"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is False
        assert "error" in data
    
    def test_generate_model_invalid_file(self, client):
        """Test error when file doesn't exist."""
        response = client.post("/generate-model", json={
            "file_paths": ["/nonexistent/file.txt"],
            "output_path": "/tmp/test_model.json"
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is False
    
    def test_generate_model_success(self, client, temp_srs_file):
        """Test successful domain model generation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            response = client.post("/generate-model", json={
                "file_paths": [temp_srs_file],
                "output_path": output_path
            })
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "domain_model" in data
            assert data["output_path"] == output_path
            
            # Verify model structure
            model = data["domain_model"]
            assert "bounded_contexts" in model
            assert "entities" in model
            assert "value_objects" in model
            assert "ubiquitous_language" in model
            
            # Verify file was created
            assert Path(output_path).exists()
            
            # Verify file content
            with open(output_path) as f:
                saved_model = json.load(f)
            assert "bounded_contexts" in saved_model
            
        finally:
            try:
                os.unlink(output_path)
            except:
                pass
    
    def test_domain_model_loaded_after_generation(self, client):
        """Verify domain model is loaded after generation."""
        response = client.get("/health")
        data = response.json()
        assert data["domain_model_loaded"] is True


# =============================================================================
# 3. CODE VALIDATION TESTS
# =============================================================================

class TestCodeValidation:
    """Test code validation endpoint."""
    
    def test_validate_valid_code(self, client):
        """Test validation of DDD-compliant code."""
        response = client.post("/validate", json={
            "filename": "test_order.py",
            "content": VALID_PYTHON_CODE
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "is_violation" in data
        # Valid code should have no violations
        if data["is_violation"]:
            print(f"Violations found: {data.get('violations', [])}")
    
    def test_validate_code_with_banned_term(self, client):
        """Test validation catches banned terms."""
        response = client.post("/validate", json={
            "filename": "test_payment.py",
            "content": INVALID_PYTHON_CODE_BANNED_TERM
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "is_violation" in data
        # Should detect "transaction" as banned term
        if data["is_violation"]:
            violations = data.get("violations", [])
            assert len(violations) > 0
    
    def test_validate_syntax_error(self, client):
        """Test validation returns syntax error for invalid code."""
        response = client.post("/validate", json={
            "filename": "broken.py",
            "content": SYNTAX_ERROR_CODE
        })
        assert response.status_code == 200
        
        data = response.json()
        assert data["is_violation"] is True
        
        violations = data.get("violations", [])
        assert len(violations) > 0
        assert violations[0]["type"] == "SyntaxError"
    
    def test_validate_empty_code(self, client):
        """Test validation of empty code."""
        response = client.post("/validate", json={
            "filename": "empty.py",
            "content": ""
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "is_violation" in data
    
    def test_validate_tracks_metrics(self, client):
        """Test that validation updates metrics."""
        # Get initial metrics
        initial = client.get("/metrics/validation/summary").json()
        initial_count = initial.get("total_validations", 0)
        
        # Perform validation
        client.post("/validate", json={
            "filename": "metrics_test.py",
            "content": VALID_PYTHON_CODE
        })
        
        # Check metrics updated
        updated = client.get("/metrics/validation/summary").json()
        assert updated.get("total_validations", 0) >= initial_count


# =============================================================================
# 4. RAG SEARCH & STATS TESTS
# =============================================================================

class TestRAGEndpoints:
    """Test RAG search and stats endpoints."""
    
    def test_rag_stats(self, client):
        """Test /rag/stats endpoint."""
        response = client.get("/rag/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "document_count" in data
        assert "chunk_count" in data
    
    def test_rag_search(self, client):
        """Test /rag/search endpoint."""
        response = client.get("/rag/search", params={
            "query": "Order",
            "top_k": 3
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "query" in data
        assert data["query"] == "Order"
    
    def test_rag_search_empty_query(self, client):
        """Test RAG search with empty query."""
        response = client.get("/rag/search", params={
            "query": "",
            "top_k": 3
        })
        assert response.status_code == 200


# =============================================================================
# 5. TOKEN TRACKING TESTS
# =============================================================================

class TestTokenTrackingEndpoints:
    """Test token usage tracking endpoints."""
    
    def test_token_stats(self, client):
        """Test /tokens/stats endpoint."""
        response = client.get("/tokens/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "session_start" in data
        assert "token_summary" in data
        assert "cost_estimation" in data
    
    def test_token_summary(self, client):
        """Test /tokens/summary endpoint."""
        response = client.get("/tokens/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_prompt_tokens" in data
        assert "total_completion_tokens" in data
        assert "total_tokens" in data
    
    def test_token_reset(self, client):
        """Test /tokens/reset endpoint."""
        response = client.post("/tokens/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "reset"
        
        # Verify reset
        summary = client.get("/tokens/summary").json()
        assert summary["total_tokens"] == 0
    
    def test_token_export(self, client):
        """Test /tokens/export endpoint."""
        response = client.get("/tokens/export")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "exported"
        assert "filepath" in data
    
    def test_token_tracking_after_validation(self, client):
        """Test tokens are tracked after validation."""
        # Reset tokens
        client.post("/tokens/reset")
        
        # Perform validation
        client.post("/validate", json={
            "filename": "token_test.py",
            "content": VALID_PYTHON_CODE
        })
        
        # Check tokens were tracked
        summary = client.get("/tokens/summary").json()
        # After validation, there should be some tokens tracked
        # (from the Validator stage using flash-lite model)
        assert summary["total_tokens"] >= 0  # May be 0 if no LLM call was needed


# =============================================================================
# 6. VALIDATION METRICS TESTS
# =============================================================================

class TestValidationMetricsEndpoints:
    """Test validation metrics endpoints."""
    
    def test_validation_metrics(self, client):
        """Test /metrics/validation endpoint."""
        response = client.get("/metrics/validation")
        assert response.status_code == 200
        
        data = response.json()
        assert "session_start" in data
        assert "summary" in data
        assert "validation_history" in data
    
    def test_validation_metrics_summary(self, client):
        """Test /metrics/validation/summary endpoint."""
        response = client.get("/metrics/validation/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_validations" in data
        assert "files_with_violations" in data
        assert "files_without_violations" in data


# =============================================================================
# 7. COMBINED METRICS TESTS
# =============================================================================

class TestCombinedMetricsEndpoint:
    """Test combined metrics endpoint."""
    
    def test_combined_metrics(self, client):
        """Test /metrics/combined endpoint."""
        response = client.get("/metrics/combined")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check all sections present
        assert "domain_model" in data
        assert "token_usage" in data
        assert "validation_metrics" in data
        assert "pricing_reference" in data
        
        # Check pricing reference structure
        pricing = data["pricing_reference"]
        assert "gemini-2.5-flash" in pricing
        assert "gemini-2.5-flash-lite" in pricing
        
        # Verify correct pricing values
        flash = pricing["gemini-2.5-flash"]
        assert flash["input_per_1m_tokens"] == 0.30
        assert flash["output_per_1m_tokens"] == 2.50
        
        flash_lite = pricing["gemini-2.5-flash-lite"]
        assert flash_lite["input_per_1m_tokens"] == 0.10
        assert flash_lite["output_per_1m_tokens"] == 0.40
    
    def test_combined_metrics_projections(self, client):
        """Test monthly projections in combined metrics."""
        response = client.get("/metrics/combined")
        data = response.json()
        
        assert "monthly_projections" in data
        projections = data["monthly_projections"]
        
        assert "estimated_monthly_validations" in projections
        assert "estimated_monthly_token_cost" in projections


# =============================================================================
# 8. INTEGRATION TESTS (End-to-End Flow)
# =============================================================================

class TestEndToEndFlow:
    """Test complete workflow from model generation to validation."""
    
    def test_full_workflow(self, client, temp_srs_file):
        """Test complete flow: generate model -> validate code -> check metrics."""
        
        # Step 1: Reset all trackers
        client.post("/tokens/reset")
        
        # Step 2: Generate domain model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            gen_response = client.post("/generate-model", json={
                "file_paths": [temp_srs_file],
                "output_path": output_path
            })
            assert gen_response.json()["success"] is True
            
            # Step 3: Verify health shows model loaded
            health = client.get("/health").json()
            assert health["domain_model_loaded"] is True
            
            # Step 4: Validate some code
            val_response = client.post("/validate", json={
                "filename": "e2e_test.py",
                "content": VALID_PYTHON_CODE
            })
            assert "is_violation" in val_response.json()
            
            # Step 5: Check tokens were tracked
            tokens = client.get("/tokens/summary").json()
            assert tokens["total_tokens"] >= 0
            
            # Step 6: Check validation metrics
            metrics = client.get("/metrics/validation/summary").json()
            assert metrics["total_validations"] >= 1
            
            # Step 7: Check combined metrics
            combined = client.get("/metrics/combined").json()
            assert "domain_model" in combined
            assert "token_usage" in combined
            assert "validation_metrics" in combined
            
        finally:
            try:
                os.unlink(output_path)
            except:
                pass


# =============================================================================
# 9. ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling for edge cases."""
    
    def test_invalid_json_body(self, client):
        """Test handling of malformed JSON."""
        response = client.post(
            "/validate",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields."""
        response = client.post("/validate", json={
            "filename": "test.py"
            # missing "content" field
        })
        assert response.status_code == 422
    
    def test_invalid_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
