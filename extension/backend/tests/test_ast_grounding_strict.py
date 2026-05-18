"""Phase D3 / OQ2: AST enrichment must not fabricate
InferenceSource(file='generated', rule='LLM_SYNTHESIS'). When an
entity has no SRS evidence and no AST grounding, raise
InsufficientGroundingError instead.
"""

import pytest
from core.orchestration.errors import InsufficientGroundingError
from core.AST.ast_signal_enrichment import _ensure_traceability


def test_ensure_traceability_raises_on_no_evidence():
    entity_dict = {
        "name": "PhantomEntity",
        "description": "no SRS evidence",
        "evidence_sentence_indices": [],
        "sources": [],
    }
    with pytest.raises(InsufficientGroundingError):
        _ensure_traceability(entity_dict, ast_signals_for_entity=[])


def test_ensure_traceability_passes_when_evidence_present():
    entity_dict = {
        "name": "Customer",
        "description": "buyer",
        "evidence_sentence_indices": [2, 5],
        "sources": [],
    }
    # Should not raise
    _ensure_traceability(entity_dict, ast_signals_for_entity=[])


def test_ensure_traceability_passes_when_ast_signals_present():
    entity_dict = {
        "name": "Order",
        "description": "order",
        "evidence_sentence_indices": [],
        "sources": [],
    }
    ast_signals = [{"file": "src/order.py", "line": 12, "rule": "AST_CLASS_DECL"}]
    _ensure_traceability(entity_dict, ast_signals_for_entity=ast_signals)


def test_ensure_traceability_raises_on_d1_sentinel_with_no_ast_signals():
    """D1's [-1] sentinel signals AST-emitted with no SRS evidence.
    Without supporting AST signals it must raise."""
    entity_dict = {
        "name": "ASTOnlyEntity",
        "description": "AST-derived, no SRS grounding",
        "evidence_sentence_indices": [-1],
        "sources": [],
    }
    with pytest.raises(InsufficientGroundingError):
        _ensure_traceability(entity_dict, ast_signals_for_entity=[])
