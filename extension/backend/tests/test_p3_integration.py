"""Phase C8: end-to-end integration test for the 5-stage pipeline against
the D1 SRS. Requires DDD_INTEGRATION_TEST=1 and a real GEMINI_API_KEY.

Run with:
    DDD_INTEGRATION_TEST=1 GEMINI_API_KEY=... pytest tests/test_p3_integration.py -v
"""

import os
import pytest
from pathlib import Path
from core.architect import DomainArchitect
from core.document_parser import SRSDocumentParser


pytestmark = pytest.mark.integration


@pytest.fixture
def srs_text():
    srs_path = Path("inputs/SRS.docx")
    if not srs_path.exists():
        pytest.skip("inputs/SRS.docx not present")
    parser = SRSDocumentParser()
    return parser.parse_file(str(srs_path))


@pytest.mark.skipif(
    os.getenv("DDD_INTEGRATION_TEST") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="integration test gated by DDD_INTEGRATION_TEST=1 + GEMINI_API_KEY"
)
def test_d1_srs_produces_valid_domain_model(srs_text):
    arch = DomainArchitect()
    model = arch.analyze_document(text=srs_text)
    assert model.project_name
    assert len(model.bounded_contexts) >= 1
    for bc in model.bounded_contexts:
        assert bc.ubiquitous_language.entities or bc.ubiquitous_language.value_objects
        for e in bc.ubiquitous_language.entities:
            assert 0.0 <= e.confidence <= 1.0
            assert e.justification
