"""Phase C5: section-aware chunking."""

from pathlib import Path
from core.scout.chunking import section_aware_chunks


SAMPLE_SRS = Path(__file__).parent / "fixtures" / "sample_srs.txt"


def test_section_chunker_emits_one_chunk_per_section():
    text = SAMPLE_SRS.read_text()
    chunks = section_aware_chunks(text, token_budget=10000)
    assert len(chunks) == 5  # 1, 2, 2.1, 2.2, 3
    titles = [c["section_title"] for c in chunks]
    assert any("Order Management" in t for t in titles)
    assert any("Inventory" in t for t in titles)


def test_section_chunker_assigns_unique_section_ids():
    text = SAMPLE_SRS.read_text()
    chunks = section_aware_chunks(text, token_budget=10000)
    ids = [c["section_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_section_chunker_splits_oversize_section_under_budget():
    long_section = "1 Big Section\n" + ("This is a very long paragraph. " * 5000)
    chunks = section_aware_chunks(long_section, token_budget=50)
    assert len(chunks) > 1
    for c in chunks:
        # 50 tokens ≈ 200 chars (rough heuristic in chunker)
        assert len(c["text"]) <= 250


def test_section_chunker_keeps_short_sections_intact():
    text = "1 Tiny Section\nOne sentence.\n2 Another\nAnother sentence."
    chunks = section_aware_chunks(text, token_budget=10000)
    assert len(chunks) == 2
