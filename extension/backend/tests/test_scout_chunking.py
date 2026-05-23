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


# ---------------------------------------------------------------------------
# WP-CORE truncated_chunks audit follow-up — count_truncated_sections helper
# closes the "ChunkMetadata.truncated_chunks always-zero" gap.
# ---------------------------------------------------------------------------


def test_count_truncated_sections_zero_when_no_section_subdivided():
    """Helper returns 0 when every section fits under budget."""
    from core.scout.chunking import count_truncated_sections, section_aware_chunks
    chunks = section_aware_chunks(SAMPLE_SRS.read_text(), token_budget=10000)
    assert count_truncated_sections(chunks) == 0


def test_count_truncated_sections_one_when_single_section_subdivided():
    """One oversize section subdivided into N sub-chunks counts as 1
    truncated section (not N-1, not N)."""
    from core.scout.chunking import count_truncated_sections, section_aware_chunks
    long_section = "1 Big Section\n" + ("This is a very long paragraph. " * 5000)
    chunks = section_aware_chunks(long_section, token_budget=50)
    # Subdivision produces multiple .chunk_N suffixed ids for section "1".
    assert any(".chunk_" in c["section_id"] for c in chunks)
    assert count_truncated_sections(chunks) == 1


def test_count_truncated_sections_two_distinct_subdivided_sections():
    """Two distinct oversize sections each subdivided count as 2."""
    from core.scout.chunking import count_truncated_sections, section_aware_chunks
    text = (
        "1 First Long\n" + ("alpha sentence. " * 4000)
        + "\n2 Second Long\n" + ("beta sentence. " * 4000)
    )
    chunks = section_aware_chunks(text, token_budget=50)
    base_ids = {c["section_id"].split(".chunk_")[0] for c in chunks}
    # Both top-level sections must have been retained as bases.
    assert "1" in base_ids and "2" in base_ids
    assert count_truncated_sections(chunks) == 2


def test_count_truncated_sections_mixed_short_and_long():
    """Short + long sections: only the long one counts as truncated."""
    from core.scout.chunking import count_truncated_sections, section_aware_chunks
    text = (
        "1 Short\nOne sentence here.\n"
        "2 Long\n" + ("gamma sentence. " * 4000)
    )
    chunks = section_aware_chunks(text, token_budget=50)
    assert count_truncated_sections(chunks) == 1


def test_count_truncated_sections_empty_list():
    """Helper handles empty input without crashing."""
    from core.scout.chunking import count_truncated_sections
    assert count_truncated_sections([]) == 0
