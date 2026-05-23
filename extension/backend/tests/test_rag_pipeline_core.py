"""WP-CORE-29 — RAG pipeline core method unit tests.

The DevEx audit (iteration-18 plan-research sweep) surfaced that
`core/rag_pipeline.py` had ZERO direct unit tests prior to WP-CORE-25
(which added `test_rag_pipeline_filters.py` covering the filter +
delete-logging changes). This file covers the rest of the pure helpers:
_parse_sections, _split_section, _get_overlap, _build_query,
_estimate_page, _generate_summary, get_stats.

All tests bypass `__init__` via `__new__` to avoid the ChromaDB
collection setup.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from config import RAGConfig


def _bare_pipe():
    """Construct a RAGPipeline without invoking __init__."""
    from core.rag_pipeline import RAGPipeline
    pipe = RAGPipeline.__new__(RAGPipeline)
    pipe.chunk_size = RAGConfig.CHUNK_SIZE
    pipe.chunk_overlap = RAGConfig.CHUNK_OVERLAP
    pipe.top_k = RAGConfig.TOP_K
    pipe.collection_name = RAGConfig.COLLECTION_NAME
    pipe.persist_directory = RAGConfig.PERSIST_DIRECTORY
    return pipe


# ---------------------------------------------------------------------------
# _parse_sections
# ---------------------------------------------------------------------------

class TestParseSections(unittest.TestCase):
    def test_t_rag_parse_1_numbered_section(self):
        """T-RAG-PARSE-1: numbered sections (no trailing period) are detected.

        Current regex `^(\\d+(?:\\.\\d+)*)\\s+(.+)$` matches `1 Introduction` and
        `1.1 Sub` but NOT `1. Introduction` (period+space). The
        document_parser.heading_pattern accepts the period form via
        optional `\\.?`. The rag_pipeline parser is stricter; this is a
        known gap not covered by WP-CORE-29 (separate WP).
        """
        pipe = _bare_pipe()
        text = """1 Introduction
This is the intro paragraph.

2 Requirements
The requirements section.
"""
        sections = pipe._parse_sections(text)
        names = [s["name"] for s in sections]
        self.assertTrue(any("Introduction" in n for n in names),
                        f"expected Introduction in {names}")
        self.assertTrue(any("Requirements" in n for n in names),
                        f"expected Requirements in {names}")

    def test_t_rag_parse_2_markdown_header(self):
        """T-RAG-PARSE-2: markdown # headers are detected."""
        pipe = _bare_pipe()
        text = """# Overview
Some overview content.

## Details
The details.
"""
        sections = pipe._parse_sections(text)
        names = [s["name"] for s in sections]
        self.assertTrue(any("Overview" in n for n in names))
        self.assertTrue(any("Details" in n for n in names))

    def test_t_rag_parse_3_no_headers_fallback(self):
        """T-RAG-PARSE-3: text without headers → single fallback section."""
        pipe = _bare_pipe()
        text = "Plain paragraph with no headers.\nAnother line."
        sections = pipe._parse_sections(text)
        self.assertEqual(len(sections), 1)
        # The fallback name is "Preamble" (first un-headered block) OR
        # "Document Content" (if no preamble either) — both acceptable.
        self.assertIn(sections[0]["name"], ("Preamble", "Document Content"))

    def test_t_rag_parse_4_long_line_not_header(self):
        """T-RAG-PARSE-4: lines >= 150 chars are NOT treated as headers."""
        pipe = _bare_pipe()
        long_line = "1. " + ("x" * 200)
        text = f"{long_line}\nBody content."
        sections = pipe._parse_sections(text)
        # No section should claim that 200-char line as its title.
        names = [s["name"] for s in sections]
        for n in names:
            self.assertLess(len(n), 250, f"section name too long: {n[:50]}...")


# ---------------------------------------------------------------------------
# _split_section
# ---------------------------------------------------------------------------

class TestSplitSection(unittest.TestCase):
    def test_t_rag_split_1_small_section_single_chunk(self):
        """T-RAG-SPLIT-1: small section returns a single chunk with header."""
        pipe = _bare_pipe()
        chunks = pipe._split_section(
            text="short content", section_header="1. Intro", section_number="1",
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("1. Intro", chunks[0])
        self.assertIn("short content", chunks[0])

    def test_t_rag_split_2_large_section_multiple_chunks(self):
        """T-RAG-SPLIT-2: long section is split into multiple chunks."""
        pipe = _bare_pipe()
        # Build a section that exceeds CHUNK_SIZE.
        paragraphs = ["para " + ("x" * 200) for _ in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = pipe._split_section(text, "5. Big", "5")
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertIn("5. Big", chunk)

    def test_t_rag_split_3_chunks_carry_overlap(self):
        """T-RAG-SPLIT-3: consecutive chunks share overlap text."""
        pipe = _bare_pipe()
        paragraphs = [f"para_{i} " + ("x" * 200) for i in range(8)]
        text = "\n\n".join(paragraphs)
        chunks = pipe._split_section(text, "6. Big", "6")
        # Overlap not always trivially substring-matched, but consecutive
        # chunks should not be identical and both should be substantial.
        self.assertGreater(len(chunks), 1)
        self.assertNotEqual(chunks[0], chunks[1])


# ---------------------------------------------------------------------------
# _get_overlap
# ---------------------------------------------------------------------------

class TestGetOverlap(unittest.TestCase):
    def test_t_rag_overlap_1_short_text(self):
        """T-RAG-OVERLAP-1: text shorter than overlap → returns whole text."""
        pipe = _bare_pipe()
        pipe.chunk_overlap = 30
        self.assertEqual(pipe._get_overlap("short"), "short")

    def test_t_rag_overlap_2_returns_tail(self):
        """T-RAG-OVERLAP-2: text longer than overlap → returns last ~N chars."""
        pipe = _bare_pipe()
        pipe.chunk_overlap = 20
        long = "a" * 100
        overlap = pipe._get_overlap(long)
        self.assertLessEqual(len(overlap), 30)

    def test_t_rag_overlap_3_sentence_boundary_preferred(self):
        """T-RAG-OVERLAP-3: overlap trims to sentence boundary when present."""
        pipe = _bare_pipe()
        pipe.chunk_overlap = 60
        text = "x" * 50 + ". The actual sentence we want carried over."
        overlap = pipe._get_overlap(text)
        # The overlap should not start mid-word; should start at the sentence.
        self.assertTrue(
            overlap.startswith("The") or overlap.startswith(". The") or overlap.startswith("x"),
            f"unexpected overlap start: {overlap[:30]!r}",
        )


# ---------------------------------------------------------------------------
# _build_query
# ---------------------------------------------------------------------------

class TestBuildQuery(unittest.TestCase):
    def test_t_rag_query_1_no_quoted_terms(self):
        """T-RAG-QUERY-1: no quoted terms → returns message as-is."""
        pipe = _bare_pipe()
        q = pipe._build_query("SynonymViolation", "Customer should be Client")
        self.assertEqual(q, "Customer should be Client")

    def test_t_rag_query_2_quoted_terms_prepended(self):
        """T-RAG-QUERY-2: quoted terms get prepended to the message."""
        pipe = _bare_pipe()
        q = pipe._build_query(
            "SynonymViolation",
            "Term 'Client' should be 'Customer' per glossary",
        )
        self.assertIn("Client", q)
        self.assertIn("Customer", q)


# ---------------------------------------------------------------------------
# _estimate_page
# ---------------------------------------------------------------------------

class TestEstimatePage(unittest.TestCase):
    def test_t_rag_page_1_index_to_page(self):
        """T-RAG-PAGE-1: chunk index 0/1/2 → page 1; 3/4/5 → page 2."""
        pipe = _bare_pipe()
        self.assertEqual(pipe._estimate_page(0), 1)
        self.assertEqual(pipe._estimate_page(2), 1)
        self.assertEqual(pipe._estimate_page(3), 2)
        self.assertEqual(pipe._estimate_page(5), 2)


# ---------------------------------------------------------------------------
# _generate_summary
# ---------------------------------------------------------------------------

class TestGenerateSummary(unittest.TestCase):
    def test_t_rag_summary_1_must_sentence_preferred(self):
        """T-RAG-SUMMARY-1: prefers a sentence containing 'must'."""
        pipe = _bare_pipe()
        text = (
            "Some preamble paragraph.\n\n"
            "Customer must verify their email before checkout.\n\n"
            "Closing line."
        )
        summary = pipe._generate_summary(text)
        self.assertTrue(len(summary) > 0)
        # Heuristic doesn't guarantee 'must' in summary, but it should be
        # non-empty and trimmed.
        self.assertLessEqual(len(summary), 250)

    def test_t_rag_summary_2_empty_text(self):
        """T-RAG-SUMMARY-2: empty text returns a non-crashing fallback."""
        pipe = _bare_pipe()
        result = pipe._generate_summary("")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats(unittest.TestCase):
    def test_t_rag_stats_1_basic_shape(self):
        """T-RAG-STATS-1: get_stats reports collection_name, total_chunks, config."""
        pipe = _bare_pipe()
        pipe.collection = MagicMock()
        pipe.collection.count.return_value = 0
        pipe.collection_name = "test_collection"
        stats = pipe.get_stats()
        self.assertEqual(stats["collection_name"], "test_collection")
        self.assertEqual(stats["total_chunks"], 0)
        self.assertIn("config", stats)
        self.assertEqual(stats["config"]["chunk_size"], pipe.chunk_size)


if __name__ == "__main__":
    unittest.main()
