"""WP-CORE-25 — RAG hızlı ayarlar tests.

Covers:
- CHUNK_SIZE regression guard (250 → 600).
- _filter_and_format_sources MIN_RELEVANCE_SCORE filter (the field was
  previously declared in config.py but never enforced).
- _delete_document no longer swallows exceptions silently — typed
  warning is logged.
"""
from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from config import RAGConfig


class TestRAGConfigChunkSize(unittest.TestCase):
    def test_t_rag_chunk_1_chunk_size_raised(self):
        """T-RAG-CHUNK-1: CHUNK_SIZE >= 500 after WP-CORE-25."""
        self.assertGreaterEqual(
            RAGConfig.CHUNK_SIZE, 500,
            "CHUNK_SIZE was raised from 250 in WP-CORE-25 to keep "
            "multi-clause domain rules in one chunk.",
        )


class TestFilterAndFormatSources(unittest.TestCase):
    """Tests the pure helper without spinning up ChromaDB."""

    def _build_rag_pipeline_without_init(self):
        """Construct a RAGPipeline instance bypassing __init__ to avoid
        ChromaDB collection setup. Only the helper method is exercised."""
        from core.rag_pipeline import RAGPipeline
        pipe = RAGPipeline.__new__(RAGPipeline)
        return pipe

    def test_t_rag_filter_1_below_threshold_dropped(self):
        """T-RAG-FILTER-1: results with relevance < MIN_RELEVANCE_SCORE drop."""
        pipe = self._build_rag_pipeline_without_init()
        # MIN_RELEVANCE_SCORE = 0.3 → distance > 0.7 drops.
        results = {
            "documents": [["text_kept", "text_dropped"]],
            "metadatas": [[{"doc_name": "k"}, {"doc_name": "d"}]],
            "distances": [[0.5, 0.85]],  # relevance 0.5 (keep), 0.15 (drop)
        }
        sources = pipe._filter_and_format_sources(results)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["document"], "k")
        self.assertEqual(sources[0]["relevance_score"], 0.5)

    def test_t_rag_filter_2_all_above_threshold_kept(self):
        """T-RAG-FILTER-2: all results pass when relevance > threshold."""
        pipe = self._build_rag_pipeline_without_init()
        results = {
            "documents": [["a", "b"]],
            "metadatas": [[{"doc_name": "a"}, {"doc_name": "b"}]],
            "distances": [[0.1, 0.2]],  # relevance 0.9, 0.8
        }
        sources = pipe._filter_and_format_sources(results)
        self.assertEqual(len(sources), 2)

    def test_t_rag_filter_3_empty_results(self):
        """T-RAG-FILTER-3: empty input → empty output, no crash."""
        pipe = self._build_rag_pipeline_without_init()
        self.assertEqual(pipe._filter_and_format_sources({}), [])
        self.assertEqual(pipe._filter_and_format_sources({"documents": []}), [])
        self.assertEqual(pipe._filter_and_format_sources({"documents": [[]]}), [])

    def test_t_rag_filter_4_all_below_threshold_empty(self):
        """T-RAG-FILTER-4: all results below threshold → empty list."""
        pipe = self._build_rag_pipeline_without_init()
        results = {
            "documents": [["a", "b"]],
            "metadatas": [[{"doc_name": "a"}, {"doc_name": "b"}]],
            "distances": [[0.95, 0.99]],  # relevance 0.05, 0.01 — both drop
        }
        sources = pipe._filter_and_format_sources(results)
        self.assertEqual(sources, [])


class TestDeleteDocumentLogging(unittest.TestCase):
    """Verify _delete_document logs typed warning instead of swallowing silently."""

    def test_t_rag_delete_1_exception_logged(self):
        """T-RAG-DELETE-1: collection.get() raising → warning logged, no crash."""
        from core.rag_pipeline import RAGPipeline
        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe.collection = MagicMock()
        pipe.collection.get.side_effect = RuntimeError("chromadb backend down")

        with self.assertLogs("core.rag_pipeline", level="WARNING") as cm:
            pipe._delete_document("doc-123")

        self.assertTrue(
            any("_delete_document failed" in msg for msg in cm.output),
            f"expected warning log; got {cm.output}",
        )
        self.assertTrue(
            any("RuntimeError" in msg for msg in cm.output),
            "warning should include exception type",
        )

    def test_t_rag_delete_2_success_no_log(self):
        """T-RAG-DELETE-2: successful delete logs nothing."""
        from core.rag_pipeline import RAGPipeline
        pipe = RAGPipeline.__new__(RAGPipeline)
        pipe.collection = MagicMock()
        pipe.collection.get.return_value = {"ids": ["chunk_1"]}

        logger = logging.getLogger("core.rag_pipeline")
        with patch.object(logger, "warning") as warn_mock:
            pipe._delete_document("doc-123")
            warn_mock.assert_not_called()
        pipe.collection.delete.assert_called_once_with(ids=["chunk_1"])


if __name__ == "__main__":
    unittest.main()
