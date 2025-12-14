"""
RAG Pipeline

Retrieval-Augmented Generation pipeline for tracking DDD rule sources.
Uses ChromaDB with built-in all-MiniLM-L6-v2 embedding model to index
SRS documents and retrieve source references for violations.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import chromadb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAGConfig, INPUTS_DIR


class RAGPipeline:
    """RAG pipeline for tracking DDD rule sources using ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        inputs_directory: Optional[str] = None,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_directory: Path for ChromaDB storage
            inputs_directory: Path where source documents are stored
            config: RAGConfig instance for custom settings
        """
        self.config = config or RAGConfig()

        # Set directories
        self.persist_directory = persist_directory or self.config.PERSIST_DIRECTORY
        self.inputs_directory = inputs_directory or str(INPUTS_DIR)

        # Load settings from config
        self.chunk_size = self.config.CHUNK_SIZE
        self.chunk_overlap = self.config.CHUNK_OVERLAP
        self.chunks_per_page = self.config.CHUNKS_PER_PAGE
        self.top_k = self.config.TOP_K
        self.min_relevance = self.config.MIN_RELEVANCE_SCORE
        self.max_summary_length = self.config.MAX_SUMMARY_LENGTH

        # Metadata extraction settings
        self.bounded_context_keywords = self.config.BOUNDED_CONTEXT_KEYWORDS
        self.entity_names = self.config.ENTITY_NAMES
        self.synonym_terms = self.config.SYNONYM_TERMS
        self.banned_terms = self.config.BANNED_TERMS

        # Chunk classification keywords
        self.glossary_keywords = self.config.GLOSSARY_KEYWORDS
        self.dependency_keywords = self.config.DEPENDENCY_KEYWORDS
        self.domain_rule_keywords = self.config.DOMAIN_RULE_KEYWORDS
        self.global_rule_keywords = self.config.GLOBAL_RULE_KEYWORDS

        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with default embedding (all-MiniLM-L6-v2)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.config.COLLECTION_NAME,
            metadata={"hnsw:space": self.config.DISTANCE_METRIC}
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def index_document(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        doc_type: str
    ) -> int:
        """
        Index a document into ChromaDB with section-aware chunking.

        Returns the number of chunks indexed.
        """
        self._delete_document(doc_id)
        chunks = self._create_chunks(raw_text, doc_id, doc_name, doc_type)

        if not chunks:
            return 0

        self.collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks]
        )

        return len(chunks)

    def retrieve_source(
        self,
        violation_type: str,
        violation_message: str,
        n_results: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve source documents for a violation.

        Returns list of source references with document, section, page,
        summary, and file_path.
        """
        if self.collection.count() == 0:
            return []

        n_results = n_results or self.top_k
        query = self._build_query(violation_type, violation_message)

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        sources = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                doc_name = metadata.get("doc_name", "unknown")

                sources.append({
                    "document": doc_name,
                    "section": metadata.get("section_name", "unknown"),
                    "page": metadata.get("page_number", 0),
                    "summary": self._generate_summary(doc, metadata),
                    "file_path": str(Path(self.inputs_directory) / doc_name),
                    "relevance_score": round(1 - distance, 3)
                })

        return sources

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Direct search for debugging."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        return [
            {
                "text": results["documents"][0][i][:300] + "..." if len(results["documents"][0][i]) > 300 else results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance": round(1 - results["distances"][0][i], 3)
            }
            for i in range(len(results["documents"][0]))
        ]

    def get_stats(self) -> Dict:
        """Get collection statistics and configuration."""
        count = self.collection.count()

        stats = {
            "collection_name": self.config.COLLECTION_NAME,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k,
                "min_relevance_score": self.min_relevance,
                "distance_metric": self.config.DISTANCE_METRIC,
                "max_summary_length": self.max_summary_length,
            }
        }

        if count > 0:
            sample = self.collection.get(limit=1)
            if sample["metadatas"]:
                stats["metadata_fields"] = list(sample["metadatas"][0].keys())

        return stats

    # =========================================================================
    # CHUNKING
    # =========================================================================

    def _create_chunks(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        doc_type: str
    ) -> List[Dict]:
        """Create chunks with metadata from raw text using section-aware splitting."""
        sections = self._parse_sections(raw_text)
        chunks = []
        chunk_index = 0

        for section in sections:
            section_text = '\n'.join(section["content"]).strip()
            if not section_text:
                continue

            bounded_context = self._extract_bounded_context(section["name"])
            chunk_type = self._classify_chunk_type(section["name"], section_text)
            sub_chunks = self._split_section(section_text, section["name"])

            for sub_chunk in sub_chunks:
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                chunks.append({
                    "id": chunk_id,
                    "text": sub_chunk,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "doc_type": doc_type,
                        "section_name": section["name"],
                        "section_number": section["number"],
                        "page_number": self._estimate_page(chunk_index),
                        "chunk_index": chunk_index,
                        "chunk_type": chunk_type,
                        "bounded_context": bounded_context,
                        "entities_mentioned": json.dumps(self._extract_entities(sub_chunk)),
                        "synonyms_mentioned": json.dumps(self._extract_synonyms(sub_chunk)),
                        "banned_terms": json.dumps(self._extract_banned_terms(sub_chunk))
                    }
                })
                chunk_index += 1

        return chunks

    def _parse_sections(self, raw_text: str) -> List[Dict]:
        """Parse document into sections based on numbered headers."""
        section_pattern = r'^(\d+\.?\d*\.?)\s+(.+?)$'
        lines = raw_text.split('\n')

        current_section = {"number": "0", "name": "Introduction", "content": []}
        sections = []

        for line in lines:
            stripped = line.strip()
            match = re.match(section_pattern, stripped)

            if match and len(stripped) < 100:
                if current_section["content"]:
                    sections.append(current_section)
                section_num = match.group(1).rstrip('.')
                section_title = match.group(2)
                current_section = {
                    "number": section_num,
                    "name": f"{section_num} {section_title}",
                    "content": []
                }
            else:
                current_section["content"].append(line)

        if current_section["content"]:
            sections.append(current_section)

        if not sections:
            sections = [{
                "number": "1",
                "name": "Document Content",
                "content": lines
            }]

        return sections

    def _split_section(self, text: str, section_header: str) -> List[str]:
        """Split section into chunks respecting word limit with overlap."""
        words = text.split()

        if len(words) <= self.chunk_size:
            return [f"{section_header}\n\n{text}"]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Try to end at sentence boundary
            last_period = chunk_text.rfind('. ')
            if last_period > len(chunk_text) * 0.6:
                chunk_text = chunk_text[:last_period + 1]
                end = start + len(chunk_text.split())

            chunks.append(f"{section_header}\n\n{chunk_text}")

            # Move start with overlap
            start = max(start + 1, end - self.chunk_overlap)
            if start >= len(words) - self.chunk_overlap:
                break

        return chunks

    # =========================================================================
    # QUERY BUILDING
    # =========================================================================

    def _build_query(self, violation_type: str, message: str) -> str:
        """Build search query from violation information."""
        query_parts = []
        message_lower = message.lower()

        # Extract quoted terms
        quoted = re.findall(r"'([^']+)'", message)
        double_quoted = re.findall(r'"([^"]+)"', message)
        extracted_terms = quoted + double_quoted

        # Check for banned global terms
        banned_terms_lower = [t.lower() for t in self.banned_terms]
        is_banned_term = any(term in message_lower for term in banned_terms_lower)

        if is_banned_term or "banned" in message_lower:
            query_parts.extend(["banned", "generic terms", "violate DDD principles"])
            for term in extracted_terms:
                if term.lower() in banned_terms_lower:
                    query_parts.append(term)

        elif "synonym" in message_lower or "should not be used" in message_lower:
            query_parts.extend(["synonym", "should NOT be used", "Important"])
            for term in extracted_terms:
                if term not in ["NamingViolation", "ContextViolation"]:
                    query_parts.append(term)

        else:
            query_parts.extend(extracted_terms)
            query_parts.append("domain rule")

        return ' '.join(query_parts) if query_parts else message

    # =========================================================================
    # SUMMARY GENERATION
    # =========================================================================

    def _generate_summary(self, text: str, metadata: Dict) -> str:
        """Generate a short one-line summary from the source text."""
        lines = text.split('\n')
        content = '\n'.join(lines[1:]) if len(lines) > 1 else text
        section_name = metadata.get("section_name", "").lower()

        # Special handling for known sections
        if "banned" in section_name and "term" in section_name:
            return "Generic terms banned as they violate DDD principles"

        if "naming convention" in section_name:
            return "Use PascalCase for entity and class names"

        if "glossary" in section_name:
            return "Domain terminology definitions"

        # Look for "Important:" rules
        match = re.search(r'Important:\s*(.+?)(?:\.|$)', content, re.IGNORECASE)
        if match:
            summary = re.sub(r'\s+', ' ', match.group(1).strip())
            return summary[:self.max_summary_length]

        # Look for "should NOT be used" patterns
        match = re.search(r'(Terms?\s+like\s+.+?should\s+NOT\s+be\s+used)', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:self.max_summary_length]

        # Look for "must be used" / "must not" patterns
        match = re.search(r'(.+?must\s+(?:be\s+used|not)[^.]*)', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:self.max_summary_length]

        # Fall back to bounded context description
        context = metadata.get("bounded_context", "")
        if context and context != "global":
            return f"DDD rule from {context}"

        # Last resort: first meaningful sentence
        for line in content.split('\n'):
            line = line.strip()
            if line and len(line) > 10 and not line.startswith('-'):
                return line[:self.max_summary_length]

        return "Source reference"

    # =========================================================================
    # METADATA EXTRACTION
    # =========================================================================

    def _extract_bounded_context(self, section_name: str) -> str:
        """Extract bounded context name from section."""
        lower_name = section_name.lower()
        for ctx in self.bounded_context_keywords:
            if ctx.lower() in lower_name:
                return ctx
        return "global"

    def _classify_chunk_type(self, section_name: str, text: str) -> str:
        """Classify chunk type for filtering."""
        lower_name = section_name.lower()
        lower_text = text.lower()

        if any(kw in lower_name for kw in self.glossary_keywords):
            return "glossary"

        if any(kw in lower_name for kw in self.dependency_keywords):
            return "context_dependency"

        if any(kw in lower_text for kw in self.domain_rule_keywords):
            return "domain_rule"

        if any(kw in lower_text for kw in self.global_rule_keywords):
            return "global_rule"

        return "general"

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity names mentioned in text."""
        return [e for e in self.entity_names if e in text]

    def _extract_synonyms(self, text: str) -> List[str]:
        """Extract synonym terms mentioned in text."""
        return [s for s in self.synonym_terms if s in text]

    def _extract_banned_terms(self, text: str) -> List[str]:
        """Extract banned terms mentioned in text."""
        return [b for b in self.banned_terms if b in text]

    def _estimate_page(self, chunk_index: int) -> int:
        """Estimate page number from chunk index."""
        return (chunk_index // self.chunks_per_page) + 1

    def _delete_document(self, doc_id: str):
        """Delete all chunks for a document (for re-indexing)."""
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        except Exception:
            pass
