"""
RAG Pipeline for DDD Rule Source Tracking

Uses ChromaDB with built-in all-MiniLM-L6-v2 embedding model
to index SRS documents and retrieve source references for violations.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import chromadb


class RAGPipeline:
    """
    RAG pipeline for tracking DDD rule sources using ChromaDB.
    Uses ChromaDB's built-in all-MiniLM-L6-v2 embedding model (no external API needed).
    """

    CHUNK_SIZE = 250  # Target words per chunk (model limit: 256)
    CHUNK_OVERLAP = 30  # Words overlap between chunks
    COLLECTION_NAME = "srs_documents"

    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB with persistent storage.

        Args:
            persist_directory: Path for ChromaDB storage. Defaults to backend/data/chroma_db/
        """
        if persist_directory is None:
            persist_directory = str(
                Path(__file__).parent.parent / "data" / "chroma_db"
            )

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
            # Uses default all-MiniLM-L6-v2 embedding automatically
        )

    def index_document(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        doc_type: str
    ) -> int:
        """
        Index a document into ChromaDB with section-aware chunking.

        Args:
            raw_text: The full text content of the document
            doc_id: Unique identifier for the document
            doc_name: Original filename
            doc_type: File extension (pdf, docx, txt)

        Returns:
            Number of chunks indexed
        """
        # Delete existing chunks for this document (re-indexing)
        self._delete_document(doc_id)

        # Parse sections and create chunks
        chunks = self._create_chunks(raw_text, doc_id, doc_name, doc_type)

        if not chunks:
            return 0

        # Add to ChromaDB
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
        n_results: int = 2
    ) -> List[Dict]:
        """
        Retrieve source documents for a violation.

        Args:
            violation_type: Type of violation (NamingViolation, ContextViolation, etc.)
            violation_message: The violation explanation message
            n_results: Number of sources to return

        Returns:
            List of source references with document, section, page, and excerpt
        """
        if self.collection.count() == 0:
            return []

        # Build query based on violation
        query = self._build_query(violation_type, violation_message)

        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        sources = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                sources.append({
                    "document": metadata.get("doc_name", "unknown"),
                    "section": metadata.get("section_name", "unknown"),
                    "page": metadata.get("page_number", 0),
                    "excerpt": doc[:300] + "..." if len(doc) > 300 else doc,
                    "relevance_score": round(1 - distance, 3)  # Convert distance to similarity
                })

        return sources

    def _create_chunks(
        self,
        raw_text: str,
        doc_id: str,
        doc_name: str,
        doc_type: str
    ) -> List[Dict]:
        """
        Create chunks with metadata from raw text using section-aware splitting.
        """
        chunks = []

        # Parse sections (pattern: "N.N Title" or "N. Title")
        section_pattern = r'^(\d+\.?\d*\.?)\s+(.+?)$'
        lines = raw_text.split('\n')

        current_section = {"number": "0", "name": "Introduction", "content": []}
        sections = []

        for line in lines:
            stripped = line.strip()
            match = re.match(section_pattern, stripped)
            if match and len(stripped) < 100:  # Section headers are typically short
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                # Start new section
                section_num = match.group(1).rstrip('.')
                section_title = match.group(2)
                current_section = {
                    "number": section_num,
                    "name": f"{section_num} {section_title}",
                    "content": []
                }
            else:
                current_section["content"].append(line)

        # Add last section
        if current_section["content"]:
            sections.append(current_section)

        # If no sections detected, treat entire document as one section
        if not sections:
            sections = [{
                "number": "1",
                "name": "Document Content",
                "content": lines
            }]

        # Create chunks from sections
        chunk_index = 0
        for section in sections:
            section_text = '\n'.join(section["content"]).strip()
            if not section_text:
                continue

            # Determine bounded context from section name
            bounded_context = self._extract_bounded_context(section["name"])

            # Determine chunk type
            chunk_type = self._classify_chunk_type(section["name"], section_text)

            # Split into sub-chunks if needed
            sub_chunks = self._split_section(section_text, section["name"])

            for sub_chunk in sub_chunks:
                # Extract entities and synonyms mentioned
                entities = self._extract_entities(sub_chunk)
                synonyms = self._extract_synonyms(sub_chunk)
                banned = self._extract_banned_terms(sub_chunk)

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
                        "entities_mentioned": json.dumps(entities),
                        "synonyms_mentioned": json.dumps(synonyms),
                        "banned_terms": json.dumps(banned)
                    }
                })
                chunk_index += 1

        return chunks

    def _split_section(self, text: str, section_header: str) -> List[str]:
        """Split section into chunks respecting word limit with overlap."""
        words = text.split()

        if len(words) <= self.CHUNK_SIZE:
            return [f"{section_header}\n\n{text}"]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.CHUNK_SIZE, len(words))

            # Try to end at sentence boundary
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            # Find last sentence boundary (period followed by space or end)
            last_period = chunk_text.rfind('. ')
            if last_period > len(chunk_text) * 0.6:  # At least 60% of chunk
                chunk_text = chunk_text[:last_period + 1]
                end = start + len(chunk_text.split())

            chunks.append(f"{section_header}\n\n{chunk_text}")

            # Move start with overlap
            start = max(start + 1, end - self.CHUNK_OVERLAP)

            # Safety check to prevent infinite loop
            if start >= len(words) - self.CHUNK_OVERLAP:
                break

        return chunks

    def _build_query(self, violation_type: str, message: str) -> str:
        """Build search query from violation information."""
        query_parts = []

        # Add violation type context
        if "NamingViolation" in violation_type or "synonym" in message.lower():
            query_parts.append("synonym")
            query_parts.append("should NOT be used")
            query_parts.append("Important")

        if "banned" in message.lower() or "Manager" in message or "Util" in message:
            query_parts.append("banned term")
            query_parts.append("naming convention")

        # Extract quoted terms from message (e.g., 'Client', 'Manager')
        quoted = re.findall(r"'([^']+)'", message)
        query_parts.extend(quoted)

        # Extract terms in double quotes
        double_quoted = re.findall(r'"([^"]+)"', message)
        query_parts.extend(double_quoted)

        return ' '.join(query_parts) if query_parts else message

    def _extract_bounded_context(self, section_name: str) -> str:
        """Extract bounded context name from section."""
        context_keywords = [
            "Customer Management", "Order Processing",
            "Inventory Management", "Payment Processing",
            "Product Catalog", "Shopping Cart", "Shipping",
            "User Management", "Account Management"
        ]
        lower_name = section_name.lower()
        for ctx in context_keywords:
            if ctx.lower() in lower_name:
                return ctx
        return "global"

    def _classify_chunk_type(self, section_name: str, text: str) -> str:
        """Classify chunk type for filtering."""
        lower_name = section_name.lower()
        lower_text = text.lower()

        if "glossary" in lower_name or "terminology" in lower_name:
            return "glossary"
        if "dependencies" in lower_name or "relationship" in lower_name:
            return "context_dependency"
        if "important:" in lower_text or "should not be used" in lower_text:
            return "domain_rule"
        if "must not" in lower_text or "must be" in lower_text:
            return "domain_rule"
        if "banned" in lower_text or "forbidden" in lower_text:
            return "global_rule"
        return "general"

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity names mentioned in text."""
        # Common DDD entity names
        entities = [
            "Customer", "Order", "Product", "Payment", "Address",
            "OrderItem", "StockLevel", "Category", "Wishlist", "Money",
            "Invoice", "Shipment", "Cart", "Discount", "Coupon",
            "User", "Account", "Transaction", "Inventory", "Supplier"
        ]
        return [e for e in entities if e in text]

    def _extract_synonyms(self, text: str) -> List[str]:
        """Extract synonym terms mentioned in text."""
        synonyms = [
            "Client", "User", "Buyer", "Shopper", "Consumer",
            "Item", "Good", "Merchandise", "Article",
            "Purchase", "Transaction", "Sale", "Charge",
            "Basket", "Bag"
        ]
        return [s for s in synonyms if s in text]

    def _extract_banned_terms(self, text: str) -> List[str]:
        """Extract banned terms mentioned in text."""
        banned = [
            "Manager", "Helper", "Util", "Utility", "Data", "Info",
            "Handler", "Processor", "Controller", "Bean", "DTO"
        ]
        return [b for b in banned if b in text]

    def _estimate_page(self, chunk_index: int, chunks_per_page: int = 4) -> int:
        """Estimate page number from chunk index."""
        return (chunk_index // chunks_per_page) + 1

    def _delete_document(self, doc_id: str):
        """Delete all chunks for a document (for re-indexing)."""
        try:
            # Get all IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
        except Exception:
            pass  # Collection might be empty or document not indexed

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()

        stats = {
            "collection_name": self.COLLECTION_NAME,
            "total_chunks": count,
            "persist_directory": self.persist_directory
        }

        if count > 0:
            sample = self.collection.get(limit=1)
            if sample["metadatas"]:
                stats["metadata_fields"] = list(sample["metadatas"][0].keys())

        return stats

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
