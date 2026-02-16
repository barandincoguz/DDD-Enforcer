"""
Pure RAG Pipeline for Document Retrieval

This pipeline does ONE thing: retrieve relevant specification sections.
It has ZERO knowledge of:
- Domain entities (Customer, Order, etc.)
- Synonyms (Client, Buyer, etc.)
- Banned terms (Manager, Helper, etc.)
- Business rules

It only knows how to:
1. Chunk documents by structure
2. Store them in ChromaDB
3. Retrieve relevant chunks based on queries
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from config import RAGConfig


class RAGPipeline:
    """
    Generic document retrieval pipeline.
    No domain knowledge. No hardcoded rules. Pure retrieval.
    """
    
    def __init__(
        self,
        persist_directory: str = RAGConfig.PERSIST_DIRECTORY,
        collection_name: str = RAGConfig.COLLECTION_NAME,
        chunk_size: int = RAGConfig.CHUNK_SIZE,
        chunk_overlap: int = RAGConfig.CHUNK_OVERLAP,
        top_k: int = RAGConfig.TOP_K
    ):
        """
        Initialize ChromaDB with persistent storage.
        
        Args:
            persist_directory: Where to store ChromaDB data
            collection_name: Name of the collection
            chunk_size: Target characters per chunk
            chunk_overlap: Character overlap between chunks
            top_k: Default number of results to retrieve
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB with built-in embeddings (all-MiniLM-L6-v2)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    # =========================================================================
    # PUBLIC API
    # =========================================================================
    
    def index_document(
        self, 
        raw_text: str, 
        doc_id: str, 
        doc_name: str, 
        doc_type: str = "SRS"
    ) -> int:
        """
        Index a document into ChromaDB.
        
        Args:
            raw_text: Full document text
            doc_id: Unique identifier for the document
            doc_name: Human-readable document name
            doc_type: Type of document (e.g., "SRS", "spec")
            
        Returns:
            Number of chunks indexed
        """
        # Delete existing chunks for this document (for re-indexing)
        self._delete_document(doc_id)
        
        # Create chunks from document
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
        n_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant source chunks for a violation.
        
        This method answers: "Where in the spec is the rule for this violation?"
        It does NOT interpret the violation or know domain rules.
        
        Args:
            violation_type: Type of violation (e.g., "SynonymViolation")
            violation_message: Description of the violation
            n_results: Number of results to return (default: self.top_k)
            
        Returns:
            List of source references with metadata
        """
        if self.collection.count() == 0:
            return []
        
        n_results = n_results or self.top_k
        
        # Build search query (minimal processing)
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
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                sources.append({
                    "document": metadata.get("doc_name", "unknown"),
                    "section": metadata.get("section_name", "unknown"),
                    "page": metadata.get("page_number", 0),
                    "summary": self._generate_summary(doc_text),
                    "file_path": metadata.get("file_path", ""),
                    "relevance_score": round(1 - distance, 3),
                    "full_text": doc_text
                })
        
        return sources
    
    def search(
        self, 
        query: str, 
        n_results: int = RAGConfig.TOP_K,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Direct search interface for debugging and testing.
        
        Args:
            query: Search query
            n_results: Number of results
            filter_metadata: Optional metadata filters (e.g., {"doc_type": "SRS"})
            
        Returns:
            List of search results
        """
        if self.collection.count() == 0:
            return []
        
        query_params = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filter_metadata:
            query_params["where"] = filter_metadata
        
        results = self.collection.query(**query_params)
        
        if not results["documents"] or not results["documents"][0]:
            return []
        
        return [
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance": round(1 - results["distances"][0][i], 3)
            }
            for i in range(len(results["documents"][0]))
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed data."""
        count = self.collection.count()
        
        stats = {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
            "config": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.top_k
            }
        }
        
        if count > 0:
            # Get a sample to show metadata structure
            sample = self.collection.get(limit=1)
            if sample["metadatas"]:
                stats["metadata_fields"] = list(sample["metadatas"][0].keys())
        
        return stats
    
    # =========================================================================
    # INTERNAL: CHUNKING (PURE DOCUMENT PROCESSING)
    # =========================================================================
    
    def _create_chunks(
        self, 
        raw_text: str, 
        doc_id: str, 
        doc_name: str, 
        doc_type: str
    ) -> List[Dict[str, Any]]:
        """
        Create chunks from raw text.
        Uses section-aware splitting to keep related content together.
        """
        sections = self._parse_sections(raw_text)
        chunks = []
        chunk_index = 0
        
        for section in sections:
            section_text = "\n".join(section["content"]).strip()
            
            if not section_text:
                continue
            
            # Split section into chunks if needed
            section_chunks = self._split_section(
                section_text, 
                section["name"],
                section["number"]
            )
            
            for chunk_text in section_chunks:
                chunk_id = f"{doc_id}_chunk_{chunk_index}"
                
                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "doc_type": doc_type,
                        "section_name": section["name"],
                        "section_number": section["number"],
                        "page_number": self._estimate_page(chunk_index),
                        "chunk_index": chunk_index,
                        "file_path": f"inputs/{doc_name}"
                    }
                })
                chunk_index += 1
        
        return chunks
    
    def _parse_sections(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Parse document into sections based on headers.
        
        Detects:
        - Numbered sections: "1. Introduction", "3.1 Customer Management"
        - Markdown headers: "# Introduction", "## Overview"
        
        This is STRUCTURAL parsing, not semantic understanding.
        """
        # Pattern for numbered sections
        section_pattern = r'^(\d+(?:\.\d+)*)\s+(.+)$'
        
        lines = raw_text.split('\n')
        sections = []
        
        current_section = {
            "number": "0",
            "name": "Preamble",
            "content": []
        }
        
        for line in lines:
            stripped = line.strip()
        
            match = re.match(section_pattern, stripped)
            
            if match and len(stripped) < 150:  # Likely a header, not content
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                section_num = match.group(1)
                section_title = match.group(2)
                current_section = {
                    "number": section_num,
                    "name": f"{section_num} {section_title}",
                    "content": []
                }
            
            # Check for markdown header (e.g., "## Overview")
            elif stripped.startswith('#') and len(stripped) < 150:
                if current_section["content"]:
                    sections.append(current_section)
                
                header_text = stripped.lstrip('#').strip()
                current_section = {
                    "number": str(len(sections) + 1),
                    "name": header_text,
                    "content": []
                }
            
            else:
                # Regular content line
                current_section["content"].append(line)
        
        # Don't forget last section
        if current_section["content"]:
            sections.append(current_section)
        
        # Fallback: if no sections detected, treat as one big section
        if not sections:
            sections = [{
                "number": "1",
                "name": "Document Content",
                "content": lines
            }]
        
        return sections
    
    def _split_section(
        self, 
        text: str, 
        section_header: str,
        section_number: str
    ) -> List[str]:
        """
        Split a section into chunks with overlap.
        Uses paragraph boundaries to avoid mid-sentence breaks.
        """
        # If section is small enough, keep it whole
        if len(text) <= self.chunk_size:
            return [f"{section_header}\n\n{text}"]
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Would adding this paragraph exceed chunk size?
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if len(test_chunk) > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(f"{section_header}\n\n{current_chunk}")
                
                # Start new chunk with overlap
                overlap = self._get_overlap(current_chunk)
                current_chunk = overlap + "\n\n" + para if overlap else para
            else:
                current_chunk = test_chunk
        
        # Save last chunk
        if current_chunk:
            chunks.append(f"{section_header}\n\n{current_chunk}")
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """
        Get last portion of text for overlap between chunks.
        Tries to find sentence boundary.
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap_start = len(text) - self.chunk_overlap
        overlap_text = text[overlap_start:]
        
        # Try to start at sentence boundary
        first_period = overlap_text.find('. ')
        if first_period > 0 and first_period < len(overlap_text) - 10:
            return overlap_text[first_period + 2:]
        
        return overlap_text
    
    def _estimate_page(self, chunk_index: int) -> int:
        """
        Estimate page number from chunk index.
        Assumes ~3 chunks per page.
        """
        return (chunk_index // 3) + 1
    
    # =========================================================================
    # INTERNAL: QUERY BUILDING (MINIMAL PROCESSING)
    # =========================================================================
    
    def _build_query(self, violation_type: str, violation_message: str) -> str:
        """
        Build search query from violation information.
        
        The embedding model (all-MiniLM-L6-v2) handles semantic matching.
        """
        # Extract quoted terms from the message
        # These are usually the specific terms mentioned in the violation
        quoted_terms = re.findall(r"['\"]([^'\"]+)['\"]", violation_message)
        
        # Simple query: quoted terms + violation message
        # Let embeddings do the heavy lifting
        if quoted_terms:
            query_parts = quoted_terms + [violation_message]
            return " ".join(query_parts)
        
        # If no quoted terms, just use the message
        return violation_message
    
    # =========================================================================
    # INTERNAL: SUMMARY GENERATION (HEURISTIC EXTRACTION)
    # =========================================================================
    
    def _generate_summary(self, text: str) -> str:
        """
        Generate a concise summary from chunk text.
        
        Uses simple heuristics to find key sentences:
        - "Important:" statements
        - "must" / "should" statements
        - First meaningful sentence
        
        This is NOT semantic understanding, just pattern matching.
        """
        lines = text.split('\n')
        
        # Skip section header (usually first 1-2 lines)
        content_lines = lines[2:] if len(lines) > 2 else lines
        content = '\n'.join(content_lines)
        
        # Look for "Important:" statements (common in specs)
        match = re.search(r'Important:\s*(.+?)(?:\.|$)', content, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:200]
        
        # Look for "should NOT" or "must NOT" statements
        match = re.search(
            r'(.{0,50}(?:should|must)\s+NOT.+?)(?:\.|$)', 
            content, 
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip()[:200]
        
        # Look for "must" statements
        match = re.search(
            r'(.{0,30}must.+?)(?:\.|$)', 
            content, 
            re.IGNORECASE
        )
        if match:
            return match.group(1).strip()[:200]
        
        # Fallback: first meaningful sentence
        for line in content_lines:
            line = line.strip()
            if len(line) > 20 and not line.startswith(('-', '*', '#')):
                return line[:200]
        
        return "Source reference"
    
    # =========================================================================
    # INTERNAL: UTILITIES
    # =========================================================================
    
    def _delete_document(self, doc_id: str):
        """Delete all chunks for a document (for re-indexing)."""
        try:
            existing = self.collection.get(
                where={"doc_id": doc_id},
                include=[]
            )
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass 