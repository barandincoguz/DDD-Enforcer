"""
Shared domain-model generation service.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.architect import DomainArchitect
from core.ast_model_signals import ASTModelSignalExtractor
from core.document_parser import SRSDocumentParser
from core.research_metrics import ResearchMetricsStore
from core.rag_pipeline import RAGPipeline
from core.schemas import DomainModel
from core.token_tracker import TokenTracker


class ModelGenerationService:
    """Coordinates document parsing, multi-agent generation, and enrichment."""

    def __init__(
        self,
        *,
        document_parser: Optional[SRSDocumentParser] = None,
        token_tracker: Optional[TokenTracker] = None,
        research_metrics: Optional[ResearchMetricsStore] = None,
    ):
        self.document_parser = document_parser or SRSDocumentParser()
        self.token_tracker = token_tracker or TokenTracker.get_instance()
        self.research_metrics = research_metrics or ResearchMetricsStore.get_instance()

    def generate_from_files(
        self,
        *,
        file_paths: List[str],
        output_path: Optional[str] = None,
        workspace_path: str = "",
        progress_callback=None,
    ) -> Dict[str, Any]:
        if not file_paths:
            raise ValueError("No input files provided")

        start_time = time.perf_counter()
        snapshot = self.token_tracker.snapshot()

        combined_text = ""
        srs_docs: List[Dict[str, Any]] = []
        parsed_documents = []
        section_parse_start = time.perf_counter()
        for file_path in file_paths:
            parsed_document = self.document_parser.parse_structured_file(file_path)
            parsed_documents.append(parsed_document)
            raw_text = parsed_document.clean_text
            combined_text += f"\n\n--- Document: {Path(file_path).name} ---\n\n{raw_text}"
            srs_docs.append({"path": file_path, "content": raw_text})
        section_parse_latency_ms = round((time.perf_counter() - section_parse_start) * 1000, 4)

        if not combined_text.strip():
            raise ValueError("All documents are empty or could not be parsed")

        architect = DomainArchitect(progress_callback=progress_callback)
        analyses = architect.analyze_document(
            raw_text=combined_text,
            parsed_documents=parsed_documents,
        )
        final_model: DomainModel = architect.synthesize_final_model(analyses)

        if workspace_path:
            extractor = ASTModelSignalExtractor()
            final_model = extractor.enrich_domain_model(
                final_model,
                workspace_path,
                srs_docs=srs_docs,
            )

        if output_path:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as handle:
                handle.write(final_model.model_dump_json(indent=2))

        metrics_delta = self.token_tracker.delta(snapshot)
        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 4)
        stage_latencies_ms = dict(architect.stage_timings_ms)
        stage_latencies_ms["SectionParser"] = section_parse_latency_ms
        stage_latencies_ms["total"] = total_latency_ms
        generation_result = {
            "model": final_model.model_dump(mode="json"),
            "project_name": final_model.project_name,
            "bounded_contexts_count": len(final_model.bounded_contexts),
            "stage_latencies_ms": stage_latencies_ms,
            "total_latency_ms": total_latency_ms,
            "verification_report": architect.last_verification_report.model_dump(mode="json"),
            "metrics": {
                "provider": getattr(architect.provider, "provider_name", None),
                "model": architect.model_name,
                **metrics_delta,
            },
            "documents": [
                {
                    "path": doc["path"],
                    "chars": len(doc["content"]),
                    "requirements": len(parsed_documents[index].requirements),
                    "sections": len(parsed_documents[index].sections),
                }
                for index, doc in enumerate(srs_docs)
            ],
        }

        self.research_metrics.record_generation_run(
            {
                "project_name": final_model.project_name,
                "bounded_contexts_count": len(final_model.bounded_contexts),
                "stage_latencies_ms": stage_latencies_ms,
                "total_latency_ms": total_latency_ms,
                "verification_report": generation_result["verification_report"],
                "metrics": generation_result["metrics"],
                "documents": generation_result["documents"],
            }
        )
        return generation_result

    def initialize_rag(self, file_paths: List[str]) -> Optional[RAGPipeline]:
        """Create and populate a RAG pipeline from the supplied documents."""
        if not file_paths:
            return None
        rag = RAGPipeline()
        for file_path in file_paths:
            raw_text = self.document_parser.parse_file(file_path)
            if raw_text.strip():
                filename = Path(file_path).name
                extension = Path(file_path).suffix[1:]
                rag.index_document(
                    raw_text=raw_text,
                    doc_id=f"srs_{filename}",
                    doc_name=filename,
                    doc_type=extension,
                )
        return rag
