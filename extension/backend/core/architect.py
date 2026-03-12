"""
Domain Architect

Evidence-backed multi-stage domain model generation pipeline:
0. Deterministic section / requirement parsing
1. Scout - Normalize requirements into domain signals
2. Architect - Identify bounded contexts and ownership
3. Specialist - Extract per-context deep structure
4. Synthesizer - Merge analyses into the final model
5. Verifier - Fail generation if supported fields are dropped
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from pydantic import BaseModel

from config import ArchitectConfig
from core.llm_provider import GeminiLLMProvider, LLMProvider
from core.prompts import (
    build_architect_prompt,
    build_scout_prompt,
    build_specialist_prompt,
    build_synthesizer_prompt,
    build_verifier_prompt,
)
from core.schemas import (
    ActorCandidate,
    Aggregate,
    BoundedContext,
    BusinessRule,
    Capability,
    CapabilityCandidate,
    ConstraintCandidate,
    ContextAnalysis,
    ContextMap,
    DomainActor,
    DomainEvent,
    DomainModel,
    Entity,
    EntityCandidate,
    EvidenceSpan,
    ExternalReference,
    GlobalRules,
    InferenceSource,
    ParsedSection,
    ParsedSRSDocument,
    ProjectMetadata,
    RequirementRecord,
    RequirementSummary,
    ScoutExtraction,
    Service,
    TableCandidate,
    ValueObject,
    ValueObjectCandidate,
    VerificationReport,
)

INTERMEDIATE_DIR = os.path.join(os.path.dirname(__file__), "intermediate")
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


class DomainArchitect:
    """AI-powered domain model extraction from structured SRS inputs."""

    LLMConfig = ArchitectConfig()
    DEFAULT_BANNED_TERMS = ["Manager", "Util", "Helper", "Data", "Info"]
    DEFAULT_ENTITY_ALIASES = {
        "User": ["Account", "Member", "Customer", "Profile"],
        "Product": ["Item", "Listing", "Good", "Merchandise"],
        "Category": ["Type", "Classification", "Group"],
        "Offer": ["Proposal", "Bid", "Quote"],
        "Request": ["Inquiry", "Ask"],
        "Deal": ["Agreement", "Transaction", "Negotiation"],
    }
    CONTEXT_HEURISTICS = {
        "IdentityAndAccess": [
            "register",
            "login",
            "user",
            "profile",
            "password",
            "email",
            "verification",
            "authorized",
        ],
        "ProductCatalog": [
            "product",
            "category",
            "browse",
            "search",
            "detail",
            "catalog",
        ],
        "SellerWorkspace": [
            "addproduct",
            "dashboard",
            "seller",
            "sell",
            "listing",
        ],
        "NegotiationAndCommunication": [
            "offer",
            "request",
            "deal",
            "contact",
            "buyer",
            "seller",
        ],
        "HelpAndSupport": [
            "manual",
            "help",
            "developer",
            "support",
        ],
    }

    def __init__(
        self,
        model: str = LLMConfig.MODEL_NAME,
        progress_callback: ProgressCallback = None,
        provider: Optional[LLMProvider] = None,
    ):
        self.provider = provider or GeminiLLMProvider()
        self.model_name = model
        self.last_request_time = 0.0
        self.min_delay = 6.0
        self.request_count = 0
        self.progress_callback = progress_callback
        self.stage_timings_ms: Dict[str, float] = {}
        self.last_run_summary: Dict[str, Any] = {}
        self.last_parsed_documents: List[ParsedSRSDocument] = []
        self.last_scout_summary = ScoutExtraction()
        self.last_context_map = ContextMap()
        self.last_context_analyses: List[ContextAnalysis] = []
        self.last_verification_report = VerificationReport(passed=False, notes=["Verifier not run yet"])

        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")

        print("\n" + "=" * 70)
        print("DOMAIN ARCHITECT INITIALIZED")
        print("=" * 70)
        print(f"  Model: {model}")
        print(f"  Rate Limit: {self.min_delay}s between requests")
        print("=" * 70 + "\n")

    def _report_progress(self, stage: str, status: str, detail: str = "", progress: int = 0) -> None:
        if self.progress_callback:
            self.progress_callback(
                {
                    "stage": stage,
                    "status": status,
                    "detail": detail,
                    "progress": progress,
                }
            )

    def _mark_stage_timing(self, stage: str, start_time: float) -> None:
        self.stage_timings_ms[stage] = round((time.perf_counter() - start_time) * 1000, 4)

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            print(f"  Waiting {sleep_time:.1f}s for rate limit")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
        print(f"  API Request #{self.request_count}")

    def _handle_quota_error(self, error: Exception, retry_count: int) -> float:
        error_str = str(error)
        is_quota_error = (
            "429" in error_str
            or "quota" in error_str.lower()
            or "ResourceExhausted" in str(type(error))
        )
        if not is_quota_error:
            return 0

        retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
        if retry_match:
            wait_time = max(float(retry_match.group(1)), 10)
        else:
            wait_time = min(15 * (2**retry_count), 300)

        print(f"  Quota exceeded, backing off {wait_time:.1f}s")
        time.sleep(wait_time)
        return wait_time

    def analyze_document(
        self,
        raw_text: str,
        parsed_documents: Optional[List[ParsedSRSDocument]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full generation pipeline up to Specialist outputs."""
        total_start = time.perf_counter()
        self.stage_timings_ms = {}
        self.last_verification_report = VerificationReport(passed=False, notes=["Verifier not run yet"])

        print("\n" + "#" * 70)
        print("# DOMAIN MODEL GENERATION PIPELINE STARTED".center(70))
        print("#" * 70)
        print(f"  Input size: {len(raw_text):,} characters")

        try:
            self.last_parsed_documents = parsed_documents or self._build_fallback_documents(raw_text)
            scout_summary = self.extract_requirement_signals(self.last_parsed_documents)
            context_map = self.identify_contexts(scout_summary)
            analyses = self.extract_all_contexts_details(
                context_map=context_map,
                scout_summary=scout_summary,
                parsed_documents=self.last_parsed_documents,
            )

            self.last_scout_summary = scout_summary
            self.last_context_map = context_map
            self.last_context_analyses = analyses
            self.last_run_summary = {
                "request_count": self.request_count,
                "stage_latencies_ms": dict(self.stage_timings_ms),
                "pipeline_total_latency_ms": round((time.perf_counter() - total_start) * 1000, 4),
                "contexts": [analysis.context for analysis in analyses],
            }
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 70)
            return [analysis.model_dump(mode="json") for analysis in analyses]
        except Exception as exc:
            self.last_run_summary = {
                "request_count": self.request_count,
                "stage_latencies_ms": dict(self.stage_timings_ms),
                "pipeline_total_latency_ms": round((time.perf_counter() - total_start) * 1000, 4),
                "error": str(exc),
            }
            print("\n" + "=" * 70)
            print("PIPELINE FAILED")
            print("=" * 70)
            print(f"  Error: {exc}")
            raise

    def extract_requirement_signals(self, parsed_documents: List[ParsedSRSDocument]) -> ScoutExtraction:
        """Stage 1: normalize parsed requirements into domain signals."""
        stage_start = time.perf_counter()
        try:
            print("\n[Stage 1] Scout - Requirement normalization")
            self._report_progress("Scout", "started", "Normalizing parsed requirements", 0)

            deterministic = self._build_deterministic_scout(parsed_documents)
            requirement_rows = [record.model_dump(mode="json") for record in self._iter_requirements(parsed_documents)]
            chunks = self._chunk_requirement_rows(requirement_rows)

            llm_outputs: List[ScoutExtraction] = []
            for index, chunk in enumerate(chunks, start=1):
                self._report_progress("Scout", "in_progress", f"Processing chunk {index}/{len(chunks)}", int(index / max(len(chunks), 1) * 100))
                prompt = build_scout_prompt(index, len(chunks), chunk)
                result = self._call_stage(
                    stage="Scout",
                    operation=f"normalize_requirements_chunk_{index}",
                    prompt=prompt,
                    response_schema=ScoutExtraction,
                )
                if result is not None:
                    llm_outputs.append(result)

            merged = self._merge_scout_extractions([deterministic, *llm_outputs])
            self._save_intermediate(
                stage="1_scout",
                data={
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "chunks_processed": len(chunks),
                    "requirements": len(merged.requirements),
                    "actors": len(merged.actors),
                    "entities": len(merged.entities),
                    "capabilities": len(merged.capabilities),
                    "constraints": len(merged.constraints),
                    "tables": len(merged.tables),
                    "scout_summary": merged.model_dump(mode="json"),
                },
            )
            self._report_progress("Scout", "completed", f"Normalized {len(merged.requirements)} requirements", 100)
            return merged
        finally:
            self._mark_stage_timing("Scout", stage_start)

    def identify_contexts(self, scout_summary: ScoutExtraction) -> ContextMap:
        """Stage 2: identify bounded contexts and ownership boundaries."""
        stage_start = time.perf_counter()
        try:
            print("\n[Stage 2] Architect - Context mapping")
            self._report_progress("Architect", "started", "Mapping bounded contexts", 0)

            fallback = self._build_fallback_context_map(scout_summary)
            prompt = build_architect_prompt(scout_summary.model_dump(mode="json"))
            llm_map = self._call_stage(
                stage="Architect",
                operation="identify_contexts",
                prompt=prompt,
                response_schema=ContextMap,
            )
            merged = self._merge_context_maps(fallback, llm_map or ContextMap())
            self._save_intermediate(
                stage="2_architect",
                data={
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "contexts_identified": len(merged.contexts),
                    "contexts": merged.model_dump(mode="json"),
                },
            )
            self._report_progress("Architect", "completed", f"Found {len(merged.contexts)} contexts", 100)
            return merged
        finally:
            self._mark_stage_timing("Architect", stage_start)

    def extract_all_contexts_details(
        self,
        *,
        context_map: ContextMap,
        scout_summary: ScoutExtraction,
        parsed_documents: List[ParsedSRSDocument],
    ) -> List[ContextAnalysis]:
        """Stage 3: extract deep per-context structure."""
        stage_start = time.perf_counter()
        analyses: List[ContextAnalysis] = []
        try:
            print("\n[Stage 3] Specialist - Context detail extraction")
            self._report_progress("Specialist", "started", f"Analyzing {len(context_map.contexts)} contexts", 0)
            for index, proposal in enumerate(context_map.contexts, start=1):
                relevant_requirements = self._select_requirements_for_context(
                    context_name=proposal.context_name,
                    context_proposal=proposal,
                    scout_summary=scout_summary,
                    parsed_documents=parsed_documents,
                )
                fallback = self._build_fallback_context_analysis(
                    context_proposal=proposal,
                    relevant_requirements=relevant_requirements,
                    scout_summary=scout_summary,
                )
                prompt = build_specialist_prompt(
                    context_name=proposal.context_name,
                    context_proposal=proposal.model_dump(mode="json"),
                    relevant_requirements=relevant_requirements,
                    scout_summary=scout_summary.model_dump(mode="json"),
                )
                llm_analysis = self._call_stage(
                    stage="Specialist",
                    operation=f"analyze_context_{proposal.context_name}",
                    prompt=prompt,
                    response_schema=ContextAnalysis,
                )
                merged = self._merge_context_analysis(fallback, llm_analysis or ContextAnalysis(context=proposal.context_name, description=proposal.description))
                analyses.append(merged)
                self._report_progress(
                    "Specialist",
                    "in_progress",
                    f"Analyzed {proposal.context_name}",
                    int(index / max(len(context_map.contexts), 1) * 100),
                )

            self._save_intermediate(
                stage="3_specialist",
                data={
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "contexts_analyzed": len(analyses),
                    "analyses": [analysis.model_dump(mode="json") for analysis in analyses],
                },
            )
            self._report_progress("Specialist", "completed", f"Analyzed {len(analyses)} contexts", 100)
            return analyses
        finally:
            self._mark_stage_timing("Specialist", stage_start)

    def synthesize(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 4: synthesize final model without dropping Specialist fields."""
        stage_start = time.perf_counter()
        try:
            print("\n[Stage 4] Synthesizer - Final model merge")
            self._report_progress("Synthesizer", "started", "Merging context analyses", 0)

            fallback_model = self._build_deterministic_domain_model(self.last_context_analyses)
            prompt = build_synthesizer_prompt(
                context_analyses=analyses,
                scout_summary=self.last_scout_summary.model_dump(mode="json"),
                glossary_aliases=self._build_glossary_aliases(self.last_parsed_documents),
                banned_terms=self.DEFAULT_BANNED_TERMS,
            )
            llm_model = self._call_stage(
                stage="Synthesizer",
                operation="synthesize_final_model",
                prompt=prompt,
                response_schema=DomainModel,
            )
            merged = self._merge_domain_models(
                fallback_model,
                llm_model.model_dump(mode="json") if llm_model is not None else {},
            )
            merged = self._cleanup_domain_data(merged)
            merged = self._resolve_cross_context_ownership(merged)
            self._save_intermediate(
                stage="4_synthesizer",
                data={
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "final_model": merged,
                },
            )
            self._report_progress("Synthesizer", "completed", "Domain model created", 100)
            return merged
        finally:
            self._mark_stage_timing("Synthesizer", stage_start)

    def synthesize_final_model(self, analyses: List[Dict[str, Any]]) -> DomainModel:
        """Create the final verified DomainModel."""
        raw_model = self.synthesize(analyses)
        hydrated_model = self._apply_metadata_and_evidence(raw_model)
        model = DomainModel(**hydrated_model)
        verification = self.verify_model(model)
        self.last_verification_report = verification
        self._save_intermediate(
            stage="5_verifier",
            data={
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "verification_report": verification.model_dump(mode="json"),
            },
        )
        self._report_progress("Verifier", "completed", "Verification completed", 100)
        self.stage_timings_ms["Verifier"] = self.stage_timings_ms.get("Verifier", 0.0)
        if not verification.passed:
            raise ValueError(f"Verifier rejected generated domain model: {verification.model_dump_json()}")
        return model

    def verify_model(self, model: DomainModel) -> VerificationReport:
        """Stage 5: deterministic coverage and consistency verification."""
        stage_start = time.perf_counter()
        try:
            requirement_records = list(self._iter_requirements(self.last_parsed_documents))
            model_data = model.model_dump(mode="json")
            llm_report = self._run_llm_verifier(requirement_records, model_data)
            final_contexts = {ctx["context_name"]: ctx for ctx in model_data.get("bounded_contexts", [])}
            covered_evidence_ids = self._collect_model_evidence_ids(model_data)

            missing_requirement_ids = [
                record.requirement_id
                for record in requirement_records
                if not self._requirement_is_represented(
                    record=record,
                    model_data=model_data,
                    covered_evidence_ids=covered_evidence_ids,
                )
            ]

            model_actor_names = {
                actor["name"]
                for context in model_data.get("bounded_contexts", [])
                for actor in context.get("actors", [])
            }
            model_capability_names = {
                capability["name"]
                for context in model_data.get("bounded_contexts", [])
                for capability in context.get("capabilities", [])
            }
            model_entity_names = {
                item["name"]
                for context in model_data.get("bounded_contexts", [])
                for bucket in (
                    context.get("ubiquitous_language", {}).get("entities", []),
                    context.get("ubiquitous_language", {}).get("value_objects", []),
                    context.get("ubiquitous_language", {}).get("aggregates", []),
                    context.get("ubiquitous_language", {}).get("services", []),
                    context.get("ubiquitous_language", {}).get("domain_events", []),
                )
                for item in bucket
            }

            uncovered_actors = sorted(
                actor.name
                for actor in self.last_scout_summary.actors
                if actor.name not in model_actor_names
            )
            uncovered_capabilities = sorted(
                capability.name
                for capability in self.last_scout_summary.capabilities
                if capability.name not in model_capability_names
            )
            uncovered_entities = sorted(
                entity.name
                for entity in self.last_scout_summary.entities
                if entity.name not in model_entity_names
            )

            evidence_less_items: List[str] = []
            for context in model_data.get("bounded_contexts", []):
                if not context.get("evidence_ids") and not context.get("evidence"):
                    evidence_less_items.append(f"context:{context['context_name']}")
                for actor in context.get("actors", []):
                    if not actor.get("evidence_ids") and not actor.get("sources"):
                        evidence_less_items.append(f"actor:{context['context_name']}:{actor['name']}")
                for capability in context.get("capabilities", []):
                    if not capability.get("evidence_ids") and not capability.get("sources"):
                        evidence_less_items.append(f"capability:{context['context_name']}:{capability['name']}")
                for rule in context.get("business_rules", []):
                    if not rule.get("evidence_ids") and not rule.get("sources"):
                        evidence_less_items.append(f"business_rule:{context['context_name']}:{rule['text']}")
                ul = context.get("ubiquitous_language", {})
                for bucket_name in ("entities", "value_objects", "aggregates", "services", "domain_events"):
                    for item in ul.get(bucket_name, []):
                        if not item.get("evidence_ids") and not item.get("sources"):
                            evidence_less_items.append(f"{bucket_name}:{context['context_name']}:{item['name']}")

            duplicate_tracker: Dict[str, List[str]] = defaultdict(list)
            for context in model_data.get("bounded_contexts", []):
                for entity in context.get("ubiquitous_language", {}).get("entities", []):
                    duplicate_tracker[entity["name"]].append(context["context_name"])
            duplicate_entities = sorted(
                f"{name}:{','.join(contexts)}"
                for name, contexts in duplicate_tracker.items()
                if len(contexts) > 1
            )

            missing_fields: List[str] = []
            for analysis in self.last_context_analyses:
                final_context = final_contexts.get(analysis.context)
                if final_context is None:
                    missing_fields.append(f"context:{analysis.context}")
                    continue
                missing_fields.extend(
                    self._compare_analysis_to_context(analysis, final_context)
                )

            contradictions: List[str] = []
            known_context_names = set(final_contexts.keys())
            for context in model_data.get("bounded_contexts", []):
                for dependency in context.get("allowed_dependencies", []):
                    if dependency not in known_context_names:
                        contradictions.append(
                            f"unknown_dependency:{context['context_name']}->{dependency}"
                        )

            notes: List[str] = []
            if not self.last_parsed_documents:
                notes.append("Verifier ran without structured parsed documents.")
            if llm_report is not None:
                notes.extend(
                    f"llm_verifier:{note}"
                    for note in llm_report.notes
                    if note
                )
                for contradiction in llm_report.contradictions:
                    if contradiction not in contradictions:
                        contradictions.append(contradiction)
                for duplicate in llm_report.duplicate_entities:
                    if duplicate not in duplicate_entities:
                        duplicate_entities.append(duplicate)
                for missing_field in llm_report.missing_fields:
                    if missing_field not in missing_fields:
                        missing_fields.append(missing_field)

            passed = not any(
                [
                    missing_requirement_ids,
                    uncovered_capabilities,
                    uncovered_actors,
                    uncovered_entities,
                    evidence_less_items,
                    duplicate_entities,
                    missing_fields,
                    contradictions,
                ]
            )

            return VerificationReport(
                passed=passed,
                missing_requirement_ids=missing_requirement_ids,
                uncovered_capabilities=uncovered_capabilities,
                uncovered_actors=uncovered_actors,
                uncovered_entities=uncovered_entities,
                evidence_less_items=evidence_less_items,
                duplicate_entities=duplicate_entities,
                missing_fields=missing_fields,
                contradictions=contradictions,
                notes=notes,
            )
        finally:
            self._mark_stage_timing("Verifier", stage_start)

    def evaluate_acceptance_coverage(
        self,
        model: DomainModel,
        acceptance_spec: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Compute deterministic coverage against a manually curated acceptance set."""
        model_data = model.model_dump(mode="json")
        actor_names = {
            actor.get("name", "")
            for context in model_data.get("bounded_contexts", [])
            for actor in context.get("actors", [])
        }
        capability_names = {
            capability.get("name", "")
            for context in model_data.get("bounded_contexts", [])
            for capability in context.get("capabilities", [])
        }
        concept_names = {
            item.get("name", "")
            for context in model_data.get("bounded_contexts", [])
            for bucket in (
                context.get("ubiquitous_language", {}).get("entities", []),
                context.get("ubiquitous_language", {}).get("value_objects", []),
                context.get("ubiquitous_language", {}).get("aggregates", []),
                context.get("ubiquitous_language", {}).get("services", []),
                context.get("ubiquitous_language", {}).get("domain_events", []),
            )
            for item in bucket
        }
        constraint_texts = {
            self._normalize_text(rule.get("text", ""))
            for context in model_data.get("bounded_contexts", [])
            for rule in context.get("business_rules", [])
        }
        constraint_texts.update(
            self._normalize_text(text)
            for text in (model_data.get("global_rules") or {}).get("cross_cutting_constraints", [])
        )

        report = {
            "actors": self._coverage_bucket(acceptance_spec.get("actors", []), actor_names),
            "entities": self._coverage_bucket(acceptance_spec.get("entities", []), concept_names),
            "capabilities": self._coverage_bucket(acceptance_spec.get("capabilities", []), capability_names),
            "constraints": self._coverage_bucket(
                acceptance_spec.get("constraints", []),
                constraint_texts,
                normalize=self._normalize_text,
            ),
        }
        covered = sum(bucket["covered_count"] for bucket in report.values())
        expected = sum(bucket["expected_count"] for bucket in report.values())
        report["overall"] = {
            "covered_count": covered,
            "expected_count": expected,
            "coverage_percent": round((covered / expected) * 100, 2) if expected else 100.0,
        }
        return report

    def _call_stage(
        self,
        *,
        stage: str,
        operation: str,
        prompt: str,
        response_schema: Type[BaseModel],
    ) -> Optional[BaseModel]:
        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.provider.generate_json(
                    model=self.model_name,
                    prompt=prompt,
                    stage=stage,
                    operation=operation,
                    response_schema=response_schema,
                    response_mime_type="application/json",
                    temperature=0.05,
                    seed=42,
                    retry_count=retry,
                )
                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue
                if response.parse_success and response.parsed is not None:
                    return response.parsed
                parsed = self._parse_json_response(response.text)
                if isinstance(parsed, dict) and parsed.get("error") == "json_parse_failed":
                    if retry < 4:
                        time.sleep(2)
                        continue
                    return None
                return response_schema.model_validate(parsed)
            except Exception as exc:
                if self._handle_quota_error(exc, retry) == 0:
                    print(f"  [WARN] {stage} {operation} failed: {exc}")
                    if retry >= 4:
                        return None
        return None

    def _iter_requirements(self, parsed_documents: Sequence[ParsedSRSDocument]) -> List[RequirementRecord]:
        return [record for document in parsed_documents for record in document.requirements]

    def _build_fallback_documents(self, raw_text: str) -> List[ParsedSRSDocument]:
        evidence = EvidenceSpan(
            evidence_id="E0001",
            document="inline.txt",
            section="Document",
            excerpt=raw_text[:500],
            line=1,
            requirement_id="REQ-001",
        )
        document = ParsedSRSDocument(
            file_path="inline.txt",
            document_name="inline.txt",
            clean_text=raw_text,
            sections=[
                ParsedSection(
                    section_id="SEC-001",
                    heading="Document",
                    category="general",
                    content=raw_text,
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            requirements=[
                RequirementRecord(
                    requirement_id="REQ-001",
                    category="general",
                    title="InlineDocument",
                    description=raw_text[:500],
                    actor=None,
                    section="Document",
                    evidence_ids=[evidence.evidence_id],
                )
            ],
            evidence_spans=[evidence],
        )
        return [document]

    def _chunk_requirement_rows(
        self,
        requirement_rows: List[Dict[str, Any]],
        max_chars: int = 9000,
    ) -> List[List[Dict[str, Any]]]:
        if not requirement_rows:
            return [[]]
        chunks: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_len = 0
        for row in requirement_rows:
            row_len = len(json.dumps(row, ensure_ascii=False))
            if current and current_len + row_len > max_chars:
                chunks.append(current)
                current = []
                current_len = 0
            current.append(row)
            current_len += row_len
        if current:
            chunks.append(current)
        return chunks

    def _build_deterministic_scout(self, parsed_documents: List[ParsedSRSDocument]) -> ScoutExtraction:
        requirements = []
        actors: Dict[str, ActorCandidate] = {}
        capabilities: Dict[tuple[str, Optional[str]], CapabilityCandidate] = {}
        constraints: Dict[str, ConstraintCandidate] = {}
        tables: Dict[str, TableCandidate] = {}
        entities: Dict[str, EntityCandidate] = {}
        evidence_spans: Dict[str, EvidenceSpan] = {}

        for document in parsed_documents:
            for span in document.evidence_spans:
                evidence_spans[span.evidence_id] = span
            for record in document.requirements:
                requirements.append(
                    RequirementSummary(
                        requirement_id=record.requirement_id,
                        title=record.title,
                        category=record.category,
                        description=record.description,
                        actor=record.actor,
                        evidence_ids=list(record.evidence_ids),
                    )
                )

                if record.actor:
                    actors.setdefault(
                        record.actor,
                        ActorCandidate(
                            name=record.actor,
                            description=f"Actor extracted from {record.section}.",
                            evidence_ids=list(record.evidence_ids),
                        ),
                    )

                if record.category in {"functional", "product_function"}:
                    capability_name = self._capability_name_from_title(record.title)
                    capabilities.setdefault(
                        (capability_name, record.actor),
                        CapabilityCandidate(
                            name=capability_name,
                            description=record.description,
                            actor=record.actor,
                            evidence_ids=list(record.evidence_ids),
                        ),
                    )

                if record.category in {"security", "performance", "safety", "non_functional"}:
                    constraints.setdefault(
                        record.description,
                        ConstraintCandidate(
                            text=record.description,
                            category=record.category,
                            evidence_ids=list(record.evidence_ids),
                        ),
                    )

                if record.category == "database":
                    tables.setdefault(
                        record.title,
                        TableCandidate(
                            name=self._normalize_term(record.title, singular=False),
                            description=record.description,
                            evidence_ids=list(record.evidence_ids),
                        ),
                    )
                    entities.setdefault(
                        self._normalize_term(record.title),
                        EntityCandidate(
                            name=self._normalize_term(record.title),
                            description=record.description,
                            evidence_ids=list(record.evidence_ids),
                        ),
                    )

                for extracted in self._extract_entity_candidates_from_record(record):
                    if extracted.name not in entities:
                        entities[extracted.name] = extracted

        return ScoutExtraction(
            requirements=requirements,
            actors=list(actors.values()),
            entities=list(entities.values()),
            constraints=list(constraints.values()),
            tables=list(tables.values()),
            capabilities=list(capabilities.values()),
            evidence_spans=list(evidence_spans.values()),
        )

    def _extract_entity_candidates_from_record(self, record: RequirementRecord) -> List[EntityCandidate]:
        candidates: List[EntityCandidate] = []
        keyword_map = {
            "user": "User",
            "product": "Product",
            "category": "Category",
            "offer": "Offer",
            "request": "Request",
            "dashboard": "Dashboard",
            "profile": "Profile",
            "seller": "Seller",
            "buyer": "Buyer",
            "manual": "UserManual",
            "contact": "ContactInformation",
            "price": "Price",
        }
        text = f"{record.title} {record.description}".lower()
        for keyword, normalized in keyword_map.items():
            if keyword in text:
                candidates.append(
                    EntityCandidate(
                        name=normalized,
                        description=f"Inferred from requirement: {record.title}",
                        evidence_ids=list(record.evidence_ids),
                    )
                )
        return candidates

    def _merge_scout_extractions(self, extractions: List[ScoutExtraction]) -> ScoutExtraction:
        requirements: Dict[str, RequirementSummary] = {}
        actors: Dict[str, ActorCandidate] = {}
        entities: Dict[str, EntityCandidate] = {}
        constraints: Dict[str, ConstraintCandidate] = {}
        tables: Dict[str, TableCandidate] = {}
        capabilities: Dict[tuple[str, Optional[str]], CapabilityCandidate] = {}
        evidence_spans: Dict[str, EvidenceSpan] = {}

        for extraction in extractions:
            for record in extraction.requirements:
                existing = requirements.get(record.requirement_id)
                if existing is None:
                    requirements[record.requirement_id] = record
                else:
                    existing.evidence_ids = sorted(set(existing.evidence_ids + record.evidence_ids))

            for actor in extraction.actors:
                existing = actors.get(actor.name)
                if existing is None:
                    actors[actor.name] = actor
                else:
                    existing.evidence_ids = sorted(set(existing.evidence_ids + actor.evidence_ids))
                    if len(actor.description) > len(existing.description):
                        existing.description = actor.description

            for entity in extraction.entities:
                existing = entities.get(entity.name)
                if existing is None:
                    entities[entity.name] = entity
                else:
                    existing.evidence_ids = sorted(set(existing.evidence_ids + entity.evidence_ids))
                    if len(entity.description) > len(existing.description):
                        existing.description = entity.description

            for constraint in extraction.constraints:
                existing = constraints.get(constraint.text)
                if existing is None:
                    constraints[constraint.text] = constraint
                else:
                    existing.evidence_ids = sorted(set(existing.evidence_ids + constraint.evidence_ids))

            for table in extraction.tables:
                existing = tables.get(table.name)
                if existing is None:
                    tables[table.name] = table
                else:
                    existing.evidence_ids = sorted(set(existing.evidence_ids + table.evidence_ids))
                    if len(table.description) > len(existing.description):
                        existing.description = table.description

            for capability in extraction.capabilities:
                key = (capability.name, capability.actor)
                existing = capabilities.get(key)
                if existing is None:
                    capabilities[key] = capability
                else:
                    existing.evidence_ids = sorted(set(existing.evidence_ids + capability.evidence_ids))
                    if len(capability.description) > len(existing.description):
                        existing.description = capability.description

            for span in extraction.evidence_spans:
                evidence_spans.setdefault(span.evidence_id, span)

        return ScoutExtraction(
            requirements=list(requirements.values()),
            actors=list(actors.values()),
            entities=list(entities.values()),
            constraints=list(constraints.values()),
            tables=list(tables.values()),
            capabilities=list(capabilities.values()),
            evidence_spans=list(evidence_spans.values()),
        )

    def _build_fallback_context_map(self, scout_summary: ScoutExtraction) -> ContextMap:
        contexts: Dict[str, Dict[str, Any]] = {}

        def ensure_context(context_name: str) -> Dict[str, Any]:
            if context_name not in contexts:
                contexts[context_name] = {
                    "context_name": context_name,
                    "description": f"Context inferred for {context_name}.",
                    "ownership_rationale": f"Deterministic fallback inferred ownership for {context_name}.",
                    "included_capabilities": [],
                    "excluded_capabilities": [],
                    "primary_entities": [],
                    "allowed_dependencies": [],
                    "evidence_ids": [],
                }
            return contexts[context_name]

        for capability in scout_summary.capabilities:
            assigned = False
            capability_key = capability.name.lower()
            combined = f"{capability.name} {capability.description} {capability.actor or ''}".lower()
            for context_name, keywords in self.CONTEXT_HEURISTICS.items():
                if any(keyword in capability_key or keyword in combined for keyword in keywords):
                    ctx = ensure_context(context_name)
                    ctx["included_capabilities"].append(capability.name)
                    ctx["evidence_ids"].extend(capability.evidence_ids)
                    assigned = True
                    break
            if not assigned:
                ctx = ensure_context("CoreDomain")
                ctx["included_capabilities"].append(capability.name)
                ctx["evidence_ids"].extend(capability.evidence_ids)

        for entity in scout_summary.entities:
            assigned = False
            combined = f"{entity.name} {entity.description}".lower()
            for context_name, keywords in self.CONTEXT_HEURISTICS.items():
                if any(keyword in combined for keyword in keywords):
                    ctx = ensure_context(context_name)
                    ctx["primary_entities"].append(entity.name)
                    ctx["evidence_ids"].extend(entity.evidence_ids)
                    assigned = True
                    break
            if not assigned:
                ctx = ensure_context("CoreDomain")
                ctx["primary_entities"].append(entity.name)
                ctx["evidence_ids"].extend(entity.evidence_ids)

        dependency_defaults = {
            "SellerWorkspace": ["IdentityAndAccess", "ProductCatalog"],
            "NegotiationAndCommunication": ["IdentityAndAccess", "ProductCatalog"],
            "HelpAndSupport": ["IdentityAndAccess"],
        }
        ordered_contexts = []
        for context_name, payload in contexts.items():
            payload["included_capabilities"] = sorted(set(payload["included_capabilities"]))
            payload["primary_entities"] = sorted(set(payload["primary_entities"]))
            payload["allowed_dependencies"] = dependency_defaults.get(context_name, [])
            payload["evidence_ids"] = sorted(set(payload["evidence_ids"]))
            ordered_contexts.append(payload)

        if not ordered_contexts:
            ordered_contexts.append(
                {
                    "context_name": "CoreDomain",
                    "description": "Fallback context with unresolved ownership.",
                    "ownership_rationale": "No strong context partition was available.",
                    "included_capabilities": [cap.name for cap in scout_summary.capabilities],
                    "excluded_capabilities": [],
                    "primary_entities": [entity.name for entity in scout_summary.entities],
                    "allowed_dependencies": [],
                    "evidence_ids": [span.evidence_id for span in scout_summary.evidence_spans[:10]],
                }
            )

        return ContextMap.model_validate({"contexts": ordered_contexts})

    def _merge_context_maps(self, fallback: ContextMap, llm_map: ContextMap) -> ContextMap:
        merged: Dict[str, Dict[str, Any]] = {
            context.context_name: context.model_dump(mode="json")
            for context in fallback.contexts
        }
        for context in llm_map.contexts:
            current = merged.setdefault(context.context_name, context.model_dump(mode="json"))
            current["description"] = context.description or current.get("description")
            current["ownership_rationale"] = context.ownership_rationale or current.get("ownership_rationale")
            current["included_capabilities"] = sorted(
                set(current.get("included_capabilities", []) + context.included_capabilities)
            )
            current["excluded_capabilities"] = sorted(
                set(current.get("excluded_capabilities", []) + context.excluded_capabilities)
            )
            current["primary_entities"] = sorted(
                set(current.get("primary_entities", []) + context.primary_entities)
            )
            current["allowed_dependencies"] = sorted(
                set(current.get("allowed_dependencies", []) + context.allowed_dependencies)
            )
            current["evidence_ids"] = sorted(
                set(current.get("evidence_ids", []) + context.evidence_ids)
            )
        return ContextMap.model_validate({"contexts": list(merged.values())})

    def _select_requirements_for_context(
        self,
        *,
        context_name: str,
        context_proposal,
        scout_summary: ScoutExtraction,
        parsed_documents: List[ParsedSRSDocument],
    ) -> List[Dict[str, Any]]:
        keywords = {
            token.lower()
            for token in (
                context_name,
                *context_proposal.primary_entities,
                *context_proposal.included_capabilities,
                *(context_name.split("And")),
            )
            if token
        }
        keywords.update(
            token.lower()
            for token in re.findall(r"[A-Z][a-z]*", context_name)
            if token
        )

        selected: List[RequirementRecord] = []
        for record in self._iter_requirements(parsed_documents):
            haystack = f"{record.title} {record.description} {record.actor or ''}".lower()
            if set(record.evidence_ids).intersection(context_proposal.evidence_ids):
                selected.append(record)

        if not selected:
            for record in self._iter_requirements(parsed_documents):
                haystack = f"{record.title} {record.description} {record.actor or ''}".lower()
                if any(keyword and keyword in haystack for keyword in keywords):
                    selected.append(record)
                    continue
                for capability in scout_summary.capabilities:
                    if capability.name in context_proposal.included_capabilities and set(capability.evidence_ids).intersection(record.evidence_ids):
                        selected.append(record)
                        break

        if not selected:
            selected = [
                record
                for record in self._iter_requirements(parsed_documents)
                if set(record.evidence_ids).intersection(context_proposal.evidence_ids)
            ]

        return [record.model_dump(mode="json") for record in selected]

    def _build_fallback_context_analysis(
        self,
        *,
        context_proposal,
        relevant_requirements: List[Dict[str, Any]],
        scout_summary: ScoutExtraction,
    ) -> ContextAnalysis:
        relevant_evidence = {
            evidence_id
            for requirement in relevant_requirements
            for evidence_id in requirement.get("evidence_ids", [])
        }
        relevant_actor_names = {
            requirement.get("actor")
            for requirement in relevant_requirements
            if requirement.get("actor")
        }

        actors = [
            actor
            for actor in scout_summary.actors
            if actor.name in relevant_actor_names
        ]
        if not actors:
            actors = [
                ActorCandidate(
                    name=actor_name,
                    description=f"Actor participating in {context_proposal.context_name}.",
                    evidence_ids=sorted(relevant_evidence),
                )
                for actor_name in sorted(relevant_actor_names)
            ]
        capabilities = [
            capability
            for capability in scout_summary.capabilities
            if capability.name in context_proposal.included_capabilities
        ]
        entities = [
            entity
            for entity in scout_summary.entities
            if entity.name in context_proposal.primary_entities
        ]
        if not entities:
            entities = [
                entity
                for entity in scout_summary.entities
                if set(entity.evidence_ids).intersection(relevant_evidence)
            ]
        constraints = [
            constraint
            for constraint in scout_summary.constraints
            if set(constraint.evidence_ids).intersection(relevant_evidence)
        ]

        value_objects = self._infer_value_objects_from_requirements(relevant_requirements)
        domain_events = self._infer_domain_events_from_capabilities(capabilities)

        aggregate_roots = [
            EntityCandidate(
                name=entity.name,
                description=f"Aggregate root candidate for {context_proposal.context_name}.",
                evidence_ids=list(entity.evidence_ids),
            )
            for entity in entities[:2]
        ]
        domain_services = [
            EntityCandidate(
                name=f"{context_proposal.context_name}Service",
                description=f"Coordinating domain service for {context_proposal.context_name}.",
                evidence_ids=list(capabilities[0].evidence_ids if capabilities else context_proposal.evidence_ids),
            )
        ] if len(capabilities) >= 2 else []

        return ContextAnalysis(
            context=context_proposal.context_name,
            description=context_proposal.description,
            actors=actors,
            capabilities=capabilities,
            aggregate_roots=aggregate_roots,
            entities=entities,
            value_objects=value_objects,
            business_rules=constraints,
            domain_events=domain_events,
            domain_services=domain_services,
            allowed_dependencies=context_proposal.allowed_dependencies,
            evidence_ids=sorted(relevant_evidence or set(context_proposal.evidence_ids)),
        )

    def _infer_value_objects_from_requirements(
        self,
        relevant_requirements: List[Dict[str, Any]],
    ) -> List[ValueObjectCandidate]:
        definitions: Dict[str, ValueObjectCandidate] = {}
        for requirement in relevant_requirements:
            text = f"{requirement.get('title', '')} {requirement.get('description', '')}".lower()
            evidence_ids = requirement.get("evidence_ids", [])
            if "price" in text:
                definitions.setdefault(
                    "Price",
                    ValueObjectCandidate(
                        name="Price",
                        description="Monetary amount associated with a product or offer.",
                        attributes=["amount", "currency"],
                        evidence_ids=list(evidence_ids),
                    ),
                )
            if "password" in text:
                definitions.setdefault(
                    "PasswordHash",
                    ValueObjectCandidate(
                        name="PasswordHash",
                        description="Protected representation of a password.",
                        attributes=["hashValue", "salt"],
                        evidence_ids=list(evidence_ids),
                    ),
                )
            if "email" in text or "contact number" in text or "contact information" in text:
                definitions.setdefault(
                    "ContactInformation",
                    ValueObjectCandidate(
                        name="ContactInformation",
                        description="Contact channels associated with a user or listing.",
                        attributes=["email", "contactNumber"],
                        evidence_ids=list(evidence_ids),
                    ),
                )
            if "warranty" in text or "delivery" in text or "condition" in text:
                definitions.setdefault(
                    "ProductDetails",
                    ValueObjectCandidate(
                        name="ProductDetails",
                        description="Detailed sellable-product attributes.",
                        attributes=["condition", "warranty", "homeDelivery"],
                        evidence_ids=list(evidence_ids),
                    ),
                )
        return list(definitions.values())

    def _infer_domain_events_from_capabilities(
        self,
        capabilities: List[CapabilityCandidate],
    ) -> List[EntityCandidate]:
        events: Dict[str, EntityCandidate] = {}
        for capability in capabilities:
            name = capability.name.lower()
            if "register" in name:
                events["UserRegistered"] = EntityCandidate(
                    name="UserRegistered",
                    description="A user registration flow completed.",
                    evidence_ids=list(capability.evidence_ids),
                )
            elif "login" in name:
                events["UserLoggedIn"] = EntityCandidate(
                    name="UserLoggedIn",
                    description="A user successfully authenticated.",
                    evidence_ids=list(capability.evidence_ids),
                )
            elif "addproduct" in name or "product" in name and "add" in name:
                events["ProductListed"] = EntityCandidate(
                    name="ProductListed",
                    description="A product was listed for discovery.",
                    evidence_ids=list(capability.evidence_ids),
                )
            elif "offer" in name or "request" in name:
                events["OfferSubmitted"] = EntityCandidate(
                    name="OfferSubmitted",
                    description="A negotiation offer or request was submitted.",
                    evidence_ids=list(capability.evidence_ids),
                )
        return list(events.values())

    def _merge_context_analysis(
        self,
        fallback: ContextAnalysis,
        llm_analysis: ContextAnalysis,
    ) -> ContextAnalysis:
        payload = fallback.model_dump(mode="json")
        llm_payload = llm_analysis.model_dump(mode="json")
        payload["description"] = llm_payload.get("description") or payload["description"]
        payload["allowed_dependencies"] = sorted(
            set(payload.get("allowed_dependencies", []) + llm_payload.get("allowed_dependencies", []))
        )
        payload["evidence_ids"] = sorted(
            set(payload.get("evidence_ids", []) + llm_payload.get("evidence_ids", []))
        )
        payload["actors"] = self._merge_named_lists(payload.get("actors", []), llm_payload.get("actors", []))
        payload["capabilities"] = self._merge_named_lists(payload.get("capabilities", []), llm_payload.get("capabilities", []), extra_keys=["actor"])
        payload["aggregate_roots"] = self._merge_named_lists(payload.get("aggregate_roots", []), llm_payload.get("aggregate_roots", []))
        payload["entities"] = self._merge_named_lists(payload.get("entities", []), llm_payload.get("entities", []))
        payload["value_objects"] = self._merge_named_lists(payload.get("value_objects", []), llm_payload.get("value_objects", []))
        payload["business_rules"] = self._merge_text_lists(payload.get("business_rules", []), llm_payload.get("business_rules", []))
        payload["domain_events"] = self._merge_named_lists(payload.get("domain_events", []), llm_payload.get("domain_events", []))
        payload["domain_services"] = self._merge_named_lists(payload.get("domain_services", []), llm_payload.get("domain_services", []))
        return ContextAnalysis.model_validate(payload)

    def _build_deterministic_domain_model(self, analyses: List[ContextAnalysis]) -> Dict[str, Any]:
        project_name = self._derive_project_name(self.last_parsed_documents)
        cross_cutting_constraints = sorted(
            {
                constraint.text
                for constraint in self.last_scout_summary.constraints
                if constraint.category in {"security", "performance", "safety", "non_functional"}
            }
        )
        assumptions = [
            record.description
            for record in self._iter_requirements(self.last_parsed_documents)
            if record.category == "assumptions"
        ]

        contexts: List[Dict[str, Any]] = []
        for analysis in analyses:
            contexts.append(
                {
                    "context_name": analysis.context,
                    "description": analysis.description,
                    "allowed_dependencies": list(analysis.allowed_dependencies),
                    "actors": [self._actor_from_candidate(actor) for actor in analysis.actors],
                    "capabilities": [self._capability_from_candidate(capability) for capability in analysis.capabilities],
                    "ubiquitous_language": {
                        "entities": [self._entity_from_candidate(entity) for entity in analysis.entities],
                        "value_objects": [self._value_object_from_candidate(value_object) for value_object in analysis.value_objects],
                        "services": [self._service_from_candidate(service) for service in analysis.domain_services],
                        "aggregates": [self._aggregate_from_candidate(aggregate) for aggregate in analysis.aggregate_roots],
                        "domain_events": [self._domain_event_from_candidate(event) for event in analysis.domain_events],
                    },
                    "business_rules": [self._business_rule_from_candidate(rule) for rule in analysis.business_rules],
                    "external_references": [
                        {
                            "name": dependency,
                            "relationship": "allowed_dependency",
                            "target_context": dependency,
                            "confidence": 0.6,
                            "evidence_ids": list(analysis.evidence_ids),
                            "sources": [],
                        }
                        for dependency in analysis.allowed_dependencies
                    ],
                    "evidence_ids": list(analysis.evidence_ids),
                    "evidence": [],
                }
            )

        return {
            "schema_version": "2.0.0",
            "project_name": project_name,
            "project_metadata": {
                "version": "1.0.0",
                "generated_at": "SET_BY_CODE",
                "description": f"Evidence-backed domain model synthesized for {project_name}.",
            },
            "bounded_contexts": contexts,
            "global_rules": {
                "naming_convention": "PascalCase",
                "banned_global_terms": list(self.DEFAULT_BANNED_TERMS),
                "cross_cutting_constraints": cross_cutting_constraints,
                "assumptions": assumptions,
            },
        }

    def _merge_domain_models(self, fallback: Dict[str, Any], llm_model: Dict[str, Any]) -> Dict[str, Any]:
        if not llm_model:
            return fallback

        merged = json.loads(json.dumps(fallback))
        merged["schema_version"] = llm_model.get("schema_version") or merged["schema_version"]
        merged["project_name"] = llm_model.get("project_name") or merged["project_name"]
        merged["project_metadata"]["description"] = (
            llm_model.get("project_metadata", {}).get("description")
            or merged["project_metadata"]["description"]
        )

        merged_contexts = {
            context["context_name"]: context
            for context in merged.get("bounded_contexts", [])
        }
        for context in llm_model.get("bounded_contexts", []):
            current = merged_contexts.setdefault(context["context_name"], context)
            current["description"] = context.get("description") or current.get("description")
            current["allowed_dependencies"] = sorted(
                set(current.get("allowed_dependencies", []) + context.get("allowed_dependencies", []))
            )
            current["evidence_ids"] = sorted(set(current.get("evidence_ids", []) + context.get("evidence_ids", [])))
            current["actors"] = self._merge_named_lists(current.get("actors", []), context.get("actors", []))
            current["capabilities"] = self._merge_named_lists(current.get("capabilities", []), context.get("capabilities", []), extra_keys=["actor"])
            current["business_rules"] = self._merge_text_lists(current.get("business_rules", []), context.get("business_rules", []))
            current["external_references"] = self._merge_named_lists(
                current.get("external_references", []),
                context.get("external_references", []),
                extra_keys=["relationship", "target_context"],
            )
            current_ul = current.setdefault("ubiquitous_language", {})
            context_ul = context.get("ubiquitous_language", {})
            current_ul["entities"] = self._merge_named_lists(current_ul.get("entities", []), context_ul.get("entities", []))
            current_ul["value_objects"] = self._merge_named_lists(current_ul.get("value_objects", []), context_ul.get("value_objects", []))
            current_ul["services"] = self._merge_named_lists(current_ul.get("services", []), context_ul.get("services", []))
            current_ul["aggregates"] = self._merge_named_lists(current_ul.get("aggregates", []), context_ul.get("aggregates", []))
            current_ul["domain_events"] = self._merge_named_lists(current_ul.get("domain_events", []), context_ul.get("domain_events", []))

        llm_global = llm_model.get("global_rules", {})
        merged["bounded_contexts"] = list(merged_contexts.values())
        merged["global_rules"]["banned_global_terms"] = sorted(
            set(merged["global_rules"].get("banned_global_terms", []) + llm_global.get("banned_global_terms", []))
        )
        merged["global_rules"]["cross_cutting_constraints"] = sorted(
            set(merged["global_rules"].get("cross_cutting_constraints", []) + llm_global.get("cross_cutting_constraints", []))
        )
        merged["global_rules"]["assumptions"] = sorted(
            set(merged["global_rules"].get("assumptions", []) + llm_global.get("assumptions", []))
        )
        return merged

    def _cleanup_domain_data(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        json_data.setdefault("schema_version", "2.0.0")
        json_data.setdefault("global_rules", {})
        json_data["global_rules"].setdefault("naming_convention", "PascalCase")
        json_data["global_rules"].setdefault("banned_global_terms", list(self.DEFAULT_BANNED_TERMS))
        json_data["global_rules"].setdefault("cross_cutting_constraints", [])
        json_data["global_rules"].setdefault("assumptions", [])
        json_data.setdefault("bounded_contexts", [])

        for context in json_data["bounded_contexts"]:
            context.setdefault("allowed_dependencies", [])
            context.setdefault("actors", [])
            context.setdefault("capabilities", [])
            context.setdefault("business_rules", [])
            context.setdefault("external_references", [])
            context.setdefault("evidence_ids", [])
            context.setdefault("evidence", [])
            ul = context.setdefault("ubiquitous_language", {})
            ul.setdefault("entities", [])
            ul.setdefault("value_objects", [])
            ul.setdefault("services", [])
            ul.setdefault("aggregates", [])
            ul.setdefault("domain_events", [])
            for entity in ul.get("entities", []):
                entity.setdefault("synonyms_to_avoid", [])
                entity.setdefault("sources", [])
                entity.setdefault("evidence_ids", [])
            for bucket in ("value_objects", "services", "aggregates", "domain_events"):
                for item in ul.get(bucket, []):
                    item.setdefault("sources", [])
                    item.setdefault("evidence_ids", [])
                    if bucket == "value_objects":
                        item.setdefault("attributes", [])
            for actor in context.get("actors", []):
                actor.setdefault("sources", [])
                actor.setdefault("evidence_ids", [])
            for capability in context.get("capabilities", []):
                capability.setdefault("sources", [])
                capability.setdefault("evidence_ids", [])
            for rule in context.get("business_rules", []):
                rule.setdefault("sources", [])
                rule.setdefault("evidence_ids", [])
            for reference in context.get("external_references", []):
                reference.setdefault("sources", [])
                reference.setdefault("evidence_ids", [])
        return json_data

    def _apply_metadata_and_evidence(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        data = self._cleanup_domain_data(json.loads(json.dumps(json_data)))
        evidence_lookup = self._build_evidence_lookup(self.last_parsed_documents)
        data.setdefault("project_metadata", {})
        data["project_metadata"]["version"] = data["project_metadata"].get("version") or "1.0.0"
        data["project_metadata"]["generated_at"] = time.strftime("%Y-%m-%d")
        data["project_metadata"]["description"] = (
            data["project_metadata"].get("description")
            or f"Evidence-backed domain model for {data.get('project_name', 'GeneratedDomainModel')}."
        )
        data["project_name"] = data.get("project_name") or self._derive_project_name(self.last_parsed_documents)

        glossary_aliases = self._build_glossary_aliases(self.last_parsed_documents)
        for context in data.get("bounded_contexts", []):
            context["evidence"] = self._sources_from_evidence_ids(context.get("evidence_ids", []), evidence_lookup, "ContextEvidence")
            for actor in context.get("actors", []):
                actor["sources"] = self._sources_from_evidence_ids(actor.get("evidence_ids", []), evidence_lookup, "ActorEvidence")
            for capability in context.get("capabilities", []):
                capability["sources"] = self._sources_from_evidence_ids(capability.get("evidence_ids", []), evidence_lookup, "CapabilityEvidence")
            for rule in context.get("business_rules", []):
                rule["sources"] = self._sources_from_evidence_ids(rule.get("evidence_ids", []), evidence_lookup, "BusinessRuleEvidence")
            for reference in context.get("external_references", []):
                reference["sources"] = self._sources_from_evidence_ids(reference.get("evidence_ids", []), evidence_lookup, "ExternalReferenceEvidence")

            ul = context.get("ubiquitous_language", {})
            for entity in ul.get("entities", []):
                entity["sources"] = self._sources_from_evidence_ids(entity.get("evidence_ids", []), evidence_lookup, "EntityEvidence")
                entity["synonyms_to_avoid"] = self._normalize_synonyms(
                    entity.get("name", ""),
                    entity.get("synonyms_to_avoid", []),
                    glossary_aliases,
                )
            for value_object in ul.get("value_objects", []):
                value_object["sources"] = self._sources_from_evidence_ids(value_object.get("evidence_ids", []), evidence_lookup, "ValueObjectEvidence")
            for service in ul.get("services", []):
                service["sources"] = self._sources_from_evidence_ids(service.get("evidence_ids", []), evidence_lookup, "ServiceEvidence")
            for aggregate in ul.get("aggregates", []):
                aggregate["sources"] = self._sources_from_evidence_ids(aggregate.get("evidence_ids", []), evidence_lookup, "AggregateEvidence")
            for event in ul.get("domain_events", []):
                event["sources"] = self._sources_from_evidence_ids(event.get("evidence_ids", []), evidence_lookup, "DomainEventEvidence")

        return data

    def _build_evidence_lookup(self, parsed_documents: List[ParsedSRSDocument]) -> Dict[str, EvidenceSpan]:
        return {
            span.evidence_id: span
            for document in parsed_documents
            for span in document.evidence_spans
        }

    def _sources_from_evidence_ids(
        self,
        evidence_ids: List[str],
        evidence_lookup: Dict[str, EvidenceSpan],
        rule_name: str,
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        seen = set()
        for evidence_id in evidence_ids:
            span = evidence_lookup.get(evidence_id)
            if span is None:
                continue
            key = (span.document, span.section, span.excerpt)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                InferenceSource(
                    file=span.document,
                    line=span.line,
                    rule=rule_name,
                    evidence=span.excerpt,
                    document=span.document,
                    section=span.section,
                    requirement_id=span.requirement_id,
                ).model_dump(mode="json")
            )
        return sources

    def _derive_project_name(self, parsed_documents: List[ParsedSRSDocument]) -> str:
        text = "\n".join(document.clean_text for document in parsed_documents)
        match = re.search(r"\bKinmail\b", text, re.IGNORECASE)
        if match:
            return "KinmailDomainModel"
        first_name = parsed_documents[0].document_name if parsed_documents else "GeneratedDomainModel"
        stem = os.path.splitext(first_name)[0]
        normalized = re.sub(r"[^A-Za-z0-9]+", " ", stem).strip() or "GeneratedDomain"
        return "".join(part.capitalize() for part in normalized.split()) + "DomainModel"

    def _build_glossary_aliases(self, parsed_documents: List[ParsedSRSDocument]) -> Dict[str, List[str]]:
        aliases: Dict[str, List[str]] = {}
        for record in self._iter_requirements(parsed_documents):
            if record.category != "glossary":
                continue
            normalized = self._normalize_term(record.title, singular=False)
            aliases.setdefault(normalized, [])
        return aliases

    def _normalize_synonyms(
        self,
        entity_name: str,
        synonyms: List[str],
        glossary_aliases: Dict[str, List[str]],
    ) -> List[str]:
        normalized = []
        for synonym in synonyms or []:
            cleaned = self._normalize_term(synonym, singular=False)
            if cleaned and cleaned.lower() != entity_name.lower():
                normalized.append(cleaned)
        normalized.extend(glossary_aliases.get(entity_name, []))
        normalized.extend(self.DEFAULT_ENTITY_ALIASES.get(entity_name, []))
        deduped: List[str] = []
        seen = set()
        for synonym in normalized:
            key = synonym.lower()
            if key == entity_name.lower() or key in seen:
                continue
            seen.add(key)
            deduped.append(synonym)
        return deduped

    def _compare_analysis_to_context(self, analysis: ContextAnalysis, context: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        actual_actor_names = {item["name"] for item in context.get("actors", [])}
        actual_capability_names = {item["name"] for item in context.get("capabilities", [])}
        actual_entity_names = {item["name"] for item in context.get("ubiquitous_language", {}).get("entities", [])}
        actual_value_object_names = {item["name"] for item in context.get("ubiquitous_language", {}).get("value_objects", [])}
        actual_aggregate_names = {item["name"] for item in context.get("ubiquitous_language", {}).get("aggregates", [])}
        actual_service_names = {item["name"] for item in context.get("ubiquitous_language", {}).get("services", [])}
        actual_event_names = {item["name"] for item in context.get("ubiquitous_language", {}).get("domain_events", [])}
        actual_rule_texts = {item["text"] for item in context.get("business_rules", [])}
        actual_reference_names = {item["name"] for item in context.get("external_references", [])}

        for actor in analysis.actors:
            if actor.name not in actual_actor_names:
                missing.append(f"actor:{analysis.context}:{actor.name}")
        for capability in analysis.capabilities:
            if capability.name not in actual_capability_names:
                missing.append(f"capability:{analysis.context}:{capability.name}")
        for entity in analysis.entities:
            if entity.name not in actual_entity_names and entity.name not in actual_reference_names:
                missing.append(f"entity:{analysis.context}:{entity.name}")
        for value_object in analysis.value_objects:
            if value_object.name not in actual_value_object_names:
                missing.append(f"value_object:{analysis.context}:{value_object.name}")
        for aggregate in analysis.aggregate_roots:
            if aggregate.name not in actual_aggregate_names and aggregate.name not in actual_reference_names:
                missing.append(f"aggregate:{analysis.context}:{aggregate.name}")
        for service in analysis.domain_services:
            if service.name not in actual_service_names:
                missing.append(f"service:{analysis.context}:{service.name}")
        for event in analysis.domain_events:
            if event.name not in actual_event_names:
                missing.append(f"domain_event:{analysis.context}:{event.name}")
        for rule in analysis.business_rules:
            if rule.text not in actual_rule_texts:
                missing.append(f"business_rule:{analysis.context}:{rule.text}")
        return missing

    def _collect_model_evidence_ids(self, model_data: Dict[str, Any]) -> set[str]:
        covered = set()
        for context in model_data.get("bounded_contexts", []):
            covered.update(context.get("evidence_ids", []))
            for actor in context.get("actors", []):
                covered.update(actor.get("evidence_ids", []))
            for capability in context.get("capabilities", []):
                covered.update(capability.get("evidence_ids", []))
            for rule in context.get("business_rules", []):
                covered.update(rule.get("evidence_ids", []))
            for reference in context.get("external_references", []):
                covered.update(reference.get("evidence_ids", []))
            ul = context.get("ubiquitous_language", {})
            for bucket_name in ("entities", "value_objects", "aggregates", "services", "domain_events"):
                for item in ul.get(bucket_name, []):
                    covered.update(item.get("evidence_ids", []))
        return covered

    def _requirement_is_represented(
        self,
        *,
        record: RequirementRecord,
        model_data: Dict[str, Any],
        covered_evidence_ids: set[str],
    ) -> bool:
        if not record.evidence_ids:
            return True
        if set(record.evidence_ids).intersection(covered_evidence_ids):
            return True

        normalized_title = self._normalize_term(record.title, singular=False)
        normalized_entity_title = self._normalize_term(record.title)
        normalized_description = self._normalize_text(record.description)
        model_capability_names = {
            capability.get("name", "")
            for context in model_data.get("bounded_contexts", [])
            for capability in context.get("capabilities", [])
        }
        model_reference_names = {
            reference.get("name", "")
            for context in model_data.get("bounded_contexts", [])
            for reference in context.get("external_references", [])
        }
        global_rules = model_data.get("global_rules") or {}
        model_concept_names = {
            item.get("name", "")
            for context in model_data.get("bounded_contexts", [])
            for bucket in (
                context.get("ubiquitous_language", {}).get("entities", []),
                context.get("ubiquitous_language", {}).get("value_objects", []),
                context.get("ubiquitous_language", {}).get("aggregates", []),
                context.get("ubiquitous_language", {}).get("services", []),
                context.get("ubiquitous_language", {}).get("domain_events", []),
            )
            for item in bucket
        }

        if record.category in {"functional", "product_function"}:
            capability_name = self._capability_name_from_title(record.title)
            if capability_name in model_capability_names:
                return True

        if record.category in {"security", "performance", "safety", "non_functional"}:
            rule_texts = {
                self._normalize_text(rule.get("text", ""))
                for context in model_data.get("bounded_contexts", [])
                for rule in context.get("business_rules", [])
            }
            rule_texts.update(
                self._normalize_text(text)
                for text in global_rules.get("cross_cutting_constraints", [])
            )
            if any(
                normalized_description == rule_text
                or normalized_description in rule_text
                or rule_text in normalized_description
                for rule_text in rule_texts
                if rule_text
            ):
                return True

        if record.category == "database":
            if normalized_entity_title in model_concept_names or normalized_title in model_reference_names:
                return True

        if record.category == "glossary":
            if normalized_entity_title in model_concept_names:
                return True

        if record.category == "assumptions":
            assumption_texts = {
                self._normalize_text(text)
                for text in global_rules.get("assumptions", [])
            }
            if normalized_description in assumption_texts:
                return True

        return False

    def _merge_named_lists(
        self,
        left: List[Dict[str, Any]],
        right: List[Dict[str, Any]],
        *,
        extra_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        extra_keys = extra_keys or []
        merged: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        for item in left + right:
            key = tuple([item.get("name")] + [item.get(key_name) for key_name in extra_keys])
            current = merged.get(key)
            if current is None:
                merged[key] = json.loads(json.dumps(item))
                merged[key]["evidence_ids"] = list(item.get("evidence_ids", []))
                continue
            current["evidence_ids"] = sorted(set(current.get("evidence_ids", []) + item.get("evidence_ids", [])))
            for field, value in item.items():
                if field == "evidence_ids":
                    continue
                if field == "sources" and not value:
                    continue
                if current.get(field) in (None, "", [], {}) and value not in (None, "", [], {}):
                    current[field] = value
            if "attributes" in item:
                current["attributes"] = sorted(set(current.get("attributes", []) + item.get("attributes", [])))
        return list(merged.values())

    def _resolve_cross_context_ownership(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        contexts = json_data.get("bounded_contexts", [])
        if not contexts:
            return json_data

        proposal_by_context = {
            proposal.context_name: proposal
            for proposal in self.last_context_map.contexts
        }
        entity_occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        aggregate_occurrences: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for context in contexts:
            ul = context.get("ubiquitous_language", {})
            for entity in ul.get("entities", []):
                entity_occurrences[entity.get("name", "")].append(
                    {"context": context, "item": entity}
                )
            for aggregate in ul.get("aggregates", []):
                aggregate_occurrences[aggregate.get("name", "")].append(
                    {"context": context, "item": aggregate}
                )

        for name, occurrences in entity_occurrences.items():
            if not name or len(occurrences) < 2:
                continue
            owner_context_name = self._select_primary_owner(name, occurrences, proposal_by_context)
            for occurrence in occurrences:
                context = occurrence["context"]
                if context.get("context_name") == owner_context_name:
                    continue
                self._move_item_to_external_reference(
                    context=context,
                    item_name=name,
                    relationship="references_entity",
                    target_context=owner_context_name,
                    evidence_ids=occurrence["item"].get("evidence_ids", []),
                )
                context["ubiquitous_language"]["entities"] = [
                    item
                    for item in context.get("ubiquitous_language", {}).get("entities", [])
                    if item.get("name") != name
                ]

        for name, occurrences in aggregate_occurrences.items():
            if not name or len(occurrences) < 2:
                continue
            owner_context_name = self._select_primary_owner(name, occurrences, proposal_by_context)
            for occurrence in occurrences:
                context = occurrence["context"]
                if context.get("context_name") == owner_context_name:
                    continue
                self._move_item_to_external_reference(
                    context=context,
                    item_name=name,
                    relationship="references_aggregate",
                    target_context=owner_context_name,
                    evidence_ids=occurrence["item"].get("evidence_ids", []),
                )
                context["ubiquitous_language"]["aggregates"] = [
                    item
                    for item in context.get("ubiquitous_language", {}).get("aggregates", [])
                    if item.get("name") != name
                ]
        return json_data

    def _select_primary_owner(
        self,
        item_name: str,
        occurrences: List[Dict[str, Any]],
        proposal_by_context: Dict[str, Any],
    ) -> str:
        best_context = occurrences[0]["context"].get("context_name")
        best_score = -1
        normalized_name = item_name.lower()

        for occurrence in occurrences:
            context = occurrence["context"]
            context_name = context.get("context_name")
            proposal = proposal_by_context.get(context_name)
            evidence_ids = set(occurrence["item"].get("evidence_ids", []))
            score = 0
            if proposal is not None:
                normalized_primary = {entity.lower() for entity in proposal.primary_entities}
                if normalized_name in normalized_primary:
                    score += 100
                score += len(evidence_ids.intersection(proposal.evidence_ids)) * 5
                score += len(proposal.included_capabilities)
            if any(
                aggregate.get("name", "").lower() == normalized_name
                for aggregate in context.get("ubiquitous_language", {}).get("aggregates", [])
            ):
                score += 10
            score += len(evidence_ids)
            if score > best_score:
                best_score = score
                best_context = context_name
        return best_context

    def _move_item_to_external_reference(
        self,
        *,
        context: Dict[str, Any],
        item_name: str,
        relationship: str,
        target_context: str,
        evidence_ids: List[str],
    ) -> None:
        references = context.setdefault("external_references", [])
        for reference in references:
            if (
                reference.get("name") == item_name
                and reference.get("relationship") == relationship
                and reference.get("target_context") == target_context
            ):
                reference["evidence_ids"] = sorted(
                    set(reference.get("evidence_ids", []) + evidence_ids)
                )
                return
        references.append(
            {
                "name": item_name,
                "relationship": relationship,
                "target_context": target_context,
                "confidence": 0.6,
                "evidence_ids": sorted(set(evidence_ids)),
                "sources": [],
            }
        )

    def _run_llm_verifier(
        self,
        requirement_records: List[RequirementRecord],
        final_model: Dict[str, Any],
    ) -> Optional[VerificationReport]:
        provider = getattr(self, "provider", None)
        if provider is None:
            return None
        if getattr(provider, "provider_name", None) == "static-json":
            responses = getattr(provider, "responses", {})
            if "verify_domain_model" not in responses and "Verifier" not in responses:
                return None
        prompt = build_verifier_prompt(
            requirements=[record.model_dump(mode="json") for record in requirement_records],
            final_model=final_model,
        )
        response = self._call_stage(
            stage="Verifier",
            operation="verify_domain_model",
            prompt=prompt,
            response_schema=VerificationReport,
        )
        if response is None:
            return None
        return response

    def _merge_text_lists(self, left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for item in left + right:
            key = item.get("text")
            if not key:
                continue
            current = merged.get(key)
            if current is None:
                merged[key] = json.loads(json.dumps(item))
                merged[key]["evidence_ids"] = list(item.get("evidence_ids", []))
                continue
            current["evidence_ids"] = sorted(set(current.get("evidence_ids", []) + item.get("evidence_ids", [])))
            if not current.get("category") and item.get("category"):
                current["category"] = item["category"]
        return list(merged.values())

    def _normalize_term(self, value: str, *, singular: bool = True) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", value).strip()
        if not cleaned:
            return ""
        words = cleaned.split()
        normalized = "".join(word.capitalize() for word in words)
        if singular and normalized.endswith("ies"):
            normalized = normalized[:-3] + "y"
        elif singular and normalized.endswith("s") and len(normalized) > 3 and not normalized.endswith("ss"):
            normalized = normalized[:-1]
        return normalized

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9]+", " ", value).strip().lower())

    def _coverage_bucket(
        self,
        expected_items: List[str],
        actual_items: set[str],
        *,
        normalize: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        normalize = normalize or (lambda value: value)
        actual_normalized = {normalize(item) for item in actual_items if item}
        covered = [item for item in expected_items if normalize(item) in actual_normalized]
        missing = [item for item in expected_items if normalize(item) not in actual_normalized]
        return {
            "expected_count": len(expected_items),
            "covered_count": len(covered),
            "coverage_percent": round((len(covered) / len(expected_items)) * 100, 2) if expected_items else 100.0,
            "covered": covered,
            "missing": missing,
        }

    def _capability_name_from_title(self, title: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9]+", " ", title).strip()
        if not cleaned:
            return "UnnamedCapability"
        lowered = cleaned.lower()
        replacements = {
            "register to the website": "RegisterAccount",
            "login to the website": "Login",
            "search for products": "SearchProducts",
            "view product details": "ViewProductDetails",
            "view product detail": "ViewProductDetails",
            "add product": "AddProduct",
            "view profile": "ViewProfile",
            "view product dashboard": "ViewProductDashboard",
            "view the website user manual": "ViewUserManual",
            "contact the developer": "ContactDeveloper",
        }
        if lowered in replacements:
            return replacements[lowered]
        return "".join(part.capitalize() for part in cleaned.split())

    def _actor_from_candidate(self, actor: ActorCandidate) -> Dict[str, Any]:
        return DomainActor(
            name=actor.name,
            description=actor.description,
            confidence=0.72,
            evidence_ids=list(actor.evidence_ids),
            sources=[],
        ).model_dump(mode="json")

    def _capability_from_candidate(self, capability: CapabilityCandidate) -> Dict[str, Any]:
        return Capability(
            name=capability.name,
            description=capability.description,
            actor=capability.actor,
            confidence=0.72,
            evidence_ids=list(capability.evidence_ids),
            sources=[],
        ).model_dump(mode="json")

    def _entity_from_candidate(self, entity: EntityCandidate) -> Dict[str, Any]:
        return Entity(
            name=entity.name,
            description=entity.description,
            confidence=0.7,
            evidence_ids=list(entity.evidence_ids),
            sources=[],
            synonyms_to_avoid=[],
        ).model_dump(mode="json")

    def _value_object_from_candidate(self, value_object: ValueObjectCandidate) -> Dict[str, Any]:
        return ValueObject(
            name=value_object.name,
            description=value_object.description,
            attributes=list(value_object.attributes),
            confidence=0.68,
            evidence_ids=list(value_object.evidence_ids),
            sources=[],
        ).model_dump(mode="json")

    def _service_from_candidate(self, service: EntityCandidate) -> Dict[str, Any]:
        return Service(
            name=service.name,
            description=service.description,
            confidence=0.6,
            evidence_ids=list(service.evidence_ids),
            sources=[],
        ).model_dump(mode="json")

    def _aggregate_from_candidate(self, aggregate: EntityCandidate) -> Dict[str, Any]:
        return Aggregate(
            name=aggregate.name,
            description=aggregate.description,
            confidence=0.65,
            evidence_ids=list(aggregate.evidence_ids),
            sources=[],
        ).model_dump(mode="json")

    def _domain_event_from_candidate(self, event: EntityCandidate) -> Dict[str, Any]:
        return DomainEvent(
            name=event.name,
            description=event.description,
            confidence=0.6,
            evidence_ids=list(event.evidence_ids),
            sources=[],
        ).model_dump(mode="json")

    def _business_rule_from_candidate(self, rule: ConstraintCandidate) -> Dict[str, Any]:
        return BusinessRule(
            text=rule.text,
            category=rule.category,
            confidence=0.78,
            evidence_ids=list(rule.evidence_ids),
            sources=[],
        ).model_dump(mode="json")

    def _check_response_completion(self, response, retry: int) -> bool:
        finish_reason = getattr(response, "finish_reason", None)
        if finish_reason is None:
            print("  No finish reason in response")
            return False

        normalized = getattr(finish_reason, "name", None) or str(finish_reason)
        if "." in normalized:
            normalized = normalized.split(".")[-1]
        normalized = normalized.strip()
        if normalized == "STOP":
            return True

        print(f"  Response incomplete: finish_reason={finish_reason}")
        if retry < 4:
            print(f"  Retrying ({retry + 1}/5)...")
            return False
        return False

    def _save_intermediate(self, stage: str, data: Dict[str, Any]) -> None:
        try:
            filename = f"{self.run_timestamp}_{stage}.json"
            filepath = os.path.join(INTERMEDIATE_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False)
            print(f"  Saved intermediate output: {filename}")
        except Exception as exc:
            print(f"  Failed to save intermediate output: {exc}")

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as exc:
            try:
                cleaned = response_text.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"  JSON parse failed: {exc}")
                return {"error": "json_parse_failed", "raw_response": response_text[:500]}

    def get_last_run_summary(self) -> Dict[str, Any]:
        return {
            "request_count": self.request_count,
            "stage_latencies_ms": dict(self.stage_timings_ms),
            "verification_report": self.last_verification_report.model_dump(mode="json"),
            **self.last_run_summary,
        }
