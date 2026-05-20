"""
Domain Architect

Multi-stage AI pipeline for extracting Domain-Driven Design models from SRS documents.
Uses Google Gemini for intelligent analysis with rate limiting and error handling.

Pipeline stages:
1. Scout - Extract domain-relevant sentences with smart chunking
2. Architect - Identify bounded contexts from domain knowledge
3. Specialist - Analyze all contexts for entities, value objects, and rules
4. Synthesizer - Merge analyses into final DomainModel
"""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import ValidationError

from core.llm.gemini import GeminiClient
from core.llm._response_adapter import LLMResponseAdapter
from core.schemas import DomainModel
from core.orchestration.errors import (
    ArchitectExtractionError,
    SpecialistFailureError,
    SpecialistShapeError,
    SynthesizerEmptyModelError,
)
from core.pipeline_contracts import ContextHypothesis, SpecialistAnalysis
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.scout.chunking import section_aware_chunks
from core.verifier.checks_deterministic import (
    check_d1_supporting_sentence_ids_subset,
    check_d2_entity_evidence_nonempty,
    check_d3_entity_names_unique_across_contexts,
    check_d4_aggregate_members_exist_in_context,
    check_d5_allowed_dependencies_reference_existing_contexts,
)
from core.verifier.types import VerifierResult
from core.token_tracker import TokenTracker
from configs.models import stage_config

load_dotenv()

# Intermediate outputs directory
INTERMEDIATE_DIR = os.path.join(os.path.dirname(__file__), "intermediate")

# Conservative default — free-tier safe. Override via env var
# DDD_MIN_DELAY_SECONDS or constructor kwarg `min_delay` (Pro tier supports
# higher RPM and benefits from lower delays, e.g. 1.0s).
DEFAULT_MIN_DELAY_SECONDS = 6.0

# Default sequential (1 worker). Set DDD_SCOUT_MAX_WORKERS > 1 (Pro tier with
# headroom on RPM) for parallel chunk extraction. Free tier should keep 1.
DEFAULT_SCOUT_MAX_WORKERS = max(1, int(os.getenv("DDD_SCOUT_MAX_WORKERS", "1")))

# Type alias for progress callback
ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _truncate_with_head_tail(text: str, max_chars: int, head_ratio: float = 0.6) -> str:
    """Truncate `text` to `max_chars` by keeping the head and tail and
    dropping the middle, with an explicit marker so the LLM knows the
    document was truncated.

    Default ratio: 60% head / 40% tail. SRS documents typically place
    intro / core concepts up front and acceptance criteria / domain
    events near the end; both should survive context-window squeezes.
    """
    if len(text) <= max_chars:
        return text
    marker = "\n\n... [middle truncated for context window] ...\n\n"
    budget = max_chars - len(marker)
    head_size = int(budget * head_ratio)
    tail_size = budget - head_size
    return f"{text[:head_size]}{marker}{text[-tail_size:]}"


class DomainArchitect:
    """AI-powered domain model extraction from SRS documents."""

    def __init__(
        self,
        model: Optional[str] = None,
        progress_callback: ProgressCallback = None,
        min_delay: Optional[float] = None,
        scout_max_workers: Optional[int] = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = GeminiClient(api_key=api_key)
        self.model_name = model or stage_config("Architect").model_id
        self.last_request_time = 0
        self.min_delay = (
            min_delay
            if min_delay is not None
            else float(os.getenv("DDD_MIN_DELAY_SECONDS", DEFAULT_MIN_DELAY_SECONDS))
        )
        self.request_count = 0
        self._rate_limit_lock = threading.Lock()
        self.scout_max_workers = (
            scout_max_workers
            if scout_max_workers is not None
            else DEFAULT_SCOUT_MAX_WORKERS
        )
        self.token_tracker = TokenTracker.get_instance()
        self.progress_callback = progress_callback
        
        # Ensure intermediate directory exists
        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        self.run_timestamp = time.strftime("%Y%m%d_%H%M%S")

        print("\n" + "="*70)
        print("🏗️  DOMAIN ARCHITECT INITIALIZED")
        print("="*70)
        print(f"  Model: {self.model_name}")
        print(f"  Rate Limit: {self.min_delay}s between requests")
        print("="*70 + "\n")

    def _report_progress(self, stage: str, status: str, detail: str = "", progress: int = 0):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback({
                "stage": stage,
                "status": status,
                "detail": detail,
                "progress": progress
            })

    # =========================================================================
    # RATE LIMITING & ERROR HANDLING
    # =========================================================================

    def _wait_for_rate_limit(self):
        """Enforce minimum delay between API requests. Thread-safe."""
        with self._rate_limit_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed
                print(f"  ⏳ Rate limiting... waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self.last_request_time = time.time()
            self.request_count += 1
            print(f"  📡 API Request #{self.request_count}")

    def _is_quota_error_and_backoff(self, error: Exception, retry_count: int) -> bool:
        """Return True iff `error` was a quota / rate-limit error AND we slept
        for the recommended backoff duration. Return False for any other error
        (caller decides how to handle).
        """
        error_str = str(error)
        is_quota = (
            "429" in error_str
            or "quota" in error_str.lower()
            or "ResourceExhausted" in str(type(error))
        )
        if not is_quota:
            return False

        retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
        if retry_match:
            wait_time = max(float(retry_match.group(1)), 10)
        else:
            # Exponential backoff: 15s, 30s, 60s, 120s
            wait_time = min(15 * (2 ** retry_count), 300)

        print(f"  ⚠️  QUOTA EXCEEDED - Backing off {wait_time:.1f}s...")
        time.sleep(wait_time)
        return True

    # =========================================================================
    # STAGE 1: SCOUT - Extract Domain Sentences
    # =========================================================================

    def extract_domain_sentences(self, clean_text: str) -> List[str]:
        """
        Extract domain-relevant sentences from document text.

        Uses smart chunking to process entire documents while respecting
        token limits. Each chunk is processed independently and results
        are combined.
        """
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│ STAGE 1: SCOUT - Extracting Domain Knowledge                   │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        self._report_progress("Scout", "started", "Extracting domain sentences", 0)

        chunk_size = 10000
        chunks = self._split_text_into_chunks(clean_text, chunk_size)
        all_sentences = []

        print(f"  📄 Document: {len(clean_text):,} characters")
        print(f"  🔪 Chunks: {len(chunks)} (max {chunk_size:,} chars each)")

        if self.scout_max_workers <= 1:
            # Sequential — preserves original behavior exactly.
            for i, chunk in enumerate(chunks):
                progress = int((i + 1) / len(chunks) * 100)
                print(f"  ▶️  Chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) [{progress}%]")
                self._report_progress(
                    "Scout", "in_progress",
                    f"Processing chunk {i + 1}/{len(chunks)}",
                    progress,
                )
                sentences = self._extract_sentences_from_chunk(chunk, i + 1, len(chunks))
                all_sentences.extend(sentences)
        else:
            # Parallel — coalesce per-chunk progress to start/end.
            print(f"  ⚡ Parallel mode: {self.scout_max_workers} workers")
            self._report_progress(
                "Scout", "in_progress",
                f"Processing {len(chunks)} chunks in parallel ({self.scout_max_workers} workers)",
                50,
            )
            args = [(chunk, i + 1, len(chunks)) for i, chunk in enumerate(chunks)]
            with ThreadPoolExecutor(max_workers=self.scout_max_workers) as ex:
                chunk_results = list(ex.map(
                    lambda a: self._extract_sentences_from_chunk(*a),
                    args,
                ))
            for r in chunk_results:
                all_sentences.extend(r)

        print(f"  ✅ Extracted {len(all_sentences)} domain-relevant sentences\n")
        self._report_progress("Scout", "completed", f"Extracted {len(all_sentences)} sentences", 100)
        
        # Save intermediate output
        self._save_intermediate(
            stage="1_scout",
            data={
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "input_length": len(clean_text),
                "chunks_processed": len(chunks),
                "sentences_extracted": len(all_sentences),
                "sentences": all_sentences
            }
        )
        
        return all_sentences if all_sentences else [clean_text[:1000]]

    def _split_text_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Find last period to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                if last_period > start:
                    end = last_period + 1
            chunks.append(text[start:end])
            start = end
        return chunks

    def _extract_sentences_from_chunk(
        self, chunk: str, chunk_num: int, total_chunks: int
    ) -> List[str]:
        """Extract domain sentences from a single chunk."""
        prompt = f"""Extract EXACTLY the domain-relevant sentences from this SRS document chunk.

STRICT INCLUSION CRITERIA (include ONLY if sentence matches):
1. Defines a business entity with attributes (e.g., "Customer has name, email, address")
2. States a business rule with keywords: "must", "shall", "cannot", "only if", "required"
3. Describes a workflow step (e.g., "Order is placed", "Payment is processed")
4. Defines a relationship (e.g., "Customer places Orders", "Order contains Items")
5. Contains a formula or calculation rule

STRICT EXCLUSION CRITERIA (NEVER include):
- Headers, titles, section numbers
- Legal/copyright text
- UI descriptions (colors, buttons, layouts)
- Technical details (database, API, code)
- General descriptions without specific rules

EXTRACTION RULES:
- Extract sentences verbatim - do not paraphrase
- One sentence = one array element
- Skip duplicate or near-duplicate sentences
- Minimum sentence length: 15 characters

TEXT CHUNK {chunk_num}/{total_chunks}:
{chunk}

RESPOND WITH JSON:
{{
  "sentences": ["exact sentence 1", "exact sentence 2", ...]
}}

Return empty array [] if no sentences match the criteria."""

        sc = stage_config("Scout")
        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                llm_response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=sc.temperature,
                    seed=sc.seed,
                    response_mime_type="application/json",
                )
                response = LLMResponseAdapter(llm_response)

                # Check if response was truncated
                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue

                result = self._parse_json_response(self._safe_response_text(response))

                if (
                    isinstance(result, dict)
                    and result.get("error") == "json_parse_failed"
                ):
                    print(f"      ⚠️  Parse failed - Retry {retry + 1}/5")
                    if retry < 4:
                        continue
                    # Final fallback: split chunk into sentences
                    return [s.strip() for s in chunk.split(".") if len(s.strip()) > 20][
                        :50
                    ]

                # Track token usage only for successful responses
                self.token_tracker.track_api_call(
                    response, 
                    stage="Scout",
                    operation=f"extract_sentences_chunk_{chunk_num}"
                )

                if isinstance(result, dict) and "sentences" in result:
                    return result["sentences"]
                elif isinstance(result, list):
                    return result
                return []

            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    print(f"      [WARN] Chunk {chunk_num} error: {e}")
                    if retry >= 4:
                        return [
                            s.strip() for s in chunk.split(".") if len(s.strip()) > 20
                        ][:50]

        return []

    # =========================================================================
    # STAGE 2: ARCHITECT - Identify Bounded Contexts
    # =========================================================================

    def identify_contexts(self, domain_sentences: List[str]) -> List[str]:
        """
        Identify bounded contexts from domain sentences.

        Analyzes all extracted domain knowledge to identify distinct
        business areas with their own terminology and rules.
        """
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│ STAGE 2: ARCHITECT - Identifying Bounded Contexts              │")
        print("└─────────────────────────────────────────────────────────────────┘")
        
        self._report_progress("Architect", "started", "Identifying bounded contexts", 0)

        text = "\n".join(domain_sentences)
        max_chars = 50000
        if len(text) > max_chars:
            print(f"  ✂️  Truncating input: {len(text):,} → {max_chars:,} chars (head + tail preserved)")
            text = _truncate_with_head_tail(text, max_chars=max_chars)

        prompt = f"""Identify distinct Bounded Contexts from the domain knowledge below.

A Bounded Context is a cohesive business area with:
- Its own terminology and ubiquitous language
- Clear ownership of specific entities
- Defined boundaries and responsibilities

DOMAIN KNOWLEDGE:
{text}

CONSTRAINTS:
- Identify 3-8 contexts (avoid over/under-fragmentation)
- Use PascalCase business names (OrderManagement, not order_mgmt)
- Each entity belongs to ONE primary context
- Avoid generic names (CoreDomain, BusinessLogic, MainSystem)

RESPOND WITH JSON:
{{
  "contexts": ["ContextName1", "ContextName2", ...]
}}"""

        sc = stage_config("Architect")
        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                llm_response = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=sc.temperature,
                    seed=sc.seed,
                    response_mime_type="application/json",
                )
                response = LLMResponseAdapter(llm_response)

                # Check if response was truncated
                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue

                result = self._parse_json_response(self._safe_response_text(response))

                if (
                    isinstance(result, dict)
                    and result.get("error") == "json_parse_failed"
                ):
                    print(f"  ⚠️  Parse failed - Retry {retry + 1}/5")
                    if retry < 4:
                        continue
                    print("  ⚠️  Max retries reached, raising ArchitectExtractionError")
                    self._report_progress("Architect", "error", "Architect exhausted JSON parse retries", 100)
                    raise ArchitectExtractionError(
                        srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                        message="Architect exhausted JSON parse retries (5/5)",
                    )

                # Track token usage only for successful responses
                self.token_tracker.track_api_call(
                    response,
                    stage="Architect",
                    operation="identify_contexts"
                )

                if isinstance(result, dict) and "contexts" in result:
                    contexts = result["contexts"]
                    if contexts and len(contexts) > 0:
                        # Save intermediate output
                        self._save_intermediate(
                            stage="2_architect",
                            data={
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "contexts_identified": len(contexts),
                                "contexts": contexts,
                                "input_sentences_count": len(domain_sentences)
                            }
                        )
                        self._report_progress("Architect", "completed", f"Found {len(contexts)} contexts", 100)
                        return contexts
                elif isinstance(result, list) and len(result) > 0:
                    # Save intermediate output
                    self._save_intermediate(
                        stage="2_architect",
                        data={
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "contexts_identified": len(result),
                            "contexts": result,
                            "input_sentences_count": len(domain_sentences)
                        }
                    )
                    self._report_progress("Architect", "completed", f"Found {len(result)} contexts", 100)
                    return result

                print(f"  ⚠️  Empty response - Retry {retry + 1}/5")
                if retry < 4:
                    continue
                self._report_progress("Architect", "error", "Architect produced empty contexts", 100)
                raise ArchitectExtractionError(
                    srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                    message="Architect produced empty contexts list after 5 retries",
                )

            except ArchitectExtractionError:
                raise
            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    print(f"   [WARN] Context identification error: {e}")
                    if retry >= 4:
                        self._report_progress("Architect", "error", str(e), 100)
                        raise ArchitectExtractionError(
                            srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                            message=f"Architect failed with {type(e).__name__}: {e}",
                        ) from e

        raise ArchitectExtractionError(
            srs_path=getattr(self, "_current_srs_path", "<unknown>"),
            message="Architect exhausted retry loop without producing contexts",
        )

    @staticmethod
    def _unwrap_singleton_list(payload: Any) -> Dict[str, Any]:
        """Defensive unwrap of LLM-parsed JSON.

        Gemini-Pro occasionally returns the prompted object inside a
        single-element top-level array. This helper unwraps that case
        explicitly while rejecting ambiguous (empty / multi-element) and
        type-incompatible inputs.
        """
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            if not payload:
                raise ValueError("Specialist payload is an empty list; cannot unwrap")
            if len(payload) > 1:
                raise ValueError(
                    f"Specialist payload is a list with multiple elements "
                    f"({len(payload)}); cannot unwrap unambiguously"
                )
            first = payload[0]
            if not isinstance(first, dict):
                raise ValueError(
                    f"Specialist payload list contains a non-dict element "
                    f"(type={type(first).__name__})"
                )
            return first
        raise ValueError(
            f"Specialist payload has unexpected type {type(payload).__name__}; "
            f"expected dict or single-element list of dict"
        )

    @staticmethod
    def _validate_specialist_payload(
        payload: Any, ctx: ContextHypothesis,
    ) -> SpecialistAnalysis:
        """Convert a raw LLM-parsed payload into a typed SpecialistAnalysis.

        Handles the list-unwrap quirk and the Pydantic strict validation
        in one place. On any failure, raises SpecialistShapeError with the
        validation errors so the retry loop can re-prompt the LLM with
        structured feedback.
        """
        try:
            unwrapped = DomainArchitect._unwrap_singleton_list(payload)
        except ValueError as e:
            raise SpecialistShapeError(
                context_name=ctx.context_name,
                errors=[{"shape": str(e)}],
                raw_excerpt=str(payload)[:200],
            ) from e

        # Compose the SpecialistAnalysis payload: context from ctx + content
        # from the LLM. The LLM's own "context" key (if present) is dropped
        # in favor of the trusted ContextHypothesis.
        composed = {
            "context": ctx.model_dump(),
            **{k: v for k, v in unwrapped.items() if k != "context"},
        }

        try:
            return SpecialistAnalysis.model_validate(composed)
        except ValidationError as e:
            raise SpecialistShapeError(
                context_name=ctx.context_name,
                errors=e.errors(),
                raw_excerpt=str(unwrapped)[:200],
            ) from e

    def extract_per_context_details(
        self, contexts: List[str], domain_sentences: List[str]
    ) -> List[SpecialistAnalysis]:
        """Per-context Specialist loop (FM-23).

        Issues one LLM call per bounded context with a focused prompt
        that mentions only that one context. Forces exclusive entity
        ownership at the prompt level.
        """
        results: List[SpecialistAnalysis] = []
        numbered_sentences_text = "\n".join(
            f"[{i}] {s}" for i, s in enumerate(domain_sentences)
        )
        for ctx_name in contexts:
            prompt = self._build_specialist_prompt_per_context(
                context_name=ctx_name,
                numbered_sentences_text=numbered_sentences_text,
            )
            sc = stage_config("Specialist")
            success = False
            for retry in range(5):
                try:
                    self._wait_for_rate_limit()
                    llm_response = self.client.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model_name,
                        temperature=sc.temperature,
                        seed=sc.seed,
                        response_mime_type="application/json",
                    )
                    response = LLMResponseAdapter(llm_response)
                    if not self._check_response_completion(response, retry):
                        if retry < 4:
                            time.sleep(2)
                            continue
                    result = self._parse_json_response(self._safe_response_text(response))
                    if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                        if retry < 4:
                            time.sleep(2)
                            continue
                        raise SpecialistFailureError(
                            context_name=ctx_name,
                            message=f"Specialist parse failed for {ctx_name} after 5 retries",
                        )
                    # Typed boundary validation — converts list-not-dict crashes (the
                    # live pipeline failure mode at the old L692) into a SpecialistShapeError
                    # that the retry loop can act on.
                    # identify_contexts returns bare context-name strings (no descriptions).
                    # Synthesizer fills BoundedContext.description later. Empty here is honest.
                    ctx = ContextHypothesis(context_name=ctx_name, description="")
                    try:
                        analysis = DomainArchitect._validate_specialist_payload(result, ctx)
                    except SpecialistShapeError as shape_err:
                        print(
                            f"  ⚠️  Specialist shape error for {ctx_name} - Retry {retry + 1}/5 "
                            f"({len(shape_err.validation_errors)} validation error(s))"
                        )
                        if retry < 4:
                            time.sleep(2)
                            continue
                        raise

                    self.token_tracker.track_api_call(
                        response, stage="Specialist", operation=f"per_context:{ctx_name}",
                    )
                    results.append(analysis)
                    success = True
                    break
                except SpecialistFailureError:
                    raise
                except Exception as e:
                    if not self._is_quota_error_and_backoff(e, retry):
                        if retry >= 4:
                            raise SpecialistFailureError(
                                context_name=ctx_name,
                                message=f"Specialist failed for {ctx_name}: {type(e).__name__}: {e}",
                            ) from e
            if not success:
                raise SpecialistFailureError(
                    context_name=ctx_name,
                    message=f"Specialist exhausted retry loop for {ctx_name}",
                )
        self._save_intermediate(
            stage="3_specialist",
            data={
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "contexts_analyzed": len(results),
                "analyses": [a.model_dump(mode="json") for a in results],
            },
        )
        return results

    # =============================================================================
    # STAGE 3: SPECIALIST - Per-Context Entity / VO / Service / Aggregate Analysis
    # =============================================================================

    def _build_specialist_prompt_per_context(
        self, *, context_name: str, numbered_sentences_text: str
    ) -> str:
        """Build the per-context Specialist prompt string (called once per bounded context)."""
        return f"""You are analyzing exactly ONE Bounded Context: {context_name}.

Do NOT include entities, aggregates, services, value objects, or
domain events that belong to other contexts.

Extract the 5 DDD building blocks for {context_name}:
1. Entities       - Objects with unique identity
2. Value Objects  - Immutable objects defined by attributes
3. Services       - Stateless operations
4. Aggregates     - Consistency boundaries with explicit members
5. Domain Events  - Past-tense business facts

DOMAIN KNOWLEDGE (numbered sentences):
{numbered_sentences_text}

RESPOND WITH JSON for {context_name} only:
{{
  "context": "{context_name}",
  "entities": [{{
    "name": "EntityName",
    "description": "Represents a customer who places orders and holds an account balance.",
    "attributes": ["attr1"],
    "confidence": 0.9,
    "justification": "Cited in 3 sentences",
    "evidence_sentence_indices": [2, 7]
  }}],
  "value_objects": [],
  "services": [],
  "aggregates": [{{"name": "X", "description": "Consistency boundary around X and its members.", "members": ["EntityName"]}}],
  "domain_events": [],
  "business_rules": []
}}

EVERY entity MUST emit `description` (1-2 sentences). Pydantic strict validation rejects any entity missing this field.
Every entity.evidence_sentence_indices must contain at least one Scout-numbered sentence index.
Do not invent data not present in the sentences."""

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def analyze_document(self, text: str) -> DomainModel:
        """Run the 5-stage pipeline on raw SRS text and return a typed DomainModel.

        Each stage produces a typed envelope from core.pipeline_contracts.
        Synthesizer is deterministic + per-context narrow LLM enrichment
        + D6/D7/D8 hard-fail invariants (via core.synthesizer).
        """
        from core.synthesizer import synthesize_domain_model
        from core.pipeline_contracts import (
            ScoutOutput, ArchitectOutput, VerifierResult as ContractVerifierResult,
            VerifierIssue as ContractVerifierIssue,
        )

        def _to_contract_issue(old_issue) -> ContractVerifierIssue:
            """Adapt core.verifier.types.VerifierIssue (dataclass) to
            core.pipeline_contracts.VerifierIssue (Pydantic)."""
            sev = old_issue.severity
            # IssueSeverity enum values are "error"/"warn"; contract expects "ERROR"/"WARN"
            sev_str = sev.value.upper() if hasattr(sev, "value") else str(sev).upper()
            return ContractVerifierIssue(
                severity=sev_str,
                check_id=getattr(old_issue, "issue_type", "unknown"),
                target=getattr(old_issue, "location", ""),
                message=old_issue.message,
            )

        def scout_fn(srs_text: str) -> ScoutOutput:
            chunks = section_aware_chunks(srs_text, token_budget=10000)
            from core.pipeline_contracts import SectionedSentence, ChunkMetadata
            sentences = [
                SectionedSentence(
                    index=i,
                    text=chunk["text"],
                    section=chunk.get("section"),
                )
                for i, chunk in enumerate(chunks)
            ]
            return ScoutOutput(
                sentences=sentences,
                chunk_metadata=ChunkMetadata(
                    chunk_count=len(chunks),
                    total_chars=sum(len(c["text"]) for c in chunks),
                ),
            )

        def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
            sentence_texts = [s.text for s in scout.sentences]
            ctx_names = self.identify_contexts(sentence_texts)
            contexts = [
                ContextHypothesis(context_name=n, description=f"{n} context")
                for n in ctx_names
            ]
            return ArchitectOutput(contexts=contexts)

        def specialist_fn(
            arch: ArchitectOutput, scout: ScoutOutput
        ) -> List[SpecialistAnalysis]:
            ctx_names = [c.context_name for c in arch.contexts]
            sentence_texts = [s.text for s in scout.sentences]
            return self.extract_per_context_details(ctx_names, sentence_texts)

        def synthesizer_fn(analyses: List[SpecialistAnalysis]) -> DomainModel:
            return synthesize_domain_model(
                analyses,
                llm_client=self.client,
                project_name=getattr(self, "project_name", "DomainModel"),
            )

        def verifier_fn(snapshot: Dict[str, Any]) -> ContractVerifierResult:
            scout_obj: ScoutOutput = snapshot["scout"]
            architect_obj: ArchitectOutput = snapshot["architect"]
            specialist_obj: List[SpecialistAnalysis] = snapshot["specialist"]

            scout_indices = {s.index for s in scout_obj.sentences}
            contexts_dicts = [
                {"name": c.context_name, "supporting_sentence_ids": c.supporting_sentence_ids}
                for c in architect_obj.contexts
            ]
            entities_by_context = {
                a.context.context_name: [e.model_dump() for e in a.entities]
                for a in specialist_obj
            }

            legacy_issues = []
            legacy_issues.extend(
                check_d1_supporting_sentence_ids_subset(contexts_dicts, scout_indices)
            )
            legacy_issues.extend(
                check_d3_entity_names_unique_across_contexts(entities_by_context)
            )
            for a in specialist_obj:
                legacy_issues.extend(check_d2_entity_evidence_nonempty(
                    context_name=a.context.context_name,
                    entities=[e.model_dump() for e in a.entities],
                ))
                legacy_issues.extend(check_d4_aggregate_members_exist_in_context(
                    a.context.context_name,
                    [e.model_dump() for e in a.entities],
                    [agg.model_dump() for agg in a.aggregates],
                ))
            legacy_issues.extend(
                check_d5_allowed_dependencies_reference_existing_contexts(contexts_dicts)
            )

            contract_issues = [_to_contract_issue(i) for i in legacy_issues]
            has_error = any(i.severity == "ERROR" for i in contract_issues)
            return ContractVerifierResult(ok=(not has_error), issues=contract_issues)

        deps = PipelineDeps(
            scout=scout_fn,
            architect=architect_fn,
            specialist=specialist_fn,
            synthesizer=synthesizer_fn,
            verifier=verifier_fn,
        )
        return run_pipeline(srs_text=text, deps=deps)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _safe_response_text(self, response) -> str:
        """Return response.text or empty string if None.

        Gemini may return `response.text = None` when finish_reason is SAFETY,
        RECITATION, or when no candidates were produced. The existing
        _parse_json_response treats empty input as a JSON parse failure and
        routes through the retry path; surfacing None directly would raise
        AttributeError instead.
        """
        text = getattr(response, "text", None)
        return text if text is not None else ""

    def _check_response_completion(self, response, retry: int) -> bool:
        """
        Check if API response was complete or truncated.
        
        Returns:
            True if response is complete and valid
            False if truncated (needs retry)
        """
        if not response.candidates:
            print("      ⚠️  No candidates in response")
            return False

        finish_reason = response.candidates[0].finish_reason

        if finish_reason == "STOP":
            return True

        if finish_reason == "MAX_TOKENS":
            # Same prompt would re-truncate; accept whatever JSON arrived and
            # let _parse_json_response handle truncated/incomplete output.
            print("      💡 Hit max output tokens — accepting partial response")
            return True

        # SAFETY / RECITATION / OTHER may be transient — retry can help.
        print(f"      ⚠️  Response incomplete: finish_reason={finish_reason}")
        if finish_reason == "SAFETY":
            print("      🛡️  Response blocked by safety filters")
        elif finish_reason == "RECITATION":
            print("      📝 Response blocked due to citation/recitation")
        elif finish_reason == "OTHER":
            print("      ❓ Response stopped for unknown reason")

        if retry < 4:
            print(f"      🔄 Retrying ({retry + 1}/5)...")
            return False
        print("      ❌ Max retries reached")
        return False

    def _save_intermediate(self, stage: str, data: Dict[str, Any]):
        """Save intermediate pipeline output to JSON file."""
        try:
            filename = f"{self.run_timestamp}_{stage}.json"
            filepath = os.path.join(INTERMEDIATE_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"  💾 Saved intermediate output: {filename}")
        except Exception as e:
            print(f"  ⚠️  Failed to save intermediate output: {e}")
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any] | List[Any]:
        """Parse JSON from LLM response.

        Returns the parsed structure (dict or list — Gemini sometimes returns
        a top-level array). On parse failure returns the sentinel
        {"error": "json_parse_failed", "raw_response": <first 500 chars>}.
        Callers detect failure with:
            isinstance(result, dict) and result.get("error") == "json_parse_failed"
        """
        try:
            # Direct parse - Gemini returns valid JSON with application/json mime type
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # Check if response was truncated
            error_msg = str(e)
            is_incomplete = "Expecting" in error_msg or "Unterminated" in error_msg
            
            if is_incomplete:
                print(f"      ⚠️  Response appears truncated: {error_msg[:80]}")
                print(f"      📏 Response length: {len(response_text):,} chars")
                print(f"      📝 Ends with: ...{response_text[-50:]}")
            
            # Rare case: Remove markdown code blocks if present
            try:
                cleaned = response_text.replace("```json", "").replace("```", "").strip()
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Log full error for debugging
                print(f"      ❌ JSON parse failed: {error_msg}")
                print(f"      📄 First 200 chars: {response_text[:200]}")
                return {"error": "json_parse_failed", "raw_response": response_text[:500]}
