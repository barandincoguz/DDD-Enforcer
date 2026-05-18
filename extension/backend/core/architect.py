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
from google import genai
from google.genai import types

from core.schemas import DomainModel
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

        self.client = genai.Client(api_key=api_key)
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
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=sc.temperature,
                        seed=sc.seed,
                    ),
                )
                
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
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=sc.temperature,
                        seed=sc.seed,
                    ),
                )

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
                    print("  ⚠️  Max retries reached, using fallback context")
                    self._report_progress("Architect", "completed", "Using fallback context", 100)
                    return ["CoreDomain"]

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
                self._report_progress("Architect", "completed", "Using fallback context", 100)
                return ["CoreDomain"]

            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    print(f"   [WARN] Context identification error: {e}")
                    self._report_progress("Architect", "error", str(e), 100)
                    return ["CoreDomain"]

        return ["CoreDomain"]

    # =========================================================================
    # STAGE 3: SPECIALIST - Analyze All Contexts
    # =========================================================================

    def extract_all_contexts_details(
        self, contexts: List[str], domain_sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze all contexts in a single API call for efficiency.

        Extracts aggregate roots, entities, value objects, and business
        rules for each identified bounded context.
        """
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│ STAGE 3: SPECIALIST - Analyzing Context Details                │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print(f"  🔍 Analyzing {len(contexts)} contexts in single request")
        
        self._report_progress("Specialist", "started", f"Analyzing {len(contexts)} contexts", 0)

        contexts_text = ", ".join(contexts)
        sentences_text = "\n".join(domain_sentences)

        max_chars = 60000
        if len(sentences_text) > max_chars:
            print(f"  ✂️  Truncating input: {len(sentences_text):,} → {max_chars:,} chars (head + tail preserved)")
            sentences_text = _truncate_with_head_tail(sentences_text, max_chars=max_chars)

        prompt = f"""Analyze the domain knowledge for these Bounded Contexts: {contexts_text}

For EACH context, extract the 5 DDD building blocks:
1. Entities       - Objects with unique identity (Customer, Order, Product)
2. Value Objects  - Immutable objects defined by attributes (Address, Money)
3. Services       - Stateless operations that don't naturally belong to an entity
4. Aggregates     - Consistency boundaries; each aggregate has a name and lists
                    the entities (`members`) that live inside it
5. Domain Events  - Past-tense business facts (OrderPlaced, PaymentReceived)

DOMAIN KNOWLEDGE:
{sentences_text}

RESPOND WITH JSON:
{{
  "analyses": [
    {{
      "context": "ContextName",
      "entities": [{{"name": "Entity1", "attributes": ["id", "name", "status"]}}],
      "value_objects": [{{"name": "Money", "attributes": ["amount", "currency"]}}],
      "services": [{{"name": "PricingService", "description": "Computes order totals"}}],
      "aggregates": [{{"name": "Order", "members": ["Order", "OrderLine"]}}],
      "domain_events": ["OrderPlaced", "OrderCancelled"],
      "business_rules": ["Orders must have at least one item"]
    }}
  ]
}}

If a category has no data, use empty arrays. Do not invent data. Every aggregate.members entry must also appear in the same context's entities list."""

        sc = stage_config("Specialist")
        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=sc.temperature,
                        seed=sc.seed,
                    ),
                )
                
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
                        time.sleep(2)
                        continue
                    # Final fallback
                    return [
                        {"context": ctx, "analysis": {"error": "parse_failed"}}
                        for ctx in contexts
                    ]

                # Track token usage only for successful responses
                self.token_tracker.track_api_call(
                    response,
                    stage="Specialist",
                    operation="analyze_all_contexts"
                )

                if isinstance(result, dict) and "analyses" in result:
                    analyses = [
                        {"context": a.get("context", "Unknown"), "analysis": a}
                        for a in result["analyses"]
                    ]
                    # Save intermediate output
                    self._save_intermediate(
                        stage="3_specialist",
                        data={
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "contexts_analyzed": len(analyses),
                            "analyses": analyses
                        }
                    )
                    self._report_progress("Specialist", "completed", f"Analyzed {len(analyses)} contexts", 100)
                    return analyses
                self._report_progress("Specialist", "completed", "Analysis completed", 100)
                return [{"context": ctx, "analysis": {}} for ctx in contexts]

            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    print(f"   [WARN] Analysis error: {e}")
                    if retry >= 4:
                        self._report_progress("Specialist", "error", str(e), 100)
                        return [
                            {"context": ctx, "analysis": {"error": str(e)}}
                            for ctx in contexts
                        ]

        self._report_progress("Specialist", "error", "Retries exhausted", 100)
        return [
            {"context": ctx, "analysis": {"error": "retries_exhausted"}}
            for ctx in contexts
        ]

    # =========================================================================
    # STAGE 4: SYNTHESIZER - Create Final Model
    # =========================================================================

    def synthesize(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize all context analyses into a cohesive domain model.

        Resolves duplicates, ensures naming consistency, and produces
        the final JSON structure.
        """
        print("\n┌─────────────────────────────────────────────────────────────────┐")
        print("│ STAGE 4: SYNTHESIZER - Creating Final Domain Model             │")
        print("└─────────────────────────────────────────────────────────────────┘")
        print("  🔨 Merging analyses into cohesive model...")
        
        self._report_progress("Synthesizer", "started", "Creating final domain model", 0)

        prompt = f"""Synthesize Bounded Context analyses into a cohesive Domain Model.

SYNTHESIS RULES:
1. Resolve duplicates: Each entity belongs to ONE primary context.
2. Generate synonyms_to_avoid for each entity (common alternative names to flag for V1).
3. Define allowed_dependencies between contexts (which contexts can reference which).
4. Ensure naming consistency: PascalCase for all names.
5. Every aggregate must list its members (entity names that live inside the aggregate).

CONTEXT ANALYSES:
{json.dumps(analyses, indent=2)}

OUTPUT SCHEMA (must match exactly — populate every field even if empty):
{{
  "project_name": "ProjectNameDomainModel",
  "project_metadata": {{
    "version": "1.0",
    "generated_at": "YYYY-MM-DD",
    "description": "Brief description"
  }},
  "bounded_contexts": [
    {{
      "context_name": "ContextName",
      "description": "What this context manages",
      "allowed_dependencies": ["OtherContext1"],
      "ubiquitous_language": {{
        "entities": [{{
          "name": "EntityName",
          "description": "Entity purpose",
          "confidence": 0.85,
          "justification": "Mentioned in 3 SRS sentences as a primary actor",
          "synonyms_to_avoid": ["Synonym1", "Synonym2"]
        }}],
        "value_objects": [{{
          "name": "ValueObjectName",
          "attributes": ["attr1", "attr2"],
          "description": "Value object purpose"
        }}],
        "services": [{{
          "name": "ServiceName",
          "description": "Service responsibility"
        }}],
        "aggregates": [{{
          "name": "AggregateName",
          "description": "Aggregate consistency boundary",
          "members": ["EntityName1", "EntityName2"]
        }}],
        "domain_events": ["EventName1"]
      }}
    }}
  ],
  "global_rules": {{
    "naming_convention": "PascalCase",
    "banned_global_terms": ["Manager", "Util", "Helper", "Data", "Info"]
  }}
}}

CRITICAL: synonyms_to_avoid must be populated for V1 detection. Every aggregate.members must reference an entity in the same context. Do not invent data not present in the analyses."""

        sc = stage_config("Synthesizer")
        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=sc.temperature,
                        seed=sc.seed,
                    ),
                )

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
                    print(f"  ⚠️  JSON parse failed - Retry {retry + 1}/5")
                    if retry < 4:
                        time.sleep(2)
                        continue
                    print("  ❌ All retries failed - Using fallback model")
                    return self._create_fallback_model()

                # Verify required fields exist. A list-typed result (which can
                # arrive from _parse_json_response since Gemini occasionally
                # returns a top-level array) is treated as missing-everything
                # and falls through to the fallback path below.
                if not isinstance(result, dict):
                    print(f"  ⚠️  Synthesizer returned {type(result).__name__} not dict - Retry {retry + 1}/5")
                    if retry < 4:
                        time.sleep(2)
                        continue
                    print("  ❌ Non-dict response - Using fallback model")
                    return self._create_fallback_model()

                required = ["project_name", "project_metadata", "bounded_contexts"]
                missing = [f for f in required if f not in result]

                if missing:
                    print(f"  ⚠️  Missing fields: {', '.join(missing)} - Retry {retry + 1}/5")
                    if retry < 4:
                        time.sleep(2)
                        continue
                    print("  ❌ Incomplete response - Using fallback model")
                    return self._create_fallback_model()

                # Track token usage only for successful, valid responses
                self.token_tracker.track_api_call(
                    response,
                    stage="Synthesizer",
                    operation="synthesize_final_model"
                )
                
                # Save intermediate output
                self._save_intermediate(
                    stage="4_synthesizer",
                    data={
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "final_model": result
                    }
                )
                
                self._report_progress("Synthesizer", "completed", "Domain model created", 100)

                return result

            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    print(f"   [WARN] Synthesis error: {e}")
                    if retry >= 4:
                        self._report_progress("Synthesizer", "error", str(e), 100)
                        raise

        self._report_progress("Synthesizer", "error", "Retries exhausted", 100)
        raise Exception("Failed to synthesize after all retries")

    def synthesize_final_model(self, analyses: List[Dict[str, Any]]) -> DomainModel:
        """Synthesize per-context analyses into a final DomainModel.

        Phase A: Propagates Pydantic ValidationError instead of returning an
        empty fallback model. The previous bare except (FM-21) silently turned
        validation errors into successful empty-model responses, which is
        removed in Phase B together with _create_fallback_model.
        """
        raw_dict = self.synthesize(analyses)
        raw_dict = self._cleanup_domain_data(raw_dict)
        # Pydantic raises ValidationError on shape/constraint failures —
        # let it propagate. The narrow except below only catches the
        # KeyError that _cleanup_domain_data may surface from a stage-3
        # error payload, and re-raises with context.
        return DomainModel(**raw_dict)

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def analyze_document(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Run the full analysis pipeline on an SRS document.

        Returns context analyses ready for synthesis.
        """
        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + "  🚀 DOMAIN MODEL GENERATION PIPELINE STARTED".center(67) + "#")
        print("#" + " "*68 + "#")
        print("#"*70)
        print(f"\n  📊 Input Document: {len(raw_text):,} characters")

        try:
            # Stage 1: Extract domain sentences
            domain_sentences = self.extract_domain_sentences(raw_text)

            if not domain_sentences:
                domain_sentences = [raw_text[:1000]]

            # Stage 2: Identify contexts
            contexts = self.identify_contexts(domain_sentences)
            print(f"  ✅ Identified {len(contexts)} contexts: {', '.join(contexts)}\n")

            # Stage 3: Analyze contexts
            results = self.extract_all_contexts_details(contexts, domain_sentences)

            print(f"  ✅ Analyzed {len(results)} contexts:\n")
            for i, r in enumerate(results, 1):
                print(f"      {i}. {r['context']}")

            print("\n" + "="*70)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            print(f"  📊 Total API Requests: {self.request_count}")
            print(f"  🎯 Returning {len(results)} context analyses; caller invokes synthesize() next")
            print("="*70 + "\n")
            return results

        except Exception as e:
            print("\n" + "="*70)
            print("❌ PIPELINE FAILED")
            print("="*70)
            print(f"  Error: {e}")
            print("="*70 + "\n")
            import traceback
            traceback.print_exc()
            raise

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _cleanup_domain_data(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Structural-only normalization of the Synthesizer's raw JSON.

        Phase A6 (FM-20): no longer fabricates defaults. If the LLM omitted
        a field, leave it omitted — let downstream Pydantic validation or
        the Verifier surface the gap. The only transformation kept is
        coercing domain_events from `[{"name": "X"}]` (object form some
        models emit) to `["X"]` (list-of-string form the schema expects).
        """
        if "bounded_contexts" in json_data:
            for ctx in json_data["bounded_contexts"]:
                ul = ctx.get("ubiquitous_language", {})
                events = ul.get("domain_events")
                if isinstance(events, list):
                    ul["domain_events"] = [
                        e["name"] if isinstance(e, dict) and "name" in e else e
                        for e in events
                    ]
        return json_data

    def _create_fallback_model(self) -> Dict[str, Any]:
        """Create minimal valid model structure."""
        return {
            "project_name": "Generated Domain Model",
            "project_metadata": {
                "version": "1.0",
                "generated_at": time.strftime("%Y-%m-%d"),
                "description": "Auto-generated fallback model",
            },
            "bounded_contexts": [],
            "global_rules": {
                "naming_convention": "PascalCase",
                "banned_global_terms": ["Manager", "Util"],
            },
        }

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
