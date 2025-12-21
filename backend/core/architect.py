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
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from core.schemas import DomainModel, GlobalRules, ProjectMetadata
from core.token_tracker import TokenTracker
from config import ArchitectConfig

load_dotenv()


class DomainArchitect:
    """AI-powered domain model extraction from SRS documents."""

    LLMConfig = ArchitectConfig()

    def __init__(self, model: str = LLMConfig.MODEL_NAME):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model
        self.last_request_time = 0
        self.min_delay = 6.0
        self.request_count = 0
        self.token_tracker = TokenTracker.get_instance()

        print("\n" + "="*70)
        print("🏗️  DOMAIN ARCHITECT INITIALIZED")
        print("="*70)
        print(f"  Model: {model}")
        print(f"  Rate Limit: {self.min_delay}s between requests")
        print("="*70 + "\n")

    # =========================================================================
    # RATE LIMITING & ERROR HANDLING
    # =========================================================================

    def _wait_for_rate_limit(self):
        """Enforce minimum delay between API requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_delay:
            sleep_time = self.min_delay - elapsed
            print(f"  ⏳ Rate limiting... waiting {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
        print(f"  📡 API Request #{self.request_count}")

    def _handle_quota_error(self, error: Exception, retry_count: int) -> float:
        """Handle quota exceeded errors with exponential backoff."""
        error_str = str(error)
        is_quota_error = (
            "429" in error_str
            or "quota" in error_str.lower()
            or "ResourceExhausted" in str(type(error))
        )

        if not is_quota_error:
            return 0

        # Try to extract suggested retry delay
        retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
        if retry_match:
            wait_time = max(float(retry_match.group(1)), 10)
        else:
            # Exponential backoff: 15s, 30s, 60s, 120s
            wait_time = min(15 * (2**retry_count), 300)

        print(f"  ⚠️  QUOTA EXCEEDED - Backing off {wait_time:.1f}s...")
        time.sleep(wait_time)
        return wait_time

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

        chunk_size = 10000
        chunks = self._split_text_into_chunks(clean_text, chunk_size)
        all_sentences = []

        print(f"  📄 Document: {len(clean_text):,} characters")
        print(f"  🔪 Chunks: {len(chunks)} (max {chunk_size:,} chars each)")

        for i, chunk in enumerate(chunks):
            progress = (i + 1) / len(chunks) * 100
            print(f"  ▶️  Chunk {i + 1}/{len(chunks)} ({len(chunk):,} chars) [{progress:.0f}%]")
            sentences = self._extract_sentences_from_chunk(chunk, i + 1, len(chunks))
            all_sentences.extend(sentences)

        print(f"  ✅ Extracted {len(all_sentences)} domain-relevant sentences\n")
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
        prompt = f"""Extract ALL domain-relevant sentences from this SRS document chunk.

Focus on:
- Business entities and their attributes
- Business rules, constraints, and validations
- Workflows, processes, and use cases
- Data relationships and dependencies
- Business calculations and formulas

TEXT CHUNK {chunk_num}/{total_chunks}:
{chunk}

RESPOND WITH JSON:
{{
  "sentences": ["sentence1", "sentence2", ...]
}}

Extract ALL relevant sentences, no limit."""

        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                
                # Check if response was truncated
                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue
                
                result = self._parse_json_response(response.text)

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
                if self._handle_quota_error(e, retry) == 0:
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

        text = "\n".join(domain_sentences)
        max_chars = 50000
        if len(text) > max_chars:
            print(f"  ✂️  Truncating input: {len(text):,} → {max_chars:,} chars")
            text = text[:max_chars]

        prompt = f"""Analyze the domain knowledge below and identify distinct Bounded Contexts.

A Bounded Context is a cohesive business area with its own:
- Terminology and language
- Entities and rules
- Team ownership
- Change frequency

DOMAIN KNOWLEDGE:
{text}

RESPOND WITH JSON:
{{
  "contexts": ["ContextName1", "ContextName2", ...]
}}

Identify ALL contexts. Use business-meaningful names (e.g., OrderManagement)."""

        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )

                # Check if response was truncated
                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue
                
                result = self._parse_json_response(response.text)

                if (
                    isinstance(result, dict)
                    and result.get("error") == "json_parse_failed"
                ):
                    print(f"  ⚠️  Parse failed - Retry {retry + 1}/5")
                    if retry < 4:
                        continue
                    print("  ⚠️  Max retries reached, using fallback context")
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
                        return contexts
                elif isinstance(result, list) and len(result) > 0:
                    return result

                print(f"  ⚠️  Empty response - Retry {retry + 1}/5")
                if retry < 4:
                    continue
                return ["CoreDomain"]

            except Exception as e:
                if self._handle_quota_error(e, retry) == 0:
                    print(f"   [WARN] Context identification error: {e}")
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

        contexts_text = ", ".join(contexts)
        sentences_text = "\n".join(domain_sentences)

        max_chars = 60000
        if len(sentences_text) > max_chars:
            print(f"  ✂️  Truncating input: {len(sentences_text):,} → {max_chars:,} chars")
            sentences_text = sentences_text[:max_chars]

        prompt = f"""You are a DDD Expert. Analyze the domain knowledge for ALL these contexts: {contexts_text}

For EACH context, extract:
1. Aggregate Roots
2. Entities (with attributes)
3. Value Objects
4. Business Rules

DOMAIN KNOWLEDGE:
{sentences_text}

RESPOND WITH JSON:
{{
  "analyses": [
    {{
      "context": "ContextName1",
      "aggregate_roots": ["Root1", "Root2"],
      "entities": [{{"name": "Entity1", "attributes": ["attr1"]}}],
      "value_objects": [{{"name": "VO1", "attributes": ["attr1"]}}],
      "business_rules": ["Rule 1", "Rule 2"]
    }}
  ]
}}"""

        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )
                
                # Check if response was truncated
                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue
                
                result = self._parse_json_response(response.text)

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
                    return [
                        {"context": a.get("context", "Unknown"), "analysis": a}
                        for a in result["analyses"]
                    ]
                return [{"context": ctx, "analysis": {}} for ctx in contexts]

            except Exception as e:
                if self._handle_quota_error(e, retry) == 0:
                    print(f"   [WARN] Analysis error: {e}")
                    if retry >= 4:
                        return [
                            {"context": ctx, "analysis": {"error": str(e)}}
                            for ctx in contexts
                        ]

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

        prompt = f"""You are a Chief Software Architect. Synthesize the following Bounded Context analyses into ONE cohesive Domain Model JSON.

RULES:
1. Resolve duplicate entities intelligently
2. Maintain naming consistency
3. Generate realistic sample values
4. Place entities in their primary context
5. Create meaningful project metadata

CONTEXT ANALYSES:
{json.dumps(analyses, indent=2)}

RESPOND WITH JSON matching this schema:
{{
  "project_name": "ProjectName",
  "project_metadata": {{
    "version": "1.0",
    "generated_at": "2025-12-11",
    "description": "Description"
  }},
  "bounded_contexts": [
    {{
      "context_name": "ContextName",
      "description": "What this context does",
      "ubiquitous_language": {{
        "entities": [{{
          "name": "EntityName",
          "description": "Entity description",
          "synonyms_to_avoid": ["Synonym1"]
        }}],
        "value_objects": [{{
          "name": "ValueObjectName",
          "attributes": ["attribute1", "attribute2"],
          "description": "Value object description"
        }}],
        "domain_events": ["EventName1", "EventName2"]
      }},
      "allowed_dependencies": []
    }}
  ],
  "global_rules": {{
    "naming_convention": "PascalCase",
    "banned_global_terms": ["Manager", "Util"]
  }}
}}"""

        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                    ),
                )

                # Check if response was truncated
                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue
                
                result = self._parse_json_response(response.text)

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

                # Verify required fields exist
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

                return result

            except Exception as e:
                if self._handle_quota_error(e, retry) == 0:
                    print(f"   [WARN] Synthesis error: {e}")
                    if retry >= 4:
                        raise

        raise Exception("Failed to synthesize after all retries")

    def synthesize_final_model(self, analyses: List[Dict[str, Any]]) -> DomainModel:
        """Create validated DomainModel from analyses."""
        try:
            json_data = self.synthesize(analyses)
            cleaned_data = self._cleanup_domain_data(json_data)
            return DomainModel(**cleaned_data)
        except Exception as e:
            print(f"  ❌ Model creation error: {e}")
            print("  🔄 Creating minimal fallback model...")
            return DomainModel(
                project_name="Generated Domain Model",
                project_metadata=ProjectMetadata(
                    version="1.0",
                    generated_at=time.strftime("%Y-%m-%d"),
                    description="Auto-generated domain model from SRS document",
                ),
                bounded_contexts=[],
                global_rules=GlobalRules(
                    naming_convention="PascalCase",
                    banned_global_terms=["Manager", "Util", "Helper"],
                ),
            )

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
            print(f"  🎯 Ready for final synthesis")
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
        """Clean up JSON data to match DomainModel schema."""
        if "global_rules" not in json_data:
            json_data["global_rules"] = {
                "naming_convention": "PascalCase",
                "banned_global_terms": [],
            }

        if "bounded_contexts" in json_data:
            for context in json_data["bounded_contexts"]:
                if "allowed_dependencies" not in context:
                    context["allowed_dependencies"] = None

                ub_lang = context.get("ubiquitous_language", {})

                # Ensure entities have synonyms_to_avoid
                for entity in ub_lang.get("entities", []):
                    if isinstance(entity, dict) and "synonyms_to_avoid" not in entity:
                        entity["synonyms_to_avoid"] = None

                # Ensure value objects have attributes
                for vo in ub_lang.get("value_objects", []):
                    if isinstance(vo, dict) and "attributes" not in vo:
                        vo["attributes"] = []

                # Ensure domain events are strings
                events = ub_lang.get("domain_events", [])
                if isinstance(events, list):
                    cleaned = []
                    for event in events:
                        if isinstance(event, dict) and "name" in event:
                            cleaned.append(event["name"])
                        elif isinstance(event, str):
                            cleaned.append(event)
                    ub_lang["domain_events"] = cleaned

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
    
    def _check_response_completion(self, response, retry: int) -> bool:
        """
        Check if API response was complete or truncated.
        
        Returns:
            True if response is complete and valid
            False if truncated (needs retry)
        """
        if not response.candidates:
            print(f"      ⚠️  No candidates in response")
            return False
            
        finish_reason = response.candidates[0].finish_reason
        
        if finish_reason == "STOP":
            return True  # Normal completion
            
        # Response was cut off
        print(f"      ⚠️  Response incomplete: finish_reason={finish_reason}")
        
        if finish_reason == "MAX_TOKENS":
            print(f"      💡 Hit token limit ({self.LLMConfig.MAX_OUTPUT_TOKENS})")
            print(f"      💡 Response was cut mid-generation")
        elif finish_reason == "SAFETY":
            print(f"      🛡️  Response blocked by safety filters")
        elif finish_reason == "RECITATION":
            print(f"      📝 Response blocked due to citation/recitation")
        elif finish_reason == "OTHER":
            print(f"      ❓ Response stopped for unknown reason")
            
        if retry < 4:
            print(f"      🔄 Retrying ({retry + 1}/5)...")
            return False
        else:
            print(f"      ❌ Max retries reached")
            return False

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response. Simple strategy since we use response_mime_type='application/json'."""
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
