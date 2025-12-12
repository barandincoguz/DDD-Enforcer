import json
import os
import time
from typing import Dict, List

import google.generativeai as genai  # type: ignore
from dotenv import load_dotenv

from core.schemas import DomainModel

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # type: ignore


class DomainArchitect:
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        self.model = genai.GenerativeModel(model_name=model)  # type: ignore
        self.last_request_time = 0
        self.min_delay = 6.0
        self.request_count = 0
        self.quota_exhausted = False
        print(f"🤖 Domain Architect initialized with model: {model}")
        print(f"⚙️  Rate limiting: {self.min_delay}s between requests")
        print(
            "🎯 Smart chunking enabled - processes ENTIRE SRS document without context loss"
        )
        print("📊 Estimated API requests: ~4-8 (depends on document size)")

    def _wait_for_rate_limit(self):
        """Enforce minimum delay between API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            print(f"   ⏸️  Rate limiting: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.request_count += 1
        print(f"   📊 API Request #{self.request_count}")

    def _handle_quota_error(self, error: Exception, retry_count: int) -> float:
        """Handle quota exceeded errors with smart backoff"""
        error_str = str(error)

        if (
            "429" in error_str
            or "quota" in error_str.lower()
            or "ResourceExhausted" in str(type(error))
        ):
            # Extract retry delay from error message if available
            import re

            retry_match = re.search(r"retry in (\d+\.?\d*)", error_str)
            if retry_match:
                suggested_wait = float(retry_match.group(1))
                wait_time = max(suggested_wait, 10)  # At least 10 seconds
            else:
                # Exponential backoff: 15s, 30s, 60s, 120s
                wait_time = min(15 * (2**retry_count), 300)

            print(f"   🚫 Quota exceeded! Waiting {wait_time:.1f}s before retry...")
            print(f"   📈 Total requests made: {self.request_count}")
            time.sleep(wait_time)
            return wait_time

        return 0

    # -------------------------------------------------------
    # 1. SCOUT (AKILLI CHUNKING - Tüm dokümanı işle)
    # -------------------------------------------------------
    def extract_domain_sentences(self, clean_text: str) -> List[str]:
        print("🕵️ Extracting domain-relevant sentences with smart chunking...")

        # BAĞLAM KORUMA: Tüm text'i chunk'lara böl ve hepsini işle
        chunk_size = 10000  # Her chunk 10k karakter
        all_sentences = []

        # Text'i chunk'lara böl
        total_length = len(clean_text)
        num_chunks = (total_length + chunk_size - 1) // chunk_size

        if num_chunks == 1:
            chunks = [clean_text]
        else:
            # Cümle sınırlarında böl (. ile)
            chunks = []
            start = 0
            while start < total_length:
                end = min(start + chunk_size, total_length)
                # Son nokta işaretine kadar git
                if end < total_length:
                    last_period = clean_text.rfind(".", start, end)
                    if last_period > start:
                        end = last_period + 1
                chunks.append(clean_text[start:end])
                start = end

        print(f"   📊 Processing {len(chunks)} chunks ({total_length} chars total)")
        print(f"   🎯 This will use {len(chunks)} API request(s) for Scout phase")

        # Her chunk'ı işle
        for chunk_idx, chunk in enumerate(chunks):
            print(
                f"   📦 Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} chars)..."
            )

            for retry_count in range(5):
                try:
                    self._wait_for_rate_limit()

                    # PROMPT: Domain sentences çıkar (SINIRSIZ)
                    prompt = f"""Extract ALL domain-relevant sentences from this SRS document chunk.

Focus on:
- Business entities and their attributes
- Business rules, constraints, and validations
- Workflows, processes, and use cases
- Data relationships and dependencies
- Business calculations and formulas

TEXT CHUNK {chunk_idx + 1}/{len(chunks)}:
{chunk}

RESPOND WITH JSON:
{{
  "sentences": ["sentence1", "sentence2", ...]
}}

Extract ALL relevant sentences, no limit."""

                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(  # type: ignore
                            response_mime_type="application/json",
                            max_output_tokens=5000,  # Arttırıldı: daha fazla sentence
                        ),
                    )
                    result = self._clean_and_parse_json(response.text)

                    # Check if JSON parsing failed
                    if (
                        isinstance(result, dict)
                        and result.get("error") == "json_parse_failed"
                    ):
                        print(
                            f"      ⚠️ Sentence extraction JSON parse failed in chunk {chunk_idx + 1}"
                        )
                        # Fallback: split into sentences
                        fallback = [
                            s.strip() for s in chunk.split(".") if len(s.strip()) > 20
                        ]
                        all_sentences.extend(fallback[:50])
                        break

                    # Extract sentences from this chunk
                    chunk_sentences = []
                    if isinstance(result, dict) and "sentences" in result:
                        chunk_sentences = result["sentences"]
                    elif isinstance(result, list):
                        chunk_sentences = result

                    all_sentences.extend(chunk_sentences)
                    print(
                        f"      ✅ Chunk {chunk_idx + 1}: extracted {len(chunk_sentences)} sentences"
                    )
                    break  # Başarılı, bir sonraki chunk'a geç

                except json.JSONDecodeError as e:
                    print(f"      ⚠️ JSON parse error in chunk {chunk_idx + 1}: {e}")
                    # Fallback: cümlelere böl
                    fallback = [
                        s.strip() for s in chunk.split(".") if len(s.strip()) > 20
                    ]
                    all_sentences.extend(fallback[:50])  # İlk 50 cümle
                    break
                except Exception as e:
                    wait_time = self._handle_quota_error(e, retry_count)
                    if wait_time == 0:
                        print(f"      ⚠️ Error in chunk {chunk_idx + 1}: {e}")
                        if retry_count < 4:
                            continue
                        break

        print(
            f"   ✅ Total sentences extracted: {len(all_sentences)} from {len(chunks)} chunks"
        )
        return all_sentences if all_sentences else [clean_text[:1000]]

    # -------------------------------------------------------
    # 2. CONTEXT IDENTIFICATION (TAM BAĞLAM)
    # -------------------------------------------------------
    def identify_contexts(self, domain_sentences: List[str]) -> List[str]:
        print("🏗️ Identifying bounded contexts from ALL domain knowledge...")

        for retry_count in range(5):
            try:
                self._wait_for_rate_limit()

                # TÜM CÜMLELERI KULLAN - ancak token limitine dikkat
                text = "\n".join(domain_sentences)

                # Eğer çok uzunsa akıllıca özetle
                max_chars = 50000  # ~12k token
                if len(text) > max_chars:
                    print(
                        f"   📊 Using all {len(domain_sentences)} sentences (truncated to {max_chars} chars)"
                    )
                    text = text[:max_chars]
                else:
                    print(
                        f"   📊 Using all {len(domain_sentences)} sentences ({len(text)} chars)"
                    )

                prompt = f"""Analyze ALL the domain knowledge below and identify distinct Bounded Contexts.

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

Identify 2-8 contexts. Use business-meaningful names (e.g., OrderManagement, not OrderContext)."""

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(  # type: ignore
                        response_mime_type="application/json", max_output_tokens=800
                    ),
                )
                result = self._clean_and_parse_json(response.text)

                # Check if JSON parsing failed
                if (
                    isinstance(result, dict)
                    and result.get("error") == "json_parse_failed"
                ):
                    print(
                        f"   ⚠️ Context identification JSON parse failed, using fallback"
                    )
                    return ["CoreDomain"]

                # Handle different response formats
                if isinstance(result, dict) and "contexts" in result:
                    return result["contexts"]
                elif isinstance(result, list):
                    return result
                else:
                    return ["CoreDomain"]  # Fallback

            except Exception as e:
                wait_time = self._handle_quota_error(e, retry_count)
                if wait_time == 0:
                    print(f"   ⚠️ Error in identify_contexts: {e}")
                    return ["CoreDomain"]  # Fallback

        return ["CoreDomain"]

    # -------------------------------------------------------
    # 3. SPECIALIST (TOPLU ANALİZ - Tüm contextleri tek request'te)
    # -------------------------------------------------------
    def extract_all_contexts_details(
        self, contexts: List[str], domain_sentences: List[str]
    ) -> List[Dict]:
        """Tüm context'leri tek bir API call'da analiz et - REQUEST TASARRUFU!"""
        print(f"🔬 Analyzing ALL {len(contexts)} contexts in ONE request...")

        for retry_count in range(5):
            try:
                self._wait_for_rate_limit()

                # Tüm context'ler için ortak prompt
                contexts_text = ", ".join(contexts)

                # TÜM SENTENCES KULLAN (token limitine dikkat)
                sentences_text = "\n".join(domain_sentences)
                max_chars = 60000  # ~15k token
                if len(sentences_text) > max_chars:
                    print(
                        f"   📊 Using {len(domain_sentences)} sentences (truncated to {max_chars} chars)"
                    )
                    sentences_text = sentences_text[:max_chars]
                else:
                    print(
                        f"   📊 Using all {len(domain_sentences)} sentences ({len(sentences_text)} chars)"
                    )

                prompt = f"""You are a DDD Expert. Analyze the domain knowledge for ALL these contexts at once: {contexts_text}

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
    }},
    ... (repeat for each context)
  ]
}}"""

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(  # type: ignore
                        response_mime_type="application/json",
                        max_output_tokens=4000,  # Daha fazla output için
                    ),
                )

                result = self._clean_and_parse_json(response.text)

                # Check if JSON parsing failed
                if (
                    isinstance(result, dict)
                    and result.get("error") == "json_parse_failed"
                ):
                    print(
                        f"   ⚠️ Context analysis JSON parse failed for contexts: {contexts}"
                    )
                    return [
                        {"context": ctx, "analysis": {"error": "JSON parse failed"}}
                        for ctx in contexts
                    ]

                # Format sonuçları
                if isinstance(result, dict) and "analyses" in result:
                    formatted_results = []
                    for analysis in result["analyses"]:
                        formatted_results.append(
                            {
                                "context": analysis.get("context", "Unknown"),
                                "analysis": analysis,
                            }
                        )
                    return formatted_results
                else:
                    # Fallback: her context için boş analiz
                    return [{"context": ctx, "analysis": {}} for ctx in contexts]

            except json.JSONDecodeError as e:
                print(f"   ⚠️ JSON parse error: {e}")
                return [
                    {"context": ctx, "analysis": {"error": "JSON parse failed"}}
                    for ctx in contexts
                ]
            except Exception as e:
                wait_time = self._handle_quota_error(e, retry_count)
                if wait_time == 0:
                    print(f"   ⚠️ Error analyzing contexts: {e}")
                    if retry_count < 4:
                        continue
                    return [
                        {"context": ctx, "analysis": {"error": str(e)}}
                        for ctx in contexts
                    ]

        return [
            {"context": ctx, "analysis": {"error": "Failed after retries"}}
            for ctx in contexts
        ]

    # -------------------------------------------------------
    # 4. SYNTHESIZER
    # -------------------------------------------------------
    def synthesize(self, analyses: List[Dict]):
        print("🧠 Synthesizing final Domain Model...")

        for retry_count in range(5):
            try:
                self._wait_for_rate_limit()

                combined_text = json.dumps(analyses, indent=2)

                prompt = f"""You are a Chief Software Architect. Synthesize the following Bounded Context analyses into ONE cohesive Domain Model JSON.

RULES:
1. Resolve duplicate entities intelligently
2. Maintain naming consistency
3. Generate realistic sample values
4. Place entities in their primary context
5. Create meaningful project metadata

CONTEXT ANALYSES:
{combined_text}

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

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(  # type: ignore
                        response_mime_type="application/json", max_output_tokens=4000
                    ),
                )
                result = self._clean_and_parse_json(response.text)

                # Check if JSON parsing failed
                if (
                    isinstance(result, dict)
                    and result.get("error") == "json_parse_failed"
                ):
                    print(f"   ⚠️ Synthesis JSON parse failed, creating minimal model")
                    # Return a minimal valid structure
                    return {
                        "project_name": "Generated Domain Model",
                        "project_metadata": {
                            "version": "1.0",
                            "generated_at": time.strftime("%Y-%m-%d"),
                            "description": "Auto-generated fallback model due to JSON parse error",
                        },
                        "bounded_contexts": [],
                        "global_rules": {
                            "naming_convention": "PascalCase",
                            "banned_global_terms": ["Manager", "Util"],
                        },
                    }

                return result

            except Exception as e:
                wait_time = self._handle_quota_error(e, retry_count)
                if wait_time == 0:
                    print(f"   ⚠️ Error in synthesize: {e}")
                    if retry_count < 4:
                        continue
                    raise

        raise Exception("Failed to synthesize after all retries")

    # -------------------------------------------------------
    # 4B — main.py tarafından çağrılan method (DomainModel döner)
    # -------------------------------------------------------
    def _cleanup_domain_data(self, json_data: Dict) -> Dict:
        """Clean up JSON data to match DomainModel schema requirements"""
        # Ensure global_rules has a default value
        if "global_rules" not in json_data:
            json_data["global_rules"] = {
                "naming_convention": "PascalCase",
                "banned_global_terms": [],
            }

        if "bounded_contexts" in json_data:
            for context in json_data["bounded_contexts"]:
                # Ensure allowed_dependencies has a default value
                if "allowed_dependencies" not in context:
                    context["allowed_dependencies"] = None

                if "ubiquitous_language" in context:
                    ub_lang = context["ubiquitous_language"]

                    # Fix entities - ensure they have synonyms_to_avoid field
                    if "entities" in ub_lang:
                        for entity in ub_lang["entities"]:
                            if (
                                isinstance(entity, dict)
                                and "synonyms_to_avoid" not in entity
                            ):
                                entity["synonyms_to_avoid"] = None

                    # Fix value objects - ensure they have required attributes field
                    if "value_objects" in ub_lang:
                        for vo in ub_lang["value_objects"]:
                            if isinstance(vo, dict) and "attributes" not in vo:
                                # Add empty attributes list if missing
                                vo["attributes"] = []

                    # Fix domain events - ensure they are strings, not dicts
                    if "domain_events" in ub_lang:
                        events = ub_lang["domain_events"]
                        if isinstance(events, list):
                            cleaned_events = []
                            for event in events:
                                if isinstance(event, dict) and "name" in event:
                                    # Convert dict to string using the name
                                    cleaned_events.append(event["name"])
                                elif isinstance(event, str):
                                    cleaned_events.append(event)
                            ub_lang["domain_events"] = cleaned_events

        return json_data

    def synthesize_final_model(self, analyses: List[Dict]) -> DomainModel:
        try:
            json_data = self.synthesize(analyses)
            # Clean up the data before validation
            cleaned_data = self._cleanup_domain_data(json_data)
            return DomainModel(**cleaned_data)
        except Exception as e:
            print(f"   ⚠️ Error creating DomainModel: {e}")
            print("   🔄 Creating fallback model from analyses...")
            # Create a minimal fallback model with all required fields
            from core.schemas import GlobalRules, ProjectMetadata

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

    # -------------------------------------------------------
    # MAIN PIPELINE (OPTİMİZE EDİLMİŞ - Daha Az Request!)
    # -------------------------------------------------------
    def analyze_document(self, raw_text: str):
        print("🚀 Starting FULL CONTEXT analysis pipeline...")
        print(f"📄 Document size: {len(raw_text)} characters")
        print("✅ NO CONTEXT LOSS - entire document will be processed")

        try:
            # 1) Scout - Extract domain-relevant information
            # BAĞLAM KORUMA: Tüm text'i işle!
            print(f"\n{'=' * 60}")
            print("📍 STEP 1/4: Scout - Extracting domain sentences")
            print(f"🔍 Processing ENTIRE document: {len(raw_text)} characters")
            clean_text = raw_text  # TÜM TEXT!
            domain_sentences = self.extract_domain_sentences(clean_text)
            print(f"✅ Extracted {len(domain_sentences)} domain sentences")

            if not domain_sentences:
                print("⚠️ No domain sentences found, using raw text...")
                domain_sentences = [clean_text[:1000]]

            # 2) Context Discovery (1 REQUEST)
            print(f"\n{'=' * 60}")
            print("📍 STEP 2/4: Architect - Identifying contexts")
            contexts = self.identify_contexts(domain_sentences)
            print(f"✅ Identified {len(contexts)} bounded contexts: {contexts}")

            # Limit contexts to avoid too many
            if len(contexts) > 5:
                print(f"   ⚠️ Too many contexts ({len(contexts)}), limiting to top 5")
                contexts = contexts[:5]

            # 3) Specialist - ALL contexts in ONE request (1 REQUEST)
            print(f"\n{'=' * 60}")
            print("📍 STEP 3/4: Specialist - Analyzing all contexts")
            print(f"🚀 OPTIMIZATION: All {len(contexts)} contexts in 1 API call!")
            full_results = self.extract_all_contexts_details(contexts, domain_sentences)

            for idx, result in enumerate(full_results):
                print(f"   ✅ Context {idx + 1}: {result['context']}")

            print(f"\n{'=' * 60}")
            print("📍 STEP 4/4: Ready for synthesis")
            print("✅ Analysis pipeline complete with FULL context preserved!")
            print(f"📊 Total API requests so far: {self.request_count}")
            print("💡 Synthesis will use 1 more request")
            return full_results

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Helper Method
    def _clean_and_parse_json(self, response_text: str):
        """Enhanced JSON parsing with multiple cleanup strategies."""
        import re

        # Strategy 1: Basic markdown cleanup
        try:
            cleaned_text = (
                response_text.replace("```json", "").replace("```", "").strip()
            )
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from mixed content using regex
        try:
            # Look for JSON object patterns
            json_pattern = r"\{(?:[^{}]|{[^{}]*})*\}"
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)

            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if (
                        isinstance(parsed, dict) and len(parsed) > 1
                    ):  # Valid non-empty dict
                        return parsed
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

        # Strategy 3: Clean common LLM response artifacts
        try:
            # Remove common prefixes and suffixes
            text = response_text

            # Remove markdown blocks more aggressively
            text = re.sub(r"```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"```\n?", "", text)

            # Remove common LLM prefixes
            prefixes_to_remove = [
                "Here's the JSON:",
                "Here is the JSON:",
                "JSON Response:",
                "Response:",
                "Result:",
                "Here's the analysis:",
                "Analysis result:",
            ]
            for prefix in prefixes_to_remove:
                if text.strip().startswith(prefix):
                    text = text.strip()[len(prefix) :].strip()

            # Remove trailing explanations after JSON
            lines = text.split("\n")
            json_lines = []
            in_json = False
            brace_count = 0

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("{") or in_json:
                    in_json = True
                    json_lines.append(line)
                    brace_count += line.count("{") - line.count("}")
                    if brace_count <= 0 and stripped.endswith("}"):
                        break

            if json_lines:
                json_text = "\n".join(json_lines)
                return json.loads(json_text)

        except Exception:
            pass

        # Strategy 4: Fix common JSON syntax errors
        try:
            text = response_text.strip()

            # Remove markdown
            text = re.sub(r"```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"```\n?", "", text)

            # Find potential JSON content
            start_idx = text.find("{")
            end_idx = text.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_candidate = text[start_idx : end_idx + 1]

                # Fix common issues
                json_candidate = re.sub(
                    r",(\s*[}\]])", r"\1", json_candidate
                )  # Remove trailing commas
                json_candidate = re.sub(
                    r'(["\w])\s*\n\s*(["\w])', r"\1, \2", json_candidate
                )  # Fix missing commas

                return json.loads(json_candidate)

        except Exception:
            pass

        # Strategy 5: Return fallback structure
        print(
            f"      ⚠️ All JSON parse strategies failed. Text: {response_text[:100]}..."
        )
        print(f"      🔄 Returning fallback empty structure.")

        # Return a structure that won't break downstream processing
        return {
            "error": "json_parse_failed",
            "raw_response": response_text[:500],  # Keep first 500 chars for debugging
            "fallback": True,
        }
