# Component Catalog — Domain-pipeline IN scope

Populated by Explore subagent 2026-05-21 01:36 GMT+3. Each row: `path | LOC_effective | responsibility | audit_priority | audit_status`.

`audit_status` ∈ {PENDING, IN-PROGRESS, DONE, DEFERRED}. Lower `audit_priority` = audited first.

## Inventory

| path | LOC_effective | responsibility | audit_priority | audit_status |
|---|---|---|---|---|
| extension/backend/core/document_parser.py | 182 | Parse SRS docs (DOCX/PDF/TXT) + extract domain-relevant content via regex + content filtering. | 1 | DONE (10 findings F-1..F-10) |
| extension/backend/core/document_parser_readers.py | 127 | Format-specific readers (DOCX/PDF/TXT) with content extraction + text normalization. | 1 | DONE (covered with parser.py) |
| extension/backend/core/architect.py | 752 | 4-stage pipeline orchestrator (Scout→Architect→Specialist→Synthesizer) with threading, retry, LLM integration. | 2 | PENDING |
| extension/backend/core/orchestration/pipeline.py | 60 | Typed stage driver with dependency injection for stage callables. | 2 | PENDING |
| extension/backend/core/orchestration/errors.py | 51 | Exception classes for orchestration failures (ArchitectExtractionError, SpecialistFailureError, SynthesizerEmptyModelError). | 2 | PENDING |
| extension/backend/core/pipeline_contracts.py | 113 | Pydantic envelopes for stage boundaries (ScoutOutput, ArchitectOutput, SpecialistAnalysis, VerifierResult, SynthesizerOutput). | 3 | PENDING |
| extension/backend/core/schemas.py | 169 | Domain model content classes (Entity, ValueObject, Aggregate, Service, DomainEvent) with validation + serialization. | 3 | PENDING |
| extension/backend/core/llm/base.py | 62 | Abstract LLMClient interface; common retry/rate-limit contract for Gemini + Ollama. | 4 | PENDING |
| extension/backend/core/llm/registry.py | 150 | LLM provider registry; model factory; routing to Gemini/Ollama by config. | 4 | PENDING |
| extension/backend/core/llm/retry.py | 74 | Retry decorator with exponential backoff + transient failure classification. | 4 | PENDING |
| extension/backend/core/llm/errors.py | 49 | LLM exception classes (RateLimitError, InvalidResponseError, LLMClientError). | 4 | PENDING |
| extension/backend/core/llm/gemini.py | 232 | Google Gemini client; streaming, tool use, structured output, response validation. | 4 | PENDING |
| extension/backend/core/llm/ollama.py | 176 | Ollama local LLM client; streaming + JSON-only response parsing. | 4 | PENDING |
| extension/backend/core/llm/_response_adapter.py | 39 | Bridge adapter converting Gemini responses to internal LLMResponse shape. | 4 | PENDING |
| extension/backend/core/llm/schema_probe.py | 335 | JSON schema compliance inspector for LLM responses with detailed error diagnostics. | 4 | PENDING |
| extension/backend/core/llm/__init__.py | 51 | LLM package init exporting base client interface + response types. | 4 | PENDING |
| extension/backend/core/synthesizer/__init__.py | 63 | Synthesizer orchestrator binding merge→validate→enrich→refine. | 5 | PENDING |
| extension/backend/core/synthesizer/merge.py | 45 | Deterministic multi-context merger with deduplication + conflict resolution. | 5 | PENDING |
| extension/backend/core/synthesizer/enrich.py | 73 | Post-merge model enrichment with cross-refs, implicit rules, semantic links. | 5 | PENDING |
| extension/backend/core/synthesizer/metadata.py | 14 | Synthesizer metadata container (context merge stats + field-level change tracking). | 5 | PENDING |
| extension/backend/core/synthesizer/errors.py | 11 | Synthesizer exception classes (MergeConflictError, EnrichmentError). | 5 | PENDING |
| extension/backend/core/verifier/checks_deterministic.py | 122 | D1–D5 deterministic checks (evidence nonempty, unique names, context references). | 6 | PENDING |
| extension/backend/core/verifier/checks_semantic.py | 51 | S1–S3 semantic checks (completeness, coherence, consistency via LLM). | 6 | PENDING |
| extension/backend/core/verifier/checks_semantic_d6_d7_d8.py | 97 | Specialized semantic checks D6–D8 for aggregates, entities, VO integrity. | 6 | PENDING |
| extension/backend/core/verifier/types.py | 21 | VerifierResult + ViolationReport Pydantic models. | 6 | PENDING |
| extension/backend/core/verifier/__init__.py | 3 | Verifier package init. | 6 | PENDING |
| extension/backend/core/refiner/loop.py | 32 | Feedback loop: apply verifier violations → re-prompt specialist/synthesizer until clean. | 7 | PENDING |
| extension/backend/core/refiner/prompts.py | 33 | Refiner prompt templates for violation remediation context. | 7 | PENDING |
| extension/backend/core/refiner/__init__.py | 3 | Refiner package init. | 7 | PENDING |
| extension/backend/core/AST/ast_signal_types.py | 99 | Signal type definitions + classification enums (ASTSignalType, Confidence). | 9 | PENDING |
| extension/backend/core/AST/ast_signal_discovery.py | 200 | AST traversal for signal discovery with visitor pattern + code element classification. | 9 | PENDING |
| extension/backend/core/AST/ast_signal_grounding.py | 39 | Map AST signals to source line ranges + file paths. | 9 | PENDING |
| extension/backend/core/AST/ast_signal_utils.py | 56 | Utility functions for AST traversal, filtering, signal normalization. | 9 | PENDING |
| extension/backend/core/AST/ast_signal_enrichment.py | 292 | Semantic enrichment of signals via DDD pattern heuristics + relationship inference. | 9 | PENDING |
| extension/backend/core/AST/ast_signal_classification.py | 233 | Multi-layer classification of AST signals into DDD patterns (Entity, ValueObject, Service). | 9 | PENDING |
| extension/backend/core/AST/ast_model_signals.py | 74 | ASTModelSignalExtractor orchestrator binding discovery→grounding→enrichment→classification. | 9 | PENDING |
| extension/backend/core/AST/__init__.py | 3 | AST package init. | 9 | PENDING |
| extension/backend/core/token_tracker.py | 117 | Thread-safe per-call token counting with cumulative reporting + quota enforcement. | 10 | PENDING |
| extension/backend/core/token_tracker_report.py | 164 | Token usage report aggregation + display (by provider, model, stage). | 10 | PENDING |
| extension/backend/core/token_tracker_types.py | 52 | TokenCount + TokenReport Pydantic models. | 10 | PENDING |
| extension/backend/configs/models.py | 150 | Model selection registry mapping stages to LLM provider/model pairs. | 11 | PENDING |
| extension/backend/main.py | 726 | FastAPI server with /analyze endpoint; orchestrates parsing, architect, validation, RAG, streaming. | 12 | PENDING |
| extension/backend/config.py | 104 | Centralized config: paths (domain_dir, inputs_dir), LLM keys, parser settings, model selection. | 12 | PENDING |

## Tests in scope (close-lookup batched at priority 12)

| test file | purpose |
|---|---|
| tests/test_document_parser.py | Document parser unit tests. |
| tests/test_architect_facade.py | Architect orchestrator contract tests. |
| tests/test_architect_extraction_error.py | Architect error handling + retry. |
| tests/test_architect_helpers.py | Architect helper function tests. |
| tests/test_llm_base.py | LLM base client interface tests. |
| tests/test_llm_registry.py | LLM provider registry routing tests. |
| tests/test_llm_gemini.py | Gemini client integration tests. |
| tests/test_llm_ollama.py | Ollama client integration tests. |
| tests/test_llm_retry.py | Retry decorator + backoff tests. |
| tests/test_llm_schema_probe.py | JSON schema compliance validation tests. |
| tests/test_pipeline_contracts.py | Stage boundary Pydantic contract validation tests. |
| tests/test_orchestration_errors.py | Orchestration exception handling tests. |
| tests/test_synthesizer_deterministic_merge.py | Merge dedup + conflict resolution tests. |
| tests/test_synthesizer_enrich.py | Synthesizer enrichment post-processing tests. |
| tests/test_synthesizer_empty_model_error.py | Synthesizer empty-result error handling. |
| tests/test_synthesizer_replay_historical.py | Synthesizer historical replay + determinism tests. |
| tests/test_verifier_deterministic.py | D1–D5 deterministic check unit tests. |
| tests/test_verifier_semantic.py | S1–S3 semantic check tests. |
| tests/test_verifier_d6_d7_d8.py | D6–D8 specialized semantic check tests. |
| tests/test_verifier_types.py | VerifierResult + violation model serialization tests. |
| tests/test_refiner_loop.py | Refiner feedback loop + violation remediation tests. |
| tests/test_ast_extractor.py | AST signal discovery + extraction tests. |
| tests/test_ast_grounding_strict.py | AST signal line-grounding accuracy tests. |
| tests/test_ast_collect_signals_raises.py | AST signal error handling tests. |
| tests/test_ast_model_signals_enrichment.py | AST signal enrichment + classification integration tests. |
| tests/test_token_tracker_v2.py | Token counting + quota enforcement tests. |
| tests/test_token_tracker_concurrency.py | Token tracker thread-safety tests. |
| tests/test_models_registry.py | Model selection registry + stage routing tests. |
| tests/test_p3_integration.py | End-to-end 4-stage pipeline integration tests. |
| tests/test_pipeline_orchestration.py | Pipeline stage driver + DI tests. |
| tests/test_specialist_boundary_parse.py | Specialist output contract parsing tests. |
| tests/test_specialist_per_context_loop.py | Per-context Specialist retry + refinement tests. |
| tests/test_scout_chunking.py | Scout section-aware chunking strategy tests. |
| tests/test_grounding_regression.py | Regression tests for AST signal grounding accuracy. |
| tests/test_registry_snapshot.py | LLM registry snapshot + provider selection tests. |
| tests/test_schemas_strict.py | Domain model schema validation + serialization tests. |

(Out-of-scope tests skipped: tests touching RAG, code_parser, or validator-only path.)

## Anomalies (Explore subagent observations — NO fix yet)

- **Stale `__pycache__/llm_client.cpython-313.pyc`** — ghost binary from removed `llm_client.py` (WP-01a deletion). Indicates incomplete refactor cleanup; harmless but noisy. Defer to a `chore(cleanup):` follow-up.
- **`core/llm/validator.py` 479 LOC OUT-of-scope** — legacy validate-path facade retained for backwards compat. Domain pipeline does not touch this; do not audit unless RQ4 work pulls it in.
- **`core/parser.py` exists but OUT-scope** — facade for `code_parser/`, not part of SRS→model.json path.
- **AST + token_tracker orthogonal to Scout→Architect→Specialist→Synthesizer→Verifier core** — integrated as cross-validation (AST) + accounting (token_tracker), not on hot path. Lower priority for fixes affecting domain-extraction quality.
- **`tests/test_violations.py` excluded** — validator-only path; not relevant to model.json generation.

---

**Last refresh:** 2026-05-21 01:36 GMT+3
