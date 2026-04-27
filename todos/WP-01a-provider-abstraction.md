# WP-01a: Provider Abstraction Layer (LLMClient ABC)

**Owner:** Ali
**Depends-on:** [WP-00]
**Effort:** M (1.5 weeks for one engineer; less if `llm_client.py` already provides 80% of the structure — see context report Q8)
**Status:** TODO
**Addresses instructor feedback:** [Hoca-1] (enables M=4 model run for RQ2)

## Goal

Wrap the existing Gemini-only `LLMClient` (`extension/backend/core/llm_client.py`) in a provider-agnostic interface so the same pipeline code can run against Gemini, OpenAI, Anthropic, and one OSS model without code edits. This is the gating dependency for **WP-04 RQ2 model comparison**: the paper claims a 4-provider study, but the codebase currently instantiates only Gemini (`00-context-report.md` Q8–Q10). After WP-01a, swapping providers is one config change, not a code edit.

**Important context:** `00-context-report.md` Q8 confirms `LLMClient` already centralizes Gemini calls and parameterizes via `AnalyzerConfig`. WP-01a is therefore a **refactor + extend**, not a from-scratch build. Estimate stays M (not L) because of this.

## Acceptance criteria

- [ ] Abstract `LLMClient` base class in `extension/backend/core/llm/base.py` with method `complete(prompt: str, schema: Type[BaseModel], **kwargs) -> StructuredResponse`.
- [ ] Concrete implementations:
  - [ ] `GeminiClient` (refactored from existing) — reuses current Pydantic flow.
  - [ ] `OpenAIClient` — uses `openai.beta.chat.completions.parse(response_format=...)` for structured output.
  - [ ] `AnthropicClient` — uses tool-use trick (`tool_choice={"type":"tool","name":"emit_violations"}`) to enforce schema.
  - [ ] `LocalClient` — wraps Ollama or vLLM with a JSON-mode prompt postfix; tolerates schema-noncompliance with one retry.
- [ ] Provider selection driven by `configs/model.yaml` (sibling of `configs/scope.yaml`). Single env-var indirection: `LLM_PROVIDER=anthropic`.
- [ ] `StructuredResponse` carries: `parsed: BaseModel`, `prompt_tokens: int`, `completion_tokens: int`, `latency_seconds: float`, `model_id: str`, `provider: str`, `raw_text: str` (for audit trail).
- [ ] Retry + rate-limit handling lives in the base class; concrete clients only implement `_call_provider`.
- [ ] Smoke test (`tests/test_llm_clients_smoke.py`): same SRS chunk + violation-detection prompt run through all 4 providers; all return identical `Violation` schema. Tested in CI.
- [ ] Cost-trace contract: each `StructuredResponse` arrives with token counts that WP-01c can multiply by `pricing.yaml` to compute USD without provider-specific code.

## Implementation steps

1. Read `llm_client.py` + `architect.py` to identify every Gemini call site (already mapped in `00-context-report.md` Q4: lines 226, 326, 462, 604).
2. Extract the shared "prompt → JSON parse → Pydantic validate" loop into the base class.
3. Implement `GeminiClient` first (90% of the existing code); regression-test against existing `validation_metrics_report.json`.
4. Implement `OpenAIClient` with `response_format=PydanticModel`; smoke-test on a 1-violation example.
5. Implement `AnthropicClient` via tool-use; smoke-test.
6. Implement `LocalClient` with Ollama (`OLLAMA_HOST=http://localhost:11434`) and a `--json-mode` prompt postfix. If Ollama JSON-mode is unreliable, fallback to `instructor` library for client-side schema repair.
7. Add `configs/model.yaml` schema:
   ```yaml
   provider: gemini  # gemini | openai | anthropic | local
   model_id: gemini-2.5-pro
   temperature: 0.05
   seed: 42
   max_retries: 2
   ```
8. Replace every direct `genai.Client(...)` call site in `architect.py` with `LLMClient.from_config()`.
9. Verify: pre-existing intermediate runs (154 files, 2026-03-12/13) reproduce *byte-identically* under the refactored Gemini path (regression guarantee).

## Outputs (file paths)

- `extension/backend/core/llm/base.py`
- `extension/backend/core/llm/gemini.py`
- `extension/backend/core/llm/openai.py`
- `extension/backend/core/llm/anthropic.py`
- `extension/backend/core/llm/local.py`
- `extension/backend/core/llm/__init__.py` (factory)
- `configs/model.yaml`
- `extension/backend/tests/test_llm_clients_smoke.py`
- Updated `extension/backend/core/architect.py` (call sites refactored)
- Updated `extension/backend/requirements.txt` (`openai`, `anthropic`, `ollama` or `instructor`)

## Risks & mitigations

- **Risk:** OSS model JSON-mode unreliable (R2 in `01-risks.md`). **Mitigation:** Fall back to cloud-hosted OSS via Together AI / Replicate; clearly note in §4.3 that "open-source" includes "open-weights, third-party-hosted" if local hardware insufficient. Time-box the local-OSS attempt to 3 days.
- **Risk:** Anthropic tool-use schema differs from OpenAI/Gemini. **Mitigation:** All 4 providers must produce the **same Pydantic class** (`Violation`); the base class is the contract, not the wire format.
- **Risk:** Refactor breaks pre-existing intermediate runs and we lose regression confidence. **Mitigation:** Tag commit just before refactor; smoke test compares old vs new `GeminiClient` output byte-by-byte for the SRS in `inputs/SRS.docx`.
