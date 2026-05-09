# WP-01a: Provider Abstraction (`core/llm/` package, big-bang refactor)

**Owner:** Baran
**Depends-on:** [WP-00 scope lock]
**Effort:** M (1.5 weeks for one engineer; 9 atomic commits)
**Status:** TODO
**Addresses:** [D1 6-model lockdown, Hoca-1 RQ2 enabler]
**Refs:** `MASTER_PLAN.md` §7 implementation plan

---

## Goal

Eski `extension/backend/core/llm_client.py` (258 satır, sadece Gemini) **silinecek**; yerine `core/llm/` paketi geliyor. Provider-agnostic, **6 model destekler** (G1, G2 Gemini + 4 OSS via Ollama Cloud), key rotation + retry + json_failed metric integration ile.

**Big-bang refactor scope** (no shim): eski modül tek seferde silinecek — ama 9 atomic commit'e bölünmüş, her biri yeşil-CI + rollback'e açık.

**Önemli context**: Mevcut `llm_client.py` `from google import genai` kullanıyor (yeni SDK), Pydantic Gemini flow temel olarak çalışıyor. Yeni paket bunu **yeniden yazıp** Ollama Cloud'u da OpenAI-compatible API üzerinden ekleyecek.

---

## Architecture

### Module structure
```
core/llm/
  __init__.py           # public API: get_client, LLMClient ABC, errors
  base.py               # LLMResponse, LLMClient ABC, TokenUsage
  errors.py             # RateLimitError, AuthError, SchemaError, RetryExhausted
  retry.py              # @with_retry_and_rotation (saf decorator)
  registry.py           # model_id → ModelSpec (provider, name, pricing, capabilities)
  gemini.py             # GeminiClient (google-genai SDK)
  ollama.py             # OllamaClient (OpenAI SDK → ollama.com/v1)
  schema_probe.py       # CLI: 6 model × 3 schema smoke
```

### LLMResponse contract (the key abstraction)

```python
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int = 0  # if provider supports caching

@dataclass
class LLMResponse:
    content: str                    # raw text response
    parsed: BaseModel | None        # Pydantic-parsed; None if json_failed
    usage: TokenUsage
    model_id: str                   # e.g., "gemini-3.1-pro-preview"
    provider: str                   # "gemini" | "ollama"
    json_failed: bool               # True if structured_output requested but parsing failed
    json_fail_reason: str | None    # "schema_mismatch" | "invalid_json" | "missing_field" | None
    latency_ms: float
    raw_response: dict              # full provider-specific dict (debug/audit)
```

### LLMClient ABC

```python
from abc import ABC, abstractmethod
from typing import Type
from pydantic import BaseModel

class LLMClient(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], model: str, **kwargs) -> LLMResponse:
        """Plain chat completion (no schema enforcement)."""

    @abstractmethod
    def structured_output(
        self,
        messages: list[dict],
        schema: Type[BaseModel],
        model: str,
        **kwargs,
    ) -> LLMResponse:
        """Schema-enforced output. Sets json_failed=True if model violates schema."""
```

### Retry + Key Rotation

`core/llm/retry.py`:
```python
RETRYABLE_STATUS = {429, 403, 500, 502, 503, 504}

def with_retry_and_rotation(
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    keys: list[str],
):
    """
    Algorithm:
      1. attempt 0: keys[0]
      2. retryable error → keys[1] (no delay), then keys[2], etc.
      3. all keys exhausted with retryable error → exponential backoff (1s, 2s, 4s)
      4. 5xx with single key → backoff
      5. 400/401 → RAISE (no retry)
      6. max_retries exhausted → RetryExhausted exception
    """
```

**.env format**:
```
GEMINI_API_KEY=AIza...
OLLAMA_API_KEYS=key1,key2,key3,key4,key5,key6
```

### OllamaClient (the interesting one — OpenAI-compatible API)

```python
from openai import OpenAI

class OllamaClient(LLMClient):
    def __init__(self):
        self._keys = os.environ["OLLAMA_API_KEYS"].split(",")

    def _make_client(self, key: str) -> OpenAI:
        return OpenAI(base_url="https://ollama.com/v1", api_key=key)

    def structured_output(
        self,
        messages: list[dict],
        schema: Type[BaseModel],
        model: str,
        **kwargs,
    ) -> LLMResponse:
        @with_retry_and_rotation(keys=self._keys)
        def _call(api_key: str) -> dict:
            client = self._make_client(api_key)
            return client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_schema", "json_schema": schema.model_json_schema()},
                temperature=kwargs.get("temperature", 0.05),
                seed=kwargs.get("seed", 42),
            )
        # parse response into LLMResponse, set json_failed if Pydantic fails
        ...
```

### GeminiClient (uses google-genai SDK)

Uses `from google import genai` (already in requirements.lock as `google-genai==1.75.0`). Single key for now (1 Gemini key).

---

## Acceptance Criteria

- [ ] `core/llm/__init__.py` exports: `LLMClient`, `LLMResponse`, `TokenUsage`, `get_client(provider)`, errors module
- [ ] `core/llm/base.py`: ABC contract with `chat()` and `structured_output()` methods
- [ ] `core/llm/errors.py`: `LLMError`, `RateLimitError`, `AuthError`, `SchemaError`, `RetryExhausted`
- [ ] `core/llm/retry.py`: `with_retry_and_rotation` decorator (testable in isolation with mocked time.sleep)
- [ ] `core/llm/registry.py`: `ModelSpec` dataclass + 6-model registry with pricing
- [ ] `core/llm/gemini.py`: `GeminiClient` for G1, G2 (G2 falls back to gemini-2.5-flash if 3.1-flash-lite unavailable)
- [ ] `core/llm/ollama.py`: `OllamaClient` for O1, O2, O3, O4 (key rotation + retry)
- [ ] `core/llm/schema_probe.py`: standalone CLI — `python -m core.llm.schema_probe` runs 6 model × 3 schema smoke, writes `runs/probe.json`
- [ ] **Old `core/llm_client.py` deleted** (commit 8 of 9)
- [ ] All existing tests still pass (`tests/test_unit.py`, `test_architect_helpers.py`, etc.) after migration commits 6-7
- [ ] New test coverage: `tests/test_llm/{test_base,test_errors,test_registry,test_retry,test_gemini,test_ollama,test_schema_probe}.py`
- [ ] `pyright` 0 error in `core/llm/` package
- [ ] CI green throughout — no commit leaves repo in red state

---

## 9-Commit TDD Sequence

| # | Commit message (suggested) | Files | Test count | Risk |
|---|----------------------------|-------|------------|------|
| 1 | `feat(llm): add LLMClient ABC + LLMResponse + errors` | `core/llm/{__init__,base,errors}.py` + tests | ~8 | Düşük — interface only |
| 2 | `feat(llm): add ModelSpec registry for 6 models + pricing` | `core/llm/registry.py` + tests | ~10 | Düşük — pure data |
| 3 | `feat(llm): add retry+key-rotation decorator` | `core/llm/retry.py` + tests (mock time.sleep) | ~12 | Düşük |
| 4 | `feat(llm): add OllamaClient (OpenAI-compatible API)` | `core/llm/ollama.py` + tests (mock OpenAI client) | ~10 | Orta |
| 5 | `feat(llm): add GeminiClient (google-genai SDK)` | `core/llm/gemini.py` + tests (mock genai) | ~10 | Orta |
| 6 | `refactor(architect): migrate to core.llm.GeminiClient` | `core/architect.py` + test updates | mevcut testler güncellenir | **Yüksek** — pipeline kodu değişiyor |
| 7 | `refactor(main): migrate FastAPI handlers to core.llm` | `main.py` + integration test marker | regression | Yüksek |
| 8 | `chore(llm): remove deprecated core/llm_client.py` | -`core/llm_client.py` | tüm test geçmeli | Düşük (artık kullanılmıyor) |
| 9 | `feat(llm): add schema_probe CLI for 6-model smoke` | `core/llm/schema_probe.py` + tests | ~5 | Düşük |

Her commit yeşil-CI + her commit'ten önceki hâle revert mümkün.

---

## Implementation Notes

### Commit 1 — base + errors
- `LLMResponse` dataclass with all fields above
- `LLMClient` ABC with `@abstractmethod` for `chat` and `structured_output`
- Custom exception hierarchy with `RetryExhausted` storing `last_exception` and `attempt_count`
- Tests: type checks, abstract enforcement, exception construction

### Commit 2 — registry
```python
@dataclass(frozen=True)
class ModelSpec:
    model_id: str          # e.g., "gemini-3.1-pro-preview"
    provider: str          # "gemini" | "ollama"
    full_name: str         # human-readable
    capabilities: set[str] # {"json_strict", "tool_use", "vision"}
    pricing_per_million_tokens: dict  # {"input": float, "output": float, "cached": float}
    context_window: int

REGISTRY: dict[str, ModelSpec] = {
    "gemini-3.1-pro-preview": ModelSpec(...),
    "gemini-3.1-flash-lite": ModelSpec(...),
    "gpt-oss:120b-cloud": ModelSpec(...),
    "qwen3-coder-next:cloud": ModelSpec(...),
    "minimax-m2:cloud": ModelSpec(...),
    "gemma4:31b-cloud": ModelSpec(...),
}
```
Pricing: G1 ~$1.25/$5, G2 ~$0.25/$1.50, OSS = 0 within Ollama Cloud free tier.

### Commit 3 — retry
- Decorator factory `with_retry_and_rotation(*, max_retries, base_delay, keys)`
- Inner function tracks `attempt_count` and rotates `key_index`
- Tests: mock `time.sleep`, simulate 429 rotation, simulate 5xx backoff, simulate 400 raise, simulate exhaustion

### Commit 4 — OllamaClient
- Uses `openai.OpenAI(base_url="https://ollama.com/v1", api_key=...)`
- `structured_output` uses `response_format={"type": "json_schema", ...}`
- Wraps the `chat.completions.create` call in `with_retry_and_rotation`
- On JSON parse failure: sets `json_failed=True`, `json_fail_reason="invalid_json"` or `"schema_mismatch"`
- Tests: mock `openai.OpenAI`, simulate successful response, simulate 429+rotation success, simulate schema mismatch

### Commit 5 — GeminiClient
- Uses `google.genai.Client(api_key=...)`
- `structured_output` uses Gemini's `response_schema` parameter
- Single key (no rotation needed; can extend later)
- Same json_failed metric tracking
- Tests: mock `genai.Client`, similar coverage

### Commit 6 — architect.py migration (HIGH RISK)
- All `from google import genai` calls in `architect.py` → `from core.llm import get_client`
- `client = get_client("gemini").structured_output(...)` replaces direct API calls
- Existing tests (`test_architect_helpers.py`, `test_unit.py` Architect tests) must pass unchanged
- Regression check: pre-existing intermediate JSON outputs (`extension/backend/core/intermediate/`) reproduce on a sample SRS

### Commit 7 — main.py migration
- FastAPI handlers in `main.py:164` (`from google import genai`) → `from core.llm import get_client`
- `test_api.py` (integration, marker'lı) bunu test eder; CI'da skip

### Commit 8 — delete old client
- `git rm extension/backend/core/llm_client.py`
- Verify no remaining imports: `git grep "from core.llm_client" extension/backend/` → boş
- All tests still green

### Commit 9 — schema probe
- CLI: `python -m core.llm.schema_probe --models all --schemas basic,medium,complex`
- 3 test schemas of varying complexity (simple object, nested object, deep recursion)
- Per model × schema: 1 call, check json_failed, record pass/fail
- Output: `runs/probe_results.json` + console summary

---

## Outputs (file paths)

**Created**:
- `extension/backend/core/llm/__init__.py`
- `extension/backend/core/llm/base.py`
- `extension/backend/core/llm/errors.py`
- `extension/backend/core/llm/retry.py`
- `extension/backend/core/llm/registry.py`
- `extension/backend/core/llm/gemini.py`
- `extension/backend/core/llm/ollama.py`
- `extension/backend/core/llm/schema_probe.py`
- `extension/backend/tests/test_llm/test_base.py`
- `extension/backend/tests/test_llm/test_errors.py`
- `extension/backend/tests/test_llm/test_retry.py`
- `extension/backend/tests/test_llm/test_registry.py`
- `extension/backend/tests/test_llm/test_gemini.py`
- `extension/backend/tests/test_llm/test_ollama.py`
- `extension/backend/tests/test_llm/test_schema_probe.py`

**Modified**:
- `extension/backend/core/architect.py` (LLM call sites)
- `extension/backend/main.py` (FastAPI handlers)
- `extension/backend/tests/conftest.py` (placeholder OLLAMA_API_KEYS for tests)

**Deleted**:
- `extension/backend/core/llm_client.py`

---

## Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| OSS model JSON-mode unreliable for Pydantic strict | Orta | `schema_probe` (commit 9) ölçer; başarısız olan modeli "Tier 2" işaretle, prompt-engineering fallback |
| Ollama Cloud session limit (5 saatte reset) | Düşük | Key rotation + retry handle eder; hâlâ takılırsa Pro abone yedek (kullanıcı $20'a açık) |
| Gemini API contract değişir (preview model) | Düşük | google-genai SDK semver kullanır; pin version |
| Architect.py migration regression | **Yüksek** | Commit 6 öncesi pre-existing intermediate output snapshot al; migration sonrası diff ile compare |
| Test mocking kompleksitesi | Orta | OpenAI ve google-genai SDK mocking için her test'te explicit `Mock(spec=...)` kullan |

---

## Sync Points

- **End of WP-01a (~ end of W4)**: `S1` handoff to Ali — `core/llm/__init__.py` API contract documented, Ali codebase generation için kullanır
