# Spec — Model Registry Consolidation

**Date:** 2026-04-27
**Branch:** `feat/EnhancedDocumentParserModule` (existing; no new branch)
**Status:** APPROVED (Sections 1 + 2 + JSON-shape break + branch choice all confirmed)
**Scope:** Internal refactor — single source of truth for LLM model selection + pricing.
**Out of scope:** Full multi-provider abstraction (deferred to WP-01a in `INDEX.md`).

---

## 1. Goal

Replace the current fragmented model configuration (5+ locations: `config.py`, `token_tracker.py`, `main.py`, `architect.py`, tests, `TOKEN_TRACKING.md`) with a single registry module at `extension/backend/configs/models.py`. After this refactor, swapping a model is **one edit in one file**. Pricing, provider tagging, tiered context-based pricing, and import-time validation all live in the registry.

The refactor also upgrades the active models from the current Gemini 2.5 Flash / Flash-Lite pair to:
- **Domain extraction** (Scout / Architect / Specialist / Synthesizer): `gemini-3.1-pro-preview`
- **Validation**: `gemini-3-flash-preview`

The Flash-Lite tier is removed entirely — no references remain anywhere in the codebase or documentation per the user directive "hiçbir yerde lite kullanmayalım".

## 2. Context

Diagnostic findings (full evidence in `00-context-report.md` and grep results captured during brainstorming):

- `config.py:100, 110` — hardcoded model name strings split across `AnalyzerConfig` and `ArchitectConfig`.
- `core/token_tracker.py` — `FLASH_PRICING`, `FLASH_LITE_PRICING`, `STAGE_MODEL_MAP` dicts; `_calculate_call_cost` if/else branched on short codes "flash" / "flash-lite"; `f"gemini-2.5-{model}"` interpolations at lines 316, 331, 399 (this format string silently breaks under any non-2.5 model).
- `main.py:651-728, 873-939` — direct internal-field access (`token_tracker.stats.flash_lite_prompt_tokens`) and hardcoded literal model strings throughout API responses.
- `core/architect.py:40-42` — confusing `LLMConfig = ArchitectConfig()` alias.
- `core/architect.py:825` — references `self.LLMConfig.MAX_OUTPUT_TOKENS`, a field that **does not exist anywhere** in the codebase (latent `AttributeError`). **Out of scope for this refactor**; flagged with a `# TODO` comment for a separate fix.
- `tests/test_unit.py:32-36, 451-454`; `tests/test_api.py:520-528` — assertions on hardcoded pricing floats and model string literals.
- VS Code extension (`extension/src/extension.ts`): grep confirmed it does **not** consume the changing JSON keys directly — communicates through documented endpoints whose externally-meaningful semantics are preserved.

## 3. Constraints

- **AGENTS.md compliance.** Each file ≤ 300 lines. Smallest correct change. No speculative abstractions. Fail-loud over silent degradation. No backwards-compat hacks unless explicitly required.
- **Generic-A scope.** Registry interface is provider-agnostic (`provider` field, full `model_id` strings, no version-prefix interpolation, tiered pricing). LLM call implementation remains Gemini-only — full multi-provider abstraction is WP-01a.
- **No flash-lite anywhere.** Removed from `MODELS`, all comments, all tests, all docs.
- **JSON shape break is acceptable.** Confirmed by user; no backwards-compat shim for the old `flash_model` / `flash_lite_model` / `gemini-2.5-*` keys.
- **No new dependencies.** Pure-Python config module; no PyYAML / TOML.

## 4. Registry Module — `extension/backend/configs/models.py`

### 4.1 Types

```python
@dataclass(frozen=True)
class PricingTier:
    """One pricing tier. Tiers are matched in order; first match wins."""
    max_prompt_tokens: Optional[int]   # None = unlimited (last tier)
    input_per_1m_usd: float
    output_per_1m_usd: float


@dataclass(frozen=True)
class Pricing:
    """Cost calculator. Supports flat or context-tiered pricing."""
    tiers: Tuple[PricingTier, ...]    # at least one entry; ordered ascending

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost for a single API call. Raises if no tier matches."""
        for tier in self.tiers:
            if tier.max_prompt_tokens is None or prompt_tokens <= tier.max_prompt_tokens:
                return (prompt_tokens * tier.input_per_1m_usd
                        + completion_tokens * tier.output_per_1m_usd) / 1_000_000
        raise ValueError(f"No pricing tier matched prompt_tokens={prompt_tokens}")


def flat(input_per_1m_usd: float, output_per_1m_usd: float) -> Pricing:
    """Pricing helper for models with a single flat rate."""
    return Pricing(tiers=(PricingTier(None, input_per_1m_usd, output_per_1m_usd),))


@dataclass(frozen=True)
class ModelInfo:
    model_id: str            # exact string the provider API expects
    provider: str            # "gemini" | "openai" | "anthropic" | "local"
    pricing: Pricing
    context_window: Optional[int]


@dataclass(frozen=True)
class StageConfig:
    model_id: str            # must reference a key in MODELS
    temperature: float
    seed: Optional[int]
```

### 4.2 Defaults (initial registry content)

```python
MODELS: Dict[str, ModelInfo] = {
    "gemini-3.1-pro-preview": ModelInfo(
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        pricing=Pricing(tiers=(
            PricingTier(max_prompt_tokens=200_000, input_per_1m_usd=2.0,  output_per_1m_usd=12.0),
            PricingTier(max_prompt_tokens=None,    input_per_1m_usd=4.0,  output_per_1m_usd=18.0),
        )),
        context_window=1_000_000,
    ),
    "gemini-3-flash-preview": ModelInfo(
        model_id="gemini-3-flash-preview",
        provider="gemini",
        pricing=flat(input_per_1m_usd=0.50, output_per_1m_usd=3.0),
        context_window=1_000_000,
    ),
}

STAGE_GROUPS: Dict[str, StageConfig] = {
    "domain_extraction": StageConfig(model_id="gemini-3.1-pro-preview", temperature=0.05, seed=42),
    "validation":        StageConfig(model_id="gemini-3-flash-preview",  temperature=0.05, seed=42),
}

STAGE_TO_GROUP: Dict[str, str] = {
    "Scout":       "domain_extraction",
    "Architect":   "domain_extraction",
    "Specialist":  "domain_extraction",
    "Synthesizer": "domain_extraction",
    "Validator":   "validation",
}
```

### 4.3 Helpers

```python
def stage_config(stage: str) -> StageConfig:
    """Return the StageConfig for a pipeline stage. KeyError on unknown stage."""
    return STAGE_GROUPS[STAGE_TO_GROUP[stage]]

def model_for_stage(stage: str) -> ModelInfo:
    """Return ModelInfo (model_id + pricing + provider) for a pipeline stage."""
    return MODELS[stage_config(stage).model_id]

def model_info(model_id: str) -> ModelInfo:
    """Return ModelInfo for a given model_id. KeyError on unknown model."""
    return MODELS[model_id]
```

### 4.4 Import-time validation

```python
# Validate at import time: every STAGE_GROUPS model_id must exist in MODELS.
for _group_name, _sc in STAGE_GROUPS.items():
    if _sc.model_id not in MODELS:
        raise RuntimeError(
            f"STAGE_GROUPS['{_group_name}'].model_id={_sc.model_id!r} not in MODELS"
        )
del _group_name, _sc
```

### 4.5 File header

Module docstring documents:
- Purpose (single source of truth for model selection + pricing).
- "How to upgrade a model" (1 line edit in STAGE_GROUPS).
- "How to add a model" (add MODELS entry, then reference it).
- Pricing snapshot date + source URL.
- "All Gemini 3 entries are PREVIEW" warning (subject to provider-side change).

Estimated file length: ~120 lines.

## 5. Refactor — Per-File Changes

### 5.1 `extension/backend/configs/__init__.py` (NEW)

Empty marker file. One-line docstring.

### 5.2 `extension/backend/config.py` (EDIT, ~12-line delta)

`AnalyzerConfig` and `ArchitectConfig` are kept (consumers reference them as `Config.MODEL_NAME`). Their model-related class attributes become **derived from the registry**:

```python
from configs.models import stage_config

class AnalyzerConfig:
    """Configuration for the Code Analyzer LLM client (validation stage)."""
    _STAGE_CONFIG = stage_config("Validator")
    MODEL_NAME: str = _STAGE_CONFIG.model_id
    TEMPERATURE: float = _STAGE_CONFIG.temperature
    SEED: int = _STAGE_CONFIG.seed
    RESPONSE_MIME_TYPE: str = "application/json"
    VALIDATION_RETRIES: int = 2
    RETRY_BACKOFF_SECONDS: float = 1.0


class ArchitectConfig:
    """Configuration for the Domain Architect LLM client (domain extraction stages)."""
    _STAGE_CONFIG = stage_config("Architect")
    MODEL_NAME: str = _STAGE_CONFIG.model_id
    TEMPERATURE: float = _STAGE_CONFIG.temperature
    SEED: int = _STAGE_CONFIG.seed
    RESPONSE_MIME_TYPE: str = "application/json"
```

`SEED` is now also exposed on `AnalyzerConfig` (was missing — minor consistency fix). All other fields preserved.

### 5.3 `extension/backend/core/token_tracker.py` (EDIT, ~250 lines after rewrite, down from 453)

**Removed entirely:**
- `FLASH_PRICING`, `FLASH_LITE_PRICING` module-level dicts.
- `STAGE_MODEL_MAP` module-level dict.
- `TokenUsageStats.flash_prompt_tokens`, `.flash_completion_tokens`, `.flash_lite_prompt_tokens`, `.flash_lite_completion_tokens` fields.
- `_calculate_call_cost`'s `if model == "flash" / else` branching.
- All `f"gemini-2.5-{model}"` interpolations (3 sites).
- All "flash"/"flash-lite" short-code references.

**Added:**

```python
from configs.models import ModelInfo, model_for_stage, model_info

@dataclass
class ModelTokenAccumulator:
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0


@dataclass
class StageTokenAccumulator:
    stage: str
    model_id: str        # snapshot at first call for this stage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0


@dataclass
class TokenUsageStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_api_calls: int = 0
    by_model: Dict[str, ModelTokenAccumulator] = field(default_factory=dict)
    by_stage: Dict[str, StageTokenAccumulator] = field(default_factory=dict)
    call_history: List[APICallRecord] = field(default_factory=list)
```

**Public API preserved (signatures unchanged):**
- `track_api_call(response, stage: str, operation: str) -> None`
- `get_report(detailed: bool = False) -> Dict`
- `print_summary() -> None`
- `get_combined_metrics() -> Dict`
- `export_to_json(filepath: str, detailed: bool = True) -> None`
- `calculate_cost() -> Dict` — return shape changes per §6 below.

**New public method:**
- `tokens_for_stage(stage: str) -> StageTokenAccumulator` — eliminates main.py's reach into `tracker.stats.flash_lite_*` private fields.

**Cost calculation:**
```python
def _record_call(self, stage: str, prompt_tokens: int, completion_tokens: int, operation: str) -> None:
    info = model_for_stage(stage)
    cost = info.pricing.cost_for(prompt_tokens, completion_tokens)
    # ... update by_model[info.model_id], by_stage[stage], totals, call_history.
```

The function `model_info()` is used when iterating over `by_model` for report generation (caller knows model_id, needs provider/pricing for display).

### 5.4 `extension/backend/main.py` (EDIT, ~30-line delta)

**Lines 651-652, 727-728** — replace direct field access:
```python
# OLD
pre_input  = token_tracker.stats.flash_lite_prompt_tokens
pre_output = token_tracker.stats.flash_lite_completion_tokens

# NEW
validator = token_tracker.tokens_for_stage("Validator")
pre_input  = validator.prompt_tokens
pre_output = validator.completion_tokens
```

**Lines 873-939** — replace hardcoded model literal strings:
```python
# OLD
"gemini-2.5-flash-lite": { ... }
"generation_model": "gemini-2.5-flash"

# NEW
from configs.models import model_for_stage
validator_model = model_for_stage("Validator").model_id   # "gemini-3-flash-preview"
architect_model = model_for_stage("Architect").model_id   # "gemini-3.1-pro-preview"
# ... use these variables in dict construction.
```

Cost field accessors update to read from new `cost_estimation.by_model[<model_id>]` shape.

### 5.5 `extension/backend/core/architect.py` (EDIT, ~3-line delta)

**Line 40-42:**
```python
# OLD
class DomainArchitect:
    LLMConfig = ArchitectConfig()
    def __init__(self, model: str = LLMConfig.MODEL_NAME, ...):

# NEW
from configs.models import stage_config

class DomainArchitect:
    def __init__(self, model: Optional[str] = None, ...):
        ...
        self.model_name = model or stage_config("Architect").model_id
```

**Line 825** — annotate latent bug, do not fix:
```python
# TODO(architect-bug-001): self.LLMConfig.MAX_OUTPUT_TOKENS does not exist.
# This branch will raise AttributeError when triggered. Tracked for separate fix.
print(f"      💡 Hit token limit ({self.LLMConfig.MAX_OUTPUT_TOKENS})")
```

### 5.6 `extension/backend/tests/test_unit.py` (EDIT, ~10-line delta)

**Lines 32-36** — pricing assertions read from registry:
```python
# OLD
# Flash pricing assertions on FLASH_PRICING dict literal values.

# NEW
from configs.models import MODELS
arch_pricing = MODELS["gemini-3.1-pro-preview"].pricing
assert arch_pricing.cost_for(prompt_tokens=1_000_000, completion_tokens=0) == 2.0
```

**Lines 451, 454** — config assertions read from registry:
```python
# NEW
from configs.models import stage_config
assert AnalyzerConfig.MODEL_NAME == stage_config("Validator").model_id == "gemini-3-flash-preview"
assert ArchitectConfig.MODEL_NAME == stage_config("Architect").model_id == "gemini-3.1-pro-preview"
```

**New snapshot test** (drift guard):
```python
def test_default_models_unchanged():
    """Drift guard. If a default model changes, this test fails — update consciously."""
    from configs.models import STAGE_GROUPS
    assert STAGE_GROUPS["domain_extraction"].model_id == "gemini-3.1-pro-preview"
    assert STAGE_GROUPS["validation"].model_id        == "gemini-3-flash-preview"
```

### 5.7 `extension/backend/tests/test_api.py` (EDIT, ~5-line delta)

API JSON shape assertions updated for the new key set (see §6 below).

### 5.8 `extension/backend/TOKEN_TRACKING.md` (EDIT)

Existing model-name table replaced with:
> Model selection and pricing live in `extension/backend/configs/models.py`.
> To upgrade a model: edit `STAGE_GROUPS` there. No other file should need changes.

## 6. JSON Shape Break (intentional, no compat shim)

**Old shape** (in `validation_metrics_report.json` and the relevant API response paths):
```json
{
  "model_usage": {
    "gemini-2.5-flash":      { "prompt_tokens": ..., "completion_tokens": ..., "stages": [...] },
    "gemini-2.5-flash-lite": { "prompt_tokens": ..., "completion_tokens": ..., "stages": [...] }
  },
  "cost_estimation": {
    "flash_model":      { "input_cost": ..., "output_cost": ..., "total_cost": ... },
    "flash_lite_model": { "input_cost": ..., "output_cost": ..., "total_cost": ... },
    "total_input_cost": ..., "total_output_cost": ..., "total_cost": ..., "currency": "USD"
  }
}
```

**New shape:**
```json
{
  "model_usage": {
    "gemini-3.1-pro-preview": { "prompt_tokens": ..., "completion_tokens": ..., "stages": [...] },
    "gemini-3-flash-preview": { "prompt_tokens": ..., "completion_tokens": ..., "stages": [...] }
  },
  "cost_estimation": {
    "by_model": {
      "gemini-3.1-pro-preview": { "input_cost": ..., "output_cost": ..., "total_cost": ... },
      "gemini-3-flash-preview": { "input_cost": ..., "output_cost": ..., "total_cost": ... }
    },
    "total_input_cost": ..., "total_output_cost": ..., "total_cost": ..., "currency": "USD"
  }
}
```

Stage-keyed names in `model_usage`'s `stages` arrays remain Pythonic ("Scout", "Architect", "Specialist", "Synthesizer", "Validator"). The model_id values reflect whatever is currently in `STAGE_GROUPS` — they are not Gemini-3-hardcoded in code.

External consumers (Jupyter notebooks, ad-hoc scripts) that parse the OLD shape will break. The VS Code extension (`extension/src/extension.ts`) does not parse these keys directly and is unaffected.

## 7. Implementation Order

Each step is its own commit. Branch: `feat/EnhancedDocumentParserModule`. Commit prefix: `refactor(model-registry): `.

| Step | Action | Verification |
|------|--------|--------------|
| A | Create `configs/__init__.py` and `configs/models.py`. Run `python -c "from configs.models import MODELS, STAGE_GROUPS, stage_config; ..."` to confirm import-time validation passes. | Module loads; helpers callable; validation block raises if breakage introduced. |
| B | Pre-refactor baseline: `pytest extension/backend/tests/` — capture which tests pass today (some already known-broken: `test_unit.py:32-36, 451, 454` reference fields we'll change). | Baseline test pass/fail counts recorded. |
| C | Rewrite `core/token_tracker.py` per §5.3. Public-API-level smoke: instantiate tracker, fake a Gemini response, verify `tokens_for_stage("Validator")` returns plausible values. | Token tracker module imports cleanly; smoke instantiation works. |
| D | Edit `config.py` per §5.2. | `python -c "from config import AnalyzerConfig, ArchitectConfig; print(AnalyzerConfig.MODEL_NAME, ArchitectConfig.MODEL_NAME)"` prints the two preview model ids. |
| E | Edit `main.py` per §5.4. | FastAPI server loads (`python main.py`-equivalent dry-run). |
| F | Edit `core/architect.py` per §5.5. | `from core.architect import DomainArchitect` succeeds; `DomainArchitect()` (with mocked GEMINI_API_KEY) instantiates. |
| G | Update `tests/test_unit.py` and `tests/test_api.py` per §5.6, §5.7. Run full test suite. | All tests green except the known latent `MAX_OUTPUT_TOKENS` bug (which we're not triggering). |
| H | Update `TOKEN_TRACKING.md` per §5.8. Final commit. | Doc grep returns no `flash-lite` references. |

## 8. Acceptance Criteria

- [ ] `extension/backend/configs/models.py` exists; ≤ 200 lines; no `# TODO` / `# FIXME` items.
- [ ] `grep -r "flash-lite\|flash_lite\|FLASH_LITE\|FLASH_PRICING\|gemini-2\.5" extension/backend/ --include='*.py'` returns **only** legacy intermediate JSON files (not source code).
- [ ] `grep -r "f\"gemini-" extension/backend/ --include='*.py'` returns no hits (no version-prefix interpolation anywhere).
- [ ] `pytest extension/backend/tests/` is green.
- [ ] FastAPI server starts and serves `/health` without import errors.
- [ ] DomainArchitect smoke run on D1 SRS produces a `validation_metrics_report.json` with the new shape (§6).
- [ ] `configs/models.py` import-time validation works: introduce a typo in `STAGE_GROUPS` to confirm `RuntimeError` fires; revert.
- [ ] Snapshot test `test_default_models_unchanged` passes.
- [ ] No file in `extension/backend/` exceeds 300 lines (AGENTS.md).

## 9. Out of Scope

- Multi-provider (`OpenAIClient`, `AnthropicClient`, `LocalClient`) — full WP-01a; deferred.
- Fixing `architect.py:825` `MAX_OUTPUT_TOKENS` `AttributeError` — separate bug, separate fix.
- Adding more models to the registry beyond the two preview ids — user can add when needed via 1-line edits.
- ENV-var override of `STAGE_GROUPS` model selection — useful for cost-comparison runs but YAGNI for now; can be added in 5 lines later if needed.
- Refactoring `RAG`, `Document Parser`, `AST Visitor` modules — these have nothing to do with model config.

## 10. Risks & Mitigations

| Risk | Likelihood × Impact | Mitigation |
|------|---------------------|------------|
| `gemini-3.1-pro-preview` or `gemini-3-flash-preview` rejected by provider API (preview models can be revoked) | LOW × HIGH | Step C smoke run hits the API; if either rejects, pricing stays in registry but `STAGE_GROUPS` is changed back to `gemini-2.5-pro` / `gemini-2.5-flash` (which we add to MODELS in the same edit). Total revert-time: 5 minutes. |
| `main.py:873-939` JSON consumers (Jupyter notebooks, dashboards) we are unaware of break silently | LOW × LOW | User explicitly accepted JSON-shape break. We document the change in `TOKEN_TRACKING.md`. |
| Refactor introduces regression in token-cost arithmetic | MEDIUM × MEDIUM | Step B baseline + Step C smoke run; existing `validation_metrics_report.json` provides a known-good cost reference for a Gemini 2.5 Flash run that we can replay (with old pricing temporarily added) to confirm the new arithmetic produces the same number. |
| `LLMConfig` class attribute removal in `architect.py` reaches a caller we haven't grepped | LOW × MEDIUM | One more `grep -rn "LLMConfig" extension/` before Step F (in addition to the brainstorm-time grep). Found references: line 40, 42, 825 only. |
| Pricing tier breakpoint logic miscounts at boundary (e.g., 200_000 + 1 prompt tokens) | LOW × LOW | Unit test in Step C verifies cost crossing the 200k boundary returns the higher tier; assert with explicit example values. |

---

## Tracking

- **Spec:** this file (`todos/03-model-registry-spec.md`).
- **Companion docs:** `todos/00-context-report.md` (existing-code inventory), `todos/INDEX.md` (broader EMSE plan; this refactor is a precursor sub-step within WP-01a's eventual scope).
- **Implementation plan:** to be produced by `superpowers:writing-plans` after this spec is approved.
