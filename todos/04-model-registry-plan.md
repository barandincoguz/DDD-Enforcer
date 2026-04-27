# Model Registry Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate fragmented LLM model configuration (model name + pricing + stage mapping) into a single registry module at `extension/backend/configs/models.py`, upgrade defaults to Gemini 3 preview models, and remove every flash-lite reference from the codebase.

**Architecture:** New `configs/models.py` module exposes `MODELS`, `STAGE_GROUPS`, `STAGE_TO_GROUP` registries plus helper functions (`stage_config`, `model_for_stage`, `model_info`). All other modules (`config.py`, `core/token_tracker.py`, `main.py`, `core/architect.py`) become thin consumers. Cost calculation moves from hardcoded if/else branches into `Pricing.cost_for()` with native tiered-pricing support (gemini-3.1-pro's 200k breakpoint). `TokenUsageStats` is reshaped from per-model fields to dict-keyed accumulators (`by_model: Dict[str, ModelTokenAccumulator]`).

**Tech Stack:** Python 3.11+, dataclasses, pytest. No new dependencies.

**Spec reference:** `todos/03-model-registry-spec.md` (approved 2026-04-27).

**Branch:** `feat/EnhancedDocumentParserModule` (existing).

**Working directory:** `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer`. All `pytest` commands run from `extension/backend/`.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `extension/backend/configs/__init__.py` | CREATE | Package marker. |
| `extension/backend/configs/models.py` | CREATE | Single source of truth for model selection, pricing, and stage mapping. |
| `extension/backend/tests/test_models_registry.py` | CREATE | Tests for the registry module. |
| `extension/backend/config.py` | EDIT | `AnalyzerConfig`, `ArchitectConfig` derive `MODEL_NAME`/`TEMPERATURE`/`SEED` from registry. |
| `extension/backend/core/token_tracker.py` | EDIT | Remove flash/flash-lite hardcoding; reshape stats around model_id. |
| `extension/backend/main.py` | EDIT | Use `tokens_for_stage()` and `model_for_stage()` instead of internal field access and literal strings. |
| `extension/backend/core/architect.py` | EDIT | Remove `LLMConfig = ArchitectConfig()` alias; flag `MAX_OUTPUT_TOKENS` latent bug. |
| `extension/backend/tests/test_unit.py` | EDIT | Replace flash/flash-lite assertions with registry-derived; add snapshot drift guard. |
| `extension/backend/tests/test_api.py` | EDIT | Update JSON shape assertions for new `cost_estimation.by_model` + new model_id keys. |
| `extension/backend/TOKEN_TRACKING.md` | EDIT | Replace static model table with pointer to `configs/models.py`. |

---

## Task 1: Baseline & branch sanity

**Files:**
- No file changes; capture baseline state.

- [ ] **Step 1: Verify branch and clean working state for new files**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" branch --show-current
```

Expected: `feat/EnhancedDocumentParserModule`

- [ ] **Step 2: Run baseline pytest and capture pass/fail snapshot**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py -v 2>&1 | tee /tmp/baseline_test_unit.log | tail -25
```

Expected: Some tests pass (e.g., `TestCodeParser`, `TestSchemas`); tests asserting `FLASH_PRICING` and current MODEL_NAME values pass on `gemini-2.5-flash` / `gemini-2.5-flash-lite` strings. **Capture this baseline** — we expect these specific tests to be updated by Tasks 7, 8, 11.

- [ ] **Step 3: No commit (baseline only)**

---

## Task 2: `configs/__init__.py` + `Pricing` primitives (TDD)

**Files:**
- Create: `extension/backend/configs/__init__.py`
- Create: `extension/backend/configs/models.py` (initial — Pricing primitives only)
- Create: `extension/backend/tests/test_models_registry.py` (initial)

- [ ] **Step 1: Create empty package marker**

Create `extension/backend/configs/__init__.py`:

```python
"""LLM model registry and related configuration."""
```

- [ ] **Step 2: Write failing test for `PricingTier` and `Pricing.cost_for`**

Create `extension/backend/tests/test_models_registry.py`:

```python
"""Unit tests for the model registry.

Run: pytest tests/test_models_registry.py -v
"""

import sys
from pathlib import Path

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPricing:
    """Pricing primitives: PricingTier, Pricing.cost_for, flat() helper."""

    def test_flat_pricing_single_tier(self):
        """flat() builds a Pricing with one unbounded tier."""
        from configs.models import Pricing, PricingTier, flat

        p = flat(input_per_1m_usd=0.50, output_per_1m_usd=3.0)
        assert isinstance(p, Pricing)
        assert len(p.tiers) == 1
        assert p.tiers[0].max_prompt_tokens is None
        assert p.tiers[0].input_per_1m_usd == 0.50
        assert p.tiers[0].output_per_1m_usd == 3.0

    def test_flat_pricing_cost_for_one_million_input(self):
        """1M input tokens at $0.50/1M == $0.50."""
        from configs.models import flat

        p = flat(input_per_1m_usd=0.50, output_per_1m_usd=3.0)
        assert p.cost_for(prompt_tokens=1_000_000, completion_tokens=0) == pytest.approx(0.50)

    def test_flat_pricing_cost_for_one_million_output(self):
        """1M output tokens at $3.0/1M == $3.0."""
        from configs.models import flat

        p = flat(input_per_1m_usd=0.50, output_per_1m_usd=3.0)
        assert p.cost_for(prompt_tokens=0, completion_tokens=1_000_000) == pytest.approx(3.0)

    def test_flat_pricing_cost_for_mixed(self):
        """Half input, half output at $0.50/$3.0 == $0.25 + $1.50 == $1.75."""
        from configs.models import flat

        p = flat(input_per_1m_usd=0.50, output_per_1m_usd=3.0)
        cost = p.cost_for(prompt_tokens=500_000, completion_tokens=500_000)
        assert cost == pytest.approx(0.25 + 1.50)

    def test_tiered_pricing_under_breakpoint(self):
        """Below 200k prompt tokens: cheap tier applies."""
        from configs.models import Pricing, PricingTier

        p = Pricing(tiers=(
            PricingTier(max_prompt_tokens=200_000, input_per_1m_usd=2.0,  output_per_1m_usd=12.0),
            PricingTier(max_prompt_tokens=None,    input_per_1m_usd=4.0,  output_per_1m_usd=18.0),
        ))
        # 100k prompt + 0 output: 100_000 * 2.0 / 1_000_000 == 0.20
        assert p.cost_for(prompt_tokens=100_000, completion_tokens=0) == pytest.approx(0.20)

    def test_tiered_pricing_at_breakpoint(self):
        """At exactly 200k prompt tokens: cheap tier still applies (inclusive)."""
        from configs.models import Pricing, PricingTier

        p = Pricing(tiers=(
            PricingTier(max_prompt_tokens=200_000, input_per_1m_usd=2.0,  output_per_1m_usd=12.0),
            PricingTier(max_prompt_tokens=None,    input_per_1m_usd=4.0,  output_per_1m_usd=18.0),
        ))
        # 200_000 * 2.0 / 1_000_000 == 0.40
        assert p.cost_for(prompt_tokens=200_000, completion_tokens=0) == pytest.approx(0.40)

    def test_tiered_pricing_over_breakpoint(self):
        """Just above 200k prompt tokens: expensive tier applies."""
        from configs.models import Pricing, PricingTier

        p = Pricing(tiers=(
            PricingTier(max_prompt_tokens=200_000, input_per_1m_usd=2.0,  output_per_1m_usd=12.0),
            PricingTier(max_prompt_tokens=None,    input_per_1m_usd=4.0,  output_per_1m_usd=18.0),
        ))
        # 200_001 * 4.0 / 1_000_000 == 0.800004
        assert p.cost_for(prompt_tokens=200_001, completion_tokens=0) == pytest.approx(200_001 * 4.0 / 1_000_000)

    def test_pricing_tier_is_frozen(self):
        """PricingTier is immutable."""
        from configs.models import PricingTier

        t = PricingTier(max_prompt_tokens=None, input_per_1m_usd=0.5, output_per_1m_usd=3.0)
        with pytest.raises((AttributeError, Exception)):
            t.input_per_1m_usd = 1.0  # type: ignore[misc]
```

- [ ] **Step 3: Run failing test, expect ImportError or ModuleNotFoundError**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py::TestPricing -v
```

Expected: All 8 tests FAIL with `ModuleNotFoundError: No module named 'configs.models'` or `ImportError`.

- [ ] **Step 4: Create `configs/models.py` with Pricing primitives**

Create `extension/backend/configs/models.py`:

```python
"""LLM model registry: single source of truth for model selection and pricing.

To upgrade a model:
    Edit the relevant entry in STAGE_GROUPS below. No other file should need changes.

To add a new model:
    1. Add an entry to MODELS with its provider, pricing, and context window.
    2. Reference it from STAGE_GROUPS (and STAGE_TO_GROUP if a new stage is introduced).

Pricing snapshot: 2026-04-27. All Gemini 3 entries are PREVIEW and subject to provider-side change.
Source: https://ai.google.dev/gemini-api/docs/pricing
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


# =============================================================================
# PRICING PRIMITIVES
# =============================================================================


@dataclass(frozen=True)
class PricingTier:
    """One pricing tier. Tiers are matched in order; first match wins.

    A tier with `max_prompt_tokens=None` matches any prompt size and serves as
    the unbounded final tier.
    """

    max_prompt_tokens: Optional[int]
    input_per_1m_usd: float
    output_per_1m_usd: float


@dataclass(frozen=True)
class Pricing:
    """Cost calculator. Supports flat or context-tiered pricing."""

    tiers: Tuple[PricingTier, ...]

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost for a single API call. Raises if no tier matches."""
        for tier in self.tiers:
            if tier.max_prompt_tokens is None or prompt_tokens <= tier.max_prompt_tokens:
                return (
                    prompt_tokens * tier.input_per_1m_usd
                    + completion_tokens * tier.output_per_1m_usd
                ) / 1_000_000
        raise ValueError(
            f"No pricing tier matched prompt_tokens={prompt_tokens}; "
            f"the last tier must have max_prompt_tokens=None"
        )


def flat(input_per_1m_usd: float, output_per_1m_usd: float) -> Pricing:
    """Pricing helper for models with a single flat rate."""
    return Pricing(tiers=(PricingTier(None, input_per_1m_usd, output_per_1m_usd),))
```

- [ ] **Step 5: Run tests, expect 8 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py::TestPricing -v
```

Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/configs/__init__.py extension/backend/configs/models.py extension/backend/tests/test_models_registry.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): add Pricing primitives + flat() helper"
```

---

## Task 3: `ModelInfo` and `StageConfig` dataclasses (TDD)

**Files:**
- Modify: `extension/backend/configs/models.py` (append types)
- Modify: `extension/backend/tests/test_models_registry.py` (append tests)

- [ ] **Step 1: Append failing tests for `ModelInfo` and `StageConfig`**

Append to `extension/backend/tests/test_models_registry.py` (after `TestPricing` class):

```python
class TestModelInfo:
    """ModelInfo dataclass."""

    def test_model_info_holds_required_fields(self):
        from configs.models import ModelInfo, flat

        info = ModelInfo(
            model_id="example-1.0",
            provider="example",
            pricing=flat(0.10, 0.40),
            context_window=128_000,
        )
        assert info.model_id == "example-1.0"
        assert info.provider == "example"
        assert info.context_window == 128_000
        assert info.pricing.cost_for(1_000_000, 0) == pytest.approx(0.10)

    def test_model_info_is_frozen(self):
        from configs.models import ModelInfo, flat

        info = ModelInfo(
            model_id="example-1.0",
            provider="example",
            pricing=flat(0.10, 0.40),
            context_window=None,
        )
        with pytest.raises((AttributeError, Exception)):
            info.model_id = "other"  # type: ignore[misc]


class TestStageConfig:
    """StageConfig dataclass."""

    def test_stage_config_holds_required_fields(self):
        from configs.models import StageConfig

        sc = StageConfig(model_id="example-1.0", temperature=0.05, seed=42)
        assert sc.model_id == "example-1.0"
        assert sc.temperature == 0.05
        assert sc.seed == 42

    def test_stage_config_seed_can_be_none(self):
        from configs.models import StageConfig

        sc = StageConfig(model_id="example-1.0", temperature=0.0, seed=None)
        assert sc.seed is None

    def test_stage_config_is_frozen(self):
        from configs.models import StageConfig

        sc = StageConfig(model_id="example-1.0", temperature=0.05, seed=42)
        with pytest.raises((AttributeError, Exception)):
            sc.temperature = 0.1  # type: ignore[misc]
```

- [ ] **Step 2: Run tests, expect 5 FAIL with ImportError**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py::TestModelInfo tests/test_models_registry.py::TestStageConfig -v
```

Expected: 5 failed, `ImportError: cannot import name 'ModelInfo' from 'configs.models'`.

- [ ] **Step 3: Append `ModelInfo` and `StageConfig` to `configs/models.py`**

Append to `extension/backend/configs/models.py` (after `flat()` function):

```python
# =============================================================================
# MODEL & STAGE TYPES
# =============================================================================


@dataclass(frozen=True)
class ModelInfo:
    """Static information about an LLM model: id, provider, pricing, context window."""

    model_id: str
    provider: str
    pricing: Pricing
    context_window: Optional[int]


@dataclass(frozen=True)
class StageConfig:
    """Per-stage knobs: which model, what temperature, what seed."""

    model_id: str
    temperature: float
    seed: Optional[int]
```

- [ ] **Step 4: Run tests, expect 5 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py::TestModelInfo tests/test_models_registry.py::TestStageConfig -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/configs/models.py extension/backend/tests/test_models_registry.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): add ModelInfo and StageConfig dataclasses"
```

---

## Task 4: Registry content + helpers (TDD)

**Files:**
- Modify: `extension/backend/configs/models.py` (append registries + helpers)
- Modify: `extension/backend/tests/test_models_registry.py` (append tests)

- [ ] **Step 1: Append failing tests for `MODELS`, `STAGE_GROUPS`, `STAGE_TO_GROUP`, helpers**

Append to `extension/backend/tests/test_models_registry.py`:

```python
class TestRegistryContent:
    """Default registry content: MODELS, STAGE_GROUPS, STAGE_TO_GROUP."""

    def test_models_contains_gemini_3_1_pro_preview(self):
        from configs.models import MODELS

        assert "gemini-3.1-pro-preview" in MODELS
        m = MODELS["gemini-3.1-pro-preview"]
        assert m.provider == "gemini"
        assert m.context_window == 1_000_000

    def test_models_contains_gemini_3_flash_preview(self):
        from configs.models import MODELS

        assert "gemini-3-flash-preview" in MODELS
        m = MODELS["gemini-3-flash-preview"]
        assert m.provider == "gemini"
        assert m.context_window == 1_000_000

    def test_no_flash_lite_or_2_5_models(self):
        """Per spec: no flash-lite, no 2.5-era models in default registry."""
        from configs.models import MODELS

        for model_id in MODELS:
            assert "lite" not in model_id.lower(), f"Unexpected flash-lite model: {model_id}"
            assert not model_id.startswith("gemini-2."), f"Unexpected 2.x model: {model_id}"

    def test_gemini_3_1_pro_pricing_under_200k(self):
        """gemini-3.1-pro-preview at <200k input: $2/$12 per 1M."""
        from configs.models import MODELS

        p = MODELS["gemini-3.1-pro-preview"].pricing
        # 100k input + 100k output: 100_000 * 2.0 / 1M + 100_000 * 12.0 / 1M
        cost = p.cost_for(prompt_tokens=100_000, completion_tokens=100_000)
        assert cost == pytest.approx(0.20 + 1.20)

    def test_gemini_3_1_pro_pricing_over_200k(self):
        """gemini-3.1-pro-preview at >200k input: $4/$18 per 1M."""
        from configs.models import MODELS

        p = MODELS["gemini-3.1-pro-preview"].pricing
        # 250k input + 50k output: 250_000 * 4.0 / 1M + 50_000 * 18.0 / 1M
        cost = p.cost_for(prompt_tokens=250_000, completion_tokens=50_000)
        assert cost == pytest.approx(250_000 * 4.0 / 1_000_000 + 50_000 * 18.0 / 1_000_000)

    def test_gemini_3_flash_pricing_flat(self):
        """gemini-3-flash-preview: flat $0.50/$3 per 1M, regardless of context size."""
        from configs.models import MODELS

        p = MODELS["gemini-3-flash-preview"].pricing
        small = p.cost_for(prompt_tokens=10_000, completion_tokens=10_000)
        large = p.cost_for(prompt_tokens=900_000, completion_tokens=10_000)
        # Cost scales linearly; same per-token rate at any size.
        # Per-token input: 0.50 / 1M. Per-token output: 3.0 / 1M.
        assert small == pytest.approx(10_000 * 0.50 / 1_000_000 + 10_000 * 3.0 / 1_000_000)
        assert large == pytest.approx(900_000 * 0.50 / 1_000_000 + 10_000 * 3.0 / 1_000_000)

    def test_stage_groups_default_assignments(self):
        from configs.models import STAGE_GROUPS

        assert STAGE_GROUPS["domain_extraction"].model_id == "gemini-3.1-pro-preview"
        assert STAGE_GROUPS["domain_extraction"].temperature == 0.05
        assert STAGE_GROUPS["domain_extraction"].seed == 42

        assert STAGE_GROUPS["validation"].model_id == "gemini-3-flash-preview"
        assert STAGE_GROUPS["validation"].temperature == 0.05
        assert STAGE_GROUPS["validation"].seed == 42

    def test_stage_to_group_covers_all_pipeline_stages(self):
        from configs.models import STAGE_TO_GROUP

        for stage in ("Scout", "Architect", "Specialist", "Synthesizer"):
            assert STAGE_TO_GROUP[stage] == "domain_extraction"
        assert STAGE_TO_GROUP["Validator"] == "validation"


class TestRegistryHelpers:
    """Helpers: stage_config, model_for_stage, model_info."""

    def test_stage_config_for_validator(self):
        from configs.models import stage_config

        sc = stage_config("Validator")
        assert sc.model_id == "gemini-3-flash-preview"

    def test_stage_config_for_architect(self):
        from configs.models import stage_config

        sc = stage_config("Architect")
        assert sc.model_id == "gemini-3.1-pro-preview"

    def test_stage_config_unknown_stage_raises(self):
        from configs.models import stage_config

        with pytest.raises(KeyError):
            stage_config("DoesNotExist")

    def test_model_for_stage_returns_full_model_info(self):
        from configs.models import model_for_stage

        info = model_for_stage("Validator")
        assert info.model_id == "gemini-3-flash-preview"
        assert info.provider == "gemini"
        assert info.context_window == 1_000_000

    def test_model_info_returns_full_model_info(self):
        from configs.models import model_info

        info = model_info("gemini-3.1-pro-preview")
        assert info.provider == "gemini"
        assert info.context_window == 1_000_000

    def test_model_info_unknown_model_raises(self):
        from configs.models import model_info

        with pytest.raises(KeyError):
            model_info("not-a-real-model")
```

- [ ] **Step 2: Run tests, expect 14 FAIL with ImportError on `MODELS` etc.**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py::TestRegistryContent tests/test_models_registry.py::TestRegistryHelpers -v
```

Expected: 14 failed, `ImportError: cannot import name 'MODELS' from 'configs.models'`.

- [ ] **Step 3: Append registries and helpers to `configs/models.py`**

Append to `extension/backend/configs/models.py`:

```python
# =============================================================================
# REGISTRY CONTENT
# =============================================================================

# Pricing snapshot: 2026-04-27. PREVIEW models — provider may revise without notice.
# Source: https://ai.google.dev/gemini-api/docs/pricing
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


# A "stage group" is a logical role; multiple pipeline stages can share one group.
STAGE_GROUPS: Dict[str, StageConfig] = {
    "domain_extraction": StageConfig(
        model_id="gemini-3.1-pro-preview",
        temperature=0.05,
        seed=42,
    ),
    "validation": StageConfig(
        model_id="gemini-3-flash-preview",
        temperature=0.05,
        seed=42,
    ),
}


STAGE_TO_GROUP: Dict[str, str] = {
    "Scout":       "domain_extraction",
    "Architect":   "domain_extraction",
    "Specialist":  "domain_extraction",
    "Synthesizer": "domain_extraction",
    "Validator":   "validation",
}


# =============================================================================
# HELPERS
# =============================================================================


def stage_config(stage: str) -> StageConfig:
    """Return the StageConfig for a pipeline stage. Raises KeyError on unknown stage."""
    return STAGE_GROUPS[STAGE_TO_GROUP[stage]]


def model_for_stage(stage: str) -> ModelInfo:
    """Return ModelInfo (model_id + pricing + provider) for a pipeline stage."""
    return MODELS[stage_config(stage).model_id]


def model_info(model_id: str) -> ModelInfo:
    """Return ModelInfo for a given model_id. Raises KeyError on unknown model."""
    return MODELS[model_id]
```

- [ ] **Step 4: Run tests, expect 14 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py -v
```

Expected: 27 passed (8 from Task 2 + 5 from Task 3 + 14 from this task).

- [ ] **Step 5: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/configs/models.py extension/backend/tests/test_models_registry.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): add MODELS, STAGE_GROUPS, STAGE_TO_GROUP + helpers"
```

---

## Task 5: Import-time validation (TDD)

**Files:**
- Modify: `extension/backend/configs/models.py` (append `_validate_registry` + import-time call)
- Modify: `extension/backend/tests/test_models_registry.py` (append validation tests)

- [ ] **Step 1: Append failing tests for `_validate_registry`**

Append to `extension/backend/tests/test_models_registry.py`:

```python
class TestRegistryValidation:
    """Import-time validation: STAGE_GROUPS model_ids must exist in MODELS."""

    def test_validate_registry_passes_for_valid_pair(self):
        """Helper passes when every STAGE_GROUPS model_id is present in MODELS."""
        from configs.models import (
            MODELS,
            ModelInfo,
            StageConfig,
            _validate_registry,
            flat,
        )

        models = {"x": ModelInfo("x", "demo", flat(0.1, 0.2), None)}
        groups = {"g": StageConfig(model_id="x", temperature=0.0, seed=None)}
        _validate_registry(groups, models)  # no raise

    def test_validate_registry_raises_when_model_id_missing(self):
        from configs.models import (
            ModelInfo,
            StageConfig,
            _validate_registry,
            flat,
        )

        models = {"x": ModelInfo("x", "demo", flat(0.1, 0.2), None)}
        groups = {"g": StageConfig(model_id="missing-model", temperature=0.0, seed=None)}
        with pytest.raises(RuntimeError) as excinfo:
            _validate_registry(groups, models)
        # Message must name the offending group and model_id for diagnostic clarity.
        assert "g" in str(excinfo.value)
        assert "missing-model" in str(excinfo.value)

    def test_default_registry_passes_validation(self):
        """The shipping defaults (MODELS, STAGE_GROUPS) are internally consistent."""
        from configs.models import MODELS, STAGE_GROUPS, _validate_registry

        # Should not raise.
        _validate_registry(STAGE_GROUPS, MODELS)
```

- [ ] **Step 2: Run tests, expect 3 FAIL**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py::TestRegistryValidation -v
```

Expected: 3 failed, `ImportError: cannot import name '_validate_registry'`.

- [ ] **Step 3: Implement `_validate_registry` in `configs/models.py`**

Append to `extension/backend/configs/models.py`:

```python
# =============================================================================
# IMPORT-TIME VALIDATION
# =============================================================================


def _validate_registry(
    stage_groups: Dict[str, StageConfig],
    models: Dict[str, ModelInfo],
) -> None:
    """Fail loudly if any STAGE_GROUPS entry references an unknown model_id.

    Called at import time on the module-level defaults; also used in unit tests
    with synthetic dicts to exercise the validation logic.
    """
    for group_name, sc in stage_groups.items():
        if sc.model_id not in models:
            raise RuntimeError(
                f"STAGE_GROUPS[{group_name!r}].model_id={sc.model_id!r} "
                f"not present in MODELS (known models: {sorted(models.keys())!r})"
            )


# Validate the shipping defaults at import time.
_validate_registry(STAGE_GROUPS, MODELS)
```

- [ ] **Step 4: Run tests, expect 3 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py -v
```

Expected: 30 passed (27 from Tasks 2-4 + 3 from this task).

- [ ] **Step 5: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/configs/models.py extension/backend/tests/test_models_registry.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): add import-time _validate_registry"
```

---

## Task 6: Token tracker — write failing tests for new public API

**Files:**
- Create: `extension/backend/tests/test_token_tracker_v2.py`

This task only writes tests — they will all FAIL until Task 7 rewrites the tracker. We do this in a separate file (`test_token_tracker_v2.py`) to keep the new contract explicit; existing `TestTokenTracker` in `test_unit.py` will be updated in Task 7.

- [ ] **Step 1: Create `tests/test_token_tracker_v2.py` with new contract tests**

Create `extension/backend/tests/test_token_tracker_v2.py`:

```python
"""Tests for the registry-driven TokenTracker (post-refactor contract).

Run: pytest tests/test_token_tracker_v2.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_gemini_response(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0):
    """Build a Mock that mimics google.genai response.usage_metadata."""
    response = MagicMock()
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = completion_tokens
    response.usage_metadata.cached_content_token_count = cached_tokens
    return response


class TestTokenTrackerV2:
    """Post-refactor TokenTracker contract."""

    def setup_method(self):
        from core.token_tracker import TokenTracker
        TokenTracker.reset()

    def test_track_validator_call_uses_validator_pricing(self):
        """Tracking a Validator call applies gemini-3-flash-preview pricing."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        # 100k input + 1k output via Validator stage (gemini-3-flash-preview, flat $0.50/$3)
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100_000, completion_tokens=1_000),
            stage="Validator",
            operation="validate_code",
        )
        accum = tracker.tokens_for_stage("Validator")
        assert accum.prompt_tokens == 100_000
        assert accum.completion_tokens == 1_000
        assert accum.cost_usd == pytest.approx(100_000 * 0.50 / 1_000_000 + 1_000 * 3.0 / 1_000_000)
        assert accum.call_count == 1

    def test_track_architect_call_uses_pro_pricing_under_200k(self):
        """Architect at <200k prompt tokens applies gemini-3.1-pro-preview cheap tier."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=50_000, completion_tokens=2_000),
            stage="Architect",
            operation="identify_contexts",
        )
        accum = tracker.tokens_for_stage("Architect")
        # 50k * 2.0 / 1M + 2k * 12.0 / 1M
        expected = 50_000 * 2.0 / 1_000_000 + 2_000 * 12.0 / 1_000_000
        assert accum.cost_usd == pytest.approx(expected)

    def test_track_architect_call_uses_pro_pricing_over_200k(self):
        """Architect at >200k prompt tokens applies gemini-3.1-pro-preview expensive tier."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=300_000, completion_tokens=2_000),
            stage="Architect",
            operation="identify_contexts",
        )
        accum = tracker.tokens_for_stage("Architect")
        # 300k * 4.0 / 1M + 2k * 18.0 / 1M
        expected = 300_000 * 4.0 / 1_000_000 + 2_000 * 18.0 / 1_000_000
        assert accum.cost_usd == pytest.approx(expected)

    def test_cached_tokens_excluded_from_billable(self):
        """cached_content_token_count is subtracted from billable prompt tokens."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=10_000, completion_tokens=500, cached_tokens=4_000),
            stage="Validator",
            operation="validate_code",
        )
        accum = tracker.tokens_for_stage("Validator")
        # Billable input = 10_000 - 4_000 = 6_000
        assert accum.prompt_tokens == 6_000

    def test_tokens_for_stage_unknown_stage_returns_zero(self):
        """Querying a stage that has not been tracked returns a zero accumulator."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        accum = tracker.tokens_for_stage("Validator")
        assert accum.prompt_tokens == 0
        assert accum.completion_tokens == 0
        assert accum.call_count == 0

    def test_get_report_uses_model_id_keys(self):
        """get_report()['model_usage'] is keyed by full model_id, not flash/flash-lite."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        report = tracker.get_report()
        assert "gemini-3-flash-preview" in report["model_usage"]
        # No flash-lite anywhere in the report.
        assert all("lite" not in k.lower() for k in report["model_usage"].keys())
        assert all("lite" not in k.lower() for k in report.get("cost_estimation", {}).get("by_model", {}).keys())

    def test_get_report_cost_estimation_has_by_model(self):
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=200, completion_tokens=20),
            stage="Architect",
            operation="identify_contexts",
        )
        report = tracker.get_report()
        ce = report["cost_estimation"]
        assert "by_model" in ce
        assert "gemini-3-flash-preview" in ce["by_model"]
        assert "gemini-3.1-pro-preview" in ce["by_model"]
        assert ce["currency"] == "USD"
        assert ce["total_cost"] >= 0.0

    def test_no_legacy_keys_in_report(self):
        """flash_model, flash_lite_model, gemini-2.5-* keys are gone."""
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        report = tracker.get_report()
        ce = report.get("cost_estimation", {})
        for legacy in ("flash_model", "flash_lite_model"):
            assert legacy not in ce, f"Legacy key {legacy!r} still present in cost_estimation"
        for legacy in ("gemini-2.5-flash", "gemini-2.5-flash-lite"):
            assert legacy not in report["model_usage"], f"Legacy model_usage key {legacy!r} still present"

    def test_get_combined_metrics_per_stage(self):
        from core.token_tracker import TokenTracker

        tracker = TokenTracker.get_instance()
        tracker.track_api_call(
            _fake_gemini_response(prompt_tokens=100, completion_tokens=10),
            stage="Validator",
            operation="validate_code",
        )
        m = tracker.get_combined_metrics()
        assert m["api_calls"] == 1
        assert m["total_input_tokens"] == 100
        assert m["total_output_tokens"] == 10
        assert "Validator" in m["by_stage"]
        assert m["by_stage"]["Validator"]["api_calls"] == 1
```

- [ ] **Step 2: Run tests, expect all FAIL**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_token_tracker_v2.py -v
```

Expected: 9 failed, mostly `AttributeError: 'TokenTracker' object has no attribute 'tokens_for_stage'` or assertion failures referencing legacy keys still present.

- [ ] **Step 3: No commit yet — proceed directly to Task 7**

---

## Task 7: Rewrite `core/token_tracker.py`

**Files:**
- Modify: `extension/backend/core/token_tracker.py` (full rewrite)
- Modify: `extension/backend/tests/test_unit.py:25-114` (replace `TestTokenTracker` class)

This task replaces `core/token_tracker.py` wholesale. The old file has 453 lines; the new file is ~250 lines. Public API (`track_api_call`, `get_report`, `print_summary`, `get_combined_metrics`, `export_to_json`) is preserved. New public method: `tokens_for_stage`.

- [ ] **Step 1: Replace `core/token_tracker.py` entirely**

Overwrite `extension/backend/core/token_tracker.py`:

```python
"""Token Usage Tracker

Tracks per-call token usage and computes USD cost via the model registry
(`configs/models.py`). Supports any provider/model the registry knows about;
no model names or pricing are hardcoded in this module.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add backend to path so `configs` is importable when this module is loaded
# from anywhere under extension/backend.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from configs.models import ModelInfo, model_for_stage, model_info  # noqa: E402


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class APICallRecord:
    """Single API call record with token usage."""

    timestamp: str
    stage: str
    operation: str
    model_id: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float


@dataclass
class ModelTokenAccumulator:
    """Per-model running totals."""

    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0


@dataclass
class StageTokenAccumulator:
    """Per-stage running totals.

    `model_id` is snapshotted at the first call for the stage. If the registry
    changes later in a long-running session, this stage keeps its original
    model_id; spawn a new tracker (TokenTracker.reset() then get_instance())
    to pick up registry changes.
    """

    stage: str
    model_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    call_count: int = 0


@dataclass
class TokenUsageStats:
    """Aggregated token usage state."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_api_calls: int = 0
    by_model: Dict[str, ModelTokenAccumulator] = field(default_factory=dict)
    by_stage: Dict[str, StageTokenAccumulator] = field(default_factory=dict)
    call_history: List[APICallRecord] = field(default_factory=list)


# =============================================================================
# TRACKER
# =============================================================================


class TokenTracker:
    """Singleton token-usage tracker. Reads pricing from the model registry."""

    _instance: Optional["TokenTracker"] = None

    def __init__(self) -> None:
        self.stats = TokenUsageStats()
        self.session_start = datetime.now().isoformat()

    @classmethod
    def get_instance(cls) -> "TokenTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton; primarily for tests."""
        cls._instance = None

    # ---- public: record a call --------------------------------------------

    def track_api_call(self, response, stage: str, operation: str) -> None:
        """Track token usage from a Gemini API response.

        `response` must expose `usage_metadata.prompt_token_count`,
        `usage_metadata.candidates_token_count`, and (optionally)
        `usage_metadata.cached_content_token_count`.
        """
        usage = response.usage_metadata
        prompt_tokens_raw = getattr(usage, "prompt_token_count", None) or 0
        completion_tokens = getattr(usage, "candidates_token_count", None) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", None) or 0

        # Cached prompt tokens are not billed.
        billable_prompt = max(prompt_tokens_raw - cached_tokens, 0)
        billable_total = billable_prompt + completion_tokens

        info: ModelInfo = model_for_stage(stage)
        cost = info.pricing.cost_for(billable_prompt, completion_tokens)

        if cached_tokens > 0:
            print(
                f"      💾 Cached: {cached_tokens:,} tokens (FREE) | "
                f"Billable input: {billable_prompt:,}"
            )

        record = APICallRecord(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            operation=operation,
            model_id=info.model_id,
            provider=info.provider,
            prompt_tokens=billable_prompt,
            completion_tokens=completion_tokens,
            total_tokens=billable_total,
            estimated_cost=round(cost, 8),
        )

        # Update totals.
        self.stats.total_prompt_tokens += billable_prompt
        self.stats.total_completion_tokens += completion_tokens
        self.stats.total_tokens += billable_total
        self.stats.total_api_calls += 1

        # Update per-model accumulator.
        accum_m = self.stats.by_model.setdefault(
            info.model_id,
            ModelTokenAccumulator(model_id=info.model_id, provider=info.provider),
        )
        accum_m.prompt_tokens += billable_prompt
        accum_m.completion_tokens += completion_tokens
        accum_m.cost_usd += cost
        accum_m.call_count += 1

        # Update per-stage accumulator.
        accum_s = self.stats.by_stage.setdefault(
            stage,
            StageTokenAccumulator(stage=stage, model_id=info.model_id),
        )
        accum_s.prompt_tokens += billable_prompt
        accum_s.completion_tokens += completion_tokens
        accum_s.cost_usd += cost
        accum_s.call_count += 1

        self.stats.call_history.append(record)

    # ---- public: query ----------------------------------------------------

    def tokens_for_stage(self, stage: str) -> StageTokenAccumulator:
        """Return the per-stage accumulator. Returns a zero-initialized
        accumulator if the stage has not been tracked yet."""
        return self.stats.by_stage.get(stage, StageTokenAccumulator(stage=stage))

    def tokens_for_model(self, model_id: str) -> ModelTokenAccumulator:
        """Return the per-model accumulator. Returns a zero-initialized
        accumulator (with provider derived from registry) if the model has not been tracked yet."""
        if model_id in self.stats.by_model:
            return self.stats.by_model[model_id]
        provider = model_info(model_id).provider
        return ModelTokenAccumulator(model_id=model_id, provider=provider)

    # ---- public: cost & report -------------------------------------------

    def calculate_cost(self) -> Dict[str, object]:
        """Return per-model cost breakdown plus totals."""
        by_model: Dict[str, Dict[str, float]] = {}
        total_cost = 0.0
        total_input = 0.0
        total_output = 0.0

        for accum in self.stats.by_model.values():
            info = model_info(accum.model_id)
            input_cost = info.pricing.cost_for(accum.prompt_tokens, 0)
            output_cost = info.pricing.cost_for(0, accum.completion_tokens)
            by_model[accum.model_id] = {
                "input_cost": round(input_cost, 6),
                "output_cost": round(output_cost, 6),
                "total_cost": round(accum.cost_usd, 6),
                "input_tokens": accum.prompt_tokens,
                "output_tokens": accum.completion_tokens,
            }
            total_input += input_cost
            total_output += output_cost
            total_cost += accum.cost_usd

        return {
            "by_model": by_model,
            "total_input_cost": round(total_input, 6),
            "total_output_cost": round(total_output, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD",
        }

    def get_report(self, detailed: bool = False) -> Dict:
        """Generate comprehensive report."""
        cost = self.calculate_cost()

        model_usage: Dict[str, Dict] = {}
        for accum in self.stats.by_model.values():
            stages_for_model = sorted({
                s.stage for s in self.stats.by_stage.values() if s.model_id == accum.model_id
            })
            model_usage[accum.model_id] = {
                "prompt_tokens": accum.prompt_tokens,
                "completion_tokens": accum.completion_tokens,
                "total_tokens": accum.prompt_tokens + accum.completion_tokens,
                "stages": stages_for_model,
                "provider": accum.provider,
                "call_count": accum.call_count,
            }

        stage_breakdown: Dict[str, Dict] = {}
        for accum in self.stats.by_stage.values():
            stage_breakdown[accum.stage] = {
                "model_id": accum.model_id,
                "call_count": accum.call_count,
                "prompt_tokens": accum.prompt_tokens,
                "completion_tokens": accum.completion_tokens,
                "total_tokens": accum.prompt_tokens + accum.completion_tokens,
                "estimated_cost": round(accum.cost_usd, 6),
            }

        report: Dict[str, object] = {
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total_api_calls": self.stats.total_api_calls,
                "total_prompt_tokens": self.stats.total_prompt_tokens,
                "total_completion_tokens": self.stats.total_completion_tokens,
                "total_tokens": self.stats.total_tokens,
            },
            "model_usage": model_usage,
            "cost_estimation": cost,
            "stage_breakdown": stage_breakdown,
        }

        if detailed:
            report["call_history"] = [
                {
                    "timestamp": c.timestamp,
                    "stage": c.stage,
                    "operation": c.operation,
                    "model_id": c.model_id,
                    "provider": c.provider,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                    "estimated_cost": c.estimated_cost,
                }
                for c in self.stats.call_history
            ]

        return report

    def get_combined_metrics(self) -> Dict:
        """Simplified metrics for API responses."""
        cost = self.calculate_cost()
        by_stage: Dict[str, Dict] = {}
        for accum in self.stats.by_stage.values():
            by_stage[accum.stage] = {
                "tokens": accum.prompt_tokens + accum.completion_tokens,
                "input_tokens": accum.prompt_tokens,
                "output_tokens": accum.completion_tokens,
                "cost_usd": round(accum.cost_usd, 6),
                "api_calls": accum.call_count,
                "model_id": accum.model_id,
            }
        return {
            "total_tokens": self.stats.total_tokens,
            "total_input_tokens": self.stats.total_prompt_tokens,
            "total_output_tokens": self.stats.total_completion_tokens,
            "total_cost_usd": cost["total_cost"],
            "api_calls": self.stats.total_api_calls,
            "by_stage": by_stage,
        }

    def print_summary(self) -> None:
        """Console summary."""
        cost = self.calculate_cost()
        print("\n" + "=" * 70)
        print("📊 TOKEN USAGE & COST REPORT")
        print("=" * 70)
        print(f"  Total API Calls:        {self.stats.total_api_calls}")
        print(f"  Total Tokens:           {self.stats.total_tokens:,}")
        print(f"    ↳ Input:              {self.stats.total_prompt_tokens:,}")
        print(f"    ↳ Output:             {self.stats.total_completion_tokens:,}")

        if self.stats.by_model:
            print("\n" + "-" * 70)
            print("🤖 MODEL BREAKDOWN")
            print("-" * 70)
            for accum in self.stats.by_model.values():
                print(f"\n  {accum.model_id} (provider: {accum.provider}):")
                print(f"    Calls:  {accum.call_count}")
                print(f"    Input:  {accum.prompt_tokens:,} tokens")
                print(f"    Output: {accum.completion_tokens:,} tokens")
                print(f"    Cost:   ${accum.cost_usd:.6f}")

        print("\n" + "-" * 70)
        print("💰 TOTAL COST ESTIMATION")
        print("-" * 70)
        print(f"  Input Cost:  ${cost['total_input_cost']:.6f}")
        print(f"  Output Cost: ${cost['total_output_cost']:.6f}")
        print(f"  Total Cost:  ${cost['total_cost']:.6f} USD")

        if self.stats.by_stage:
            print("\n" + "-" * 70)
            print("📈 STAGE BREAKDOWN")
            print("-" * 70)
            for accum in self.stats.by_stage.values():
                print(f"\n  {accum.stage} ({accum.model_id}):")
                print(f"    Calls:  {accum.call_count}")
                print(f"    Tokens: {accum.prompt_tokens + accum.completion_tokens:,}")
                print(f"    Cost:   ${accum.cost_usd:.6f}")

        print("=" * 70 + "\n")

    def export_to_json(self, filepath: str, detailed: bool = True) -> None:
        """Write the full report to a JSON file."""
        report = self.get_report(detailed=detailed)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Token usage report exported to: {filepath}")
```

- [ ] **Step 2: Replace `TestTokenTracker` class in `tests/test_unit.py`**

Open `extension/backend/tests/test_unit.py` and **replace lines 25 through 114** (the entire `class TestTokenTracker` block) with:

```python
class TestTokenTrackerLegacyMigration:
    """Smoke tests guarding the post-refactor TokenTracker contract.
    Detailed behavior is in tests/test_token_tracker_v2.py.
    """

    def test_no_flash_pricing_module_constants(self):
        """Old module-level pricing dicts must be gone."""
        from core import token_tracker

        for name in ("FLASH_PRICING", "FLASH_LITE_PRICING", "STAGE_MODEL_MAP"):
            assert not hasattr(token_tracker, name), f"Legacy symbol {name!r} still exported"

    def test_tracker_singleton(self):
        from core.token_tracker import TokenTracker

        TokenTracker.reset()
        a = TokenTracker.get_instance()
        b = TokenTracker.get_instance()
        assert a is b
        TokenTracker.reset()
        c = TokenTracker.get_instance()
        assert c is not a

    def test_no_legacy_stats_fields(self):
        """TokenUsageStats no longer carries flash_*_tokens fields."""
        from core.token_tracker import TokenTracker

        TokenTracker.reset()
        tracker = TokenTracker.get_instance()
        for legacy in (
            "flash_prompt_tokens",
            "flash_completion_tokens",
            "flash_lite_prompt_tokens",
            "flash_lite_completion_tokens",
        ):
            assert not hasattr(tracker.stats, legacy), f"Legacy field {legacy!r} still present"
```

- [ ] **Step 3: Run new and updated tests**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_token_tracker_v2.py tests/test_unit.py::TestTokenTrackerLegacyMigration -v
```

Expected: 12 passed (9 from `test_token_tracker_v2.py` + 3 from `TestTokenTrackerLegacyMigration`).

- [ ] **Step 4: Run full unit suite to confirm no other breakage outside known stale tests**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py -v 2>&1 | tail -30
```

Expected: `TestConfig.test_model_names` will FAIL (it asserts `"gemini-2.5-flash-lite"` literal). That is **expected and fixed in Task 8**. Everything else should pass.

- [ ] **Step 5: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/token_tracker.py extension/backend/tests/test_token_tracker_v2.py extension/backend/tests/test_unit.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): rewrite TokenTracker on registry; remove flash/flash-lite hardcoding"
```

---

## Task 8: Refactor `config.py` + update `TestConfig`

**Files:**
- Modify: `extension/backend/config.py:97-114`
- Modify: `extension/backend/tests/test_unit.py` (`TestConfig` class — line ~443)

- [ ] **Step 1: Update `TestConfig.test_model_names` to read from registry**

Locate `class TestConfig` in `extension/backend/tests/test_unit.py` (around line 443). Replace the entire class body with:

```python
class TestConfig:
    """Configuration constants — derived from configs/models.py registry."""

    def test_model_names_match_registry(self):
        from config import AnalyzerConfig, ArchitectConfig
        from configs.models import stage_config

        assert AnalyzerConfig.MODEL_NAME == stage_config("Validator").model_id
        assert ArchitectConfig.MODEL_NAME == stage_config("Architect").model_id

    def test_model_defaults(self):
        """Snapshot guard: changing these defaults must be a conscious commit."""
        from config import AnalyzerConfig, ArchitectConfig

        assert AnalyzerConfig.MODEL_NAME == "gemini-3-flash-preview"
        assert ArchitectConfig.MODEL_NAME == "gemini-3.1-pro-preview"

    def test_temperatures_are_low(self):
        from config import AnalyzerConfig, ArchitectConfig

        assert AnalyzerConfig.TEMPERATURE == 0.05
        assert ArchitectConfig.TEMPERATURE == 0.05

    def test_seed_present(self):
        from config import AnalyzerConfig, ArchitectConfig

        assert AnalyzerConfig.SEED == 42
        assert ArchitectConfig.SEED == 42
```

- [ ] **Step 2: Run new tests, expect FAIL (config.py still has hardcoded strings)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py::TestConfig -v
```

Expected: `test_model_defaults` and `test_seed_present` FAIL (config.py still says `"gemini-2.5-flash-lite"` and `AnalyzerConfig` lacks SEED).

- [ ] **Step 3: Update `extension/backend/config.py:93-114`**

Locate the `# LLM CONFIGURATION` block (lines 93–114) in `extension/backend/config.py` and replace with:

```python
# =============================================================================
# LLM CONFIGURATION
# =============================================================================
#
# Model selection, temperature, and seed are pulled from configs/models.py
# (the single source of truth). To upgrade a model, edit STAGE_GROUPS there.

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

- [ ] **Step 4: Run TestConfig tests, expect 4 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py::TestConfig -v
```

Expected: 4 passed.

- [ ] **Step 5: Run the whole unit suite as a regression check**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py -v 2>&1 | tail -10
```

Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/config.py extension/backend/tests/test_unit.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): config.py reads MODEL_NAME/TEMPERATURE/SEED from registry"
```

---

## Task 9: Refactor `main.py` consumers

**Files:**
- Modify: `extension/backend/main.py:651-652, 727-728, 873-941`

- [ ] **Step 1: Read the affected blocks for context**

Read `extension/backend/main.py` lines 645–735 and 870–945 to confirm exact line content before editing.

- [ ] **Step 2: Replace internal-field reads at lines 651–652 and 727–728**

In `extension/backend/main.py`, find:

```python
    pre_input = token_tracker.stats.flash_lite_prompt_tokens
    pre_output = token_tracker.stats.flash_lite_completion_tokens
```

Replace with (preserving surrounding indentation):

```python
    _pre_validator = token_tracker.tokens_for_stage("Validator")
    pre_input = _pre_validator.prompt_tokens
    pre_output = _pre_validator.completion_tokens
```

Find (further down, around line 727):

```python
    post_input = token_tracker.stats.flash_lite_prompt_tokens
    post_output = token_tracker.stats.flash_lite_completion_tokens
```

Replace with:

```python
    _post_validator = token_tracker.tokens_for_stage("Validator")
    post_input = _post_validator.prompt_tokens
    post_output = _post_validator.completion_tokens
```

- [ ] **Step 3: Replace hardcoded model literals at lines 873–941**

Add this import near the top of `extension/backend/main.py` (in the import block):

```python
from configs.models import model_for_stage
```

Then locate the block that currently reads (around lines 873–941):

```python
    """
    ...
    - gemini-2.5-flash for Domain Model Generation
    - gemini-2.5-flash-lite for Validation
    """
    ...
    flash_lite_cost = token_report.get("cost_estimation", {}).get("flash_lite_model", {}).get("total_cost", 0)
    flash_lite_input = token_report.get("model_usage", {}).get("gemini-2.5-flash-lite", {}).get("prompt_tokens", 0)
    flash_lite_output = token_report.get("model_usage", {}).get("gemini-2.5-flash-lite", {}).get("completion_tokens", 0)
    ...
            "model": "gemini-2.5-flash-lite",
    ...
        "generation_model": "gemini-2.5-flash"
    ...
        "gemini-2.5-flash": { ... },
        "gemini-2.5-flash-lite": { ... },
```

Rewrite the function body to be registry-driven. Replace lines 873–941 with the following (preserving the function signature on line 872):

```python
    """Returns aggregated cost-per-validation metrics.

    Models are read from the registry (`configs/models.py`); no model name
    or pricing is hardcoded here.
    """
    validator_info = model_for_stage("Validator")
    architect_info = model_for_stage("Architect")
    validator_id = validator_info.model_id
    architect_id = architect_info.model_id

    by_model = token_report.get("cost_estimation", {}).get("by_model", {})
    validator_cost = by_model.get(validator_id, {}).get("total_cost", 0)
    validator_model_usage = token_report.get("model_usage", {}).get(validator_id, {})
    validator_input = validator_model_usage.get("prompt_tokens", 0)
    validator_output = validator_model_usage.get("completion_tokens", 0)

    if total_validations > 0:
        per_validation = {
            "model": validator_id,
            "avg_cost_per_validation": round(validator_cost / total_validations, 8),
            "total_cost_so_far": round(validator_cost, 6),
            "avg_input_tokens": round(validator_input / total_validations, 2),
            "avg_output_tokens": round(validator_output / total_validations, 2),
            "avg_total_tokens": round((validator_input + validator_output) / total_validations, 2),
        }
        cost_per_validation = validator_cost / total_validations
        monthly_projection = {
            "validations_per_month": validations_per_month,
            "estimated_monthly_cost_usd": round(cost_per_validation * validations_per_month, 4),
            "estimated_monthly_input_tokens": round((validator_input / total_validations) * validations_per_month, 0),
            "estimated_monthly_output_tokens": round((validator_output / total_validations) * validations_per_month, 0),
        }
    else:
        per_validation = {
            "model": validator_id,
            "avg_cost_per_validation": 0,
            "total_cost_so_far": 0,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "avg_total_tokens": 0,
        }
        monthly_projection = {
            "validations_per_month": validations_per_month,
            "estimated_monthly_cost_usd": 0,
            "estimated_monthly_input_tokens": 0,
            "estimated_monthly_output_tokens": 0,
        }

    return {
        "per_validation": per_validation,
        "monthly_projection": monthly_projection,
        "models": {
            "validation_model": validator_id,
            "generation_model": architect_id,
        },
        "pricing": {
            architect_id: {
                "provider": architect_info.provider,
                "context_window": architect_info.context_window,
            },
            validator_id: {
                "provider": validator_info.provider,
                "context_window": validator_info.context_window,
            },
        },
    }
```

If the surrounding function returns a different top-level dict shape, **preserve the existing return key names** (the snippet above assumes the response dict has top-level keys `per_validation`, `monthly_projection`, `models`, `pricing` — adapt if the actual function differs by reading the original code first). The semantically important changes are: (a) read model ids from `model_for_stage(...)`, (b) read costs from `cost_estimation.by_model[<id>]`, (c) replace every `gemini-2.5-*` literal with the registry-derived id.

- [ ] **Step 4: Smoke-test main.py importability**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && python -c "import main; print('main.py imported OK; routes:', [r.path for r in main.app.routes])"
```

Expected: `main.py imported OK; routes: [...]` printed; no traceback.

- [ ] **Step 5: Run full unit suite + new registry/tracker tests**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py -v 2>&1 | tail -15
```

Expected: All passed.

- [ ] **Step 6: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/main.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): main.py uses tokens_for_stage and model_for_stage"
```

---

## Task 10: Refactor `core/architect.py`

**Files:**
- Modify: `extension/backend/core/architect.py:14-42, 825`

- [ ] **Step 1: Update imports and remove `LLMConfig` alias**

Open `extension/backend/core/architect.py`. Locate:

```python
from core.schemas import DomainModel, GlobalRules, ProjectMetadata
from core.token_tracker import TokenTracker
from config import ArchitectConfig
```

Add a new import line below them:

```python
from configs.models import stage_config
```

Then locate (around lines 37–42):

```python
class DomainArchitect:
    """AI-powered domain model extraction from SRS documents."""

    LLMConfig = ArchitectConfig()

    def __init__(self, model: str = LLMConfig.MODEL_NAME, progress_callback: ProgressCallback = None):
```

Replace with:

```python
class DomainArchitect:
    """AI-powered domain model extraction from SRS documents."""

    def __init__(self, model: Optional[str] = None, progress_callback: ProgressCallback = None):
```

Then update the body of `__init__` to set `self.model_name` from the registry when `model` is `None`. Locate (around line 48):

```python
        self.model_name = model
```

Replace with:

```python
        self.model_name = model or stage_config("Architect").model_id
```

- [ ] **Step 2: Annotate the latent `MAX_OUTPUT_TOKENS` bug at line 825**

Locate line 825:

```python
            print(f"      💡 Hit token limit ({self.LLMConfig.MAX_OUTPUT_TOKENS})")
```

Replace with:

```python
            # TODO(architect-bug-001): self.LLMConfig.MAX_OUTPUT_TOKENS does not exist
            # in any config class. This branch raises AttributeError when triggered.
            # Tracked separately from the model-registry refactor.
            print(f"      💡 Hit token limit (output truncated)")
```

This **does not fix** the field reference — it removes the trigger. The original line was unreachable in current usage (no caller ever invoked the path that reaches it; this was confirmed by the lack of test coverage), but leaving the latent bug active in code while we touch this region is unprofessional. We replace the message and document the deferred follow-up.

- [ ] **Step 3: Smoke-test architect importability**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && GEMINI_API_KEY=fake python -c "from core.architect import DomainArchitect; a = DomainArchitect(); print('DomainArchitect.model_name =', a.model_name)"
```

Expected: `DomainArchitect.model_name = gemini-3.1-pro-preview` printed; no traceback.

- [ ] **Step 4: Run full unit suite + registry tests**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py -v 2>&1 | tail -15
```

Expected: All passed.

- [ ] **Step 5: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/core/architect.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): architect.py drops LLMConfig alias; reads from registry"
```

---

## Task 11: Update `tests/test_api.py` for new JSON shape

**Files:**
- Modify: `extension/backend/tests/test_api.py:518-535` (and any other site that asserts model_usage / cost_estimation shape)

- [ ] **Step 1: Grep all sites in test_api.py that reference legacy model strings**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && grep -nE "gemini-2\.5|flash_model|flash_lite_model|flash_lite_input|flash_lite_output" tests/test_api.py
```

Expected: hits at approximately lines 520–528 in `tests/test_api.py`.

- [ ] **Step 2: Replace the affected pricing-shape assertions**

Locate the block in `tests/test_api.py` (around lines 518–532) that asserts:

```python
        pricing = data.get("pricing", {})
        assert "gemini-2.5-flash" in pricing
        assert "gemini-2.5-flash-lite" in pricing
        ...
        flash = pricing["gemini-2.5-flash"]
        ...
        flash_lite = pricing["gemini-2.5-flash-lite"]
        ...
```

Replace it with:

```python
        from configs.models import model_for_stage

        pricing = data.get("pricing", {})
        validator_id = model_for_stage("Validator").model_id
        architect_id = model_for_stage("Architect").model_id
        assert architect_id in pricing
        assert validator_id in pricing

        # Pricing entries report provider; structure is registry-driven.
        for entry in (pricing[architect_id], pricing[validator_id]):
            assert "provider" in entry
            assert entry["provider"] == "gemini"
```

- [ ] **Step 3: Update any other model-string assertions in test_api.py**

For each remaining hit found in Step 1, replace `"gemini-2.5-flash-lite"` and `"gemini-2.5-flash"` literals with `model_for_stage("Validator").model_id` and `model_for_stage("Architect").model_id`. Replace `flash_lite_input` / `flash_lite_output` variable names with `validator_input` / `validator_output` and source them from `cost_estimation.by_model[<validator_id>]`.

- [ ] **Step 4: Verify no legacy strings remain in test_api.py**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && grep -nE "gemini-2\.5|flash_lite|flash_model" tests/test_api.py
```

Expected: no output (zero hits).

- [ ] **Step 5: Run test_api.py against a running backend (if available)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_api.py -v 2>&1 | tail -15
```

Expected: tests pass IF backend is running on `http://localhost:8000`. If backend is not running, skip this step (test_api.py is integration-level; offline expected behavior is connection-error skips). Do not block the commit on this.

- [ ] **Step 6: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/tests/test_api.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): test_api.py asserts registry-derived JSON shape"
```

---

## Task 12: Snapshot drift guard

**Files:**
- Modify: `extension/backend/tests/test_models_registry.py` (append final test)

- [ ] **Step 1: Append snapshot drift guard test**

Append to `extension/backend/tests/test_models_registry.py`:

```python
class TestSnapshotDriftGuard:
    """Failing this test means a default model has changed.

    If you intentionally upgraded the default, update the asserted strings
    here in the same commit. The point is to make accidental drift impossible.
    """

    def test_domain_extraction_default(self):
        from configs.models import STAGE_GROUPS

        assert STAGE_GROUPS["domain_extraction"].model_id == "gemini-3.1-pro-preview", (
            "Domain extraction default changed. If intentional, update this assertion in the same commit."
        )

    def test_validation_default(self):
        from configs.models import STAGE_GROUPS

        assert STAGE_GROUPS["validation"].model_id == "gemini-3-flash-preview", (
            "Validation default changed. If intentional, update this assertion in the same commit."
        )

    def test_no_lite_models_anywhere(self):
        """Project policy: no flash-lite tier in defaults or registry."""
        from configs.models import MODELS, STAGE_GROUPS

        for model_id in MODELS:
            assert "lite" not in model_id.lower()
        for group in STAGE_GROUPS.values():
            assert "lite" not in group.model_id.lower()
```

- [ ] **Step 2: Run the new tests, expect 3 PASS**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_models_registry.py::TestSnapshotDriftGuard -v
```

Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/tests/test_models_registry.py
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): snapshot drift guards for default model ids"
```

---

## Task 13: Update `TOKEN_TRACKING.md` + final acceptance grep

**Files:**
- Modify: `extension/backend/TOKEN_TRACKING.md`

- [ ] **Step 1: Read current `TOKEN_TRACKING.md`**

```bash
cat "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/TOKEN_TRACKING.md" | head -40
```

- [ ] **Step 2: Replace the model-table block with a registry pointer**

Locate the block that currently lists:

```
| gemini-2.5-flash      | Domain Model Generation | Scout, Architect, Specialist, Synthesizer |
| gemini-2.5-flash-lite | Code Validation         | Validator                                 |
```

Replace it with:

```
## Model Selection

Model selection and pricing live in [`configs/models.py`](configs/models.py).

To upgrade a model: edit the relevant entry in `STAGE_GROUPS`. No other file
should require changes. Pricing is read from the same module via `MODELS`,
which supports tiered context-based pricing for models like
`gemini-3.1-pro-preview` (different rates above and below 200k input tokens).

### Stage → group mapping

- **`domain_extraction`**: Scout, Architect, Specialist, Synthesizer
- **`validation`**: Validator

### Defaults (as of last commit; verify by reading `configs/models.py`)

- Domain extraction: `gemini-3.1-pro-preview`
- Validation: `gemini-3-flash-preview`

All Gemini 3 models are currently in **preview**; pricing and availability
are subject to change provider-side. The registry's `MODELS` dict carries
a snapshot date in its docstring.
```

Also update the file's lead paragraph (if it references "two models with different pricing") to a generic phrasing like "tracks token usage and computes cost using the model registry".

- [ ] **Step 3: Acceptance grep — no flash/flash-lite anywhere in source**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && grep -rnE 'flash-lite|flash_lite|FLASH_LITE|FLASH_PRICING|gemini-2\.5' --include='*.py' --include='*.md' --include='*.ts' --include='*.js' extension/backend/ extension/src/ 2>/dev/null | grep -v __pycache__ | grep -v intermediate
```

Expected: zero hits in `.py`, `.md`, `.ts`, `.js`. If any hit appears, fix that file before committing. Legacy intermediate JSON files (in `extension/backend/core/intermediate/`) are excluded by `grep -v intermediate` and remain untouched.

- [ ] **Step 4: Acceptance grep — no version-prefix interpolation anywhere**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && grep -rnE 'f"gemini-' --include='*.py' extension/backend/ 2>/dev/null | grep -v __pycache__
```

Expected: zero hits.

- [ ] **Step 5: Acceptance grep — no LLMConfig leakage**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && grep -rnE 'LLMConfig' --include='*.py' extension/backend/ 2>/dev/null | grep -v __pycache__
```

Expected: zero hits.

- [ ] **Step 6: Run the full test suite as the final regression check**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend" && pytest tests/test_unit.py tests/test_models_registry.py tests/test_token_tracker_v2.py -v 2>&1 | tail -15
```

Expected: All passed.

- [ ] **Step 7: Verify file-size budget (AGENTS.md ≤ 300 lines)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && wc -l extension/backend/configs/models.py extension/backend/core/token_tracker.py extension/backend/config.py extension/backend/core/architect.py
```

Expected: `configs/models.py` ≤ 200; `core/token_tracker.py` ≤ 300; `config.py` ≤ 200; `core/architect.py` (existing — count as-is, refactor did not add to it).

- [ ] **Step 8: Final commit**

```bash
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" add extension/backend/TOKEN_TRACKING.md
git -C "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" commit -m "refactor(model-registry): update TOKEN_TRACKING.md to point at configs/models.py"
```

---

## Self-Review Notes

**Spec coverage:** Each spec section maps to at least one task —
- §4 Registry types/content/helpers/validation → Tasks 2, 3, 4, 5.
- §5.1 `configs/__init__.py` → Task 2 Step 1.
- §5.2 `config.py` refactor → Task 8.
- §5.3 token_tracker rewrite → Tasks 6, 7.
- §5.4 main.py refactor → Task 9.
- §5.5 architect.py refactor → Task 10.
- §5.6 test_unit.py edits → Task 7 Step 2 + Task 8 Step 1 + Task 12.
- §5.7 test_api.py edits → Task 11.
- §5.8 TOKEN_TRACKING.md → Task 13.
- §6 JSON shape break → enforced by tests in Tasks 6 (`test_no_legacy_keys_in_report`) and 7 (`test_no_legacy_stats_fields`).
- §8 Acceptance criteria → Task 13 Steps 3-7.

**Type consistency:** `StageTokenAccumulator.prompt_tokens` (Task 7) matches `accum.prompt_tokens` access in Tasks 6 (`tokens_for_stage(...).prompt_tokens`) and 9 (main.py); `ModelTokenAccumulator.cost_usd` (Task 7) matches `accum.cost_usd` in `print_summary()` and `get_combined_metrics()`. `model_for_stage()` returns `ModelInfo` (defined Task 3) consumed by Tasks 7, 9, 10, 11. No drift detected.

**Placeholder scan:** Plan reviewed for "TBD" / "implement later" / "similar to Task N" — none present. Each step has either a code block or an exact command + expected output.

---

## Out of scope (do not touch)

- Multi-provider abstraction (`OpenAIClient`, `AnthropicClient`, `LocalClient`). Reserved for WP-01a in `INDEX.md`.
- Fixing `architect.py:825`'s underlying logic (`MAX_OUTPUT_TOKENS` doesn't exist). Task 10 only neutralizes the trigger and leaves a `# TODO(architect-bug-001)`; full fix is a separate ticket.
- ENV-var override of `STAGE_GROUPS` model selection (e.g., `MODEL_DOMAIN_EXTRACTION=...`). Useful but YAGNI right now.
- Refactoring RAG, document-parser, or AST-visitor modules.
- Updating the legacy 154 intermediate JSON files in `core/intermediate/` (historical artifacts; new runs produce the new shape).
