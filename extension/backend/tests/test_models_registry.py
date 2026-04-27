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
        with pytest.raises(AttributeError):
            t.input_per_1m_usd = 1.0  # type: ignore[misc]


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
        with pytest.raises(AttributeError):
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
        with pytest.raises(AttributeError):
            sc.temperature = 0.1  # type: ignore[misc]


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
