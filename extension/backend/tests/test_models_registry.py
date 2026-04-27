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
