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
