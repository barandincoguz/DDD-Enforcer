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
