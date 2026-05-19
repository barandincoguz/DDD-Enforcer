"""6-model registry per D1 lockdown.

Two closed Gemini models + four OSS via Ollama Cloud. This module is
the single source of truth for what models the project knows about;
core.llm clients consult it to resolve a model_id into a ModelSpec
with provider + pricing + context window.

Pricing snapshot: 2026-05-19.
- Gemini: https://ai.google.dev/gemini-api/docs/pricing
- Ollama Cloud: flat-rate subscription; per-token cost is amortized,
  so the in-token price for OSS models is reported as 0.0 USD/M and
  the compute_mode marker carries the billing semantics.

Adding a new model: append an entry to MODELS below. STAGE_GROUPS in
configs/models.py is the place to wire a model_id to a pipeline
stage; this registry stays provider-data-only.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple


Provider = Literal["gemini", "ollama"]
ComputeMode = Literal["per_token", "subscription_flat"]


@dataclass(frozen=True)
class PriceTier:
    """One pricing tier. Tiers are matched in order; first match wins."""
    max_prompt_tokens: Optional[int]
    input_per_1m_usd: float
    output_per_1m_usd: float


@dataclass(frozen=True)
class Pricing:
    """Cost calculator. Supports flat or context-tiered pricing."""
    tiers: Tuple[PriceTier, ...]

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
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


def _flat(input_usd: float, output_usd: float) -> Pricing:
    return Pricing(tiers=(PriceTier(None, input_usd, output_usd),))


@dataclass(frozen=True)
class ModelSpec:
    """Static info about an LLM model."""
    model_id: str
    provider: Provider
    pricing: Pricing
    context_window: Optional[int]
    compute_mode: ComputeMode
    supports_json_schema: bool
    supports_seed: bool
    notes: str = ""


# =============================================================================
# D1 LOCKDOWN: 6 models — 2 Gemini closed + 4 OSS Ollama Cloud
# =============================================================================

MODELS: Dict[str, ModelSpec] = {
    # G1 — Gemini Pro Preview (top closed)
    "gemini-3.1-pro-preview": ModelSpec(
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        pricing=Pricing(tiers=(
            PriceTier(max_prompt_tokens=200_000, input_per_1m_usd=2.0,  output_per_1m_usd=12.0),
            PriceTier(max_prompt_tokens=None,    input_per_1m_usd=4.0,  output_per_1m_usd=18.0),
        )),
        context_window=1_000_000,
        compute_mode="per_token",
        supports_json_schema=True,
        supports_seed=True,
        notes="D1 G1 — top closed model for RQ2a scaling baseline.",
    ),
    # G2 — Gemini Flash-Lite (cheap closed). Fallback to gemini-2.5-flash if
    # 3.1-flash-lite is unavailable at run time (handled in GeminiClient).
    "gemini-3.1-flash-lite": ModelSpec(
        model_id="gemini-3.1-flash-lite",
        provider="gemini",
        pricing=_flat(input_usd=0.10, output_usd=0.40),
        context_window=1_000_000,
        compute_mode="per_token",
        supports_json_schema=True,
        supports_seed=True,
        notes="D1 G2 — cheap closed model. Runtime falls back to gemini-2.5-flash if unavailable.",
    ),
    # O1 — OSS GPT large
    "gpt-oss:120b-cloud": ModelSpec(
        model_id="gpt-oss:120b-cloud",
        provider="ollama",
        pricing=_flat(input_usd=0.0, output_usd=0.0),
        context_window=128_000,
        compute_mode="subscription_flat",
        supports_json_schema=True,
        supports_seed=True,
        notes="D1 O1 — OSS GPT 120B via Ollama Cloud; billed flat subscription.",
    ),
    # O2 — OSS coder-tuned
    "qwen3-coder-next:cloud": ModelSpec(
        model_id="qwen3-coder-next:cloud",
        provider="ollama",
        pricing=_flat(input_usd=0.0, output_usd=0.0),
        context_window=128_000,
        compute_mode="subscription_flat",
        supports_json_schema=True,
        supports_seed=True,
        notes="D1 O2 — Qwen3 coder via Ollama Cloud; flat subscription billing.",
    ),
    # O3 — OSS MoE
    "minimax-m2:cloud": ModelSpec(
        model_id="minimax-m2:cloud",
        provider="ollama",
        pricing=_flat(input_usd=0.0, output_usd=0.0),
        context_window=128_000,
        compute_mode="subscription_flat",
        supports_json_schema=True,
        supports_seed=True,
        notes="D1 O3 — MiniMax M2 via Ollama Cloud; flat subscription billing.",
    ),
    # O4 — OSS Gemma
    "gemma4:31b-cloud": ModelSpec(
        model_id="gemma4:31b-cloud",
        provider="ollama",
        pricing=_flat(input_usd=0.0, output_usd=0.0),
        context_window=64_000,
        compute_mode="subscription_flat",
        supports_json_schema=True,
        supports_seed=True,
        notes="D1 O4 — Gemma4 31B via Ollama Cloud; flat subscription billing.",
    ),
}


# =============================================================================
# HELPERS
# =============================================================================


def model_spec(model_id: str) -> ModelSpec:
    """Return ModelSpec for a known model_id. Raises KeyError on unknown."""
    try:
        return MODELS[model_id]
    except KeyError as e:
        raise KeyError(
            f"Unknown model_id {model_id!r}. Known: {sorted(MODELS.keys())!r}"
        ) from e


def models_by_provider(provider: Provider) -> Dict[str, ModelSpec]:
    """Return all models for a given provider."""
    return {mid: spec for mid, spec in MODELS.items() if spec.provider == provider}


def _validate_registry() -> None:
    """Fail loudly at import time on structural inconsistencies."""
    for model_id, spec in MODELS.items():
        if spec.model_id != model_id:
            raise RuntimeError(
                f"MODELS[{model_id!r}].model_id={spec.model_id!r} mismatch"
            )
        tiers = spec.pricing.tiers
        if not tiers:
            raise RuntimeError(f"MODELS[{model_id!r}].pricing.tiers is empty")
        if tiers[-1].max_prompt_tokens is not None:
            raise RuntimeError(
                f"MODELS[{model_id!r}]: last pricing tier must have "
                f"max_prompt_tokens=None"
            )
        for i, t in enumerate(tiers[:-1]):
            if t.max_prompt_tokens is None:
                raise RuntimeError(
                    f"MODELS[{model_id!r}]: only the last tier may have "
                    f"max_prompt_tokens=None (offending index: {i})"
                )


_validate_registry()
