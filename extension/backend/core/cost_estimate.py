"""Cost estimation helpers for pre-flight USD estimates.

Reads pricing from the D1 model registry (core/llm/registry.py) and the
stage→model mapping in configs/models.py. Used by scripts/cost_estimate.py
and by tests in tests/test_token_tracker_multi_provider.py.

Public API:
    estimate_single_call_cost(model_id, prompt_tokens, completion_tokens) -> float
    estimate_run_cost(model_id, prompt_tokens, completion_tokens, runs) -> dict
    estimate_per_stage_cost(prompt_tokens, completion_tokens, runs) -> list[dict]
"""

from typing import Any, Dict, List

from core.llm.registry import MODELS


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def estimate_single_call_cost(
    model_id: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Look up the model's Pricing and compute cost for one API call (USD).

    Raises ValueError if model_id is unknown in the D1 registry.
    """
    if model_id not in MODELS:
        raise ValueError(
            f"Unknown model_id {model_id!r}. "
            f"Known models: {sorted(MODELS.keys())!r}"
        )
    return MODELS[model_id].pricing.cost_for(prompt_tokens, completion_tokens)


def estimate_run_cost(
    model_id: str, prompt_tokens: int, completion_tokens: int, runs: int
) -> Dict[str, float]:
    """Compute total cost for N identical runs of a single model.

    Returns a dict with keys: 'per_call' (float), 'total' (float), 'runs' (int).
    Raises ValueError on unknown model_id or runs <= 0.
    """
    if runs <= 0:
        raise ValueError(f"runs must be > 0, got {runs!r}")
    per_call = estimate_single_call_cost(model_id, prompt_tokens, completion_tokens)
    return {
        "per_call": per_call,
        "total": per_call * runs,
        "runs": runs,
    }


def estimate_per_stage_cost(
    prompt_tokens: int, completion_tokens: int, runs: int
) -> List[Dict[str, Any]]:
    """Compute per-stage cost for each pipeline stage in configs.models.STAGE_TO_GROUP.

    For each stage, looks up the model via configs.models.stage_config, then
    retrieves pricing from the D1 registry (core/llm/registry.py) if the model
    is present there, or from configs.models otherwise.

    Returns a list of dicts, one per stage:
        {'stage': str, 'model_id': str, 'per_call': float, 'total': float}
    plus a final sentinel entry:
        {'stage': '__total__', 'model_id': '', 'per_call': 0.0, 'total': float}

    The __total__.total equals the sum of all stage totals.
    """
    from configs.models import stage_config, STAGE_TO_GROUP, MODELS as OLD_MODELS

    entries: List[Dict[str, Any]] = []
    grand_total = 0.0

    for stage in sorted(STAGE_TO_GROUP.keys()):
        sc = stage_config(stage)
        model_id = sc.model_id

        # Prefer pricing from D1 registry; fall back to configs.models registry
        if model_id in MODELS:
            pricing = MODELS[model_id].pricing
        elif model_id in OLD_MODELS:
            pricing = OLD_MODELS[model_id].pricing
        else:
            raise ValueError(
                f"Stage {stage!r} model_id {model_id!r} not found in either registry"
            )

        per_call = pricing.cost_for(prompt_tokens, completion_tokens)
        total = per_call * runs
        grand_total += total
        entries.append(
            {
                "stage": stage,
                "model_id": model_id,
                "per_call": per_call,
                "total": total,
            }
        )

    entries.append(
        {
            "stage": "__total__",
            "model_id": "",
            "per_call": 0.0,
            "total": grand_total,
        }
    )
    return entries
