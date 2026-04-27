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
