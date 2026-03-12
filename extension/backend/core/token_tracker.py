"""
Model-agnostic token and cost tracker.

This tracker preserves the existing reporting endpoints while removing the
previous hardcoded stage-to-model pricing assumptions. Pricing now comes from
configuration, and per-call parseability/cached-token metadata is retained for
research analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any, Dict, List, Optional

from config import PricingConfig


def _safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(":", "_")


@dataclass
class APICallRecord:
    """Single provider call usage record."""

    timestamp: str
    provider: str
    model: str
    stage: str
    operation: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    parse_success: bool
    retry_count: int
    estimated_cost: float


@dataclass
class TokenUsageStats:
    """Aggregated provider-call statistics."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cached_tokens: int = 0
    total_api_calls: int = 0
    parseable_calls: int = 0
    unparseable_calls: int = 0
    model_usage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stage_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    call_history: List[APICallRecord] = field(default_factory=list)


class TokenTracker:
    """Singleton token tracker with dynamic model/provider accounting."""

    _instance: Optional["TokenTracker"] = None

    def __init__(self):
        self.stats = TokenUsageStats()
        self.session_start = datetime.now().isoformat()

    @classmethod
    def get_instance(cls) -> "TokenTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def _get_model_pricing(self, model: str) -> Dict[str, float]:
        return PricingConfig.MODEL_PRICING.get(
            model,
            {
                "provider": "unknown",
                "input_per_1m_tokens": 0.0,
                "output_per_1m_tokens": 0.0,
                "notes": "Pricing unavailable.",
            },
        )

    def _calculate_call_cost(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        pricing = self._get_model_pricing(model)
        return (
            prompt_tokens * (pricing.get("input_per_1m_tokens", 0.0) / 1_000_000)
            + completion_tokens * (pricing.get("output_per_1m_tokens", 0.0) / 1_000_000)
        )

    def track_usage(
        self,
        *,
        provider: str,
        model: str,
        stage: str,
        operation: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        parse_success: bool = True,
        retry_count: int = 0,
    ) -> float:
        """Track a normalized provider call and return the estimated cost."""
        billable_prompt_tokens = max(0, prompt_tokens - cached_tokens)
        total_tokens = billable_prompt_tokens + completion_tokens
        estimated_cost = self._calculate_call_cost(
            model=model,
            prompt_tokens=billable_prompt_tokens,
            completion_tokens=completion_tokens,
        )

        self.stats.total_prompt_tokens += billable_prompt_tokens
        self.stats.total_completion_tokens += completion_tokens
        self.stats.total_tokens += total_tokens
        self.stats.total_cached_tokens += cached_tokens
        self.stats.total_api_calls += 1
        if parse_success:
            self.stats.parseable_calls += 1
        else:
            self.stats.unparseable_calls += 1

        model_bucket = self.stats.model_usage.setdefault(
            model,
            {
                "provider": provider,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "call_count": 0,
                "parseable_calls": 0,
                "unparseable_calls": 0,
            },
        )
        model_bucket["prompt_tokens"] += billable_prompt_tokens
        model_bucket["completion_tokens"] += completion_tokens
        model_bucket["total_tokens"] += total_tokens
        model_bucket["cached_tokens"] += cached_tokens
        model_bucket["call_count"] += 1
        model_bucket["parseable_calls"] += 1 if parse_success else 0
        model_bucket["unparseable_calls"] += 0 if parse_success else 1

        stage_bucket = self.stats.stage_stats.setdefault(
            stage,
            {
                "provider": provider,
                "model": model,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "call_count": 0,
                "parseable_calls": 0,
                "unparseable_calls": 0,
            },
        )
        stage_bucket["prompt_tokens"] += billable_prompt_tokens
        stage_bucket["completion_tokens"] += completion_tokens
        stage_bucket["total_tokens"] += total_tokens
        stage_bucket["cached_tokens"] += cached_tokens
        stage_bucket["call_count"] += 1
        stage_bucket["parseable_calls"] += 1 if parse_success else 0
        stage_bucket["unparseable_calls"] += 0 if parse_success else 1

        self.stats.call_history.append(
            APICallRecord(
                timestamp=datetime.now().isoformat(),
                provider=provider,
                model=model,
                stage=stage,
                operation=operation,
                prompt_tokens=billable_prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens,
                parse_success=parse_success,
                retry_count=retry_count,
                estimated_cost=round(estimated_cost, 8),
            )
        )

        return estimated_cost

    def track_api_call(
        self,
        response: Any,
        stage: str,
        operation: str,
        *,
        model: str = "gemini-2.5-flash-lite",
        provider: str = "gemini",
        parse_success: bool = True,
        retry_count: int = 0,
    ) -> float:
        """Compatibility helper for older direct-Gemini call sites."""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
        completion_tokens = getattr(usage, "candidates_token_count", None) or 0
        cached_tokens = getattr(usage, "cached_content_token_count", None) or 0
        return self.track_usage(
            provider=provider,
            model=model,
            stage=stage,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            parse_success=parse_success,
            retry_count=retry_count,
        )

    def snapshot(self) -> Dict[str, Any]:
        """Return a lightweight snapshot for delta calculations."""
        return {
            "total_prompt_tokens": self.stats.total_prompt_tokens,
            "total_completion_tokens": self.stats.total_completion_tokens,
            "total_tokens": self.stats.total_tokens,
            "total_cached_tokens": self.stats.total_cached_tokens,
            "total_api_calls": self.stats.total_api_calls,
            "parseable_calls": self.stats.parseable_calls,
            "unparseable_calls": self.stats.unparseable_calls,
            "total_cost_usd": self.calculate_cost()["totals"]["total_cost"],
        }

    def delta(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Compute aggregate deltas since a previous snapshot."""
        current = self.snapshot()
        return {
            "llm_input_tokens": max(0, current["total_prompt_tokens"] - snapshot.get("total_prompt_tokens", 0)),
            "llm_output_tokens": max(0, current["total_completion_tokens"] - snapshot.get("total_completion_tokens", 0)),
            "llm_total_tokens": max(0, current["total_tokens"] - snapshot.get("total_tokens", 0)),
            "cached_tokens": max(0, current["total_cached_tokens"] - snapshot.get("total_cached_tokens", 0)),
            "api_calls": max(0, current["total_api_calls"] - snapshot.get("total_api_calls", 0)),
            "parseable_outputs": max(0, current["parseable_calls"] - snapshot.get("parseable_calls", 0)),
            "unparseable_outputs": max(0, current["unparseable_calls"] - snapshot.get("unparseable_calls", 0)),
            "cost_usd": round(max(0.0, current["total_cost_usd"] - snapshot.get("total_cost_usd", 0.0)), 8),
        }

    def calculate_cost(self) -> Dict[str, Any]:
        """Calculate cost breakdowns by model and overall totals."""
        by_model: Dict[str, Any] = {}
        total_input_cost = 0.0
        total_output_cost = 0.0

        for model, usage in self.stats.model_usage.items():
            pricing = self._get_model_pricing(model)
            input_cost = usage["prompt_tokens"] * (pricing.get("input_per_1m_tokens", 0.0) / 1_000_000)
            output_cost = usage["completion_tokens"] * (pricing.get("output_per_1m_tokens", 0.0) / 1_000_000)
            total_input_cost += input_cost
            total_output_cost += output_cost
            by_model[model] = {
                "provider": usage["provider"],
                "input_cost": round(input_cost, 8),
                "output_cost": round(output_cost, 8),
                "total_cost": round(input_cost + output_cost, 8),
                "input_tokens": usage["prompt_tokens"],
                "output_tokens": usage["completion_tokens"],
                "cached_tokens": usage["cached_tokens"],
            }

        return {
            "currency": PricingConfig.CURRENCY,
            "by_model": by_model,
            "totals": {
                "total_input_cost": round(total_input_cost, 8),
                "total_output_cost": round(total_output_cost, 8),
                "total_cost": round(total_input_cost + total_output_cost, 8),
            },
        }

    def get_report(self, detailed: bool = False) -> Dict[str, Any]:
        """Return a structured token report."""
        cost = self.calculate_cost()
        report = {
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total_api_calls": self.stats.total_api_calls,
                "total_prompt_tokens": self.stats.total_prompt_tokens,
                "total_completion_tokens": self.stats.total_completion_tokens,
                "total_tokens": self.stats.total_tokens,
                "total_cached_tokens": self.stats.total_cached_tokens,
                "parseable_calls": self.stats.parseable_calls,
                "unparseable_calls": self.stats.unparseable_calls,
                "parseable_output_rate_percent": round(
                    (self.stats.parseable_calls / self.stats.total_api_calls) * 100, 2
                )
                if self.stats.total_api_calls
                else 0.0,
            },
            "model_usage": self.stats.model_usage,
            "cost_estimation": cost,
            "pricing_reference": PricingConfig.MODEL_PRICING,
            "stage_breakdown": {},
        }

        for stage, stats in self.stats.stage_stats.items():
            stage_cost = self._calculate_call_cost(
                model=stats["model"],
                prompt_tokens=stats["prompt_tokens"],
                completion_tokens=stats["completion_tokens"],
            )
            report["stage_breakdown"][stage] = {
                "provider": stats["provider"],
                "model": stats["model"],
                "call_count": stats["call_count"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "total_tokens": stats["total_tokens"],
                "cached_tokens": stats["cached_tokens"],
                "parseable_calls": stats["parseable_calls"],
                "unparseable_calls": stats["unparseable_calls"],
                "estimated_cost": round(stage_cost, 8),
            }

        if detailed:
            report["call_history"] = [
                {
                    "timestamp": call.timestamp,
                    "provider": call.provider,
                    "model": call.model,
                    "stage": call.stage,
                    "operation": call.operation,
                    "prompt_tokens": call.prompt_tokens,
                    "completion_tokens": call.completion_tokens,
                    "total_tokens": call.total_tokens,
                    "cached_tokens": call.cached_tokens,
                    "parse_success": call.parse_success,
                    "retry_count": call.retry_count,
                    "estimated_cost": call.estimated_cost,
                }
                for call in self.stats.call_history
            ]

        return report

    def get_combined_metrics(self) -> Dict[str, Any]:
        """Return the compact metric structure consumed by the extension UI."""
        cost = self.calculate_cost()["totals"]
        by_stage = {}
        for stage, stats in self.stats.stage_stats.items():
            by_stage[stage] = {
                "provider": stats["provider"],
                "model": stats["model"],
                "tokens": stats["total_tokens"],
                "input_tokens": stats["prompt_tokens"],
                "output_tokens": stats["completion_tokens"],
                "cached_tokens": stats["cached_tokens"],
                "cost_usd": round(
                    self._calculate_call_cost(
                        model=stats["model"],
                        prompt_tokens=stats["prompt_tokens"],
                        completion_tokens=stats["completion_tokens"],
                    ),
                    8,
                ),
                "api_calls": stats["call_count"],
                "parseable_calls": stats["parseable_calls"],
                "unparseable_calls": stats["unparseable_calls"],
            }

        return {
            "total_tokens": self.stats.total_tokens,
            "total_input_tokens": self.stats.total_prompt_tokens,
            "total_output_tokens": self.stats.total_completion_tokens,
            "total_cached_tokens": self.stats.total_cached_tokens,
            "total_cost_usd": round(cost["total_cost"], 8),
            "api_calls": self.stats.total_api_calls,
            "parseable_output_rate_percent": round(
                (self.stats.parseable_calls / self.stats.total_api_calls) * 100, 2
            )
            if self.stats.total_api_calls
            else 0.0,
            "by_stage": by_stage,
        }

    def export_to_json(self, filepath: str, detailed: bool = True) -> None:
        report = self.get_report(detailed=detailed)
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Token usage report exported to: {filepath}")

    def print_summary(self) -> None:
        report = self.get_report(detailed=False)
        summary = report["summary"]
        print("\n" + "=" * 70)
        print("TOKEN USAGE & COST REPORT")
        print("=" * 70)
        print(f"  Total API Calls: {summary['total_api_calls']}")
        print(f"  Total Tokens: {summary['total_tokens']:,}")
        print(f"    Input: {summary['total_prompt_tokens']:,}")
        print(f"    Output: {summary['total_completion_tokens']:,}")
        print(f"    Cached: {summary['total_cached_tokens']:,}")
        print(f"  Parseable Output Rate: {summary['parseable_output_rate_percent']:.2f}%")
        for model, usage in report["model_usage"].items():
            cost = report["cost_estimation"]["by_model"].get(model, {})
            print(f"\n  {model} ({usage['provider']}):")
            print(f"    Calls: {usage['call_count']}")
            print(f"    Tokens: {usage['total_tokens']:,}")
            print(f"    Cost: ${cost.get('total_cost', 0.0):.6f}")
        print("=" * 70 + "\n")
