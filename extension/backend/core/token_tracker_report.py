"""Token-tracker reporting & presentation.

Pure functions over `TokenUsageStats` — they compute reports, format console
summaries, and serialize JSON, but they do not mutate state or know about the
singleton tracker. The `TokenTracker` class delegates to these functions.

Lives separately from `token_tracker.py` so each file stays focused (AGENTS.md
≤ 300-line guidance).
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

# Ensure configs is importable when this module is loaded from anywhere under
# extension/backend.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from configs.models import model_info  # noqa: E402
from core.token_tracker_types import TokenUsageStats  # noqa: E402


def compute_cost(stats: TokenUsageStats) -> Dict[str, object]:
    """Return per-model cost breakdown plus totals. (Was TokenTracker.calculate_cost.)"""
    by_model: Dict[str, Dict[str, float]] = {}
    total_cost = 0.0
    total_input = 0.0
    total_output = 0.0

    for accum in stats.by_model.values():
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


def build_report(stats: TokenUsageStats, session_start: str, detailed: bool = False) -> Dict:
    """Generate comprehensive report. (Was TokenTracker.get_report.)"""
    cost = compute_cost(stats)

    model_usage: Dict[str, Dict] = {}
    for accum in stats.by_model.values():
        stages_for_model = sorted({
            s.stage for s in stats.by_stage.values() if s.model_id == accum.model_id
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
    for accum in stats.by_stage.values():
        stage_breakdown[accum.stage] = {
            "model_id": accum.model_id,
            "call_count": accum.call_count,
            "prompt_tokens": accum.prompt_tokens,
            "completion_tokens": accum.completion_tokens,
            "total_tokens": accum.prompt_tokens + accum.completion_tokens,
            "estimated_cost": round(accum.cost_usd, 6),
        }

    report: Dict[str, object] = {
        "session_start": session_start,
        "session_end": datetime.now().isoformat(),
        "summary": {
            "total_api_calls": stats.total_api_calls,
            "total_prompt_tokens": stats.total_prompt_tokens,
            "total_completion_tokens": stats.total_completion_tokens,
            "total_tokens": stats.total_tokens,
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
            for c in stats.call_history
        ]

    return report


def combined_metrics(stats: TokenUsageStats) -> Dict:
    """Simplified metrics for API responses. (Was TokenTracker.get_combined_metrics.)"""
    cost = compute_cost(stats)
    by_stage: Dict[str, Dict] = {}
    for accum in stats.by_stage.values():
        by_stage[accum.stage] = {
            "tokens": accum.prompt_tokens + accum.completion_tokens,
            "input_tokens": accum.prompt_tokens,
            "output_tokens": accum.completion_tokens,
            "cost_usd": round(accum.cost_usd, 6),
            "api_calls": accum.call_count,
            "model_id": accum.model_id,
        }
    return {
        "total_tokens": stats.total_tokens,
        "total_input_tokens": stats.total_prompt_tokens,
        "total_output_tokens": stats.total_completion_tokens,
        "total_cost_usd": cost["total_cost"],
        "api_calls": stats.total_api_calls,
        "by_stage": by_stage,
    }


def print_summary(stats: TokenUsageStats) -> None:
    """Console summary. (Was TokenTracker.print_summary.)"""
    cost = compute_cost(stats)
    print("\n" + "=" * 70)
    print("📊 TOKEN USAGE & COST REPORT")
    print("=" * 70)
    print(f"  Total API Calls:        {stats.total_api_calls}")
    print(f"  Total Tokens:           {stats.total_tokens:,}")
    print(f"    ↳ Input:              {stats.total_prompt_tokens:,}")
    print(f"    ↳ Output:             {stats.total_completion_tokens:,}")

    if stats.by_model:
        print("\n" + "-" * 70)
        print("🤖 MODEL BREAKDOWN")
        print("-" * 70)
        for accum in stats.by_model.values():
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

    if stats.by_stage:
        print("\n" + "-" * 70)
        print("📈 STAGE BREAKDOWN")
        print("-" * 70)
        for accum in stats.by_stage.values():
            print(f"\n  {accum.stage} ({accum.model_id}):")
            print(f"    Calls:  {accum.call_count}")
            print(f"    Tokens: {accum.prompt_tokens + accum.completion_tokens:,}")
            print(f"    Cost:   ${accum.cost_usd:.6f}")

    print("=" * 70 + "\n")


def export_to_json(
    stats: TokenUsageStats, session_start: str, filepath: str, detailed: bool = True
) -> None:
    """Write the full report to a JSON file. (Was TokenTracker.export_to_json.)"""
    report = build_report(stats, session_start, detailed=detailed)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)
    print(f"📄 Token usage report exported to: {filepath}")
