"""Token Usage Tracker

Records per-call token usage and computes USD cost via the model registry
(`configs/models.py`). Reporting and presentation live in `token_tracker_report.py`;
data classes live in `token_tracker_types.py`.
"""

import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add backend to path so `configs` is importable when this module is loaded
# from anywhere under extension/backend.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from configs.models import ModelInfo, model_for_stage, model_info  # noqa: E402

# Re-export the data classes from token_tracker_types so existing
# `from core.token_tracker import TokenUsageStats` continues to work.
from core.token_tracker_types import (  # noqa: E402,F401
    APICallRecord,
    ModelTokenAccumulator,
    StageTokenAccumulator,
    TokenUsageStats,
)


class TokenTracker:
    """Singleton token-usage tracker. Reads pricing from the model registry.

    Reporting/presentation methods (`get_report`, `print_summary`, etc.) are
    thin delegates to functions in `core.token_tracker_report`.
    """

    _instance: Optional["TokenTracker"] = None

    def __init__(self) -> None:
        self.stats = TokenUsageStats()
        self.session_start = datetime.now().isoformat()
        self._lock = threading.Lock()

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

        # Lock-protected mutations — counters and dict updates atomic across threads.
        with self._lock:
            self.stats.total_prompt_tokens += billable_prompt
            self.stats.total_completion_tokens += completion_tokens
            self.stats.total_tokens += billable_total
            self.stats.total_api_calls += 1

            accum_m = self.stats.by_model.setdefault(
                info.model_id,
                ModelTokenAccumulator(model_id=info.model_id, provider=info.provider),
            )
            accum_m.prompt_tokens += billable_prompt
            accum_m.completion_tokens += completion_tokens
            accum_m.cost_usd += cost
            accum_m.call_count += 1

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

    # ---- public: report delegates ----------------------------------------
    # These thin wrappers preserve the existing public API. Implementation
    # lives in core.token_tracker_report to keep this file focused.

    def calculate_cost(self) -> Dict[str, object]:
        from core.token_tracker_report import compute_cost
        return compute_cost(self.stats)

    def get_report(self, detailed: bool = False) -> Dict:
        from core.token_tracker_report import build_report
        return build_report(self.stats, self.session_start, detailed=detailed)

    def get_combined_metrics(self) -> Dict:
        from core.token_tracker_report import combined_metrics
        return combined_metrics(self.stats)

    def print_summary(self) -> None:
        from core.token_tracker_report import print_summary
        print_summary(self.stats)

    def export_to_json(self, filepath: str, detailed: bool = True) -> None:
        from core.token_tracker_report import export_to_json
        export_to_json(self.stats, self.session_start, filepath, detailed=detailed)
