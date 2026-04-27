"""Token-tracker data classes.

These dataclasses are the shared data model used by both `token_tracker.py`
(stateful tracking) and `token_tracker_report.py` (reporting/presentation).
Lives in its own module to break what would otherwise be a circular import.
"""

from dataclasses import dataclass, field
from typing import Dict, List


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
