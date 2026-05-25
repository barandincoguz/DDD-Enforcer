"""Critic-stage error taxonomy."""
from typing import Optional


class CriticError(Exception):
    """Raised when the Critic LLM call is unrecoverable (json_failed after
    retries, or empty parse). Caught at the loop boundary and recorded;
    never silently swallowed."""

    def __init__(self, reason: str, cycle: Optional[int] = None):
        self.reason = reason
        self.cycle = cycle
        super().__init__(f"CriticError(cycle={cycle}): {reason}")
