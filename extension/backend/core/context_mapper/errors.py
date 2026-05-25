"""Context-Mapper errors."""


class ContextMapperError(Exception):
    """Raised when the mapping LLM call fails (json_failed) after retries."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(f"context-mapper failed: {self.reason}")
