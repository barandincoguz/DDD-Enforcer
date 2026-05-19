"""Synthesizer-specific exceptions."""


class SynthesizerInvariantError(Exception):
    """Raised when D6/D7/D8 invariants fail on the deterministic
    Synthesizer output. Indicates a code bug, not an LLM hiccup —
    deterministic code that violates an invariant means the code is
    broken. No retry path.
    """

    def __init__(self, *, check_id: str, message: str, details=None):
        self.check_id = check_id
        self.details = details or []
        super().__init__(f"[{check_id}] {message}")
