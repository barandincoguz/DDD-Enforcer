"""Typed exceptions for the P3 pipeline.

All silent fallbacks in core/architect.py are converted to raises of
these classes. The top-level orchestrator catches PipelineError, writes
a structured failure_log.json, and decides retry/skip/fail per RQ1
metrics policy.
"""

from typing import Any, List, Optional


class PipelineError(Exception):
    """Base for every P3 pipeline failure."""


class ScoutChunkParseError(PipelineError):
    def __init__(self, chunk_id: str, attempts: int, message: Optional[str] = None):
        self.chunk_id = chunk_id
        self.attempts = attempts
        super().__init__(message or f"Scout chunk {chunk_id} failed to parse after {attempts} attempts")


class ArchitectExtractionError(PipelineError):
    def __init__(self, srs_path: str, message: Optional[str] = None):
        self.srs_path = srs_path
        super().__init__(message or f"Architect produced zero bounded contexts for {srs_path}")


class SpecialistFailureError(PipelineError):
    def __init__(self, context_name: str, message: Optional[str] = None):
        self.context_name = context_name
        super().__init__(message or f"Specialist failed for context {context_name!r}")


class SynthesizerEmptyModelError(PipelineError):
    def __init__(self, input_summary: str, message: Optional[str] = None):
        self.input_summary = input_summary
        super().__init__(message or f"Synthesizer returned an empty DomainModel (input: {input_summary})")


class RefinementExhaustedError(PipelineError):
    def __init__(self, issues: List[Any], message: Optional[str] = None):
        self.issues = issues
        super().__init__(message or f"Refiner exhausted retries with {len(issues)} unresolved issues")


class InsufficientGroundingError(PipelineError):
    def __init__(self, entity_name: str, message: Optional[str] = None):
        self.entity_name = entity_name
        super().__init__(message or f"Entity {entity_name!r} has no SRS evidence_sentence_indices and no AST grounding")
