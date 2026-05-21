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


class SpecialistShapeError(SpecialistFailureError):
    """Specialist LLM emitted a JSON shape that fails Pydantic validation
    (e.g. top-level array instead of object, missing required fields).

    The retry loop converts the validation error into structured LLM
    feedback so the next attempt can correct the shape. Distinguished
    from SpecialistFailureError so callers can tell shape errors
    (recoverable via retry) from unrecoverable failures.
    """
    def __init__(self, *, context_name: str, errors: list, raw_excerpt: str = ""):
        self.context_name = context_name
        self.validation_errors = errors
        self.raw_excerpt = raw_excerpt
        message = (
            f"Specialist shape error for {context_name}: "
            f"{len(errors)} validation error(s); first: {errors[0] if errors else 'none'}"
        )
        super().__init__(context_name=context_name, message=message)


class IntermediateSaveError(PipelineError):
    """Raised when stage diagnostic JSON cannot be persisted to INTERMEDIATE_DIR.

    Per AGENTS.md "no silent degradation": stage diagnostic artifacts are
    EMSE reproducibility evidence; silent loss is a methodology gap.

    Carries:
        stage: the pipeline stage label (e.g., "2_architect", "3_specialist").
        filepath: the intended on-disk path.
        cause: the wrapped exception (OSError / TypeError / ValueError).
        srs_path: the SRS being processed (or "<unknown>" if pre-analyze_document).
    """

    def __init__(
        self,
        stage: str,
        filepath: str,
        cause: Exception,
        srs_path: str = "<unknown>",
    ):
        self.stage = stage
        self.filepath = filepath
        self.cause = cause
        self.srs_path = srs_path
        super().__init__(
            f"Failed to save intermediate output for stage='{stage}' "
            f"at '{filepath}' (srs={srs_path}): {type(cause).__name__}: {cause}"
        )
