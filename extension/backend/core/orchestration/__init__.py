"""Pipeline orchestration package: errors, 5-stage driver, and the
Verifier+Refiner loop wiring.
"""

from core.orchestration.errors import (
    PipelineError,
    ScoutChunkParseError,
    ArchitectExtractionError,
    SpecialistFailureError,
    SynthesizerEmptyModelError,
    RefinementExhaustedError,
    InsufficientGroundingError,
)

__all__ = [
    "PipelineError",
    "ScoutChunkParseError",
    "ArchitectExtractionError",
    "SpecialistFailureError",
    "SynthesizerEmptyModelError",
    "RefinementExhaustedError",
    "InsufficientGroundingError",
]
