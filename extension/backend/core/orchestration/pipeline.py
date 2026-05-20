"""5-stage pipeline driver: Scout → Architect → Specialist → Verifier → Synthesizer.

Each stage is injected as a Callable so this module stays test-friendly.
Real wiring (Gemini-backed stages) happens in DomainArchitect.analyze_document.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from core.schemas import DomainModel
from core.pipeline_contracts import (
    ScoutOutput,
    ArchitectOutput,
    SpecialistAnalysis,
    VerifierResult,
)
from core.orchestration.errors import SynthesizerEmptyModelError
from core.refiner.loop import refine_until_clean


ScoutFn = Callable[[str], ScoutOutput]
ArchitectFn = Callable[[ScoutOutput], ArchitectOutput]
SpecialistFn = Callable[[ArchitectOutput, ScoutOutput], List[SpecialistAnalysis]]
SynthesizerFn = Callable[[List[SpecialistAnalysis]], DomainModel]
VerifierFn = Callable[[Dict[str, Any]], VerifierResult]


@dataclass
class PipelineDeps:
    scout: ScoutFn
    architect: ArchitectFn
    specialist: SpecialistFn
    synthesizer: SynthesizerFn
    verifier: VerifierFn


def run_pipeline(*, srs_text: str, deps: PipelineDeps) -> DomainModel:
    """Run the 5-stage pipeline with typed envelopes throughout.

    Raises PipelineError subclasses on failure; otherwise returns a
    validated DomainModel.
    """
    scout: ScoutOutput = deps.scout(srs_text)
    arch: ArchitectOutput = deps.architect(scout)
    specialist_output: List[SpecialistAnalysis] = deps.specialist(arch, scout)

    # Build a combined snapshot for the Verifier.
    snapshot: Dict[str, Any] = {
        "scout": scout,
        "architect": arch,
        "specialist": specialist_output,
    }

    def _re_run_specialist(_prev, _result) -> List[SpecialistAnalysis]:
        # Phase C ships a simple re-run; Phase D wires issue-aware re-prompting.
        return deps.specialist(arch, scout)

    refined_specialist, cycles = refine_until_clean(
        stage_name="specialist",
        initial_output=specialist_output,
        stage_runner=_re_run_specialist,
        verifier=lambda s: deps.verifier({**snapshot, "specialist": s}),
        max_cycles=2,
    )

    model: DomainModel = deps.synthesizer(refined_specialist)
    if not model.bounded_contexts:
        raise SynthesizerEmptyModelError(input_summary=f"{len(refined_specialist)} contexts")
    return model
