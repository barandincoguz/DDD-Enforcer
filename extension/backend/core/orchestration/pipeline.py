"""5-stage pipeline driver: Scout → Architect → Specialist → Verifier → Synthesizer.

Each stage is injected as a Callable so this module stays test-friendly.
Real wiring (Gemini-backed stages) happens in DomainArchitect.analyze_document
in Task C7.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from core.schemas import DomainModel
from core.verifier.types import VerifierResult
from core.orchestration.errors import SynthesizerEmptyModelError
from core.refiner.loop import refine_until_clean


ScoutFn = Callable[[str], List[Dict]]
ArchitectFn = Callable[[List[Dict]], List[Dict]]
SpecialistFn = Callable[[List[Dict], List[Dict]], List[Dict]]
SynthesizerFn = Callable[[List[Dict]], Dict]
VerifierFn = Callable[[Dict], VerifierResult]


@dataclass
class PipelineDeps:
    scout: ScoutFn
    architect: ArchitectFn
    specialist: SpecialistFn
    synthesizer: SynthesizerFn
    verifier: VerifierFn


def run_pipeline(*, srs_text: str, deps: PipelineDeps) -> DomainModel:
    """Run the 5-stage pipeline. Raises PipelineError subclasses on
    failure; otherwise returns a validated DomainModel.
    """
    scout_chunks = deps.scout(srs_text)
    contexts = deps.architect(scout_chunks)
    specialist_outputs = deps.specialist(contexts, scout_chunks)

    # Build a combined snapshot for the Verifier.
    snapshot = {
        "scout": scout_chunks,
        "architect": contexts,
        "specialist": specialist_outputs,
    }

    def _re_run_specialist(_prev, _result):
        # Phase C ships a simple re-run; Phase D wires issue-aware re-prompting.
        return deps.specialist(contexts, scout_chunks)

    refined_specialist, cycles = refine_until_clean(
        stage_name="specialist",
        initial_output=specialist_outputs,
        stage_runner=_re_run_specialist,
        verifier=lambda s: deps.verifier({**snapshot, "specialist": s}),
        max_cycles=2,
    )

    raw_model = deps.synthesizer(refined_specialist)
    if not raw_model.get("bounded_contexts"):
        raise SynthesizerEmptyModelError(input_summary=f"{len(refined_specialist)} contexts")
    return DomainModel(**raw_model)
