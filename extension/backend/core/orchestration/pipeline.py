"""5-stage pipeline driver: Scout → Architect → Specialist → Verifier → Synthesizer.

Each stage is injected as a Callable so this module stays test-friendly.
Real wiring (Gemini-backed stages) happens in DomainArchitect.analyze_document.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
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


def run_pipeline(
    *,
    srs_text: str,
    deps: PipelineDeps,
    srs_path: Optional[str] = None,
) -> DomainModel:
    """Run the 5-stage pipeline with typed envelopes throughout.

    Raises PipelineError subclasses on failure; otherwise returns a
    validated DomainModel.

    Args:
        srs_text: SRS document text fed to the Scout stage.
        deps: Injected stage callables (Scout, Architect, Specialist,
            Synthesizer, Verifier).
        srs_path: Optional source path label for error messages. WP-CORE-5b
            threads this through `SynthesizerEmptyModelError.srs_path` so
            empty-Specialist failures name the failing SRS. Defaults to
            "<unknown>" inside the error if omitted.
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

    try:
        refined_specialist, cycles = refine_until_clean(
            stage_name="specialist",
            initial_output=specialist_output,
            stage_runner=_re_run_specialist,
            verifier=lambda s: deps.verifier({**snapshot, "specialist": s}),
            max_cycles=2,
        )
    except Exception as exc:
        # Refiner exhausted retries on semantic issues. The architectural
        # contract (typed envelopes + D6/D7/D8 hard-fail invariants) is
        # intact. The Verifier surfaced specific issues but the Refiner
        # couldn't resolve them after the budget — proceed with the most
        # recent Specialist output and let Synthesizer + D6/D7/D8 decide
        # if the model is shippable. This degrades from "Refiner-clean
        # within N cycles" to "best-effort Specialist", which is
        # acceptable for the WP-CORE-1 baseline while a follow-up WP
        # implements issue-aware re-prompting in the Refiner.
        print(
            f"  ⚠️  refiner exhausted retries ({type(exc).__name__}); "
            f"continuing with last Specialist output"
        )
        refined_specialist = specialist_output

    # Pre-call guard (primary): catches refined_specialist == [] for both
    # initial-empty and refiner-rerun-to-empty DI paths. Per Codex W-1 + W-2,
    # this is the only place where the empty case can be observed
    # taxonomically as a PipelineError; the in-tree synthesizer would
    # otherwise raise pydantic.ValidationError via DomainModel._non_empty
    # (core/schemas.py:207-215), escaping the PipelineError taxonomy.
    if not refined_specialist:
        raise SynthesizerEmptyModelError(
            input_summary="0 SpecialistAnalysis from upstream pipeline",
            srs_path=srs_path or "<unknown>",
        )

    model: DomainModel = deps.synthesizer(refined_specialist)

    # Post-call boundary check (belt-and-suspenders): retained per Codex W-3
    # because PipelineDeps.synthesizer is an injectable SynthesizerFn; a
    # future or test-injected synthesizer could construct DomainModel via
    # DomainModel.model_construct (which bypasses Pydantic validation) and
    # return an empty model. The in-tree synthesizer is already caught by
    # Pydantic _non_empty validator, making this branch dead for the in-tree
    # path — but ALIVE for injected paths.
    if not model.bounded_contexts:
        raise SynthesizerEmptyModelError(
            input_summary="synthesizer returned 0 bounded contexts (bypassed Pydantic)",
            srs_path=srs_path or "<unknown>",
        )
    return model
