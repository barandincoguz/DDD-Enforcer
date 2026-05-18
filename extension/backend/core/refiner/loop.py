"""Bounded retry orchestration.

refine_until_clean takes a stage's output and runs the Verifier; if
issues exist, it asks the stage_runner to produce a corrected output
and re-verifies. Capped at max_cycles cycles; on exhaustion raises
RefinementExhaustedError carrying the residual issues.
"""

from typing import Any, Callable, Tuple
from core.verifier.types import VerifierResult
from core.orchestration.errors import RefinementExhaustedError


def refine_until_clean(
    *,
    stage_name: str,
    initial_output: Any,
    stage_runner: Callable[[Any, VerifierResult], Any],
    verifier: Callable[[Any], VerifierResult],
    max_cycles: int = 2,
) -> Tuple[Any, int]:
    """Run verifier; if issues, call stage_runner with (output, result)
    to produce a corrected output; loop up to max_cycles.

    Returns (final_output, cycles_used). Raises RefinementExhaustedError
    when verifier still reports issues after max_cycles.
    """
    output = initial_output
    cycles = 0
    while True:
        result = verifier(output)
        if result.ok or result.error_count() == 0:
            return output, cycles
        if cycles >= max_cycles:
            raise RefinementExhaustedError(issues=result.issues)
        output = stage_runner(output, result)
        cycles += 1
