"""Bounded retry orchestration.

refine_until_clean takes a stage's output and runs the Verifier; if
issues exist, it asks the stage_runner to produce a corrected output
and re-verifies. Capped at max_cycles cycles; on exhaustion raises
RefinementExhaustedError carrying the residual issues.
"""

from typing import Any, Callable, Optional, Tuple
from core.verifier.types import VerifierResult
from core.orchestration.errors import RefinementExhaustedError


def refine_until_clean(
    *,
    stage_name: str,
    initial_output: Any,
    stage_runner: Callable[[Any, VerifierResult], Any],
    verifier: Callable[[Any], VerifierResult],
    max_cycles: int = 2,
    initial_result: Optional[VerifierResult] = None,
) -> Tuple[Any, int]:
    """Run verifier; if issues, call stage_runner with (output, result)
    to produce a corrected output; loop up to max_cycles.

    Returns (final_output, cycles_used). Raises RefinementExhaustedError
    when verifier still reports issues after max_cycles.

    WP-CORE-7: optional `initial_result` lets callers (run_pipeline) pre-verify
    once to dispatch architect-stage issues, then thread the result back in
    without re-verifying. Avoids double-verify on the common path. When
    `initial_result is None`, the verifier is called as the first step of
    the loop (legacy behavior).

    WP-CORE-29: `initial_result` is now properly `Optional[VerifierResult]`
    so the three `type: ignore[assignment]` directives in the pre-WP body
    are no longer needed.
    """
    output = initial_output
    cycles = 0
    result: Optional[VerifierResult] = initial_result
    while True:
        if result is None:
            result = verifier(output)
        if result.ok or result.error_count() == 0:
            return output, cycles
        if cycles >= max_cycles:
            # WP-CORE-24: carry cycles_attempted so the run manifest can
            # record refiner.metrics.cycles_used even on the exhausted path.
            raise RefinementExhaustedError(issues=result.issues, cycles_attempted=cycles)
        output = stage_runner(output, result)
        cycles += 1
        result = None  # force re-verify next iter
