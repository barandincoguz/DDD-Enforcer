"""Per-stage refinement prompts.

The Refiner takes the Verifier's issue list + the failing stage's
output + the failing stage's original prompt template and produces a
new prompt that asks the same stage to fix the cited issues.
"""

from typing import List
from core.verifier.types import VerifierIssue


REFINEMENT_PROMPT_TEMPLATE = """\
Your previous output for stage `{stage_name}` had the following issues:

{issue_list}

PREVIOUS OUTPUT:
{previous_output_json}

INSTRUCTIONS:
1. Address every issue listed above.
2. Preserve all correct parts of the previous output.
3. Re-emit the FULL stage output (not just the corrections).
4. If you cannot address an issue, leave that part of the output unchanged
   and add a `_unresolved` key to the affected element.

Respond with valid JSON matching the original stage's schema.
"""


def render_refinement_prompt(
    *, stage_name: str, previous_output_json: str, issues: List[VerifierIssue]
) -> str:
    # Uppercase severity matches the contract Literal ("ERROR" / "WARN")
    # surfaced to the LLM elsewhere; keeps the prompt consistent.
    issue_list = "\n".join(
        f"- [{i.severity.value.upper()}] {i.location} — {i.message}"
        + (f" (suggestion: {i.suggestion})" if i.suggestion else "")
        for i in issues
    )
    return REFINEMENT_PROMPT_TEMPLATE.format(
        stage_name=stage_name,
        issue_list=issue_list,
        previous_output_json=previous_output_json,
    )
