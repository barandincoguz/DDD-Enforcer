"""S1: LLM-based semantic grounding spot-check.

For each entity with cited evidence_sentence_indices, ask a small LLM
judge: "does the entity name (or a close synonym) appear in the cited
sentences?" Issues are emitted when the judge says no.

Phase C3 ships this as a Protocol-based dependency so unit tests can
inject a MagicMock judge. Phase C6 wires a real Gemini call.
"""

from typing import Dict, List, Protocol, Sequence
from core.verifier.types import IssueSeverity, VerifierIssue


class LLMJudge(Protocol):
    def judge(self, *, entity_name: str, sentences: Sequence[str]) -> Dict:
        """Return {"grounded": bool, "reason": str}."""
        ...


def check_s1_entity_grounded_in_evidence(
    entity: Dict,
    scout_sentences: Dict[int, str],
    llm_judge: LLMJudge,
) -> List[VerifierIssue]:
    """Ask the LLM judge whether the entity actually appears in its
    cited evidence sentences. Returns [] when grounded or no indices
    are claimed; one ERROR issue when the judge rules ungrounded.
    """
    indices = entity.get("evidence_sentence_indices") or []
    if not indices:
        return []
    cited = [scout_sentences[i] for i in indices if i in scout_sentences]
    if not cited:
        return [VerifierIssue(
            stage="specialist",
            location=f"specialist:entities({entity.get('name')})",
            issue_type="evidence_indices_out_of_range",
            severity=IssueSeverity.ERROR,
            message=(
                f"Entity {entity.get('name')!r} cites sentence ids "
                f"{indices} not present in scout_sentences"
            ),
        )]
    verdict = llm_judge.judge(entity_name=entity["name"], sentences=cited)
    if not verdict.get("grounded", False):
        return [VerifierIssue(
            stage="specialist",
            location=f"specialist:entities({entity.get('name')})",
            issue_type="semantic_ungrounded",
            severity=IssueSeverity.ERROR,
            message=(
                f"LLM judge: entity {entity.get('name')!r} not grounded in "
                f"cited sentences ({verdict.get('reason')})"
            ),
            suggestion="Either cite different sentences or drop this entity",
        )]
    return []
