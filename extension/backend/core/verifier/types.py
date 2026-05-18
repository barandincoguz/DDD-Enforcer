"""Verifier types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARN = "warn"


@dataclass(frozen=True)
class VerifierIssue:
    stage: Literal["scout", "architect", "specialist", "synthesizer"]
    location: str
    issue_type: str
    severity: IssueSeverity
    message: str
    suggestion: Optional[str] = None


@dataclass
class VerifierResult:
    ok: bool
    issues: List[VerifierIssue] = field(default_factory=list)

    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)
