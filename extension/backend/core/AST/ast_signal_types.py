from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Set


CandidateType = Literal[
    "entities",
    "value_objects",
    "services",
    "aggregates",
    "domain_events",
]


@dataclass
class SourceRef:
    file: str
    line: int
    rule: str
    evidence: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "rule": self.rule,
            "evidence": self.evidence,
        }


@dataclass
class GroundingMatch:
    file: str
    line: int
    snippet: str
    score: float
    rule: str

    def to_source(self) -> SourceRef:
        return SourceRef(
            file=self.file,
            line=self.line,
            rule=self.rule,
            evidence=self.snippet[:140],
        )


@dataclass
class ClassFacts:
    name: str
    file_path: str
    line: int
    bases: Set[str] = field(default_factory=set)
    decorators: Set[str] = field(default_factory=set)
    class_attributes: Set[str] = field(default_factory=set)
    instance_attributes: Set[str] = field(default_factory=set)
    constructor_params: Set[str] = field(default_factory=set)
    methods: Set[str] = field(default_factory=set)
    public_methods: Set[str] = field(default_factory=set)
    mutation_methods: Set[str] = field(default_factory=set)
    collection_attributes: Set[str] = field(default_factory=set)
    dependency_attributes: Set[str] = field(default_factory=set)
    event_names: Set[str] = field(default_factory=set)
    module_tokens: Set[str] = field(default_factory=set)
    name_tokens: Set[str] = field(default_factory=set)
    method_tokens: Set[str] = field(default_factory=set)
    is_dataclass: bool = False
    is_frozen: bool = False

    @property
    def all_attributes(self) -> Set[str]:
        return set(self.class_attributes) | set(self.instance_attributes)

    @property
    def stateful_attributes(self) -> Set[str]:
        return self.all_attributes - self.dependency_attributes


@dataclass
class CandidateSignal:
    candidate_type: CandidateType
    name: str
    description: str
    confidence: float
    reasons: List[str] = field(default_factory=list)
    sources: List[SourceRef] = field(default_factory=list)
    grounded_matches: List[GroundingMatch] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    file_path: str = ""
    module_tokens: Set[str] = field(default_factory=set)

    def to_public_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "confidence": round(self.confidence, 2),
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.candidate_type == "entities":
            payload["synonyms_to_avoid"] = []
            # Entity.justification is required (Phase A schema tightening).
            # AST signals don't ship a justification field, so synthesize one
            # from the classifier reasons that produced this candidate.
            payload["justification"] = (
                "; ".join(self.reasons[:3]) if self.reasons else "AST-detected entity"
            )
        if self.candidate_type == "aggregates":
            # Aggregate.members is required (Phase A schema tightening).
            # AST classifier doesn't yet populate aggregate members; emit empty
            # list so DomainModel validation passes. Phase D1 may surface this.
            payload["members"] = []
        if self.candidate_type == "value_objects":
            payload["attributes"] = list(self.attributes)
        return payload


@dataclass
class ContextMatch:
    index: int
    score: float
    reasons: List[str] = field(default_factory=list)
