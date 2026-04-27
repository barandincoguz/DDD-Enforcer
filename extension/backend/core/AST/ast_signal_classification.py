from __future__ import annotations

from typing import List, Optional, Tuple

from core.AST.ast_signal_types import CandidateSignal, ClassFacts, SourceRef
from core.AST.ast_signal_utils import clamp


SERVICE_SUFFIXES = ("Service", "Manager", "Processor", "Handler", "UseCase", "Provider")
INFRASTRUCTURE_SUFFIXES = (
    "Repository",
    "Factory",
    "Controller",
    "Helper",
    "Util",
    "Dto",
    "DTO",
    "Exception",
    "Config",
    "Settings",
)
EVENT_SUFFIXES = (
    "Event",
    "Created",
    "Placed",
    "Cancelled",
    "Paid",
    "Approved",
    "Rejected",
    "Completed",
    "Shipped",
)
IDENTITY_NAMES = {"id", "uuid", "guid", "code", "identifier"}
ACTION_VERBS = {"process", "place", "handle", "execute", "issue", "create", "sync", "send"}
TYPE_LABELS = {
    "entities": "entity",
    "value_objects": "value object",
    "services": "service",
    "aggregates": "aggregate",
}
TYPE_RULES = {
    "entities": "AST_ENTITY",
    "value_objects": "AST_VALUE_OBJECT",
    "services": "AST_SERVICE",
    "aggregates": "AST_AGGREGATE",
}


class SignalClassifier:
    def classify(self, facts: ClassFacts) -> List[CandidateSignal]:
        signals: List[CandidateSignal] = []
        for scorer in (
            self._score_service,
            self._score_value_object,
            self._score_aggregate,
            self._score_entity,
        ):
            candidate = scorer(facts)
            if candidate:
                signals.append(candidate)
        signals.extend(self._score_domain_events(facts))
        return signals

    def _score_service(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        score, reasons = self._base_score(facts)
        if any(facts.name.endswith(suffix) for suffix in SERVICE_SUFFIXES):
            score += 0.35
            reasons.append("service-style suffix")
        if facts.dependency_attributes or self._dependency_params(facts):
            score += 0.25
            reasons.append("injected collaborators")
        if facts.public_methods:
            score += 0.15
            reasons.append("orchestration methods")
        if len(facts.stateful_attributes) <= 1:
            score += 0.10
        if facts.method_tokens & ACTION_VERBS:
            score += 0.10
            reasons.append("action-oriented method names")
        if self._identity_attributes(facts):
            score -= 0.25
        if facts.is_dataclass and not facts.public_methods:
            score -= 0.25
        return self._build_candidate("services", facts, score, 0.60, reasons)

    def _score_value_object(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        score, reasons = self._base_score(facts)
        methods = facts.methods
        if facts.is_frozen or "ValueObject" in facts.bases:
            score += 0.45
            reasons.append("explicit immutable/value-object marker")
        if facts.is_dataclass:
            score += 0.35
            reasons.append("dataclass structure")
        if "__eq__" in methods or "__hash__" in methods:
            score += 0.15
            reasons.append("structural equality/hash")
        if len(facts.all_attributes) >= 2:
            score += 0.10
        if len(facts.public_methods) <= 2:
            score += 0.05
        if facts.mutation_methods or any(name.startswith("set") for name in methods):
            score -= 0.40
        if self._identity_attributes(facts):
            score -= 0.20
        return self._build_candidate(
            "value_objects",
            facts,
            score,
            0.58,
            reasons,
            attributes=sorted(facts.all_attributes),
        )

    def _score_aggregate(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        score, reasons = self._base_score(facts)
        if facts.name.endswith(("Aggregate", "Root")):
            score += 0.35
            reasons.append("aggregate naming")
        if facts.collection_attributes:
            score += 0.25
            reasons.append("owns collections")
        if any(name.startswith(("add_", "remove_", "update_")) for name in facts.methods):
            score += 0.20
            reasons.append("manages aggregate members")
        if self._identity_attributes(facts):
            score += 0.10
        if len(facts.public_methods) >= 2:
            score += 0.10
        return self._build_candidate("aggregates", facts, score, 0.62, reasons)

    def _score_entity(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        score, reasons = self._base_score(facts)
        identities = self._identity_attributes(facts)
        if identities:
            score += 0.45
            reasons.append("identity field")
        if len(facts.stateful_attributes) >= 2:
            score += 0.15
            reasons.append("stateful attributes")
        if facts.public_methods:
            score += 0.10
        if facts.mutation_methods:
            score += 0.15
            reasons.append("lifecycle/mutation behavior")
        if "Entity" in facts.bases or "AggregateRoot" in facts.bases:
            score += 0.15
        if facts.is_frozen:
            score -= 0.20
        if facts.dependency_attributes and not facts.stateful_attributes:
            score -= 0.20
        return self._build_candidate("entities", facts, score, 0.56, reasons)

    def _score_domain_events(self, facts: ClassFacts) -> List[CandidateSignal]:
        events = set(facts.event_names)
        if facts.name.endswith(EVENT_SUFFIXES):
            events.add(facts.name)

        signals: List[CandidateSignal] = []
        for event_name in sorted(events):
            reasons = ["event emission pattern"] if event_name in facts.event_names else ["event-style name"]
            source = SourceRef(
                file=facts.file_path,
                line=facts.line,
                rule="AST_DOMAIN_EVENT",
                evidence=event_name,
            )
            signals.append(
                CandidateSignal(
                    candidate_type="domain_events",
                    name=event_name,
                    description="AST-discovered domain event",
                    confidence=0.68,
                    reasons=reasons,
                    sources=[source],
                    file_path=facts.file_path,
                    module_tokens=set(facts.module_tokens),
                )
            )
        return signals

    def _base_score(self, facts: ClassFacts) -> Tuple[float, List[str]]:
        score = 0.10
        reasons: List[str] = []
        if not facts.name or not facts.name[0].isupper() or "_" in facts.name:
            score -= 0.35
        if any(facts.name.endswith(suffix) for suffix in INFRASTRUCTURE_SUFFIXES):
            score -= 0.50
            reasons.append("infrastructure-style suffix")
        if facts.module_tokens & {"controller", "config", "helper", "util", "migration"}:
            score -= 0.15
        return score, reasons

    def _build_candidate(
        self,
        candidate_type: str,
        facts: ClassFacts,
        score: float,
        threshold: float,
        reasons: List[str],
        attributes: Optional[List[str]] = None,
    ) -> Optional[CandidateSignal]:
        confidence = round(clamp(score), 2)
        if confidence < threshold:
            return None
        return CandidateSignal(
            candidate_type=candidate_type,
            name=facts.name,
            description=f"AST-discovered {TYPE_LABELS[candidate_type]}",
            confidence=confidence,
            reasons=list(dict.fromkeys(reasons)),
            sources=[
                SourceRef(
                    file=facts.file_path,
                    line=facts.line,
                    rule=TYPE_RULES[candidate_type],
                    evidence=f"class {facts.name}",
                )
            ],
            attributes=attributes or [],
            file_path=facts.file_path,
            module_tokens=set(facts.module_tokens),
        )

    def _identity_attributes(self, facts: ClassFacts) -> List[str]:
        identities = []
        for attr in facts.all_attributes | facts.constructor_params:
            lowered = attr.lower()
            if lowered in IDENTITY_NAMES or lowered.endswith("_id"):
                identities.append(attr)
        return identities

    def _dependency_params(self, facts: ClassFacts) -> List[str]:
        dependencies: List[str] = []
        for attr in facts.constructor_params:
            lowered = attr.lower()
            if lowered.endswith(
                (
                    "repo",
                    "repository",
                    "service",
                    "client",
                    "adapter",
                    "gateway",
                    "publisher",
                )
            ):
                dependencies.append(attr)
        return dependencies
