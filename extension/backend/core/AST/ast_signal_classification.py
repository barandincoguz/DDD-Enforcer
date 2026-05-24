from __future__ import annotations

from typing import List, Optional, Tuple

from core.AST.ast_signal_types import CandidateSignal, CandidateType, ClassFacts, SourceRef
from core.AST.ast_signal_utils import clamp


SERVICE_SUFFIXES = ("Service", "Manager", "Processor", "Handler", "UseCase", "Provider")
# WP-CORE-22: Repository + Factory removed from this list. They had a -0.50
# base-score penalty that made them impossible to detect; now they are
# first-class candidates with dedicated scorers (`_score_repository`,
# `_score_factory`). The remaining suffixes are genuine infrastructure
# concerns (controllers, DTOs, exceptions, configs) that should not be
# classified as domain building blocks.
INFRASTRUCTURE_SUFFIXES = (
    "Controller",
    "Helper",
    "Util",
    "Dto",
    "DTO",
    "Exception",
    "Config",
    "Settings",
)
REPOSITORY_SUFFIXES = ("Repository", "Repo")
FACTORY_SUFFIXES = ("Factory", "Builder")
REPOSITORY_METHOD_PREFIXES = ("find_by_", "get_by_", "find_one", "find_all")
REPOSITORY_METHOD_NAMES = {"save", "delete", "add", "remove", "find", "get", "exists"}
FACTORY_METHOD_PREFIXES = ("create_", "build_", "make_", "from_", "build")
# WP-CORE-33 (V7): Anti-Corruption Layer suffixes. Kept distinct from
# SERVICE_SUFFIXES so the ACL scorer can claim *Translator / *Adapter /
# *Mapper / *Gateway / *ACL / *AntiCorruption before _score_service runs.
ACL_SUFFIXES = (
    "Translator",
    "Adapter",
    "Mapper",
    "ACL",
    "AntiCorruption",
    "Gateway",
)
# Method-shape signals that ACL classes use to translate between
# external-system models and the bounded context's domain model.
ACL_METHOD_PREFIXES = (
    "translate_",
    "convert_",
    "adapt_",
    "to_domain",
    "from_external",
    "from_dto",
    "to_dto",
)
ACL_METHOD_NAMES = {"translate", "convert", "adapt", "to_domain", "from_external"}
# WP-CORE-33 (V8): Specification suffixes — predicate-as-object classes.
SPECIFICATION_SUFFIXES = ("Specification", "Spec", "Rule", "Predicate")
SPECIFICATION_METHOD_NAMES = {
    "is_satisfied_by",
    "and_",
    "or_",
    "not_",
    "evaluate",
    "matches",
}
# WP-CORE-33 (V9): dependency suffixes that mark a Service as
# *application*-tier (repository injection) vs *infrastructure*-tier
# (external clients / gateways / sdks only). Repository markers win
# whenever both are present.
_REPO_DEP_SUFFIXES = ("repo", "repository")
_INFRA_DEP_SUFFIXES = ("client", "gateway", "adapter", "sdk", "publisher")
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
    "repositories": "repository",
    "factories": "factory",
    # WP-CORE-33
    "anti_corruption_layers": "anti-corruption layer",
    "specifications": "specification",
}
TYPE_RULES = {
    "entities": "AST_ENTITY",
    "value_objects": "AST_VALUE_OBJECT",
    "services": "AST_SERVICE",
    "aggregates": "AST_AGGREGATE",
    "repositories": "AST_REPOSITORY",
    "factories": "AST_FACTORY",
    # WP-CORE-33
    "anti_corruption_layers": "AST_ACL",
    "specifications": "AST_SPECIFICATION",
}


class SignalClassifier:
    def classify(self, facts: ClassFacts) -> List[CandidateSignal]:
        signals: List[CandidateSignal] = []
        # WP-CORE-22: Repository + Factory run BEFORE _score_service so
        # *Repository / *Factory class names land in their dedicated
        # buckets instead of being silently dropped by base_score's
        # legacy penalty.
        # WP-CORE-33: ACL + Specification scorers slot in here too —
        # *Translator / *Adapter / *Specification class names are
        # neither Services nor ValueObjects in the DDD pattern catalog.
        for scorer in (
            self._score_repository,
            self._score_factory,
            self._score_acl,
            self._score_specification,
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

    def _score_repository(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        """WP-CORE-22: Repository pattern detector.

        Repositories are persistence-facing classes that mediate access to an
        Aggregate root. Naming convention is the strongest signal; method
        shape (find_by_*, save, delete) and explicit Repository base classes
        reinforce. Identity field on the repository itself is a strong
        negative — that would indicate Entity, not Repository.
        """
        score, reasons = self._base_score(facts)
        has_repo_suffix = any(facts.name.endswith(s) for s in REPOSITORY_SUFFIXES)
        if has_repo_suffix:
            score += 0.40
            reasons.append("repository-style suffix")
        if "Repository" in facts.bases or "AbstractRepository" in facts.bases:
            score += 0.30
            reasons.append("explicit Repository base class")
        repo_methods = [
            name for name in facts.methods
            if name.startswith(REPOSITORY_METHOD_PREFIXES)
            or name in REPOSITORY_METHOD_NAMES
        ]
        if repo_methods:
            score += min(0.25, 0.10 + 0.05 * len(repo_methods))
            reasons.append("CRUD/query methods")
        if self._identity_attributes(facts):
            # Repository itself shouldn't carry domain identity.
            score -= 0.20
        if facts.is_dataclass or facts.is_frozen:
            # Repositories aren't value objects.
            score -= 0.30
        return self._build_candidate("repositories", facts, score, 0.60, reasons)

    def _score_factory(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        """WP-CORE-22: Factory pattern detector.

        Factories create domain objects. Suffix (Factory/Builder) is the
        strongest signal; presence of create_*/build_*/make_*/from_* methods
        reinforces. Factories are typically stateless or lightly stateful;
        heavy DI implies an Application Service, not a Factory.
        """
        score, reasons = self._base_score(facts)
        has_factory_suffix = any(facts.name.endswith(s) for s in FACTORY_SUFFIXES)
        if has_factory_suffix:
            score += 0.40
            reasons.append("factory-style suffix")
        factory_methods = [
            name for name in facts.methods
            if name.startswith(FACTORY_METHOD_PREFIXES)
        ]
        if factory_methods:
            score += min(0.30, 0.15 + 0.05 * len(factory_methods))
            reasons.append("creation methods")
        if len(facts.stateful_attributes) <= 1:
            score += 0.10
            reasons.append("stateless or near-stateless")
        if self._identity_attributes(facts):
            # Factories don't carry domain identity.
            score -= 0.30
        # Heavy DI shifts the candidate toward Application Service.
        if len(facts.dependency_attributes) >= 2:
            score -= 0.15
        return self._build_candidate("factories", facts, score, 0.62, reasons)

    def _score_acl(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        """WP-CORE-33 (V7): Anti-Corruption Layer detector.

        ACLs sit at the seam between an internal bounded context and an
        external system, translating one model into the other.  The
        class-name suffix is the strongest signal; presence of
        translate_*, convert_*, to_domain, from_external methods
        reinforces.  An external collaborator (typical of ACLs) is a
        moderate signal but not required — pure translator classes can
        be implemented as static-style adapters with no DI.
        """
        score, reasons = self._base_score(facts)
        has_acl_suffix = any(facts.name.endswith(s) for s in ACL_SUFFIXES)
        if has_acl_suffix:
            score += 0.40
            reasons.append("ACL-style suffix")
        translate_methods = [
            name for name in facts.methods
            if name.startswith(ACL_METHOD_PREFIXES) or name in ACL_METHOD_NAMES
        ]
        if translate_methods:
            score += min(0.30, 0.15 + 0.05 * len(translate_methods))
            reasons.append("translation/adaptation methods")
        if facts.dependency_attributes:
            # ACLs typically hold a reference to the external collaborator.
            score += 0.15
            reasons.append("external collaborator")
        if self._identity_attributes(facts):
            # ACLs do not carry domain identity.
            score -= 0.20
        if facts.is_frozen:
            # ACLs are translation gateways, not value objects.
            score -= 0.20
        return self._build_candidate(
            "anti_corruption_layers", facts, score, 0.62, reasons,
        )

    def _score_specification(self, facts: ClassFacts) -> Optional[CandidateSignal]:
        """WP-CORE-33 (V8): Specification-pattern detector.

        Specifications encapsulate a predicate as an object.  The canonical
        `is_satisfied_by(...)` method is the highest-value signal; suffix
        (Specification / Spec / Rule / Predicate) reinforces.  Combinator
        helpers (`and_`, `or_`, `not_`) add small bonuses.  Heavy DI
        demotes — a class with several repository-like deps is an
        application service borrowing the *Spec* suffix.
        """
        score, reasons = self._base_score(facts)
        has_spec_suffix = any(facts.name.endswith(s) for s in SPECIFICATION_SUFFIXES)
        if has_spec_suffix:
            score += 0.40
            reasons.append("specification-style suffix")
        if "is_satisfied_by" in facts.methods:
            score += 0.30
            reasons.append("predicate-as-object method (is_satisfied_by)")
        combinators = {"and_", "or_", "not_"} & facts.methods
        if combinators:
            score += min(0.15, 0.05 * len(combinators))
            reasons.append("specification combinators")
        if facts.is_frozen or facts.is_dataclass:
            # Many specifications are immutable. Small bonus only.
            score += 0.05
        if self._identity_attributes(facts):
            score -= 0.20
        # Heavy DI shifts the candidate toward Application Service.
        if len(facts.dependency_attributes) >= 2:
            score -= 0.30
            reasons.append("multiple injected deps demote to service")
        return self._build_candidate(
            "specifications", facts, score, 0.62, reasons,
        )

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
        candidate = self._build_candidate("services", facts, score, 0.60, reasons)
        # WP-CORE-33 (V9): stamp the kind discriminator on the service
        # candidate before returning so downstream consumers can split
        # domain / application / infrastructure tiers without re-doing
        # the dependency analysis.
        if candidate is not None:
            candidate.service_kind = self._classify_service_kind(facts)
        return candidate

    def _classify_service_kind(self, facts: ClassFacts) -> str:
        """WP-CORE-33 (V9): deterministic Service-tier discriminator.

        * No injected deps -> "domain" (pure logic).
        * Any repository-style dep -> "application" (orchestrates a use
          case that crosses the persistence boundary).
        * Only external-client / gateway / adapter / sdk / publisher
          deps -> "infrastructure" (wrapper around an external system).
        * Mixed / unclassifiable deps default to "application" — the
          common case for orchestration code.
        """
        deps = set(facts.dependency_attributes) | set(self._dependency_params(facts))
        if not deps:
            return "domain"
        lowered = {d.lower() for d in deps}
        has_repo = any(name.endswith(_REPO_DEP_SUFFIXES) for name in lowered)
        if has_repo:
            return "application"
        if lowered and all(
            any(name.endswith(suffix) for suffix in _INFRA_DEP_SUFFIXES)
            for name in lowered
        ):
            return "infrastructure"
        return "application"

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
        candidate_type: CandidateType,
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
