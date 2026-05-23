from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from core.AST.ast_signal_grounding import rank_groundings
from core.AST.ast_signal_types import CandidateSignal, ContextMatch
from core.AST.ast_signal_utils import clamp, normalized_terms
from core.orchestration.errors import InsufficientGroundingError


INTERMEDIATE_DIR = Path(__file__).resolve().parent / "intermediate"
MERGE_THRESHOLD = 0.55
AST_ONLY_MERGE_THRESHOLD = 0.62
CONTEXT_THRESHOLD = 2.25
CONTEXT_MARGIN = 0.75
# WP-CORE-22: repositories + factories added as first-class buckets.
# WP-CORE-33: anti_corruption_layers + specifications added.
CANDIDATE_TYPES = (
    "entities",
    "value_objects",
    "services",
    "aggregates",
    "domain_events",
    "repositories",
    "factories",
    "anti_corruption_layers",
    "specifications",
)


class SignalEnricher:
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.diagnostics = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workspace_path": workspace_path,
            "counts": {
                key: {"detected": 0, "grounded": 0, "merged": 0, "dropped": 0, "unassigned": 0}
                for key in CANDIDATE_TYPES
            },
            "samples": {"dropped": [], "unassigned": []},
        }

    def deduplicate_signals(self, signals: List[CandidateSignal]) -> List[CandidateSignal]:
        deduped: Dict[tuple[str, str], CandidateSignal] = {}
        for signal in signals:
            key = (signal.candidate_type, signal.name.lower())
            self.diagnostics["counts"][signal.candidate_type]["detected"] += 1
            existing = deduped.get(key)
            if not existing:
                deduped[key] = signal
                continue
            existing.confidence = max(existing.confidence, signal.confidence)
            existing.reasons = list(dict.fromkeys(existing.reasons + signal.reasons))
            existing.sources.extend(signal.sources)
            existing.attributes = sorted(set(existing.attributes) | set(signal.attributes))
            existing.module_tokens |= signal.module_tokens
        return list(deduped.values())

    def apply_grounding(self, signals: List[CandidateSignal], docs: Optional[List[Dict[str, Any]]]) -> None:
        if not docs:
            return
        for signal in signals:
            matches = rank_groundings(signal.name, docs, limit=3)
            if not matches:
                continue
            signal.grounded_matches = matches
            signal.sources.extend(match.to_source() for match in matches)
            signal.confidence = round(clamp(signal.confidence + min(0.18, 0.06 * len(matches))), 2)
            signal.reasons.append("grounded in SRS")
            self.diagnostics["counts"][signal.candidate_type]["grounded"] += 1

    def enrich_model(
        self,
        model_data: Dict[str, Any],
        signals: List[CandidateSignal],
        srs_docs: Optional[List[Dict[str, Any]]] = None,
        mutability_index: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        contexts = model_data.get("bounded_contexts", [])
        # A model is treated as "auto-created" when there are no contexts at
        # all, OR when the only context is a blank placeholder (no entities,
        # services, aggregates, value_objects, or domain_events). Phase A's
        # schema tightening forbids the previous "empty bounded_contexts"
        # shortcut, so callers seed a blank CoreDomain context instead; we
        # honor that as equivalent to "auto_created".
        auto_created_context = not bool(contexts) or self._is_blank_seed(contexts)
        if not contexts:
            contexts = [self._default_context()]
            model_data["bounded_contexts"] = contexts

        for context in contexts:
            normalize_ul_lists(context)

        forbidden_terms = build_forbidden_terms(model_data)
        signals = [signal for signal in signals if signal.name.lower() not in forbidden_terms]

        # Per-context, per-bucket map of (name lowercase) -> list of merged
        # signals. Phase D3: we need to know which items got AST grounding so
        # _ensure_traceability can decide raise-vs-pass without fabrication.
        merged_signals_by_context: List[Dict[tuple[str, str], List[CandidateSignal]]] = [
            {} for _ in contexts
        ]

        for signal in signals:
            if not self._should_merge(signal, auto_created_context):
                self._drop(signal, "low-confidence")
                continue

            match = None if auto_created_context else self._select_context(signal, contexts)
            if not auto_created_context and not match:
                self._unassign(signal)
                continue

            target_idx = 0 if auto_created_context else match.index
            self._merge_signal(contexts[target_idx], signal)
            self.diagnostics["counts"][signal.candidate_type]["merged"] += 1
            merged_signals_by_context[target_idx].setdefault(
                (signal.candidate_type, signal.name.lower()), []
            ).append(signal)

        self._check_traceability(model_data, merged_signals_by_context)
        # WP-CORE-27a: stamp `is_mutable_in_code` on every LLM-claimed VO
        # using the AST-derived mutability index.  Items whose class
        # name does not appear in the index are left at the schema
        # default (False) — the conservative behaviour for an
        # AST-invisible VO is "no D9 violation signal".  Backward compat
        # for callers that supply `mutability_index=None`: this loop is
        # a no-op.
        if mutability_index is not None:
            for context in contexts:
                ul = context.get("ubiquitous_language", {}) or {}
                for vo in ul.get("value_objects", []) or []:
                    name_key = str(vo.get("name", "")).lower()
                    if name_key in mutability_index:
                        vo["is_mutable_in_code"] = bool(
                            mutability_index[name_key]
                        )
        self._finalize_rates()
        self._write_diagnostics()
        return model_data

    @staticmethod
    def _is_blank_seed(contexts: List[Dict[str, Any]]) -> bool:
        """Return True iff `contexts` is exactly one context with no domain
        vocabulary populated yet — i.e. a placeholder seed inserted to
        satisfy Phase A's "bounded_contexts must be non-empty" validator.
        """
        if len(contexts) != 1:
            return False
        ul = (contexts[0] or {}).get("ubiquitous_language", {}) or {}
        for bucket in ("entities", "value_objects", "services", "aggregates", "domain_events"):
            if ul.get(bucket):
                return False
        return True

    def _should_merge(self, signal: CandidateSignal, auto_created_context: bool) -> bool:
        if auto_created_context:
            return signal.confidence >= MERGE_THRESHOLD
        if signal.grounded_matches:
            return signal.confidence >= MERGE_THRESHOLD
        return signal.confidence >= AST_ONLY_MERGE_THRESHOLD

    def _select_context(
        self,
        signal: CandidateSignal,
        contexts: List[Dict[str, Any]],
    ) -> Optional[ContextMatch]:
        signal_terms = normalized_terms(signal.name) | signal.module_tokens
        matches: List[ContextMatch] = []

        for index, context in enumerate(contexts):
            context_terms = self._context_terms(context)
            context_name_terms = normalized_terms(str(context.get("context_name", "")))
            score = 0.0
            reasons: List[str] = []

            if signal.module_tokens & context_name_terms:
                score += 1.5
                reasons.append("module path matches context")
            if signal_terms & context_terms:
                score += min(1.5, 0.5 * len(signal_terms & context_terms))
                reasons.append("symbol overlaps context vocabulary")

            for grounding in signal.grounded_matches:
                grounded_terms = normalized_terms(grounding.snippet)
                if grounded_terms & context_terms:
                    score += min(2.0, 0.6 * len(grounded_terms & context_terms))
                    reasons.append("grounded requirement overlaps context vocabulary")
                if grounded_terms & context_name_terms:
                    score += 0.75
                    reasons.append("grounded requirement mentions context name")

            if score > 0:
                matches.append(ContextMatch(index=index, score=score, reasons=list(dict.fromkeys(reasons))))

        matches.sort(key=lambda item: item.score, reverse=True)
        if not matches or matches[0].score < CONTEXT_THRESHOLD:
            return None
        if len(matches) > 1 and matches[0].score - matches[1].score < CONTEXT_MARGIN:
            return None
        return matches[0]

    def _context_terms(self, context: Dict[str, Any]) -> Set[str]:
        terms = normalized_terms(str(context.get("context_name", "")))
        ul = (context or {}).get("ubiquitous_language", {})
        for bucket in ("entities", "value_objects", "services", "aggregates"):
            for item in ul.get(bucket, []) or []:
                terms.update(normalized_terms(str((item or {}).get("name", ""))))
        for event_name in ul.get("domain_events", []) or []:
            terms.update(normalized_terms(str(event_name)))
        return terms

    def _merge_signal(self, context: Dict[str, Any], signal: CandidateSignal) -> None:
        ul = context.setdefault("ubiquitous_language", {})
        normalize_ul_lists(context)
        if signal.candidate_type == "domain_events":
            events = ul.setdefault("domain_events", [])
            if signal.name not in events:
                events.append(signal.name)
            return

        bucket = ul[signal.candidate_type]
        current = next((item for item in bucket if item.get("name", "").lower() == signal.name.lower()), None)
        if current is None:
            bucket.append(signal.to_public_dict())
            return

        current["confidence"] = max(float(current.get("confidence", 0.5)), signal.confidence)
        current.setdefault("sources", []).extend(source.to_dict() for source in signal.sources)
        if signal.candidate_type == "value_objects":
            current["attributes"] = sorted(set(current.get("attributes", [])) | set(signal.attributes))

    def _check_traceability(
        self,
        model_data: Dict[str, Any],
        merged_signals_by_context: List[Dict[tuple[str, str], List[CandidateSignal]]],
    ) -> None:
        """Phase D3 (OQ2): replace fabricated 'generated' InferenceSource with
        a strict grounding check. For each entity/VO/service/aggregate dict
        in the model, ensure it has either valid SRS evidence (a non-negative
        evidence_sentence_indices entry) or AST signals merged into it.
        Raises InsufficientGroundingError on the first ungrounded item; the
        orchestrator counts the run as degraded.
        """
        contexts = model_data.get("bounded_contexts", []) or []
        for idx, context in enumerate(contexts):
            ul = context.get("ubiquitous_language", {})
            signals_map = merged_signals_by_context[idx] if idx < len(merged_signals_by_context) else {}
            for bucket in ("entities", "value_objects", "services", "aggregates"):
                for item in ul.get(bucket, []) or []:
                    item["confidence"] = round(clamp(float(item.get("confidence", 0.5))), 2)
                    item.setdefault("sources", [])
                    signal_dicts = [
                        {
                            "file": getattr(source, "file", None) or (source.get("file") if isinstance(source, dict) else None),
                            "line": getattr(source, "line", 1) if not isinstance(source, dict) else source.get("line", 1),
                            "rule": getattr(source, "rule", "AST") if not isinstance(source, dict) else source.get("rule", "AST"),
                            "evidence": getattr(source, "evidence", "") if not isinstance(source, dict) else source.get("evidence", ""),
                        }
                        for signal in signals_map.get((bucket, str(item.get("name", "")).lower()), [])
                        for source in signal.sources
                    ]
                    _ensure_traceability(item, ast_signals_for_entity=signal_dicts)

    def _drop(self, signal: CandidateSignal, reason: str) -> None:
        self.diagnostics["counts"][signal.candidate_type]["dropped"] += 1
        if len(self.diagnostics["samples"]["dropped"]) < 10:
            self.diagnostics["samples"]["dropped"].append(self._sample(signal, reason))

    def _unassign(self, signal: CandidateSignal) -> None:
        self.diagnostics["counts"][signal.candidate_type]["unassigned"] += 1
        if len(self.diagnostics["samples"]["unassigned"]) < 10:
            self.diagnostics["samples"]["unassigned"].append(self._sample(signal, "context-ambiguous"))

    def _sample(self, signal: CandidateSignal, reason: str) -> Dict[str, Any]:
        return {
            "type": signal.candidate_type,
            "name": signal.name,
            "confidence": signal.confidence,
            "reason": reason,
            "signals": signal.reasons[:5],
        }

    def _finalize_rates(self) -> None:
        totals = {"detected": 0, "grounded": 0, "merged": 0, "unassigned": 0}
        for stats in self.diagnostics["counts"].values():
            for key in totals:
                totals[key] += stats[key]
        detected = max(totals["detected"], 1)
        self.diagnostics["rates"] = {
            "grounding_hit_rate": round(totals["grounded"] / detected, 3),
            "context_assignment_success_rate": round(
                totals["merged"] / max(totals["merged"] + totals["unassigned"], 1),
                3,
            ),
        }

    def _write_diagnostics(self) -> None:
        INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        output = INTERMEDIATE_DIR / "ast_signals_diagnostics.json"
        output.write_text(json.dumps(self.diagnostics, indent=2), encoding="utf-8")

    def _default_context(self) -> Dict[str, Any]:
        return {
            "context_name": "CoreDomain",
            "description": "Auto-created context from AST signals",
            "allowed_dependencies": [],
            "ubiquitous_language": {
                "entities": [],
                "value_objects": [],
                "services": [],
                "aggregates": [],
                "domain_events": [],
            },
        }


def _ensure_traceability(
    entity_dict: Dict[str, Any],
    ast_signals_for_entity: List[Dict[str, Any]],
) -> None:
    """Ensure the entity has either SRS evidence (Phase D1's
    evidence_sentence_indices with valid non-sentinel values) or AST
    grounding (signals). Phase D3 (OQ2): raise
    InsufficientGroundingError instead of fabricating a 'generated'
    InferenceSource.

    Valid SRS evidence = evidence_sentence_indices contains at least
    one non-negative integer (the -1 sentinel emitted by AST
    enrichment for AST-derived entities is intentionally NOT valid
    SRS evidence; it must be accompanied by ast_signals_for_entity).
    """
    indices = entity_dict.get("evidence_sentence_indices") or []
    has_srs_evidence = any(i >= 0 for i in indices)
    has_ast_signal = bool(ast_signals_for_entity)
    if not has_srs_evidence and not has_ast_signal:
        raise InsufficientGroundingError(entity_name=entity_dict.get("name", "<unknown>"))
    for sig in ast_signals_for_entity:
        entity_dict.setdefault("sources", []).append({
            "file": sig.get("file"),
            "line": sig.get("line", 1),
            "rule": sig.get("rule", "AST"),
            "evidence": sig.get("evidence", ""),
        })


def build_forbidden_terms(model_data: Dict[str, Any]) -> Set[str]:
    forbidden: Set[str] = set()
    global_rules = model_data.get("global_rules") or {}
    for term in global_rules.get("banned_global_terms", []) or []:
        normalized = str(term).strip().lower()
        if normalized:
            forbidden.add(normalized)
    for context in model_data.get("bounded_contexts", []) or []:
        ul = (context or {}).get("ubiquitous_language", {})
        for entity in ul.get("entities", []) or []:
            for synonym in (entity or {}).get("synonyms_to_avoid", []) or []:
                normalized = str(synonym).strip().lower()
                if normalized:
                    forbidden.add(normalized)
    return forbidden


def normalize_ul_lists(context: Dict[str, Any]) -> None:
    ul = context.setdefault("ubiquitous_language", {})
    for key in CANDIDATE_TYPES:
        if ul.get(key) is None:
            ul[key] = []
