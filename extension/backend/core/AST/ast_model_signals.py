"""
AST Model Signals facade.

Keeps the public API stable while delegating discovery, classification,
grounding, and enrichment to smaller internal modules.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from core.AST.ast_signal_classification import SignalClassifier
from core.AST.ast_signal_discovery import DEFAULT_SKIP_DIRS, extract_class_facts, find_python_files
from core.AST.ast_signal_enrichment import SignalEnricher
from core.AST.ast_signal_types import CandidateSignal
from core.AST.import_graph import apply_import_topology_to_model
from core.schemas import DomainModel


logger = logging.getLogger(__name__)


class ASTModelSignalExtractor:
    """Extract AST-based DDD candidates and enrich the synthesized domain model."""

    def __init__(self, ignore_paths: Optional[List[str]] = None):
        self.skip_dirs: Set[str] = set(DEFAULT_SKIP_DIRS)
        self.ignore_paths = ignore_paths or []
        self.classifier = SignalClassifier()

    def find_python_files(self, workspace_path: str) -> List[str]:
        return find_python_files(
            workspace_path=workspace_path,
            skip_dirs=self.skip_dirs,
            ignore_paths=self.ignore_paths,
        )

    def extract_candidates(
        self,
        python_files: List[str],
        grounding_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        signals = self._collect_signals(python_files)
        enricher = SignalEnricher(workspace_path="")
        signals = enricher.deduplicate_signals(signals)
        enricher.apply_grounding(signals, grounding_docs)
        return self._to_public_candidate_map(signals)

    def enrich_domain_model(
        self,
        model: DomainModel,
        workspace_path: str,
        srs_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> DomainModel:
        model_data = model.model_dump(mode="json")
        python_files = self.find_python_files(workspace_path)
        signals = self._collect_signals(python_files)

        enricher = SignalEnricher(workspace_path=workspace_path)
        signals = enricher.deduplicate_signals(signals)
        enricher.apply_grounding(signals, srs_docs)
        model_data = enricher.enrich_model(model_data, signals, srs_docs=srs_docs)
        # WP-CORE-31b: auto-populate BoundedContext.allowed_dependencies
        # from the AST-derived import topology + record cross-check diff.
        # Runs after the main enrichment pass so any contexts the
        # enricher may have introduced (default CoreDomain seeding) are
        # also considered.
        self._apply_import_topology(
            model_data,
            python_files=python_files,
            workspace_root=workspace_path,
            enricher=enricher,
        )
        return DomainModel(**model_data)

    def _apply_import_topology(
        self,
        model_data: Dict[str, Any],
        python_files: List[str],
        workspace_root: str,
        enricher: SignalEnricher,
    ) -> None:
        """WP-CORE-31b: thin wrapper around `apply_import_topology_to_model`
        that stashes the diagnostics block on the supplied enricher so it
        ends up serialised next to the other AST diagnostics.

        Defensive: any error in graph construction (parse errors,
        permission issues) is propagated only via the helper's own
        silent-skip behaviour; the enrichment pipeline never crashes on
        an import-topology hiccup.
        """
        topology_diags = apply_import_topology_to_model(
            model_data,
            python_files=python_files,
            workspace_root=workspace_root,
        )
        enricher.diagnostics["import_topology"] = topology_diags

    def _collect_signals(self, python_files: List[str]) -> List[CandidateSignal]:
        signals: List[CandidateSignal] = []
        for file_path in python_files:
            try:
                for facts in extract_class_facts(file_path):
                    signals.extend(self.classifier.classify(facts))
            except (SyntaxError, UnicodeDecodeError, ValueError, OSError) as exc:
                # G01: raise instead of silently logging and continuing.
                # Malformed Python files in the workspace now propagate errors
                # instead of vanishing from the candidate set.
                raise SyntaxError(f"AST signal extraction failed for {file_path}: {exc}") from exc
        return signals

    def _to_public_candidate_map(
        self,
        signals: List[CandidateSignal],
    ) -> Dict[str, List[Dict[str, Any]]]:
        # WP-CORE-22: repositories + factories are first-class candidates.
        # WP-CORE-33: anti_corruption_layers + specifications are first-class.
        # domain_events are excluded because they use a different output
        # shape (list of strings, not list of dicts).
        result: Dict[str, List[Dict[str, Any]]] = {
            "entities": [],
            "value_objects": [],
            "services": [],
            "aggregates": [],
            "repositories": [],
            "factories": [],
            "anti_corruption_layers": [],
            "specifications": [],
        }
        for signal in signals:
            if signal.candidate_type not in result:
                continue
            result[signal.candidate_type].append(signal.to_public_dict())
        return result
