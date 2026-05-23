"""WP-CORE-31b — Cross-context import topology pipeline integration tests.

Closes the WP-CORE-31 follow-up: the standalone helpers
(`build_module_import_graph` / `map_modules_to_contexts` /
`derive_context_dependencies`) are composed into a single pipeline-aware
helper `apply_import_topology_to_model(...)` that:

1. Auto-populates `BoundedContext.allowed_dependencies` when the LLM
   left it empty / None and the AST-derived graph found cross-context
   import edges.
2. Records a cross-check diff (`extra_in_llm`, `extra_in_derived`) when
   both LLM and derived graphs are non-empty so reviewers can spot
   architectural drift.
3. Returns a serialisable diagnostics block that the enricher attaches
   to `ast_signals_diagnostics.json`.

Tests exercise the helper directly (RED-first) plus one thin
`ASTModelSignalExtractor.enrich_domain_model` smoke check confirming the
diagnostics block lands on the enricher.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.AST.import_graph import apply_import_topology_to_model


class _FixtureWorkspace(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _write(self, rel_path: str, content: str) -> str:
        full = os.path.join(self.workspace, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        return full

    @staticmethod
    def _model(contexts):
        return {"bounded_contexts": contexts}


# =============================================================================
# Direct helper behaviour
# =============================================================================


class TestApplyImportTopology(_FixtureWorkspace):
    def test_t_31b_1_auto_populates_empty_allowed_dependencies(self) -> None:
        """T-31b-1: LLM left allowed_dependencies = None; derived graph
        finds a cross-context edge -> auto-populate."""
        path_a = self._write(
            "billing/order.py", "from inventory.stock import Stock\n"
        )
        path_b = self._write("inventory/stock.py", "x = 1\n")
        model = self._model([
            {"context_name": "Billing", "allowed_dependencies": None},
            {"context_name": "Inventory", "allowed_dependencies": None},
        ])
        diags = apply_import_topology_to_model(
            model, [path_a, path_b], workspace_root=self.workspace,
        )
        billing = next(
            c for c in model["bounded_contexts"] if c["context_name"] == "Billing"
        )
        self.assertEqual(billing["allowed_dependencies"], ["Inventory"])
        self.assertIn("Billing", diags["auto_populated"])

    def test_t_31b_2_preserves_existing_allowed_dependencies(self) -> None:
        """T-31b-2: existing non-empty allowed_dependencies is NOT overwritten."""
        path_a = self._write(
            "billing/order.py", "from inventory.stock import Stock\n"
        )
        path_b = self._write("inventory/stock.py", "x = 1\n")
        model = self._model([
            {
                "context_name": "Billing",
                "allowed_dependencies": ["Inventory", "Shared"],
            },
            {"context_name": "Inventory", "allowed_dependencies": None},
        ])
        apply_import_topology_to_model(
            model, [path_a, path_b], workspace_root=self.workspace,
        )
        billing = next(
            c for c in model["bounded_contexts"] if c["context_name"] == "Billing"
        )
        self.assertEqual(
            billing["allowed_dependencies"], ["Inventory", "Shared"]
        )

    def test_t_31b_3_cross_check_diff_records_extras(self) -> None:
        """T-31b-3: LLM declared + derived non-empty, sets differ ->
        diff dict includes the LLM-only and derived-only extras."""
        path_a = self._write(
            "billing/order.py", "from inventory.stock import Stock\n"
        )
        path_b = self._write("inventory/stock.py", "x = 1\n")
        model = self._model([
            {
                "context_name": "Billing",
                "allowed_dependencies": ["Shared"],
            },
            {"context_name": "Inventory", "allowed_dependencies": None},
        ])
        diags = apply_import_topology_to_model(
            model, [path_a, path_b], workspace_root=self.workspace,
        )
        diff = diags["cross_check_diff"]
        self.assertIn("Billing", diff)
        self.assertIn("Shared", diff["Billing"]["extra_in_llm"])
        self.assertIn("Inventory", diff["Billing"]["extra_in_derived"])

    def test_t_31b_4_no_imports_keeps_dependencies_none(self) -> None:
        """T-31b-4: workspace with no cross-context imports -> no auto-populate."""
        path_a = self._write("billing/order.py", "x = 1\n")
        path_b = self._write("inventory/stock.py", "x = 1\n")
        model = self._model([
            {"context_name": "Billing", "allowed_dependencies": None},
            {"context_name": "Inventory", "allowed_dependencies": None},
        ])
        diags = apply_import_topology_to_model(
            model, [path_a, path_b], workspace_root=self.workspace,
        )
        for ctx in model["bounded_contexts"]:
            self.assertIsNone(ctx["allowed_dependencies"])
        self.assertEqual(diags["auto_populated"], [])

    def test_t_31b_5_intra_context_imports_excluded(self) -> None:
        """T-31b-5: imports inside the same context don't count as deps."""
        path_a = self._write(
            "billing/order.py", "from billing.invoice import Invoice\n"
        )
        path_b = self._write("billing/invoice.py", "x = 1\n")
        model = self._model([
            {"context_name": "Billing", "allowed_dependencies": None},
        ])
        apply_import_topology_to_model(
            model, [path_a, path_b], workspace_root=self.workspace,
        )
        billing = model["bounded_contexts"][0]
        self.assertIsNone(billing["allowed_dependencies"])

    def test_t_31b_6_unmapped_context_skipped(self) -> None:
        """T-31b-6: a context whose name matches no module is skipped."""
        path = self._write("foo/bar.py", "x = 1\n")
        model = self._model([
            {"context_name": "OrphanContext", "allowed_dependencies": None},
        ])
        diags = apply_import_topology_to_model(
            model, [path], workspace_root=self.workspace,
        )
        ctx = model["bounded_contexts"][0]
        self.assertIsNone(ctx["allowed_dependencies"])
        self.assertEqual(diags["auto_populated"], [])

    def test_t_31b_7_parse_error_file_silently_skipped(self) -> None:
        """T-31b-7: malformed .py file is silently skipped (helper-level,
        mirrors build_module_import_graph's documented behaviour)."""
        path_good = self._write(
            "billing/order.py", "from inventory.stock import Stock\n"
        )
        path_bad = self._write("inventory/stock.py", "this is not valid python ::: !\n")
        model = self._model([
            {"context_name": "Billing", "allowed_dependencies": None},
            {"context_name": "Inventory", "allowed_dependencies": None},
        ])
        # Should not raise.
        diags = apply_import_topology_to_model(
            model, [path_good, path_bad], workspace_root=self.workspace,
        )
        self.assertIsInstance(diags, dict)

    def test_t_31b_8_diagnostics_structure(self) -> None:
        """T-31b-8: returned diagnostics block carries the documented keys."""
        path_a = self._write(
            "billing/order.py", "from inventory.stock import Stock\n"
        )
        path_b = self._write("inventory/stock.py", "x = 1\n")
        model = self._model([
            {"context_name": "Billing", "allowed_dependencies": None},
            {"context_name": "Inventory", "allowed_dependencies": None},
        ])
        diags = apply_import_topology_to_model(
            model, [path_a, path_b], workspace_root=self.workspace,
        )
        self.assertIn("derived", diags)
        self.assertIn("auto_populated", diags)
        self.assertIn("cross_check_diff", diags)
        # Values are JSON-safe (sorted lists, no sets).
        derived_billing = diags["derived"].get("Billing", [])
        self.assertIsInstance(derived_billing, list)
        self.assertEqual(diags["derived"]["Billing"], ["Inventory"])

    def test_t_31b_9_empty_model_is_noop(self) -> None:
        """T-31b-9: model with zero bounded_contexts -> empty diagnostics, no error."""
        model = self._model([])
        diags = apply_import_topology_to_model(
            model, [], workspace_root=self.workspace,
        )
        self.assertEqual(diags["auto_populated"], [])
        self.assertEqual(diags["cross_check_diff"], {})
        self.assertEqual(diags["derived"], {})


# =============================================================================
# enrich_domain_model integration smoke
# =============================================================================


class TestEnrichDomainModelStashesTopology(_FixtureWorkspace):
    def test_t_31b_10_extractor_stashes_topology_on_enricher_diagnostics(self) -> None:
        """T-31b-10: ASTModelSignalExtractor.enrich_domain_model writes the
        topology block to enricher.diagnostics["import_topology"].

        Pipeline-level integration smoke. The extractor wires the new
        helper between `enricher.enrich_model(...)` and the final
        `DomainModel(**model_data)` re-validation.
        """
        from core.AST.ast_model_signals import ASTModelSignalExtractor
        from core.AST.ast_signal_enrichment import SignalEnricher

        # Direct enricher use covers the wiring without needing a full
        # DomainModel round-trip (which would force Entity grounding).
        enricher = SignalEnricher(workspace_path=self.workspace)
        path_a = self._write(
            "billing/order.py", "from inventory.stock import Stock\n"
        )
        path_b = self._write("inventory/stock.py", "x = 1\n")
        model_data = {
            "bounded_contexts": [
                {
                    "context_name": "Billing",
                    "allowed_dependencies": None,
                    "ubiquitous_language": {},
                },
                {
                    "context_name": "Inventory",
                    "allowed_dependencies": None,
                    "ubiquitous_language": {},
                },
            ],
        }

        # Mimic the wiring step that enrich_domain_model performs.
        extractor = ASTModelSignalExtractor()
        extractor._apply_import_topology(  # type: ignore[attr-defined]
            model_data,
            python_files=[path_a, path_b],
            workspace_root=self.workspace,
            enricher=enricher,
        )

        self.assertIn("import_topology", enricher.diagnostics)
        topology = enricher.diagnostics["import_topology"]
        self.assertEqual(topology["derived"].get("Billing"), ["Inventory"])
        billing = next(
            c for c in model_data["bounded_contexts"]
            if c["context_name"] == "Billing"
        )
        self.assertEqual(billing["allowed_dependencies"], ["Inventory"])


if __name__ == "__main__":
    unittest.main()
