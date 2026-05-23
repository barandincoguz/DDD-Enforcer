"""WP-CORE-31 — Cross-context import topology tests.

Tests the 3 standalone helpers in `core.AST.import_graph`. Pipeline
integration (auto-populate BoundedContext.allowed_dependencies) is a
follow-up (WP-CORE-31b); this WP ships the helpers + tests only.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from core.AST.import_graph import (
    build_module_import_graph,
    derive_context_dependencies,
    map_modules_to_contexts,
)


class _FixtureWorkspace(unittest.TestCase):
    """Mixin: tmp workspace with utility to write .py files."""
    def setUp(self):
        self.workspace = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _write(self, rel_path: str, content: str) -> str:
        full = os.path.join(self.workspace, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        return full


# ---------------------------------------------------------------------------
# build_module_import_graph
# ---------------------------------------------------------------------------

class TestBuildImportGraph(_FixtureWorkspace):
    def test_t_import_1_simple_two_files(self):
        """T-IMPORT-1: two files with import edge → graph captures it."""
        path_a = self._write("a.py", "import b\n")
        path_b = self._write("b.py", "x = 1\n")
        graph = build_module_import_graph(
            [path_a, path_b], workspace_root=self.workspace,
        )
        # Modules keyed by file stem.
        self.assertIn("a", graph)
        self.assertIn("b", graph)
        self.assertIn("b", graph["a"])
        self.assertEqual(graph["b"], set())

    def test_t_import_2_from_import(self):
        """T-IMPORT-2: `from X.Y import Z` records `X.Y` as imported module."""
        path = self._write("svc.py", "from billing.repository import OrderRepo\n")
        graph = build_module_import_graph([path], workspace_root=self.workspace)
        imports = graph["svc"]
        self.assertIn("billing.repository", imports)

    def test_t_import_3_relative_import(self):
        """T-IMPORT-3: `from . import foo` records `.foo` form."""
        path = self._write("pkg/module.py", "from . import sibling\n")
        graph = build_module_import_graph([path], workspace_root=self.workspace)
        imports = graph["pkg.module"]
        # Either ".sibling" or "." surface depending on resolver.
        self.assertTrue(
            any(imp.startswith(".") for imp in imports),
            f"expected relative import marker; got {imports}",
        )

    def test_t_import_4_empty_file(self):
        """T-IMPORT-4: empty file → empty import set."""
        path = self._write("empty.py", "")
        graph = build_module_import_graph([path], workspace_root=self.workspace)
        self.assertEqual(graph["empty"], set())

    def test_t_import_5_multiple_imports(self):
        """T-IMPORT-5: `import a, b, c` records all three."""
        path = self._write("multi.py", "import os, json, re\n")
        graph = build_module_import_graph([path], workspace_root=self.workspace)
        for stdlib in ("os", "json", "re"):
            self.assertIn(stdlib, graph["multi"])

    def test_t_import_6_parse_error_skipped(self):
        """T-IMPORT-6: malformed .py file is skipped (not in graph)."""
        good = self._write("good.py", "import os\n")
        bad = self._write("bad.py", "def broken(:\n")
        graph = build_module_import_graph(
            [good, bad], workspace_root=self.workspace,
        )
        self.assertIn("good", graph)
        self.assertNotIn("bad", graph)


# ---------------------------------------------------------------------------
# map_modules_to_contexts
# ---------------------------------------------------------------------------

class TestMapModulesToContexts(unittest.TestCase):
    def test_t_map_1_path_segment_match(self):
        """T-MAP-1: module with directory matching a context → mapped."""
        mapping = map_modules_to_contexts(
            ["billing.service", "ordering.repository"],
            context_names=["Billing", "Ordering"],
        )
        self.assertEqual(mapping["billing.service"], "Billing")
        self.assertEqual(mapping["ordering.repository"], "Ordering")

    def test_t_map_2_case_insensitive(self):
        """T-MAP-2: case mismatch between dir name and context name still matches."""
        mapping = map_modules_to_contexts(
            ["BILLING.service", "Ordering.repo"],
            context_names=["billing", "ORDERING"],
        )
        self.assertEqual(mapping["BILLING.service"], "billing")
        self.assertEqual(mapping["Ordering.repo"], "ORDERING")

    def test_t_map_3_underscore_hyphen_normalized(self):
        """T-MAP-3: 'user_management' module ↔ 'UserManagement' context."""
        mapping = map_modules_to_contexts(
            ["user_management.service"],
            context_names=["UserManagement"],
        )
        self.assertEqual(mapping["user_management.service"], "UserManagement")

    def test_t_map_4_no_match_is_none(self):
        """T-MAP-4: module with no segment matching any context → None."""
        mapping = map_modules_to_contexts(
            ["common.utils"],
            context_names=["Billing", "Ordering"],
        )
        self.assertIsNone(mapping["common.utils"])

    def test_t_map_5_first_match_wins(self):
        """T-MAP-5: when multiple segments match, first (closest to root) wins."""
        mapping = map_modules_to_contexts(
            ["billing.ordering.cross"],
            context_names=["Billing", "Ordering"],
        )
        # First segment "billing" matches first → context = "Billing".
        self.assertEqual(mapping["billing.ordering.cross"], "Billing")


# ---------------------------------------------------------------------------
# derive_context_dependencies
# ---------------------------------------------------------------------------

class TestDeriveContextDependencies(unittest.TestCase):
    def test_t_derive_1_simple_edge(self):
        """T-DERIVE-1: A imports B (different contexts) → A → B edge."""
        import_graph = {"billing.svc": {"ordering.repo"}}
        ctx_map = {
            "billing.svc": "Billing",
            "ordering.repo": "Ordering",
        }
        deps = derive_context_dependencies(import_graph, ctx_map)
        self.assertIn("Billing", deps)
        self.assertEqual(deps["Billing"], {"Ordering"})

    def test_t_derive_2_intra_context_excluded(self):
        """T-DERIVE-2: same-context imports do NOT generate dependency edges."""
        import_graph = {"billing.svc": {"billing.repo"}}
        ctx_map = {"billing.svc": "Billing", "billing.repo": "Billing"}
        deps = derive_context_dependencies(import_graph, ctx_map)
        # No outgoing edges from Billing.
        self.assertEqual(deps.get("Billing", set()), set())

    def test_t_derive_3_unmapped_modules_ignored(self):
        """T-DERIVE-3: edges touching None-mapped modules are dropped."""
        import_graph = {"common.utils": {"billing.svc"}}
        ctx_map = {"common.utils": None, "billing.svc": "Billing"}
        deps = derive_context_dependencies(import_graph, ctx_map)
        self.assertEqual(deps, {})

    def test_t_derive_4_aggregate_multi_edges(self):
        """T-DERIVE-4: Billing → Ordering and Billing → Catalog aggregate."""
        import_graph = {
            "billing.svc": {"ordering.repo", "catalog.repo"},
        }
        ctx_map = {
            "billing.svc": "Billing",
            "ordering.repo": "Ordering",
            "catalog.repo": "Catalog",
        }
        deps = derive_context_dependencies(import_graph, ctx_map)
        self.assertEqual(deps["Billing"], {"Ordering", "Catalog"})

    def test_t_derive_5_third_party_imports_skipped(self):
        """T-DERIVE-5: imports targeting stdlib / third-party (unmapped) skipped."""
        import_graph = {"billing.svc": {"os", "json", "pydantic"}}
        ctx_map = {
            "billing.svc": "Billing",
            # stdlib/3rd-party deliberately not mapped to any context.
        }
        deps = derive_context_dependencies(import_graph, ctx_map)
        self.assertEqual(deps.get("Billing", set()), set())


# ---------------------------------------------------------------------------
# End-to-end fixture
# ---------------------------------------------------------------------------

class TestEndToEnd(_FixtureWorkspace):
    def test_t_e2e_fixture_project(self):
        """T-E2E: 4-file 3-context fixture produces the expected dep graph.

        - billing/svc.py:    imports ordering.repo + os
        - billing/repo.py:   imports os
        - ordering/repo.py:  imports catalog.product
        - catalog/product.py: standalone (just defines Product)
        Expected: Billing → Ordering. Ordering → Catalog. Catalog → ∅.
        """
        b_svc = self._write("billing/svc.py", "import os\nfrom ordering.repo import OrderRepo\n")
        b_repo = self._write("billing/repo.py", "import os\n")
        o_repo = self._write("ordering/repo.py", "from catalog.product import Product\n")
        c_prod = self._write("catalog/product.py", "class Product:\n    pass\n")

        import_graph = build_module_import_graph(
            [b_svc, b_repo, o_repo, c_prod], workspace_root=self.workspace,
        )
        ctx_map = map_modules_to_contexts(
            import_graph.keys(),
            context_names=["Billing", "Ordering", "Catalog"],
        )
        deps = derive_context_dependencies(import_graph, ctx_map)

        self.assertEqual(deps.get("Billing"), {"Ordering"})
        self.assertEqual(deps.get("Ordering"), {"Catalog"})
        self.assertNotIn("Catalog", deps)


if __name__ == "__main__":
    unittest.main()
