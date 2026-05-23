"""WP-CORE-22 — Repository + Factory AST detection tests (RED phase).

Closes detection blind spot: `Repository` and `Factory` previously sat in
`INFRASTRUCTURE_SUFFIXES` with a -0.50 penalty, making them impossible to
classify. This test file drives the positive-scorer + schema additions.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.AST.ast_model_signals import ASTModelSignalExtractor
from core.schemas import Factory, Repository, UbiquitousLanguage


class TestRepositoryFactoryDetection(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.extractor = ASTModelSignalExtractor()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _write(self, filename: str, content: str) -> str:
        path = os.path.join(self.test_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    # ------------------------------------------------------------------
    # Repository detection
    # ------------------------------------------------------------------

    def test_t_repo_1_explicit_repository_class_detected(self):
        """T-REPO-1: `*Repository` suffix + repo methods → repositories candidate."""
        code = """
class OrderRepository:
    def __init__(self, session):
        self.session = session

    def find_by_id(self, order_id):
        return self.session.query(Order).filter_by(id=order_id).first()

    def save(self, order):
        self.session.add(order)

    def delete(self, order):
        self.session.delete(order)
"""
        path = self._write("repo.py", code)
        results = self.extractor.extract_candidates([path])
        repo_names = [item["name"] for item in results.get("repositories", [])]
        self.assertIn("OrderRepository", repo_names)

    def test_t_repo_2_short_repo_suffix_detected(self):
        """T-REPO-2: short `*Repo` suffix also accepted."""
        code = """
class CustomerRepo:
    def find_by_email(self, email):
        pass

    def find_all(self):
        pass
"""
        path = self._write("repo2.py", code)
        results = self.extractor.extract_candidates([path])
        repo_names = [item["name"] for item in results.get("repositories", [])]
        self.assertIn("CustomerRepo", repo_names)

    def test_t_repo_3_non_repository_not_detected(self):
        """T-REPO-3: a plain service class is NOT flagged as repository."""
        code = """
class PaymentProcessor:
    def __init__(self, gateway):
        self.gateway = gateway

    def process(self, amount):
        pass
"""
        path = self._write("svc.py", code)
        results = self.extractor.extract_candidates([path])
        repo_names = [item["name"] for item in results.get("repositories", [])]
        self.assertNotIn("PaymentProcessor", repo_names)

    def test_t_repo_4_aggregate_root_derived_from_name(self):
        """T-REPO-4: payload carries `aggregate_root` derived by suffix strip."""
        code = """
class InvoiceRepository:
    def find_by_id(self, invoice_id):
        pass

    def save(self, invoice):
        pass
"""
        path = self._write("inv.py", code)
        results = self.extractor.extract_candidates([path])
        repos = results.get("repositories", [])
        target = next((r for r in repos if r["name"] == "InvoiceRepository"), None)
        self.assertIsNotNone(target)
        self.assertEqual(target["aggregate_root"], "Invoice")

    # ------------------------------------------------------------------
    # Factory detection
    # ------------------------------------------------------------------

    def test_t_factory_1_factory_with_create_methods_detected(self):
        """T-FACTORY-1: `*Factory` with `create_*` static methods → factories candidate."""
        code = """
class OrderFactory:
    @staticmethod
    def create_order(customer_id, items):
        return Order(customer_id=customer_id, items=items)

    @staticmethod
    def create_empty():
        return Order()
"""
        path = self._write("factory.py", code)
        results = self.extractor.extract_candidates([path])
        factory_names = [item["name"] for item in results.get("factories", [])]
        self.assertIn("OrderFactory", factory_names)

    def test_t_factory_2_builder_with_build_method_detected(self):
        """T-FACTORY-2: `*Builder` with `build_*` method accepted."""
        code = """
class InvoiceBuilder:
    def __init__(self):
        self._lines = []

    def add_line(self, line):
        self._lines.append(line)

    def build_invoice(self):
        return Invoice(lines=self._lines)
"""
        path = self._write("builder.py", code)
        results = self.extractor.extract_candidates([path])
        factory_names = [item["name"] for item in results.get("factories", [])]
        self.assertIn("InvoiceBuilder", factory_names)

    def test_t_factory_3_random_class_not_detected(self):
        """T-FACTORY-3: ordinary class is NOT flagged as factory."""
        code = """
class Customer:
    def __init__(self, customer_id, name):
        self.customer_id = customer_id
        self.name = name
"""
        path = self._write("customer.py", code)
        results = self.extractor.extract_candidates([path])
        factory_names = [item["name"] for item in results.get("factories", [])]
        self.assertNotIn("Customer", factory_names)

    def test_t_factory_4_produces_derived_from_name(self):
        """T-FACTORY-4: payload carries `produces` derived by suffix strip."""
        code = """
class OrderFactory:
    @staticmethod
    def create_order():
        pass

    @staticmethod
    def build_for(customer):
        pass
"""
        path = self._write("factory2.py", code)
        results = self.extractor.extract_candidates([path])
        target = next(
            (f for f in results.get("factories", []) if f["name"] == "OrderFactory"),
            None,
        )
        self.assertIsNotNone(target)
        self.assertEqual(target["produces"], "Order")

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def test_t_schema_1_repository_pydantic_instantiation(self):
        """T-SCHEMA-1: Repository pydantic class accepts minimal valid payload."""
        repo = Repository(
            name="OrderRepository",
            aggregate_root="Order",
            confidence=0.7,
        )
        self.assertEqual(repo.name, "OrderRepository")
        self.assertEqual(repo.aggregate_root, "Order")
        self.assertEqual(repo.confidence, 0.7)
        self.assertEqual(repo.sources, [])

    def test_t_schema_2_factory_pydantic_instantiation(self):
        """T-SCHEMA-2: Factory pydantic class accepts minimal valid payload."""
        factory = Factory(name="OrderFactory", produces="Order", confidence=0.65)
        self.assertEqual(factory.name, "OrderFactory")
        self.assertEqual(factory.produces, "Order")

    def test_t_schema_3_ubiquitous_language_repositories_field(self):
        """T-SCHEMA-3: UbiquitousLanguage accepts repositories + factories optional fields."""
        ul = UbiquitousLanguage(
            entities=[],
            value_objects=None,
            services=None,
            aggregates=None,
            domain_events=None,
            repositories=[Repository(name="OrderRepository", aggregate_root="Order")],
            factories=[Factory(name="OrderFactory", produces="Order")],
        )
        assert ul.repositories is not None and len(ul.repositories) == 1
        assert ul.factories is not None and len(ul.factories) == 1


if __name__ == "__main__":
    unittest.main()
