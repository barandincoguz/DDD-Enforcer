"""WP-CORE-27 — Verifier D9 (VO mutability) + S3 (empty aggregate) + D11
(dependency cycle) tests (RED phase).

- D9: ValueObject claimed in the domain model but AST evidence shows
  the class has mutation methods → mismatch ERROR. v1 ships the schema
  field + check; AST cross-reference that populates `is_mutable_in_code`
  is a follow-up WP.
- S3: Aggregate with empty `members` list → WARN companion to D4.
- D11: Cycle in BoundedContext.allowed_dependencies graph → ERROR.
  Pure deterministic DFS.
"""
from __future__ import annotations

import unittest

from core.observability._verifier_mapping import (
    CANONICAL_CHECK_IDS,
    canonical_check_id,
    empty_issue_counts,
)
from core.schemas import InferenceSource, ValueObject
from core.verifier.checks_deterministic import (
    check_d11_dependency_cycle_free,
    check_d9_value_object_mutability_consistency,
    check_s3_aggregate_members_nonempty,
)
from core.verifier.types import IssueSeverity


# ---------------------------------------------------------------------------
# D9 ValueObject mutability claim mismatch
# ---------------------------------------------------------------------------

class TestD9ValueObjectMutability(unittest.TestCase):
    def test_t_d9_1_mutable_flagged(self):
        """T-D9-1: VO with is_mutable_in_code=True → ERROR."""
        vos = [{"name": "Money", "is_mutable_in_code": True}]
        issues = check_d9_value_object_mutability_consistency(
            context_name="Billing", value_objects=vos,
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, "value_object_mutable")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)
        self.assertIn("Money", issues[0].message)

    def test_t_d9_2_immutable_pass(self):
        """T-D9-2: VO with is_mutable_in_code=False → no issue."""
        vos = [{"name": "Money", "is_mutable_in_code": False}]
        issues = check_d9_value_object_mutability_consistency(
            context_name="Billing", value_objects=vos,
        )
        self.assertEqual(issues, [])

    def test_t_d9_3_default_false_pass(self):
        """T-D9-3: VO without the field defaults to immutable → no issue."""
        vos = [{"name": "Money"}]
        issues = check_d9_value_object_mutability_consistency(
            context_name="Billing", value_objects=vos,
        )
        self.assertEqual(issues, [])

    def test_t_d9_4_mixed(self):
        """T-D9-4: 2 VOs, only the mutable one fires."""
        vos = [
            {"name": "Money", "is_mutable_in_code": False},
            {"name": "Address", "is_mutable_in_code": True},
        ]
        issues = check_d9_value_object_mutability_consistency(
            context_name="Billing", value_objects=vos,
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].message.count("Address"), 1)

    def test_t_d9_schema_field_exists(self):
        """T-D9-SCHEMA: ValueObject Pydantic model accepts is_mutable_in_code."""
        vo = ValueObject(
            name="Money",
            attributes=["amount", "currency"],
            description="Monetary value",
            confidence=0.9,
            sources=[],
            is_mutable_in_code=False,
        )
        self.assertFalse(vo.is_mutable_in_code)
        vo_mut = ValueObject(
            name="MutableThing",
            attributes=["x"],
            description="...",
            confidence=0.9,
            sources=[],
            is_mutable_in_code=True,
        )
        self.assertTrue(vo_mut.is_mutable_in_code)


# ---------------------------------------------------------------------------
# S3 Empty aggregate
# ---------------------------------------------------------------------------

class TestS3EmptyAggregate(unittest.TestCase):
    def test_t_s3_1_empty_members_warn(self):
        """T-S3-1: Aggregate with empty members → WARN."""
        aggregates = [{"name": "OrderAggregate", "members": []}]
        issues = check_s3_aggregate_members_nonempty(
            context_name="Ordering", aggregates=aggregates,
        )
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, "empty_aggregate")
        self.assertEqual(issues[0].severity, IssueSeverity.WARN)

    def test_t_s3_2_with_members_pass(self):
        """T-S3-2: Aggregate with members → no issue."""
        aggregates = [{"name": "OrderAggregate", "members": ["Order", "OrderLine"]}]
        issues = check_s3_aggregate_members_nonempty(
            context_name="Ordering", aggregates=aggregates,
        )
        self.assertEqual(issues, [])

    def test_t_s3_3_missing_members_field_warn(self):
        """T-S3-3: Aggregate missing members field treated as empty."""
        aggregates = [{"name": "OrderAggregate"}]
        issues = check_s3_aggregate_members_nonempty(
            context_name="Ordering", aggregates=aggregates,
        )
        self.assertEqual(len(issues), 1)

    def test_t_s3_4_mixed_aggregates(self):
        """T-S3-4: 2 aggregates, only the empty one fires."""
        aggregates = [
            {"name": "OrderAggregate", "members": ["Order"]},
            {"name": "Empty", "members": []},
        ]
        issues = check_s3_aggregate_members_nonempty(
            context_name="Ordering", aggregates=aggregates,
        )
        self.assertEqual(len(issues), 1)
        self.assertIn("Empty", issues[0].message)


# ---------------------------------------------------------------------------
# D11 Dependency cycle
# ---------------------------------------------------------------------------

class TestD11DependencyCycle(unittest.TestCase):
    def test_t_d11_1_linear_no_cycle(self):
        """T-D11-1: A→B→C linear graph → no issue."""
        contexts = [
            {"name": "A", "allowed_dependencies": ["B"]},
            {"name": "B", "allowed_dependencies": ["C"]},
            {"name": "C", "allowed_dependencies": []},
        ]
        issues = check_d11_dependency_cycle_free(contexts)
        self.assertEqual(issues, [])

    def test_t_d11_2_two_cycle(self):
        """T-D11-2: A→B→A → ERROR."""
        contexts = [
            {"name": "A", "allowed_dependencies": ["B"]},
            {"name": "B", "allowed_dependencies": ["A"]},
        ]
        issues = check_d11_dependency_cycle_free(contexts)
        self.assertGreaterEqual(len(issues), 1)
        self.assertEqual(issues[0].issue_type, "dependency_cycle")
        self.assertEqual(issues[0].severity, IssueSeverity.ERROR)

    def test_t_d11_3_three_cycle(self):
        """T-D11-3: A→B→C→A → ERROR."""
        contexts = [
            {"name": "A", "allowed_dependencies": ["B"]},
            {"name": "B", "allowed_dependencies": ["C"]},
            {"name": "C", "allowed_dependencies": ["A"]},
        ]
        issues = check_d11_dependency_cycle_free(contexts)
        self.assertGreaterEqual(len(issues), 1)

    def test_t_d11_4_dag_fanout(self):
        """T-D11-4: A→B, A→C fan-out DAG → no cycle."""
        contexts = [
            {"name": "A", "allowed_dependencies": ["B", "C"]},
            {"name": "B", "allowed_dependencies": []},
            {"name": "C", "allowed_dependencies": []},
        ]
        issues = check_d11_dependency_cycle_free(contexts)
        self.assertEqual(issues, [])

    def test_t_d11_5_dag_diamond(self):
        """T-D11-5: A→B, A→C, B→D, C→D diamond DAG → no cycle."""
        contexts = [
            {"name": "A", "allowed_dependencies": ["B", "C"]},
            {"name": "B", "allowed_dependencies": ["D"]},
            {"name": "C", "allowed_dependencies": ["D"]},
            {"name": "D", "allowed_dependencies": []},
        ]
        issues = check_d11_dependency_cycle_free(contexts)
        self.assertEqual(issues, [])

    def test_t_d11_6_self_loop(self):
        """T-D11-6: A→A self-loop → ERROR."""
        contexts = [
            {"name": "A", "allowed_dependencies": ["A"]},
        ]
        issues = check_d11_dependency_cycle_free(contexts)
        self.assertGreaterEqual(len(issues), 1)

    def test_t_d11_7_unknown_dep_ignored(self):
        """T-D11-7: dependency on non-existent context is silently ignored
        (D5 catches that separately; D11 only acts on the known graph)."""
        contexts = [
            {"name": "A", "allowed_dependencies": ["NonExistent"]},
        ]
        issues = check_d11_dependency_cycle_free(contexts)
        self.assertEqual(issues, [])


# ---------------------------------------------------------------------------
# Mapping updates
# ---------------------------------------------------------------------------

class TestVerifierMappingUpdates(unittest.TestCase):
    def test_t_mapping_d9(self):
        """T-MAPPING-D9: value_object_mutable → D9."""
        class Fake:
            issue_type = "value_object_mutable"
            check_id = None
        self.assertEqual(canonical_check_id(Fake()), "D9")

    def test_t_mapping_s3(self):
        """T-MAPPING-S3: empty_aggregate → S3."""
        class Fake:
            issue_type = "empty_aggregate"
            check_id = None
        self.assertEqual(canonical_check_id(Fake()), "S3")

    def test_t_mapping_d11(self):
        """T-MAPPING-D11: dependency_cycle → D11."""
        class Fake:
            issue_type = "dependency_cycle"
            check_id = None
        self.assertEqual(canonical_check_id(Fake()), "D11")

    def test_t_mapping_canonical_ids_extended(self):
        """T-MAPPING-CANONICAL: CANONICAL_CHECK_IDS contains D9, D11, S3."""
        self.assertIn("D9", CANONICAL_CHECK_IDS)
        self.assertIn("D11", CANONICAL_CHECK_IDS)
        self.assertIn("S3", CANONICAL_CHECK_IDS)

    def test_t_mapping_empty_counts_initializes_new(self):
        """T-MAPPING-EMPTY: empty_issue_counts() seeds D9/D11/S3 with 0."""
        counts = empty_issue_counts()
        self.assertEqual(counts["D9"], 0)
        self.assertEqual(counts["D11"], 0)
        self.assertEqual(counts["S3"], 0)


if __name__ == "__main__":
    unittest.main()
