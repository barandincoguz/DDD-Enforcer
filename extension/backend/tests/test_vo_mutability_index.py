"""WP-CORE-27a — AST-derived `is_mutable_in_code` population.

WP-CORE-27 shipped the D9 verifier check
(`check_d9_value_object_mutability_consistency`) but left the
`ValueObject.is_mutable_in_code` field defaulting to `False` — the AST
cross-reference that would set the field was deferred as a follow-up.
Without that wiring, D9 was vacuously satisfied for every LLM-claimed
Value Object.

This WP delivers two pieces:

1. **`core/AST/mutability_index.py`** — a pure helper
   `build_mutability_index(python_files)` that walks every class in the
   workspace and returns `{class_name_lower: bool}`.  `True` means the
   class has at least one mutation method (set_*, add_*, change_*, ...);
   `False` is the conservative default (no mutation methods seen).

2. **`SignalEnricher.enrich_model`** gains an optional
   `mutability_index` parameter.  After the main merge loop, each
   LLM-claimed VO whose name appears in the index gets
   `is_mutable_in_code = index[name.lower()]`.  Items not in the index
   stay at the schema default.

The wiring funnel is `ASTModelSignalExtractor.enrich_domain_model`:
build the index from the workspace's Python files, pass it through.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.AST.ast_signal_enrichment import SignalEnricher
from core.AST.ast_signal_types import CandidateSignal, SourceRef
from core.AST.mutability_index import build_mutability_index


class _FixtureWorkspace(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace, ignore_errors=True)

    def _write(self, rel_path: str, content: str) -> str:
        full = os.path.join(self.workspace, rel_path)
        os.makedirs(os.path.dirname(full) or self.workspace, exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(content)
        return full


# =============================================================================
# build_mutability_index
# =============================================================================


class TestBuildMutabilityIndex(_FixtureWorkspace):
    def test_t_27a_1_class_with_set_method_flagged_mutable(self) -> None:
        """T-27a-1: a class with a `set_foo` mutation method is flagged
        mutable in the index."""
        path = self._write(
            "amount.py",
            """
class Amount:
    def __init__(self, value):
        self.value = value

    def set_value(self, new_value):
        self.value = new_value
""",
        )
        index = build_mutability_index([path])
        self.assertTrue(index.get("amount"))

    def test_t_27a_2_class_without_mutation_methods_flagged_false(self) -> None:
        """T-27a-2: a class with no mutation-style methods is flagged False."""
        path = self._write(
            "money.py",
            """
class Money:
    def __init__(self, value, currency):
        self.value = value
        self.currency = currency
""",
        )
        index = build_mutability_index([path])
        self.assertFalse(index.get("money", False))

    def test_t_27a_3_frozen_dataclass_not_mutable(self) -> None:
        """T-27a-3: a frozen dataclass without mutation methods is False."""
        path = self._write(
            "address.py",
            """
from dataclasses import dataclass

@dataclass(frozen=True)
class Address:
    street: str
    city: str
""",
        )
        index = build_mutability_index([path])
        self.assertFalse(index.get("address", False))

    def test_t_27a_4_parse_error_file_silently_skipped(self) -> None:
        """T-27a-4: a malformed .py file does not raise; just contributes
        nothing to the index."""
        good = self._write(
            "good.py",
            "class Good:\n    def set_x(self, v):\n        self.x = v\n",
        )
        bad = self._write("bad.py", "this is not valid python :::\n")
        index = build_mutability_index([good, bad])
        self.assertTrue(index.get("good"))
        # The bad file contributes nothing.
        self.assertNotIn("bad", index)

    def test_t_27a_5_mutable_wins_over_non_mutable_duplicate(self) -> None:
        """T-27a-5: if two files define a class with the same name, the
        index entry collapses to True iff ANY definition is mutable."""
        a = self._write(
            "a.py",
            "class Order:\n    def __init__(self, x): self.x = x\n",
        )
        b = self._write(
            "b.py",
            "class Order:\n    def add_item(self, item): self.items.append(item)\n",
        )
        index = build_mutability_index([a, b])
        self.assertTrue(index.get("order"))


# =============================================================================
# SignalEnricher.enrich_model populates is_mutable_in_code
# =============================================================================


def _vo_signal(name: str) -> CandidateSignal:
    """Build a minimal value-object CandidateSignal for enrichment input."""
    return CandidateSignal(
        candidate_type="value_objects",
        name=name,
        description=f"AST-discovered value object {name}",
        confidence=0.7,
        reasons=["frozen marker"],
        sources=[SourceRef(file="x.py", line=1, rule="AST_VALUE_OBJECT", evidence=f"class {name}")],
        attributes=["value", "currency"],
    )


def _model_with_vo(name: str) -> dict:
    """Minimal model_data dict carrying one LLM-claimed VO.

    `evidence_sentence_indices=[0]` satisfies the
    `_check_traceability` guard (Phase D3) so the test focuses on the
    is_mutable_in_code field population without having to wire a full
    SRS grounding fixture.
    """
    return {
        "bounded_contexts": [
            {
                "context_name": "CoreDomain",
                "description": "test",
                "ubiquitous_language": {
                    "entities": [],
                    "value_objects": [
                        {
                            "name": name,
                            "attributes": ["value", "currency"],
                            "description": "test VO",
                            "confidence": 0.8,
                            "evidence_sentence_indices": [0],
                        }
                    ],
                    "services": [],
                    "aggregates": [],
                    "domain_events": [],
                },
            }
        ],
    }


class TestEnrichPopulatesMutability(unittest.TestCase):
    def test_t_27a_6_llm_vo_with_mutable_backing_class_flagged_true(self) -> None:
        """T-27a-6: LLM claims `Amount` as a VO; the AST mutability index
        marks `amount` as mutable -> enrich_model sets
        is_mutable_in_code=True on the merged dict."""
        enricher = SignalEnricher(workspace_path="")
        signals = [_vo_signal("Amount")]
        model = _model_with_vo("Amount")
        result = enricher.enrich_model(
            model, signals, mutability_index={"amount": True},
        )
        vo = result["bounded_contexts"][0]["ubiquitous_language"]["value_objects"][0]
        self.assertTrue(vo.get("is_mutable_in_code"))

    def test_t_27a_7_llm_vo_with_immutable_backing_class_flagged_false(self) -> None:
        """T-27a-7: a VO whose backing class is immutable -> is_mutable_in_code False."""
        enricher = SignalEnricher(workspace_path="")
        signals = [_vo_signal("Money")]
        model = _model_with_vo("Money")
        result = enricher.enrich_model(
            model, signals, mutability_index={"money": False},
        )
        vo = result["bounded_contexts"][0]["ubiquitous_language"]["value_objects"][0]
        self.assertFalse(vo.get("is_mutable_in_code", False))

    def test_t_27a_8_no_index_supplied_field_defaults_false(self) -> None:
        """T-27a-8: backward compat — enrich_model called without the
        mutability_index keyword (legacy callers) leaves the field at
        the schema default of False."""
        enricher = SignalEnricher(workspace_path="")
        signals = [_vo_signal("Amount")]
        model = _model_with_vo("Amount")
        result = enricher.enrich_model(model, signals)
        vo = result["bounded_contexts"][0]["ubiquitous_language"]["value_objects"][0]
        # The field should either be absent (schema default applies on
        # Pydantic re-validation) or explicitly False — both are OK.
        self.assertFalse(vo.get("is_mutable_in_code", False))


if __name__ == "__main__":
    unittest.main()
