"""WP-CORE-23 — V6 Domain Event detection false-positive guards (RED phase).

Two known false-positive paths in `_score_domain_events` + `_event_name_from_arg`:

1. ast.Name argument to publish/emit → the variable NAME (not its value) is
   recorded as an event. Example: `self.mailer.dispatch(template_name)` records
   "template_name" as a domain event.

2. Class-name suffix path: any class whose name ends in EVENT_SUFFIXES
   ("Event", "Paid", "Created", ...) emits itself as an event regardless of
   what's actually in the body. Example: `class PaidLeavePolicy:` ending in
   "Paid" emits "PaidLeavePolicy" as a domain event.

Fix: (a) `_event_name_from_arg` no longer accepts `ast.Name` — only Constant
and Attribute. (b) Suffix-name path requires class body to be event-like:
either empty (`pass`), only dataclass fields, or strict event base
classes — NOT a class with methods.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.AST.ast_model_signals import ASTModelSignalExtractor
from core.AST.ast_signal_discovery import extract_class_facts


class TestDomainEventFalsePositiveGuards(unittest.TestCase):
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

    def _event_names(self, path: str) -> list[str]:
        """Collect domain event names emitted across the file's classes."""
        results = self.extractor.extract_candidates([path])
        # extract_candidates filters out domain_events; sample via class facts.
        names: list[str] = []
        for facts in extract_class_facts(path):
            names.extend(facts.event_names)
        return names

    # ------------------------------------------------------------------
    # _event_name_from_arg ast.Name skip
    # ------------------------------------------------------------------

    def test_t_v6_name_arg_dropped(self):
        """T-V6-NAME-ARG: variable name passed to publish() is NOT an event."""
        code = """
class NotificationService:
    def __init__(self, bus):
        self.bus = bus

    def send_template(self, template_name):
        self.bus.publish(template_name)
"""
        path = self._write("notif.py", code)
        names = self._event_names(path)
        self.assertNotIn(
            "template_name", names,
            "ast.Name variable should NOT be treated as event name",
        )

    def test_t_v6_string_literal_kept(self):
        """T-V6-LITERAL: string literal passed to publish() IS an event."""
        code = """
class OrderService:
    def __init__(self, bus):
        self.bus = bus

    def place_order(self):
        self.bus.publish("OrderPlaced")
"""
        path = self._write("ord.py", code)
        names = self._event_names(path)
        self.assertIn("OrderPlaced", names)

    def test_t_v6_attribute_arg_kept(self):
        """T-V6-ATTR: ast.Attribute (Events.OrderPlaced) accepted (existing
        behavior preserved)."""
        code = """
class OrderService:
    def __init__(self, bus):
        self.bus = bus

    def place(self):
        self.bus.dispatch(Events.OrderPlaced)
"""
        path = self._write("ord2.py", code)
        names = self._event_names(path)
        self.assertIn("OrderPlaced", names)

    # ------------------------------------------------------------------
    # suffix-name path body-purity check
    # ------------------------------------------------------------------

    def test_t_v6_suffix_with_methods_dropped(self):
        """T-V6-SUFFIX-METHODS: PaidLeavePolicy ends in 'Paid' but has methods;
        must NOT be flagged as event."""
        code = """
class PaidLeavePolicy:
    def __init__(self, employee_id):
        self.employee_id = employee_id

    def approve(self, manager_id):
        return True
"""
        path = self._write("policy.py", code)
        results = self.extractor.extract_candidates([path])
        # The class has methods → not an event. extract_candidates omits
        # domain_events from the public map, so we check via class facts:
        # the suffix-name path should NOT have added this class to facts.event_names.
        for facts in extract_class_facts(path):
            if facts.name == "PaidLeavePolicy":
                # Body-purity check: NotInPolicyName.
                # The classifier will only emit event signals when name+body match.
                from core.AST.ast_signal_classification import SignalClassifier
                signals = SignalClassifier().classify(facts)
                event_signals = [s for s in signals if s.candidate_type == "domain_events"]
                self.assertEqual(
                    event_signals, [],
                    "Class with methods should NOT be flagged as event by suffix",
                )

    def test_t_v6_suffix_empty_class_kept(self):
        """T-V6-SUFFIX-EMPTY: bare `class OrderPlaced: pass` IS flagged as event."""
        code = """
class OrderPlaced:
    pass
"""
        path = self._write("evt_empty.py", code)
        from core.AST.ast_signal_classification import SignalClassifier
        for facts in extract_class_facts(path):
            if facts.name == "OrderPlaced":
                signals = SignalClassifier().classify(facts)
                event_signals = [s for s in signals if s.candidate_type == "domain_events"]
                self.assertTrue(
                    any(s.name == "OrderPlaced" for s in event_signals),
                    "Empty event-suffix class should be flagged as event",
                )

    def test_t_v6_suffix_dataclass_kept(self):
        """T-V6-SUFFIX-DATACLASS: a dataclass event payload class IS flagged."""
        code = """
from dataclasses import dataclass

@dataclass
class OrderShipped:
    order_id: str
    timestamp: str
"""
        path = self._write("evt_dc.py", code)
        from core.AST.ast_signal_classification import SignalClassifier
        for facts in extract_class_facts(path):
            if facts.name == "OrderShipped":
                signals = SignalClassifier().classify(facts)
                event_signals = [s for s in signals if s.candidate_type == "domain_events"]
                self.assertTrue(
                    any(s.name == "OrderShipped" for s in event_signals),
                    "Dataclass event class should be flagged as event",
                )


if __name__ == "__main__":
    unittest.main()
