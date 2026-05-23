"""WP-CORE-30 — Refiner cycle_history + flapping detection tests.

Adds per-cycle issue snapshot to RefinementExhaustedError and a
`_detect_flapping` helper that compares the last two cycles' issue
signatures. The flapped flag flows to the run manifest via
`_record_refiner_metrics_safely(flapped=..., cycle_history_len=...)`.
"""
from __future__ import annotations

import unittest

from core.observability import RunManifest, StageEmitter
from core.observability.emitter import _emitter_var
from core.orchestration.errors import RefinementExhaustedError
from core.orchestration.pipeline import (
    _detect_flapping,
    _flapping_signature,
    _record_refiner_metrics_safely,
)
from core.refiner.loop import refine_until_clean
from core.verifier.types import IssueSeverity, VerifierIssue, VerifierResult


def _issue(loc: str, itype: str = "missing_evidence") -> VerifierIssue:
    return VerifierIssue(
        stage="specialist",
        location=loc,
        issue_type=itype,
        severity=IssueSeverity.ERROR,
        message=f"problem at {loc}",
    )


# ---------------------------------------------------------------------------
# _flapping_signature & _detect_flapping
# ---------------------------------------------------------------------------

class TestFlappingSignature(unittest.TestCase):
    def test_t_sig_1_order_independent(self):
        """T-SIG-1: signature is invariant to issue ordering."""
        a = [_issue("loc-a"), _issue("loc-b")]
        b = [_issue("loc-b"), _issue("loc-a")]
        self.assertEqual(_flapping_signature(a), _flapping_signature(b))

    def test_t_sig_2_type_distinguishes(self):
        """T-SIG-2: different issue_type at same location → different sig."""
        a = [_issue("loc-a", "missing_evidence")]
        b = [_issue("loc-a", "value_object_mutable")]
        self.assertNotEqual(_flapping_signature(a), _flapping_signature(b))

    def test_t_sig_3_empty_issues(self):
        """T-SIG-3: empty list → empty tuple signature."""
        self.assertEqual(_flapping_signature([]), ())


class TestDetectFlapping(unittest.TestCase):
    def test_t_flap_1_short_history_false(self):
        """T-FLAP-1: history < 2 → never flapping."""
        self.assertFalse(_detect_flapping([]))
        self.assertFalse(_detect_flapping([[_issue("x")]]))

    def test_t_flap_2_same_set_flap(self):
        """T-FLAP-2: last 2 cycles identical issue sets → flap."""
        sig_set = [_issue("loc-a"), _issue("loc-b")]
        history = [sig_set, sig_set]
        self.assertTrue(_detect_flapping(history))

    def test_t_flap_3_different_sets_no_flap(self):
        """T-FLAP-3: last 2 cycles differ → no flap."""
        history = [
            [_issue("loc-a")],
            [_issue("loc-b")],
        ]
        self.assertFalse(_detect_flapping(history))

    def test_t_flap_4_reorder_still_flaps(self):
        """T-FLAP-4: same issues different order across cycles → still flapping."""
        history = [
            [_issue("a"), _issue("b")],
            [_issue("b"), _issue("a")],
        ]
        self.assertTrue(_detect_flapping(history))


# ---------------------------------------------------------------------------
# RefinementExhaustedError.cycle_history
# ---------------------------------------------------------------------------

class TestRefinementExhaustedHistory(unittest.TestCase):
    def test_t_err_history_default_empty(self):
        """T-ERR-HISTORY-DEFAULT: cycle_history defaults to empty list."""
        err = RefinementExhaustedError(issues=[])
        self.assertEqual(err.cycle_history, [])

    def test_t_err_history_passthrough(self):
        """T-ERR-HISTORY-PASS: cycle_history stored as provided."""
        history = [[_issue("a")], [_issue("a")]]
        err = RefinementExhaustedError(
            issues=[_issue("a")],
            cycles_attempted=2,
            cycle_history=history,
        )
        self.assertEqual(len(err.cycle_history), 2)


# ---------------------------------------------------------------------------
# refine_until_clean wires cycle_history into the exception
# ---------------------------------------------------------------------------

class TestRefineUntilCleanHistory(unittest.TestCase):
    def test_t_loop_history_captured(self):
        """T-LOOP-HISTORY: refine_until_clean fills cycle_history on exhaust."""
        # Verifier always returns same issue → flap + exhaust.
        persistent_issue = _issue("specialist:Ordering.entities[0]")
        result = VerifierResult(ok=False, issues=[persistent_issue])

        def verifier_fn(_output):
            return result

        def stage_runner(_prev, _result):
            return _prev  # no-op runner; issue persists

        with self.assertRaises(RefinementExhaustedError) as cm:
            refine_until_clean(
                stage_name="specialist",
                initial_output={},
                stage_runner=stage_runner,
                verifier=verifier_fn,
                max_cycles=2,
            )
        err = cm.exception
        self.assertEqual(err.cycles_attempted, 2)
        # Two cycles, both with the same issue set → flap-detectable.
        self.assertEqual(len(err.cycle_history), 2)
        self.assertTrue(_detect_flapping(err.cycle_history))


# ---------------------------------------------------------------------------
# Manifest fields wired
# ---------------------------------------------------------------------------

class TestRefinerManifestFlappingFields(unittest.TestCase):
    def setUp(self):
        self.manifest = RunManifest()
        self.emitter = StageEmitter(self.manifest)
        self._token = _emitter_var.set(self.emitter)

    def tearDown(self):
        try:
            _emitter_var.reset(self._token)
        except Exception:
            pass

    def test_t_manifest_flap_true(self):
        """T-MANIFEST-FLAP-TRUE: flapped=True surfaces in manifest."""
        _record_refiner_metrics_safely(
            cycles_used=2,
            exhausted=True,
            residual_count=3,
            max_cycles=2,
            flapped=True,
            cycle_history_len=2,
        )
        refiner = self.manifest.stages["refiner"]
        self.assertTrue(refiner.metrics["flapped"])
        self.assertEqual(refiner.metrics["cycle_history_len"], 2)

    def test_t_manifest_flap_false(self):
        """T-MANIFEST-FLAP-FALSE: flapped=False on clean exit."""
        _record_refiner_metrics_safely(
            cycles_used=1,
            exhausted=False,
            residual_count=0,
            max_cycles=2,
            flapped=False,
            cycle_history_len=1,
        )
        refiner = self.manifest.stages["refiner"]
        self.assertFalse(refiner.metrics["flapped"])
        self.assertEqual(refiner.metrics["cycle_history_len"], 1)


if __name__ == "__main__":
    unittest.main()
