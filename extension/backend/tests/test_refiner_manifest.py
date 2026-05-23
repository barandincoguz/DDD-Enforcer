"""WP-CORE-24 — Refiner manifest fields tests (RED→GREEN).

Asserts that the orchestrator writes refiner.cycles_used + exhausted to
the active StageEmitter manifest on both the clean path and the
exhausted path, and that the metric is absent when no emitter is in
context (CLI / standalone runs).
"""
from __future__ import annotations

import unittest

from core.observability import RunManifest, StageEmitter
from core.observability.emitter import _emitter_var
from core.orchestration.errors import RefinementExhaustedError
from core.orchestration.pipeline import _record_refiner_metrics_safely


class TestRefinerManifestMetrics(unittest.TestCase):
    def setUp(self):
        self.manifest = RunManifest()
        self.emitter = StageEmitter(self.manifest)
        self._token = _emitter_var.set(self.emitter)

    def tearDown(self):
        _emitter_var.reset(self._token)

    # ------------------------------------------------------------------

    def test_t_refiner_manifest_1_clean_writes_zero_cycles(self):
        """T-REFINER-MANIFEST-1: clean path (cycles=0) writes status=clean."""
        _record_refiner_metrics_safely(
            cycles_used=0, exhausted=False, residual_count=0, max_cycles=2,
        )
        refiner = self.manifest.stages.get("refiner")
        self.assertIsNotNone(refiner)
        assert refiner is not None  # for type checker
        self.assertEqual(refiner.status, "clean")
        self.assertEqual(refiner.metrics["cycles_used"], 0)
        self.assertEqual(refiner.metrics["exhausted"], False)
        self.assertEqual(refiner.metrics["exhausted_residual_count"], 0)
        self.assertEqual(refiner.metrics["max_cycles"], 2)

    def test_t_refiner_manifest_2_exhausted_records_residual(self):
        """T-REFINER-MANIFEST-2: exhausted path writes status=exhausted + residual count."""
        _record_refiner_metrics_safely(
            cycles_used=2, exhausted=True, residual_count=3, max_cycles=2,
        )
        refiner = self.manifest.stages.get("refiner")
        assert refiner is not None
        self.assertEqual(refiner.status, "exhausted")
        self.assertEqual(refiner.metrics["cycles_used"], 2)
        self.assertEqual(refiner.metrics["exhausted"], True)
        self.assertEqual(refiner.metrics["exhausted_residual_count"], 3)

    def test_t_refiner_manifest_3_no_emitter_is_silent_noop(self):
        """T-REFINER-MANIFEST-3: without emitter in context, the helper is no-op."""
        # Reset emitter, call helper, restore.
        _emitter_var.reset(self._token)
        try:
            # Should not raise.
            _record_refiner_metrics_safely(
                cycles_used=1, exhausted=False, residual_count=0,
            )
            # No manifest to assert against — helper silently dropped.
        finally:
            self._token = _emitter_var.set(self.emitter)


class TestRefinementExhaustedErrorCyclesAttempted(unittest.TestCase):
    def test_t_refiner_error_carries_cycles(self):
        """T-REFINER-ERR-CYCLES: RefinementExhaustedError exposes cycles_attempted."""
        err = RefinementExhaustedError(issues=["x", "y"], cycles_attempted=2)
        self.assertEqual(err.cycles_attempted, 2)
        self.assertEqual(len(err.issues), 2)

    def test_t_refiner_error_default_cycles_zero(self):
        """T-REFINER-ERR-DEFAULT: cycles_attempted default is 0 for backward compat."""
        err = RefinementExhaustedError(issues=[])
        self.assertEqual(err.cycles_attempted, 0)


if __name__ == "__main__":
    unittest.main()
