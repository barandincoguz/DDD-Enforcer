"""WP-CORE-26 — Caller-side JSON parse failure → manifest wiring tests.

Closes WP-CORE-20a: Scout/Architect/Specialist hot path uses
`client.chat(...)` + `_parse_json_response(...)` manually. Provider-side
`LLMResponse.json_failed` is False on that path; without explicit
caller-side reporting, the run manifest under-reports `json_failed_rate`.

The helper `_record_json_parse_failure_if_emitter` sets `_stage_var`
temporarily so the emitter accepts the call even when no
`with emitter.stage(...)` block is active (the production code does
NOT wrap stages — that's a WP-CORE-20b follow-up).
"""
from __future__ import annotations

import unittest

from core.architect import _record_json_parse_failure_if_emitter
from core.observability import RunManifest, StageEmitter
from core.observability.emitter import _emitter_var, _stage_var


class TestJSONParseFailureWiring(unittest.TestCase):
    def setUp(self):
        self.manifest = RunManifest()
        self.emitter = StageEmitter(self.manifest)
        self._token = _emitter_var.set(self.emitter)

    def tearDown(self):
        try:
            _emitter_var.reset(self._token)
        except Exception:
            pass

    def test_t_json_wire_scout(self):
        """T-JSON-WIRE-SCOUT: Scout parse failure bumps json_parse_failure_count."""
        _record_json_parse_failure_if_emitter(
            stage="scout",
            operation="extract_sentences_chunk_1 attempt-3",
            model_id="gemini-3.1-pro-preview",
        )
        self.assertEqual(self.manifest.llm.json_parse_failure_count, 1)
        self.assertEqual(self.manifest.llm.json_failed_total_count, 1)
        self.assertEqual(
            self.manifest.llm.json_fail_reasons.get("caller_parse"), 1
        )
        scout = self.manifest.stages.get("scout")
        self.assertIsNotNone(scout)
        assert scout is not None
        self.assertEqual(len(scout.json_parse_failures), 1)
        self.assertEqual(scout.json_parse_failures[0].operation,
                         "extract_sentences_chunk_1 attempt-3")
        self.assertEqual(scout.json_parse_failures[0].model_id,
                         "gemini-3.1-pro-preview")

    def test_t_json_wire_architect(self):
        """T-JSON-WIRE-ARCHITECT: Architect parse failure records under stage=architect."""
        _record_json_parse_failure_if_emitter(
            stage="architect",
            operation="identify_contexts attempt-2",
            model_id="gemini-3.1-pro-preview",
        )
        architect = self.manifest.stages.get("architect")
        assert architect is not None
        self.assertEqual(len(architect.json_parse_failures), 1)

    def test_t_json_wire_specialist(self):
        """T-JSON-WIRE-SPECIALIST: Specialist parse failure records under stage=specialist."""
        _record_json_parse_failure_if_emitter(
            stage="specialist",
            operation="per_context:Ordering:attempt-4",
            model_id="gemini-3.1-pro-preview",
        )
        spec = self.manifest.stages.get("specialist")
        assert spec is not None
        self.assertEqual(len(spec.json_parse_failures), 1)
        self.assertIn("Ordering", spec.json_parse_failures[0].operation)

    def test_t_json_wire_multi_stage_aggregates(self):
        """T-JSON-WIRE-MULTI: 3 calls across 3 stages → llm.json_parse_failure_count == 3."""
        _record_json_parse_failure_if_emitter(
            stage="scout", operation="op1", model_id="m1",
        )
        _record_json_parse_failure_if_emitter(
            stage="architect", operation="op2", model_id="m1",
        )
        _record_json_parse_failure_if_emitter(
            stage="specialist", operation="op3", model_id="m1",
        )
        self.assertEqual(self.manifest.llm.json_parse_failure_count, 3)
        self.assertEqual(self.manifest.llm.json_failed_total_count, 3)

    def test_t_json_wire_no_emitter_noop(self):
        """T-JSON-WIRE-NOEMITTER: no active emitter → helper is no-op, no crash."""
        # Clear emitter ContextVar
        _emitter_var.reset(self._token)
        try:
            # Should not raise.
            _record_json_parse_failure_if_emitter(
                stage="scout", operation="orphan", model_id="m1",
            )
        finally:
            self._token = _emitter_var.set(self.emitter)

    def test_t_json_wire_stage_var_reset_after_call(self):
        """T-JSON-WIRE-STAGE-RESET: _stage_var is restored after the helper call."""
        before = _stage_var.get()
        _record_json_parse_failure_if_emitter(
            stage="scout", operation="op", model_id="m1",
        )
        after = _stage_var.get()
        self.assertEqual(before, after)  # ContextVar reset


if __name__ == "__main__":
    unittest.main()
