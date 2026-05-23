"""Tests for core.metrics — T-01b-B-1 through T-01b-B-12.

TDD RED first: all tests written before core/metrics.py exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers to build minimal PaperRunManifest and Violation objects
# ---------------------------------------------------------------------------

from core.run_manifest import PaperRunManifest, Violation, EMPTY_TREE_SHA256


def _violation(
    violation_type: str = "ubiquitous_language_drift",
    location: str = "myapp/order.py::Order",
    severity: str = "ERROR",
    message: str = "drift detected",
) -> Violation:
    return Violation(
        violation_type=violation_type,
        location=location,
        severity=severity,  # type: ignore[arg-type]
        message=message,
    )


def _manifest(violations: List[Violation] | None = None) -> PaperRunManifest:
    return PaperRunManifest(
        run_id="test-run-001",
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        srs_path="/inputs/SRS.docx",
        srs_sha256="a" * 64,
        violations=violations or [],
    )


# ---------------------------------------------------------------------------
# Imports under test (will ImportError until core/metrics.py exists)
# ---------------------------------------------------------------------------

from core.metrics import (  # noqa: E402
    JudgeVerdict,
    JudgeVerdictEntry,
    JudgeMissedExpectation,
    PrecisionRecallF1,
    load_judge_verdict,
    compute_metrics,
    aggregate_scoreline,
)


# ---------------------------------------------------------------------------
# T-01b-B-1: JudgeVerdict instantiation + field validation
# ---------------------------------------------------------------------------


class TestJudgeVerdictSchema:
    """T-01b-B-1: JudgeVerdict provider Literal accepts gemini/ollama/human; rejects others."""

    def test_valid_gemini_provider(self) -> None:
        v = JudgeVerdict(
            run_id="r1",
            judge_model_id="gemini-3.1-pro-preview",
            judge_provider="gemini",
        )
        assert v.run_id == "r1"
        assert v.judge_provider == "gemini"

    def test_valid_ollama_provider(self) -> None:
        v = JudgeVerdict(
            run_id="r1",
            judge_model_id="gpt-oss:120b-cloud",
            judge_provider="ollama",
        )
        assert v.judge_provider == "ollama"

    def test_valid_human_provider(self) -> None:
        v = JudgeVerdict(
            run_id="r1",
            judge_model_id="human-reviewer",
            judge_provider="human",
        )
        assert v.judge_provider == "human"

    def test_invalid_provider_openai(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(
                run_id="r1",
                judge_model_id="gpt-4",
                judge_provider="openai",  # type: ignore[arg-type]
            )

    def test_default_empty_entries_and_missed(self) -> None:
        v = JudgeVerdict(run_id="r1", judge_model_id="x", judge_provider="human")
        assert v.entries == []
        assert v.missed == []


# ---------------------------------------------------------------------------
# T-01b-B-2: JudgeVerdictEntry.label Literal validation
# ---------------------------------------------------------------------------


class TestJudgeVerdictEntryLabel:
    """T-01b-B-2: label accepts true_positive, false_positive; rejects others."""

    def test_true_positive_accepted(self) -> None:
        e = JudgeVerdictEntry(
            violation_type="drift",
            location="file.py",
            label="true_positive",
        )
        assert e.label == "true_positive"

    def test_false_positive_accepted(self) -> None:
        e = JudgeVerdictEntry(
            violation_type="drift",
            location="file.py",
            label="false_positive",
        )
        assert e.label == "false_positive"

    def test_false_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdictEntry(
                violation_type="drift",
                location="file.py",
                label="false_negative",  # type: ignore[arg-type]
            )

    def test_ignored_label_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdictEntry(
                violation_type="drift",
                location="file.py",
                label="ignored",  # type: ignore[arg-type]
            )

    def test_notes_is_optional(self) -> None:
        e = JudgeVerdictEntry(
            violation_type="drift",
            location="file.py",
            label="true_positive",
        )
        assert e.notes is None


# ---------------------------------------------------------------------------
# T-01b-B-3: compute_metrics happy path (3 predictions, 2 TP + 1 FP, no FN)
# ---------------------------------------------------------------------------


class TestComputeMetricsHappyPath:
    """T-01b-B-3: 3 predictions same type, 2 TP + 1 FP → P=2/3, R=1.0, F1=0.8."""

    def test_precision_recall_f1(self) -> None:
        vtype = "ubiquitous_language_drift"
        violations = [
            _violation(violation_type=vtype, location="f.py::A"),
            _violation(violation_type=vtype, location="f.py::B"),
            _violation(violation_type=vtype, location="f.py::C"),
        ]
        manifest = _manifest(violations=violations)

        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="gemini-3.1-pro-preview",
            judge_provider="gemini",
            entries=[
                JudgeVerdictEntry(violation_type=vtype, location="f.py::A", label="true_positive"),
                JudgeVerdictEntry(violation_type=vtype, location="f.py::B", label="true_positive"),
                JudgeVerdictEntry(violation_type=vtype, location="f.py::C", label="false_positive"),
            ],
            missed=[],
        )

        results = compute_metrics(manifest, verdict)
        # One per-type + one aggregate
        per_type = [r for r in results if r.violation_type != "__aggregate__"]
        assert len(per_type) == 1
        score = per_type[0]
        assert score.true_positives == 2
        assert score.false_positives == 1
        assert score.false_negatives == 0
        assert abs(score.precision - 2 / 3) < 1e-9
        assert abs(score.recall - 1.0) < 1e-9
        expected_f1 = 2 * (2 / 3) * 1.0 / (2 / 3 + 1.0)
        assert abs(score.f1 - expected_f1) < 1e-9

    def test_aggregate_entry_present(self) -> None:
        vtype = "ubiquitous_language_drift"
        violations = [_violation(violation_type=vtype, location="f.py::A")]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[JudgeVerdictEntry(violation_type=vtype, location="f.py::A", label="true_positive")],
        )
        results = compute_metrics(manifest, verdict)
        aggregates = [r for r in results if r.violation_type == "__aggregate__"]
        assert len(aggregates) == 1


# ---------------------------------------------------------------------------
# T-01b-B-4: compute_metrics with false negatives
# ---------------------------------------------------------------------------


class TestComputeMetricsFalseNegatives:
    """T-01b-B-4: 1 TP + 2 FN → P=1.0, R=1/3, F1≈0.5."""

    def test_with_missed_expectations(self) -> None:
        vtype = "aggregate_boundary_violation"
        violations = [_violation(violation_type=vtype, location="a.py::X")]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype, location="a.py::X", label="true_positive"),
            ],
            missed=[
                JudgeMissedExpectation(violation_type=vtype, location="b.py::Y"),
                JudgeMissedExpectation(violation_type=vtype, location="c.py::Z"),
            ],
        )
        results = compute_metrics(manifest, verdict)
        per_type = [r for r in results if r.violation_type != "__aggregate__"]
        assert len(per_type) == 1
        score = per_type[0]
        assert score.true_positives == 1
        assert score.false_positives == 0
        assert score.false_negatives == 2
        assert abs(score.precision - 1.0) < 1e-9
        assert abs(score.recall - 1 / 3) < 1e-9
        expected_f1 = 2 * 1.0 * (1 / 3) / (1.0 + 1 / 3)
        assert abs(score.f1 - expected_f1) < 1e-9


# ---------------------------------------------------------------------------
# T-01b-B-5: Multi-type with correct per-type and micro-aggregate
# ---------------------------------------------------------------------------


class TestComputeMetricsMultiType:
    """T-01b-B-5: 2 types, each with own TP/FP/FN; aggregate is micro-average."""

    def test_multi_type_scorelines(self) -> None:
        # Type A: 2 TP, 1 FP, 0 FN
        # Type B: 1 TP, 0 FP, 1 FN
        vtype_a = "type_a"
        vtype_b = "type_b"
        violations = [
            _violation(violation_type=vtype_a, location="f.py::A1"),
            _violation(violation_type=vtype_a, location="f.py::A2"),
            _violation(violation_type=vtype_a, location="f.py::A3"),
            _violation(violation_type=vtype_b, location="f.py::B1"),
        ]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A1", label="true_positive"),
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A2", label="true_positive"),
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A3", label="false_positive"),
                JudgeVerdictEntry(violation_type=vtype_b, location="f.py::B1", label="true_positive"),
            ],
            missed=[
                JudgeMissedExpectation(violation_type=vtype_b, location="f.py::B2"),
            ],
        )
        results = compute_metrics(manifest, verdict)
        per_type = {r.violation_type: r for r in results if r.violation_type != "__aggregate__"}
        assert set(per_type.keys()) == {vtype_a, vtype_b}

        sa = per_type[vtype_a]
        assert sa.true_positives == 2
        assert sa.false_positives == 1
        assert sa.false_negatives == 0
        assert abs(sa.precision - 2 / 3) < 1e-9
        assert abs(sa.recall - 1.0) < 1e-9

        sb = per_type[vtype_b]
        assert sb.true_positives == 1
        assert sb.false_positives == 0
        assert sb.false_negatives == 1
        assert abs(sb.precision - 1.0) < 1e-9
        assert abs(sb.recall - 0.5) < 1e-9

    def test_aggregate_is_micro_not_macro(self) -> None:
        # Same setup: aggregate should be micro (sum counts first)
        vtype_a = "type_a"
        vtype_b = "type_b"
        violations = [
            _violation(violation_type=vtype_a, location="f.py::A1"),
            _violation(violation_type=vtype_a, location="f.py::A2"),
            _violation(violation_type=vtype_a, location="f.py::A3"),
            _violation(violation_type=vtype_b, location="f.py::B1"),
        ]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A1", label="true_positive"),
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A2", label="true_positive"),
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A3", label="false_positive"),
                JudgeVerdictEntry(violation_type=vtype_b, location="f.py::B1", label="true_positive"),
            ],
            missed=[
                JudgeMissedExpectation(violation_type=vtype_b, location="f.py::B2"),
            ],
        )
        results = compute_metrics(manifest, verdict)
        agg = next(r for r in results if r.violation_type == "__aggregate__")
        # Micro: TP=3, FP=1, FN=1
        assert agg.true_positives == 3
        assert agg.false_positives == 1
        assert agg.false_negatives == 1
        # P = 3/(3+1) = 0.75, R = 3/(3+1) = 0.75, F1 = 0.75
        assert abs(agg.precision - 3 / 4) < 1e-9
        assert abs(agg.recall - 3 / 4) < 1e-9


# ---------------------------------------------------------------------------
# T-01b-B-6: Division-by-zero guard (all-FP case)
# ---------------------------------------------------------------------------


class TestDivisionByZero:
    """T-01b-B-6: all-FP (0 TP, 0 FN) → P=0, R=0, F1=0; no exception, no NaN."""

    def test_all_fp_no_nan_no_exception(self) -> None:
        vtype = "drift"
        violations = [_violation(violation_type=vtype, location="f.py::X")]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype, location="f.py::X", label="false_positive"),
            ],
            missed=[],
        )
        results = compute_metrics(manifest, verdict)
        per_type = [r for r in results if r.violation_type != "__aggregate__"]
        score = per_type[0]
        assert score.true_positives == 0
        assert score.false_positives == 1
        assert score.false_negatives == 0
        # Division by zero: TP+FP=1, P=0.0; TP+FN=0, R=0.0; P+R=0, F1=0.0
        assert score.precision == 0.0
        assert score.recall == 0.0
        assert score.f1 == 0.0
        import math
        assert not math.isnan(score.precision)
        assert not math.isnan(score.recall)
        assert not math.isnan(score.f1)


# ---------------------------------------------------------------------------
# T-01b-B-7: ValueError on missing judge entry
# ---------------------------------------------------------------------------


class TestMissingJudgeEntryRaisesValueError:
    """T-01b-B-7: predicted violation with no matching judge entry raises ValueError."""

    def test_missing_entry_raises(self) -> None:
        vtype = "drift"
        violations = [
            _violation(violation_type=vtype, location="f.py::A"),
            _violation(violation_type=vtype, location="f.py::B"),  # no entry
        ]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype, location="f.py::A", label="true_positive"),
                # f.py::B intentionally absent
            ],
        )
        with pytest.raises(ValueError, match="no judge entry"):
            compute_metrics(manifest, verdict)


# ---------------------------------------------------------------------------
# T-01b-B-8: load_judge_verdict round-trip
# ---------------------------------------------------------------------------


class TestLoadJudgeVerdictRoundTrip:
    """T-01b-B-8: write via model_dump_json, read back with load_judge_verdict."""

    def test_round_trip(self, tmp_path: Path) -> None:
        verdict = JudgeVerdict(
            run_id="run-xyz",
            judge_model_id="gemini-3.1-pro-preview",
            judge_provider="gemini",
            entries=[
                JudgeVerdictEntry(
                    violation_type="drift",
                    location="a.py::B",
                    label="true_positive",
                    notes="looks right",
                ),
            ],
            missed=[
                JudgeMissedExpectation(
                    violation_type="drift",
                    location="b.py::C",
                    notes="model missed this",
                ),
            ],
        )
        path = tmp_path / "run-xyz.judge.json"
        path.write_text(verdict.model_dump_json(indent=2), encoding="utf-8")

        loaded = load_judge_verdict(path)
        assert loaded.run_id == verdict.run_id
        assert loaded.judge_provider == verdict.judge_provider
        assert len(loaded.entries) == 1
        assert loaded.entries[0].label == "true_positive"
        assert len(loaded.missed) == 1
        assert loaded.missed[0].location == "b.py::C"


# ---------------------------------------------------------------------------
# T-01b-B-9: load_judge_verdict raises ValidationError on broken file
# ---------------------------------------------------------------------------


class TestLoadJudgeVerdictBrokenFile:
    """T-01b-B-9: load_judge_verdict raises ValidationError when run_id is missing."""

    def test_missing_required_field(self, tmp_path: Path) -> None:
        broken = {"judge_model_id": "x", "judge_provider": "human"}
        path = tmp_path / "broken.judge.json"
        path.write_text(json.dumps(broken), encoding="utf-8")

        with pytest.raises(ValidationError):
            load_judge_verdict(path)


# ---------------------------------------------------------------------------
# T-01b-B-10: Type-only-FN case (violation type only in verdict.missed)
# ---------------------------------------------------------------------------


class TestTypeOnlyFalseNegative:
    """T-01b-B-10: violation type only in verdict.missed gets a per-type scoreline."""

    def test_fn_only_type_present_in_output(self) -> None:
        vtype_a = "known_type"
        fn_only_type = "fn_only_type"
        violations = [_violation(violation_type=vtype_a, location="f.py::A")]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A", label="true_positive"),
            ],
            missed=[
                JudgeMissedExpectation(violation_type=fn_only_type, location="b.py::X"),
                JudgeMissedExpectation(violation_type=fn_only_type, location="b.py::Y"),
                JudgeMissedExpectation(violation_type=fn_only_type, location="b.py::Z"),
            ],
        )
        results = compute_metrics(manifest, verdict)
        per_type = {r.violation_type: r for r in results if r.violation_type != "__aggregate__"}
        assert fn_only_type in per_type
        fn_score = per_type[fn_only_type]
        assert fn_score.true_positives == 0
        assert fn_score.false_positives == 0
        assert fn_score.false_negatives == 3
        assert fn_score.precision == 0.0
        assert fn_score.recall == 0.0
        assert fn_score.f1 == 0.0


# ---------------------------------------------------------------------------
# T-01b-B-11: Micro-averaging vs macro-averaging
# ---------------------------------------------------------------------------


class TestMicroAveraging:
    """T-01b-B-11: micro aggregate != mean of per-type F1 when types are unequal."""

    def test_micro_not_macro(self) -> None:
        # Type A: TP=1, FP=0, FN=0 → P=1, R=1, F1=1
        # Type B: TP=0, FP=1, FN=0 → P=0, R=0, F1=0
        # Macro mean of F1 = 0.5
        # Micro: TP=1, FP=1, FN=0 → P=1/2, R=1/1, F1=2/3
        vtype_a = "type_a"
        vtype_b = "type_b"
        violations = [
            _violation(violation_type=vtype_a, location="f.py::A1"),
            _violation(violation_type=vtype_b, location="f.py::B1"),
        ]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype_a, location="f.py::A1", label="true_positive"),
                JudgeVerdictEntry(violation_type=vtype_b, location="f.py::B1", label="false_positive"),
            ],
            missed=[],
        )
        results = compute_metrics(manifest, verdict)
        agg = next(r for r in results if r.violation_type == "__aggregate__")
        # Micro: TP=1, FP=1, FN=0
        assert agg.true_positives == 1
        assert agg.false_positives == 1
        assert agg.false_negatives == 0
        # P = 1/2, R = 1/1 = 1, F1 = 2*(0.5*1)/(0.5+1) = 2/3
        assert abs(agg.precision - 0.5) < 1e-9
        assert abs(agg.recall - 1.0) < 1e-9
        expected_f1 = 2 * 0.5 * 1.0 / (0.5 + 1.0)
        assert abs(agg.f1 - expected_f1) < 1e-9
        # Not 0.5 (macro mean)
        assert agg.f1 != 0.5


# ---------------------------------------------------------------------------
# T-01b-B-12: aggregate_scoreline ignores __aggregate__ entries (idempotency)
# ---------------------------------------------------------------------------


class TestAggregateScourelineIdempotency:
    """T-01b-B-12: aggregate_scoreline ignores input entries with violation_type == '__aggregate__'."""

    def test_excludes_aggregate_entries(self) -> None:
        per_type = [
            PrecisionRecallF1(
                violation_type="type_a",
                true_positives=2,
                false_positives=1,
                false_negatives=0,
                precision=2 / 3,
                recall=1.0,
                f1=0.8,
            ),
            PrecisionRecallF1(
                violation_type="__aggregate__",
                true_positives=999,
                false_positives=999,
                false_negatives=999,
                precision=0.0,
                recall=0.0,
                f1=0.0,
            ),
        ]
        result = aggregate_scoreline(per_type)
        # Should only count type_a (TP=2, FP=1, FN=0), ignore __aggregate__ entry
        assert result.violation_type == "__aggregate__"
        assert result.true_positives == 2
        assert result.false_positives == 1
        assert result.false_negatives == 0

    def test_calling_twice_stable(self) -> None:
        """Calling aggregate_scoreline on its own output produces the same result."""
        per_type = [
            PrecisionRecallF1(
                violation_type="drift",
                true_positives=3,
                false_positives=1,
                false_negatives=2,
                precision=0.75,
                recall=0.6,
                f1=0.667,
            ),
        ]
        first = aggregate_scoreline(per_type)
        second = aggregate_scoreline([*per_type, first])
        assert second.true_positives == first.true_positives
        assert second.false_positives == first.false_positives
        assert second.false_negatives == first.false_negatives


# ---------------------------------------------------------------------------
# T-01b-B-13: Duplicate prediction raises ValueError
# ---------------------------------------------------------------------------


class TestDuplicatePrediction:
    """T-01b-B-13: two manifest violations with the same (violation_type, location) raise ValueError."""

    def test_T01b_B13_duplicate_prediction_raises(self) -> None:
        vtype = "V1"
        location = "f.py::A"
        violations = [
            _violation(violation_type=vtype, location=location),
            _violation(violation_type=vtype, location=location),  # duplicate
        ]
        manifest = _manifest(violations=violations)
        verdict = JudgeVerdict(
            run_id="test-run-001",
            judge_model_id="x",
            judge_provider="human",
            entries=[
                JudgeVerdictEntry(violation_type=vtype, location=location, label="true_positive"),
            ],
        )
        with pytest.raises(ValueError, match=r"duplicate prediction"):
            compute_metrics(manifest, verdict)


# ---------------------------------------------------------------------------
# T-01b-B-14: run_id mismatch raises ValueError
# ---------------------------------------------------------------------------


class TestRunIdMismatch:
    """T-01b-B-14: verdict.run_id != manifest.run_id raises ValueError."""

    def test_T01b_B14_run_id_mismatch_raises(self) -> None:
        manifest = _manifest()  # run_id="test-run-001"
        verdict = JudgeVerdict(
            run_id="run_B",  # intentionally different
            judge_model_id="x",
            judge_provider="human",
        )
        with pytest.raises(ValueError, match=r"does not match"):
            compute_metrics(manifest, verdict)
