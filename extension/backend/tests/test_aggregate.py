"""Tests for core/aggregate.py — WP-01b Task C.

TDD RED phase: all 14 tests written BEFORE core/aggregate.py exists.

T-01b-C-1  through T-01b-C-14 per spec.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from core.metrics import (
    JudgeMissedExpectation,
    JudgeVerdict,
    JudgeVerdictEntry,
    PrecisionRecallF1,
)
from core.run_manifest import EMPTY_TREE_SHA256, PaperRunManifest


# ---------------------------------------------------------------------------
# Helpers to build fake manifests and verdicts
# ---------------------------------------------------------------------------

_FAKE_SRS_SHA = "a" * 64  # 64-char hex string


def _make_manifest(
    run_id: str,
    model_id: str = "gemini-3.1-flash-lite",
    srs_path: str = "inputs/SRS_D1.docx",
    pipeline: str = "P1",
    violations=None,
) -> PaperRunManifest:
    return PaperRunManifest(
        run_id=run_id,
        pipeline=pipeline,  # type: ignore[arg-type]
        model_id=model_id,
        provider="gemini",
        srs_path=srs_path,
        srs_sha256=_FAKE_SRS_SHA,
        violations=violations or [],
    )


def _make_verdict(
    run_id: str,
    entries: List[JudgeVerdictEntry] | None = None,
    missed: List[JudgeMissedExpectation] | None = None,
) -> JudgeVerdict:
    return JudgeVerdict(
        run_id=run_id,
        entries=entries or [],
        missed=missed or [],
        judge_model_id="human",
        judge_provider="human",
    )


def _prf1(vtype: str, p: float, r: float, f: float) -> PrecisionRecallF1:
    return PrecisionRecallF1(violation_type=vtype, precision=p, recall=r, f1=f)


# ---------------------------------------------------------------------------
# T-01b-C-1: summarize_metric happy-path N=5
# ---------------------------------------------------------------------------


def test_c1_summarize_metric_n5():
    from core.aggregate import summarize_metric

    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = summarize_metric(values, "precision")

    assert result.metric_name == "precision"
    assert result.n_samples == 5
    assert abs(result.mean - 0.7) < 1e-9
    # Sample std ddof=1
    expected_std = statistics.stdev(values)
    assert abs(result.std - expected_std) < 1e-9
    # Q1 = 0.6, Q3 = 0.8 (numpy linear interp on 5 samples)
    q1, q3 = np.percentile(values, [25, 75])
    assert abs(result.q1 - q1) < 1e-9
    assert abs(result.q3 - q3) < 1e-9
    assert abs(result.iqr - (q3 - q1)) < 1e-9
    # CI brackets mean
    assert result.ci95_low < result.mean
    assert result.ci95_high > result.mean


# ---------------------------------------------------------------------------
# T-01b-C-2: summarize_metric determinism across two calls with same seed
# ---------------------------------------------------------------------------


def test_c2_summarize_metric_determinism():
    from core.aggregate import summarize_metric

    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    r1 = summarize_metric(values, "recall", seed=42)
    r2 = summarize_metric(values, "recall", seed=42)
    assert r1 == r2


# ---------------------------------------------------------------------------
# T-01b-C-3: n_samples == 1 degenerate case
# ---------------------------------------------------------------------------


def test_c3_summarize_metric_n1():
    from core.aggregate import summarize_metric

    result = summarize_metric([0.75], "f1")
    assert result.n_samples == 1
    assert result.std == 0.0
    assert result.iqr == 0.0
    assert result.ci95_low == result.mean == result.ci95_high == 0.75


# ---------------------------------------------------------------------------
# T-01b-C-4: n_samples == 0 raises ValueError
# ---------------------------------------------------------------------------


def test_c4_summarize_metric_empty_raises():
    from core.aggregate import summarize_metric

    with pytest.raises(ValueError, match="empty"):
        summarize_metric([], "precision")


# ---------------------------------------------------------------------------
# T-01b-C-5: aggregate_scorelines happy path 3 runs × 2 types
# ---------------------------------------------------------------------------


def test_c5_aggregate_scorelines_happy_path():
    from core.aggregate import aggregate_scorelines

    # 3 runs, each returning scorelines for V1 + V2 + __aggregate__
    def _run_scorelines(p_v1, r_v1, f_v1, p_v2, r_v2, f_v2):
        agg_p = (p_v1 + p_v2) / 2
        agg_r = (r_v1 + r_v2) / 2
        agg_f = (f_v1 + f_v2) / 2
        return [
            _prf1("V1", p_v1, r_v1, f_v1),
            _prf1("V2", p_v2, r_v2, f_v2),
            _prf1("__aggregate__", agg_p, agg_r, agg_f),
        ]

    per_run = [
        _run_scorelines(0.8, 0.6, 0.7, 0.9, 0.5, 0.65),
        _run_scorelines(0.7, 0.7, 0.7, 0.8, 0.6, 0.68),
        _run_scorelines(0.9, 0.5, 0.67, 0.7, 0.8, 0.75),
    ]

    result = aggregate_scorelines(per_run)

    types_present = {s.violation_type for s in result}
    assert "V1" in types_present
    assert "V2" in types_present
    assert "__aggregate__" in types_present
    assert len(result) == 3

    for sc in result:
        assert sc.n_runs == 3
        assert sc.precision_summary.n_samples == 3
        assert sc.recall_summary.n_samples == 3
        assert sc.f1_summary.n_samples == 3


# ---------------------------------------------------------------------------
# T-01b-C-6: aggregate_scorelines raises on violation_type set mismatch
# ---------------------------------------------------------------------------


def test_c6_aggregate_scorelines_type_mismatch_raises():
    from core.aggregate import aggregate_scorelines

    run_a = [
        _prf1("V1", 0.8, 0.6, 0.7),
        _prf1("__aggregate__", 0.8, 0.6, 0.7),
    ]
    run_b = [
        _prf1("V2", 0.9, 0.5, 0.65),  # different type!
        _prf1("__aggregate__", 0.9, 0.5, 0.65),
    ]

    with pytest.raises(ValueError, match="violation_type"):
        aggregate_scorelines([run_a, run_b])


# ---------------------------------------------------------------------------
# T-01b-C-7: discover_run_pairs finds pairs, skips dirs without judge
# ---------------------------------------------------------------------------


def test_c7_discover_run_pairs(tmp_path):
    from core.aggregate import discover_run_pairs

    # Run A: has both manifest + judge
    run_a = tmp_path / "run_A"
    run_a.mkdir()
    (run_a / "manifest.json").write_text("{}", encoding="utf-8")
    (run_a / "run_A.judge.json").write_text("{}", encoding="utf-8")

    # Run B: has both manifest + judge
    run_b = tmp_path / "run_B"
    run_b.mkdir()
    (run_b / "manifest.json").write_text("{}", encoding="utf-8")
    (run_b / "run_B.judge.json").write_text("{}", encoding="utf-8")

    # Run C: manifest only — no judge → should be skipped
    run_c = tmp_path / "run_C"
    run_c.mkdir()
    (run_c / "manifest.json").write_text("{}", encoding="utf-8")

    pairs = discover_run_pairs(tmp_path)

    assert len(pairs) == 2
    manifest_paths = [p[0] for p in pairs]
    # Sorted by manifest path
    assert manifest_paths == sorted(manifest_paths)
    # run_C not present
    for m, _ in pairs:
        assert "run_C" not in str(m)


# ---------------------------------------------------------------------------
# T-01b-C-8: group_by_configuration groups into correct configs
# ---------------------------------------------------------------------------


def test_c8_group_by_configuration(tmp_path):
    from core.aggregate import group_by_configuration
    from core.run_manifest import write_paper_run_manifest

    # Config 1: P1 / model-A / SRS_D1.docx  → 2 runs
    m1 = _make_manifest("run_P1_A_1", model_id="model-A", srs_path="inputs/SRS_D1.docx", pipeline="P1")
    m2 = _make_manifest("run_P1_A_2", model_id="model-A", srs_path="inputs/SRS_D1.docx", pipeline="P1")
    # Config 2: P2 / model-B / SRS_D2.docx  → 2 runs
    m3 = _make_manifest("run_P2_B_1", model_id="model-B", srs_path="inputs/SRS_D2.docx", pipeline="P2")
    m4 = _make_manifest("run_P2_B_2", model_id="model-B", srs_path="inputs/SRS_D2.docx", pipeline="P2")

    for m in [m1, m2, m3, m4]:
        write_paper_run_manifest(m, tmp_path)
        # Create a fake judge sidecar so discover_run_pairs includes it
        judge_path = tmp_path / m.run_id / f"{m.run_id}.judge.json"
        judge_path.write_text("{}", encoding="utf-8")

    pairs = discover_run_pairs(tmp_path)
    groups = group_by_configuration(pairs)

    assert len(groups) == 2
    config_keys = list(groups.keys())
    # Both configs should have 2 pairs
    for k, v in groups.items():
        assert len(v) == 2

    # Verify that srs_label is basename without extension
    srs_labels = {k[2] for k in config_keys}
    assert "SRS_D1" in srs_labels
    assert "SRS_D2" in srs_labels


# ---------------------------------------------------------------------------
# Helper import needed for test_c8
# ---------------------------------------------------------------------------

from core.aggregate import discover_run_pairs  # noqa: E402  (needed after write)


# ---------------------------------------------------------------------------
# T-01b-C-9: compose_aggregate_key produces filesystem-safe slug
# ---------------------------------------------------------------------------


def test_c9_compose_aggregate_key_filesystem_safe():
    from core.aggregate import AggregatedConfiguration, compose_aggregate_key

    config = AggregatedConfiguration(
        pipeline="P1",
        model_id="gemini-3.1-pro-preview",
        srs_label="SRS_D1",
        n_runs=5,
        run_ids=["r1", "r2", "r3", "r4", "r5"],
        scorelines=[],
    )
    key = compose_aggregate_key(config)
    # Must not contain /, ., whitespace, :, <, >, ", |, ?, *, \
    import re
    assert not re.search(r'[./\s:<>"|?*\\]', key), f"Unsafe chars in key: {key!r}"
    assert len(key) > 0


# ---------------------------------------------------------------------------
# T-01b-C-10: aggregate_configuration end-to-end N=2 fake runs
# ---------------------------------------------------------------------------


def test_c10_aggregate_configuration_e2e(tmp_path):
    from core.aggregate import aggregate_configuration
    from core.metrics import compute_metrics
    from core.run_manifest import Violation, write_paper_run_manifest

    # Build 2 identical-structure runs (same types, different scores)
    for i in range(1, 3):
        run_id = f"run_{i:03d}"
        manifest = _make_manifest(
            run_id=run_id,
            violations=[
                Violation(
                    violation_type="ubiquitous_language_drift",
                    location=f"module/file_{i}.py",
                    severity="ERROR",
                    message="drift",
                )
            ],
        )
        write_paper_run_manifest(manifest, tmp_path)

        # Build judge verdict: mark the prediction as TP
        verdict = JudgeVerdict(
            run_id=run_id,
            entries=[
                JudgeVerdictEntry(
                    violation_type="ubiquitous_language_drift",
                    location=f"module/file_{i}.py",
                    label="true_positive",
                )
            ],
            missed=[],
            judge_model_id="human",
            judge_provider="human",
        )
        judge_path = tmp_path / run_id / f"{run_id}.judge.json"
        judge_path.write_text(verdict.model_dump_json(), encoding="utf-8")

    from core.aggregate import discover_run_pairs

    pairs = discover_run_pairs(tmp_path)
    assert len(pairs) == 2

    agg_config = aggregate_configuration(pairs)

    assert agg_config.n_runs == 2
    assert len(agg_config.run_ids) == 2
    # At least one scoreline per type seen + __aggregate__
    vtypes = {s.violation_type for s in agg_config.scorelines}
    assert "ubiquitous_language_drift" in vtypes
    assert "__aggregate__" in vtypes


# ---------------------------------------------------------------------------
# T-01b-C-11: write_aggregated_configuration_atomic round-trips
# ---------------------------------------------------------------------------


def test_c11_write_atomic_round_trip(tmp_path):
    from core.aggregate import (
        AggregatedConfiguration,
        AggregatedScoreline,
        StatisticalSummary,
        write_aggregated_configuration_atomic,
    )

    def _ss(metric: str) -> StatisticalSummary:
        return StatisticalSummary(
            metric_name=metric,  # type: ignore[arg-type]
            n_samples=3,
            mean=0.7,
            std=0.1,
            q1=0.65,
            q3=0.75,
            iqr=0.10,
            ci95_low=0.6,
            ci95_high=0.8,
        )

    scoreline = AggregatedScoreline(
        violation_type="ubiquitous_language_drift",
        n_runs=3,
        precision_summary=_ss("precision"),
        recall_summary=_ss("recall"),
        f1_summary=_ss("f1"),
    )

    config = AggregatedConfiguration(
        pipeline="P1",
        model_id="gemini-3.1-flash-lite",
        srs_label="SRS_D1",
        n_runs=3,
        run_ids=["r1", "r2", "r3"],
        scorelines=[scoreline],
    )

    out_path = write_aggregated_configuration_atomic(config, tmp_path)
    assert out_path.exists()

    raw = out_path.read_text(encoding="utf-8")
    loaded = AggregatedConfiguration.model_validate_json(raw)
    assert loaded == config


# ---------------------------------------------------------------------------
# T-01b-C-12: Atomic write — patched os.replace leaves only .tmp on disk
# ---------------------------------------------------------------------------


def test_c12_atomic_write_no_partial_file(tmp_path):
    from core.aggregate import (
        AggregatedConfiguration,
        write_aggregated_configuration_atomic,
    )

    config = AggregatedConfiguration(
        pipeline="P2",
        model_id="gemini-3.1-flash-lite",
        srs_label="SRS_D2",
        n_runs=1,
        run_ids=["r1"],
        scorelines=[],
    )

    with patch("core.aggregate.os.replace", side_effect=OSError("injected")):
        with pytest.raises(OSError, match="injected"):
            write_aggregated_configuration_atomic(config, tmp_path)

    # Target must NOT exist; tmp may remain
    from core.aggregate import compose_aggregate_key

    key = compose_aggregate_key(config)
    target = tmp_path / "_aggregated" / f"{key}.json"
    assert not target.exists(), f"Target file must not exist after failed os.replace: {target}"


# ---------------------------------------------------------------------------
# T-01b-C-13: Constant-input bootstrap → ci95_low == ci95_high == mean
# ---------------------------------------------------------------------------


def test_c13_summarize_metric_constant_input():
    from core.aggregate import summarize_metric

    result = summarize_metric([0.5, 0.5, 0.5, 0.5, 0.5], "precision")
    assert result.mean == 0.5
    assert result.ci95_low == 0.5
    assert result.ci95_high == 0.5


# ---------------------------------------------------------------------------
# T-01b-C-14: __aggregate__ rollup preserved across runs
# ---------------------------------------------------------------------------


def test_c14_aggregate_rollup_preserved():
    from core.aggregate import aggregate_scorelines

    # Each run has one real type + __aggregate__
    def _run(p, r, f):
        return [
            _prf1("V1", p, r, f),
            _prf1("__aggregate__", p, r, f),
        ]

    per_run = [_run(0.8, 0.6, 0.7), _run(0.9, 0.5, 0.65), _run(0.7, 0.7, 0.7)]
    result = aggregate_scorelines(per_run)

    agg_scorelines = [s for s in result if s.violation_type == "__aggregate__"]
    assert len(agg_scorelines) == 1
    agg = agg_scorelines[0]
    assert agg.n_runs == 3
    assert agg.precision_summary.n_samples == 3
