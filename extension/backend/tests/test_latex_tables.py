"""Tests for core/latex_tables.py — WP-01b Task D.

TDD RED phase: all 14 tests written BEFORE core/latex_tables.py exists.

T-01b-D-1  through T-01b-D-14 per spec.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from core.aggregate import (
    AggregatedConfiguration,
    AggregatedScoreline,
    StatisticalSummary,
    compose_aggregate_key,
)


# ---------------------------------------------------------------------------
# Helpers to build fake AggregatedConfiguration instances
# ---------------------------------------------------------------------------


def _ss(metric: str, mean: float = 0.7, std: float = 0.05,
        ci_low: float = 0.65, ci_high: float = 0.75) -> StatisticalSummary:
    """Build a minimal StatisticalSummary for tests."""
    return StatisticalSummary(
        metric_name=metric,  # type: ignore[arg-type]
        n_samples=5,
        mean=mean,
        std=std,
        q1=mean - 0.05,
        q3=mean + 0.05,
        iqr=0.1,
        ci95_low=ci_low,
        ci95_high=ci_high,
    )


def _scoreline(vtype: str, p: float = 0.8, r: float = 0.7, f: float = 0.75) -> AggregatedScoreline:
    return AggregatedScoreline(
        violation_type=vtype,
        n_runs=5,
        precision_summary=_ss("precision", mean=p),
        recall_summary=_ss("recall", mean=r),
        f1_summary=_ss("f1", mean=f),
    )


def _config(
    pipeline: str | None = "P1",
    model_id: str = "gemini-3.1-flash-lite",
    srs_label: str = "SRS_D1",
    extra_scorelines: List[AggregatedScoreline] | None = None,
) -> AggregatedConfiguration:
    """Build a minimal AggregatedConfiguration for tests."""
    scorelines = [
        _scoreline("V1"),
        _scoreline("V2"),
        _scoreline("__aggregate__", p=0.85, r=0.75, f=0.8),
    ]
    if extra_scorelines:
        scorelines.extend(extra_scorelines)
    return AggregatedConfiguration(
        pipeline=pipeline,  # type: ignore[arg-type]
        model_id=model_id,
        srs_label=srs_label,
        n_runs=5,
        run_ids=["r1", "r2", "r3", "r4", "r5"],
        scorelines=scorelines,
    )


# ---------------------------------------------------------------------------
# T-01b-D-1: sanitize_latex_label — alphanumerics + `-` + `_` survive, `.` replaced
# ---------------------------------------------------------------------------


def test_d1_sanitize_latex_label_valid_chars():
    from core.latex_tables import sanitize_latex_label

    result = sanitize_latex_label("gemini-3.1-pro_preview")
    # The `.` must be replaced but alphanumerics, `-`, `_` survive
    assert result == "gemini-3_1-pro_preview"


# ---------------------------------------------------------------------------
# T-01b-D-2: sanitize_latex_label — space, `/`, `#` all replaced
# ---------------------------------------------------------------------------


def test_d2_sanitize_latex_label_special_chars():
    from core.latex_tables import sanitize_latex_label

    result = sanitize_latex_label("foo bar/baz#qux")
    # Must contain no space, /, #
    assert " " not in result
    assert "/" not in result
    assert "#" not in result
    # Only alphanumerics + `-` + `_` remain
    assert re.fullmatch(r"[A-Za-z0-9_-]+", result), f"Non-whitelisted chars in: {result!r}"


# ---------------------------------------------------------------------------
# T-01b-D-3: escape_latex_value — wraps in \detokenize{...}
# ---------------------------------------------------------------------------


def test_d3_escape_latex_value_wraps_detokenize():
    from core.latex_tables import escape_latex_value

    result = escape_latex_value("Hoca's & students")
    assert result.startswith(r"\detokenize{")
    assert result.endswith("}")


# ---------------------------------------------------------------------------
# T-01b-D-4: format_mean_std — correct rounding + Unicode ±
# ---------------------------------------------------------------------------


def test_d4_format_mean_std_rounding_and_unicode():
    from core.latex_tables import format_mean_std

    result = format_mean_std(0.8234, 0.0451, places=3)
    # Correct rounding: 0.8234 → "0.823", 0.0451 → "0.045"
    assert result == "0.823 ± 0.045"


# ---------------------------------------------------------------------------
# T-01b-D-5: build_rq1_table — 2 configs same pipeline + 1 different pipeline
# ---------------------------------------------------------------------------


def test_d5_build_rq1_table_groups_by_pipeline():
    from core.latex_tables import build_rq1_table

    configs = [
        _config(pipeline="P1", model_id="model-A"),
        _config(pipeline="P1", model_id="model-B"),  # same pipeline, different model
        _config(pipeline="P2", model_id="model-C"),
    ]
    spec = build_rq1_table(configs)

    assert spec.rq == "RQ1"
    assert len(spec.rows) == 2

    row_labels = {row.row_label for row in spec.rows}
    assert "P1" in row_labels
    assert "P2" in row_labels

    # P1 row: mean of 2 configs' __aggregate__ precision means (both = 0.85)
    p1_row = next(r for r in spec.rows if r.row_label == "P1")
    # precision cell mean should be average of the two P1 configs' __aggregate__ precision means
    # Both have mean=0.85 → averaged = 0.85
    p_cell = p1_row.cells[0]
    assert abs(p_cell.mean - 0.85) < 1e-9


# ---------------------------------------------------------------------------
# T-01b-D-6: build_rq1_table — all None pipelines → single row "unspecified"
# ---------------------------------------------------------------------------


def test_d6_build_rq1_table_none_pipeline():
    from core.latex_tables import build_rq1_table

    configs = [
        _config(pipeline=None, model_id="model-X"),
    ]
    spec = build_rq1_table(configs)

    assert len(spec.rows) == 1
    assert spec.rows[0].row_label == "unspecified"


# ---------------------------------------------------------------------------
# T-01b-D-7: build_rq2_table — groups by model_id, sanitized labels
# ---------------------------------------------------------------------------


def test_d7_build_rq2_table_groups_by_model():
    from core.latex_tables import build_rq2_table

    configs = [
        _config(model_id="gemini-3.1-flash-lite", pipeline="P1"),
        _config(model_id="gemini-3.1-flash-lite", pipeline="P2"),  # same model, diff pipeline
        _config(model_id="gpt-oss:120b-cloud", pipeline="P1"),
    ]
    spec = build_rq2_table(configs)

    assert spec.rq == "RQ2"
    assert len(spec.rows) == 2

    # Labels must be sanitized (no `:` in them)
    for row in spec.rows:
        assert ":" not in row.row_label, f"Unsanitized label: {row.row_label!r}"


# ---------------------------------------------------------------------------
# T-01b-D-8: build_rq3_table — excludes __aggregate__, keeps per-type rows
# ---------------------------------------------------------------------------


def test_d8_build_rq3_table_excludes_aggregate():
    from core.latex_tables import build_rq3_table

    configs = [_config(pipeline="P1")]
    spec = build_rq3_table(configs)

    assert spec.rq == "RQ3"
    vtypes = {row.row_label for row in spec.rows}
    # __aggregate__ must not appear in row labels
    assert "__aggregate__" not in vtypes
    # The real violation types must be present
    assert "V1" in vtypes
    assert "V2" in vtypes


# ---------------------------------------------------------------------------
# T-01b-D-9: build_rq4_table — CI width = ci95_high - ci95_low of __aggregate__ F1
# ---------------------------------------------------------------------------


def test_d9_build_rq4_table_ci_width():
    from core.latex_tables import build_rq4_table

    # Build a config with a known CI width for __aggregate__ F1
    scorelines = [
        _scoreline("V1"),
        AggregatedScoreline(
            violation_type="__aggregate__",
            n_runs=5,
            precision_summary=_ss("precision"),
            recall_summary=_ss("recall"),
            f1_summary=StatisticalSummary(
                metric_name="f1",
                n_samples=5,
                mean=0.75,
                std=0.05,
                q1=0.72,
                q3=0.78,
                iqr=0.06,
                ci95_low=0.60,   # known low
                ci95_high=0.90,  # known high → width = 0.30
            ),
        ),
    ]
    config = AggregatedConfiguration(
        pipeline="P1",
        model_id="gemini-3.1-flash-lite",
        srs_label="SRS_D1",
        n_runs=5,
        run_ids=["r1", "r2", "r3", "r4", "r5"],
        scorelines=scorelines,
    )

    spec = build_rq4_table([config])

    assert spec.rq == "RQ4"
    assert len(spec.rows) == 1

    # The CI width cell should be the last cell in the row
    row = spec.rows[0]
    ci_width_cell = row.cells[-1]
    # Expected width: 0.90 - 0.60 = 0.30
    assert abs(ci_width_cell.mean - 0.30) < 1e-9


# ---------------------------------------------------------------------------
# T-01b-D-10: render_table_spec — starts/ends with table tags, has toprule etc.
# ---------------------------------------------------------------------------


def test_d10_render_table_spec_structure():
    from core.latex_tables import TableCell, TableRow, TableSpec, render_table_spec

    spec = TableSpec(
        rq="RQ1",
        caption=r"\detokenize{RQ1 caption}",
        label="tab:rq1",
        column_headers=["Pipeline", "Precision"],
        rows=[
            TableRow(
                row_label="P1",
                cells=[
                    TableCell(label="Precision", mean=0.8, std=0.05, formatted="0.800 ± 0.050"),
                ],
            )
        ],
    )

    output = render_table_spec(spec)

    assert output.startswith(r"\begin{table}")
    assert output.rstrip().endswith(r"\end{table}")
    assert r"\toprule" in output
    assert r"\midrule" in output
    assert r"\bottomrule" in output
    # Row terminator
    assert r"\\" in output


# ---------------------------------------------------------------------------
# T-01b-D-11: render_table_spec — injection probe: sanitized content is safe
# ---------------------------------------------------------------------------


def test_d11_render_table_spec_injection_probe():
    from core.latex_tables import (
        TableCell,
        TableRow,
        TableSpec,
        escape_latex_value,
        render_table_spec,
        sanitize_latex_label,
    )

    # Adversarial inputs that pass through the sanitizers
    dangerous_caption = escape_latex_value("Caption with & and $ and % and #")
    dangerous_label = sanitize_latex_label("model#id$percent%amp&brace{}")

    spec = TableSpec(
        rq="RQ1",
        caption=dangerous_caption,
        label="tab:rq1",
        column_headers=["Pipeline", "Precision"],
        rows=[
            TableRow(
                row_label=dangerous_label,
                cells=[
                    TableCell(label="Precision", mean=0.8, std=0.05, formatted="0.800 ± 0.050"),
                ],
            )
        ],
    )

    output = render_table_spec(spec)

    # The raw dangerous characters must NOT appear OUTSIDE of \detokenize{...} wrappers
    # Find all \detokenize{...} segments and mask them out
    masked = re.sub(r"\\detokenize\{[^}]*\}", "SAFE", output)

    # After masking, no bare `&` (the LaTeX column separator is fine)
    # but $ and % and # outside detokenize are dangerous
    # We do not mask `&` because it IS the column separator — check only %, $, # outside wrappers
    assert "$" not in masked, f"Raw $ found outside detokenize in output"
    assert "%" not in masked, f"Raw % found outside detokenize in output"
    # `#` is only dangerous in certain positions — the sanitizer replaced it with `_`
    # so it shouldn't appear at all
    assert "#" not in masked, f"Raw # found in output"


# ---------------------------------------------------------------------------
# T-01b-D-12: load_aggregated_configs — reads valid, skips malformed with [WARN]
# ---------------------------------------------------------------------------


def test_d12_load_aggregated_configs_skips_malformed(tmp_path, capsys):
    from core.latex_tables import load_aggregated_configs

    agg_dir = tmp_path / "_aggregated"
    agg_dir.mkdir()

    # Write 2 valid configs
    for i in range(1, 3):
        config = _config(pipeline="P1", model_id=f"model-{i}", srs_label=f"SRS_D{i}")
        key = compose_aggregate_key(config)
        (agg_dir / f"{key}.json").write_text(config.model_dump_json(), encoding="utf-8")

    # Write 1 malformed JSON
    (agg_dir / "malformed.json").write_text("{not valid json###}", encoding="utf-8")

    configs = load_aggregated_configs(tmp_path)

    # Should return 2 valid configs
    assert len(configs) == 2

    # Should have logged a [WARN] to stderr
    captured = capsys.readouterr()
    assert "[WARN]" in captured.err


# ---------------------------------------------------------------------------
# T-01b-D-13: write_table_tex_atomic — round-trip + os.replace patch
# ---------------------------------------------------------------------------


def test_d13_write_table_tex_atomic_round_trip(tmp_path):
    from core.latex_tables import write_table_tex_atomic

    content = r"\begin{table}" + "\n" + r"\end{table}"
    output_path = tmp_path / "rq1.tex"

    write_table_tex_atomic(content, output_path)

    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == content


def test_d13b_write_table_tex_atomic_replace_called(tmp_path):
    from core.latex_tables import write_table_tex_atomic

    import os as _os

    content = r"\begin{table}\end{table}"
    output_path = tmp_path / "out.tex"

    # Capture the call args while still performing the real replace via
    # the original (un-patched) os.replace so the .tmp file gets renamed.
    _real_replace = _os.replace
    captured: list[tuple[object, object]] = []

    def _spy_replace(src: object, dst: object) -> None:
        captured.append((src, dst))
        _real_replace(src, dst)  # type: ignore[arg-type]

    with patch("core.latex_tables.os.replace", side_effect=_spy_replace):
        write_table_tex_atomic(content, output_path)

    assert len(captured) == 1, "os.replace should be called exactly once"
    src, dst = captured[0]
    assert str(src).endswith(".tmp"), f"Expected .tmp src, got: {src!r}"
    assert Path(str(dst)) == output_path


# ---------------------------------------------------------------------------
# T-01b-D-14: end-to-end — build _aggregated dir, call builders + writer, check .tex
# ---------------------------------------------------------------------------


def test_d14_end_to_end(tmp_path):
    from core.latex_tables import (
        build_rq1_table,
        load_aggregated_configs,
        render_table_spec,
        write_table_tex_atomic,
    )

    agg_dir = tmp_path / "_aggregated"
    agg_dir.mkdir()

    config = _config(pipeline="P1", model_id="gemini-3.1-flash-lite", srs_label="SRS_D1")
    key = compose_aggregate_key(config)
    (agg_dir / f"{key}.json").write_text(config.model_dump_json(), encoding="utf-8")

    # Load + build + render + write
    configs = load_aggregated_configs(tmp_path)
    assert len(configs) == 1

    spec = build_rq1_table(configs)
    rendered = render_table_spec(spec)
    output_path = tmp_path / "rq1.tex"
    write_table_tex_atomic(rendered, output_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert len(content) > 0
    assert r"\begin{table}" in content
    assert r"\end{table}" in content
