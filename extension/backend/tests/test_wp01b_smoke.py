"""WP-01b Task F — E2E smoke test for the full WP-01b pipeline.

T-01b-F-1: Acceptance criterion #7 from the WP-01b spec:
  3 pipelines × 1 model × 1 domain × 2 runs = 6 manifests
  → full chain: discover → group → aggregate → build_rqN_table
    → render_table_spec → write_table_tex_atomic
  → 4 syntactically-valid LaTeX table files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from core.metrics import JudgeVerdict, JudgeVerdictEntry, JudgeMissedExpectation
from core.run_manifest import PaperRunManifest, Violation, write_paper_run_manifest


# ---------------------------------------------------------------------------
# Constants for the 6-run fixture
# ---------------------------------------------------------------------------

_FAKE_SRS_SHA = "b" * 64  # 64-char hex string
_MODEL_ID = "gemini-3.1-flash-lite"
_SRS_PATH = "inputs/SRS_D1.docx"   # single domain → srs_label = "SRS_D1"
_PIPELINES = ["P1", "P2", "P3"]    # 3 distinct pipelines
_RUNS_PER_PIPELINE = 2              # 2 runs each → 6 total


# ---------------------------------------------------------------------------
# Helpers to build fake manifests and verdicts
# ---------------------------------------------------------------------------


def _make_manifest(run_id: str, pipeline: str) -> PaperRunManifest:
    """Build a minimal PaperRunManifest with one violation."""
    return PaperRunManifest(
        run_id=run_id,
        pipeline=pipeline,  # type: ignore[arg-type]
        model_id=_MODEL_ID,
        provider="gemini",
        srs_path=_SRS_PATH,
        srs_sha256=_FAKE_SRS_SHA,
        violations=[
            Violation(
                violation_type="ubiquitous_language_drift",
                location=f"module/file_{run_id}.py",
                severity="ERROR",
                message="drift detected",
            )
        ],
    )


def _make_verdict(run_id: str) -> JudgeVerdict:
    """Build a JudgeVerdict that marks the single predicted violation as TP."""
    return JudgeVerdict(
        run_id=run_id,
        entries=[
            JudgeVerdictEntry(
                violation_type="ubiquitous_language_drift",
                location=f"module/file_{run_id}.py",
                label="true_positive",
            )
        ],
        missed=[],
        judge_model_id="human",
        judge_provider="human",
    )


# ---------------------------------------------------------------------------
# T-01b-F-1: full WP-01b chain smoke test
# ---------------------------------------------------------------------------


class TestWP01bSmoke:
    """E2E smoke tests for the WP-01b pipeline (Task F)."""

    def test_T01b_F1_full_chain(self, tmp_path: Path) -> None:
        """Drive the full WP-01b chain with 6 fake runs and verify outputs.

        Fixture: 3 pipelines × 1 model × 1 domain × 2 runs = 6 manifests.

        Chain exercised:
          1. write_paper_run_manifest + judge sidecar (setup)
          2. discover_run_pairs → group_by_configuration
          3. aggregate_configuration per group
          4. write_aggregated_configuration_atomic per group
          5. load_aggregated_configs
          6. build_rq1_table / build_rq2_table / build_rq3_table / build_rq4_table
          7. render_table_spec per table
          8. write_table_tex_atomic per table

        Assertions:
          A. _aggregated/ contains exactly 3 JSON files (one per pipeline config).
          B. tables/rq1.tex through tables/rq4.tex exist and are non-empty.
          C. Each rqN.tex contains \\begin{table}, \\end{table}, \\toprule, \\bottomrule.
          D. RQ1 table has 3 rows (one per pipeline).
          E. RQ2 table has 1 row (single model).
          F. RQ4 table has 3 rows (one per configuration).
        """
        from core.aggregate import (
            aggregate_configuration,
            discover_run_pairs,
            group_by_configuration,
            write_aggregated_configuration_atomic,
        )
        from core.latex_tables import (
            build_rq1_table,
            build_rq2_table,
            build_rq3_table,
            build_rq4_table,
            load_aggregated_configs,
            render_table_spec,
            write_table_tex_atomic,
        )

        runs_root = tmp_path

        # ------------------------------------------------------------------ #
        # Step 1: Write 6 fake manifests + judge sidecars                     #
        # ------------------------------------------------------------------ #
        for pipeline in _PIPELINES:
            for run_idx in range(1, _RUNS_PER_PIPELINE + 1):
                run_id = f"run_{pipeline}_{run_idx:02d}"

                manifest = _make_manifest(run_id=run_id, pipeline=pipeline)
                write_paper_run_manifest(manifest, runs_root)

                verdict = _make_verdict(run_id=run_id)
                judge_path = runs_root / run_id / f"{run_id}.judge.json"
                judge_path.write_text(verdict.model_dump_json(), encoding="utf-8")

        # ------------------------------------------------------------------ #
        # Step 2: discover → group                                            #
        # ------------------------------------------------------------------ #
        pairs = discover_run_pairs(runs_root)
        assert len(pairs) == 6, f"Expected 6 pairs, got {len(pairs)}"

        groups = group_by_configuration(pairs)
        # 3 pipelines × 1 model × 1 srs_label → 3 groups
        assert len(groups) == 3, f"Expected 3 config groups, got {len(groups)}"

        # ------------------------------------------------------------------ #
        # Step 3–4: aggregate each group and write to _aggregated/            #
        # ------------------------------------------------------------------ #
        for _config_key, group_pairs in groups.items():
            assert len(group_pairs) == 2, (
                f"Expected 2 runs per group, got {len(group_pairs)}"
            )
            agg_config = aggregate_configuration(group_pairs)
            write_aggregated_configuration_atomic(agg_config, runs_root)

        # Assertion A: _aggregated/ has exactly 3 JSON files
        agg_dir = runs_root / "_aggregated"
        agg_json_files = sorted(agg_dir.glob("*.json"))
        assert len(agg_json_files) == 3, (
            f"Expected 3 aggregated JSON files, found {len(agg_json_files)}: "
            f"{[f.name for f in agg_json_files]}"
        )

        # ------------------------------------------------------------------ #
        # Step 5: load aggregated configs                                     #
        # ------------------------------------------------------------------ #
        configs = load_aggregated_configs(runs_root)
        assert len(configs) == 3, f"Expected 3 loaded configs, got {len(configs)}"

        # ------------------------------------------------------------------ #
        # Step 6–8: build + render + write all 4 RQ tables                   #
        # ------------------------------------------------------------------ #
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        rq_builders = {
            "rq1": build_rq1_table,
            "rq2": build_rq2_table,
            "rq3": build_rq3_table,
            "rq4": build_rq4_table,
        }

        rendered: dict[str, str] = {}
        for rq_name, builder in rq_builders.items():
            spec = builder(configs)
            tex_content = render_table_spec(spec)
            out_path = tables_dir / f"{rq_name}.tex"
            write_table_tex_atomic(tex_content, out_path)
            rendered[rq_name] = tex_content

        # ------------------------------------------------------------------ #
        # Assertions B, C: all 4 .tex files exist, non-empty, well-formed    #
        # ------------------------------------------------------------------ #
        for rq_name in ["rq1", "rq2", "rq3", "rq4"]:
            tex_path = tables_dir / f"{rq_name}.tex"

            # B: file exists and is non-empty
            assert tex_path.exists(), f"{rq_name}.tex was not written"
            content = tex_path.read_text(encoding="utf-8")
            assert len(content) > 0, f"{rq_name}.tex is empty"

            # C: syntactic well-formedness markers
            assert r"\begin{table}" in content, (
                f"{rq_name}.tex missing \\begin{{table}}"
            )
            assert r"\end{table}" in content, (
                f"{rq_name}.tex missing \\end{{table}}"
            )
            assert r"\toprule" in content, (
                f"{rq_name}.tex missing \\toprule"
            )
            assert r"\bottomrule" in content, (
                f"{rq_name}.tex missing \\bottomrule"
            )

        # ------------------------------------------------------------------ #
        # Assertion D: RQ1 has 3 rows (one per pipeline P1/P2/P3)            #
        # ------------------------------------------------------------------ #
        rq1_spec = build_rq1_table(configs)
        assert len(rq1_spec.rows) == 3, (
            f"RQ1 expected 3 pipeline rows, got {len(rq1_spec.rows)}: "
            f"{[r.row_label for r in rq1_spec.rows]}"
        )
        rq1_labels = {r.row_label for r in rq1_spec.rows}
        assert "P1" in rq1_labels, "RQ1 missing P1 row"
        assert "P2" in rq1_labels, "RQ1 missing P2 row"
        assert "P3" in rq1_labels, "RQ1 missing P3 row"

        # ------------------------------------------------------------------ #
        # Assertion E: RQ2 has 1 row (single model)                          #
        # ------------------------------------------------------------------ #
        rq2_spec = build_rq2_table(configs)
        assert len(rq2_spec.rows) == 1, (
            f"RQ2 expected 1 model row, got {len(rq2_spec.rows)}: "
            f"{[r.row_label for r in rq2_spec.rows]}"
        )

        # ------------------------------------------------------------------ #
        # Assertion F: RQ4 has 3 rows (one per configuration)                #
        # ------------------------------------------------------------------ #
        rq4_spec = build_rq4_table(configs)
        assert len(rq4_spec.rows) == 3, (
            f"RQ4 expected 3 configuration rows, got {len(rq4_spec.rows)}: "
            f"{[r.row_label for r in rq4_spec.rows]}"
        )
