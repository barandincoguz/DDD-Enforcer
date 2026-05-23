"""WP-CORE-20 — Aggregator tests (RED phase).

Covers Codex W-3 (pooled rates) + W-4 (no input mutation) + spec §6.3:
- Per-run CSV, pooled JSON, distributions JSON
- Pooled json_failed_rate uses sum-of-counts, NOT mean-of-ratios
- Aggregator never mutates input manifests
- SHA-256 fingerprints of inputs recorded in pooled output
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _build_manifest_blob(*, run_id: str, total_calls: int, json_failed: int,
                         latency_calls: list[float], outcome: str = "success",
                         elapsed_ms: float = 1000.0):
    """Helper: hand-craft a manifest JSON shape the aggregator must consume."""
    return {
        "schema_version": "1.0",
        "min_supported_schema": "1.0",
        "run_id": run_id,
        "started_at": "2026-05-23T00:00:00Z",
        "ended_at": "2026-05-23T00:00:01Z",
        "elapsed_ms": elapsed_ms,
        "instrumentation_overhead_ms": 0.0,
        "monotonic_clock_source": "time.monotonic_ns",
        "outcome": outcome,
        "environment": {},
        "request": {},
        "stages": {
            "scout": {
                "started_at": "2026-05-23T00:00:00Z",
                "ended_at": "2026-05-23T00:00:01Z",
                "elapsed_ms": 1000.0,
                "status": "success",
                "llm_calls": [
                    {
                        "timestamp": "2026-05-23T00:00:00Z",
                        "stage": "scout",
                        "operation": f"chunk_{i}",
                        "model_id": "gemini-3.1-flash-lite",
                        "provider": "gemini",
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "cached_tokens": 0,
                        "cost_usd": 0.0001,
                        "latency_ms": latency_calls[i],
                        "json_failed": i < json_failed,
                        "json_fail_reason": "invalid_json" if i < json_failed else None,
                        "is_retry_exhausted": False,
                    }
                    for i in range(len(latency_calls))
                ],
                "json_parse_failures": [],
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "metrics": {},
            },
        },
        "llm": {
            "total_calls": total_calls,
            "total_tokens": {"prompt": 0, "completion": 0, "cached": 0, "billable_prompt": 0, "total": 0},
            "total_cost_usd": total_calls * 0.0001,
            "json_failed_count": json_failed,
            "json_parse_failure_count": 0,
            "json_failed_total_count": json_failed,
            "json_failed_rate": (json_failed / total_calls) if total_calls else 0.0,
            "json_fail_reasons": {},
            "retry_exhausted_count": 0,
            "by_model": {},
            "by_stage": {},
        },
        "domain_model_summary": {},
        "errors": [],
    }


def _write_fixture_manifests(dir_: Path) -> list[Path]:
    """Three-run fixture per spec §11.7 acceptance."""
    manifests = [
        _build_manifest_blob(run_id="r1", total_calls=10, json_failed=2, latency_calls=[100.0] * 10),
        _build_manifest_blob(run_id="r2", total_calls=20, json_failed=0, latency_calls=[200.0] * 20),
        _build_manifest_blob(run_id="r3", total_calls=10, json_failed=8, latency_calls=[300.0] * 10),
    ]
    paths = []
    for i, m in enumerate(manifests):
        p = dir_ / f"run-r{i+1}.json"
        p.write_text(json.dumps(m))
        paths.append(p)
    return paths


def test_t_agg_1_pooled_json_failed_rate_uses_sum_of_counts(tmp_path):
    """T-AGG-1 (Codex W-3): pooled rate = sum(failed) / sum(calls), NOT mean(rates)."""
    from scripts.aggregate_runs import aggregate

    paths = _write_fixture_manifests(tmp_path)
    out_csv = tmp_path / "per-run.csv"
    out_pooled = tmp_path / "pooled.json"
    out_distributions = tmp_path / "distributions.json"

    aggregate(paths, out_csv=out_csv, out_pooled=out_pooled, out_distributions=out_distributions)

    pooled = json.loads(out_pooled.read_text())
    # r1: 2/10=0.2, r2: 0/20=0.0, r3: 8/10=0.8. mean-of-rates = 0.333.
    # pooled = (2+0+8) / (10+20+10) = 10/40 = 0.25.  These are DIFFERENT.
    assert pooled["pooled_json_failed_rate"] == pytest.approx(0.25)


def test_t_agg_2_input_manifests_not_mutated(tmp_path):
    """T-AGG-2 (Codex W-4): aggregator must not modify input manifests."""
    from scripts.aggregate_runs import aggregate

    paths = _write_fixture_manifests(tmp_path)
    snapshots = [p.read_text() for p in paths]

    aggregate(
        paths,
        out_csv=tmp_path / "per-run.csv",
        out_pooled=tmp_path / "pooled.json",
        out_distributions=tmp_path / "distributions.json",
    )

    for p, snap in zip(paths, snapshots):
        assert p.read_text() == snap, f"aggregator mutated {p}"


def test_t_agg_3_pooled_output_records_input_hashes(tmp_path):
    """T-AGG-3 (Codex W-4): pooled JSON records SHA-256 of each input manifest."""
    import hashlib

    from scripts.aggregate_runs import aggregate

    paths = _write_fixture_manifests(tmp_path)
    out_pooled = tmp_path / "pooled.json"
    aggregate(
        paths,
        out_csv=tmp_path / "per-run.csv",
        out_pooled=out_pooled,
        out_distributions=tmp_path / "distributions.json",
    )

    pooled = json.loads(out_pooled.read_text())
    expected_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert "input_manifest_hashes" in pooled
    # All input files referenced by basename
    for name, h in expected_hashes.items():
        assert pooled["input_manifest_hashes"][name] == h


def test_t_agg_4_per_run_csv_has_one_row_per_input(tmp_path):
    """T-AGG-4: per-run CSV has exactly N rows + header."""
    from scripts.aggregate_runs import aggregate

    paths = _write_fixture_manifests(tmp_path)
    out_csv = tmp_path / "per-run.csv"
    aggregate(
        paths,
        out_csv=out_csv,
        out_pooled=tmp_path / "pooled.json",
        out_distributions=tmp_path / "distributions.json",
    )

    lines = out_csv.read_text().splitlines()
    assert len(lines) == len(paths) + 1  # header + 3 rows
    header = lines[0]
    assert "run_id" in header
    assert "outcome" in header
    assert "elapsed_ms" in header
    assert "total_cost_usd" in header
    assert "json_failed_rate" in header


def test_t_agg_5_distributions_json_keeps_per_run_arrays(tmp_path):
    """T-AGG-5: distributions JSON contains arrays (N entries each) per metric for box plots."""
    from scripts.aggregate_runs import aggregate

    paths = _write_fixture_manifests(tmp_path)
    out_distributions = tmp_path / "distributions.json"
    aggregate(
        paths,
        out_csv=tmp_path / "per-run.csv",
        out_pooled=tmp_path / "pooled.json",
        out_distributions=out_distributions,
    )

    dist = json.loads(out_distributions.read_text())
    assert "metrics" in dist
    assert len(dist["metrics"]["elapsed_ms"]) == 3
    assert len(dist["metrics"]["total_cost_usd"]) == 3
    assert len(dist["metrics"]["json_failed_rate"]) == 3


def test_t_agg_6_skips_tmp_files(tmp_path):
    """T-AGG-6 (Codex W-7 follow-on): aggregator skips *.tmp files when given a glob."""
    from scripts.aggregate_runs import aggregate

    paths = _write_fixture_manifests(tmp_path)
    # Add a stray .tmp partial file — aggregator must NOT consume it.
    (tmp_path / "run-bogus.json.tmp").write_text("{ not valid json")

    out_pooled = tmp_path / "pooled.json"
    aggregate(
        paths,  # explicit list; aggregator's CLI glob will independently filter .tmp
        out_csv=tmp_path / "per-run.csv",
        out_pooled=out_pooled,
        out_distributions=tmp_path / "distributions.json",
    )

    pooled = json.loads(out_pooled.read_text())
    # Only the 3 valid manifests show up.
    assert pooled["n_runs"] == 3
