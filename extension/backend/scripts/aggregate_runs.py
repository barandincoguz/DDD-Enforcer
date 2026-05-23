"""WP-CORE-20 — Cross-run manifest aggregator (F-9 part 6/7).

CLI:
    python -m scripts.aggregate_runs runs/manifests/run-*.json \
        --out-csv runs/aggregates/n10-per-run.csv \
        --out-pooled runs/aggregates/n10-pooled.json \
        --out-distributions runs/aggregates/n10-distributions.json

Outputs three files (Codex W-3 + W-4):
- per-run CSV: one row per input manifest (raw scalars, no aggregation).
- pooled JSON: pooled rates and percentiles from raw counts (NOT mean-of-ratios).
- distributions JSON: per-metric arrays for box plots / outlier inspection.

Input manifests are NEVER mutated. SHA-256 of every input manifest is recorded
in the pooled JSON for replication-package fidelity.

Aggregator skips `*.tmp` files (Codex W-7 — atomic-write partials).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, List


MIN_SUPPORTED_SCHEMA = "1.0"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any] | None:
    """Load + version-check a single manifest. Returns None if unsupported/corrupt."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[aggregator] skip {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None
    sv = data.get("schema_version", "0.0")
    if sv < MIN_SUPPORTED_SCHEMA:
        print(
            f"[aggregator] skip {path.name}: schema_version={sv} < min {MIN_SUPPORTED_SCHEMA}",
            file=sys.stderr,
        )
        return None
    return data


def _per_run_row(manifest: dict[str, Any]) -> dict[str, Any]:
    """One CSV row per manifest with the EMSE-paper-facing scalars."""
    stages = manifest.get("stages", {})
    llm = manifest.get("llm", {})

    def _stage_metric(stage: str, key: str, default: float = 0.0) -> float:
        return stages.get(stage, {}).get(key, default)

    def _stage_inner_metric(stage: str, key: str, default: float = 0.0) -> float:
        return stages.get(stage, {}).get("metrics", {}).get(key, default)

    return {
        "run_id": manifest.get("run_id", ""),
        "outcome": manifest.get("outcome", "unknown"),
        "elapsed_ms": manifest.get("elapsed_ms", 0.0),
        "instrumentation_overhead_ms": manifest.get("instrumentation_overhead_ms", 0.0),
        "total_cost_usd": llm.get("total_cost_usd", 0.0),
        "total_calls": llm.get("total_calls", 0),
        "json_failed_count": llm.get("json_failed_count", 0),
        "json_parse_failure_count": llm.get("json_parse_failure_count", 0),
        "json_failed_total_count": llm.get("json_failed_total_count", 0),
        "json_failed_rate": llm.get("json_failed_rate", 0.0),
        "retry_exhausted_count": llm.get("retry_exhausted_count", 0),
        "scout_elapsed_ms": _stage_metric("scout", "elapsed_ms"),
        "scout_p50_latency_ms": _stage_metric("scout", "p50_latency_ms"),
        "scout_p95_latency_ms": _stage_metric("scout", "p95_latency_ms"),
        "architect_elapsed_ms": _stage_metric("architect", "elapsed_ms"),
        "architect_attempts": _stage_inner_metric("architect", "attempts"),
        "architect_re_run_count": _stage_inner_metric("architect", "re_run_count"),
        "specialist_elapsed_ms": _stage_metric("specialist", "elapsed_ms"),
        "specialist_degraded_count": _stage_inner_metric("specialist", "degraded_context_count"),
        "verifier_elapsed_ms": _stage_metric("verifier", "elapsed_ms"),
        "refiner_cycles": _stage_inner_metric("refiner", "cycles_used"),
        "refiner_exhausted_count": _stage_inner_metric("refiner", "exhausted_residual_count"),
        "synthesizer_elapsed_ms": _stage_metric("synthesizer", "elapsed_ms"),
        "bounded_context_count": manifest.get("domain_model_summary", {}).get("bounded_context_count", 0),
        "entity_count": manifest.get("domain_model_summary", {}).get("entity_count", 0),
        "aggregate_count": manifest.get("domain_model_summary", {}).get("aggregate_count", 0),
    }


def _write_csv(paths: List[Path], manifests: List[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [_per_run_row(m) for m in manifests]
    fieldnames = list(rows[0].keys()) if rows else ["run_id"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pooled_metrics(manifests: List[dict[str, Any]]) -> dict[str, Any]:
    """Compute pooled rates from raw call counts (Codex W-3).

    json_failed_rate = sum(json_failed_total_count) / sum(total_calls)
                  NOT mean(per-run json_failed_rate)

    The two diverge when runs differ in call count: a run with 0 calls
    contributes 0 to the pooled rate; a per-run avg of rates over-weights
    short runs.
    """
    rows = [_per_run_row(m) for m in manifests]

    def _summ(key: str) -> dict[str, float]:
        vs = [r[key] for r in rows if isinstance(r[key], (int, float))]
        if not vs:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0,
                    "p95": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": statistics.mean(vs),
            "std": statistics.pstdev(vs) if len(vs) >= 2 else 0.0,
            "median": statistics.median(vs),
            "p25": _percentile(vs, 25),
            "p75": _percentile(vs, 75),
            "p95": _percentile(vs, 95),
            "min": min(vs),
            "max": max(vs),
        }

    total_calls_sum = sum(r["total_calls"] for r in rows)
    json_failed_sum = sum(r["json_failed_total_count"] for r in rows)
    pooled_rate = (json_failed_sum / total_calls_sum) if total_calls_sum else 0.0

    return {
        "n_runs": len(manifests),
        "pooled_json_failed_rate": pooled_rate,
        "pooled_total_cost_usd": sum(r["total_cost_usd"] for r in rows),
        "pooled_total_calls": total_calls_sum,
        "pooled_json_failed_total_count": json_failed_sum,
        "metrics": {
            "elapsed_ms": _summ("elapsed_ms"),
            "total_cost_usd": _summ("total_cost_usd"),
            "json_failed_rate": _summ("json_failed_rate"),
            "scout_p95_latency_ms": _summ("scout_p95_latency_ms"),
            "architect_attempts": _summ("architect_attempts"),
            "refiner_cycles": _summ("refiner_cycles"),
            "bounded_context_count": _summ("bounded_context_count"),
            "entity_count": _summ("entity_count"),
        },
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    qs = statistics.quantiles(values, n=100, method="inclusive")
    return qs[int(pct) - 1]


def _distributions(manifests: List[dict[str, Any]]) -> dict[str, Any]:
    rows = [_per_run_row(m) for m in manifests]
    keys = [
        "elapsed_ms", "total_cost_usd", "json_failed_rate",
        "scout_p95_latency_ms", "architect_attempts", "refiner_cycles",
        "bounded_context_count", "entity_count",
    ]
    return {"metrics": {k: [r[k] for r in rows] for k in keys}}


def aggregate(
    paths: Iterable[Path],
    out_csv: Path,
    out_pooled: Path,
    out_distributions: Path,
) -> None:
    """Read manifests, write three outputs. Inputs are not mutated."""
    paths = [p for p in paths if not p.name.endswith(".tmp")]
    manifests: List[dict[str, Any]] = []
    loaded_paths: List[Path] = []
    for p in paths:
        m = _load_manifest(p)
        if m is None:
            continue
        manifests.append(m)
        loaded_paths.append(p)

    _write_csv(loaded_paths, manifests, out_csv)

    pooled = _pooled_metrics(manifests)
    pooled["input_manifest_hashes"] = {p.name: _hash_file(p) for p in loaded_paths}
    pooled["input_manifest_count"] = len(loaded_paths)
    out_pooled.parent.mkdir(parents=True, exist_ok=True)
    out_pooled.write_text(json.dumps(pooled, indent=2))

    dist = _distributions(manifests)
    out_distributions.parent.mkdir(parents=True, exist_ok=True)
    out_distributions.write_text(json.dumps(dist, indent=2))


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate WP-CORE-20 run manifests.")
    parser.add_argument("inputs", nargs="+", type=Path, help="manifest paths or globs")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-pooled", type=Path, required=True)
    parser.add_argument("--out-distributions", type=Path, required=True)
    args = parser.parse_args(argv)

    # Expand globs that the shell didn't expand.
    expanded: List[Path] = []
    for inp in args.inputs:
        if any(c in str(inp) for c in "*?["):
            expanded.extend(Path().glob(str(inp)))
        else:
            expanded.append(inp)

    aggregate(expanded, args.out_csv, args.out_pooled, args.out_distributions)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
