"""Statistical aggregation of N-run paper-data results.

WP-01b Task C deliverables:
  1. StatisticalSummary       — mean / std / IQR / bootstrap 95% CI for one metric.
  2. AggregatedScoreline      — per-violation-type summary across N runs.
  3. AggregatedConfiguration  — one config (pipeline, model_id, srs_label) aggregated.
  4. summarize_metric()       — compute summary for a list of float samples.
  5. aggregate_scorelines()   — group N runs' scorelines by type, produce AggregatedScorelines.
  6. discover_run_pairs()     — glob runs_root for manifest+judge pairs.
  7. group_by_configuration() — group pairs by (pipeline, model_id, srs_label) tuple.
  8. aggregate_configuration()— load pairs, compute_metrics per run, aggregate.
  9. write_aggregated_configuration_atomic() — atomic write to _aggregated/<key>.json.
  10. compose_aggregate_key() — filesystem-safe slug for a configuration.

No LLM calls. Pure deterministic statistics (seeded bootstrap). Pydantic v2 + numpy + stdlib.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import tempfile
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from core.metrics import PrecisionRecallF1, compute_metrics, load_judge_verdict
from core.run_manifest import PaperRunManifest


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class StatisticalSummary(BaseModel):
    """Mean / std / IQR / bootstrap 95% CI for a single metric (e.g. precision)."""

    metric_name: Literal["precision", "recall", "f1"]
    n_samples: int
    mean: float
    std: float
    """Sample std with ddof=1; 0.0 when n_samples < 2."""
    q1: float
    q3: float
    iqr: float
    """q3 - q1."""
    ci95_low: float
    """Bootstrap 95% CI lower bound."""
    ci95_high: float
    """Bootstrap 95% CI upper bound."""


class AggregatedScoreline(BaseModel):
    """Per-violation-type AND per-aggregate rollup, across N runs of the same configuration."""

    violation_type: str
    """The violation type key, or '__aggregate__' for the overall rollup."""
    n_runs: int
    precision_summary: StatisticalSummary
    recall_summary: StatisticalSummary
    f1_summary: StatisticalSummary


class AggregatedConfiguration(BaseModel):
    """One configuration (pipeline, model_id, srs_label) aggregated across N runs."""

    pipeline: Optional[Literal["P1", "P2", "P3"]] = None
    model_id: str
    srs_label: str
    """Basename of srs_path, no extension."""
    n_runs: int
    run_ids: List[str]
    scorelines: List[AggregatedScoreline]
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Core statistical function
# ---------------------------------------------------------------------------


def summarize_metric(
    values: List[float],
    metric_name: Literal["precision", "recall", "f1"],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> StatisticalSummary:
    """Compute mean, std, IQR, and bootstrap 95% CI for a single metric across N samples.

    Bootstrap: numpy.random.default_rng(seed); resample with replacement
    ``bootstrap_resamples`` times; compute mean of each resample; take 2.5th
    and 97.5th percentiles as ci95_low/high.

    When n_samples < 2: std = 0.0, IQR = 0.0, ci95_low = ci95_high = mean.
    When n_samples == 0: raises ValueError — caller must not feed empty lists.
    """
    n = len(values)
    if n == 0:
        raise ValueError(
            "summarize_metric: values list is empty; "
            "this is a join error in the caller — each run must produce scorelines."
        )

    arr = np.array(values, dtype=float)
    mean_val = float(arr.mean())

    if n < 2:
        return StatisticalSummary(
            metric_name=metric_name,
            n_samples=n,
            mean=mean_val,
            std=0.0,
            q1=mean_val,
            q3=mean_val,
            iqr=0.0,
            ci95_low=mean_val,
            ci95_high=mean_val,
        )

    # Sample standard deviation (ddof=1)
    std_val = float(statistics.stdev(values))

    # IQR via numpy.percentile with default linear interpolation
    q1_val, q3_val = np.percentile(arr, [25, 75], method="linear")
    q1_val = float(q1_val)
    q3_val = float(q3_val)
    iqr_val = q3_val - q1_val

    # Bootstrap 95% CI
    rng = np.random.default_rng(seed)
    boot_samples = rng.choice(arr, size=(bootstrap_resamples, n), replace=True)
    boot_means = boot_samples.mean(axis=1)
    ci95_low, ci95_high = np.percentile(boot_means, [2.5, 97.5])

    return StatisticalSummary(
        metric_name=metric_name,
        n_samples=n,
        mean=mean_val,
        std=std_val,
        q1=q1_val,
        q3=q3_val,
        iqr=iqr_val,
        ci95_low=float(ci95_low),
        ci95_high=float(ci95_high),
    )


# ---------------------------------------------------------------------------
# Scoreline aggregation
# ---------------------------------------------------------------------------


def aggregate_scorelines(
    per_run_scorelines: List[List[PrecisionRecallF1]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> List[AggregatedScoreline]:
    """Group N runs' per-type scorelines by violation_type and produce one
    AggregatedScoreline per type (including __aggregate__).

    Each inner list (one per run) MUST have the same set of violation_types
    AND the __aggregate__ entry — else raises ValueError. (compute_metrics
    is deterministic about which types appear; mismatches signal a join bug
    in the caller.)
    """
    if not per_run_scorelines:
        raise ValueError("aggregate_scorelines: per_run_scorelines is empty")

    n_runs = len(per_run_scorelines)

    # Determine the canonical set of violation_types from the first run
    canonical_types = frozenset(s.violation_type for s in per_run_scorelines[0])

    # Validate all runs have the same set of types
    for i, run_scorelines in enumerate(per_run_scorelines[1:], start=1):
        run_types = frozenset(s.violation_type for s in run_scorelines)
        if run_types != canonical_types:
            missing = canonical_types - run_types
            extra = run_types - canonical_types
            raise ValueError(
                f"aggregate_scorelines: run {i} has a different set of violation_types "
                f"than run 0. Missing from run {i}: {missing!r}. "
                f"Extra in run {i}: {extra!r}. "
                "This is a join bug in the caller."
            )

    # Group values by violation_type
    precision_by_type: Dict[str, List[float]] = {vt: [] for vt in canonical_types}
    recall_by_type: Dict[str, List[float]] = {vt: [] for vt in canonical_types}
    f1_by_type: Dict[str, List[float]] = {vt: [] for vt in canonical_types}

    for run_scorelines in per_run_scorelines:
        for score in run_scorelines:
            precision_by_type[score.violation_type].append(score.precision)
            recall_by_type[score.violation_type].append(score.recall)
            f1_by_type[score.violation_type].append(score.f1)

    result: List[AggregatedScoreline] = []
    for vtype in sorted(canonical_types):
        # Use seed offsets +0/+1/+2 per metric so precision/recall/f1 bootstrap
        # draws use independent RNG streams.  Deterministic replay is preserved:
        # the outer seed still determines all derived seeds.
        result.append(
            AggregatedScoreline(
                violation_type=vtype,
                n_runs=n_runs,
                precision_summary=summarize_metric(
                    precision_by_type[vtype],
                    "precision",
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed,
                ),
                recall_summary=summarize_metric(
                    recall_by_type[vtype],
                    "recall",
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed + 1,
                ),
                f1_summary=summarize_metric(
                    f1_by_type[vtype],
                    "f1",
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed + 2,
                ),
            )
        )

    return result


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_run_pairs(runs_root: Path) -> List[Tuple[Path, Path]]:
    """Find every <runs_root>/<run_id>/manifest.json that has a sibling
    *.judge.json file. Returns sorted list of (manifest_path, judge_path)
    tuples. Skips run directories without a judge file (paper-data not
    yet labeled by judge).
    """
    pairs: List[Tuple[Path, Path]] = []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        # Find *.judge.json sibling
        judge_files = sorted(run_dir.glob("*.judge.json"))
        if not judge_files:
            continue  # silently skip — not yet labeled

        if len(judge_files) > 1:
            raise ValueError(
                f"ambiguous judge files in {run_dir}: {sorted(judge_files)}"
            )

        judge_path = judge_files[0]
        pairs.append((manifest_path, judge_path))

    return pairs


# ---------------------------------------------------------------------------
# Configuration grouping
# ---------------------------------------------------------------------------


def group_by_configuration(
    pairs: List[Tuple[Path, Path]],
) -> Dict[Tuple[Optional[str], str, str], List[Tuple[Path, Path]]]:
    """Group manifest+judge pairs by configuration tuple
    (pipeline, model_id, srs_label). Loads each PaperRunManifest to read
    the grouping keys. ``srs_label`` is the basename without extension of
    ``manifest.srs_path``.
    """
    groups: Dict[Tuple[Optional[str], str, str], List[Tuple[Path, Path]]] = {}

    for manifest_path, judge_path in pairs:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PaperRunManifest.model_validate(raw)

        srs_label = Path(manifest.srs_path).stem  # basename without extension

        key: Tuple[Optional[str], str, str] = (
            manifest.pipeline,
            manifest.model_id,
            srs_label,
        )
        if key not in groups:
            groups[key] = []
        groups[key].append((manifest_path, judge_path))

    return groups


# ---------------------------------------------------------------------------
# Per-configuration aggregation
# ---------------------------------------------------------------------------


def aggregate_configuration(
    pairs: List[Tuple[Path, Path]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> AggregatedConfiguration:
    """For one configuration's run pairs, load each PaperRunManifest +
    JudgeVerdict, compute_metrics per run, then aggregate via
    aggregate_scorelines. Returns the AggregatedConfiguration.
    """
    if not pairs:
        raise ValueError("aggregate_configuration: pairs list is empty")

    run_ids: List[str] = []
    per_run_scorelines: List[List[PrecisionRecallF1]] = []

    # Read pipeline/model_id/srs_label from first pair — caller guarantees homogeneity
    first_manifest_raw = json.loads(pairs[0][0].read_text(encoding="utf-8"))
    first_manifest = PaperRunManifest.model_validate(first_manifest_raw)
    pipeline = first_manifest.pipeline
    model_id = first_manifest.model_id
    srs_label = Path(first_manifest.srs_path).stem

    for manifest_path, judge_path in pairs:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = PaperRunManifest.model_validate(raw)

        verdict = load_judge_verdict(judge_path)

        # compute_metrics raises if verdict.run_id != manifest.run_id
        scorelines = compute_metrics(manifest, verdict)

        run_ids.append(manifest.run_id)
        per_run_scorelines.append(scorelines)

    aggregated_scorelines = aggregate_scorelines(
        per_run_scorelines,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )

    return AggregatedConfiguration(
        pipeline=pipeline,
        model_id=model_id,
        srs_label=srs_label,
        n_runs=len(pairs),
        run_ids=run_ids,
        scorelines=aggregated_scorelines,
        bootstrap_seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )


# ---------------------------------------------------------------------------
# Filesystem-safe slug
# ---------------------------------------------------------------------------


def compose_aggregate_key(config: AggregatedConfiguration) -> str:
    """Filesystem-safe slug: ``f"{pipeline or 'pipelineNone'}_{model_id}_{srs_label}"``
    sanitized with the same regex as :func:`core.run_manifest.compose_run_id`.
    """

    def _sanitize(s: str) -> str:
        return re.sub(r'[./\s:<>"|?*\\]', "_", s)

    pipeline_str = config.pipeline if config.pipeline is not None else "pipelineNone"
    parts = [pipeline_str, config.model_id, config.srs_label]
    return "_".join(_sanitize(p) for p in parts)


# ---------------------------------------------------------------------------
# Atomic writer
# ---------------------------------------------------------------------------


def write_aggregated_configuration_atomic(
    config: AggregatedConfiguration,
    runs_root: Path,
) -> Path:
    """Write to ``<runs_root>/_aggregated/<config_key>.json`` atomically.

    Uses the ``tmp + fsync + os.replace`` pattern: a partial write leaves
    only the ``.tmp`` file on disk — the target is never corrupt.
    """
    key = compose_aggregate_key(config)
    agg_dir = runs_root / "_aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)

    target = agg_dir / f"{key}.json"
    tmp = target.with_suffix(target.suffix + ".tmp")

    payload = config.model_dump_json(indent=2)

    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())

    os.replace(tmp, target)
    return target
