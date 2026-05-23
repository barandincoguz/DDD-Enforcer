"""Precision / Recall / F1 computation for DDD-Enforcer paper runs.

WP-01b Task B deliverables:
  1. JudgeVerdictEntry   — per-prediction truth label (true_positive | false_positive).
  2. JudgeMissedExpectation — violations the model missed entirely (FN source).
  3. JudgeVerdict        — container with judge_provider Literal (gemini | ollama | human).
  4. PrecisionRecallF1   — scoreline for one violation_type or the micro-aggregate.
  5. load_judge_verdict(path) -> JudgeVerdict
  6. compute_metrics(manifest, verdict) -> List[PrecisionRecallF1]
  7. aggregate_scoreline(per_type) -> PrecisionRecallF1 (micro-average)

No LLM calls. Pure deterministic arithmetic. Pydantic v2 + stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from core.run_manifest import PaperRunManifest


# ---------------------------------------------------------------------------
# Judge schemas
# ---------------------------------------------------------------------------


class JudgeVerdictEntry(BaseModel):
    """One per predicted Violation in the PaperRunManifest, indexed by (violation_type, location)."""

    violation_type: str
    location: str
    label: Literal["true_positive", "false_positive"]
    notes: Optional[str] = None


class JudgeMissedExpectation(BaseModel):
    """A violation the model SHOULD have produced but didn't (false negative source)."""

    violation_type: str
    location: str
    notes: Optional[str] = None


class JudgeVerdict(BaseModel):
    """Judge's verdict for one PaperRunManifest. Maps to a *.judge.json sidecar."""

    run_id: str
    """MUST equal the manifest.run_id this verdict corresponds to (checked by caller)."""

    entries: List[JudgeVerdictEntry] = Field(default_factory=list)
    """One entry per predicted Violation; label is true_positive or false_positive."""

    missed: List[JudgeMissedExpectation] = Field(default_factory=list)
    """Violations the model should have emitted but did not (FN source)."""

    judge_model_id: str
    """Model (or human identifier) that produced this verdict."""

    judge_provider: Literal["gemini", "ollama", "human"]
    """Provider of the judge. Three-value lock; 'openai' and others are rejected."""


# ---------------------------------------------------------------------------
# Scoring schema
# ---------------------------------------------------------------------------


class PrecisionRecallF1(BaseModel):
    """One scoreline for either a single violation_type or the aggregate."""

    violation_type: str
    """Violation category key, or '__aggregate__' for the overall micro-average."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    precision: float = 0.0
    """TP / (TP + FP). Coerced to 0.0 when TP + FP == 0."""

    recall: float = 0.0
    """TP / (TP + FN). Coerced to 0.0 when TP + FN == 0."""

    f1: float = 0.0
    """2 * precision * recall / (precision + recall). Coerced to 0.0 when both == 0."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_precision(tp: int, fp: int) -> float:
    """TP / (TP + FP), returns 0.0 on division-by-zero."""
    denom = tp + fp
    return tp / denom if denom else 0.0


def _safe_recall(tp: int, fn: int) -> float:
    """TP / (TP + FN), returns 0.0 on division-by-zero."""
    denom = tp + fn
    return tp / denom if denom else 0.0


def _safe_f1(precision: float, recall: float) -> float:
    """2 * P * R / (P + R), returns 0.0 on division-by-zero."""
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


def _make_scoreline(violation_type: str, tp: int, fp: int, fn: int) -> PrecisionRecallF1:
    p = _safe_precision(tp, fp)
    r = _safe_recall(tp, fn)
    f = _safe_f1(p, r)
    return PrecisionRecallF1(
        violation_type=violation_type,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=p,
        recall=r,
        f1=f,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_judge_verdict(path: Path) -> JudgeVerdict:
    """Read a judge verdict JSON file and validate it as a JudgeVerdict.

    Raises:
        ValidationError: if the file contents do not conform to JudgeVerdict.
        FileNotFoundError / OSError: if the file cannot be opened.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    return JudgeVerdict.model_validate(raw)


def compute_metrics(
    manifest: PaperRunManifest,
    verdict: JudgeVerdict,
) -> List[PrecisionRecallF1]:
    """Compute per-violation-type AND micro-aggregate P/R/F1.

    Returns a list with one entry per distinct violation_type seen across
    manifest.violations and verdict.missed, plus a final '__aggregate__' entry.

    Counting:
    - For each Violation in manifest.violations, the matching JudgeVerdictEntry
      is looked up by (violation_type, location). label == 'true_positive' → TP;
      label == 'false_positive' → FP. No match → ValueError.
    - For each JudgeMissedExpectation → FN for its violation_type.

    Division-by-zero is coerced to 0.0; no NaN, no exception.
    """
    # Build lookup: (violation_type, location) -> JudgeVerdictEntry
    entry_index: Dict[Tuple[str, str], JudgeVerdictEntry] = {}
    for entry in verdict.entries:
        key = (entry.violation_type, entry.location)
        entry_index[key] = entry

    # Counters per violation_type
    tp_counts: Dict[str, int] = {}
    fp_counts: Dict[str, int] = {}
    fn_counts: Dict[str, int] = {}

    def _ensure(vtype: str) -> None:
        if vtype not in tp_counts:
            tp_counts[vtype] = 0
            fp_counts[vtype] = 0
            fn_counts[vtype] = 0

    # Process predicted violations
    for violation in manifest.violations:
        key = (violation.violation_type, violation.location)
        entry = entry_index.get(key)
        if entry is None:
            raise ValueError(
                f"predicted violation has no judge entry: "
                f"violation_type={violation.violation_type!r}, "
                f"location={violation.location!r}"
            )
        vtype = violation.violation_type
        _ensure(vtype)
        if entry.label == "true_positive":
            tp_counts[vtype] += 1
        else:  # false_positive
            fp_counts[vtype] += 1

    # Process missed expectations (FN)
    for missed in verdict.missed:
        vtype = missed.violation_type
        _ensure(vtype)
        fn_counts[vtype] += 1

    # Build per-type scorelines
    per_type: List[PrecisionRecallF1] = []
    for vtype in tp_counts:  # all types present in at least one of the three dicts
        score = _make_scoreline(
            vtype,
            tp_counts[vtype],
            fp_counts[vtype],
            fn_counts[vtype],
        )
        per_type.append(score)

    # Append micro-aggregate
    per_type.append(aggregate_scoreline(per_type))
    return per_type


def aggregate_scoreline(per_type: List[PrecisionRecallF1]) -> PrecisionRecallF1:
    """Micro-average P/R/F1 over all per-type scorelines.

    Ignores any entry whose violation_type == '__aggregate__' to ensure
    idempotency (calling this function twice on its own output is safe).

    Returns a single PrecisionRecallF1 with violation_type='__aggregate__'.
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    for score in per_type:
        if score.violation_type == "__aggregate__":
            continue
        total_tp += score.true_positives
        total_fp += score.false_positives
        total_fn += score.false_negatives

    return _make_scoreline("__aggregate__", total_tp, total_fp, total_fn)
