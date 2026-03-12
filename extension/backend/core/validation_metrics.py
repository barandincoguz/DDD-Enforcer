"""
Validation metrics tracker.

Tracks validation-level summaries while preserving a history useful for API
inspection. Raw event streams live in `research_metrics.py`; this tracker keeps
the higher-level view used by backend endpoints and docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from typing import Any, Dict, List, Optional


@dataclass
class ValidationRecord:
    """Single validation execution record."""

    timestamp: str
    filename: str
    mode: str
    provider: Optional[str]
    model: Optional[str]
    file_size_chars: int
    code_file_tokens: int
    validation_time_ms: float
    stage_latencies_ms: Dict[str, float]
    llm_input_tokens: int
    llm_output_tokens: int
    llm_total_tokens: int
    cached_tokens: int
    cost_usd: float
    api_calls: int
    parseable_outputs: int
    unparseable_outputs: int
    violations_count: int
    violation_types: List[str]
    has_sources: bool
    retrieval_top1_hit: Optional[bool] = None
    retrieval_top3_hit: Optional[bool] = None


@dataclass
class ValidationStats:
    """Aggregated validation statistics."""

    total_validations: int = 0
    total_violations_found: int = 0
    files_with_violations: int = 0
    files_without_violations: int = 0
    validation_modes: Dict[str, int] = field(default_factory=dict)
    validation_history: List[ValidationRecord] = field(default_factory=list)
    violation_type_counts: Dict[str, int] = field(default_factory=dict)
    total_validation_time_ms: float = 0.0
    total_code_size_chars: int = 0
    total_code_tokens: int = 0
    total_llm_input_tokens: int = 0
    total_llm_output_tokens: int = 0
    total_llm_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    total_api_calls: int = 0
    parseable_outputs: int = 0
    unparseable_outputs: int = 0
    validations_with_sources: int = 0
    stage_totals_ms: Dict[str, float] = field(default_factory=dict)
    retrieval_comparable_runs: int = 0
    retrieval_top1_hits: int = 0
    retrieval_top3_hits: int = 0


class ValidationMetricsTracker:
    """Singleton validation metrics tracker."""

    _instance: Optional["ValidationMetricsTracker"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.stats = ValidationStats()
        self.session_start = datetime.now().isoformat()

    @classmethod
    def get_instance(cls) -> "ValidationMetricsTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def track_validation(
        self,
        *,
        filename: str,
        file_size_chars: int,
        code_file_tokens: int,
        validation_time_ms: float,
        violations: List[Dict[str, Any]],
        has_sources: bool = False,
        mode: str = "pipeline",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        stage_latencies_ms: Optional[Dict[str, float]] = None,
        llm_input_tokens: int = 0,
        llm_output_tokens: int = 0,
        llm_total_tokens: int = 0,
        cached_tokens: int = 0,
        cost_usd: float = 0.0,
        api_calls: int = 0,
        parseable_outputs: int = 0,
        unparseable_outputs: int = 0,
        retrieval_top1_hit: Optional[bool] = None,
        retrieval_top3_hit: Optional[bool] = None,
    ) -> None:
        with self._lock:
            stage_latencies_ms = stage_latencies_ms or {}
            violations_count = len(violations)
            violation_types = [violation.get("type", "Unknown") for violation in violations]

            record = ValidationRecord(
                timestamp=datetime.now().isoformat(),
                filename=filename,
                mode=mode,
                provider=provider,
                model=model,
                file_size_chars=file_size_chars,
                code_file_tokens=code_file_tokens,
                validation_time_ms=validation_time_ms,
                stage_latencies_ms=stage_latencies_ms,
                llm_input_tokens=llm_input_tokens,
                llm_output_tokens=llm_output_tokens,
                llm_total_tokens=llm_total_tokens,
                cached_tokens=cached_tokens,
                cost_usd=cost_usd,
                api_calls=api_calls,
                parseable_outputs=parseable_outputs,
                unparseable_outputs=unparseable_outputs,
                violations_count=violations_count,
                violation_types=violation_types,
                has_sources=has_sources,
                retrieval_top1_hit=retrieval_top1_hit,
                retrieval_top3_hit=retrieval_top3_hit,
            )

            self.stats.total_validations += 1
            self.stats.total_violations_found += violations_count
            self.stats.total_validation_time_ms += validation_time_ms
            self.stats.total_code_size_chars += file_size_chars
            self.stats.total_code_tokens += code_file_tokens
            self.stats.total_llm_input_tokens += llm_input_tokens
            self.stats.total_llm_output_tokens += llm_output_tokens
            self.stats.total_llm_tokens += llm_total_tokens
            self.stats.total_cached_tokens += cached_tokens
            self.stats.total_cost_usd += cost_usd
            self.stats.total_api_calls += api_calls
            self.stats.parseable_outputs += parseable_outputs
            self.stats.unparseable_outputs += unparseable_outputs
            self.stats.validation_modes[mode] = self.stats.validation_modes.get(mode, 0) + 1

            if violations_count > 0:
                self.stats.files_with_violations += 1
            else:
                self.stats.files_without_violations += 1

            if has_sources:
                self.stats.validations_with_sources += 1

            for stage_name, stage_ms in stage_latencies_ms.items():
                self.stats.stage_totals_ms[stage_name] = (
                    self.stats.stage_totals_ms.get(stage_name, 0.0) + float(stage_ms)
                )

            for violation_type in violation_types:
                self.stats.violation_type_counts[violation_type] = (
                    self.stats.violation_type_counts.get(violation_type, 0) + 1
                )

            if retrieval_top1_hit is not None or retrieval_top3_hit is not None:
                self.stats.retrieval_comparable_runs += 1
                self.stats.retrieval_top1_hits += 1 if retrieval_top1_hit else 0
                self.stats.retrieval_top3_hits += 1 if retrieval_top3_hit else 0

            self.stats.validation_history.append(record)
            self._auto_export()

    def _auto_export(self) -> None:
        def _export_in_background() -> None:
            try:
                from pathlib import Path

                backend_dir = Path(__file__).parent.parent
                export_path = backend_dir / "validation_metrics_report.json"
                with open(export_path, "w", encoding="utf-8") as handle:
                    json.dump(self.get_report(detailed=True), handle, indent=2)
            except Exception as exc:
                print(f"[Metrics] Export warning: {exc}")

        thread = threading.Thread(target=_export_in_background, daemon=True)
        thread.start()

    def get_report(self, detailed: bool = False) -> Dict[str, Any]:
        with self._lock:
            total = self.stats.total_validations
            avg = lambda value: round(value / total, 4) if total else 0.0
            report = {
                "session_start": self.session_start,
                "session_end": datetime.now().isoformat(),
                "summary": {
                    "total_validations": total,
                    "files_with_violations": self.stats.files_with_violations,
                    "files_without_violations": self.stats.files_without_violations,
                    "violation_rate_percent": round(
                        (self.stats.files_with_violations / total) * 100, 2
                    )
                    if total
                    else 0.0,
                    "total_violations_found": self.stats.total_violations_found,
                    "avg_violations_per_file": avg(self.stats.total_violations_found),
                    "validation_modes": self.stats.validation_modes,
                },
                "performance": {
                    "avg_validation_time_ms": avg(self.stats.total_validation_time_ms),
                    "total_validation_time_ms": round(self.stats.total_validation_time_ms, 4),
                    "avg_code_size_chars": avg(self.stats.total_code_size_chars),
                    "total_code_size_chars": self.stats.total_code_size_chars,
                    "avg_code_tokens": avg(self.stats.total_code_tokens),
                    "total_code_tokens": self.stats.total_code_tokens,
                    "avg_stage_latencies_ms": {
                        stage_name: avg(total_ms)
                        for stage_name, total_ms in self.stats.stage_totals_ms.items()
                    },
                },
                "llm_usage": {
                    "avg_input_tokens": avg(self.stats.total_llm_input_tokens),
                    "avg_output_tokens": avg(self.stats.total_llm_output_tokens),
                    "avg_total_tokens": avg(self.stats.total_llm_tokens),
                    "avg_cached_tokens": avg(self.stats.total_cached_tokens),
                    "avg_cost_usd": avg(self.stats.total_cost_usd),
                    "avg_api_calls": avg(self.stats.total_api_calls),
                    "parseable_output_rate_percent": round(
                        (self.stats.parseable_outputs / max(self.stats.parseable_outputs + self.stats.unparseable_outputs, 1))
                        * 100,
                        2,
                    ),
                },
                "rag_integration": {
                    "validations_with_sources": self.stats.validations_with_sources,
                    "source_attachment_rate_percent": round(
                        (self.stats.validations_with_sources / total) * 100,
                        2,
                    )
                    if total
                    else 0.0,
                    "retrieval_top1_accuracy_percent": round(
                        (self.stats.retrieval_top1_hits / self.stats.retrieval_comparable_runs) * 100,
                        2,
                    )
                    if self.stats.retrieval_comparable_runs
                    else None,
                    "retrieval_top3_accuracy_percent": round(
                        (self.stats.retrieval_top3_hits / self.stats.retrieval_comparable_runs) * 100,
                        2,
                    )
                    if self.stats.retrieval_comparable_runs
                    else None,
                },
                "violation_breakdown": self.stats.violation_type_counts,
            }

            if detailed:
                report["validation_history"] = [
                    {
                        "timestamp": rec.timestamp,
                        "filename": rec.filename,
                        "mode": rec.mode,
                        "provider": rec.provider,
                        "model": rec.model,
                        "file_size_chars": rec.file_size_chars,
                        "code_file_tokens": rec.code_file_tokens,
                        "validation_time_ms": rec.validation_time_ms,
                        "stage_latencies_ms": rec.stage_latencies_ms,
                        "llm_input_tokens": rec.llm_input_tokens,
                        "llm_output_tokens": rec.llm_output_tokens,
                        "llm_total_tokens": rec.llm_total_tokens,
                        "cached_tokens": rec.cached_tokens,
                        "cost_usd": rec.cost_usd,
                        "api_calls": rec.api_calls,
                        "parseable_outputs": rec.parseable_outputs,
                        "unparseable_outputs": rec.unparseable_outputs,
                        "violations_count": rec.violations_count,
                        "violation_types": rec.violation_types,
                        "has_sources": rec.has_sources,
                        "retrieval_top1_hit": rec.retrieval_top1_hit,
                        "retrieval_top3_hit": rec.retrieval_top3_hit,
                    }
                    for rec in self.stats.validation_history
                ]
            return report

    def print_summary(self) -> None:
        report = self.get_report(detailed=False)
        print("\n" + "=" * 70)
        print("VALIDATION METRICS SUMMARY")
        print("=" * 70)
        print(json.dumps(report["summary"], indent=2))
        print(json.dumps(report["performance"], indent=2))
        print(json.dumps(report["llm_usage"], indent=2))
        print(json.dumps(report["rag_integration"], indent=2))
        print("=" * 70 + "\n")
