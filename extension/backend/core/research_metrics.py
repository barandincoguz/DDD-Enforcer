"""
Structured research metrics recorder.

Persists analysis-ready JSONL event streams for provider calls, validation runs,
generation runs, and retrieval events. Also provides lightweight summary and CSV
export helpers for experiments and backend endpoints.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import csv
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

from config import ResearchArtifactsConfig


@dataclass
class EventBuffer:
    """In-memory event buffer for a single event type."""

    items: List[Dict[str, Any]] = field(default_factory=list)


class ResearchMetricsStore:
    """Singleton event recorder for research-facing metrics."""

    _instance: Optional["ResearchMetricsStore"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.session_start = datetime.now().isoformat()
        self.provider_calls = EventBuffer()
        self.validation_runs = EventBuffer()
        self.generation_runs = EventBuffer()
        self.retrieval_events = EventBuffer()

        self._event_paths = {
            "provider_calls": ResearchArtifactsConfig.EVENTS_DIR / "provider_calls.jsonl",
            "validation_runs": ResearchArtifactsConfig.EVENTS_DIR / "validation_runs.jsonl",
            "generation_runs": ResearchArtifactsConfig.EVENTS_DIR / "generation_runs.jsonl",
            "retrieval_events": ResearchArtifactsConfig.EVENTS_DIR / "retrieval_events.jsonl",
        }
        for path in self._event_paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "ResearchMetricsStore":
        """Return the global metrics store instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset in-memory state and truncate event streams."""
        with cls._lock:
            existing = cls._instance
            if existing is not None:
                for path in existing._event_paths.values():
                    if path.exists():
                        path.unlink()
            cls._instance = None

    def _append_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        buffer = getattr(self, event_type)
        buffer.items.append(payload)
        path = self._event_paths[event_type]
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def record_provider_call(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("timestamp", datetime.now().isoformat())
        self._append_event("provider_calls", payload)

    def record_validation_run(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("timestamp", datetime.now().isoformat())
        self._append_event("validation_runs", payload)

    def record_generation_run(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("timestamp", datetime.now().isoformat())
        self._append_event("generation_runs", payload)

    def record_retrieval_event(self, payload: Dict[str, Any]) -> None:
        payload.setdefault("timestamp", datetime.now().isoformat())
        self._append_event("retrieval_events", payload)

    def _average(self, values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def _summarize_provider_calls(self) -> Dict[str, Any]:
        calls = self.provider_calls.items
        model_counter = Counter()
        stage_counter = Counter()
        total_cost = 0.0
        parseable = 0
        cached_tokens = 0
        for call in calls:
            model_counter[f"{call.get('provider')}:{call.get('model')}"] += 1
            stage_counter[call.get("stage", "unknown")] += 1
            total_cost += float(call.get("cost_usd", 0.0))
            parseable += 1 if call.get("parse_success") else 0
            cached_tokens += int(call.get("cached_tokens", 0))

        return {
            "total_calls": len(calls),
            "parseable_output_rate_percent": round((parseable / len(calls)) * 100, 2) if calls else 0.0,
            "total_cost_usd": round(total_cost, 8),
            "cached_tokens": cached_tokens,
            "by_model": dict(model_counter),
            "by_stage": dict(stage_counter),
        }

    def _summarize_validation_runs(self) -> Dict[str, Any]:
        runs = self.validation_runs.items
        stage_values: Dict[str, List[float]] = defaultdict(list)
        total_latencies = []
        cost_values = []
        token_values = []
        mode_counter = Counter()
        violation_counter = Counter()
        scaling_points = []

        for run in runs:
            metrics = run.get("metrics", {})
            stage_latencies = metrics.get("stage_latencies_ms", {})
            for stage_name, value in stage_latencies.items():
                stage_values[stage_name].append(float(value))
            total_latencies.append(float(metrics.get("validation_time_ms", 0.0)))
            cost_values.append(float(metrics.get("cost_usd", 0.0)))
            token_values.append(float(metrics.get("llm_total_tokens", 0)))
            mode_counter[run.get("mode", "pipeline")] += 1
            for violation in run.get("violations", []):
                violation_counter[violation.get("type", "Unknown")] += 1
            scaling_points.append(
                {
                    "filename": run.get("filename", ""),
                    "file_size_chars": run.get("file_size_chars", 0),
                    "latency_ms": metrics.get("validation_time_ms", 0.0),
                }
            )

        return {
            "total_runs": len(runs),
            "avg_validation_time_ms": self._average(total_latencies),
            "avg_cost_usd": self._average(cost_values),
            "avg_llm_tokens": self._average(token_values),
            "avg_stage_latencies_ms": {
                stage_name: self._average(values) for stage_name, values in stage_values.items()
            },
            "modes": dict(mode_counter),
            "violation_types": dict(violation_counter),
            "scaling_points": scaling_points,
        }

    def _summarize_generation_runs(self) -> Dict[str, Any]:
        runs = self.generation_runs.items
        stage_values: Dict[str, List[float]] = defaultdict(list)
        totals = []
        cost_values = []
        for run in runs:
            stage_latencies = run.get("stage_latencies_ms", {})
            for stage_name, value in stage_latencies.items():
                stage_values[stage_name].append(float(value))
            totals.append(float(run.get("total_latency_ms", 0.0)))
            cost_values.append(float(run.get("metrics", {}).get("cost_usd", 0.0)))

        return {
            "total_runs": len(runs),
            "avg_total_latency_ms": self._average(totals),
            "avg_cost_usd": self._average(cost_values),
            "avg_stage_latencies_ms": {
                stage_name: self._average(values) for stage_name, values in stage_values.items()
            },
        }

    def _summarize_retrieval_events(self) -> Dict[str, Any]:
        events = self.retrieval_events.items
        latencies = []
        top1_hits = 0
        top3_hits = 0
        comparable = 0
        for event in events:
            latencies.append(float(event.get("latency_ms", 0.0)))
            expected = event.get("expected_sections") or []
            observed = event.get("observed_sections") or []
            if expected:
                comparable += 1
                top1_hits += 1 if observed[:1] and observed[0] in expected else 0
                top3_hits += 1 if any(section in expected for section in observed[:3]) else 0

        return {
            "total_queries": len(events),
            "avg_latency_ms": self._average(latencies),
            "top1_accuracy_percent": round((top1_hits / comparable) * 100, 2) if comparable else None,
            "top3_accuracy_percent": round((top3_hits / comparable) * 100, 2) if comparable else None,
            "comparable_queries": comparable,
        }

    def get_report(self, detailed: bool = False) -> Dict[str, Any]:
        """Return a structured report across all recorded research events."""
        report = {
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "provider_calls": self._summarize_provider_calls(),
            "validation_runs": self._summarize_validation_runs(),
            "generation_runs": self._summarize_generation_runs(),
            "retrieval": self._summarize_retrieval_events(),
        }
        if detailed:
            report["events"] = {
                "provider_calls": self.provider_calls.items,
                "validation_runs": self.validation_runs.items,
                "generation_runs": self.generation_runs.items,
                "retrieval_events": self.retrieval_events.items,
            }
        return report

    def export_csvs(self, output_dir: Path) -> Dict[str, str]:
        """Export each event stream to CSV for analysis workflows."""
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, str] = {}

        for name, buffer in (
            ("provider_calls", self.provider_calls.items),
            ("validation_runs", self.validation_runs.items),
            ("generation_runs", self.generation_runs.items),
            ("retrieval_events", self.retrieval_events.items),
        ):
            csv_path = output_dir / f"{name}.csv"
            rows = self._flatten_records(buffer)
            if rows:
                fieldnames = sorted({key for row in rows for key in row.keys()})
                with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                    handle.write("")
            exported[name] = str(csv_path)

        summary_path = output_dir / "research_metrics_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(self.get_report(detailed=True), handle, indent=2)
        exported["summary"] = str(summary_path)
        return exported

    def _flatten_records(
        self,
        records: List[Dict[str, Any]],
        prefix: str = "",
    ) -> List[Dict[str, Any]]:
        flattened = []
        for record in records:
            flattened.append(self._flatten_record(record, prefix=prefix))
        return flattened

    def _flatten_record(
        self,
        value: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for key, item in value.items():
            flat_key = f"{prefix}{key}"
            if isinstance(item, dict):
                flat.update(self._flatten_record(item, prefix=f"{flat_key}."))
            elif isinstance(item, list):
                flat[flat_key] = json.dumps(item, ensure_ascii=True)
            else:
                flat[flat_key] = item
        return flat
