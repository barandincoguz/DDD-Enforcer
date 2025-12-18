"""
Validation Metrics Tracker

Tracks validation statistics for UBMK presentation.
Records every validation request, violations found, and processing metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json
import threading


@dataclass
class ValidationRecord:
    """Single validation request record."""
    
    timestamp: str
    filename: str
    file_size_chars: int
    code_file_tokens: int  # Token count of the validated code file
    validation_time_ms: float
    violations_count: int
    violation_types: List[str]
    has_sources: bool  # Whether RAG sources were added


@dataclass
class ValidationStats:
    """Aggregated validation statistics."""
    
    total_validations: int = 0
    total_violations_found: int = 0
    files_with_violations: int = 0
    files_without_violations: int = 0
    
    # Per violation type breakdown
    violation_type_counts: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    avg_validation_time_ms: float = 0.0
    total_validation_time_ms: float = 0.0
    
    # RAG metrics
    validations_with_sources: int = 0
    
    # File metrics
    total_code_size_chars: int = 0
    avg_code_size_chars: float = 0.0
    
    # Token metrics
    total_code_tokens: int = 0
    avg_code_tokens: float = 0.0
    
    # Detailed history
    validation_history: List[ValidationRecord] = field(default_factory=list)


class ValidationMetricsTracker:
    """
    Singleton tracker for validation metrics.
    
    Tracks every validation request for UBMK presentation reporting.
    Thread-safe for concurrent validation requests.
    """
    
    _instance: Optional['ValidationMetricsTracker'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.stats = ValidationStats()
        self.session_start = datetime.now().isoformat()
    
    @classmethod
    def get_instance(cls) -> 'ValidationMetricsTracker':
        """Singleton pattern to ensure single tracker."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset tracker (useful for testing)."""
        with cls._lock:
            cls._instance = None
    
    def track_validation(
        self,
        filename: str,
        file_size_chars: int,
        code_file_tokens: int,
        validation_time_ms: float,
        violations: List[Dict],
        has_sources: bool = False
    ) -> None:
        """
        Track a single validation request.
        
        Args:
            filename: Name of the validated file
            file_size_chars: Size of code in characters
            code_file_tokens: Token count of the code file
            validation_time_ms: Time taken for validation
            violations: List of violations found
            has_sources: Whether RAG sources were attached
        """
        with self._lock:
            violations_count = len(violations)
            violation_types = [v.get("type", "Unknown") for v in violations]
            
            # Create record
            record = ValidationRecord(
                timestamp=datetime.now().isoformat(),
                filename=filename,
                file_size_chars=file_size_chars,
                code_file_tokens=code_file_tokens,
                validation_time_ms=validation_time_ms,
                violations_count=violations_count,
                violation_types=violation_types,
                has_sources=has_sources
            )
            
            # Update aggregates
            self.stats.total_validations += 1
            self.stats.total_violations_found += violations_count
            
            if violations_count > 0:
                self.stats.files_with_violations += 1
            else:
                self.stats.files_without_violations += 1
            
            # Update violation type counts
            for vtype in violation_types:
                self.stats.violation_type_counts[vtype] = \
                    self.stats.violation_type_counts.get(vtype, 0) + 1
            
            # Update performance metrics
            self.stats.total_validation_time_ms += validation_time_ms
            self.stats.avg_validation_time_ms = \
                self.stats.total_validation_time_ms / self.stats.total_validations
            
            # Update file metrics
            self.stats.total_code_size_chars += file_size_chars
            self.stats.avg_code_size_chars = \
                self.stats.total_code_size_chars / self.stats.total_validations
            
            # Update token metrics
            self.stats.total_code_tokens += code_file_tokens
            self.stats.avg_code_tokens = \
                self.stats.total_code_tokens / self.stats.total_validations
            
            # Update RAG metrics
            if has_sources:
                self.stats.validations_with_sources += 1
            
            # Add to history
            self.stats.validation_history.append(record)
            
            # Auto-export after each validation (like token_tracker)
            self._auto_export()
    
    def _auto_export(self) -> None:
        """Automatically export report to JSON file (like token_tracker does)."""
        def _export_in_background():
            try:
                # Import here to avoid circular dependency
                from pathlib import Path
                backend_dir = Path(__file__).parent.parent
                export_path = backend_dir / "validation_metrics_report.json"
                
                report = self.get_report(detailed=True)
                with open(export_path, 'w') as f:
                    json.dump(report, f, indent=2)
            except Exception as e:
                # Silent fail, don't break validation
                print(f"[Metrics] Export warning: {e}")
        
        # Run export in background thread to avoid blocking
        import threading
        thread = threading.Thread(target=_export_in_background, daemon=True)
        thread.start()
    
    def get_report(self, detailed: bool = False) -> Dict:
        """
        Generate comprehensive validation metrics report.
        
        Args:
            detailed: Include per-validation breakdown
            
        Returns:
            Dict with all statistics
        """
        with self._lock:
            violation_rate = 0.0
            if self.stats.total_validations > 0:
                violation_rate = (self.stats.files_with_violations / 
                                self.stats.total_validations) * 100
            
            avg_violations_per_file = 0.0
            if self.stats.total_validations > 0:
                avg_violations_per_file = (self.stats.total_violations_found / 
                                          self.stats.total_validations)
            
            report = {
                "session_start": self.session_start,
                "session_end": datetime.now().isoformat(),
                "summary": {
                    "total_validations": self.stats.total_validations,
                    "files_with_violations": self.stats.files_with_violations,
                    "files_without_violations": self.stats.files_without_violations,
                    "violation_rate_percent": round(violation_rate, 2),
                    "total_violations_found": self.stats.total_violations_found,
                    "avg_violations_per_file": round(avg_violations_per_file, 2),
                },
                "performance": {
                    "avg_validation_time_ms": round(self.stats.avg_validation_time_ms, 2),
                    "total_validation_time_ms": round(self.stats.total_validation_time_ms, 2),
                    "avg_code_size_chars": round(self.stats.avg_code_size_chars, 2),
                    "total_code_size_chars": self.stats.total_code_size_chars,
                    "avg_code_tokens": round(self.stats.avg_code_tokens, 2),
                    "total_code_tokens": self.stats.total_code_tokens,
                },
                "rag_integration": {
                    "validations_with_sources": self.stats.validations_with_sources,
                    "source_attachment_rate_percent": round(
                        (self.stats.validations_with_sources / max(self.stats.total_validations, 1)) * 100, 
                        2
                    ),
                },
                "violation_breakdown": self.stats.violation_type_counts,
            }
            
            # Add detailed history if requested
            if detailed:
                report["validation_history"] = [
                    {
                        "timestamp": rec.timestamp,
                        "filename": rec.filename,
                        "file_size_chars": rec.file_size_chars,
                        "code_file_tokens": rec.code_file_tokens,
                        "validation_time_ms": rec.validation_time_ms,
                        "violations_count": rec.violations_count,
                        "violation_types": rec.violation_types,
                        "has_sources": rec.has_sources,
                    }
                    for rec in self.stats.validation_history
                ]
            
            return report
    
    def print_summary(self) -> None:
        """Print formatted summary to console."""
        report = self.get_report()
        
        print("\n" + "="*70)
        print("📊 VALIDATION METRICS SUMMARY")
        print("="*70)
        
        summary = report["summary"]
        print(f"  Total Validations: {summary['total_validations']}")
        print(f"  Files with Violations: {summary['files_with_violations']} "
              f"({summary['violation_rate_percent']}%)")
        print(f"  Clean Files: {summary['files_without_violations']}")
        print(f"  Total Violations: {summary['total_violations_found']}")
        print(f"  Avg Violations/File: {summary['avg_violations_per_file']}")
        
        print("\n  Performance:")
        perf = report["performance"]
        print(f"    Avg Time: {perf['avg_validation_time_ms']:.2f}ms")
        print(f"    Avg Code Size: {perf['avg_code_size_chars']:.0f} chars")
        
        print("\n  RAG Integration:")
        rag = report["rag_integration"]
        print(f"    Sources Attached: {rag['validations_with_sources']} "
              f"({rag['source_attachment_rate_percent']:.1f}%)")
        
        print("\n  Violation Types:")
        for vtype, count in sorted(report["violation_breakdown"].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"    {vtype}: {count}")
        
        print("="*70 + "\n")
