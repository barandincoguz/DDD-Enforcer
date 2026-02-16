# Validation Metrics

## Overview

Automatically tracks validation statistics for every code validation request. Similar to token tracking, metrics are saved to `validation_metrics_report.json` after each validation.

## How It Works

**Automatic Tracking**: When you validate code (via VS Code extension save), the system automatically:

1. Records filename, code size, validation time
2. Counts violations and types
3. Tracks if RAG sources were attached
4. Updates `validation_metrics_report.json` file

**Just like token tracking** - no manual action needed!

## Report Contents

`validation_metrics_report.json` includes:

```json
{
  "session_start": "2025-12-16T10:00:00",
  "session_end": "2025-12-16T11:30:00",
  "summary": {
    "total_validations": 15,
    "files_with_violations": 10,
    "files_without_violations": 5,
    "violation_rate_percent": 66.7,
    "total_violations_found": 28,
    "avg_violations_per_file": 1.87
  },
  "performance": {
    "avg_validation_time_ms": 245.3,
    "avg_code_size_chars": 1250
  },
  "rag_integration": {
    "validations_with_sources": 8,
    "source_attachment_rate_percent": 53.3
  },
  "violation_breakdown": {
    "SynonymViolation": 10,
    "BannedTermViolation": 8,
    "ContextBoundaryViolation": 6,
    "ValueObjectViolation": 4
  },
  "validation_history": [...]
}
```

## Console Summary

Print summary to console anytime:

```python
from core.validation_metrics import ValidationMetricsTracker

tracker = ValidationMetricsTracker.get_instance()
tracker.print_summary()
```

Output:

```
======================================================================
📊 VALIDATION METRICS SUMMARY
======================================================================
  Total Validations: 15
  Files with Violations: 10 (66.7%)
  Clean Files: 5
  Total Violations: 28
  Avg Violations/File: 1.87

  Performance:
    Avg Time: 245.30ms
    Avg Code Size: 1250 chars

  RAG Integration:
    Sources Attached: 8 (53.3%)

  Violation Types:
    SynonymViolation: 10
    BannedTermViolation: 8
    ContextBoundaryViolation: 6
    ValueObjectViolation: 4
======================================================================
```

## Files Generated

- `validation_metrics_report.json` - Updated after each validation
- `token_usage_report.json` - Updated after domain model generation
