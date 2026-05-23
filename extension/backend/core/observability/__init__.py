"""WP-CORE-20 — EMSE-grade structured logging & run-manifest.

Public API:
- RunManifest: Pydantic v2 model written one-per-pipeline-run to runs/manifests/.
- StageEmitter: per-run emitter exposing `with emitter.stage(name) as rec:` + call recorders.
- get_current_emitter: contextvar-backed accessor (returns None outside a stage).

See `docs/superpowers/specs/2026-05-23-wp-core-20-emse-grade-logging-design.md` (v2)
for the contract and Codex review history.
"""

from core.observability.emitter import StageEmitter, get_current_emitter
from core.observability.run_manifest import (
    JSONParseFailureRecord,
    LLMAggregate,
    LLMCallRecord,
    RunManifest,
    StageRecord,
    write_manifest_atomic,
)

__all__ = [
    "RunManifest",
    "StageRecord",
    "LLMCallRecord",
    "JSONParseFailureRecord",
    "LLMAggregate",
    "StageEmitter",
    "get_current_emitter",
    "write_manifest_atomic",
]
