"""schema_probe — 6-model × 3-schema smoke CLI for D1 conformance.

Run with:
    cd extension/backend && python -m core.llm.schema_probe [--out runs/probe.json] [--trials N]

For each (model, schema) cell the probe calls structured_output with a
minimal prompt and records whether the model returned valid JSON
matching the schema. Output JSON has shape:

{
  "timestamp": "...",
  "trials_per_cell": N,
  "results": [
    {
      "model_id": "gemini-3.1-pro-preview",
      "provider": "gemini",
      "schema": "basic",
      "trials": N,
      "success": k,
      "json_failed": N - k,
      "errors": [str, ...],
      "mean_latency_ms": 123.4,
      "total_tokens": 1234
    },
    ...
  ]
}

The probe deliberately avoids the project's real domain prompts; it
just measures whether each provider+model+schema combination produces
parseable Pydantic JSON. WP-NEW-B (paper-side) consumes the output.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from core.llm import get_client_for_model
from core.llm.registry import MODELS


# =============================================================================
# THE 3 PROBE SCHEMAS — basic, medium, complex
# =============================================================================


class BasicViolation(BaseModel):
    """2 fields, flat shape."""
    name: str = Field(description="Entity name")
    description: str = Field(description="Short description")


class _MediumAttribute(BaseModel):
    name: str
    type: str


class MediumViolation(BaseModel):
    """4 fields, 1 nested list. Mirrors real Entity contract."""
    name: str = Field(description="Entity name")
    description: str = Field(description="Short description")
    confidence: float = Field(ge=0.0, le=1.0)
    attributes: List[_MediumAttribute] = Field(default_factory=list)


class _ComplexSource(BaseModel):
    file: str
    line: int = Field(ge=1)
    rule: str


class _ComplexEntity(BaseModel):
    name: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str
    evidence_sentence_indices: List[int]
    sources: List[_ComplexSource] = Field(default_factory=list)


class ComplexViolation(BaseModel):
    """Deep nested with optionals — mirrors the project's BoundedContext shape."""
    context_name: str
    description: str
    supporting_sentence_ids: List[int] = Field(default_factory=list)
    entities: List[_ComplexEntity]
    allowed_dependencies: Optional[List[str]] = None


SCHEMAS: Dict[str, type] = {
    "basic": BasicViolation,
    "medium": MediumViolation,
    "complex": ComplexViolation,
}


PROMPTS: Dict[str, str] = {
    "basic": (
        "Emit a JSON object matching the schema. Use the example entity "
        "Customer with description 'A buyer in an e-commerce domain'."
    ),
    "medium": (
        "Emit a JSON object matching the schema. Use the example entity "
        "Order with description 'A customer purchase', confidence 0.9, and "
        "two attributes: order_id (string) and total (number)."
    ),
    "complex": (
        "Emit a JSON object matching the schema. Use the bounded context "
        "OrderManagement with two entities (Order with attributes "
        "order_id and total; OrderLine with attributes line_id and "
        "quantity). Each entity must have a confidence between 0 and 1, "
        "a justification string, and at least one evidence_sentence_index. "
        "Sources may be empty."
    ),
}


# =============================================================================
# RUNNER
# =============================================================================


@dataclass
class CellResult:
    model_id: str
    provider: str
    schema: str
    trials: int
    success: int = 0
    json_failed: int = 0
    errors: List[str] = field(default_factory=list)
    mean_latency_ms: float = 0.0
    total_tokens: int = 0


def probe_cell(model_id: str, schema_name: str, trials: int) -> CellResult:
    """Run `trials` calls of (model_id × schema_name) and aggregate."""
    spec = MODELS[model_id]
    schema_cls = SCHEMAS[schema_name]
    prompt = PROMPTS[schema_name]
    result = CellResult(
        model_id=model_id,
        provider=spec.provider,
        schema=schema_name,
        trials=trials,
    )
    client = get_client_for_model(model_id)
    latencies: List[float] = []
    for _ in range(trials):
        try:
            resp = client.structured_output(
                messages=[{"role": "user", "content": prompt}],
                schema=schema_cls,
                model=model_id,
            )
        except Exception as e:
            result.json_failed += 1
            result.errors.append(f"{type(e).__name__}: {e}")
            continue
        latencies.append(resp.latency_ms)
        result.total_tokens += resp.usage.total_tokens
        if resp.json_failed:
            result.json_failed += 1
            if resp.json_fail_reason:
                result.errors.append(resp.json_fail_reason)
        else:
            result.success += 1
    if latencies:
        result.mean_latency_ms = sum(latencies) / len(latencies)
    return result


def run_probe(
    models: Optional[List[str]] = None,
    schemas: Optional[List[str]] = None,
    trials: int = 1,
) -> Dict:
    """Iterate selected (model × schema) cells and aggregate."""
    if models is None:
        models = list(MODELS.keys())
    if schemas is None:
        schemas = list(SCHEMAS.keys())

    results: List[CellResult] = []
    for model_id in models:
        for schema_name in schemas:
            cell = probe_cell(model_id, schema_name, trials)
            results.append(cell)
            print(
                f"[{model_id}] {schema_name}: "
                f"success={cell.success}/{cell.trials} "
                f"json_failed={cell.json_failed} "
                f"mean_latency={cell.mean_latency_ms:.0f}ms"
            )

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trials_per_cell": trials,
        "results": [asdict(r) for r in results],
    }


# =============================================================================
# CLI
# =============================================================================


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m core.llm.schema_probe",
        description="6-model × 3-schema conformance probe for D1.",
    )
    parser.add_argument(
        "--out",
        default="runs/probe.json",
        help="Output JSON path (default: runs/probe.json)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Trials per cell (default: 1)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        help="Subset of models to probe (default: all 6).",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        choices=list(SCHEMAS.keys()),
        help="Subset of schemas to probe (default: basic medium complex).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_probe(
        models=args.models,
        schemas=args.schemas,
        trials=args.trials,
    )
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path} ({len(report['results'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
