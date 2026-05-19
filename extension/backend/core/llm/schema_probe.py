"""schema_probe — 6-model × 3-schema smoke CLI for D1 conformance.

Run with:
    cd extension/backend && python -m core.llm.schema_probe [--out runs/probe-YYYYMMDD-HHMMSS.json] [--trials N]

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
      "json_failed": M,
      "transport_error": P,
      "errors": [str, ...],
      "raw_failures": [str, ...],
      "mean_latency_ms": 123.4,
      "total_tokens": 1234,
      "seeds_used": [42, 43, ...]
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
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core.llm import get_client_for_model
from core.llm.gemini import _RUNTIME_FALLBACKS as _GEMINI_FALLBACKS  # private import: paper-integrity check needs source of truth
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


# Constants kept in sync with the manifest. If you change these, update the
# manifest documentation and re-run the probe so the artifact label is honest.
_SEED_BASE: int = 42
_SEED_STRATEGY: str = "per_trial_42_plus_i"
_DEFAULT_TEMPERATURE: float = 0.05


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
    transport_error: int = 0
    errors: List[str] = field(default_factory=list)
    raw_failures: List[str] = field(default_factory=list)
    mean_latency_ms: float = 0.0
    total_tokens: int = 0
    seeds_used: List[int] = field(default_factory=list)


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
    # Wrap client construction — failure counts as transport_error for all trials
    try:
        client = get_client_for_model(model_id)
    except Exception as e:
        result.transport_error = trials
        result.errors.append(
            f"ClientConstructorError: {type(e).__name__}: {e}"
        )
        return result

    seeds = [_SEED_BASE + i for i in range(trials)]
    result.seeds_used = seeds
    latencies: List[float] = []
    for seed in seeds:
        try:
            resp = client.structured_output(
                messages=[{"role": "user", "content": prompt}],
                schema=schema_cls,
                model=model_id,
                seed=seed,
                temperature=_DEFAULT_TEMPERATURE,
            )
        except Exception as e:
            # Provider/transport exceptions (auth, rate-limit, 5xx) go to
            # transport_error; json_failed is reserved for resp.json_failed=True
            result.transport_error += 1
            result.errors.append(f"{type(e).__name__}: {e}")
            continue
        # Defense-in-depth: pre-flight catches declared fallbacks; this guard
        # catches any unexpected resolved-model mismatch during execution
        if resp.model_id != model_id:
            raise RuntimeError(
                f"Runtime fallback fired: requested={model_id!r} "
                f"resolved={resp.model_id!r}. Refusing to mislabel "
                f"D1 lock artifact. Either disable fallback in "
                f"GeminiClient or remove this model from D1."
            )
        latencies.append(resp.latency_ms)
        result.total_tokens += resp.usage.total_tokens
        if resp.json_failed:
            result.json_failed += 1
            if resp.json_fail_reason:
                result.errors.append(resp.json_fail_reason)
            # Capture first 500 chars of raw output for auditability
            raw = (resp.content or "")[:500]
            result.raw_failures.append(raw)
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
# Manifest helpers
# =============================================================================


def _git_head_or_raise() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_dirty_status() -> Tuple[bool, List[str]]:
    """Return (is_dirty, dirty_files) from a single `git status --porcelain` call."""
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not out:
        return False, []
    return True, out.split("\n")


def _pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


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
        default=None,
        help="Output JSON path (default: runs/probe-YYYYMMDD-HHMMSS.json)",
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
    # Load .env before any client resolution so env-only deployments work
    load_dotenv()
    args = _parse_args(argv)

    # Pre-flight: refuse models that have a declared runtime fallback in
    # GeminiClient — paper-integrity for D1 lock requires accurate model labels
    requested_models = list(args.models or MODELS.keys())
    fallback_offenders = [m for m in requested_models if m in _GEMINI_FALLBACKS]
    if fallback_offenders:
        print(
            "ERROR: refusing to probe models with declared runtime "
            "fallback in core/llm/gemini.py — paper-integrity for D1 "
            f"lock requires accurate model labels. Offenders: "
            f"{fallback_offenders}. Each would silently resolve to: "
            + ", ".join(f"{m} -> {_GEMINI_FALLBACKS[m]}" for m in fallback_offenders)
            + ". Options: (a) remove the entry from _RUNTIME_FALLBACKS in "
            "gemini.py for this run; (b) drop these models from --models; "
            "(c) register the resolved model under D1 instead.",
            file=sys.stderr,
        )
        return 1

    # Compute timestamped default output path
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"runs/probe-{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Git provenance — aborts loudly if unavailable (no "unknown" fallback)
    git_commit = _git_head_or_raise()
    git_dirty, git_dirty_files = _git_dirty_status()

    start_iso = _now_iso()
    aborted: Optional[str] = None
    try:
        report = run_probe(
            models=args.models,
            schemas=args.schemas,
            trials=args.trials,
        )
    except RuntimeError as e:
        aborted = str(e)
        report = {
            "timestamp": _now_iso(),
            "trials_per_cell": args.trials,
            "results": [],
            "aborted": str(e),
        }
    end_iso = _now_iso()

    # Warn on high transport_error cells before writing artifacts
    for cell in report["results"]:
        if cell["transport_error"] >= args.trials / 2:
            print(
                f"WARNING: cell {cell['model_id']}×{cell['schema']} "
                f"has {cell['transport_error']}/{cell['trials']} "
                f"transport errors — investigate before treating as data"
            )

    # Write manifest sidecar
    manifest = {
        "timestamp_start": start_iso,
        "timestamp_end": end_iso,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_dirty_files": git_dirty_files,
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "google-genai": _pkg_version("google-genai"),
            "openai": _pkg_version("openai"),
            "pydantic": _pkg_version("pydantic"),
        },
        "models": list(args.models or MODELS.keys()),
        "schemas": list(args.schemas or SCHEMAS.keys()),
        "trials_per_cell": args.trials,
        "seed_strategy": _SEED_STRATEGY,
        "temperature_default": _DEFAULT_TEMPERATURE,
        "prompts": PROMPTS,
        "schema_fidelity_note": (
            "ComplexViolation approximates BoundedContext nesting depth "
            "and constraint variety; not the full production contract."
        ),
        "aborted": aborted,
    }
    manifest_path = out_path.with_name(out_path.stem + ".manifest.json")

    out_path.write_text(json.dumps(report, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path} + {manifest_path}")

    if aborted:
        raise RuntimeError(aborted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
