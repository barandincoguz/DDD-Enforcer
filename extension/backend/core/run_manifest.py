"""PaperRunManifest — paper-data run record schema + atomic writer.

This is the PUBLIC paper-data record for EMSE submission. It is DISTINCT from
`core.observability.run_manifest.RunManifest`, which is the INTERNAL
observability/telemetry record for stage accounting. The two coexist and serve
different audiences.

WP-01b Task A deliverables:
  1. PaperRunManifest (Pydantic v2 BaseModel)
  2. Violation (nested schema, same module)
  3. write_paper_run_manifest() — atomic per-run writer
  4. SHA-256 helpers: sha256_of_file(), sha256_of_code_tree()
  5. compose_run_id() — filesystem-safe composite key classmethod + standalone
  6. default_runs_root() — env-var-aware path helper
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sentinel: SHA-256 of zero bytes (hashlib.sha256(b"").hexdigest())
# ---------------------------------------------------------------------------
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

# ---------------------------------------------------------------------------
# Read chunk size for streaming SHA-256 computation (64 KiB)
# ---------------------------------------------------------------------------
_CHUNK = 64 * 1024


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def default_runs_root() -> Path:
    """Return the canonical runs root directory.

    Resolution order:
      1. ``DDD_RUNS_DIR`` environment variable (absolute path).
      2. ``<backend_dir>/runs/`` where ``<backend_dir>`` is the directory that
         contains this module (``extension/backend/core/../`` → ``extension/backend/``).
    """
    env_val = os.environ.get("DDD_RUNS_DIR")
    if env_val:
        return Path(env_val)
    # __file__ is extension/backend/core/run_manifest.py → parent.parent = extension/backend/
    return Path(__file__).parent.parent / "runs"


def sha256_of_file(path: Path) -> str:
    """Return the lower-case hex SHA-256 digest of the file at *path*.

    Reads in 64 KiB chunks to stay memory-bounded on large SRS files.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_of_code_tree(root: Path) -> str:
    """Return the lower-case hex SHA-256 of all ``.py`` files under *root*.

    Hash is computed over a deterministic concatenation of:
      ``<relative_path>\\0<file_bytes>\\0``
    for every ``.py`` file found under *root*, sorted lexicographically by
    relative path. This ensures the digest is canonical regardless of filesystem
    enumeration order.

    If *root* does not exist or contains no ``.py`` files, returns the SHA-256
    of zero bytes (the "empty" sentinel) without raising.
    """
    if not root.exists():
        return _EMPTY_SHA256

    py_files = sorted(root.rglob("*.py"), key=lambda p: str(p.relative_to(root)))
    if not py_files:
        return _EMPTY_SHA256

    h = hashlib.sha256()
    for filepath in py_files:
        rel = str(filepath.relative_to(root))
        # Feed: <relative_path>\0
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        # Feed file contents in chunks, then a \0 separator
        with open(filepath, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
        h.update(b"\x00")

    return h.hexdigest()


def compose_run_id(
    pipeline: str,
    model_id: str,
    srs: str,
    timestamp_utc: str,
    seed: str,
) -> str:
    """Build a filesystem-safe composite run identifier.

    Sanitizes each component by replacing ``.``, ``/``, and `` `` with ``_``,
    then joins with ``_``. The result contains only alphanumerics, ``_``, and
    ``-``.

    Args:
        pipeline:      Pipeline tag, e.g. ``"P1"``.
        model_id:      D1 registry key, e.g. ``"gemini-3.1-pro-preview"``.
        srs:           Short SRS identifier, e.g. ``"D1_SRS"``.
        timestamp_utc: ISO-8601 UTC timestamp string.
        seed:          Seed tag, e.g. ``"seed-42"``.

    Returns:
        A path-safe string with no ``/``, ``.``, or space characters.
    """

    def _sanitize(s: str) -> str:
        return re.sub(r"[./\s]", "_", s)

    parts = [pipeline, model_id, srs, timestamp_utc, seed]
    return "_".join(_sanitize(p) for p in parts)


# ---------------------------------------------------------------------------
# Violation — nested schema for PaperRunManifest.violations
# ---------------------------------------------------------------------------


class Violation(BaseModel):
    """A single DDD violation recorded during a paper-data run."""

    violation_type: str
    """Violation category key, e.g. ``"ubiquitous_language_drift"``."""

    location: str
    """File path or fully-qualified symbol identifier where the violation was found."""

    severity: Literal["ERROR", "WARN"]
    """Severity level. Exactly two values per D1 lock; ``"INFO"`` is rejected."""

    message: str
    """Human-readable description of the violation."""

    srs_path: Optional[str] = None
    """Optional path to the SRS passage that grounds this violation (RAG source)."""

    suggestion: Optional[str] = None
    """Optional auto-generated remediation suggestion."""


# ---------------------------------------------------------------------------
# PaperRunManifest — the public paper-data record
# ---------------------------------------------------------------------------


class PaperRunManifest(BaseModel):
    """Per-run paper-data record for the EMSE submission.

    Each run of the DDD-Enforcer pipeline against a specific (model, SRS,
    codebase, seed) configuration produces exactly one manifest. Downstream
    Tasks B/C/D consume these manifests to produce LaTeX table cells — no
    paper table is hand-typed.

    Schema version is frozen at ``"1.0"`` for this WP-01b Task A release.
    """

    # --- Identity ----------------------------------------------------------

    run_id: str
    """Composite, filesystem-safe identifier. Use :func:`compose_run_id` to build."""

    timestamp_utc: str = Field(
        default_factory=lambda: (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    )
    """ISO-8601 UTC timestamp (``Z`` suffix). Auto-populated at instantiation."""

    # --- Pipeline configuration --------------------------------------------

    pipeline: Optional[Literal["P1", "P2", "P3"]] = None
    """Pipeline variant. ``None`` until WP-01d ships P1/P2/P3 classes."""

    model_id: str
    """D1 model registry key, e.g. ``"gemini-3.1-pro-preview"``."""

    provider: Literal["gemini", "ollama"]
    """LLM provider. Two-provider lock per D1; ``"openai"`` and others are rejected."""

    # --- Inputs / provenance -----------------------------------------------

    srs_path: str
    """Absolute or workspace-relative path to the SRS source file."""

    srs_sha256: str
    """64-hex-char lower-case SHA-256 of the SRS file contents. Use :func:`sha256_of_file`."""

    code_root: Optional[str] = None
    """Workspace root of the validated codebase. ``None`` for generation-only runs."""

    code_sha256: Optional[str] = None
    """64-hex-char SHA-256 of all ``.py`` files under ``code_root``. ``None`` iff ``code_root is None``."""

    # --- Outputs -----------------------------------------------------------

    violations: List[Violation] = Field(default_factory=list)
    """DDD violations discovered during the validation phase."""

    # --- Metrics -----------------------------------------------------------

    latency_seconds: float = 0.0
    """Wall-clock duration of the entire run."""

    prompt_tokens: int = 0
    """Sum of prompt tokens across all LLM calls in this run."""

    completion_tokens: int = 0
    """Sum of completion tokens across all LLM calls in this run."""

    cost_usd: float = 0.0
    """Total cost in USD, computed using registry pricing."""

    # --- Cross-run linkage (RQ2/RQ3/RQ4 inputs) ----------------------------

    judge_verdict_path: Optional[str] = None
    """Path to the cross-family judge's verdict JSON (RQ2/RQ3 inputs)."""

    audit_overrides_path: Optional[str] = None
    """Path to the human-rater override file (RQ3)."""

    seed_manifest_path: Optional[str] = None
    """RQ4 only: path to the seed list used for stochastic-sweep variance."""

    # --- Schema bookkeeping ------------------------------------------------

    schema_version: str = "1.0"
    """Frozen at ``"1.0"`` for this WP-01b Task A release."""

    # --- Classmethod helper ------------------------------------------------

    @classmethod
    def compose_run_id(
        cls,
        pipeline: str,
        model_id: str,
        srs: str,
        timestamp_utc: str,
        seed: str,
    ) -> str:
        """Thin wrapper around the module-level :func:`compose_run_id`.

        Provided as a classmethod so callers can use
        ``PaperRunManifest.compose_run_id(...)`` without a separate import.
        """
        return compose_run_id(pipeline, model_id, srs, timestamp_utc, seed)


# ---------------------------------------------------------------------------
# Atomic writer
# ---------------------------------------------------------------------------


def write_paper_run_manifest(
    manifest: PaperRunManifest,
    runs_root: Path,
) -> Path:
    """Write *manifest* to ``runs_root/<run_id>/manifest.json`` atomically.

    Uses the same ``tmp + fsync + os.replace`` pattern as
    ``core.observability.run_manifest.write_manifest_atomic`` (Codex W-7):
    a partial write leaves only the ``.tmp`` file on disk — the target is
    never corrupt.

    Args:
        manifest:  The :class:`PaperRunManifest` to persist.
        runs_root: Parent directory for all run subdirectories.

    Returns:
        The :class:`Path` of the written ``manifest.json`` file.
    """
    target = runs_root / manifest.run_id / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = target.with_suffix(target.suffix + ".tmp")
    payload = manifest.model_dump_json(indent=2, exclude_none=False)

    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())

    os.replace(tmp, target)
    return target
