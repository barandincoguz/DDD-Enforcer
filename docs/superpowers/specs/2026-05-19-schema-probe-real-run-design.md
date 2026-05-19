# schema_probe Real Run (WP-NEW-B Stage 1) — Design

**Status**: Approved (Codex 5.5-xhigh juri review applied 2026-05-19)
**Branch policy**: feature branch `feat/schema-probe-real-run` → FF merge to main
**Commits**: 2 (code fix + run artifact)

---

## 1. Goal

Produce D1 6-model × 3-schema conformance data for the EMSE paper RQ2 Table 7 (`json_failed` column source data). Output the run as a reproducible artifact committed to the repo.

**Deliverables**:
- `runs/probe-{ts}.json` — per-cell aggregated results
- `runs/probe-{ts}.manifest.json` — reproducibility metadata sidecar
- 2 commits on `main` (code fix; artifact)

## 2. Scope

**In scope**:
- Minimal patch to `extension/backend/core/llm/schema_probe.py`
- New smoke tests in `extension/backend/tests/test_llm_schema_probe.py`
- 1 pre-flight smoke (2 cells: 1 Gemini + 1 Ollama) — validates plumbing
- 1 full run (18 cells × 5 trials = 90 calls)

**Out of scope**:
- Paper-side Markdown table render (separate WP-NEW-B Stage 2)
- `--seed` CLI flag for variance ablation studies
- Per-cell checkpointing / resume support
- Token tracker integration
- Cross-prompt sensitivity (deferred to WP-NEW-C)

## 3. Juri review (Codex 5.5-xhigh) — accepted concerns

| # | Severity | Concern | Resolution |
|---|---|---|---|
| 1 | BLOCKER | Runtime fallback `gemini-3.1-flash-lite → gemini-2.5-flash` silently mislabels artifact | `probe_cell` asserts `resp.model_id == requested`; mismatch raises `RuntimeError` → probe stops loudly |
| 2 | BLOCKER | Current `json_failed` counter mixes schema-invalid output with transport errors (auth, 429, 5xx) | `CellResult` gains `transport_error` counter. Provider exceptions go there; only `resp.json_failed=True` increments `json_failed` |
| 3 | HIGH | Pre-flight on Gemini only leaves all 12 Ollama cells unvalidated | Pre-flight expanded to 2 cells: `gemini-3.1-flash-lite × basic` AND `gpt-oss:120b-cloud × basic` |
| 4 | HIGH | `get_client_for_model()` constructor failure aborts cell silently | Constructor wrapped in try/except; failure → cell records `transport_error = trials`, probe continues |
| 5 | HIGH | `git_commit: "unknown"` fallback lets unprovenanced artifact ship | No fallback. `git rev-parse HEAD` failure raises. Working tree status captured: `git_dirty` bool + dirty file list |
| 6 | MEDIUM | Manifest missing exact SDK versions | Add `package_versions: {google-genai, openai, pydantic}` via `importlib.metadata` |
| 7 | MEDIUM | Acceptance criterion `success+json_failed==N` accepts all-transport-error cells as complete | Refined: `success+json_failed+transport_error==N`; **warning** logged if any cell has `transport_error ≥ N/2` |
| 8 | MEDIUM | "mirrors BoundedContext shape" overclaims fidelity | Spec/manifest copy: "approximates BoundedContext nesting depth and constraint variety" |
| 9 | MEDIUM (override) | No raw-response capture means reviewers cannot audit failure modes | **Override applied (autonomous mode)**: truncated raw output (500 char) captured on `json_failed=True` only. Adds ~10 KB to artifact, large audit value |
| 10 | QUESTION | Fixed `seed=42` across 5 trials = reproducibility test, not 5 independent samples | Per-trial seed: `seeds = [42, 43, 44, 45, 46]`. Manifest records the list |

Codex concern #5 (clean tree provenance) and #1 (model labeling) together raise the artifact's evidentiary bar high enough for EMSE reviewers.

## 4. Architecture

### 4.1 Code changes (`core/llm/schema_probe.py`)

```python
# new imports
from dotenv import load_dotenv
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata

# CellResult gains transport_error + raw_failures
@dataclass
class CellResult:
    model_id: str
    provider: str
    schema: str
    trials: int
    success: int = 0
    json_failed: int = 0
    transport_error: int = 0          # NEW
    errors: List[str] = field(default_factory=list)
    raw_failures: List[str] = field(default_factory=list)  # NEW, truncated 500ch
    mean_latency_ms: float = 0.0
    total_tokens: int = 0
    seeds_used: List[int] = field(default_factory=list)    # NEW

# probe_cell revisions
def probe_cell(model_id: str, schema_name: str, trials: int) -> CellResult:
    spec = MODELS[model_id]
    schema_cls = SCHEMAS[schema_name]
    prompt = PROMPTS[schema_name]
    result = CellResult(
        model_id=model_id, provider=spec.provider,
        schema=schema_name, trials=trials,
    )

    try:
        client = get_client_for_model(model_id)
    except Exception as e:                                 # NEW — H4
        result.transport_error = trials
        result.errors.append(f"ClientConstructorError: {type(e).__name__}: {e}")
        return result

    seeds = [42 + i for i in range(trials)]                # NEW — Q11
    result.seeds_used = seeds
    latencies: List[float] = []
    for seed in seeds:
        try:
            resp = client.structured_output(
                messages=[{"role": "user", "content": prompt}],
                schema=schema_cls,
                model=model_id,
                seed=seed,
            )
        except Exception as e:
            result.transport_error += 1                    # CHANGED — B2
            result.errors.append(f"{type(e).__name__}: {e}")
            continue

        # B1: hard fail on silent runtime fallback
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
            # M10: capture truncated raw output for audit
            raw = (resp.content or "")[:500]
            result.raw_failures.append(raw)
        else:
            result.success += 1
    if latencies:
        result.mean_latency_ms = sum(latencies) / len(latencies)
    return result

# main()
def main(argv=None) -> int:
    load_dotenv()                                          # NEW — required
    args = _parse_args(argv)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"runs/probe-{ts}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_iso = _now_iso()
    report = run_probe(
        models=args.models,
        schemas=args.schemas,
        trials=args.trials,
    )
    end_iso = _now_iso()

    # M7-style transport_error warning loop
    for cell in report["results"]:
        if cell["transport_error"] >= args.trials / 2:
            print(
                f"WARNING: cell {cell['model_id']}×{cell['schema']} "
                f"has {cell['transport_error']}/{cell['trials']} "
                f"transport errors — investigate before treating as data"
            )

    manifest = {
        "timestamp_start": start_iso,
        "timestamp_end": end_iso,
        "git_commit": _git_head_or_raise(),                # H5: no fallback
        "git_dirty": _git_is_dirty(),
        "git_dirty_files": _git_dirty_files() if _git_is_dirty() else [],
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": {                              # M7
            "google-genai": _pkg_version("google-genai"),
            "openai": _pkg_version("openai"),
            "pydantic": _pkg_version("pydantic"),
        },
        "models": list(args.models or MODELS.keys()),
        "schemas": list(args.schemas or SCHEMAS.keys()),
        "trials_per_cell": args.trials,
        "seed_strategy": "per_trial_42_plus_i",
        "temperature_default": 0.05,
        "prompts": PROMPTS,
        "schema_fidelity_note": (
            "ComplexViolation approximates BoundedContext nesting depth "
            "and constraint variety; not the full production contract."
        ),                                                  # M9
    }
    manifest_path = out_path.with_suffix(".manifest.json")

    out_path.write_text(json.dumps(report, indent=2))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_path} + {manifest_path}")
    return 0

# helpers (private, module-level)
def _git_head_or_raise() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()

def _git_is_dirty() -> bool:
    return bool(subprocess.run(
        ["git", "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip())

def _git_dirty_files() -> List[str]:
    return subprocess.run(
        ["git", "status", "--porcelain"], check=True,
        capture_output=True, text=True,
    ).stdout.strip().split("\n")

def _pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
```

### 4.2 Smoke tests (`tests/test_llm_schema_probe.py`)

8 tests (TDD: write first, then implementation):

1. `test_main_loads_dotenv` — `patch("core.llm.schema_probe.load_dotenv")`, invoke `main(["--trials","0",...])` with mocked client, assert called
2. `test_default_out_path_has_timestamp` — regex `^runs/probe-\d{8}-\d{6}\.json$`
3. `test_manifest_sidecar_written_with_all_keys` — invoke `main(...)`, parse manifest, assert all 14 keys present
4. `test_transport_error_separated_from_json_failed` — client raises `RateLimitError`, assert `transport_error+=1` and `json_failed==0`
5. `test_client_construction_failure_yields_transport_error_cell` — `get_client_for_model` raises, assert cell `transport_error == trials`, probe completes
6. `test_runtime_fallback_raises_loudly` — mock client returns `resp.model_id="other"`, assert `RuntimeError` propagates
7. `test_seed_varies_per_trial` — mock client records kwargs, assert `seeds=[42,43,44]` for trials=3
8. `test_raw_output_captured_on_json_failed` — mock returns `json_failed=True, content="bad{...}"`, assert truncated raw in `raw_failures`
9. `test_git_head_failure_aborts_run` — patch subprocess to raise, assert `CalledProcessError` propagates
10. `test_transport_warning_printed` — synthesize cell with `transport_error >= trials/2`, capture stdout, assert "WARNING:" line

All mocked, no live LLM. CI green required.

### 4.3 Pre-flight smoke (no commit)

```bash
cd extension/backend
source .venv/bin/activate
python -m core.llm.schema_probe \
  --models gemini-3.1-flash-lite gpt-oss:120b-cloud \
  --schemas basic \
  --trials 1
```

Expected: 2 cells, each `success=1/1`, manifest written, both providers' auth verified.

If fails: STOP. Diagnose. Common causes — dotenv path, Ollama model 404, Gemini auth.

### 4.4 Full run (no commit)

```bash
python -m core.llm.schema_probe --trials 5
```

Expected: 18 cells, 90 calls, ~15-20 min. Per-cell stdout. Final output:
- `runs/probe-{ts}.json` (~100 KB)
- `runs/probe-{ts}.manifest.json` (~4 KB)

### 4.5 Commit structure

**Commit 1** — `feat(llm/schema_probe): dotenv + manifest + transport_error split + fallback guard (WP-NEW-B prep)`
- `core/llm/schema_probe.py` (modified)
- `tests/test_llm_schema_probe.py` (new or extended)
- Working tree currently dirty (`AGENTS.md`, 2 JSON artifacts) — **do not stage**, only stage these 2 files

**Commit 2** — `chore(artifacts): schema_probe full run N=5 6×3 (WP-NEW-B Stage 1)`
- `runs/probe-{ts}.json`
- `runs/probe-{ts}.manifest.json`
- Only stage these 2 files

## 5. Error handling policy

| Error class | Behavior | Rationale |
|---|---|---|
| Pre-flight failure | STOP. Diagnose. Fix. Re-test. | Plumbing must work before 90-call run |
| Client constructor exception | Record cell `transport_error = trials`, probe continues | Cell-level isolation |
| Per-trial exception (RateLimit, Auth, 5xx, timeout) | `transport_error += 1`, error string appended, continue | Distinguished from schema failures |
| Per-trial `resp.json_failed = True` | `json_failed += 1`, raw output (500ch) captured | This is the **target metric** |
| Runtime fallback fired (`resp.model_id != requested`) | `raise RuntimeError` — probe aborts immediately | D1 lock integrity is non-negotiable for paper |
| `git rev-parse` failure | `CalledProcessError` propagates — probe aborts | No unprovenanced artifact |
| `transport_error >= N/2` for a cell | Warning logged to stdout, probe continues | Reviewer signal: cell may be unusable data |

## 6. Acceptance criteria

- [ ] Commit 1 lands: 224 + 10 new tests green, `pyright` continue-on-error ok
- [ ] Pre-flight: 2 cells, `success == 1/1` each
- [ ] Full run: 18 cells produced
- [ ] Per-cell invariant: `success + json_failed + transport_error == 5`
- [ ] Manifest has 15 keys: timestamps (×2: start, end), git provenance (×3: commit, dirty, dirty_files), system (×3: python_version, platform, package_versions), run config (×5: models, schemas, trials_per_cell, seed_strategy, temperature_default), prompts, schema_fidelity_note
- [ ] `git_dirty` honest (not silently false)
- [ ] No cell with `transport_error >= 3` (50% of N=5); if any → investigate before Commit 2
- [ ] Commit 2 stages only the 2 probe files; no AGENTS.md, no validation_metrics_report.json, no intermediate AST JSON

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `gemini-3.1-pro-preview` access revoked (preview lifecycle) | Med | G1 cells transport_error | No silent fallback wired; manifest shows transport failures honestly; follow-up commit decides replacement |
| `gemma4:31b-cloud` / `qwen3-coder-next:cloud` IDs not on Ollama Cloud | Med | Up to 6 Ollama cells transport_error | Pre-flight only covers `gpt-oss:120b-cloud`; if it works, other Ollama IDs are likely fine but not guaranteed. Surfaces in full run; D1 may need ID corrections |
| `gemini-3.1-flash-lite` triggers silent fallback to `gemini-2.5-flash` | High | Probe aborts (intentional) | Resolution: either disable fallback in GeminiClient (preferred for paper) or remove flash-lite from D1. **Decision deferred to user when probe aborts.** |
| Gemini single-key rate limit | Low (paid tier) | Slower run | Retry decorator backoff |
| Working tree stays dirty during run | Cert. | Manifest records `git_dirty: true` with file list | Reviewers see honest provenance; if blocking, separate `chore(artifacts):` commit first to clean tree, then run |
| Full run exceeds 30 min | Low | Wasted wall time | Pre-flight latency observation extrapolated; abort if first 3 cells > 60s each |

## 8. Out of scope (explicit)

- Paper Markdown table renderer
- `--seed` CLI override (per-trial varying built into probe_cell)
- Per-cell checkpointing / resume
- Token tracker integration
- Cross-prompt sensitivity (WP-NEW-C)
- Re-running on alternate models if D1 IDs are 404 (separate follow-up commit)
