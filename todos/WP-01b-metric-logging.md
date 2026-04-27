# WP-01b: Metric Logging + Run Manifest Module

**Owner:** Ali
**Depends-on:** [WP-00]
**Effort:** M (existing `validation_metrics.py` covers ~50%; gap is multi-run aggregation + LaTeX table emission)
**Status:** TODO
**Addresses instructor feedback:** [Hoca-1] (enables N-run averaging tables), [Hoca-6] (variance source)

## Goal

Enforce the rule "**no paper table is hand-typed**". Every cell in tables 4–9 of `paper.tex` is rendered from a `runs/` directory by `scripts/build_tables.py`. This is the operational backbone of Hoca-1 closure: scope numbers in `configs/scope.yaml` + run manifests on disk + auto-rendered LaTeX = zero placeholder cells once experiments complete.

**Existing infrastructure to leverage:** `extension/backend/core/validation_metrics.py` (`ValidationMetricsTracker`, `ValidationStats`) and `validation_metrics_report.json` already exist. This WP **extends** them; it does not replace.

## Acceptance criteria

- [ ] `RunManifest` Pydantic schema with fields: `run_id`, `timestamp_utc`, `pipeline` (P1/P2/P3), `model_id`, `provider`, `srs_path`, `srs_sha256`, `code_root`, `code_sha256`, `violations: List[Violation]`, `latency_seconds`, `prompt_tokens`, `completion_tokens`, `cost_usd`, `judge_verdict_path: Optional[str]`, `audit_overrides_path: Optional[str]`, `seed_manifest_path: Optional[str]` (RQ4 only).
- [ ] Each pipeline run writes exactly one `runs/<run_id>/manifest.json` (run_id = `{pipeline}_{model}_{srs}_{timestamp}_{seed}`).
- [ ] Per-run `metrics.py` computes precision / recall / F1 per-violation-type and aggregate (input: `RunManifest` + Judge verdict file). 
- [ ] `scripts/aggregate.py` reads N=5 runs for a (pipeline, model, SRS) configuration and emits `mean ± std`, IQR, and bootstrap 95% CI.
- [ ] `scripts/build_tables.py runs/ --rq <1|2|3|4> --output paper/tables/rq<N>.tex` renders the LaTeX `tabular` body for each RQ.
- [ ] `paper.tex` has `\input{tables/rq1.tex}` etc. inside each `\begin{table}` block.
- [ ] Smoke test: 3 pipelines × 1 model × 1 domain × 2 runs = 6 manifests → table builder produces a syntactically-valid LaTeX file that compiles.

## Implementation steps

1. Define `RunManifest` Pydantic schema in `extension/backend/core/run_manifest.py`.
2. Write a thin "save manifest" hook in the pipeline orchestrator that produces `runs/<run_id>/manifest.json` after each run.
3. Extend `validation_metrics.py` to accept a `RunManifest` and a Judge verdict file, returning `PrecisionRecallF1` per violation type and aggregate.
4. Write `scripts/aggregate.py` (input: glob over `runs/`; output: per-config aggregates).
5. Write `scripts/build_tables.py` per-RQ renderers (4 separate template strings, one per table).
6. Edit `paper.tex` to `\input{tables/rq1.tex}` etc.
7. Add `make tables` Makefile target (also creates the WP-01b sub-deliverable for "make orchestration").
8. Mark `extension/backend/core/intermediate/*.json` (154 legacy files) as `legacy_pre_emse/` — they predate this manifest format and will not be reused.

## Outputs (file paths)

- `extension/backend/core/run_manifest.py`
- `extension/backend/core/metrics.py` (refactored from `validation_metrics.py`)
- `scripts/aggregate.py`
- `scripts/build_tables.py`
- `LaTeX_DL_468198_240419/tables/rq{1,2,3,4}.tex` (auto-generated)
- `Makefile` target `tables`
- `tests/test_table_builder.py` (smoke test)
- Updated `paper.tex` with `\input{tables/...}`

## Risks & mitigations

- **Risk:** Existing `validation_metrics_report.json` schema diverges from new `RunManifest`. **Mitigation:** New schema is a *superset* of the old one; legacy reports remain readable with the old loader, but are not consumed by `build_tables.py`.
- **Risk:** Bootstrap CI computation is slow for N=5 runs × 6 violation types × 4 models × 3 domains = 360 cells. **Mitigation:** Vectorize with numpy; cache aggregate results in `runs/_aggregated/<config>.json`.
- **Risk:** LaTeX template injection escapes user-provided strings (model names, domain names). **Mitigation:** Use `\detokenize{...}` for any string content; whitelist alphanumerics + `-` + `_` in `model_id` and `srs_label`.
