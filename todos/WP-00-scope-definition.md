# WP-00: Scope Definition (N projects / M models / K domains / Q seeds)

**Owner:** Baran + Ali (joint decision; one decision artifact)
**Depends-on:** []
**Effort:** S
**Status:** TODO
**Addresses instructor feedback:** [Hoca-1] (primary)

## Goal

Pin down all sample-size and scope numbers in a single config file `configs/scope.yaml`, then import them into the paper via `\input{scope-numbers.tex}` so that no numerical scope claim in the manuscript is hand-typed. This single decision unblocks WP-01b (table aggregation), WP-04 (RQ2 model count), WP-06 (RQ4 seed catalogue), and WP-17 (statistical sample size justification). Without WP-00 every other WP either guesses or hardcodes — the worst possible state.

## Acceptance criteria

- [ ] `configs/scope.yaml` exists with keys: `runs_per_config`, `models`, `domains`, `violation_types`, `seeds_per_type_per_domain`, `audit_random_fraction`, `audit_low_confidence_inclusion`, `practitioner_survey_target`.
- [ ] `scripts/render_scope.py` produces `paper/scope-numbers.tex` containing `\newcommand{\Nruns}{5}\newcommand{\Mmodels}{4}` etc.
- [ ] `paper.tex` line 834 (`\placeholder{N}`) replaced with `\Nruns`.
- [ ] All RQ tables (`tab:rq1_pipeline`, `tab:rq2_models`, `tab:rq3_domains`, `tab:rq4_seeded`) use macros from `scope-numbers.tex` for header text and explanatory captions.
- [ ] Hoca sign-off recorded in commit message (e.g., `WP-00: scope locked, Hoca approved 2026-MM-DD`).

## Implementation steps

1. Recommend defaults to Hoca for review:
   - `runs_per_config: 5` (variance reporting min N=3, ideal N=5)
   - `models: ["gemini-2.5-pro", "gpt-5", "claude-sonnet-4-7", "qwen2.5-coder-32b"]` (4 models, one per family + 1 OSS)
   - `domains: ["D1: e-commerce-or-similar", "D2: TBD", "D3: TBD"]` (K=3, justified in §4.2)
   - `violation_types: 6` (V1–V6, fixed)
   - `seeds_per_type_per_domain: 5` (5 × 6 × 3 = 90 total seeds for RQ4)
   - `audit_random_fraction: 0.25` + `audit_low_confidence_inclusion: true`
   - `practitioner_survey_target: 12` (only relevant if WP-09 / RQ5=(D) pursued)
2. Hoca review gate: confirm or push back on N=5 (Hoca may want N=10 for stronger statistical power; if so, ripple impact on compute budget — flag to WP-01c / R5).
3. Lock the YAML, commit, tag.
4. Write `scripts/render_scope.py`: trivial — read YAML, write `\newcommand{\X}{value}` lines.
5. Edit `paper.tex` to `\input{scope-numbers.tex}` after the preamble; replace `\placeholder{N}` and any other scope-dependent macro.
6. Update `00-paper-baseline.md` line 834 placeholder list to "RESOLVED — see scope-numbers.tex".

## Outputs (file paths)

- `configs/scope.yaml`
- `scripts/render_scope.py`
- `LaTeX_DL_468198_240419/scope-numbers.tex` (auto-generated, NOT manually edited)
- `paper.tex` edits at 1 location (line 834) + preamble `\input`
- Commit on `wp-00-scope` branch, merged to main after Hoca approval

## Risks & mitigations

- **Risk:** Hoca pushes N=5 → N=10. **Mitigation:** WP-17 power analysis runs *before* RQ1 to validate. If N=5 insufficient, accept the extra ~10% compute cost (`R5` in `01-risks.md`).
- **Risk:** D2/D3 not yet selected, blocking WP-02. **Mitigation:** WP-00 commits `D2/D3 = "TBD-week-2"`; WP-02 must finalize within 5 days of WP-00 sign-off.
- **Risk:** OSS model choice forces hardware investment. **Mitigation:** Verify GPU availability *before* committing Qwen2.5-Coder-32B; fall back to Llama-3.1-70B via Together AI / Replicate if local hardware insufficient (R2).
