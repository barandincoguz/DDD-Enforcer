# WP-04: RQ2 Experiments — LLM Provider Comparison

**Owner:** Ali (run, cost monitoring) + Baran (analysis)
**Depends-on:** [WP-01a, WP-01b, WP-01c, WP-01d, WP-02, WP-03, WP-07]
**Effort:** L (4 model providers × cost monitoring × per-violation-type breakdown)
**Status:** BLOCKED on WP-03 (winner pipeline must be known)
**Addresses instructor feedback:** [Hoca-1] (fills Table 7 and Pareto figure)

## Goal

Using the WP-03 winning pipeline (likely P3), swap in 4 LLM providers (Gemini 2.5 Pro, GPT-5, Claude Sonnet 4.7, Qwen2.5-Coder-32B or fallback OSS) and run N=5 each on D1. Produce Table 7 cells + Figure 2 (Pareto frontier). Determine **winner model** for RQ3.

## Acceptance criteria

- [ ] `runs/rq2/<model>_<run>/manifest.json` exists for all 20 runs (4 models × 5 runs).
- [ ] Each manifest has populated `cost_usd` field (depends on WP-01c).
- [ ] Judge verdicts produced for all 20 runs (WP-07 cross-family logic kicks in: Judge belongs to a different family than the tested model where possible).
- [ ] `LaTeX_DL_468198_240419/tables/rq2.tex` auto-rendered with: P, R, F1, Avg latency, Cost-per-validation. All 19 cells of Table 7 non-empty (OSS row's cost says "compute only").
- [ ] `LaTeX_DL_468198_240419/figures/rq2_pareto.pdf` rendered by `scripts/figures/pareto.py` (matplotlib scatter, log-x cost, y F1, Pareto-frontier annotated).
- [ ] §6 analysis paragraph (line 677 AUTHOR-TODO) drafted: top-F1 dominator analysis, OSS practicality verdict, Pareto-frontier shape commentary.
- [ ] §6 summary box (line 681) names winner model.
- [ ] Per-violation-type breakdown in `replication_package/rq2_per_type.csv`.

## Implementation steps

1. Run `python scripts/cost_estimate.py --pipeline P3 --model claude-sonnet-4-7 --srs D1 --runs 5` for **each model**. If total > $300, escalate (R9 in `01-risks.md`).
2. Smoke-test all 4 providers via WP-01a smoke test on a 1-violation example.
3. Verify OSS model (Qwen2.5-Coder-32B or fallback Llama-3.1-70B / DeepSeek-Coder-V2 via Together AI) returns parseable structured output. Time-box to 3 days; if failing, switch to cloud-hosted OSS (R2 mitigation).
4. Run `make rq2` end-to-end: 20 runs × ~30 files = 600 file-validations.
5. Judge verdicts (WP-07) for all 600 validations.
6. Author audit (WP-08) — 25% sample of each model's TPs/FPs/FNs.
7. Build Table 7 + Figure 2 (Pareto): `make rq2-tables && make rq2-figures`.
8. Write §6 analysis prose + summary box.
9. Commit "WP-04 RQ2 done" → unblocks WP-05.

## Outputs (file paths)

- `runs/rq2/` (20 manifest dirs)
- `judge_verdicts/rq2/`
- `audit/rq2/`
- `LaTeX_DL_468198_240419/tables/rq2.tex`
- `LaTeX_DL_468198_240419/figures/rq2_pareto.pdf`
- `scripts/figures/pareto.py`
- `replication_package/rq2_per_type.csv`
- `paper.tex` §6 analysis + summary
- `docs/RQ2_RUNBOOK.md`

## Risks & mitigations

- **Risk:** OSS model fails to produce structured output (R2). **Mitigation:** Already mitigated in WP-01a — fall back to cloud-hosted OSS. The "open source" framing is preserved as long as model weights are public.
- **Risk:** RQ2 cloud bill exceeds $500 (R9). **Mitigation:** Pre-flight `cost_estimate.py` ; if over budget, swap Claude Opus → Sonnet, GPT-5 → mini, etc. **Always do this check before launching.**
- **Risk:** Top-F1 model also wins on cost — boring result. **Mitigation:** This is a real possibility (and a publishable finding!) — frame as "the cheap-but-strong model dominates the frontier; OSS alternatives are X% behind, suggesting on-prem is still sub-Pareto today".
- **Risk:** Cross-family Judge selection fails (Judge family = tested family for some pair). **Mitigation:** WP-07 logic flags these as sensitivity checks; reported in §9.3.2.
