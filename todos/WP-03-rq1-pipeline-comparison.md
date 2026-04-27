# WP-03: RQ1 Experiments — Pipeline Comparison (P1 vs P2 vs P3)

**Owner:** Ali (run) + Baran (analysis)
**Depends-on:** [WP-01a, WP-01b, WP-01c, WP-01d, WP-02, WP-07]
**Effort:** M
**Status:** TODO
**Addresses instructor feedback:** [Hoca-1] (fills Table 6)

## Goal

Run the canonical RQ1 grid: D1 × 1 fixed LLM × 3 pipelines × N=5 runs × all source files in D1's codebase. Produce all numeric cells in Table 6 (`tab:rq1_pipeline`, lines 626–639) plus per-violation-type breakdowns (in replication package only). Determine the **winner pipeline** that flows into RQ2 (WP-04).

## Acceptance criteria

- [ ] `runs/rq1/<config>_<run>/manifest.json` exists for all 15 runs (3 pipelines × 5 runs).
- [ ] All 15 runs have associated Judge verdicts in `judge_verdicts/` (depends on WP-07).
- [ ] `LaTeX_DL_468198_240419/tables/rq1.tex` auto-rendered with: P, R, F1, Parseable %, Rulebook quality (P3 only), Wall-clock seconds. All 17 cells of Table 6 non-empty.
- [ ] §5 analysis paragraph (line 641 AUTHOR-TODO) drafted with: V1/V4 hypothesis test, P2 fallback narrative, P1 unparseable rate, qualitative example contrasting P1 hallucination vs. P3 grounded violation.
- [ ] §5 summary box (line 645) names the winner pipeline.
- [ ] WP-13 receives "RQ1 done" signal so Discussion §9.1 prose can fill.

## Implementation steps

1. Verify WP-01d's `make rq1` command runs end-to-end on a smoke configuration (1 run, 1 file).
2. Pick the "fixed LLM" for RQ1. Recommendation: **Gemini 2.5 Flash** (already validated in legacy runs; cheapest of the 4 models; controls confounding while pipelines are compared).
3. Run `make rq1` end-to-end: 15 runs × ~30 files = 450 file-validations. Parallelize across 3 worker processes if rate-limit allows.
4. Run WP-07 Judge on all 450 validations (Judge LLM is GPT-5 or Claude Opus 4.7 — see WP-07).
5. Run WP-08 audit (25% random + low-confidence) for the per-pipeline TPs/FPs/FNs.
6. Build tables: `python scripts/build_tables.py runs/rq1/ --rq 1 --output paper/tables/rq1.tex`.
7. Inspect `tables/rq1.tex` and write the §5 analysis prose.
8. Commit "WP-03 RQ1 done" — WP-04 unblocked.

## Outputs (file paths)

- `runs/rq1/` (15 manifest dirs)
- `judge_verdicts/rq1/` (15 verdict files)
- `audit/rq1/` (sampled override CSVs from WP-08)
- `LaTeX_DL_468198_240419/tables/rq1.tex` (auto-generated)
- `paper.tex` §5 analysis paragraph + summary box (lines 641, 645)
- `docs/RQ1_RUNBOOK.md` documenting the exact configuration used

## Risks & mitigations

- **Risk:** P1 unparseable rate is so high that the comparison is unfair. **Mitigation:** Document the parseable% honestly. If P1 consistently produces unparseable output, that *is* the result — no gymnastics required. (Hoca + reviewer will appreciate honest reporting.)
- **Risk:** N=5 variance is so large that no winner is statistically distinguishable from another. **Mitigation:** Triggers WP-17 to either bump N=10 or report effect-size ranges; prose adjusts framing from "P3 wins" to "P3 is in the top tier".
- **Risk:** Rate-limit on Gemini blocks parallel runs. **Mitigation:** Worker pool with backoff is already in WP-01a base class; if still slow, run sequentially overnight.
