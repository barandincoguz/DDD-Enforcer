# WP-18: RQ5 Design + Execution (Hoca-5)

**Owner:** Ali (if RQ5 = (A) Ablation) OR Baran (if RQ5 = (D) Developer Study) — depends on choice
**Depends-on:** [WP-00, WP-01a/b/c/d (for ablation); WP-09 (if developer study)]
**Effort:** M (Ablation) | L (Developer Study) | XL (combined RQ5+RQ6)
**Status:** AWAITING DECISION (Baran chooses 1 of 3 favorites from `01-brainstorming.md` §C)
**Addresses instructor feedback:** [Hoca-5] (primary)

## Goal

Land the 5th research question that Hoca requested. Default proposal: **RQ5 = (A) Ablation: AST features removed** — cheap, dataset-reuse, defends the architectural claim. Strong second: **(D) Developer Study** if industry-relevance evidence is desired. See `01-brainstorming.md` §C for full decision matrix.

## Acceptance criteria (varies by chosen RQ5)

### If RQ5 = (A) Ablation (AST removed)

- [ ] `--no-ast` flag added to `pipelines/p3_multi_agent.py` (and optionally p2): when set, AST features are not injected into the LLM prompt.
- [ ] `runs/rq5_ablation/` produced: WP-04 winning model × {with-AST, without-AST} × 5 runs × D1.
- [ ] `LaTeX_DL_468198_240419/tables/rq5_ablation.tex`: with-AST vs without-AST F1, per violation type.
- [ ] §4.1 RQ5 added: "Does AST grounding contribute beyond what LLM-only achieves?"
- [ ] §6.5 (or new §) "RQ5 Results: AST Ablation" written: per-type degradation, clear narrative on V4 Context Boundary (likely most AST-dependent).
- [ ] §9.1 conclusion bullet (5th) added.

### If RQ5 = (D) Developer Study (merge with WP-09)

- [ ] All WP-09 acceptance criteria.
- [ ] §4.1 RQ5: "Do practitioners find the framework's violations useful?"
- [ ] §"Practitioner Perspective" promoted from appendix to §"RQ5 Results".
- [ ] §9.1 conclusion bullet (5th) added.

### If RQ5+RQ6 (combo)

- [ ] Both above. Effort: +4 weeks. Recommended only if team capacity allows.

## Implementation steps (Ablation default)

1. Open `pipelines/p3_multi_agent.py`; add boolean flag in `Pipeline.run(srs, code, model_config, with_ast=True)`.
2. Branch where AST features are injected into prompt: skip when `with_ast=False`.
3. Smoke test with-AST vs without-AST on 1 file.
4. Run `make rq5-ablation`: winning model × with/without × 5 runs × D1.
5. Build `tables/rq5_ablation.tex` (paired comparison; Wilcoxon test if N=5 enough).
6. Write §4.1 RQ5 + §6.5 results + §9.1 bullet.

## Outputs (file paths, Ablation default)

- Updated `extension/backend/core/pipelines/p3_multi_agent.py` with `with_ast` flag
- `runs/rq5_ablation/`
- `LaTeX_DL_468198_240419/tables/rq5_ablation.tex`
- `paper.tex` §4.1 RQ5 description + §6.5 results + §9.1 bullet (lines 101–107 for RQ list, ~755 for §6.5)
- `docs/RQ5_RUNBOOK.md`

## Risks & mitigations

- **Risk (Ablation):** Without AST, LLM completely fails to detect V4/V5/V6, making the comparison degenerate. **Mitigation:** This *is* the result — AST is load-bearing. Frame in §6.5 as "AST contributes 30–60% of recall on V4–V6; less on V1–V3 which are already syntactic-pattern-driven".
- **Risk (Developer Study):** IRB delay pushes RQ5 past submission deadline. **Mitigation:** Default to (A); fallback to (D) only if IRB cleared by week 4.
- **Risk (Either):** RQ5 distracts from the 4-RQ funnel narrative. **Mitigation:** Keep RQ5 self-contained; do not weave it into RQ1–RQ4 conclusions.
