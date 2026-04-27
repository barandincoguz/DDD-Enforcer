# WP-17: Statistical Methodology (Variance, CI, Significance, Effect Size, Power)

**Owner:** Ali (implements analysis scripts) + Baran (writes §"Statistical Analysis Plan")
**Depends-on:** [WP-01b (variance source), WP-03..06 (raw data)]
**Effort:** M
**Status:** TODO
**Addresses instructor feedback:** [Hoca-6] (primary — backbone of methodological credibility)

## Goal

Provide every statistical number a reviewer would ask for: per-cell std/IQR, 95% CIs (bootstrap), per-claim significance tests (Wilcoxon for paired, Friedman+Nemenyi for ranking), effect sizes (Cliff's δ), multiple-comparison correction (Holm), and a power-analysis justification for N=5. Pre-register the analysis plan in §4.7 to defuse "p-hacking" objections. Update §9.3.4 + §9.3.7 prose accordingly.

## Acceptance criteria

- [ ] `analysis/statistical_tests.py` with functions: `bootstrap_ci(values, n=10000, alpha=0.05)`, `wilcoxon_paired(a, b)`, `friedman_nemenyi(matrix)`, `cliffs_delta(a, b)`, `holm_correct(p_values)`.
- [ ] `analysis/power_analysis.py`: pre-registers expected effect size + computes minimum N. If N=5 insufficient at α=0.05, escalate (R5 in `01-risks.md`).
- [ ] `analysis/results.csv`: one row per claim (e.g., "P3 > P2 on V1 Synonym recall"), columns: `claim`, `metric`, `n`, `mean_a`, `mean_b`, `delta`, `ci_lower`, `ci_upper`, `p_value`, `cliffs_delta`, `holm_corrected_p`.
- [ ] §4.7 (or new §) "Statistical Analysis Plan" added: pre-registered tests, multiple-comparison correction approach.
- [ ] §9.3.4 internal validity expanded: variance reporting, CI methodology, significance test framework.
- [ ] §9.3.7 conclusion validity: power-analysis-grounded sample size justification.
- [ ] All RQ tables in §5–§8 have CI annotations or `±std` columns.
- [ ] Inline numerics in §9.1 prose cite p-values where they apply.

## Implementation steps

1. Implement `bootstrap_ci`, `wilcoxon_paired`, `cliffs_delta` (small set of well-tested functions; ≤200 lines total).
2. **Pilot run on D1 with N=5** in week 4: if cell-level std > 0.15 F1, escalate N to 10 *before* expensive runs commit (R5 mitigation).
3. Power analysis: assume effect size d=0.5 (medium) for P3 vs P2 F1; compute minimum N.
4. After WP-03..06 produce raw data, run all tests; populate `analysis/results.csv`.
5. Write §4.7 pre-registered Statistical Analysis Plan (paragraph or two).
6. Write §9.3.4 expansion + §9.3.7 sample-size justification.
7. Update RQ tables (§5–§8) with `±std` or CI columns where space allows.

## Outputs (file paths)

- `analysis/statistical_tests.py`
- `analysis/power_analysis.py`
- `analysis/results.csv`
- `paper.tex` §4.7 (new), §9.3.4 (expanded), §9.3.7 (sample-size justification), §5–§8 table CI columns
- `replication_package/analysis/`

## Risks & mitigations

- **Risk:** N=5 insufficient for significance; headline claims weakened. **Mitigation:** Pilot in week 4 catches this early. If insufficient, bump to N=10 (~1 extra week of compute) OR reframe claims as "differences observed; sample size limits significance" (honest, not weak).
- **Risk:** Multiple comparisons inflate Type I error; Holm correction is too conservative. **Mitigation:** Report both raw and corrected p-values in `results.csv`; let reader judge.
- **Risk:** Bootstrap CI is wide due to N=5; CI overlaps in adjacent claims. **Mitigation:** Report effect sizes alongside CIs; Cliff's δ does not depend on N as strongly.
