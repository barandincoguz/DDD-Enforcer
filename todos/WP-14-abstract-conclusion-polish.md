# WP-14: Abstract + Conclusion Polish

**Owner:** Baran
**Depends-on:** [WP-13]
**Effort:** S
**Status:** TODO
**Addresses instructor feedback:** [Hoca-2] (primary)

## Goal

Replace the current results-free abstract (lines 58–70) with a numerically grounded one. Fill the 4 conclusion bullets (lines 886–889) with real findings. Author Contributions block (line 918). Final tone audit. This is the closing paper-polish WP before submission.

## Acceptance criteria

- [ ] Abstract has at least 3 quantitative findings: pipeline winner with F1 vs runner-up Δ, model winner with cost-per-validation, one cross-domain or RQ4 number.
- [ ] Abstract template instantiated: "Across [N=5] runs over [M=4] LLM providers and [K=3] SRS domains, we find that the multi-agent pipeline (P3) yields F1=[fill] versus [fill] for retrieval-augmented baseline (P2) (Wilcoxon p<[fill])…"
- [ ] All 4 (or 5 if RQ5) conclusion bullets in §10 (lines 886–889) filled with real numbers + cautious framing.
- [ ] §10 Author Contributions block (line 918) written: who did what (Baran: paper writing + literature + corpus design + survey if pursued; Ali: infrastructure + experiment runs + statistical analysis; Hoca: supervision + review).
- [ ] Tone audit: no "novel", no "first", no "prove"; consistent "framework", "proof-of-concept", "today's capability level", "swappable".
- [ ] Word count: abstract ≤ 250 words (Springer EMSE limit varies; verify on EMSE author guidelines).

## Implementation steps

1. After WP-13 closes, read the latest §5–§9 prose; extract 3 strongest numbers.
2. Draft abstract using the template; iterate until ≤ 250 words and tone-consistent.
3. Fill conclusion bullets verbatim from §9.1 prose summaries; trim to 1 sentence each.
4. Write Author Contributions (1 paragraph).
5. Run tone-audit grep on full paper: `grep -niE 'novel|prove|first|best' paper.tex`. For each hit, justify or rewrite.
6. Run claim-modesty review one last time.

## Outputs (file paths)

- Updated `paper.tex` lines 58–70 (abstract), 886–889 (conclusion bullets), 918 (author contributions)
- `docs/glossary.md` (consistency reference for "framework", "proof-of-concept", etc.)

## Risks & mitigations

- **Risk:** Abstract feels under-claiming and editor desk-rejects on weak novelty. **Mitigation:** §3.5's 8-row delta table is the novelty story; reference it in abstract by saying "extends the conference version with a multi-agent pipeline (vs. single prompt), structured rulebook (vs. flat term list), 6-type taxonomy (vs. 3), and an LLM-Assisted Human Evaluation protocol".
- **Risk:** Conclusion bullets repeat findings from §9.1 verbatim — boring. **Mitigation:** §10 conclusion is *deliberately* a recap; that's its purpose. Just keep each bullet under 30 words.
