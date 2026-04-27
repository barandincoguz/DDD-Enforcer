# WP-13: Discussion + Threats Prose

**Owner:** Baran (writing) + Ali (statistical numbers from WP-17)
**Depends-on:** [WP-03, WP-04, WP-05, WP-06, WP-08, WP-17, WP-18 if RQ5 chosen]
**Effort:** M
**Status:** TODO
**Addresses instructor feedback:** [Hoca-1] (RQ analysis prose), [Hoca-2] (findings narrative for abstract source), [Hoca-6] (threats prose with stats)

## Goal

Fill all ANALYSIS-PROSE placeholders in §5–§9 of `paper.tex` (12 placeholder locations identified in `00-paper-baseline.md`). Tighten threats-to-validity (§9.3) by integrating WP-17's statistical numbers, dual-author κ from WP-08, and (where applicable) practitioner data from WP-09/RQ5. The output of this WP is the prose foundation for WP-14 (abstract polish).

## Acceptance criteria

- [ ] §5 (RQ1) lines 641, 645 filled: where P3 outperforms P2 (which violation types), where P2 is "good enough", P1 unparseable rate, qualitative example contrasting hallucinated vs. grounded violation, summary-box names winner.
- [ ] §6 (RQ2) lines 677, 681 filled: top-F1 dominator analysis, OSS practical viability for data-egress-constrained shops, Pareto-frontier shape commentary, summary-box names winner with "today's-capability" framing.
- [ ] §7 (RQ3) lines 705, 709 filled: hardest-domain identified + reason, common per-domain failure modes, framework-level weakness if any, summary-box reiterates "probe not benchmark".
- [ ] §8 (RQ4) lines 738, 742 filled: most/least reliable types, non-seeded confirmed bonus violations, type-confusion analysis, summary-box reports recall floor + weakest type.
- [ ] §9.1 lines 763, 766, 774 filled: multi-agent decomposition payoff narrative, provider spread + OSS narrative, author-Judge agreement rate value + characterization (κ from WP-08).
- [ ] §9.3.4 (line 834) `\placeholder{N}` resolved + variance + 95% CI + significance test results integrated (from WP-17).
- [ ] §9.3.7 (line 871) Conclusion-validity sample-size justification written (from WP-17 power analysis).
- [ ] If RQ5 = Ablation pursued (WP-18 = (A)), add §6.5 or §"RQ5 Ablation Results" subsection + bullet in §9.1.
- [ ] If practitioner appendix (WP-09 / RQ5=(D)) pursued, add §"Practitioner Perspective" subsection.
- [ ] Tone audit: "we provide", "we suggest", "we report", "today's capability level" — no "we prove", no "novel".

## Implementation steps

1. After WP-03..06 + WP-08 + WP-17 land, gather all numeric outputs into one consolidated `analysis/numbers_for_prose.md`.
2. For each placeholder location (12 total), write 1–3 sentences citing specific numbers.
3. Run a "claim modesty" pass: search for "prove", "novel", "first", "best" — replace with hedged language unless explicitly grounded.
4. Cross-check: every claim in §9.1 has a corresponding number in §5–§8 tables.
5. Cross-check: every threat in §9.3 has a mitigation + residual-risk paragraph.
6. Hoca review gate: share §5–§9 PDF + 1-bullet diff vs. previous version.

## Outputs (file paths)

- Updated `paper.tex` at all listed line ranges
- `analysis/numbers_for_prose.md` (working notes; not in submission)

## Risks & mitigations

- **Risk:** Numbers shift slightly during final WP-17 stats run; prose locked too early. **Mitigation:** Prose written after `analysis/numbers_for_prose.md` is "frozen" by Ali (commit-tagged).
- **Risk:** Excessive hedging makes paper read as wishy-washy. **Mitigation:** Hedge claims, not findings. "We find F1 = 0.78 ± 0.04 (p < 0.05)" is direct; "this suggests P3 may be preferable in domains where global SRS knowledge matters" is appropriately hedged.
