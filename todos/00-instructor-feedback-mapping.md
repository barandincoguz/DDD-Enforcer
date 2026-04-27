# Hoca (Prof. Dr. Murat Karakaya) Feedback × Work Package Mapping

**Date:** 2026-04-27
**Source:** Direct verbal notes from advisor on the conference→journal extension.
**Treatment:** Each note is treated as an EMSE reviewer attack vector. **Every note is mapped to ≥1 WP.** Deliverable rule: a paper version cannot be marked "submission-ready" until every Hoca-* tag below is closed.

---

## 1. Hoca's six notes (verbatim)

> 1. N proje / M model sayılar boş
> 2. Abstract sonuç kısmı boş
> 3. Related work tamamla
> 4. Sistem mimarisi şekli / extension workflow'ları net ve detaylı olmalı
> 5. RQ1–RQ5 yapılacak
> 6. Threats to validity içinde run sayısı, varyant, statistical power eksik

---

## 2. Note → WP mapping (matrix)

| Tag | Hoca's note | Severity | Primary WPs | Secondary WPs | Closure criterion |
|-----|-------------|----------|-------------|---------------|-------------------|
| **Hoca-1** | N proje / M model boş | CRITICAL | WP-00 (scope), WP-01b (metric logging), WP-17 (statistical methodology) | WP-02 (corpus), WP-04 (RQ2 model exec) | All paper tables auto-render from `runs/`; `\input{scope-numbers.tex}` injects N,M,K,Q,R; §9.3.4 `\placeholder{N}` resolved to numeric. |
| **Hoca-2** | Abstract sonuç kısmı boş | HIGH | WP-14 (abstract + conclusion polish) | WP-13 (discussion prose), WP-17 (numbers source) | Abstract contains 3+ numeric findings ("Across N=5 runs / M=4 providers / K=3 domains, P3 yields F1=…, Δ=… vs P2 (p<…)"). |
| **Hoca-3** | Related work tamamla | HIGH | WP-10 (bibliography cleanup), Phase 1.2 literature subagent | WP-13 (positioning prose) | All 5 in-text `\placeholder{REFERENCE}` blocks replaced with concrete cited paragraphs; bibliography has ≥8 new peer-reviewed refs (2022–2026); `automatingddd2024` arXiv ID corrected or removed; `li2024llm_bugs`, `nam2024llm_comprehension`, `dddllmindustry2024` resolved to real papers. |
| **Hoca-4** | Mimari şekli + extension workflow | HIGH | WP-11 (figures), WP-16 (extension architecture documentation) | WP-13 (architecture prose) | Figure 1 (architecture) is a vector PDF showing the full SRS→VS Code path; §3.4 grows from ~7 lines to ~1.5 pages with Figure 4 (extension sequence diagram) + 3–5 screenshots. |
| **Hoca-5** | RQ1–RQ5 (currently 4) | DECISION-NEEDED | WP-18 (RQ5 design + execution) | WP-13, WP-14 (results discussion + conclusion bullet) | Paper §4.1 lists RQ5 with motivation; §6.5 (or new §) reports its results; §9.1 has bullet for it; §10 conclusion has a 5th bullet. |
| **Hoca-6** | Threats: run sayısı, varyant, statistical power | CRITICAL | WP-17 (statistical methodology) | WP-13 (threats prose), WP-08 (audit + κ) | §9.3.4 explicitly states N, std/IQR, 95% CI; §4.7 (or new §) presents pre-registered Statistical Analysis Plan; significance tests (Wilcoxon for P3>P2, Friedman+Nemenyi for model rankings) reported with p-values; effect size (Cliff's δ); Cohen's κ from independent dual-author audit. |

---

## 3. WP → Hoca-tag reverse index

For each WP, which Hoca notes does it (partially or fully) address?

| WP | Hoca tags addressed |
|----|---------------------|
| **WP-00** Scope definition | Hoca-1 (primary) |
| **WP-01a** Provider abstraction | Hoca-1 (enables M=4 model run) |
| **WP-01b** Metric logging + run manifest | Hoca-1 (enables N-run averaging tables), Hoca-6 (variance source) |
| **WP-01c** Token tracking + cost telemetry | Hoca-1 (enables RQ2 cost column) |
| **WP-01d** Pipeline implementations (P1/P2/P3) | (enabling) |
| **WP-02** Subject corpus (D1, D2, D3) | Hoca-1 (K=3 fixed) |
| **WP-03** RQ1 experiments | Hoca-1 (fills Table 6) |
| **WP-04** RQ2 experiments | Hoca-1 (fills Table 7) |
| **WP-05** RQ3 experiments | Hoca-1 (fills Table 8) |
| **WP-06** RQ4 experiments | Hoca-1 (fills Table 9) |
| **WP-07** Judge LLM + rubric pipeline | Hoca-6 (Cohen's κ requires Judge → audit chain) |
| **WP-08** Author audit + Cohen's κ | Hoca-6 (κ + dual-author independence) |
| **WP-09** Practitioner survey (optional) | (industry-relevance, not directly Hoca) |
| **WP-10** Bibliography + placeholder cleanup | Hoca-3 (primary) |
| **WP-11** Figures + diagrams | Hoca-4 (primary) |
| **WP-12** Replication package | (EMSE Open Science, not directly Hoca but EMSE-required) |
| **WP-13** Discussion + threats prose | Hoca-1 (analysis prose), Hoca-2 (findings narrative), Hoca-6 (threats prose) |
| **WP-14** Abstract + Conclusion polish | Hoca-2 (primary) |
| **WP-15** Submission package | (gate, not directly Hoca) |
| **WP-16** Extension architecture documentation | Hoca-4 (primary, extension half) |
| **WP-17** Statistical methodology | Hoca-6 (primary) |
| **WP-18** RQ5 design + execution | Hoca-5 (primary) |

**Coverage check:** Every one of Hoca-1, -2, -3, -4, -5, -6 has at least one **primary** owner WP. ✓

---

## 4. Reviewer-attack rehearsal (linking Hoca's notes → reviewer questions)

| If reviewer asks… | Source Hoca tag | We answer with |
|-------------------|------------------|----------------|
| "How many independent runs averaged the headline numbers?" | Hoca-1, Hoca-6 | "N=5 runs per configuration; per-cell std and 95% CI in Table X; raw runs in replication package" — WP-00 + WP-01b + WP-17. |
| "Without an expert annotator, what is your inter-rater reliability?" | Hoca-6 | "Cohen's κ from dual-author independent audit on a 25% random sample of Judge verdicts" — WP-08. |
| "Did you control for prompt variance?" | Hoca-6 | "Prompt is held fixed (released in replication package). Prompt sensitivity flagged in §9.3.4 as known limitation; future-work item." — WP-13. |
| "Is the P3 > P2 difference significant?" | Hoca-6 | "Wilcoxon signed-rank, p < 0.05 after Holm correction; Cliff's δ = …" — WP-17. |
| "Where is the figure for the system architecture?" | Hoca-4 | Figure 1 (vector PDF, full pipeline) — WP-11. |
| "What does the developer actually see in the extension?" | Hoca-4 | §3.4 expanded + Figure 4 sequence diagram + 3–5 screenshots — WP-16. |
| "Why these three domains and not one or ten?" | Hoca-1 | §4.2 with explicit selection rationale per domain (WP-02 deliverable) + acknowledged "probe, not benchmark" (already in §9.3.5). |
| "Why is your `automatingddd2024` arXiv ID 2603.26244 — that's in the future?" | Hoca-3 | Either real ID after literature subagent verification or citation removed; positioning paragraphs adjusted — WP-10. |
| "What's RQ5?" | Hoca-5 | Default proposal: ablation (no AST features) using existing RQ1–4 dataset — WP-18. |
| "Where is your replication package?" | (EMSE-required) | Zenodo DOI + GitHub release tag — WP-12. |
| "What is your sample size for the precision/recall estimates?" | Hoca-6 | §9.3.7 conclusion validity quantifies it; bootstrap CIs reported — WP-17. |

---

## 5. Closure tracking

Each Hoca tag is **closed** when its primary WP(s) are marked DONE in `INDEX.md` AND the corresponding paper edits have landed in `paper.tex` (verified via `\placeholder` grep returning zero hits in the relevant section).

| Hoca tag | Status (2026-04-27) |
|----------|---------------------|
| Hoca-1 | OPEN — depends on WP-00 + WP-01b + WP-17 + experiments |
| Hoca-2 | OPEN — depends on Hoca-1 closure (numbers must exist before abstract can be filled) |
| Hoca-3 | OPEN — Phase 1.2 literature subagent in flight |
| Hoca-4 | OPEN — depends on WP-11 + WP-16 |
| Hoca-5 | OPEN — RQ5 selection awaiting Baran's decision (Phase 1.1.C) |
| Hoca-6 | OPEN — depends on WP-17 + WP-08 |

**Critical observation:** Hoca-2 cannot close before Hoca-1 closes (numbers gate the abstract). This is the longest single dependency chain in the project.

---

## 6. Process for staying in sync with Hoca

- Each WP marks itself "ready for Hoca review" when its primary deliverable is complete.
- Recommend a **review gate** at the end of each phase (Phase 0/1 → Hoca skim, Phase 2 → Hoca approval before code starts, Phase RQ-execution → Hoca check after RQ1+RQ2, Phase pre-submit → 1-week internal review with Hoca + 1 outside reader).
- Hoca's role is **review-only** — he is not assigned as Owner on any WP. (Confirmed in user's prompt.)
- Any deviation from this mapping requires updating *this file* before doing the work.
