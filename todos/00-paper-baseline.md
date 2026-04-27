# Paper Baseline Analysis — `LaTeX_DL_468198_240419/paper.tex`

**Date:** 2026-04-27
**File:** 1019 lines, single-file `svjour3` Springer EMSE template.
**Conference baseline cited as:** `dincoguz2025ddd_conference`.
**Currently 4 RQs** (RQ1 Pipeline, RQ2 Model, RQ3 Cross-domain, RQ4 Synthetic-violation). Hoca wants **5 RQs** → at least 1 must be added (see WP-18).

---

## 1. Placeholder Inventory (by line, by category)

Placeholders are introduced via `\newcommand{\placeholder}[1]{\textcolor{red}{\textbf{[PLACEHOLDER: #1]}}}` (defined paper.tex:26). Every occurrence below is a deliverable.

### REFERENCE — bibliography or in-text citations missing

| Line | Section | What's missing |
|------|---------|----------------|
| 146 | §2.1 DDD principles | 2–3 DDD-related refs (Mazlami 2017 microservice extraction, Kapferer & Zimmermann 2020, DDD anti-patterns) |
| 160 | §2.2 Architecture enforcement | Knodel & Popescu 2007, Passos et al. 2010, architectural erosion |
| 173 | §2.3 Microservice quality | Microservice anti-pattern detection, decomposition quality metrics, API analysis |
| 182 | §2.4 LLMs for SE | 3–4 LLM-based architecture analysis refs |
| 215 | §2.6 Multi-agent LLM | 2–3 multi-agent LLM-in-SE refs |
| 965 | bib `li2024llm_bugs` | LLM-based bug detection paper — fully placeholder |
| 971 | bib `nam2024llm_comprehension` | LLM for program comprehension — fully placeholder |
| 982 | bib `automatingddd2024` | **arXiv:2603.26244 is IMPOSSIBLE** (YYMM format → March 2026, not 2024). Either find real paper or delete |
| 988 | bib `dddllmindustry2024` | Industry writeup (Mülder 2024 / Sebastian 2024) — choose concrete one |
| 1002 | bib `llmcodequality2026` | EMSE 2026 paper — author list missing; verify DOI `10.1007/s10664-026-10858-8` |
| 1006 | bib `dincoguz2025ddd_conference` | Conference name and proceedings details — UBMK |
| 1009–1015 | bibliography TODO comments | Mazlami, Kapferer, Knodel/Popescu, Passos + more LLM-for-arch + microservice anti-pattern + DDD adoption |

→ **Routes to:** Phase 1.2 literature subagent (`02-literature.md`) + WP-10 (bibliography cleanup) + arXiv ID forensics for `automatingddd2024`.

### FIGURE — vector PDF diagrams to produce

| Line | Section | What's missing |
|------|---------|----------------|
| 258 | §3 Figure 1 | System architecture diagram: SRS → Multi-Agent (Scout→Architect→Specialist→Synthesizer) → Domain Model JSON → Code Analysis (AST+LLM) → Violations → Traceability → VS Code |
| 386 | §3.4 IDE Integration | Extension screenshot: in-editor diagnostic for Synonym Violation + "View Source" side panel |
| 672 | §6 Figure 2 | RQ2 cost–accuracy Pareto frontier (matplotlib scatter) |

→ **Routes to:** WP-11 (figures) + WP-16 (extension architecture documentation, adds Figure 4 sequence diagram).

### TABLE-DATA — empty numeric cells

| Line range | Table | RQ | # Empty cells |
|------------|-------|----|---|
| 464–466 | Table 4 (subjects D1/D2/D3) | exp design | 12 (services, LOC, codebase origin × 3 domains, plus 3 domain-name placeholders) |
| 487–490 | Table 5 (LLM providers under study) | exp design | 8 (model name + cost × 4 models, plus 1 free) |
| 634–636 | Table 6 (RQ1 pipeline comparison) | RQ1 | 17 (P, R, F1, Parseable, Rulebook quality, Wall-clock × 3 pipelines) |
| 662–665 | Table 7 (RQ2 LLM provider) | RQ2 | 19 (P, R, F1, Latency, Cost × 4 models, minus 1 "compute only") |
| 698–700 | Table 8 (RQ3 cross-domain) | RQ3 | 15 (Violations, TP, P, R/F1, Domain-name × 3 domains) |
| 726–733 | Table 9 (RQ4 seeded recall) | RQ4 | 21 (Seeded, Detected, Recall × 6 violation types + Overall row) |

**Total empty numeric cells: ~92** (excluding figures). All flow from `runs/` directory once experiments are scripted (WP-01b/d → WP-03/04/05/06).

→ **Routes to:** WP-01b (metric logging + table builder script) + WP-02 (subject corpus) + WP-03/04/05/06 (RQ experiments).

### ANALYSIS-PROSE — RQ analysis paragraphs to write after data

| Line | Section | What's missing |
|------|---------|----------------|
| 641 | §5 RQ1 | Discussion: where P3 outperforms P2, V1/V4 hypothesis, P1 unparseable rate, qualitative example |
| 645 | §5 RQ1 summary | "Which pipeline best balances detection quality and cost" |
| 677 | §6 RQ2 | Whether top-F1 model dominates across types, OSS practical viability, Pareto frontier shape |
| 681 | §6 RQ2 summary | Best model + framing as "today's capability level" |
| 705 | §7 RQ3 | Hardest domain + why, common failure modes per domain, framework-level weaknesses |
| 709 | §7 RQ3 summary | Generalization commentary, explicit "3 domains is a probe not a benchmark" |
| 738 | §8 RQ4 | Most/least reliable types, non-seeded confirmed bonuses, type-confusion analysis |
| 742 | §8 RQ4 summary | Recall floor framing, weakest type, target for improvement |
| 763 | §9.1 Findings (RQ1) | Multi-agent decomposition payoff narrative — fills once RQ1 numbers in |
| 766 | §9.1 Findings (RQ2) | Provider spread + OSS practicality narrative |
| 774 | §9.1 Caveats | Author–Judge agreement rate value + reliability characterization |
| 886 | §10 Conclusion bullet | RQ1 narrative (P3 vs P2 framing) |
| 887 | §10 Conclusion bullet | RQ2 narrative (winner + OSS viability) |
| 888 | §10 Conclusion bullet | RQ3 narrative (range + hardest domain + reason) |
| 889 | §10 Conclusion bullet | RQ4 narrative (recall fraction + weakest type) |

→ **Routes to:** WP-13 (Discussion + threats prose) + WP-14 (Abstract + Conclusion polish).

### AUTHOR-TODO — explicit author actions

| Line | Section | What's missing |
|------|---------|----------------|
| 386 | §3.4 | IDE screenshot figure (also FIGURE) |
| 471 | §4.2 | D1/D2/D3 origin, selection rationale, DDD-relevant characteristics, codebase origin (third-party vs generated), licensing |
| 871 | §9.3.6 Conclusion validity | Sample-size justification once Judge label counts known (statistical power) |
| 908 | §10 Data availability | Replication package URL, configs, rubric prompts, Judge verdicts, audit overrides, seeded manifests, raw run dirs |
| 918 | Declarations | Author contributions block |

→ **Routes to:** WP-02 (corpus), WP-12 (replication package), WP-14 (author contributions), WP-17 (statistical analysis), WP-16 (screenshot via WP-11).

### SCOPE-NUMBER — sample/run sizes to commit

| Line | Section | What's missing |
|------|---------|----------------|
| 834 | §9.3.4 Internal validity | "averaged over `\placeholder{N}` independent runs" — N must be defined and reported |

Plus implicit scope numbers in paper text:
- L66 abstract: "three pipeline variants" (fixed at 3 ✓), "four LLM providers" (fixed at 4 ✓), "three SRS domains" (fixed at 3 ✓ but not declared as variable)
- L102 RQ1: 3 pipelines (P1/P2/P3) — fixed
- L103 RQ2: 4 models — fixed
- L104 RQ3: 3 domains — fixed
- L105 RQ4: seeded violation count not specified

→ **Routes to:** WP-00 (scope definition: N=5 runs, M=4 models, K=3 domains, Q=5 seeds × 6 types × 3 domains = 90, R=25% audit, S practitioner survey size).

---

## 2. Suspicious / Broken References

### `automatingddd2024` — IMPOSSIBLE arXiv ID

```
\bibitem{automatingddd2024}
\placeholder{AUTHOR TODO: confirm author list and venue.} (2024) Automating Domain-Driven Design: Experience with a Prompting Framework. arXiv:2603.26244.
\url{https://arxiv.org/abs/2603.26244}
```

**Diagnosis.** arXiv IDs follow `YYMM.NNNNN` format. `26.03` = March 2026. This is in the future relative to the paper's claimed 2024 date. The URL **does not resolve**.

**Action.** Three possibilities:
1. The author confused `2403.26244` (March 2024) with `2603.26244`. Check arXiv:2403.26244 — but `26244` exceeds the typical numbering range; likely doesn't exist either.
2. The author meant a different paper entirely. Literature subagent will hunt.
3. Delete the citation if no real paper backs it. Replace with `evans2024infoq` plus an industry writeup (`dddllmindustry2024` once filled).

**Cited at:** lines 91, 198, 224, 789. Removal cascades into 3 in-text positioning paragraphs that need to compensate.

→ **Routes to:** Phase 1.2 literature subagent + WP-10.

### `llmcodequality2026` — DOI verification needed

```
\bibitem{llmcodequality2026}
\placeholder{AUTHOR TODO: confirm author list.} (2026) An evaluation study of large language models for addressing code quality issues. Empirical Software Engineering.
\url{https://doi.org/10.1007/s10664-026-10858-8}
```

**Diagnosis.** DOI prefix `10.1007/s10664-026-10858-8` looks like a Springer EMSE forthcoming-issue DOI. Author list and metadata must be confirmed. The 2026 year is plausible if paper was accepted late 2025 / early 2026.

**Action.** Literature subagent confirms or replaces. If the DOI resolves, fill author list. Currently cited only as a metric precedent — could also be replaced with a more solid existing reference.

### Fully placeholder bibliography entries

- `li2024llm_bugs` (line 965, used at line 178)
- `nam2024llm_comprehension` (line 971, used at line 178)
- `dddllmindustry2024` (line 988, used at line 200)

→ Each requires 1 real reference from literature subagent.

### Conference paper missing details

```
\bibitem{dincoguz2025ddd_conference}
Dincoguz AB, Kendir A, Karakaya M (2025) DDD-Enforcer: An AI-Powered Multi-Agent System for Real-Time Domain-Driven Design Enforcement. \placeholder{Conference name and proceedings details}
```

**Cited at:** lines 99, 391, 401. Author should know venue (UBMK 2025?).

→ **Routes to:** WP-10 (Baran fills in directly from his own records).

---

## 3. Section-by-Section Health Check

| Section | Lines | Health | Notes |
|---------|-------|--------|-------|
| Abstract (§) | 58–70 | **Skeleton OK, sayısal sonuç eksik** | "Across [N] runs, [M] models, [K] domains, P3 yields F1=…" şablonu hazır, deneyler bittikten sonra dolar. |
| §1 Introduction | 75–123 | **Solid** | RQs net, framing ("proof-of-concept", "framework", "today's capability level") tutarlı. **4 RQ var, 5 olacak.** |
| §2.1 DDD principles | 132–146 | Yeterli + 1 placeholder | Ozkan ref güzel; Mazlami/Kapferer eklenmeli |
| §2.2 Architecture enforcement | 148–160 | Yeterli + 1 placeholder | Knodel/Popescu/Passos ekle |
| §2.3 Microservice quality | 162–173 | Yeterli + 1 placeholder | Anti-pattern + decomposition refs ekle |
| §2.4 LLMs for SE | 175–192 | Orta + 1 placeholder | LLM-for-arch refs gerekiyor; bug + comprehension refs yer tutucu |
| §2.5 LLMs for DDD | 194–204 | **automatingddd2024 patladı** | arXiv ID düzelt veya kaldır |
| §2.6 Multi-agent | 206–215 | Orta + 1 placeholder | CodePlan/MetaGPT/ChatDev SE evals ekle |
| §2.7 Positioning | 217–244 | **Mükemmel** | Table 1 hazır, 6 boyutta net karşılaştırma |
| §3 Architecture | 249–413 | **Figür eksik** | Figure 1 placeholder; §3.4 IDE çok kısa (Hoca-4) |
| §3 V1–V6 definitions | 333–367 | **Mükemmel** | Her violation için negative/positive/detection signal var. Reviewer-friendly. |
| §3.5 Conf vs Journal | 388–413 | **Mükemmel Table 2** | 8 satırlık delta tablosu çok güçlü, EMSE editörünü ikna eder |
| §4 Experimental Design | 418–615 | Skeleton iyi + scope numbers boş | Table 3 (subjects), Table 4 (models) cell'leri boş |
| §4.5 Eval Protocol (Judge) | 523–550 | **Methodologically güçlü** | Cross-family Judge, audit, agreement rate iyi düşünülmüş |
| §4.7 Reproducibility | 609–615 | İyi planlama, **infra eksik** | Run dir naming, manifest schema yazılı; uygulamada multi-run orchestrator yok |
| §5 RQ1 | 620–646 | Skeleton + 17 boş cell + 2 prose placeholder | |
| §6 RQ2 | 648–682 | Skeleton + 19 boş cell + Pareto figure + 2 prose placeholder | |
| §7 RQ3 | 684–710 | Skeleton + 15 boş cell + 2 prose placeholder | |
| §8 RQ4 | 712–743 | Skeleton + 21 boş cell + 2 prose placeholder | |
| §9.1 Key Findings | 752–774 | Skeleton iyi + 3 prose placeholder | "framework not point solution" mesajı net |
| §9.2 Comparison | 776–790 | Yeterli | Önceki çalışmalarla karşılaştırma sağlam |
| §9.3.1 Context-window | 798–807 | **İyi** | Mitigation + residual risk net |
| §9.3.2 Hallucinations | 809–818 | **İyi** | Cross-family Judge mantığı iyi |
| §9.3.3 Eval subjectivity | 820–828 | **İyi** | "Cohen's κ uygulanamaz" gerekçesi açık |
| §9.3.4 Internal validity | 830–839 | **N PLACEHOLDER (Hoca-6)** | Variance + CI + significance test eksik |
| §9.3.5 External validity | 841–855 | İyi | Language + domain count + SRS quality threats listelenmiş |
| §9.3.6 Construct validity | 857–867 | İyi | Taxonomy + pipeline design choices threats |
| §9.3.7 Conclusion validity | 869–871 | **AUTHOR-TODO** | Statistical power discussion gerekli (Hoca-6) |
| §10 Conclusion | 877–908 | Skeleton + 4 RQ bullet placeholder + Data Availability TODO | |
| Bibliography | 924–1017 | **5 placeholder + 1 imkansız arXiv ID + 7 yorum-todo** | |

---

## 4. Hoca's Notes → Paper Section Mapping

| Hoca's note | Paper location | Severity | Reroutes |
|-------------|----------------|----------|----------|
| 1. N proje / M model sayılar boş | §4.1 RQs (lines 422–447); Table 4 lines 464–466 (D1–D3); Table 5 lines 487–490 (4 LLMs); §4.7 reproducibility line 612; §9.3.4 line 834 (\placeholder{N}); abstract line 66 | **CRITICAL** | WP-00 + WP-01b + WP-17 |
| 2. Abstract sonuç kısmı boş | Lines 58–70 (no F1 / Pareto / generalization numbers) | HIGH | WP-14 |
| 3. Related work tamamla | Lines 146, 160, 173, 182, 215 (5 in-text); 965, 971, 982, 988, 1002, 1006 (6 bib); 1009–1015 (7 todo comments) | HIGH | WP-10 + Phase 1.2 literature |
| 4. Mimari şekli + extension workflow | Line 258 (Figure 1); §3.4 IDE Integration (lines 380–386, only 7 lines!) | HIGH | WP-11 + WP-16 |
| 5. RQ1–RQ5 (paper'da 4 var) | Lines 101–106 (RQ1–RQ4 listed); §5–§8 result sections | DECISION-NEEDED | WP-18 + Phase 1.1.C |
| 6. Threats: run sayısı, varyant, statistical power | §9.3.4 line 834 (\placeholder{N}); §9.3.7 line 871 (sample size justification); no significance test wording anywhere | **CRITICAL** | WP-17 + WP-13 |

---

## 5. Quick Wins vs. Heavy Lifts

**Quick wins (≤ 2 days each, no new experiments):**
- Fill `dincoguz2025ddd_conference` venue + proceedings (Baran knows this)
- Fix `automatingddd2024` (delete + adjust §2.5 + §9.2)
- Add 2–3 confirmed Mazlami/Kapferer/Knodel/Passos refs (literature subagent)
- Migrate `\begin{thebibliography}` → `refs.bib` for Springer style (BibTeX migration)
- Write Author Contributions block (line 918)
- Update `\placeholder{N}` → `5` once WP-00 confirms

**Heavy lifts (≥ 1 week each):**
- All 4 RQ result tables (depends on WP-01a/b/c/d → WP-03/04/05/06)
- Figure 1 (architecture diagram, TikZ or draw.io)
- Statistical analysis (WP-17): variance, CI, significance, effect size, multiple comparison
- Practitioner survey (WP-09 if pursued, IRB ~3 weeks)
- RQ5 (WP-18 — design + execute + write)

---

## 6. Springer EMSE Format Compliance

Pre-flight check items for WP-15 (submission):
- [x] `\documentclass[smallextended]{svjour3}` — line 11 ✓
- [x] `\smartqed` — line 12 ✓
- [x] Author + institute blocks — lines 34–49 ✓
- [x] Abstract + keywords — lines 58–69 ✓
- [ ] **No subfolders for figures/bib** — currently only 1 file (paper.tex), `template.tex` exists alongside, no figures dir yet (when figures arrive, must be flat)
- [ ] **BibTeX migration** — currently `\begin{thebibliography}` inline; Springer prefers `\bibliographystyle{spbasic} \bibliography{refs}`
- [ ] **Open Science replication package URL** — line 907 `https://github.com/barandincoguz/DDD-Enforcer` exists; needs Zenodo DOI for permanence (WP-12)
- [ ] **Declarations** block exists (lines 914–918) but Author Contributions empty
- [ ] **Compile warning-free** — not yet verified, run `latexmk -pdf paper.tex` before submission

---

## 7. Counts Summary

| Category | Count |
|----------|-------|
| `\placeholder{...}` occurrences (in-text + table cells + bibitems combined) | **~110** (most are table cells) |
| Distinct deliverables (after grouping) | **30+** |
| Figures to produce | 4 (architecture, IDE screenshot, RQ2 Pareto, RQ4 sequence) |
| Empty table cells | ~92 |
| Bibliography entries needing fix | 6 (5 fully placeholder + 1 impossible arXiv) |
| Bibliography entries to add (Hoca-3) | 8+ (per literature subagent) |
| In-text reference placeholders | 5 |
| RQ analysis paragraphs to author | 12 (3 per RQ × 4 RQs) — once data is in |
| AUTHOR-TODO blocks | 5 |
| SCOPE-NUMBER decisions | 1 explicit (`N`) + 7 implicit (M, K, Q, R, S, audit %, etc.) |

---

## 8. Top-Level Conclusion for Planning

The paper is **structurally healthy**: framing, RQs (modulo +1 from Hoca-5), evaluation protocol (LLM-Assisted Human Eval), threats taxonomy, and positioning table are all in place. What's missing is **(a) numerical evidence**, **(b) figures**, **(c) related-work depth**, and **(d) statistical rigor**. None of these are existential — they are filled by 18 well-scoped WPs.

The biggest single risk to scope is **WP-04 RQ2 (4-model comparison)**: it requires the OSS-model integration that does not exist in the codebase yet (see `00-context-report.md` Q10) and therefore depends on WP-01a (provider abstraction) being done first. This is the planning critical path.
