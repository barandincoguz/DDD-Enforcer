# Strategic Brainstorming — EMSE Submission

**Date:** 2026-04-27
**Inputs synthesized:** `00-context-report.md`, `00-paper-baseline.md`, `00-instructor-feedback-mapping.md`, `02-literature.md`.

---

## A. Gap Statement (the 30-second editor pitch)

**Methodological gap.** Existing LLM-as-Judge protocols in SE (e.g., `llmjudgeSE2025` and the broader survey landscape) typically evaluate single-turn code-related outputs and report aggregate Pearson correlations against expert raters. Our protocol — _LLM-Assisted Human Evaluation_ — differs in three explicit ways: (a) it requires a strictly stronger Judge from a _different_ model family than every tested model (eliminating intra-family agreement bias), (b) it routes Judge verdicts through a 25%-random + all-low-confidence author audit and reports the override rate as a κ-analogue, and (c) it pre-registers the rubric so the same protocol can be re-run by a future researcher who plugs a new pipeline or model into the same harness. Where prior LLM-as-Judge work asks "does the Judge agree with humans?", we ask "in what fraction of audited cases does the Judge survive a human pass?" — a stricter test for a proof-of-concept.

**Application gap.** The published LLM-DDD literature is concentrated on the _design_ side: event-storming acceleration, ubiquitous-language drafting, bounded-context discovery (typified by the prompting-framework citation we currently flag as `automatingddd2024` plus `evans2024infoq` and industry write-ups). To the best of our knowledge, no peer-reviewed work has systematically studied _enforcement_ — keeping an evolving codebase consistent with a previously-established domain model — and certainly none has compared multiple pipeline architectures (naive vs. retrieval-augmented vs. multi-agent) head-to-head on the enforcement task. This is the gap DDD-Enforcer occupies.

**Conference→Journal gap.** Table 2 in §3.5 already enumerates 8 measurable deltas from `dincoguz2025ddd_conference`. Crucially, every delta is _quantitative_: 3 violation types → 6 (V1–V6); single LLM prompt over the full SRS → 4-stage Pydantic-enforced pipeline; free-form text post-hoc parsed → Pydantic-enforced ValidationResponse; illustrative examples → 3-pipeline × 4-model × 3-domain × N=5-run empirical study; CLI-only → VS Code extension with 4 commands and proper diagnostic API; no traceability → embedding-based traceability (decoupled from detection); no methodology framing → LLM-Assisted Human Evaluation protocol. This is roughly **2× novelty** by EMSE-extension standards (~30–50% new content typically required).

**One-sentence editor pitch.** "DDD-Enforcer fills the LLM-DDD enforcement gap with the first head-to-head empirical comparison of pipeline architectures, model providers, and SRS domains for automatic Domain-Driven Design violation detection, evaluated under a transparent and reproducible LLM-Assisted Human Evaluation protocol."

---

## B. Reviewer Threat Modeling — 9 Attack Vectors

For each, we list (a) the likely reviewer phrasing, (b) the paper section that defends, (c) the WP that produces the supporting evidence, (d) the residual risk that remains.

### B.1 "N belirsiz, kaç ölçüm aldınız?" (Hoca-1)

- **Reviewer phrasing.** "All your tables report point estimates with no indication of how many independent runs were averaged or what the run-to-run variance looks like. Without this we cannot interpret the headline numbers."
- **Defended in.** §4.7 Reproducibility (run dir naming declares N), §9.3.4 Internal validity (states N + variance + bootstrap CI).
- **Evidence WPs.** WP-00 (commits N=5), WP-01b (auto-aggregation produces mean/std/CI), WP-17 (statistical methodology section authored), WP-13 (threats prose updated).
- **Residual risk.** None if WP-00 + WP-01b + WP-17 land cleanly. This is our strongest correctable threat.

### B.2 "Variance / CI / significance test yok." (Hoca-6)

- **Reviewer phrasing.** "Even if N is 5, you do not perform any significance test for the headline P3 > P2 or model-ranking claims; the differences may be within noise."
- **Defended in.** New §4.7 (Statistical Analysis Plan, pre-registered), expanded §9.3.4.
- **Evidence WPs.** WP-17 — Wilcoxon signed-rank for paired pipeline comparisons, Friedman + Nemenyi for model rankings, Cliff's δ for effect size, Holm correction.
- **Residual risk.** N=5 is on the low end for non-parametric tests. Mitigation: report effect size alongside p-values; in §9.3 pre-register the statistical plan to defuse "p-hacking" objections.

### B.3 "Cohen's κ yok, tek annotator var."

- **Reviewer phrasing.** "Author-Judge agreement rate is _not_ Cohen's κ. With only one annotator per audited case, you cannot compute inter-rater reliability."
- **Defended in.** §4.5 Eval protocol — argument is currently that "agreement rate plays the κ role" but reviewer may not buy this.
- **Evidence WPs.** WP-08 — _two authors independently audit_ the 25% random sample, compute κ between authors, and _also_ report their disagreements with the Judge.
- **Residual risk.** Two TEDU authors may share systematic bias on borderline cases. Mitigation: §9.3.3 explicitly acknowledges this; future work flags external-annotator validation.

### B.4 "Üç domain, generalization claim?"

- **Reviewer phrasing.** "Three domains is not enough to claim generalization; reframe or expand."
- **Defended in.** §1, §4.2, §9.3.5 — already say "probe, not benchmark" three times. The current paper already mostly self-defends.
- **Evidence WPs.** WP-13 (tightening the prose so that no sentence in §6 RQ3 results overclaims). WP-02 (justifying the _selection_ of the three domains in §4.2).
- **Residual risk.** Reviewer may still demand 5–6 domains. Mitigation: position as "future work, framework supports it via swappable subjects/" in §10.

### B.5 "RQ4 confirmation bias — aynı yazar inject + evaluate."

- **Reviewer phrasing.** "If the same author seeds the violations and runs the framework, the seeded set may unconsciously be tuned to what the framework can detect."
- **Defended in.** §4.6 Synthetic-Violation Protocol — _currently_ says "Authors then inject…" with no role-separation.
- **Evidence WPs.** WP-06 — **blind injection protocol**: Author A injects violations and writes the manifest; Author B (without seeing the manifest) runs the pipeline; Author A then matches the pipeline output back against the manifest. The pre-commit of the manifest (timestamped git commit) makes the role-separation auditable.
- **Residual risk.** Two-author teams have a finite "blindness budget". Mitigation: pre-register the seed catalogue _before_ WP-04 RQ2 results are known.

### B.6 "Industry relevance?" (EMSE-special)

- **Reviewer phrasing.** "EMSE prizes applied research with industry impact. What evidence have you produced that practitioners would actually use this?"
- **Defended in.** Currently: VS Code extension shipped (1.0.0); abstract says "doubles as onboarding artifact" — this is a _claim_, not evidence.
- **Evidence WPs.** **WP-09 practitioner survey** (12–20 microservice developers, IRB-approved) **OR** **WP-18 RQ5 = Developer Study** (combines into a §"Practitioner Perspective" section).
- **Residual risk.** Without empirical industry-relevance evidence, this is the single biggest risk for EMSE acceptance. Recommended: pursue at least one of WP-09 or RQ5=Dev-Study.

### B.7 "Replication package?" (EMSE Open Science)

- **Reviewer phrasing.** "Where is your replication package? EMSE 2024+ Open Science Initiative requires a replication package URL in the paper."
- **Defended in.** §10 Data Availability — currently has GitHub URL + AUTHOR TODO for full replication URL.
- **Evidence WPs.** WP-12 — Zenodo DOI + GitHub release tag + REPLICATION.md.
- **Residual risk.** Low. The data exists (configs, prompts, intermediate run outputs) — just needs packaging.

### B.8 "LLM çıkacak yarın, sonuçlar geçersiz."

- **Reviewer phrasing.** "Your model rankings will be obsolete in 6 months as new models release. Why publish results that are time-bounded?"
- **Defended in.** §1, §4.3, §9.1 already use "framework, not point solution", "today's capability level", "swappable" language. Strong existing defense.
- **Evidence WPs.** WP-13/WP-14 (tightening these phrasings for consistency); WP-12 (replication package makes "re-run with new model" trivial).
- **Residual risk.** This argument is well-rehearsed in the LLM-SE literature; reviewers tend to accept it when it's framed as "framework + snapshot".

### B.9 "Related work zayıf." (Hoca-3)

- **Reviewer phrasing.** "Sections 2.1–2.6 read as 1–2-paragraph stubs. Where is the engagement with classical architecture-conformance literature, microservice quality, or recent LLM-SE empirical work?"
- **Defended in.** Currently: 5 in-text placeholder blocks + 5 placeholder bibliography entries.
- **Evidence WPs.** WP-10 — pull from `02-literature.md` shortlist; ensure each subsection has 3–5 high-quality refs. Include 2022+ EMSE/TSE/TOSEM/ICSE/FSE works.
- **Residual risk.** Low if literature subagent's output is verified. Time-bounded.

---

## C. RQ5 Decision Table — 3 Favorites Picked, 1 Awaits Baran

The original prompt asks for 3 favorites with explicit reasoning. The decision is Baran's; this section frames it.

### C.1 RQ5 candidate analysis (5 options)

| RQ5                                  | What it measures                                                                                                      | Effort | Reuses RQ1–4 data?                                               | EMSE-fit (1–5) | Downside                                              |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------- | ------ | ---------------------------------------------------------------- | -------------- | ----------------------------------------------------- |
| **(A) Ablation: AST removed**        | Does AST grounding actually help vs. LLM-only on the same SRS+code+rulebook?                                          | M      | ✅ Yes — same 90 seeded files (RQ4) and same 3-domain runs (RQ3) | 4              | Ablation is "internal", not "industry-facing"         |
| (B) Prompt sensitivity               | How much do small prompt rewrites shift the F1 numbers?                                                               | S      | ✅ Partial — re-run with 2 prompt variants                       | 3              | Reviewer can dismiss as a §9.3 threat-paragraph       |
| (C) Rulebook stability               | Same SRS, N=5 rulebook extractions — how similar are the rulebooks?                                                   | S      | ✅ Yes — already implicit in WP-01b N=5 design                   | 3              | Repeats determinism story already partially in §9.3.4 |
| **(D) Developer/practitioner study** | Show 5–10 developers a violation list; ask "useful / wrong / neither?"; report kappa-agreement and qualitative themes | L      | ❌ New data collection                                           | 5              | IRB ~3 weeks; recruitment; longest critical-path risk |
| (E) Rulebook drift over time         | How does detection degrade as the SRS evolves?                                                                        | XL     | ❌ Need 2 SRS versions per domain                                | 4              | Requires new corpus; cannot fit in 12–16 weeks        |

### C.2 The 3 favorites (default ranking)

**🥇 Default: (A) Ablation — AST features removed**

- **Why first:** Reuses every byte of the RQ1–RQ4 dataset (cost = a few extra runs with `--no-ast` flag). It directly tests the architectural claim that AST + LLM > LLM alone, which is the _core selling point_ of DDD-Enforcer over a generic LLM chat interface. Reviewer-strong.
- **Effort:** ~1 week (1 author): add a `--no-ast` mode to the validator; re-run RQ1 winner and RQ4 seeded set; produce ablation table; write half a page.
- **Implementation tap:** WP-01a already gives us the LLM client; WP-01d already has structured output; only thing to add is conditional AST injection.

**🥈 Strong second: (D) Developer study**

- **Why second:** This is the single intervention that most directly addresses the "EMSE industry relevance" reviewer threat (B.6 above). It also produces a qualitative narrative ("developers found V1 Synonym most actionable, V5 Aggregate Boundary least actionable…") that is hard to fake from training data and immediately differentiates from "yet another LLM accuracy paper".
- **Effort:** ~4 weeks if pursued in parallel with infra (1 author runs IRB + recruitment while the other does WP-01a/b/c). 3-week IRB delay is the longest single critical-path stretch in the project.
- **Risk:** IRB rejection / 1-month delay. Recruitment shortfall (target 12, achieve 6). Mitigation: start IRB **immediately**, fall back to a 5-developer "case study" framing if recruitment is short.

**🥉 Bronze: (C) Rulebook stability**

- **Why third:** Smallest cost (already partially measured), reinforces determinism story. But (1) overlaps with §9.3.4 internal-validity, (2) doesn't open a fundamentally new analysis dimension. Best as a "sub-RQ" inside RQ1 rather than a standalone RQ5.

### C.3 Default recommendation

**Pick (A) as RQ5.** Strong + cheap + EMSE-internal.

**If team capacity allows: pick (A) as RQ5 + (D) as a §"Practitioner Perspective" appendix subsection** — frame it as a complementary qualitative validation rather than a 6th RQ. This avoids "RQ creep" while still buying industry-relevance signal.

**Avoid:** (B) and (C) as RQ5 (better as threats discussion); (E) as RQ5 (out of scope for current corpus).

### C.4 What Baran needs to choose

| Option                   | Pros                                  | Cons                                     | Timeline impact                      |
| ------------------------ | ------------------------------------- | ---------------------------------------- | ------------------------------------ |
| RQ5 = (A) only           | Cheap, fast, defends the architecture | Doesn't add industry signal              | +1 week                              |
| RQ5 = (A) + appendix-(D) | Best of both                          | (D) is the long pole; might slip         | +1 week (A) + IRB-bound (D) parallel |
| RQ5 = (D) only           | Strongest industry signal             | Single-RQ relies on IRB cooperating      | +3 weeks IRB + 2 weeks study         |
| RQ5 + RQ6 = (A) + (D)    | Maximum reviewer-armor                | Two new RQs to write × 2 result sections | +4 weeks                             |

**Sevkiyat tarihi sıkıysa:** (A) only. **Sevkiyat tarihi esnek + IRB hızlı ise:** (A) + appendix-(D) (best balance).

→ **Routes to:** WP-18 (the WP detailing the chosen RQ5 once Baran decides).

---

## D. Position summary for Baran's review

The empirical core of this paper is sound: framing, RQs (with the +1), evaluation protocol, threats taxonomy, and architecture description are all in good shape. The work concentrates in three places:

1. **Numerical evidence pipeline** (WP-00/01a-d/02/07/03–06/08/17) — the longest path; ~10 weeks of total effort across two authors; depends on the OSS-model integration (WP-01a sub-deliverable) being done before WP-04 RQ2 can run.
2. **Paper polish** (WP-10 bibliography, WP-11 figures, WP-13 prose, WP-14 abstract, WP-16 extension docs) — mostly parallelizable, ~3 weeks of focused writing once data is in.
3. **One discretionary investment** (WP-09 practitioner survey OR WP-18 RQ5 = Dev-Study OR both) — IRB-bounded; if pursued, must be started in week 1 to not become the critical path.

**The overall gating risk** is whether 2 authors can land 18 WPs in 12–16 weeks. See `01-risks.md` for the quantitative risk register.
