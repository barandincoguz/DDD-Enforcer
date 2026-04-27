# Risk Register — EMSE Submission

**Date:** 2026-04-27
**Scope:** Risks to landing a submission-ready paper in 12–16 weeks with **2 authors (Baran + Ali) + 1 advisor (Murat Hoca, review-only)**.

---

## 1. Top risks (ranked)

### R1. **Two-author capacity vs. 18-WP scope** (LIKELIHOOD: HIGH, IMPACT: HIGH)

- **Description.** v1 of the planning prompt assumed 3 authors → 10–14 weeks. v3 explicitly downgrades to 2 authors and adds infra-rewrite WPs (WP-01a/b/c) → 12–16 weeks. With both authors at "moderate-intensive" pace and Hoca review gates, this is feasible but **fragile to a single major delay**.
- **Quantitative impact.**
  - Scenario A (smooth): 12 weeks if WP-09 IRB is parallelized successfully.
  - Scenario B (one major slip — e.g., IRB delay or RQ4 generator-LLM debugging): 16 weeks.
  - Scenario C (two slips): 18–20 weeks. **Submission slips to Q3.**
- **Mitigation.**
  1. **De-scope WP-09** if recruitment is hard — fall back to RQ5=(A) Ablation only.
  2. **Front-load infrastructure** (WP-00 + WP-01a-c-d) in weeks 1–3. Without this, RQ deneyleri boşa enerjiye dönüşür.
  3. **Hoca review gates ≤ 5 working days** — otherwise critical-path slips. Recommend pre-scheduling 4 review meetings (after WP-00, after RQ2 completion, after RQ4 completion, before submission).
  4. **WP-12 replication package can be parallelized** with WP-13/14 prose work — assign to whichever author has the "lighter week".
- **Trigger to escalate.** Any single WP slipping more than +3 days past its planned end.

### R2. **OSS model integration is harder than expected** (LIKELIHOOD: MEDIUM, IMPACT: HIGH)

- **Description.** The paper claims a 4-model RQ2 (Gemini, GPT, Claude, OSS-local), but the codebase has zero OSS-model integration (`00-context-report.md` Q10). Bringing up Qwen2.5-Coder-32B (or similar) with structured-output enforcement (Pydantic) on local hardware is non-trivial. Tools like Ollama's JSON-mode, vLLM, or guidance/outlines may have schema-compliance edge cases that add 1–2 weeks of debugging.
- **Quantitative impact.** +1 to +2 weeks on the critical path (delays WP-04 RQ2 → cascade).
- **Mitigation.**
  1. **WP-01a includes a "smoke-test 4 providers on a tiny SRS" subtask** — this surfaces schema-compliance issues in week 2, not week 7.
  2. **Pin a fallback OSS model** — if Qwen2.5-Coder-32B fails, fall back to Llama-3.1-70B or DeepSeek-Coder-V2 with cloud inference (Together AI / Replicate). Note in §4.3 that "open-source" includes "open-weights, third-party-hosted" if local hardware insufficient.
  3. **Hardware check immediately:** verify available GPU memory (a 32B model in 4-bit needs ~20 GB). If insufficient, swap to smaller model or third-party hosting.
- **Trigger to escalate.** WP-01a smoke-test fails in week 2 → engage cloud-hosted OSS model fallback.

### R3. **WP-09 practitioner survey gets IRB-blocked or under-recruited** (LIKELIHOOD: MEDIUM, IMPACT: MEDIUM)

- **Description.** TEDU IRB review typically takes 2–4 weeks. Recruitment of 12 microservice developers from LinkedIn / TEDU alumni / meetups can stall, especially if the call requires DDD familiarity.
- **Quantitative impact.** Up to +4 weeks if WP-09 is on the critical path. **However**, if WP-09 is the *appendix-(D)* (not RQ5 itself), it can slip without affecting submission.
- **Mitigation.**
  1. **Start IRB application in week 1**, regardless of whether WP-09 is "definitely happening". The 2-week IRB lead-time is a sunk cost worth paying for optionality.
  2. **Recruit in parallel with infra work.** Don't wait for IRB approval to draft the recruitment text + survey instrument.
  3. **Fallback:** if recruitment yields fewer than 8 developers, reframe as "case study" (n ≥ 5) — qualitative themes still publishable.
  4. **De-scope:** if RQ5 = (A) Ablation is chosen and WP-09 falls behind, **drop WP-09 entirely**. The Ablation alone is a strong RQ5 without industry-relevance evidence.
- **Trigger to escalate.** No IRB response by week 4 OR fewer than 5 confirmed developer participants by week 8.

### R4. **`automatingddd2024` arXiv ID has no real replacement** (LIKELIHOOD: LOW, IMPACT: LOW)

- **Description.** The current bibliography entry `\bibitem{automatingddd2024}` cites `arXiv:2603.26244`, which is **structurally impossible** for a 2024 paper. Web search was unavailable during planning, so we have not found a real replacement.
- **Quantitative impact.** None on critical path; ~1 day of literature work in WP-10. Worst case: prose changes in §2.5, §9.2 to drop the citation.
- **Mitigation.**
  1. **Literature WP-10 first sub-task:** verify or replace `automatingddd2024`.
  2. **Acceptable end-state:** delete the citation; rewrite §2.5/§9.2 to lean on `evans2024infoq` + a verified industry writeup.
- **Trigger to escalate.** None — this is contained in WP-10 and known to the team.

### R5. **Statistical methodology (WP-17) reveals N=5 is too small** (LIKELIHOOD: MEDIUM, IMPACT: MEDIUM)

- **Description.** Power analysis may indicate N=5 is insufficient for a Wilcoxon test with effect size < 0.5 (very common in LLM evaluation). If so, headline P3 > P2 claim may not reach significance.
- **Quantitative impact.** Either (a) bump N=5 to N=10 (+10% to RQ1–4 compute and runtime, ~1 week extra) or (b) reframe claims more cautiously ("difference observed; sample size too small for significance" — weaker but defensible).
- **Mitigation.**
  1. **WP-17 power analysis runs in week 4–5** (before RQ1 execution at scale). If N=5 is insufficient, decision is made *before* expensive runs are committed.
  2. **Pilot run on D1 with N=5** in week 4: if cell-level std is large (> 0.1 F1), bump to N=10.
  3. **Alternative metric:** report bootstrap CIs of the *differences* (P3 − P2) per file × per type, even at small N — this is more interpretable than significance testing for low-N studies.
- **Trigger to escalate.** Pilot run shows std > 0.15 F1 → trigger N expansion conversation immediately.

### R6. **Hoca's review gates take too long** (LIKELIHOOD: MEDIUM, IMPACT: MEDIUM)

- **Description.** If a Hoca review takes 2 weeks instead of 5 days, two such gates push submission by a month.
- **Mitigation.**
  1. **Pre-schedule review gates** at the kickoff: end of Phase 0–1 (week 1), after RQ1+RQ2 done (week 8), after WP-13 prose done (week 13), pre-submit (week 15).
  2. **Async reviews** — share Google Doc/PDF + bullet-point summary; don't require Hoca to read the full paper at every gate.
  3. **Default-to-proceed rule:** if no Hoca response after the agreed deadline, proceed and treat his eventual feedback as in-flight changes rather than blockers.
- **Trigger to escalate.** No Hoca response 7 days after a review gate.

### R7. **Pre-existing intermediate runs (`/extension/backend/core/intermediate/`, 154 files) lure the team into "we already have results"** (LIKELIHOOD: LOW, IMPACT: HIGH)

- **Description.** The codebase has 154 intermediate JSON files dated 2026-03-12/13. These were produced by a *single-pass* run (`temperature=0.05`, `seed=42`) on a single SRS and a single Gemini model. They are **not** N=5 averaged, **not** multi-model, **not** multi-domain. The temptation to "use what we have" risks publishing point estimates without variance.
- **Quantitative impact.** If this happens, peer-review desk-rejects on Hoca-1 + Hoca-6.
- **Mitigation.**
  1. **Explicitly mark old runs as `legacy_pre_emse/`** in the planning. Do not re-use without re-execution under WP-01b run-manifest format.
  2. **WP-01b acceptance test:** all paper tables auto-render from runs/ — manual cherry-picking blocked.

### R8. **Springer EMSE format / submission portal surprises** (LIKELIHOOD: LOW, IMPACT: LOW)

- **Description.** Springer occasionally tightens svjour3 or the Editorial Manager submission requirements (figure formats, supplementary file size limits, ORCID requirements, suggested-reviewer rules).
- **Mitigation.**
  1. **WP-15 last-week dry-run:** practice submission against the EM portal, but do NOT click "submit". Resolve any portal complaints before the real submission.
  2. **Compile warning-free:** `latexmk -pdf paper.tex` should produce zero warnings before any submission.

### R9. **Expensive RQ2 cloud bills surprise the team** (LIKELIHOOD: LOW, IMPACT: LOW–MEDIUM)

- **Description.** RQ2 = 4 models × N=5 × ~30 files/run × token-heavy violation detection. With Claude Opus or GPT-5 this could hit $200–$500 in API costs.
- **Mitigation.**
  1. **Token budget per run estimated by WP-01c** before launching RQ2 in week 7. If estimate exceeds budget, swap to smaller model variants (Sonnet vs. Opus, mini-models for OpenAI).
  2. **Smoke-test on 1 file** before launching the 30-file × 5-run × 4-model job.

### R10. **Reviewer requests "scaling to 5 domains, multiple Judge LLMs, real industrial codebase" — major revision** (LIKELIHOOD: MEDIUM in any LLM-SE paper, IMPACT: HIGH)

- **Description.** This is the typical "major revision" pattern from EMSE/TSE/TOSEM. We cannot pre-empt it fully.
- **Mitigation.**
  1. **Pre-emptively run 2 Judge LLMs sensitivity check** in WP-07 — costs ~10% extra compute, defuses the "single Judge bias" criticism.
  2. **Pre-build infrastructure to scale** (so that a 4th, 5th domain can be added after submission within 2 weeks if reviewer demands).
  3. **Frame as "framework, not benchmark"** in the submission letter — this preempts "scope creep" review.
- **Trigger.** Inevitable; treat as part of normal review cycle, not as a project failure.

---

## 2. Submission-ready realistic timeline (under different scenarios)

| Scenario | Duration | Conditions |
|----------|----------|------------|
| **Optimistic** | 11 weeks | Both authors fully available, no IRB, RQ5=(A) only, no major slips |
| **Realistic (default)** | 14 weeks | Standard slip distribution, IRB-parallelized, RQ5=(A)+appendix-(D) |
| **Pessimistic** | 18 weeks | One major slip + Hoca review gate delay + IRB extended |
| **Worst-case** | 22 weeks | Two slips + cancelled survey + significance issues require N=10 expansion |

**Recommendation:** Plan for **14 weeks**, communicate buffer up to 16, internally hold cushion to 18.

---

## 3. Pre-emptive moves (do these in week 1–2)

1. **Submit IRB application for WP-09** (3-week clock starts immediately, regardless of whether RQ5 is (A) or (D)).
2. **Smoke-test 4 LLM providers** with WP-01a — 1 file, 1 violation type, 30 minutes total — surfaces structured-output edge cases early.
3. **Lock the scope numbers in `configs/scope.yaml`** (WP-00) so that all WPs reference identical N, M, K, Q, R, S.
4. **Verify the impossible arXiv ID** in `automatingddd2024` (WP-10 sub-task) — 1 hour of literature search.
5. **Set up Hoca review gate calendar** — 4 meetings, agreed deadlines.
6. **Mark 154 legacy run artifacts as `legacy_pre_emse/`** — do not reuse without WP-01b re-execution.

---

## 4. What major-revision request would force us to redo work?

If reviewers ask for one of the following, we lose the most time:

1. **Scaling to 5+ domains** — requires WP-02 expansion to D4, D5; +3 weeks if D4/D5 SRSes are pre-staged.
2. **Multiple Judge LLMs (cross-Judge sensitivity)** — +1 week if Judge harness in WP-07 is parameterized; +3 weeks if hardcoded to one Judge.
3. **Real industrial codebase** — +6 weeks; requires industry partnership; might be infeasible for first revision.
4. **Independent annotators (non-author)** — +4 weeks if not pre-arranged; +1 week if a TEDU graduate-student annotator pool is identified up front.

**Pre-emptive insurance:** WP-07 should be parameterized for any Judge LLM (≤1-day swap), WP-02 should leave hooks for D4/D5 (a fourth SRS pre-cleaned but not analyzed), and WP-09 if pursued should ideally include 1–2 non-author annotators in the panel.
