# WP-08: Author Audit + Cohen's κ (Dual Independent Audit)

**Owner:** Baran AND Ali (each independently audits the same sample)
**Depends-on:** [WP-07, all RQ runs (WP-03..06)]
**Effort:** M (audit volume = 25% × ~1500 violations across all RQs ≈ 375 cases × 2 authors)
**Status:** TODO
**Addresses instructor feedback:** [Hoca-6] (κ + dual-author independence)

## Goal

Replace the current "single-author audit" plan with a **dual-author independent audit** that produces Cohen's κ (between authors) **and** a Judge-vs-author override rate. This directly addresses the reviewer threat "Cohen's κ yok, tek annotator var" (B.3 in `01-brainstorming.md`).

## Acceptance criteria

- [ ] `evaluation/audit_overrides.csv` exists with columns: `run_id`, `violation_id`, `judge_classification`, `author_a_classification`, `author_b_classification`, `consensus_classification`, `note`.
- [ ] Each row is independently filled by Baran and Ali — they do **not** see each other's classifications until both are complete.
- [ ] Cohen's κ computed between Baran and Ali per RQ × per violation type. Reported in §4.5 (eval reliability) and §9.3.3 (eval subjectivity).
- [ ] Disagreement resolution protocol: if Baran ≠ Ali, both convene + apply rubric → consensus. Document protocol in §4.5.
- [ ] Audit sample = 25% random + ALL low-confidence Judge verdicts. Total audited cases logged in `evaluation/audit_summary.json`.
- [ ] Audit override rate (Judge vs. consensus) reported as the κ-analogue.
- [ ] §4.5 prose updated to describe the dual-author protocol. §9.3.3 updated to acknowledge "two TEDU authors share systematic bias on borderline cases" as residual.

## Implementation steps

1. After all RQ runs + Judge verdicts produced, sample 25% random + all low-confidence cases.
2. Generate two CSVs (one for Baran, one for Ali) with columns `run_id`, `violation_id`, `framework_report`, `judge_verdict`, `srs_excerpt`, `code_excerpt`. **Authors do not see Judge classifications during their own pass.** (Override the Judge column with `[BLINDED]`.)
3. Each author independently classifies each row TP/FP/FN.
4. Merge the two CSVs. Compute κ (`scipy.stats.cohen_kappa_score` or `sklearn.metrics.cohen_kappa_score`).
5. For disagreements, both authors convene + apply rubric → consensus row.
6. Compute Judge-vs-consensus override rate per RQ.
7. Update §4.5 + §9.3.3 prose.

## Outputs (file paths)

- `evaluation/audit_baran.csv` (Baran's pass)
- `evaluation/audit_ali.csv` (Ali's pass)
- `evaluation/audit_overrides.csv` (merged + consensus)
- `evaluation/audit_summary.json` (Cohen's κ per RQ, override rate per RQ)
- Updated `paper.tex` §4.5, §9.3.3 prose
- `replication_package/audit/` (full anonymized audit log)

## Risks & mitigations

- **Risk:** Audit is tedious and gets rushed. **Mitigation:** Time-box per author to 20 minutes per 30 violations; if pace falls behind, reduce sample to 20% + flag in §4.5 transparently.
- **Risk:** Authors share systematic bias (both think V3 borderline cases are FPs when they should be TPs). **Mitigation:** §9.3.3 explicitly names this as residual; future-work item for external-annotator validation.
- **Risk:** Low-confidence cases are concentrated in one violation type, skewing per-type κ. **Mitigation:** Report aggregate κ + per-type κ; comment on per-type variance in §4.5.
