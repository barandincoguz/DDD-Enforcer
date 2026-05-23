# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 19:33 GMT+3
**Last action:** Iterations 28-32 SHIPPED — WP-CORE-33 + WP-CORE-31b +
WP-CORE-20b + WP-CORE-29b + WP-CORE-27a.  Tier 3 sweep + V7/V8/V9
pattern detection + observability gap closures.

**Session totals (this autonomous block, post-/clear):**
- 5 WPs shipped (WP-CORE-33, 31b, 20b, 29b, 27a)
- 5 commits (atomic)
- 560 → 611 (+51 tests, zero regression)
- 0 MAJOR-OPEN-live findings

**Cumulative across the multi-day run:** 348 → 611 (+263 tests, ~60 commits).

**Remaining backlog (engineering, paper-free):**
- F-8 (MINOR-OPEN — XXE hardening; needs threat model)
- F-16 (TRIVIAL-OPEN — dead extract_domain_sentences cascade)
- WP-CORE-20c — rerun StageRecord preservation (WP-CORE-20b follow-up)
- WP-CORE-30b — render_refinement_prompt wiring + per-context narrow rerun
- ChunkMetadata.truncated_chunks always-zero count fix
- WP-CORE-28 — Extension UX wave 1 (TypeScript, manual smoke risk)
- WP-CORE-32 — Extension webviews
- WP-01b — run-spec orchestrator (paper data infra; user-approved)
- Pyright `continue-on-error` tightening + main.py ~10 type errors

**Baseline:** 611 passed, 31 deselected.
**HEAD:** effd5df.
**Ahead of origin/main:** 17 commits (NOT pushed).

**Recommendation for next session:**
Default if user says "devam":
1. **F-16** dead `extract_domain_sentences` cleanup (trivial, <15 min).
2. **WP-CORE-20c** rerun StageRecord preservation — closes the
   WP-CORE-20b documented follow-up (architect rerun overwrites prior
   `manifest.stages["architect"]` record).

Full handoff: `.planning/pipeline_audit/handoff-2026-05-23-1933.md`.

**Operational rules carried forward**:
- Caveman mode ACTIVE (full)
- Communication TR, code EN
- Accuracy > cost (WP-CORE-21 + WP-CORE-34 REJECTED — see
  `feedback-accuracy-over-cost.md`)
- NO `git push` without explicit "push it"
- Codex xhigh skipped for small/mechanical/schema-additive WPs
- TDD strict: RED → GREEN → DOC → COMMIT
