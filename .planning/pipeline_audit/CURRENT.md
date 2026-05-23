# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 22:00 GMT+3
**Last action:** Iteration 19-27 SHIPPED — WP-CORE-22 through WP-CORE-31 (skip 28). Tier 1 + Tier 2 + partial Tier 3 complete.

**Session totals (single autonomous session):**
- 9 WPs shipped (WP-CORE-22, 23, 24, 25, 26, 27, 29, 30, 31)
- 11 commits (atomic)
- 459 → 560 (+101 tests, zero regression)
- 0 MAJOR-OPEN-live findings

**Remaining backlog (engineering, paper-free):**
- F-8 (MINOR-OPEN — XXE hardening; needs threat model)
- F-16 (TRIVIAL-OPEN — dead extract_domain_sentences cascade)
- WP-CORE-27a — AST→ValueObject.is_mutable_in_code population (follow-up)
- WP-CORE-30b — render_refinement_prompt wiring + per-context narrow rerun
- WP-CORE-31b — pipeline integration of import_graph helpers
- WP-CORE-20b — wrap architect stages in emitter.stage scopes
- WP-CORE-28 — Extension UX wave 1 (TypeScript, manual smoke risk)
- WP-CORE-32 — Extension webviews
- WP-CORE-33 — V7 ACL + V8 Specification + V9 Service kind (NEXT-RECOMMENDED)
- WP-01b — run-spec orchestrator (paper data infra; user-approved)
- rag_pipeline.py `_parse_sections` regex period-form gap

**Baseline:** 560 passed, 31 deselected.
**HEAD:** 273af57.

**Recommendation for next session:**
Default if user says "devam": WP-CORE-33 (V7 ACL + V8 Specification + V9 Service kind). Backend-only, autonomous-safe, ~25-30 tests, builds directly on WP-CORE-22 Repository/Factory pattern.

Full handoff: `.planning/pipeline_audit/handoff-2026-05-23-2200.md`.

**Operational rules carried forward**:
- Caveman mode ACTIVE (full)
- Communication TR, code EN
- Accuracy > cost (WP-CORE-21 + WP-CORE-34 REJECTED — see `feedback-accuracy-over-cost.md`)
- NO `git push` without explicit "push it"
- Codex xhigh skipped for small/mechanical/schema-additive WPs (established pattern)
- TDD strict: RED → GREEN → DOC → COMMIT
