# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 22:55 GMT+3
**Last action:** Iterations 33-42 SHIPPED — F-16, WP-CORE-20c,
ChunkMetadata.truncated_chunks fix, WP-CORE-30b (SDD),
ownership-deprecation, WP-01b A-F (SDD), WP-01c closure (SDD).

**Session totals (this autonomous block):**
- 9 WPs shipped (F-16, WP-CORE-20c, ChunkMetadata, WP-CORE-30b,
  ownership-doc-update, WP-01b Tasks A/B/C/D/E/F, WP-01c closure)
- 25 commits (all atomic, conventional-commits)
- 611 → 716 (+105 tests net, zero regression)
- 0 MAJOR-OPEN-live findings

**Cumulative across the multi-day run:** 348 → 716 (+368 tests, ~85 commits).

**WP-01 STATUS (the user's stop condition):**
- WP-01a (Provider abstraction) — ✅ SHIPPED prior session (2026-05-19)
- WP-01b (Run orchestrator + metrics + tables) — ✅ FULLY SHIPPED this session
  - Task A: PaperRunManifest schema + writer + provenance hashes (+ 17 tests)
  - Task B: metrics.py precision/recall/F1 per type (+ 25 tests)
  - Task C: aggregate.py N-runs mean ± std + IQR + bootstrap 95% CI (+ 16 tests)
  - Task D: build_tables.py per-RQ LaTeX renderer (+ 15 tests)
  - Task E: Makefile target + tables/ scaffolding (no tests; pure build)
  - Task F: E2E smoke + legacy intermediate JSON archive (+ 1 test, 227 files moved)
- WP-01c (Token tracking + cost telemetry) — ✅ CLOSED this session
  - Most criteria already satisfied by WP-01a + Task A (pricing in
    registry, LLMResponseAdapter normalization, cost_usd field).
  - Remaining work: scripts/cost_estimate.py + multi-provider
    regression test (+ 11 tests).
- WP-01d (P1/P2/P3 pipeline classes) — DEFERRED per user (kept on backlog).

**Remaining backlog (engineering, paper-free):**
- F-8 (MINOR-OPEN — XXE hardening; needs threat model)
- WP-01d — P1/P2/P3 pipeline classes (user-deferred)
- WP-CORE-28 — Extension UX wave 1 (TypeScript, manual smoke risk)
- WP-CORE-32 — Extension webviews
- Pyright `continue-on-error` tightening + main.py ~10 type errors
- paper.tex integration of rqN.tex \input{} blocks (human-coordinator
  task — see LaTeX_DL_468198_240419/tables/README.md for the
  candidate line numbers and TODO list).
- **Minor follow-ups deferred from WP-CORE-30b code review:**
  - `_to_legacy_issue` severity-fallback silent mapping
  - `_parse_target_ctx` partial duplication with `_issue_stage`
  - `track_api_call` test spy Pyright false positive (`# type: ignore`)
  - `render_refinement_prompt` lowercase severity label
  - `_specialist_with_feedback` short-result-list risk
  - Pre-existing `token_tracker.by_stage` capitalization divergence
- **Minor follow-ups deferred from WP-01b/01c code review:**
  - `pipeline=None` grouping test gap in aggregate.py
  - `compose_aggregate_key` could deduplicate the regex shared with
    `PaperRunManifest.compose_run_id` (small util)
  - Atomic write pattern copied 3 times (run_manifest, aggregate,
    latex_tables); could extract a shared `_write_atomic` helper
  - `AggregatedConfiguration.schema_version` writer-side has no
    SUPPORTED_VERSION guard yet (consumer side does)

**Baseline:** 716 passed, 31 deselected.
**HEAD:** 4ca1301.
**Ahead of origin/main:** 25 commits (NOT pushed).

**SDD telemetry this session:**

| WP | Implementer | Spec review | Quality review | Fix loops | Total dispatches |
|----|-------------|-------------|----------------|-----------|-----------------|
| WP-CORE-30b | 2 | 2 | 2 | 1 | 9 (with final integration) |
| WP-01b Task A | 1 | 2 | 1 | 2 | 7 |
| WP-01b Task B | 1 | 1 | 1 | 1 | 4 |
| WP-01b Task C | 1 | 1 | 1 | 1 | 4 |
| WP-01b Task D | 1 | 1 (combined) | (combined) | 1 | 3 |
| WP-01b Task E | 1 | 0 (scaffolding) | 0 | 0 | 1 |
| WP-01b Task F | 1 | 0 (smoke + mv) | 0 | 0 | 1 |
| WP-01c | 1 | 1 (combined) | (combined) | 1 | 3 |
| **Total** | **9** | **8** | **5** | **7** | **32 subagent dispatches** |

**Recommendation for next session:**

WP-01 chain is COMPLETE except WP-01d (user-deferred). Next ranked
options:

1. **Pyright tightening** — main.py ~10 type fixes + CI gate
   `continue-on-error: false`. Mid risk, deterministic. ~1-2h.
2. **WP-01d (P1/P2/P3 pipeline classes)** — if user un-defers. Big WP;
   3-5 SDD tasks; ~6-10h.
3. **WP-CORE-28 / WP-CORE-32** — Extension UX (TypeScript, manual
   smoke). Mid risk; needs human VS Code session.
4. **F-8 XXE hardening** — security audit follow-up. Small but
   needs threat model first.
5. **Minor deferred concerns sweep** — close the 10+ accumulated
   Minor findings across WP-CORE-30b + WP-01b/c reviews in one batch.
6. **paper.tex `\input{}` integration** — paper-coordinator task; NOT
   autonomous-safe. Human reviews `tables/README.md` candidate line
   numbers and inserts.

**Operational rules carried forward:**
- Caveman mode ACTIVE (full)
- Communication TR, code EN
- Accuracy > cost (WP-CORE-21 + WP-CORE-34 REJECTED — see
  `feedback-accuracy-over-cost.md`)
- **Ownership disestablished (2026-05-23)** — any agent picks up
  any WP; `WP_DAGILIM_BARAN_ALI.md` historical only. See
  `feedback-ownership-disestablished.md`.
- NO `git push` without explicit "push it"
- Codex xhigh skipped for small/mechanical/schema-additive WPs
- TDD strict: RED → GREEN → DOC → COMMIT
- Subagent-Driven Development (SDD) the default for cross-stage /
  multi-file / Codex-REQUIRE WPs per
  `superpowers:subagent-driven-development`. SDD telemetry above
  shows the pattern is high-throughput + high-quality (zero
  regression across 25 commits).
