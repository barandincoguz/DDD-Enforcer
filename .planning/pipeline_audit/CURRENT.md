# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 20:55 GMT+3
**Last action:** Iterations 33-36 SHIPPED — F-16 + WP-CORE-20c +
ChunkMetadata.truncated_chunks fix + WP-CORE-30b (Tasks 1 + 2 + cleanup
via Subagent-Driven Development).

**Session totals (this autonomous block, post-/cont):**
- 4 WPs shipped (F-16 cleanup, WP-CORE-20c, ChunkMetadata fix, WP-CORE-30b)
- 5 commits (atomic, 3 of them ship WP-CORE-30b in SDD task split)
- 611 → 632 (+21 tests net, -1 dead test removed by F-16, zero regression)
- 0 MAJOR-OPEN-live findings

**Cumulative across the multi-day run:** 348 → 632 (+284 tests, ~65 commits).

**Remaining backlog (engineering, paper-free):**
- F-8 (MINOR-OPEN — XXE hardening; needs threat model)
- WP-CORE-28 — Extension UX wave 1 (TypeScript, manual smoke risk)
- WP-CORE-32 — Extension webviews
- WP-01b — run-spec orchestrator (paper data infra; user-approved)
- Pyright `continue-on-error` tightening + main.py ~10 type errors
- **Minor follow-ups from WP-CORE-30b code review (deferred):**
  - `core/architect.py:_to_legacy_issue` severity-fallback silent mapping (any
    non-"ERROR" → WARN); contract already constrains to ERROR/WARN Literal so
    exhaustive today, but defensive log on unknowns would catch future drift.
  - `_parse_target_ctx` (architect closure) partially duplicates
    `_issue_stage` (pipeline.py); factor into shared helper if next narrow-
    rerun WP touches both.
  - `track_api_call` test spy Pyright false positive — add `# type: ignore`
    on the monkey-patch line.
  - `render_refinement_prompt` emits `[error]/[warn]` (lowercase from
    `IssueSeverity.value`) rather than `[ERROR]/[WARN]` as documented; cosmetic
    only, no LLM impact.
  - `_specialist_with_feedback` short-result-list risk on missing
    `prev_by_ctx` AND fallback failure — silently drops the context; safe-
    drop sentinel or prev fallback recommended for future hardening.
  - Pre-existing `token_tracker.by_stage` capitalization divergence
    (`"Specialist"`) vs emitter manifest key (`"specialist"`) — pre-WP-30b
    issue, paper-data consumers must read from BOTH structures or join.

**Baseline:** 632 passed, 31 deselected.
**HEAD:** 02d93e7.
**Ahead of origin/main:** 5 commits (NOT pushed).

**Recommendation for next session:**
Per § Next session menu in `handoff-2026-05-23-1933.md`, remaining ranked WPs:
1. **WP-01b** — run-spec orchestrator (paper data infra; user-approved).
   Largest single chunk left; enables N=10 paper runs.
2. **Pyright tightening** — main.py ~10 type fixes + drop CI
   `continue-on-error: true`.
3. **WP-CORE-28 / WP-CORE-32** — Extension UX (TypeScript, manual smoke).
4. **F-8 XXE hardening** — security audit follow-up.

**Operational rules carried forward**:
- Caveman mode ACTIVE (full)
- Communication TR, code EN
- Accuracy > cost (WP-CORE-21 + WP-CORE-34 REJECTED — see
  `feedback-accuracy-over-cost.md`)
- NO `git push` without explicit "push it"
- Codex xhigh skipped for small/mechanical/schema-additive WPs
- TDD strict: RED → GREEN → DOC → COMMIT
- **NEW (iter 36):** Subagent-Driven Development (SDD) applied for WP-CORE-30b
  — implementer → spec reviewer → code quality reviewer → fix loop → re-review
  per `superpowers:subagent-driven-development` skill. Pattern repeatable for
  any cross-stage / Codex-REQUIRE WP.
