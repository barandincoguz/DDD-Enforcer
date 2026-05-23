# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 21:35 GMT+3
**Last action:** Iterations 33-37 SHIPPED — F-16 + WP-CORE-20c +
ChunkMetadata.truncated_chunks fix + WP-CORE-30b (SDD) +
ownership-deprecation + WP-01b Task A (PaperRunManifest schema +
writer + provenance hashes via SDD).

**Session totals (this autonomous block, post-/cont):**
- 5 WPs shipped (F-16, WP-CORE-20c, ChunkMetadata fix, WP-CORE-30b, WP-01b Task A)
- 10 commits (atomic)
- 611 → 649 (+38 tests net, zero regression)
- 0 MAJOR-OPEN-live findings

**Cumulative across the multi-day run:** 348 → 649 (+301 tests, ~70 commits).

**Remaining backlog (engineering, paper-free):**
- F-8 (MINOR-OPEN — XXE hardening; needs threat model)
- WP-CORE-28 — Extension UX wave 1 (TypeScript, manual smoke risk)
- WP-CORE-32 — Extension webviews
- **WP-01b remaining tasks (5 of 6):**
  - Task B — `core/metrics.py` (precision/recall/F1 per violation type)
  - Task C — `scripts/aggregate.py` (N-runs → mean ± std + IQR + bootstrap CI)
  - Task D — `scripts/build_tables.py` (per-RQ LaTeX renderer)
  - Task E — Makefile target + `paper.tex` `\input{tables/rqN.tex}`
  - Task F — E2E smoke + `core/intermediate/*.json` → `legacy_pre_emse/`
- Pyright `continue-on-error` tightening + main.py ~10 type errors
- **Minor follow-ups from WP-CORE-30b code review (deferred):**
  - `_to_legacy_issue` severity-fallback silent mapping (exhaustive today
    per contract Literal, but defensive log helpful)
  - `_parse_target_ctx` partial duplication with `_issue_stage`
  - `track_api_call` test spy Pyright false positive (`# type: ignore`)
  - `render_refinement_prompt` lowercase severity label
  - `_specialist_with_feedback` short-result-list risk on missing prev
    + fallback failure
  - Pre-existing `token_tracker.by_stage` "Specialist" vs emitter
    "specialist" capitalization divergence

**Baseline:** 649 passed, 31 deselected.
**HEAD:** e18ee53.
**Ahead of origin/main:** 10 commits (NOT pushed).

**Recommendation for next session:**
Per the SDD decomposition agreed on this session, the WP-01b chain
continues in order:
1. **WP-01b Task B** — `core/metrics.py` precision/recall/F1 per
   violation type. Consumes `PaperRunManifest` + judge-verdict JSON.
   ~1-2h. Pure deterministic, low-risk; TDD strict.
2. **WP-01b Task C** — aggregator. ~2h. New `scripts/aggregate.py`.
3. **WP-01b Task D** — LaTeX renderer. ~2-3h. Sensitive paper.tex
   touch in Task E.

Other ranked options:
- **Pyright tightening** — main.py ~10 type fixes + CI gate
  `continue-on-error: false`.
- **WP-CORE-28 / WP-CORE-32** — Extension UX (TypeScript, manual smoke).
- **F-8 XXE hardening** — security audit follow-up.

**Operational rules carried forward**:
- Caveman mode ACTIVE (full)
- Communication TR, code EN
- Accuracy > cost (WP-CORE-21 + WP-CORE-34 REJECTED — see
  `feedback-accuracy-over-cost.md`)
- **Ownership disestablished (2026-05-23)** — any agent picks up any WP;
  `WP_DAGILIM_BARAN_ALI.md` historical only; risk matrix + sync points
  still active. See `feedback-ownership-disestablished.md`.
- NO `git push` without explicit "push it"
- Codex xhigh skipped for small/mechanical/schema-additive WPs
- TDD strict: RED → GREEN → DOC → COMMIT
- Subagent-Driven Development (SDD) for cross-stage / multi-file /
  Codex-REQUIRE WPs — implementer → spec reviewer → code quality
  reviewer → fix loop → re-review per
  `superpowers:subagent-driven-development`.

**SDD telemetry this session:**
- WP-CORE-30b: 2 implementer + 2 spec reviewer + 2 code quality
  reviewer + 1 fix + 1 re-review + 1 final integration = 9 dispatches.
- WP-01b Task A: 1 implementer + 1 spec reviewer + 1 fix (colon) +
  1 spec re-review + 1 code quality + 1 fix (5 findings) +
  1 code quality re-review = 7 dispatches.
- Total: 16 dispatches across 2 WPs.
- Outcome: 0 regression, 12+15 = 27 added tests across these two WPs.
