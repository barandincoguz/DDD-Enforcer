# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-24 11:50 GMT+3
**Last action:** Iteration 44 SHIPPED (commit `99285e0`) — **Pyright
tightening COMPLETE.** All 9 remaining production-code errors fixed
(config.py, ast_signal_classification.py, ast_signal_discovery.py,
ast_signal_enrichment.py, architect.py, ollama.py, pipeline.py,
rag_pipeline.py, checks_deterministic.py). `tests/` excluded from
pyright scope (~119 noise: MagicMock + Optional fixture).
`continue-on-error: true` DROPPED from backend-ci.yml. Full repo
`pyright` reports **0 errors, 0 warnings**. `pytest -m "not integration"`
still reports **716 passed**, zero regression.

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
**HEAD:** 99285e0.
**Ahead of origin/main:** 27 commits (NOT pushed).

---

## Pyright tightening COMPLETE (iteration 44, 2026-05-24)

Commit `99285e0` closes the Pyright tightening WP. Full repo pyright
reports **0 errors, 0 warnings, 0 informations**.

**Production-code fixes shipped this iteration (9 sites, 9 files):**

| File | Line(s) | Fix |
|------|---------|-----|
| `config.py` | 114, 126 | `SEED = _STAGE_CONFIG.seed if not None else 42` (deterministic fallback) |
| `core/AST/ast_signal_classification.py` | 433, 444 | Param `candidate_type: CandidateType` (was `str`) + import |
| `core/AST/ast_signal_discovery.py` | 117 | Walrus-assign in set comprehension to narrow `str \| None` → `str` |
| `core/AST/ast_signal_enrichment.py` | 118 | `assert match is not None` after the guard at line 114 |
| `core/architect.py` | top + 1002 | `from __future__ import annotations` + `TYPE_CHECKING` block for `Scout/Architect/SpecialistAnalysis` forward refs + `cast(Literal["ERROR","WARN"], sev_str)` with explicit fallback |
| `core/llm/ollama.py` | 166-168 | `cast(Iterable[ChatCompletionMessageParam], messages)` at openai-SDK call boundary |
| `core/orchestration/pipeline.py` | 355-357 | `cast(...)` to bridge `core.verifier.types.VerifierResult` (dataclass) vs `core.pipeline_contracts.VerifierResult` (Pydantic) — comment notes single-type refactor is a future WP |
| `core/rag_pipeline.py` | 152-157 | Helper param `Mapping[str, Any]` (was `Dict`) so ChromaDB's `QueryResult` TypedDict is assignable |
| `core/verifier/checks_deterministic.py` | 108 | REAL BUG: guard `if not isinstance(name, str): continue` before adding None entity names to `seen: Dict[str, str]` |

**Tests/ exclusion (`pyrightconfig.json`):**

Added `extension/backend/tests` to `exclude`. ~119 noise breakdown:

- ~80 attribute injection from `unittest.mock.patch.object`
  (pyright doesn't model `MethodType.return_value` /
  `MethodType.call_count`)
- ~25 Optional fixture access (tests deliberately exercising
  None-paths)
- ~15 intentional Literal violations (negative tests passing
  invalid str into Literal params)

Real test bugs continue to be caught by `pytest -m "not integration"`
(716 passing this iteration, zero regression). Re-enabling pyright on
`tests/` would require comprehensive MagicMock typing work that is
out of EMSE submission scope.

**CI gate change (`.github/workflows/backend-ci.yml`):**

- REMOVED: `continue-on-error: true` from the pyright step.
- Comment now points at `CURRENT.md` for the tests-exclude rationale.
- Production type errors now BLOCK merges.

**Remaining type-related follow-ups (future WPs, not blockers):**

- `core/orchestration/pipeline.py` cast bridge → single-type
  `VerifierResult` refactor (rename one of the two, update all call
  sites + tests).
- `tests/` pyright re-enable after MagicMock typing investigation
  (study pyright-strict + Protocol-based stubs for unittest.mock).
- `core/AST/mutability_index.py` import-resolution false positive
  shows up in IDE pass but not CLI; investigate if it surfaces in
  future contributor environments.

---

## Pyright tightening scope discovery (iteration 43, 2026-05-24)

Handoff §10 Rank 1 estimated "~10 type errors in main.py". Reality:

| Surface | Count | Status |
|---------|-------|--------|
| `main.py` | 7 | ✅ FIXED this iteration (commit `7a5de0e`) |
| Other production code | 9 | ⏸ DEFERRED — blocks CI gate drop |
| Tests | ~116 | ⏸ DEFERRED — mostly MagicMock/Optional fixture noise |

**Production-code error sites still open (9 total):**

- `config.py:114,126` — `Type "int | None" not assignable to "int"` (env var coercion)
- `core/AST/ast_signal_classification.py:444` — `str → CandidateType` Literal cast
- `core/architect.py:1002` — `Unknown | str → Literal["ERROR","WARN"]` severity cast
- `core/llm/ollama.py:168` — `List[Dict[str,str]] → Iterable[ChatCompletionMessageParam]`
- `core/orchestration/pipeline.py:355,357` — verifier callable signature mismatch +
  `VerifierResult vs VerifierResult | None`
- `core/rag_pipeline.py:152` — `QueryResult → Dict[str, Any]`
- `core/verifier/checks_deterministic.py:108` — `Unknown | None → str` key

**Test-side errors (~116) categories:**

- `Import "pytest" could not be resolved` (false positive — pytest now in `.venv` post-iter-43;
  re-run may drop these)
- `Cannot assign to attribute "return_value"/"call_count" for class "MethodType"` —
  pyright doesn't model `unittest.mock.patch.object` mock-attribute injection
- `Object of type "None" is not subscriptable` — test fixtures intentionally raw-test
  None paths
- `str → Literal[...]` arg type — test inputs deliberately violate Literal contracts

**Decision (this iteration, per user Option A):** Ship the surgical `main.py`
fix + `.venv` config alone. Keep CI flag non-blocking. Do NOT drop
`continue-on-error: true` until at least the 9 production-code errors
are fixed (test-side noise can be excluded via pyrightconfig if needed).

**Side artifacts (NOT committed, gitignored):**

- `extension/backend/.venv/` rebuilt with python3.13 (3.12 unavailable
  on this dev machine). requirements.txt + pytest/pytest-cov/httpx
  installed. Resolves CLAUDE.md "broken .venv" follow-up locally.

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
