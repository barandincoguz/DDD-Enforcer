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
- WP-01d (P1/P2/P3 pipeline classes) — **CANCELLED 2026-05-24** per
  user direction. Different pipeline architectures will NOT be
  explored for this paper. `PaperRunManifest.pipeline` Literal field
  stays on the schema (no migration cost) but only the current
  `DomainArchitect` chain will produce RQ1 data.

**Remaining backlog (engineering, paper-free):**
- F-8 (MINOR-OPEN — XXE hardening; needs threat model)
- WP-01d — P1/P2/P3 pipeline classes (user-deferred)
- WP-CORE-28 — Extension UX wave 1 (TypeScript, manual smoke risk)
- WP-CORE-32 — Extension webviews
- Pyright `continue-on-error` tightening + main.py ~10 type errors
- paper.tex integration of rqN.tex \input{} blocks (human-coordinator
  task — see LaTeX_DL_468198_240419/tables/README.md for the
  candidate line numbers and TODO list).
- ~~**Minor follow-ups deferred from WP-CORE-30b code review**~~ —
  ALL CLOSED in iter 46 (see iter-46 commit list above).
- ~~**Minor follow-ups deferred from WP-01b/01c code review**~~ —
  ALL CLOSED in iter 46 (see iter-46 commit list above).

**Baseline:** 729 passed, 31 deselected.
**HEAD:** `ee9d57b` (iter 48 D-fix-2 — race close).
**Ahead of origin/main:** 53 commits (NOT pushed).

---

## Iteration 48 — WP-CORE-28 Feature 2 (Backend lifecycle resilience) COMPLETE (2026-05-24)

WP-CORE-28 Iter 48 shipped via Subagent-Driven Development (SDD).
Plan at `.planning/plans/2026-05-24-wp-core-28-iter-48-backend-lifecycle.md`.
Spec at `todos/WP-CORE-28-extension-ux-wave1.md`.

**6 commits (4 feat + 2 fix), 22 new test cases, zero baseline regression.**

| Commit | Iter step | Content | Δ tests |
|--------|-----------|---------|---------|
| `16c7bde` | A | `computeBackoffMs(attempt, baseMs?, maxMs?)` exponential backoff (1s,2s,4s,8s,16s; clamp 30s) + 8 tests | +8 |
| `a6346d5` | B | `shouldAttemptRestart(attempt, maxAttempts?)` cap-5 boolean + 4 tests | +4 |
| `47876fe` | C | `formatExitReason(code, signal)` + `classifyExitForRestart(code, signal, intentional)` + `ExitDisposition` type + 10 tests | +10 |
| `b5fa9e2` | D | `intentionalStopTarget` + `backendRestartAttempts` globals + `child.on('exit')` rewrite + `stopBackend` flag + `restartBackend` reason prompt + `handleUnexpectedExit` 3-button warning + `attemptAutoRestart` loop + `findAvailablePort` log lines | 0 |
| `385de50` | D follow-up 1 | Quality-review fix 1: replace `backendIntentionalStop: boolean` with `intentionalStopTarget: ChildProcess \| null` per-process pattern + `.catch()` on fire-and-forget `handleUnexpectedExit` | 0 |
| `ee9d57b` | D follow-up 2 | Quality-review fix 2: drop defensive `intentionalStopTarget = null;` reset in `startBackend` which reintroduced the original race the per-process refactor was meant to close | 0 |

**SDD telemetry (this iter):**

| Task | Implementer | Spec review | Quality review | Fix loops | Dispatches |
|------|-------------|-------------|----------------|-----------|-----------|
| 1 | 1 | 1 | 1 | 0 | 3 |
| 2 | 1 | 1 | 1 | 0 | 3 |
| 3 | 1 | 1 | 1 | 0 | 3 |
| 4 | 1 | 1 | 1 | 2 (per-process refactor + reset removal; each followed by re-review) | 7 |
| **Total** | **4** | **4** | **4** | **2** | **16 subagent dispatches** |

Two quality-review iterations on Task 4 were warranted. The first
review flagged a real race (boolean flag global vs slow-exiting
ChildProcess); the implementer's first fix (`385de50`) correctly
introduced a per-process `intentionalStopTarget: ChildProcess | null`
reference + `.catch()` on the fire-and-forget call, but ALSO kept
a defensive `= null` reset in `startBackend` that defeated the
per-process correctness. The second review traced the scenario and
showed the race was still open. Fix 2 (`ee9d57b`) deleted the
reset and re-established correctness. The state machine is now:
module init `null` → `stopBackend` sets target = the live
ChildProcess being killed → that process's matching exit handler
clears target = null on `target === thisProcess` consumption →
next `stopBackend` overwrites if no match consumed.

**Locked invariants added this iter:**

| ID | Invariant |
|----|-----------|
| Iter 48 A-C | `extension/src/extension.ts` exports four pure backend-lifecycle helpers (`computeBackoffMs`, `shouldAttemptRestart`, `formatExitReason`, `classifyExitForRestart`) plus the `ExitDisposition` type, in a `// BACKEND LIFECYCLE (pure helpers — testable without vscode)` section placed between TYPES and GLOBAL STATE. |
| Iter 48 D | Module-level `intentionalStopTarget: ChildProcess \| null` is the per-process intent marker. **DO NOT add any code path that nulls this variable except** (a) the matching `child.on('exit')` handler when `target === thisProcess`, or (b) a subsequent `stopBackend` that overwrites with a fresh ChildProcess. Adding a defensive reset (e.g., in `startBackend`) reintroduces the original race — see commit `ee9d57b` body for the trace. |
| Iter 48 D | The `child.on('exit')` handler MUST capture `const thisProcess = backendProcess;` immediately before `backendProcess.on('exit', ...)` so the per-process comparison at exit time uses the specific ChildProcess that handler was registered for, not whatever `backendProcess` references at the time the event fires. |
| Iter 48 D | `handleUnexpectedExit` is called as a fire-and-forget Promise from the synchronous `child.on('exit')` callback. It MUST be invoked with `.catch((err) => log(...))` so a VS Code API failure becomes observable in the Output channel rather than bubbling up as an unhandledRejection. |
| Iter 48 D | `attemptAutoRestart` is bounded by `shouldAttemptRestart` (default 5 attempts). On exhaustion it surfaces a persistent error toast and sets status to `"error"`. The loop does NOT recurse, does NOT schedule additional attempts, and does NOT bypass the cap. |

**F5 smoke status:** Programmatic gates green (compile, lint, pyright,
pytest 729). F5 GUI smoke pending user decision; this iter's surface
touches the load-bearing backend lifecycle (kill -9, crash dialog,
auto-restart loop, port-collision logging) so a real F5 session is
recommended before Iter 49 starts. The smoke checklist will be posted
to the user when this CURRENT.md update is committed.

**Baseline at iter close:** 729 passed, 31 deselected, pyright 0/0/0.

---

## Iteration 47 — WP-CORE-28 Feature 1 (API key onboarding) COMPLETE (2026-05-24)

WP-CORE-28 Iter 47 shipped via Subagent-Driven Development (SDD).
Plan at `.planning/plans/2026-05-24-wp-core-28-iter-47-api-key-onboarding.md`.
Spec at `todos/WP-CORE-28-extension-ux-wave1.md`.

**5 commits (4 feat + 1 fix), 18 new test cases, zero baseline regression.**

| Commit | Iter step | Content | Δ tests |
|--------|-----------|---------|---------|
| `72cc92a` | A | `classifyApiKeyError(err): ApiKeyErrorKind` pure fn + 8 mocha tests | +8 |
| `21b667f` | B | `validateGeminiKey(key, httpProbe?)` Gemini `/v1beta/models` probe with injectable axios + 4 tests | +4 |
| `45ed96f` | C | `decideMigrationOffer(source, declined): MigrationDecision` + `ApiKeySource` + 5 tests | +5 |
| `e2e57cd` | D | `getApiKey` rewrite (source tracking → pre-validate → migration toast 3-button → `globalState` decline) + new `"validatingApiKey"` status bar state + 1 integration test | +1 |
| `6388075` | D follow-up | Quality-review fix: wrap `cfg.update` in try/catch; warning toast if settings-clear fails after `secrets.store` succeeds | 0 |

**SDD telemetry (this iter):**

| Task | Implementer | Spec review | Quality review | Fix loops | Dispatches |
|------|-------------|-------------|----------------|-----------|-----------|
| A | 1 | 1 | 1 | 0 | 3 |
| B | 1 | 1 | 1 | 0 | 3 |
| C | 1 | 1 | 1 | 0 | 3 |
| D | 1 | 1 | 1 | 1 (fix + re-review) | 5 |
| **Total** | **4** | **4** | **4** | **1** | **14 subagent dispatches** |

Quality reviewer caught one legitimate issue on D: the migration accept
path called `secrets.store()` then `cfg.update()` without a try/catch.
If `cfg.update` threw, the user saw a success toast but the settings
entry persisted; the next `getApiKey` call's discovery-order (settings
first) would silently re-read the stale entry, bypassing the freshly
saved secret-storage copy. Fix commit `6388075` wraps `cfg.update` and
surfaces a Warning toast directing the user to remove the setting
manually when the clear fails. Success toast only fires when both
operations succeed.

**Locked invariants added this iter:**

| ID | Invariant |
|----|-----------|
| Iter 47 A-C | `extension/src/extension.ts` exports three pure helpers (`classifyApiKeyError`, `validateGeminiKey`, `decideMigrationOffer`) plus the types `ApiKeyErrorKind`, `ApiKeyValidationResult`, `ApiKeyHttpProbe`, `ApiKeySource`, `MigrationDecision`. Tests in `extension/src/test/extension.test.ts` exercise them with an injectable-HTTP stub (no sinon) — precedent established for future Iter-48-onward pure helpers. |
| Iter 47 D | `getApiKey` MUST pre-validate the key against `https://generativelanguage.googleapis.com/v1beta/models` (direct, no backend round-trip) before returning. On rejection, the function returns `undefined` so the existing `startBackend` error path takes over. |
| Iter 47 D | The migration accept path is non-atomic by design (secret-storage write succeeds, settings clear may fail). The fix at `6388075` makes the failure observable to the user via a Warning toast; the spec accepted that auto-rollback (deleting the just-written secret) is not appropriate UX. |
| Iter 47 D | `apiKeyMigrationDeclined` lives in `context.globalState` (per-extension, persists across VS Code restarts). No undo command exists; user must clear via `~/Library/Application Support/Code/User/globalStorage/ddd-enforcer.ddd-enforcer/` (macOS) or a clean VS Code profile. |

**F5 smoke status:** Programmatic gates green (compile, lint, pyright,
pytest 729). User explicitly waived the F5 GUI gate ("kontrol et ve
devam et", 2026-05-24) — physical F5 smoke deferred to user discretion.
Iter 48 proceeds without blocking on a manual smoke session.

**Baseline at iter close:** 729 passed, 31 deselected, pyright 0/0/0.

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
