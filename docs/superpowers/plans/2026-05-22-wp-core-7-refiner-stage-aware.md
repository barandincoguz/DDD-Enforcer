# WP-CORE-7 — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-22-wp-core-7-refiner-stage-aware-design.md` (v2, post-Codex)
**Status:** EXECUTED (RED `aea15e4`, GREEN `ce56d99`, DOC `704e59c`, PLANNING `{this-sha}`)
**Baseline:** 348 passed, 31 deselected at HEAD `4c8580c` (post-WP-CORE-6)
**Target:** 358 passed, 31 deselected (+10 tests, zero regression)

---

## Task breakdown

### Task 1 — RED commit (`aea15e4`)

**Commit type:** `test(orchestration, architect)`
**Summary:** `WP-CORE-7 red-phase tests for stage-aware Refiner + ArchitectGroundingError`

**Files touched:**

1. `tests/test_orchestration_errors.py` — APPEND T-AGE-1 (ArchitectGroundingError carries srs_path, issues, residual_issues, cycles_attempted)
2. `tests/test_pipeline_orchestration.py` — APPEND T-DISPATCH-1..5, T-LOG-1, T-LOG-2; MODIFY T-DEGRADE-LOG-1 (flip from expect-degrade to expect-raise)
3. `tests/test_architect_identify_contexts.py` — APPEND T-FEEDBACK-1 (feedback_issues kwarg + exact prompt format per Codex W-3)
4. `tests/test_analyze_document_e2e.py` — NEW FILE — T-INT-1 (E2E ArchitectGroundingError propagation through analyze_document)

**Codex W-2 mitigation:** all imports of `ArchitectGroundingError` live INSIDE test function bodies (not at module top) so pytest collection succeeds — failures surface at body execution as ImportError/TypeError, counted as test failures not collection errors.

**Expected pytest after RED commit:** 358 collected, 348 passed, 10 failed, 31 deselected.

**Empirical result:** Confirmed at `aea15e4` — 358 collected, 348 passed, 10 failed, 31 deselected. Failure modes:
- ImportError × 6 (T-AGE-1, T-DEGRADE-LOG-1-flip, T-DISPATCH-2, T-DISPATCH-4, T-LOG-1, T-INT-1)
- TypeError × 4 (T-FEEDBACK-1 missing `feedback_issues` kwarg; T-DISPATCH-1/3/5 missing `architect_with_feedback` on PipelineDeps)

### Task 2 — GREEN commit (`ce56d99`)

**Commit type:** `fix(orchestration, architect, refiner)`
**Summary:** `WP-CORE-7 Refiner stage-aware dispatch + ArchitectGroundingError`

**Files touched:**

1. `core/orchestration/errors.py` — ADD `ArchitectGroundingError(PipelineError)` with `srs_path`, `issues`, `residual_issues`, `cycles_attempted` (Codex W-4 widening).
2. `core/orchestration/__init__.py` — RE-EXPORT `ArchitectGroundingError`.
3. `core/orchestration/pipeline.py` — ADD `_issue_stage` helper (Codex C-1 target-prefix derivation, no schema widen); `_format_issue` + `_log_architect_rerun` + `_log_architect_fail` + `_log_specialist_degrade` log helpers preserving WP-CORE-6 C-4 contract; `ArchitectWithFeedbackFn` type + `PipelineDeps.architect_with_feedback` field; restructure `run_pipeline` outer architect-refine loop with pre-check + `initial_result=` threading (Codex C-2); narrow bare `except Exception` to `except RefinementExhaustedError` only (Codex W-5).
4. `core/refiner/loop.py` — ADD optional `initial_result=` kwarg to `refine_until_clean` (skips first verifier call when supplied; resets to None after each iteration).
5. `core/architect.py` — ADD `_build_grounding_feedback_block(feedback_issues)` static helper (Codex W-3 exact format + N-1 once-per-attempt); WIDEN `identify_contexts(feedback_issues=...)` kwarg; ADD `architect_with_feedback_fn` closure in `analyze_document`; ADD `architect_with_feedback=architect_with_feedback_fn` to `PipelineDeps(...)` construction.
6. `tests/test_pipeline_orchestration.py` — UPDATE `_make_typed_deps` fixture with `architect_with_feedback=architect_with_feedback_fn` shim; UPDATE 2 direct `PipelineDeps(...)` construction sites (test_pipeline_invokes_refiner... + test_pipeline_raises_synthesizer_empty_model_error_when_refiner_rerun...).
7. `tests/test_architect_id_propagation.py` — UPDATE WP-CORE-6 happy-path E2E test: `supporting_sentence_ids` from `[0, 1]` to `[0]`. Pre-WP-CORE-7 this test passed via silent degrade on D1 fail; post-WP-CORE-7 D1 raises, so the mock must produce only valid Scout indices (real Scout chunker emits {0} for the two-sentence test input).

**Expected pytest after GREEN commit:** 358 passed, 31 deselected.

**Empirical result:** Confirmed at `ce56d99` — 358 passed, 31 deselected, 0 failed.

### Task 3 — DOC commit (`704e59c`)

**Commit type:** `chore(artifacts)`
**Summary:** `WP-CORE-7 dev_doc + audit state update + F-22 SHIPPED + F-23/24 NEW`

**Files touched:**
- `development_docs/WP-CORE-7-refiner-stage-aware.md` (created)
- `development_docs/INDEX.md` (ACTIVE row #8 added)
- `.planning/pipeline_audit/CURRENT.md` (iteration 6 SHIPPED status)
- `.planning/pipeline_audit/improvements_backlog.md` (F-22 → SHIPPED; F-23 + F-24 NEW; F-15 downgraded)
- `.planning/pipeline_audit/decision_log.md` (D-PICK-WP-CORE-7 + D-CODEX-REVIEW-WP-CORE-7)
- `.planning/pipeline_audit/findings/architect.md` (§F-22 SHIPPED status added)
- `.planning/pipeline_audit/handoff-2026-05-23-1245.md` (iteration 7 handoff)

### Task 4 — PLANNING commit (this one)

**Commit type:** `chore(planning)`
**Summary:** `WP-CORE-7 spec v2 + plan into git history`

**Files touched:**
- `docs/superpowers/specs/2026-05-22-wp-core-7-refiner-stage-aware-design.md` (v2)
- `docs/superpowers/plans/2026-05-22-wp-core-7-refiner-stage-aware.md` (this file)

---

## Dependencies + sequencing

```
Task 1 (RED) ──┐
               ├─→ Task 2 (GREEN) ─→ Task 3 (DOC) ─→ Task 4 (PLANNING)
[baseline 348] │   [baseline 358]   [baseline 358]   [baseline 358]
               │
[10 fail by    │   [0 fail; all
 ImportError/  │    10 RED tests
 TypeError on  │    flip to pass +
 ArchitectGround │  T-DEGRADE-LOG-1
 ingError or   │    flip retained]
 PipelineDeps  │
 missing kwarg]│
```

Each commit is atomic + verifiable by checkout + pytest. No `--no-verify`.

---

## Goal-backward verification

| Spec goal | Plan task | Verification at commit time |
|---|---|---|
| F-22 LIVE production bug fixed (architect-stage D1 ERRORs no longer degrade silently) | Task 2 (GREEN) | T-DISPATCH-2 + T-INT-1 + T-DEGRADE-LOG-1-flip all assert `ArchitectGroundingError` raises on persistent failure |
| Mode C hybrid (1 architect re-run with feedback + hard-fail) | Task 2 (GREEN) | T-DISPATCH-1 asserts `architect_with_feedback` invoked exactly once on resolvable architect issue; T-DISPATCH-2 asserts hard-fail after exhaustion |
| Specialist refine loop unchanged (regression contract) | Task 2 (GREEN) | T-DISPATCH-3 + T-LOG-2 + existing WP-CORE-6 tests all pass post-GREEN |
| Stage routing without VerifierIssue schema widen | Task 2 (GREEN), Codex C-1 disposition | `_issue_stage` derives from `target` prefix; no callsite migration needed; `pipeline_contracts.VerifierIssue` unchanged |
| Pre-check verifier ONCE on common path | Task 2 (GREEN), Codex C-2 disposition | `refine_until_clean(initial_result=...)` threading; verified by T-DISPATCH-3 (specialist refine works with pre-check threading) |
| `ArchitectGroundingError` carries residual non-architect issues | Task 2 (GREEN), Codex W-4 disposition | T-DISPATCH-4 asserts `len(exc.issues) == 1 (architect)` and `len(exc.residual_issues) == 1 (specialist)` for mixed-stage scenario |
| Exact feedback prompt format | Task 2 (GREEN), Codex W-3 disposition | T-FEEDBACK-1 asserts `"PREVIOUS ATTEMPT FAILED VERIFICATION:"` + `issue.target` + `issue.message` substrings + ordering (feedback before main instructions) |
| Per-attempt feedback (not per-internal-retry) | Task 2 (GREEN), Codex N-1 disposition | `_build_grounding_feedback_block` called once before `for retry in range(5)`; reused unchanged across internal retries (no per-retry re-derivation) |
| Narrowed exception handler | Task 2 (GREEN), Codex W-5 disposition | `except RefinementExhaustedError` only at `pipeline.py`; unexpected exceptions propagate per AGENTS.md explicit-failure |
| Run-manifest signal preserved | Task 2 (GREEN), WP-CORE-6 C-4 contract | T-LOG-1 (hard-fail path) + T-LOG-2 (degrade path) both assert issue list visible in stdout |
| F-23 + F-24 backlog entries | Task 3 (DOC) | `improvements_backlog.md` shows F-23 (typed PipelineError handler, MAJOR) and F-24 (srs_path in VerifierIssue, MINOR) as new OPEN entries |
| EMSE Methods claim updated to enforcement | Task 3 (DOC), `development_docs/WP-CORE-7-*` cross-references | Dev doc §Limitations references advisor-flag for paper revision |

---

## Risks materialized + mitigated during execution

| spec §Risks ID | mitigation outcome |
|---|---|
| R-1 (control-flow refactor risk) | Codex C-2 fix forced restructure to pre-check + `initial_result=` threading; broke 4 existing tests during initial GREEN attempt (double-verify on common path); resolved by extending `refine_until_clean` API. |
| R-2 (architect re-run requires specialist re-run with new arch) | Naturally handled by outer while-loop structure; T-DISPATCH-5 verifies specialist sees the new `arch`. |
| R-3 (PipelineDeps fixture migration) | 3 fixture sites updated in lockstep with GREEN. `_make_typed_deps` shim uses `architect_with_feedback=lambda scout, issues: architect_fn(scout)` for tests that don't exercise the feedback path. |
| R-7 (narrowed exception surface) | No regressions observed — all 348 existing tests pass post-narrowing. Hypothetical latent bugs in `deps.verifier` would now surface; none observed in this WP. |

---

## Post-execution status

- Pytest: 348 → 358 (+10 tests, zero regression).
- Commits: 4 (RED + GREEN + DOC + PLANNING), all atomic with Claude trailer.
- No `git push`. Loop discipline maintained.
- F-22 status: OPEN → SHIPPED.
- New OPEN backlog: F-23 (MAJOR), F-24 (MINOR). F-15 downgraded to TRIVIAL.

Iteration 6 closed clean. Iteration 7 handoff at `.planning/pipeline_audit/handoff-2026-05-23-1245.md`.
