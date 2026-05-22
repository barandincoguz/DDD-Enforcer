# WP-CORE-6 — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-6-d1-verifier-non-vacuous-design.md` (v2, post-Codex)
**Status:** READY for execution
**Baseline:** 338 passed, 31 deselected at HEAD `9608495`
**Target:** 348 passed, 31 deselected at HEAD `{green-sha}` (+10 tests)

---

## Task breakdown

### Task 1 — RED commit

**Commit type:** `test(architect, verifier, orchestration)`
**Summary:** `WP-CORE-6 red-phase tests for D1 non-vacuous + supporting_sentence_ids E2E propagation + degrade-log enrichment`
**Files touched:**

1. `tests/test_verifier_deterministic.py` — APPEND T-D1-NV-1, T-D1-NV-2
2. `tests/test_architect_identify_contexts.py` — NEW FILE — T-ARCH-1, T-ARCH-2, T-ARCH-2b, T-ARCH-3
3. `tests/test_architect_id_propagation.py` — NEW FILE — T-PROP-1, T-PROP-2, T-INT-1
4. `tests/test_pipeline_orchestration.py` — APPEND T-DEGRADE-LOG-1

**Expected pytest after RED commit**: 348 collected; 340 passed; 8 failed (T-D1-NV-1, T-ARCH-1, T-ARCH-2, T-ARCH-2b, T-ARCH-3, T-PROP-1, T-INT-1, T-DEGRADE-LOG-1).

**Commit message:**
```
test(architect, verifier, orchestration): WP-CORE-6 red-phase tests for D1 non-vacuous + supporting_sentence_ids E2E propagation + degrade-log enrichment

WP-CORE-6 red-phase. 10 new tests across 4 files:
  - tests/test_verifier_deterministic.py: T-D1-NV-1 (empty IDs → ERROR),
    T-D1-NV-2 (non-empty subset passes — regression-lock)
  - tests/test_architect_identify_contexts.py (NEW): T-ARCH-1 (return
    dict shape), T-ARCH-2 (retries on old dict shape), T-ARCH-2b (retries
    on top-level list shape per Codex W-2), T-ARCH-3 (prompt numbering)
  - tests/test_architect_id_propagation.py (NEW): T-PROP-1 (Specialist
    preserves Architect's ContextHypothesis IDs — Codex C-1), T-PROP-2
    (Synthesizer merge regression-lock), T-INT-1 (analyze_document E2E
    IDs survive — Codex C-3)
  - tests/test_pipeline_orchestration.py: T-DEGRADE-LOG-1 (degrade-log
    includes full issues list per Codex C-4)

Expected RED: 8 failing on current main (HEAD 9608495); GREEN commit
in this WP turns them green.

Codex xhigh review: 4 CRITICAL + 4 WARN + 6 NIT + 1 OQ. All CRITICAL+WARN
handled inline in spec v2; 1 OQ deferred with explicit revisit trigger
(post-F-22). 4-iteration zero-deferred streak (CORE-3/4/5b/6) ends here,
by design.

Spec: docs/superpowers/specs/2026-05-21-wp-core-6-d1-verifier-non-vacuous-design.md (v2)
Plan: docs/superpowers/plans/2026-05-21-wp-core-6-d1-verifier-non-vacuous.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

### Task 2 — GREEN commit

**Commit type:** `fix(architect, verifier, orchestration)`
**Summary:** `WP-CORE-6 Architect populates supporting_sentence_ids end-to-end + D1 non-empty clause + degrade-log enrichment`
**Files touched:**

1. `core/architect.py`:
   - Add `_truncate_numbered_pairs(pairs, max_chars, head_ratio=0.6)` helper near `_truncate_with_head_tail` (~line 78).
   - Rewrite `identify_contexts` (line 367-504):
     - Build `numbered_pairs = list(enumerate(domain_sentences))`; truncate via new helper; format as numbered text.
     - Prompt requests `{"contexts": [{"name": ..., "supporting_sentence_ids": [...]}]}` with explicit constraint "every context MUST cite ≥1 sentence index".
     - Parser strict-shape: validate each context is dict with `name: str` + `supporting_sentence_ids: List[int]`; reject anything else → retry → exhaustion → `ArchitectExtractionError`.
     - Remove top-level `elif isinstance(result, list)` branch entirely.
     - Signature change: `-> List[Dict[str, Any]]`.
   - Update `architect_fn` (line 776-783) to thread `supporting_sentence_ids` into `ContextHypothesis`.
   - Update `specialist_fn` (line 785-790) to pass `list(arch.contexts)` instead of `[c.context_name for c in arch.contexts]`.
   - Update `extract_per_context_details` (line 574+): signature `contexts: List[ContextHypothesis]`; loop body uses `ctx.context_name` from input; **delete** line 621-623 (`ctx = ContextHypothesis(context_name=ctx_name, description="")`) — reuse the input ctx so `supporting_sentence_ids` is preserved into `SpecialistAnalysis.context`.

2. `core/verifier/checks_deterministic.py`:
   - Add non-empty clause to `check_d1_supporting_sentence_ids_subset`.
   - Update docstring to reflect the two-clause invariant.

3. `core/orchestration/pipeline.py`:
   - Split `except Exception` (line 65) into `except RefinementExhaustedError as exc:` (with enriched issue-list log) and fallback `except Exception as exc:` (unchanged generic log).

4. `core/pipeline_contracts.py` — NO CHANGE (Pydantic schema's default `[]` is fine; verifier-layer ERROR is the chosen enforcement layer per OQ-4 disposition).

**Gate before commit**: `pytest -m "not integration" --tb=short -q` → 348 passed, 31 deselected.

**Commit message:**
```
fix(architect, verifier, orchestration): WP-CORE-6 Architect populates supporting_sentence_ids end-to-end + D1 non-empty + degrade-log

Four-file change for F-21 (vacuous D1 verifier pass):

  - core/architect.py:
    * identify_contexts: prompt rewrite (numbered sentences + object
      array shape); parser strict-shape (reject top-level list branch +
      dict-shape validation); signature -> List[Dict[str, Any]];
      new line-pair-aware truncation helper _truncate_numbered_pairs
      (Codex W-1)
    * architect_fn: thread supporting_sentence_ids into ContextHypothesis
    * specialist_fn: pass list(arch.contexts) instead of name strings
    * extract_per_context_details: signature accepts List[ContextHypo-
      thesis]; preserves input ctx (Codex C-1 — closes Specialist
      rebuild ID loss)
  - core/verifier/checks_deterministic.py: D1 also flags empty IDs
    as ERROR (honest-signal defense; not enforcement — see D-6 + F-22)
  - core/orchestration/pipeline.py: RefinementExhaustedError degrade
    path logs full exc.issues list (Codex C-4 / A5-risk4 partial)

Pre-WP behavior: ContextHypothesis.supporting_sentence_ids defaulted
to [] for every run; D1 subset check passed vacuously; final
DomainModel.bounded_contexts[].supporting_sentence_ids always empty.
Post-WP behavior: Architect populates IDs from prompt; IDs propagate
Architect → Specialist → Synthesizer → final DomainModel; D1 check
now non-vacuously evaluated; degrade-log enriched with issue details.

Net diff: ~107 LOC production change across 4 files. Baseline 338 →
348 (+10 tests; all WP-CORE-6 tests green; zero regression).

Limitations (deferred to F-22 per Codex W-4):
  - D1 ERRORs still degrade to best-effort via Refiner exhaustion
    (Refiner only re-runs Specialist, not Architect). F-22 tracks the
    Refiner extension to dispatch re-runs by failing stage.

Spec v2: docs/superpowers/specs/2026-05-21-wp-core-6-d1-verifier-non-vacuous-design.md
Plan: docs/superpowers/plans/2026-05-21-wp-core-6-d1-verifier-non-vacuous.md
Codex xhigh review: 4 CRITICAL + 4 WARN + 6 NIT + 1 OQ; all CRITICAL+WARN
handled inline; 1 OQ deferred with explicit revisit trigger.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

### Task 3 — DOC commit (artifacts)

**Commit type:** `chore(artifacts)`
**Summary:** `WP-CORE-6 dev_doc + audit state update + F-22 backlog entry`
**Files touched:**

1. `development_docs/WP-CORE-6-d1-verifier-non-vacuous.md` (NEW). Sections:
   - Header: status / branch / commit SHAs / spec + plan paths / TL;DR.
   - Motivation: F-21 LIVE in production (unlike F-11/F-14 dormant); EMSE methodology consequence (every project run passes D1 vacuously).
   - Architectural decisions:
     1. D-1 — prompt change + numbered sentences.
     2. D-2 — return-shape widening + identify_contexts contract change.
     3. D-2b — Specialist contract change (Codex C-1).
     4. D-3 — D1 non-empty clause as honest signal (Codex C-2 reframe; F-22 tracks enforcement).
     5. D-4 — line-pair-aware truncation (Codex W-1).
     6. D-5b — strict parser, delete top-level list branch (Codex W-2).
     7. D-6 — degrade-log enrichment (Codex C-4).
   - File-level changes: 4-row table.
   - Methodology applied: TDD with genuine RED-fail (8 failing); 4-iteration zero-deferred streak ends with explicit 1-OQ defer (A6-srs-path) carrying revisit trigger.
   - Empirical results: 338 → 348 tests, +10 tests, zero regression.
   - Limitations + follow-ups: F-22 explicit (Refiner-can't-re-run-Architect); A6-srs-path deferred; EMSE Methods section needs honest update.
   - Cross-references: `[[WP-CORE-4-intermediate-save-observability]]`, `[[WP-CORE-5b-synthesizer-empty-model-policy]]`, F-22 backlog.

2. `development_docs/INDEX.md` — APPEND ACTIVE row #7 for WP-CORE-6.
3. `.planning/pipeline_audit/CURRENT.md` — update last-action + next-pointer (iteration 6 = F-22 or pivot).
4. `.planning/pipeline_audit/improvements_backlog.md` — move F-21 to SHIPPED row; ADD F-22 row in OPEN orchestrator table.
5. `.planning/pipeline_audit/findings/architect.md` — append §F-21 SHIPPED status note + reframe.
6. `.planning/pipeline_audit/handoff-2026-05-21-{HHMM}.md` — NEW, iteration-5 → iteration-6 handoff.
7. `.planning/pipeline_audit/decision_log.md` — append `D-CODEX-REVIEW-WP-CORE-6` summary with full disposition table.

**Commit message:**
```
chore(artifacts): WP-CORE-6 dev_doc + audit state update + F-22 backlog entry

  - development_docs/WP-CORE-6-d1-verifier-non-vacuous.md (new)
  - development_docs/INDEX.md ACTIVE row #7
  - .planning/pipeline_audit/CURRENT.md next-pointer to iteration 6
  - .planning/pipeline_audit/improvements_backlog.md
    * F-21 → SHIPPED
    * F-22 (NEW) — Refiner cannot re-run Architect; degrade-mask
      gap discovered during WP-CORE-6 Codex review
  - .planning/pipeline_audit/findings/architect.md §F-21 SHIPPED
  - .planning/pipeline_audit/decision_log.md D-CODEX-REVIEW-WP-CORE-6
  - .planning/pipeline_audit/handoff-2026-05-21-{HHMM}.md (new)

Iteration-6 recommendation: F-22 (Refiner stage-aware re-runs) to
complete the enforcement story for D1 ERRORs. Alternative: pivot to
priority-3 audit walk (synthesizer or verifier deeper close-lookup).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

### Task 4 — PLANNING commit

**Commit type:** `chore(planning)`
**Summary:** `WP-CORE-6 spec v2 + plan into git history`

```
chore(planning): WP-CORE-6 spec v2 + plan into git history

WP-CORE-6 artifacts (Codex-reviewed v2; 1 OQ deferred per design):
  - docs/superpowers/specs/2026-05-21-wp-core-6-d1-verifier-non-vacuous-design.md
  - docs/superpowers/plans/2026-05-21-wp-core-6-d1-verifier-non-vacuous.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Goal-backward verification

| Iteration-5 goal | Evidence (anticipated) |
|---|---|
| Pick F-21 per Codex W-8 priority bump | F-21 chosen as iteration 5 target. |
| Spec → Codex xhigh review → plan → SDD → dev_doc → state update | Spec v1 → review (4 CRITICAL + 4 WARN + 6 NIT + 1 OQ) → v2 → this plan → SDD via RED/GREEN → dev_doc + state in DOC commit. |
| Each commit gated on pytest ≥ baseline (RED accepts known-failing tests) | RED: 340 passed + 8 failed = 348 collected; GREEN: 348 passed. |
| No `git push` | Confirmed; main remains local. |
| Atomic Conventional Commits + Claude trailer | All 4 commits conform per messages above. |
| Zero-deferred standard maintained | **No** — 4-iteration streak ends by design. 1 OQ (A6-srs-path) deferred with scope-bounded rationale + concrete revisit trigger (post-F-22). 4 CRITICAL + 4 WARN handled inline. The deferral is qualitatively different from "future work." |
| Production reachability subsection in spec | YES — §Motivation explicitly notes F-21 is LIVE (unlike F-11/F-14 dormant). Loop discipline lesson from iteration 4 applied. |

---

## Post-iteration handoff outline (to be written at task 3)

**Next iteration target (iteration 6):**

- **Option A (RECOMMENDED): F-22** — Refiner stage-aware re-runs. Completes the enforcement story started by WP-CORE-6's honest signal. M-L effort.
- **Option B: priority-3 audit walk** (synthesizer or verifier deeper close-lookup) — surfaces new findings; broadens audit coverage.
- **Option C: pivot to ingestion-layer MAJOR-OPEN** (F-1, F-2, F-4) — refreshes audit-walk lens after long orchestrator stretch.

**Backlog post-WP-CORE-6:**
- Ingestion: 2 SHIPPED + 4 MAJOR-OPEN + 3 MINOR-OPEN + 1 TRIVIAL-OPEN = 8 OPEN
- Orchestrator: 3 SHIPPED (F-13, F-14, F-21) + 1 MAJOR-OPEN-DORMANT (F-11) + 1 MAJOR-OPEN-NEW (F-22) + 6 MINOR-OPEN + 1 TRIVIAL-OPEN = 9 OPEN
- Total OPEN MAJOR (live): 5 (F-1, F-2, F-4-uncertain, F-22) + 1 DORMANT (F-11)

**Baseline post-iteration:** 348 passed, 31 deselected.
