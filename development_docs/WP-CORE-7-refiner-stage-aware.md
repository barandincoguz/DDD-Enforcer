# WP-CORE-7 — Refiner stage-aware re-runs + ArchitectGroundingError

**Status:** SHIPPED 2026-05-23
**Branch / commits:**
- RED `aea15e4` — test(orchestration, architect): WP-CORE-7 red-phase tests
- GREEN `ce56d99` — fix(orchestration, architect, refiner): WP-CORE-7 Refiner stage-aware dispatch + ArchitectGroundingError
- DOC `{this commit}` — chore(artifacts): WP-CORE-7 dev_doc + audit state update + F-22 SHIPPED + F-23/24 NEW
- PLANNING `{pending}` — chore(planning): WP-CORE-7 spec v2 + plan into git history

**Spec:** `docs/superpowers/specs/2026-05-22-wp-core-7-refiner-stage-aware-design.md` (v2; revised after Codex xhigh review)
**Plan:** `docs/superpowers/plans/2026-05-22-wp-core-7-refiner-stage-aware.md` (to be added in PLANNING commit)
**Parent finding:** `.planning/pipeline_audit/improvements_backlog.md` finding **F-22** (MAJOR, NEW from WP-CORE-6 Codex W-4 / A6-f22) — now SHIPPED.

## TL;DR

F-22 closes the half-fix left by WP-CORE-6. After WP-CORE-6, D1 verifier flagged `ungrounded_context` ERRORs on contexts with empty or invalid `supporting_sentence_ids`, but Refiner only knew how to re-run **Specialist**. Architect-stage ERRORs propagated to `RefinementExhaustedError` → caught + logged + best-effort degrade → model shipped. EMSE methodology claim "D1 enforces grounding" was honored at verifier level but not pipeline level. WP-CORE-7 promotes the signal to enforcement: Refiner becomes stage-aware via a stage-derivation helper (`_issue_stage` from target-prefix) and runs **mode C hybrid** — one architect re-run with issue-aware feedback prompt, then hard-fail via new `ArchitectGroundingError` exception carrying srs_path + issues + residual_issues + cycles_attempted.

Baseline: 348 → 358 passing (+10 tests, zero regression).

## Motivation

WP-CORE-6 §D-3 explicitly framed the D1 ERROR as **"honest signal, not enforcement."** The reason: closing enforcement required redesigning Refiner's control loop, which only re-ran Specialist (`pipeline.py:70-72` `_re_run_specialist`). Architect-stage ERRORs that surfaced after WP-CORE-6's non-empty clause (`checks_deterministic.py:24,38`) could never be auto-corrected — re-running Specialist with the same Architect output reproduces the same Architect-stage ERROR. After max_cycles=2 Refiner raised `RefinementExhaustedError`, which `pipeline.py:82-97` caught and logged (post-WP-CORE-6 C-4 with full issues list), then continued with `refined_specialist = specialist_output`. The pipeline shipped the model anyway.

Production reachability check (loop discipline): F-22 is LIVE. Every project run with a non-trivial SRS is a candidate for triggering this path. The WP-CORE-6 honest-signal degrade-log empirically confirms the path is hit on real runs.

## Architectural decisions

### D-1 — Mode C hybrid (1 feedback rerun + hard-fail) over mode A (hard-fail only) or mode B (loop with feedback)

Three modes were considered:
- **A** — hard-fail only on architect-stage ERRORs (smallest change, but zero self-correction shot).
- **B** — issue-aware re-prompt loop up to `max_cycles` (closed-loop control, but LLM-determinism risk + cost amplification + complexity).
- **C** — hybrid: 1 architect re-run with feedback prompt → if still failing, hard-fail (bounded cost ~10 LLM calls worst case; one genuine self-correction shot; explicit failure on persistent grounding violation).

Mode C chosen for AGENTS.md "smallest correct change" + "explicit failure on persistent violation" + matches EMSE methodology integrity goal.

### D-2 — Derive stage from `target` prefix instead of widening `VerifierIssue` schema (Codex C-1)

The contract `core/pipeline_contracts.VerifierIssue` (Pydantic) has no `stage` field. The legacy `core/verifier/types.VerifierIssue` (dataclass) has it but is dropped by the `_to_contract_issue` adapter (`architect.py:835-846`). By the time `RefinementExhaustedError.issues` reaches `run_pipeline`, the stage attr is gone.

**Naive fix** (schema widen): add `stage: Literal[...]` field to contract `VerifierIssue`. Migration cost: 13 call sites (5 in `checks_semantic_d6_d7_d8.py`, 2 in `checks_semantic.py`, 4 in tests, 1 in `_to_contract_issue`).

**Chosen fix** (target-prefix derivation): every Verifier check populates `target` (or `location`, in legacy) with a `{stage}:` prefix already (`checks_deterministic.py:24,38,67,91,118,145`). A small `_issue_stage(issue) -> Optional[str]` helper in `pipeline.py` derives stage from this prefix, falling back to the legacy `.stage` attribute when present. Zero schema change. The trade-off: stage routing depends on prefix invariant — documented as a non-negotiable in handoff.

### D-3 — Pre-check verifier ONCE before specialist refine loop; thread result back via `initial_result=`

Codex C-2 flagged that calling `refine_until_clean(_re_run_specialist)` first then partitioning on `RefinementExhaustedError` wastes 2 specialist cycles on architect-stage-only failures. Solution:
1. `initial_result = deps.verifier(snapshot)` once.
2. If architect-stage issues present → dispatch to architect feedback rerun (skip specialist refine entirely).
3. Else thread `initial_result` into `refine_until_clean(initial_result=initial_result)` so the refiner skips its own first verifier call — common path stays at 1 verifier call per cycle.

`refine_until_clean` gains the optional `initial_result` kwarg (`core/refiner/loop.py:14-46`) that, when supplied, is consumed before the first `verifier(output)` call and reset to `None` after each iteration (so subsequent cycles re-verify normally).

### D-4 — Hard-fail via new `ArchitectGroundingError` taxonomically distinct from `ArchitectExtractionError`

`ArchitectExtractionError` already exists (`errors.py:23-26`) and is raised by `identify_contexts` on JSON-parse-retry exhaustion or empty-contexts. That's a **syntactic** failure mode (LLM didn't emit parseable structure).

`ArchitectGroundingError` is **semantic** (LLM emitted well-formed contexts whose `supporting_sentence_ids` fail D1). Combining them would obscure the run-manifest signal and prevent typed downstream filtering. Both inherit from `PipelineError`.

Payload: `srs_path`, `issues` (architect-stage subset at exhaustion), `residual_issues` (non-architect issues observed alongside, preserved for post-mortem visibility per Codex W-4), `cycles_attempted` (always 1 in mode C).

### D-5 — Feedback block format: `"PREVIOUS ATTEMPT FAILED VERIFICATION:"` + per-issue lines, prepended once per outer attempt (Codex W-3, N-1)

`_build_grounding_feedback_block(feedback_issues)` static helper in `DomainArchitect`. Produces:
```
PREVIOUS ATTEMPT FAILED VERIFICATION:
The previous response was rejected because of the following grounding issues:
- {issue.target}: {issue.message}
- ...

For this retry, ensure every context cites valid supporting_sentence_id
values that appear in the numbered list below.
```

Prepended once to `identify_contexts`'s main prompt **before** the `for retry in range(5)` internal loop. Reused across all 5 internal JSON-parse retries (no per-retry re-derivation).

### D-6 — Narrow `except Exception` in `pipeline.py` to `except RefinementExhaustedError` (Codex W-5)

The pre-WP-CORE-7 `pipeline.py:98-112` had:
```python
except Exception as exc:
    print(f"  ⚠️  refiner exhausted retries ({type(exc).__name__}); ...")
    refined_specialist = specialist_output
```

This silently degraded on ANY exception (including bugs in `deps.verifier` or test fixtures). Codex W-5 flagged this as contradicting WP-CORE-7's explicit-failure mandate. The block is removed; unexpected exceptions now propagate.

### D-7 — `PipelineDeps` widened with `architect_with_feedback` callable

`PipelineDeps` gains the field; `DomainArchitect.analyze_document` builds an `architect_with_feedback_fn` closure that mirrors `architect_fn` but threads `feedback_issues` into `identify_contexts`. Three fixture sites in `tests/test_pipeline_orchestration.py` updated in lockstep (the in-fixture closures supply a feedback-ignoring delegate to `architect_fn` for tests that don't exercise the feedback path).

## File-level changes

| File | Change | LOC delta |
|---|---|---|
| `core/orchestration/errors.py` | + `ArchitectGroundingError(PipelineError)` with srs_path / issues / residual_issues / cycles_attempted | +40 |
| `core/orchestration/__init__.py` | Re-export `ArchitectGroundingError` | +2 |
| `core/orchestration/pipeline.py` | + `_issue_stage` helper, `_format_issue`, log helpers, restructured `run_pipeline` outer architect-refine loop with pre-check + `initial_result` threading + narrowed `except RefinementExhaustedError`; `PipelineDeps.architect_with_feedback` field | +120 / -45 |
| `core/refiner/loop.py` | + optional `initial_result=` kwarg to `refine_until_clean` | +15 / -3 |
| `core/architect.py` | + `_build_grounding_feedback_block` static helper, `identify_contexts(feedback_issues=...)` kwarg + feedback prepend, `architect_with_feedback_fn` closure, `PipelineDeps(architect_with_feedback=...)` in construction | +60 / -2 |
| `tests/test_pipeline_orchestration.py` (RED + GREEN) | + 10 new tests (T-DISPATCH-1..5, T-LOG-1, T-LOG-2 + T-DEGRADE-LOG-1 modified) + `_make_typed_deps` widened + 2 direct PipelineDeps fixture sites updated | +420 / -25 |
| `tests/test_orchestration_errors.py` (RED) | + T-AGE-1 | +35 |
| `tests/test_architect_identify_contexts.py` (RED) | + T-FEEDBACK-1 | +55 |
| `tests/test_analyze_document_e2e.py` (RED, NEW) | + T-INT-1 (architect grounding error E2E) | +120 |
| `tests/test_architect_id_propagation.py` (GREEN) | WP-CORE-6 happy-path E2E supporting_sentence_ids changed from [0, 1] to [0] (Scout emits only index 0 for this two-sentence test input; pre-WP-CORE-7 D1 silently degraded, post-WP-CORE-7 raises) | +6 / -3 |

## Methodology applied

- **TDD with genuine RED → GREEN.** RED commit `aea15e4` accepted 10 known-failing tests (verified failure-by-assertion or runtime, no collection errors per Codex W-2 import-guard). GREEN commit `ce56d99` flipped all 10 green + preserved baseline 348 specs.
- **Spec → Codex xhigh review → spec v2 → atomic commits.** Codex returned 2 CRITICAL + 6 WARN + 2 NIT + 1 OQ; all CRITICAL+WARN handled inline; 2 NIT inlined; 1 OQ (A6-srs-path follow-up) recorded as F-24 backlog post-F-22 trigger fired.
- **Production reachability subsection mandatory in spec §Motivation.** F-22 confirmed LIVE.

## Empirical results

- **Test baseline**: 348 (pre-RED) → 358 passing (GREEN; +10 new tests, zero regression).
- **LOC delta vs WP-CORE-6 HEAD `4c8580c`**: +335 / -75 (7 files in GREEN; 4 files in RED).
- **Failure surface for D1 ERROR runs**: pre-WP-CORE-7 = silent degrade with model shipped; post-WP-CORE-7 = `ArchitectGroundingError` raised after 1 issue-aware re-prompt fails. EMSE methodology claim "D1 enforces grounding" is now true at both verifier and pipeline level.

## Limitations + follow-ups

- **F-23 (NEW, backlog)**: `main.py` exception handlers are bare `except Exception` (lines 77, 180, 194, 211, 226, 410, 427, 518, 533, 721); no typed `PipelineError` catch. New `ArchitectGroundingError` is caught generically and serialized via `str(e)`. Run-manifest still surfaces failure but loses typed taxonomy at the response level. Out of WP-CORE-7 scope. Severity MAJOR (run-manifest signal completeness).
- **F-24 (NEW, backlog)**: A6-srs-path follow-up — WP-CORE-6 deferred OQ now unlocked post-F-22. Threading `srs_path` into `VerifierIssue` requires schema widening + 13-callsite migration. Out of WP-CORE-7 scope. Severity MINOR (observability completeness).
- **F-15 (downgraded)**: Refiner exhaustion fallback observability partially closed for architect path (hard-fail now). Specialist path retains degrade-log per WP-CORE-6 C-4 contract. Downgrade severity TRIVIAL.
- **Architect feedback prompt format**: current format includes only `target` + `message` per issue. Codex OQ-5 considered whether to include the LLM's previous **response** for concrete reference; deferred per AGENTS.md "no speculative generalization" (prompt token cost amplification). Revisit if empirical single-shot-fix rate is low.
- **`architect_max_cycles=1` hard-coded**: Codex OQ-2 considered configurability; deferred. Add as F-25 backlog only if empirics show feedback rarely succeeds.

## Cross-references

- **Predecessor**: `[[WP-CORE-6-d1-verifier-non-vacuous]]` — D-3 fold-in created the honest-signal-but-not-enforcement gap that WP-CORE-7 closes. Sibling Codex W-4 / A6-f22 spawned F-22.
- **Invariant chain**:
  - WP-CORE-4: any future stage retry wrapper using `_save_intermediate` MUST include `except IntermediateSaveError: raise`.
  - WP-CORE-5b: any future pipeline-orchestration code constructing `SynthesizerEmptyModelError` MUST pass `srs_path`.
  - WP-CORE-6: any future Architect stage that produces context proposals MUST include `supporting_sentence_ids`; `extract_per_context_details` signature is `List[ContextHypothesis]`.
  - **WP-CORE-7 NEW invariant**: every `VerifierIssue.target` (or `.location` in legacy) MUST be prefixed with `'{stage}:'`. `_issue_stage` dispatch depends on this prefix.
  - **WP-CORE-7 NEW invariant**: every constructor of `PipelineDeps` MUST supply `architect_with_feedback`.
- **EMSE paper**: pre-WP-CORE-7 Methods-section claim "D1 catches contexts citing un-emitted sentences" was true at verifier level only. Post-WP-CORE-7: "D1 ERRORs trigger one issue-aware re-prompt of Architect; persistent grounding violations raise `ArchitectGroundingError` and fail the run." Flag for advisor at next review.
