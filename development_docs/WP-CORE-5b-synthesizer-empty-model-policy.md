# WP-CORE-5b — `SynthesizerEmptyModelError` guard placement + taxonomy preservation

**Status:** SHIPPED 2026-05-21
**Branch:** `main`
**Commit range:** `cc82e64` (RED) → `27a5d98` (GREEN) → `{doc-sha}` (DOC, this commit) → `{planning-sha}` (PLANNING)
**Pre-WP HEAD:** `2b8602f` (WP-CORE-4 final state)
**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-5b-synthesizer-empty-model-policy-design.md` (v2, post-Codex)
**Plan:** `docs/superpowers/plans/2026-05-21-wp-core-5b-synthesizer-empty-model-policy.md`
**Parent finding:** `.planning/pipeline_audit/findings/architect.md` finding **F-14** (MAJOR)
**Predecessor in this iteration:** WP-CORE-5 ABANDONED at v1 (F-11 dormant in production; Codex review surfaced 3 CRITICALs; spec preserved with banner)
**Test delta:** 332 → 338 (+6 tests, zero regression)

## TL;DR

When the pipeline encounters an empty `refined_specialist` list (DI-reachable but production-dormant), it now raises `SynthesizerEmptyModelError` (subclass of `PipelineError`) instead of `pydantic.ValidationError` — preserving the orchestration error taxonomy. The error now carries `srs_path` (matching WP-CORE-4's `IntermediateSaveError` pattern) for diagnostic clarity. Hard-fail policy is unchanged — `_create_fallback_model` deletion stays in force; empty model is still never a legitimate pipeline output. The post-call boundary check is retained as belt-and-suspenders against injected synthesizers that bypass Pydantic via `DomainModel.model_construct`.

## Motivation

### Audit text mismatch with code reality

`.planning/pipeline_audit/findings/architect.md` §F-14 framed the bug as "Hard-fail on a degenerate case — needs explicit policy (hard-fail vs degrade)". Close-lookup invalidated that framing:

1. **Hard-fail policy is already explicit.** `tests/test_synthesizer_empty_model_error.py:test_create_fallback_model_is_gone` enforces the deletion of `_create_fallback_model`, with the rationale `"_create_fallback_model must be deleted; an empty model is no longer a legitimate pipeline output."` WP-CORE-1 made this decision deliberately. Adding a degrade path would regress that contract.
2. **The post-synthesizer `if not model.bounded_contexts:` check is dead for the in-tree synthesizer.** `core/schemas.py:207-215` has a Pydantic `_non_empty` validator on `DomainModel.bounded_contexts` that raises before the check runs. Empty input → `build_deterministic_skeleton([])` → `DomainModel(bounded_contexts=[])` → `pydantic.ValidationError` → propagates up through `synthesize_domain_model` → up to `pipeline.py:81`'s `deps.synthesizer(...)` → never reaches the post-call check.
3. **Production-reachable trigger is dormant** (parallel to F-11). Architect's `identify_contexts` raises `ArchitectExtractionError` on zero contexts (`architect.py:482-485, 501-504`); Specialist's `extract_per_context_details` raises `SpecialistFailureError` on per-context exhaustion. So `refined_specialist == []` cannot happen in production via `analyze_document`. DI paths (tests, alternative pipelines) can.

### Real F-14 gap

The actual problem: when `refined_specialist == []`, the pipeline raises `pydantic.ValidationError` — which is **not** a `PipelineError` subclass. This violates the orchestration error taxonomy documented in `core/orchestration/errors.py:1-6`:

> "All silent fallbacks in core/architect.py are converted to raises of these classes. The top-level orchestrator catches PipelineError, writes a structured failure_log.json, and decides retry/skip/fail per RQ1 metrics policy."

Today, no consumer wires `except PipelineError:`; `main.py:180/427/533` all catch generic `Exception`. So the user-visible behavior change from this WP is small — `str(e)` becomes terser + names the SRS path. But the **taxonomic correctness** is paper-relevant: the EMSE Methods section claims structured error categorization; that claim cannot hold if a major failure mode escapes through `pydantic.ValidationError`.

### Why fix now (despite dormancy)

Codex W-7 disposition reframed this WP as **"contract cleanup for paper-methodology integrity"** rather than "production hardening". The fix is small (~22 LOC of real code + ~50 LOC of doc comments), preserves an explicit existing contract, and lays the foundation for a future `except PipelineError:` handler to work correctly. Codex W-8 noted F-21 (vacuous D1 verifier) has higher paper impact and should be the next iteration target.

## Architectural decisions

### D-1 — Pre-call guard placement (Codex OQ-1)

The fix is a `if not refined_specialist: raise SynthesizerEmptyModelError(...)` guard placed **before** `deps.synthesizer(refined_specialist)`, not a `try: ... except ValidationError as exc: ...` post-call rewrap.

**Rationale**:
- Pre-call: 5 LOC, no fragile error-shape matching. Trivially correct.
- Post-call rewrap would require `if exc.errors()[0]['loc'] == ('bounded_contexts',): raise SynthesizerEmptyModelError(...) from exc` — fragile under Pydantic version changes, and accidentally swallows other unrelated `ValidationError`s.
- Codex agreed: pre-call wins on KISS.

### D-2 — Post-call boundary check retained as belt-and-suspenders (Codex W-3)

The v1 spec proposed deleting the post-call `if not model.bounded_contexts:` check entirely (because Pydantic catches it first). Codex W-3 reversed this: `PipelineDeps.synthesizer` is a freely injectable `Callable[[List[SpecialistAnalysis]], DomainModel]`. An injected synthesizer that bypasses Pydantic via `DomainModel.model_construct(...)` (which skips all validators) could return an empty model that escapes the pre-call guard.

**Layer cake** (3 defenses, each at a different layer):

| Layer | Catches | Triggered by |
|---|---|---|
| Pre-call guard at `pipeline.py` | `refined_specialist == []` | Empty Specialist output from DI or refiner-shrink |
| Pydantic `_non_empty` validator at `schemas.py:207` | `DomainModel(bounded_contexts=[])` via normal `__init__` | In-tree synthesizer (`synthesize_domain_model`) |
| Post-call check at `pipeline.py` | `model.bounded_contexts == []` from `model_construct` bypass | Injected synthesizers that skip validation |

T-EMPTY-4 in the test suite exercises the third layer with a deliberate `DomainModel.model_construct(bounded_contexts=[], ...)` injection.

### D-3 — `srs_path` symmetry with `IntermediateSaveError` (Codex OQ-2)

WP-CORE-4 added `srs_path` to `IntermediateSaveError` so error messages name the failing SRS. Codex OQ-2 asked whether to add it to `SynthesizerEmptyModelError` now or defer. Decision: **add now** for taxonomy symmetry. `run_pipeline` widened with `srs_path: Optional[str] = None` kwarg; `analyze_document` (`core/architect.py:846`) threads `self._current_srs_path` (already unconditionally assigned per WP-CORE-4 W-2). When the guard fires, the error message reads:

```
SynthesizerEmptyModelError: Synthesizer returned an empty DomainModel
  (srs=/abs/path/SRS.docx; input: 0 SpecialistAnalysis from upstream pipeline)
```

vs the pre-WP message:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for DomainModel
bounded_contexts
  Value error, bounded_contexts must be non-empty; an empty DomainModel
  indicates upstream pipeline failure and must raise instead.
```

Both contain enough information to diagnose, but the new message is shorter, names the SRS, and lives in the documented `PipelineError` taxonomy.

### D-4 — Refiner-shrink-to-empty test (Codex W-1 + W-4)

The v1 spec only covered the "initial Specialist returns `[]`" path. Codex W-1 flagged that `refine_until_clean` (`core/refiner/loop.py:28-37`) returns whatever `stage_runner` last produced on the verifier-ok path; if the refiner rerun returns `[]` AND the verifier accepts it, `refined_specialist` becomes `[]` even though `specialist_output` was non-empty. T-EMPTY-3 exercises this edge — exception-free path from non-empty input to empty output.

### D-5 — Genuine-RED-fail TDD pattern (Codex OQ-5)

WP-CORE-3 + WP-CORE-4 RED commits used per-test imports + minimal-stub fixtures so the new tests collected cleanly even though the symbols-under-test didn't exist yet. WP-CORE-5b takes a different approach: 5 of the 6 new tests deliberately fail against the pre-GREEN code (`SynthesizerEmptyModelError` constructor's `srs_path` kwarg is unknown; the pre-call guard doesn't exist; the post-call check emits the old message format). Codex OQ-5 agreed this is honest TDD — the test asserts the future contract; production must move to satisfy it.

## File-level changes

| file | change | lines | rationale |
|---|---|---|---|
| `core/orchestration/errors.py` | `SynthesizerEmptyModelError.__init__` accepts `srs_path: str = "<unknown>"`; class docstring added; default message includes `srs={path}` and `input: {summary}` | +27 / -2 | D-3 — taxonomy symmetry with `IntermediateSaveError` |
| `core/orchestration/pipeline.py` | `run_pipeline` signature widened with `srs_path: Optional[str] = None`; pre-call guard added; post-call check rewritten to emit `"bypassed Pydantic"` message; both wrap `srs_path or "<unknown>"`; doc comments explain layer-cake | +43 / -4 | D-1, D-2 — primary + belt-and-suspenders guards at the orchestration layer |
| `core/architect.py` | `analyze_document` calls `run_pipeline(srs_text=text, deps=deps, srs_path=self._current_srs_path)` | +4 / -1 | D-3 — thread `srs_path` from existing `_current_srs_path` (WP-CORE-4 invariant) |
| `tests/test_pipeline_orchestration.py` | append T-EMPTY-1 / -2 / -3 / -4 (taxonomy + synthesizer-not-invoked + refiner-shrink + injected-synthesizer) | +112 / -0 | D-4, D-5 — TDD coverage of all three layers |
| `tests/test_synthesizer_empty_model_error.py` | append T-EMPTY-5 (`srs_path` field) + T-EMPTY-6 (diagnostic message format) | +36 / -0 | D-3, D-5 — regression-lock on srs_path + str(err) contract |

**Net diff**: +148 LOC tests + +72 LOC production = +220 LOC total. Of the 72 production LOC, ~50 are doc comments explaining the layer cake; real code change is ~22 LOC.

## Methodology applied

- **TDD with genuine RED-fail** (D-5). RED commit `cc82e64` lands 6 failing tests (5 GREEN-required + 1 regression-lock-required). GREEN commit `27a5d98` turns all 6 green with zero regression.
- **Codex xhigh adversarial review** with zero-deferred standard (third consecutive iteration: WP-CORE-3 + WP-CORE-4 + WP-CORE-5b). Codex's 6 WARN + 3 NITS + 3 OQ all handled inline in spec v2; no findings carried forward.
- **Atomic Conventional Commits** with Claude trailer. RED → GREEN → DOC → PLANNING cadence matches WP-CORE-3/4.
- **Smallest correct change.** No degrade-best-effort added (would regress the explicit `_create_fallback_model = gone` contract). No taxonomy-wide refactor. Three layers of defense, each at its appropriate scope (DI / schema / injected-bypass).

## Empirical results

- **Test count**: 332 → 338 (+6, all green at GREEN HEAD `27a5d98`).
- **Regression count**: 0 (verified by full pytest at GREEN commit).
- **Pre-WP error type on `refined_specialist == []`**: `pydantic_core._pydantic_core.ValidationError`.
- **Post-WP error type on same path**: `core.orchestration.errors.SynthesizerEmptyModelError(PipelineError)`.
- **Production impact**: dormant; in-tree pipeline through `analyze_document` still cannot reach the empty case (Architect upstream guard intact). DI-test impact: clean.
- **EMSE methodology impact**: positive but small. Methods section's claim of "structured PipelineError taxonomy for all failure modes" now holds for the empty-Specialist case; previously was a known taxonomy escape.

## Limitations + follow-ups

1. **F-21 (vacuous D1 verifier pass) is the highest-priority next iteration.** Codex W-8 specifically flagged this — `ContextHypothesis.supporting_sentence_ids` defaults to `[]` because the Architect never populates it; D1 verifier check at `core/verifier/checks_deterministic.py:7-27` iterates an empty list and reports zero violations. **Every project run in history has passed D1 vacuously.** EMSE methodology consequence is much larger than F-14's dormant taxonomy edge. Queued as iteration-5 target.
2. **F-11 (parallel Scout rate-limit race) remains DORMANT.** Will reopen when `extract_domain_sentences` rewires into `analyze_document.scout_fn` or `section_aware_chunks` gains an LLM call. See `decision_log.md` D-CODEX-REVIEW-WP-CORE-5.
3. **No `except PipelineError:` handler exists yet.** `main.py:180/427/533` still catch generic `Exception`. The taxonomy fix here is foundational; an actual structured-error-logging consumer (per `errors.py:1-6` docstring) is a separate, larger WP.
4. **Pre-call message hard-codes `"0 SpecialistAnalysis from upstream pipeline"`.** A future evolution could enrich with the upstream stage that produced the empty list (Refiner vs. initial Specialist), but it's not currently surfaceable from `refine_until_clean`'s return shape. Defer.
5. **Post-call belt-and-suspenders has no current production consumer.** Only T-EMPTY-4 exercises it. If no real injected synthesizer ever ships, the layer is over-engineering. Codex W-3 judged the cost-benefit favorable; we'll revisit if maintenance burden surfaces.

## Cross-references

- Sibling pattern: [[WP-CORE-4-intermediate-save-observability]] — same `srs_path` propagation + `PipelineError` taxonomy pattern
- Predecessor in iteration 4: [[wp-core-5-abandoned]] (WP-CORE-5 F-11 spec, banner-marked ABANDONED — preserved for audit trail)
- Project-locked policy: `tests/test_synthesizer_empty_model_error.py:test_create_fallback_model_is_gone` — empty model is never a legitimate output
- Pydantic schema invariant: `core/schemas.py:_non_empty` validator (lines 207-215)
- Error taxonomy charter: `core/orchestration/errors.py:1-6` docstring
- Codex review record: `.planning/pipeline_audit/decision_log.md` D-CODEX-REVIEW-WP-CORE-5b
- AGENTS.md "Error handling: explicit failure. No silent degradation."
