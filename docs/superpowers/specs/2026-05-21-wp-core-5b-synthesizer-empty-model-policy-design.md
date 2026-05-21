# WP-CORE-5b — `SynthesizerEmptyModelError` guard placement + taxonomy preservation

**Date:** 2026-05-21
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 4 — pivot after WP-CORE-5 abandon)
**Status:** REVISED v2 — addressed Codex xhigh adversarial review (0 CRITICAL + 6 WARN + 3 NITS + 3 OQ; all handled inline)
**Parent finding:** `.planning/pipeline_audit/findings/architect.md` finding **F-14** (MAJOR)
**Loop:** Domain Pipeline Hardening Loop (fourth WP attempt; baseline 332 confirmed at HEAD `2b8602f`)
**Sibling iterations:**
- Iteration 1 — WP-CORE-2 shipped at `25e6880` (reference-heading truncation)
- Iteration 2 — WP-CORE-3 shipped at `daefeb0` (empty-input contract)
- Iteration 3 — WP-CORE-4 shipped at `02e0fe9` (`IntermediateSaveError` + `srs_path` propagation)
- Iteration 4a — WP-CORE-5 **ABANDONED at v1** (F-11 parallel Scout race; Codex review surfaced 3 CRITICALs; F-11 dormant in production; spec preserved as audit-trail banner)
**Cross-ref:** `decision_log.md` entries `D-CODEX-REVIEW-WP-CORE-5` (Codex output for the abandoned WP), `D-PICK-WP-CORE-5b` (pivot rationale), and `D-CODEX-REVIEW-WP-CORE-5b` (this review's disposition table).

---

## Revision history

- **v1 (draft, 2026-05-21 ~11:30 GMT+3)** — initial spec; sent to Codex xhigh for adversarial review.
- **v2 (this version, 2026-05-21 ~12:00 GMT+3)** — Codex xhigh review verdict: **GO with 3 conditions** (0 CRITICAL + 6 WARN + 3 NITS + 3 OQ). All findings handled inline. Dispositions:

  | # | finding | disposition |
  |---|---|---|
  | **W-1** | `refiner-empty-success-path` — `refine_until_clean` non-exception path can shrink `refined_specialist` to `[]` if `stage_runner` returns `[]` on rerun and verifier accepts (`core/refiner/loop.py:28-37`). Spec missed this DI-reachable path. | **ADOPTED.** Discovery 3 updated to acknowledge the refiner success-path edge. New test **T-EMPTY-3** added: first Specialist returns non-empty, verifier fails once, rerun returns `[]`, verifier accepts → pipeline raises `SynthesizerEmptyModelError`. Pre-call guard still catches because `refined_specialist == []` at the gate. |
  | **W-2** | `specialist-empty-is-only-blocked-upstream` — `extract_per_context_details([])` returns `[]` without raising (`architect.py:574-587, 656-664`). The "Specialist raises" claim in Discovery 3 is only true for non-empty `contexts` input. | **ADOPTED.** Discovery 3 reworded: "production chain is protected by **Architect's** `identify_contexts` upstream raise on zero contexts; if zero contexts ever reaches Specialist via DI, `extract_per_context_details([])` returns `[]` silently — and the pre-call guard at the pipeline catches that case." |
  | **W-3** | `keep-post-boundary-check` — Deleting the post-call check loses a cheap boundary invariant for injected/future synthesizers (`PipelineDeps.synthesizer` is `SynthesizerFn = Callable[[List[SpecialistAnalysis]], DomainModel]`, freely injectable). | **ADOPTED.** Post-call check stays. Pre-call guard adds as the primary; post-call retained as belt-and-suspenders for injected synthesizers that bypass Pydantic (e.g., constructing `DomainModel.model_construct(...)` to skip validation). |
  | **W-4** | `missing-refiner-empty-test` — RED tests only cover initial-empty Specialist. | **ADOPTED via T-EMPTY-3** (see W-1). |
  | **W-5** | `t-empty-4-duplicates-existing-layer` — Pydantic-validator coverage already exists at `tests/test_synthesizer_deterministic_merge.py:97-103`. | **ADOPTED.** Original T-EMPTY-4 (direct `synthesize_domain_model([])` raises `ValidationError`) **dropped**. Avoids duplicate coverage. |
  | **W-6** | `red-commit-ci-risk` — RED commit with pytest exit 1 ok locally but poor shared-history hygiene. | **NO ACTION.** Confirms existing loop discipline: commits stay local until DOC + planning land; user-driven `git push` only. Documented in §Risks. |
  | **W-7** | `taxonomy-not-consumed` — `main.py:180,427,533` all catch generic `Exception`; PipelineError taxonomy is future-facing. | **ADOPTED.** §Motivation reframed: fix is "contract cleanup for paper-methodology integrity" not "production hardening". Honest framing. |
  | **W-8** | `f21-priority` — F-21 has clearer paper impact (every project run vacuously passes D1 check). | **ACKNOWLEDGED.** WP-CORE-5b stays as quick cleanup (Codex's "ship as quick cleanup" option). F-21 explicitly queued as next iteration in the post-WP-CORE-5b handoff. |
  | **N-1** | `dead-code-scope` — Post-call check is dead only for in-tree synthesizer, not globally. | **ADOPTED.** Discovery 2 tightened. |
  | **N-2** | `taxonomy-test-overlap` — T-EMPTY-1 + T-EMPTY-2 overlap. | **ADOPTED.** Merged into single test (`pytest.raises(PipelineError) as exc; assert isinstance(exc.value, SynthesizerEmptyModelError)`). |
  | **N-3** | `emse-framing-holds` — Hard-fail grounded; do not add recovery modes. | **CONFIRMED.** §Non-goals unchanged. |
  | **OQ-1** | Pre-call vs catch-and-rewrap. | **ADOPTED PRE-CALL** (Codex agreed). |
  | **OQ-2** | `srs-path` symmetry — should `SynthesizerEmptyModelError` carry `srs_path` like `IntermediateSaveError`? | **ADOPTED.** Add `srs_path: str = "<unknown>"` to constructor; widen `run_pipeline` signature with `srs_path: Optional[str] = None` kwarg-only; thread from `analyze_document` (`self._current_srs_path` already available). Symmetry with WP-CORE-4. ~5 LOC additional change. New test **T-EMPTY-5** verifies `srs_path` carried. |
  | **OQ-3** | `main-handler-test` — Lifespan/endpoints catch generic `Exception`; switching from `ValidationError` to `PipelineError` doesn't change handler selection. | **NO ACTION** — Codex agreed no main.py test needed. |

---

## Motivation

The audit text for F-14 (`.planning/pipeline_audit/findings/architect.md` §F-14) flagged `SynthesizerEmptyModelError` as a "hard-fail on a degenerate case — needs explicit policy", proposing `degrade-best-effort` as an alternative. **Close-lookup of the codebase invalidates that framing**:

### Discovery 1 — Hard-fail policy is already explicit and intentional

`tests/test_synthesizer_empty_model_error.py:26-34`:

```python
def test_create_fallback_model_is_gone():
    """B4 deletes _create_fallback_model. DomainArchitect instance must not have
    the attribute (checked via object.__new__ to bypass __init__)."""
    from core.architect import DomainArchitect
    arch = DomainArchitect.__new__(DomainArchitect)
    assert not hasattr(arch, "_create_fallback_model"), (
        "_create_fallback_model must be deleted; an empty model is "
        "no longer a legitimate pipeline output."
    )
```

The codebase carries a regression-locked decision: **degrade-best-effort was explicitly removed** (WP-CORE-1 cleanup, "_create_fallback_model" deleted). The hard-fail invariant is a project-wide design contract, not an oversight.

### Discovery 2 — The post-synthesizer guard is dead code (Pydantic already raises)

`core/schemas.py:207-215`:

```python
@field_validator("bounded_contexts")
@classmethod
def _non_empty(cls, v: List[BoundedContext]) -> List[BoundedContext]:
    if not v:
        raise ValueError(
            "bounded_contexts must be non-empty; an empty DomainModel "
            "indicates upstream pipeline failure and must raise instead."
        )
    return v
```

When `synthesize_domain_model([])` runs (no `SpecialistAnalysis` inputs):

1. `build_deterministic_skeleton([], project_name=...)` iterates `analyses=[]` → `bounded_contexts=[]`.
2. `DomainModel(..., bounded_contexts=[], ...)` Pydantic constructor runs `_non_empty` validator.
3. `_non_empty` raises `ValueError("bounded_contexts must be non-empty; ...")`.
4. Pydantic wraps as `pydantic.ValidationError`.
5. `ValidationError` propagates up through `synthesize_domain_model` → `deps.synthesizer(refined_specialist)` at `core/orchestration/pipeline.py:81`.
6. `pipeline.py:82` `if not model.bounded_contexts: raise SynthesizerEmptyModelError(...)` **never executes** — `model` never bound; `ValidationError` already in flight.

So:
- `SynthesizerEmptyModelError` is **unreachable for the in-tree synthesizer** (`synthesize_domain_model` in `core/synthesizer/__init__.py`). Per Codex N-1, the dead-code claim is scoped to the in-tree synthesizer, not globally — `PipelineDeps.synthesizer` is `Callable[[List[SpecialistAnalysis]], DomainModel]`, freely injectable. An injected synthesizer that bypasses Pydantic (e.g., via `DomainModel.model_construct(...)`) could still return an empty model; the post-call check remains the right last-line defense for that case.
- For the production code path, the actual error type that escapes the pipeline on empty `refined_specialist` is `pydantic.ValidationError`, which is **not** a `PipelineError` subclass — violating the orchestration error taxonomy in `core/orchestration/errors.py` ("All silent fallbacks in core/architect.py are converted to raises of these classes").

### Discovery 3 — Production-reachable trigger is dormant, but DI paths include a refiner-rerun edge

For `refined_specialist` to be `[]` at `pipeline.py:81`:

- **Architect upstream guard.** `architect_fn` calls `self.identify_contexts(...)` which raises `ArchitectExtractionError` on zero contexts (`core/architect.py:482-485` and `:501-504`). So in production, `arch.contexts` is always ≥ 1 by the time `specialist_fn` runs.
- **Specialist has NO empty-input guard of its own.** Per Codex W-2: `extract_per_context_details([])` initializes `results=[]`, iterates over empty `contexts`, never enters the per-context loop, and returns `[]` cleanly (`core/architect.py:574-587, 656-664`). The protection against `len(specialist_output) == 0` comes ENTIRELY from Architect upstream, not from Specialist's own logic. If a future refactor weakens the Architect guard or a different code path produces zero contexts, Specialist silently returns `[]`.
- **Refiner success path can shrink to empty.** Per Codex W-1: `refine_until_clean` returns whatever `stage_runner` last produced when verifier returns ok. `_re_run_specialist` calls `deps.specialist(arch, scout)` (`pipeline.py:53-55`); if that rerun returns `[]` AND the verifier accepts the empty output (e.g., a verifier that returns `ok=True` on empty input by default), `refined_specialist` becomes `[]` even though `specialist_output` was non-empty. This is an exception-free path that the v1 spec missed.
- **Refiner exception path** falls back to `refined_specialist = specialist_output` (`pipeline.py:79`) — preserves length.

**Production reachability via `analyze_document`**: nil (Architect guards). **Test-DI reachability**: yes:
1. `specialist_fn` returns `[]` on first call (skips refiner cycle if verifier accepts; falls through to synthesizer).
2. `specialist_fn` returns non-empty on call 1 but `[]` on call 2 + verifier accepts (refiner-rerun edge).

The pre-call guard in the proposed fix catches both DI paths uniformly because it gates on `refined_specialist == []` regardless of how the list became empty.

Like F-11, F-14's production symptom is dormant — but unlike F-11, the **fix is small, narrow, and preserves a contract that already exists** (PipelineError taxonomy). Per Codex W-7 disposition, the honest framing is: **this WP is contract cleanup for paper-methodology integrity**, not production hardening. `main.py:180/427/533` all catch generic `Exception` today, so user-visible behavior changes only in `str(e)` content (the message becomes terser + more diagnostic). The fix's primary value is taxonomic + EMSE-reproducibility — ensuring every pipeline failure mode goes through `PipelineError` so the paper's claim of "structured failure_log.json per RQ1 metrics policy" (per `errors.py:1-6` docstring) can hold up when consumers actually wire `except PipelineError:` handlers in a future iteration.

### Restated problem

The real F-14 gap, as discovered:

> When `refined_specialist == []`, the pipeline raises `pydantic.ValidationError` instead of `SynthesizerEmptyModelError`. The error class exists but is never reached. Empty-input failure escapes the `PipelineError` taxonomy.

The fix preserves the explicit hard-fail policy (do NOT add degrade-best-effort) and re-routes the empty-input case through the taxonomically correct exception, with `srs_path` context (WP-CORE-4 pattern) for diagnostic value.

### Non-goals

- **No change to hard-fail policy.** `test_create_fallback_model_is_gone` stays green; `_create_fallback_model` stays deleted.
- **No `degrade-best-effort` mode.** Explicitly rejected per project history.
- **No retry-with-relaxed-constraints.** Same reason.
- **No `srs_path` field on `SynthesizerEmptyModelError`.** Could be added (WP-CORE-4 pattern for `IntermediateSaveError`), but `run_pipeline` signature doesn't currently carry it; adding would widen the WP scope. Defer to a follow-up if observability demands it.
- **No deletion of `SynthesizerEmptyModelError`.** It's the right taxonomic exception; we make it reachable.
- **No deletion of Pydantic `_non_empty` validator.** Last-line defense at the schema layer stays — protects against future code paths that bypass `pipeline.run_pipeline`.
- **No change to Refiner / `RefinementExhaustedError` handling.** Out of scope.
- **No change to upstream `ArchitectExtractionError` / `SpecialistFailureError` raises.** Those guard correctly upstream.

---

## Design

### Production code changes (v2 — 3 files)

Per Codex W-3 + OQ-2, the post-call check is **retained** (not deleted), and `srs_path` is added to `SynthesizerEmptyModelError` for WP-CORE-4 symmetry.

#### File 1 — `core/orchestration/errors.py`

```python
# BEFORE (lines 35-38):
class SynthesizerEmptyModelError(PipelineError):
    def __init__(self, input_summary: str, message: Optional[str] = None):
        self.input_summary = input_summary
        super().__init__(message or f"Synthesizer returned an empty DomainModel (input: {input_summary})")
```

```python
# AFTER:
class SynthesizerEmptyModelError(PipelineError):
    """Raised when the pipeline detects an empty Specialist input or an
    empty DomainModel from an injected synthesizer.

    Per AGENTS.md "Error handling: explicit failure. No silent degradation":
    an empty DomainModel is never a legitimate pipeline output (see
    test_create_fallback_model_is_gone). This exception preserves the
    PipelineError taxonomy for that case.

    Carries:
        input_summary: brief textual description of the failing input
            (e.g., "0 SpecialistAnalysis from upstream pipeline" or
            "synthesizer returned 0 bounded contexts").
        srs_path: the SRS being processed (or "<unknown>" if unset). Matches
            the WP-CORE-4 pattern for IntermediateSaveError.
    """

    def __init__(
        self,
        input_summary: str,
        srs_path: str = "<unknown>",
        message: Optional[str] = None,
    ):
        self.input_summary = input_summary
        self.srs_path = srs_path
        super().__init__(
            message
            or f"Synthesizer returned an empty DomainModel (srs={srs_path}; input: {input_summary})"
        )
```

#### File 2 — `core/orchestration/pipeline.py`

```python
# BEFORE — signature (line 36):
def run_pipeline(*, srs_text: str, deps: PipelineDeps) -> DomainModel:

# BEFORE — guard (lines 81-84):
    model: DomainModel = deps.synthesizer(refined_specialist)
    if not model.bounded_contexts:
        raise SynthesizerEmptyModelError(input_summary=f"{len(refined_specialist)} contexts")
    return model
```

```python
# AFTER — signature widened:
def run_pipeline(
    *,
    srs_text: str,
    deps: PipelineDeps,
    srs_path: Optional[str] = None,
) -> DomainModel:

# AFTER — guard becomes pre-call + post-call belt-and-suspenders:
    # Pre-call guard (primary): catches refined_specialist == [] for both
    # initial-empty and refiner-rerun-to-empty DI paths. Per Codex W-1 + W-2,
    # this is the only place where the empty case can be observed
    # taxonomically as a PipelineError; the in-tree synthesizer would
    # otherwise raise pydantic.ValidationError via DomainModel._non_empty.
    if not refined_specialist:
        raise SynthesizerEmptyModelError(
            input_summary="0 SpecialistAnalysis from upstream pipeline",
            srs_path=srs_path or "<unknown>",
        )

    model: DomainModel = deps.synthesizer(refined_specialist)

    # Post-call boundary check (belt-and-suspenders): retained per Codex W-3
    # because PipelineDeps.synthesizer is an injectable SynthesizerFn; a
    # future or test-injected synthesizer could construct DomainModel via
    # model_construct (which bypasses Pydantic validation) and return an
    # empty model. The in-tree synthesizer is already caught by Pydantic
    # _non_empty validator (core/schemas.py:207-215), making this branch
    # dead for the in-tree path — but ALIVE for injected paths.
    if not model.bounded_contexts:
        raise SynthesizerEmptyModelError(
            input_summary="synthesizer returned 0 bounded contexts (bypassed Pydantic)",
            srs_path=srs_path or "<unknown>",
        )
    return model
```

#### File 3 — `core/architect.py`

Thread `srs_path` from `analyze_document` to `run_pipeline`:

```python
# BEFORE (line 846):
        return run_pipeline(srs_text=text, deps=deps)
```

```python
# AFTER:
        return run_pipeline(
            srs_text=text,
            deps=deps,
            srs_path=self._current_srs_path,
        )
```

`self._current_srs_path` is already unconditionally assigned at the start of `analyze_document` per WP-CORE-4 W-2 (line 736).

### Diff summary (3 files, ~31 LOC net change)

| file | LOC added | LOC removed | net |
|---|---|---|---|
| `core/orchestration/errors.py` | +13 | -2 | +11 |
| `core/orchestration/pipeline.py` | +21 | -4 | +17 |
| `core/architect.py` | +4 | -1 | +3 |
| **total** | **+38** | **-7** | **+31** |

Of the +31 net LOC, ~22 are doc comments. Code-only diff is ~9 LOC.

### Why pre-call AND post-call (both layers)

| layer | catches | when |
|---|---|---|
| Pre-call guard at `pipeline.py` | `refined_specialist == []` | Any DI path that produces empty Specialist output (initial-empty OR refiner-shrink-success). Closes Codex W-1 + W-2. |
| Pydantic `_non_empty` validator at `schemas.py:207` | `DomainModel(bounded_contexts=[])` via normal `__init__` | The in-tree `synthesize_domain_model` path (analyses passed but somehow merged to nothing — currently impossible since `merge.py:28-45` always produces one BC per analysis, but defends against future merge changes). |
| Post-call belt-and-suspenders at `pipeline.py` | `model.bounded_contexts == []` despite Pydantic | Injected synthesizers that bypass Pydantic via `DomainModel.model_construct(...)`, schema-version mismatch, or future Pydantic behavior changes. Closes Codex W-3. |

Each layer fails-loud with a `SynthesizerEmptyModelError` (or `ValidationError` for the Pydantic one — direct callers of `DomainModel(...)` outside `run_pipeline` get the validator's error, which is the intended contract for non-pipeline callers).

### Why keep Pydantic `_non_empty` validator (unchanged)

It's a different defense at a different layer. The Pydantic validator catches:
- Direct `DomainModel(...)` construction from anywhere (e.g., tests, future ingest paths, schema migration code).
- Future code paths that bypass `pipeline.run_pipeline` (e.g., a separate offline analyzer or REPL session).

Removing it would create a gap at the schema layer; keeping it is belt-and-suspenders at the schema layer (separate from the pipeline-layer post-call check).

### What downstream callers see now

| Caller | Before | After |
|---|---|---|
| `analyze_document` → caller of `run_pipeline` | `pydantic.ValidationError` (escapes the `PipelineError` taxonomy) | `SynthesizerEmptyModelError` (subclass of `PipelineError`) with `srs_path` populated from `self._current_srs_path` |
| `main.py:180` lifespan handler | catches as generic `Exception`, prints traceback | catches as generic `Exception`, prints traceback (no handler-selection change today — but future `except PipelineError:` handler would now correctly catch this case, and the traceback now names the SRS path that failed) |
| `main.py:427-434` `/generate-model` endpoint | catches as `Exception`, returns `{"success": False, "error": str(e)}` with Pydantic's verbose error msg | catches as `Exception`, returns `{"success": False, "error": "Synthesizer returned an empty DomainModel (srs=/abs/path/SRS.docx; input: 0 SpecialistAnalysis from upstream pipeline)"}` — terser AND names the SRS |
| `main.py:534` `/generate-model-stream` | same as `/generate-model` | same as `/generate-model` |
| `tests/test_pipeline_orchestration.py` | n/a (no test for this path today) | new tests T-EMPTY-1..T-EMPTY-4 assert `SynthesizerEmptyModelError` raises with correct taxonomy + `srs_path` |

---

## Red-phase tests (v2 — 6 tests, post-Codex dispositions applied)

New tests go in two existing files (no new files — F-14 fits cleanly within existing test taxonomy). Per Codex N-2, T-EMPTY-1 and T-EMPTY-2 are merged into a single test using `pytest.raises(PipelineError)` + `isinstance` check. Per Codex W-5, the original T-EMPTY-4 (direct `synthesize_domain_model([])` raises `ValidationError`) is dropped — already covered by `tests/test_synthesizer_deterministic_merge.py:97-103`. Per Codex W-1 + W-4, a new test covers the refiner-shrink-to-empty success path. Per OQ-2 disposition, a new test covers `srs_path` propagation. Per W-3, a new test covers the post-call belt-and-suspenders for injected synthesizers.

### `tests/test_pipeline_orchestration.py` — append 4 tests (T-EMPTY-1..4)

These mirror the existing `test_pipeline_propagates_architect_extraction_error` / `test_pipeline_propagates_specialist_failure` pattern.

| # | id | name | invariant |
|---|---|---|---|
| 1 | T-EMPTY-1 | `test_pipeline_raises_synthesizer_empty_model_error_when_specialist_returns_empty` | `deps.specialist = MagicMock(return_value=[])` → `pytest.raises(PipelineError) as exc; assert isinstance(exc.value, SynthesizerEmptyModelError)`. Merges the v1 T-EMPTY-1 + T-EMPTY-2 per Codex N-2 (single assertion covers both empty-raise AND taxonomy invariants). |
| 2 | T-EMPTY-2 | `test_pipeline_synthesizer_not_invoked_when_specialist_empty` | `synthesizer_fn = MagicMock()`; after the empty-Specialist path raises, `synthesizer_fn.call_count == 0`. Confirms guard is pre-call, not post-call. |
| 3 | T-EMPTY-3 (**NEW per W-1**) | `test_pipeline_raises_synthesizer_empty_model_error_when_refiner_rerun_returns_empty` | `specialist_fn` returns non-empty on call 1, `[]` on call 2; verifier returns `ok=False` (issues) on first call, `ok=True` on second call. `refine_until_clean` accepts the empty rerun output (no `RefinementExhaustedError`), `refined_specialist` becomes `[]`, pre-call guard raises `SynthesizerEmptyModelError`. Closes the refiner-success-path edge Codex flagged. |
| 4 | T-EMPTY-4 (**NEW per W-3**) | `test_pipeline_post_call_check_catches_injected_synthesizer_returning_empty_model` | `synthesizer_fn = lambda analyses: DomainModel.model_construct(bounded_contexts=[], project_name="x", project_metadata=..., global_rules=None)` (bypasses Pydantic). Pre-call guard not triggered (non-empty specialist). Post-call check raises `SynthesizerEmptyModelError` with `input_summary` containing "bypassed Pydantic". Confirms belt-and-suspenders. |

### `tests/test_synthesizer_empty_model_error.py` — append 2 tests (T-EMPTY-5..6)

| # | id | name | invariant |
|---|---|---|---|
| 5 | T-EMPTY-5 (**NEW per OQ-2**) | `test_synthesizer_empty_model_error_carries_srs_path` | `SynthesizerEmptyModelError(input_summary="x", srs_path="/abs/path/SRS.docx")` exposes `err.srs_path == "/abs/path/SRS.docx"` and `str(err)` contains the path. Default `srs_path="<unknown>"` when omitted. WP-CORE-4 symmetry with `IntermediateSaveError`. |
| 6 | T-EMPTY-6 | `test_synthesizer_empty_model_error_message_diagnostic` | `SynthesizerEmptyModelError(input_summary="0 SpecialistAnalysis from upstream pipeline")` has `str(err)` containing "empty DomainModel" and "0 SpecialistAnalysis" — diagnostic for support cases. |

**Total new tests:** 6 (4 + 2). Net change from v1: +2 (added T-EMPTY-3 refiner + T-EMPTY-4 injected-synthesizer + T-EMPTY-5 srs_path), -1 (dropped v1's direct-ValidationError test), -1 (merged v1's T-EMPTY-2 into T-EMPTY-1).

**Existing-test update**: `tests/test_synthesizer_empty_model_error.py:test_synthesizer_empty_model_error_carries_input_summary` (line 13-17) currently asserts `"0 contexts" in str(err)`. The new default `input_summary="0 SpecialistAnalysis from upstream pipeline"` does NOT contain "0 contexts". **The existing test passes an explicit `input_summary="0 contexts from 3 SpecialistAnalyses"` and asserts that — so no update needed.** Verified by re-reading the test.

**Expected RED signal:**
- T-EMPTY-1, T-EMPTY-2, T-EMPTY-3 fail today because `run_pipeline` raises `pydantic.ValidationError` (or refiner exhausts retries differently).
- T-EMPTY-4 fails today because the v1 post-call check raises with `input_summary=f"{len(refined_specialist)} contexts"` (literal "0 contexts" — wrong message; new v2 wording is "bypassed Pydantic"). Updated wording is the v2 contract.
- T-EMPTY-5 fails today because `srs_path` field doesn't exist on `SynthesizerEmptyModelError`.
- T-EMPTY-6 fails today because `str(err)` doesn't contain "empty DomainModel" with `srs_path` — it does already say "empty DomainModel" though, so this passes today. Re-categorize: T-EMPTY-6 is regression-lock, not RED-by-design.

Net RED expectation: 4 failing (T-EMPTY-1, T-EMPTY-2, T-EMPTY-3, T-EMPTY-4, T-EMPTY-5), 1 passing (T-EMPTY-6 — regression-lock only).

**Black-box test fixture pattern**: reuse `_make_typed_deps()` from `test_pipeline_orchestration.py:26-72`; override `deps.specialist = MagicMock(...)` and `deps.synthesizer = ...` as needed per test. Pattern matches existing `test_pipeline_propagates_*` tests.

---

## Atomic commit sequence (4 commits — matches WP-CORE-4 cadence)

| # | type | scope | summary | gate |
|---|---|---|---|---|
| 1 | `test` | `orchestration` | WP-CORE-5b red-phase tests for `SynthesizerEmptyModelError` guard placement + taxonomy | T-EMPTY-1..T-EMPTY-5 fail (expected red, 5 failing); T-EMPTY-6 passes (regression-lock only). Baseline 332 → 338 collected, 333 passed, 5 failed (RED-by-design). pytest exit code 1; commit message documents red signal rationale; loop ritual permits this because next commit (GREEN) is in the same WP and resolves them. |
| 2 | `fix` | `orchestration, architect` | WP-CORE-5b move `SynthesizerEmptyModelError` guard pre-call + retain post-call belt-and-suspenders + add `srs_path` | 3-file change (`core/orchestration/errors.py` + `core/orchestration/pipeline.py` + `core/architect.py`); +31 net LOC (most are doc comments; code-only ~9 LOC). Baseline pytest = 338 passed, 31 deselected. All 6 new tests green; all 332 prior tests green. |
| 3 | `chore` | `artifacts` | WP-CORE-5b dev_doc + audit state update | `development_docs/WP-CORE-5b-…md` (new) + `INDEX.md` ACTIVE row (new row #6) + `improvements_backlog.md` F-14 → SHIPPED + `CURRENT.md` next-pointer + handoff doc + `findings/architect.md` §F-14 status update. |
| 4 | `chore` | `planning` | WP-CORE-5b spec v2 + plan into git history | Land spec v2 (post-Codex disposition) + plan doc into the tree under `docs/superpowers/`. |

**Why split RED from GREEN**: same TDD rationale as WP-CORE-3/4 — diff of RED commit isolates the new behavior contract; GREEN commit shows the minimal production change that turns them green. Reviewer can check out RED commit's HEAD and run the failing tests against parent `2b8602f` to confirm the red signal.

**Why no test stub renames this time** (unlike abandoned WP-CORE-5): the new tests append to existing files; no symbol rename across the test suite.

**RED-commit pushability** (per Codex W-6): the RED commit deliberately introduces 5 failing tests, which would fail CI's `--cov-fail-under=60` gate. Loop discipline keeps all 4 commits local until user issues an explicit `git push`. After both RED + GREEN land, the cumulative diff has 0 failing tests. If CI is invoked between RED and GREEN, the loop is structurally pre-merge so failure is acceptable. WP-CORE-3 and WP-CORE-4 followed the same pattern.

---

## Risks

1. **RED commit accepts known-failing tests.** Risk: CI pipeline (`backend-ci.yml`) gates on `pytest --cov-fail-under=60` succeeding; a RED commit with 3 failing tests would fail CI if pushed. Mitigation: commits stay local until DOC + planning land (loop rule: `git push` only on explicit user "push it"). Prior WPs (WP-CORE-2, -3, -4) all landed RED commits locally with the same caveat.
2. **`SynthesizerEmptyModelError` `input_summary` text changes.** Old: `f"{len(refined_specialist)} contexts"` (would have been literal "0 contexts" — misleading; "0 contexts" sounds like "Architect found 0 contexts" which is the `ArchitectExtractionError` case). New: `"0 SpecialistAnalysis from upstream pipeline"`. Risk: any existing test or external consumer parses the message string for "0 contexts". Verified via grep — no consumer parses the string; only `test_synthesizer_empty_model_error.py:17` asserts `"0 contexts" in str(err)` which I will update in the RED commit's T-EMPTY-5 (assert the new message contents).
3. **Behavior change for callers of `run_pipeline` with empty Specialist DI**. Risk: a test elsewhere depends on `pydantic.ValidationError` being raised. Verified via grep `ValidationError` in tests — only `test_response_validator.py` uses it (LLM-response validator, unrelated). No test depends on the old behavior.
4. **Mypy/Pyright type inference on `deps.synthesizer(refined_specialist)`**. Risk: removing the post-call check leaves an implicit assumption that `model.bounded_contexts` is non-empty. Pyright sees `bounded_contexts: List[BoundedContext]` (no narrowing); no regression.
5. **A future contributor re-adds a degrade-best-effort fallback**. Risk: the dead-code-removal could be misread as "this case never fires, so we can build a fallback here". Mitigation: doc comment in the spec + the new test `test_create_fallback_model_is_gone` regression-locks the policy.
6. **`input_summary` could be enriched with `srs_path` later (WP-CORE-4 pattern)**. Not in scope here. If a follow-up WP wants it, the pattern is identical to `IntermediateSaveError`: thread `srs_path` through `run_pipeline` signature → wire into the constructor.

---

## Open questions — resolved post-Codex review

All OQs from v1 dispositioned in the revision-history table at top. Summary:

| OQ # | question | resolution |
|---|---|---|
| OQ-1 | Pre-call guard vs catch-and-rewrap. | **PRE-CALL** (Codex agreed; KISS, no fragile error-shape match). |
| OQ-2 | `srs_path` symmetry with `IntermediateSaveError`? | **YES** — adopted in v2. Test T-EMPTY-5 verifies. |
| OQ-3 | `except PipelineError` wrapper in `analyze_document`? | **NO** (Codex agreed; overengineering until a consumer demands it). |
| OQ-4 | Update `_non_empty` validator error message to name `SynthesizerEmptyModelError`? | **NO** — unnecessary noise; the validator is a schema-layer defense and its caller may not be `pipeline.run_pipeline`. Naming the orchestration exception would be misleading for direct-construction callers (e.g., tests that exercise `_non_empty` directly). |
| OQ-5 | RED commit pattern — genuine-fail vs per-test-import skip? | **GENUINE-FAIL** — Codex agreed this is honest TDD. Loop discipline keeps commits local until user `git push`. |
| OQ-6 | Delete the post-call check entirely? | **NO** (Codex W-3 reversed v1 inclination). Post-call retained as belt-and-suspenders for injected synthesizers that bypass Pydantic via `model_construct`. New T-EMPTY-4 covers this path. |
| OQ-7 | `input_summary` text change backwards-compatible? | **SAFE** — confirmed; no consumer parses the substring. Test `test_synthesizer_empty_model_error_carries_input_summary` passes an explicit `input_summary` and asserts that — not the constructor default. |

No deferred OQs. Spec v2 ready for plan-phase. Two consecutive iterations (WP-CORE-3 + WP-CORE-4) achieved zero-deferred standard inline; WP-CORE-5b maintains the streak (the abandoned WP-CORE-5 does not count toward the streak since it was scope-abandoned, not deferred).

---

## Pre-mortem (what could go wrong post-merge)

1. **A reviewer reads the GREEN diff and thinks "this check was already dead — why move it?"** Mitigation: spec §Discovery 2 explains the Pydantic-vs-pipeline taxonomy issue; commit message refs the spec.
2. **A contributor adds a new `deps.synthesizer` impl that doesn't go through `build_deterministic_skeleton` and may return `DomainModel(bounded_contexts=[])` from elsewhere.** This bypasses the pre-call guard. Mitigation: Pydantic `_non_empty` validator still catches. Last-line defense intact.
3. **A new test that injects `specialist_fn=MagicMock(return_value=[])` may break if it expected `ValidationError`.** Mitigated by exhaustive grep in §Risks #3.
4. **The `input_summary` literal becomes part of an external API contract** (e.g., screen-scrapping the error message). Mitigation: not a documented API; error messages are diagnostic-only.
5. **WP-CORE-5b creates a precedent that "dormant findings are still worth fixing for taxonomy preservation".** Could be misread as "fix every dead-code path". Mitigation: spec §Motivation explicitly frames the fix as "narrow, small, preserves a contract that already exists" — not a license for broad dead-code-cleanup WPs.

---

## Cross-references

- Finding: `.planning/pipeline_audit/findings/architect.md` §F-14
- Backlog: `.planning/pipeline_audit/improvements_backlog.md` row F-14 (will move to SHIPPED)
- Sibling specs (style/cadence + pattern source):
  - `docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md` (empty-input handling pattern)
  - `docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md` (PipelineError taxonomy + WP cadence)
- Existing test files appended to: `tests/test_pipeline_orchestration.py`, `tests/test_synthesizer_empty_model_error.py`
- Project policy decision: WP-CORE-1 deleted `_create_fallback_model` (see `test_synthesizer_empty_model_error.py:test_create_fallback_model_is_gone`)
- Pydantic schema invariant: `core/schemas.py:_non_empty` validator at lines 207-215
- Error taxonomy: `core/orchestration/errors.py:12` `PipelineError` base; `:35-38` `SynthesizerEmptyModelError` (currently unreachable)
- AGENTS.md "Error handling: explicit failure. No silent degradation."
