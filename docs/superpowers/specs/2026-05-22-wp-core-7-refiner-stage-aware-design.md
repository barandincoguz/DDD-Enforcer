# WP-CORE-7 — Refiner stage-aware re-runs + `ArchitectGroundingError` (F-22 mode C hybrid)

**Date:** 2026-05-22 / revised 2026-05-23
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 6)
**Status:** REVISED v2 — addressed Codex xhigh adversarial review (2 CRITICAL + 6 WARN + 2 NIT + 1 OQ; all CRITICAL+WARN handled inline; 1 OQ tracked as post-WP-CORE-7 follow-up)
**Parent finding:** `.planning/pipeline_audit/improvements_backlog.md` finding **F-22** (MAJOR, NEW from WP-CORE-6 Codex W-4 / A6-f22)
**Loop:** Domain Pipeline Hardening Loop (sixth WP; baseline 348 confirmed at HEAD `4c8580c`)
**Sibling iterations:**
- Iteration 1 — WP-CORE-2 shipped at `25e6880` (reference-heading truncation)
- Iteration 2 — WP-CORE-3 shipped at `daefeb0` (empty-input contract)
- Iteration 3 — WP-CORE-4 shipped at `02e0fe9` (`IntermediateSaveError` + `srs_path` propagation)
- Iteration 4 — WP-CORE-5 ABANDONED + WP-CORE-5b shipped at `27a5d98` (`SynthesizerEmptyModelError` taxonomy preservation)
- Iteration 5 — WP-CORE-6 shipped at `4c8580c` (D1 verifier non-vacuous, Architect populates `supporting_sentence_ids` end-to-end)
**Codex review:** `decision_log.md` D-CODEX-REVIEW-WP-CORE-7 (to be appended at DOC commit).

---

## Revision history

- **v1 (draft, 2026-05-22 ~22:00 GMT+3)** — initial spec; sent to Codex xhigh for adversarial review.
- **v2 (this version, 2026-05-23 ~00:30 GMT+3)** — Codex xhigh review verdict: **2 CRITICAL + 6 WARN + 2 NIT + 1 OQ**. Dispositions:

  | # | finding | category | disposition |
  |---|---|---|---|
  | **C-1** | `verifier-issue-stage-drop`: `core/pipeline_contracts.py:152-163` `VerifierIssue` (Pydantic) has no `stage` field. `core/architect.py:835-846` `_to_contract_issue` adapter drops the `stage="architect"` label from `core/verifier/types.VerifierIssue` (legacy dataclass). By the time `RefinementExhaustedError.issues` reaches `run_pipeline`, the stage field is gone — D-5's `i.stage == "architect"` AttributeError at runtime. | dispatcher-blocking | **ADOPTED.** Spec v2 §D-2 introduces a new helper `_issue_stage(issue) -> Optional[str]` in `core/orchestration/pipeline.py` that derives stage from `target` prefix (`"architect:..."`, `"specialist:..."`, `"synthesizer:..."`) — already encoded by every check at `checks_deterministic.py:24,38,67,91,118,145`. **No `VerifierIssue` schema change** — keeps blast radius minimal (no migration of 4 test files + `checks_semantic_d6_d7_d8.py` + `checks_semantic.py`). Validated against every existing call site that emits a contract `VerifierIssue` (target prefix populated). |
  | **C-2** | `dispatcher-ordering`: v1 D-5 enters `refine_until_clean(_re_run_specialist)` first, then partitions on `RefinementExhaustedError`. Architect-stage issues burn 2 wasted Specialist re-runs before exhaustion. Worse: an architect-stage issue from the *initial* verify call gets routed into specialist re-run before the dispatcher sees it. | correctness + efficiency | **ADOPTED.** Spec v2 §D-5 restructures the loop: pre-check verifier result *before* entering `refine_until_clean`. If architect-stage issues present → dispatch to architect re-run immediately. Specialist refine loop only entered when no architect-stage issues. Architect-stage issues that surface *during* specialist refine (e.g., from a downstream re-evaluation) are caught in the existing `RefinementExhaustedError` handler. |
  | **W-1** | `mixed-stage-success-untested`: T-DISPATCH-4 covers persistent architect failure but not the "architect re-run succeeds, specialist still has issues, specialist refine loop runs" flow. | test gap | **ADOPTED.** New T-DISPATCH-5 added (Test Plan §T-DISPATCH-5). Verifies: architect issue first; architect re-run produces clean architect output; specialist issue surfaces post-architect-fix; specialist refine loop runs once; final pipeline succeeds. |
  | **W-2** | `red-collection-math + import-collection-failure`: spec v1 oscillates between "355 collected" and "356 collected"; T-AGE-1's top-level import of `ArchitectGroundingError` would cause pytest collection failure on the RED commit (file-level ImportError), counted as 0 pass + collection-error, not 9 fail. | test-plan correctness | **ADOPTED.** Spec v2 §Test plan reconciles math to a single number: RED commit = `355 collected, 347 passed, 8 failed` (T-DEGRADE-LOG-1 modified asserts new behavior → flips from pass to fail; 7 new RED tests fail-by-assertion; T-LOG-2 passes-from-start). T-AGE-1 + T-INT-1 import the new exception **inside the test function body** using `pytest.importorskip("core.orchestration.errors", reason="ArchitectGroundingError not yet implemented")` — collection succeeds; assertion fails predictably. |
  | **W-3** | `feedback-test-too-weak`: T-FEEDBACK-1's "feedback substring" claim leaves field name ambiguous. Per D-4, format is `- {issue.target}: {issue.message}` (using contract field `target`, not `location`). | precision | **ADOPTED.** Spec v2 §D-4 specifies exact feedback block format. T-FEEDBACK-1 asserts: (a) `"PREVIOUS ATTEMPT FAILED VERIFICATION:"` header substring, (b) `issue.target` substring, (c) `issue.message` substring. |
  | **W-4** | `aGE-payload-vs-claim-mismatch`: T-DISPATCH-4 v1 claims "specialist issues kept for post-mortem visibility" but `ArchitectGroundingError(issues=arch_issues)` only carries architect issues. | spec internal consistency | **ADOPTED with payload widening.** Spec v2 §D-3 adds `residual_issues: List[VerifierIssue]` to `ArchitectGroundingError` for the **non-architect** issues observed at exhaustion. T-DISPATCH-4 asserts both `len(exc.issues) == N_architect` and `len(exc.residual_issues) == N_other`. Closes the visibility gap without losing taxonomy precision. |
  | **W-5** | `silent-degrade-retained`: v1 D-5 retains bare `except Exception` at `pipeline.py:98-112` as "defensive." Contradicts WP-CORE-7's explicit-failure mandate. | methodology integrity | **ADOPTED.** Spec v2 §D-5 **narrows** the catch to `except RefinementExhaustedError` only. Other exceptions (e.g., unexpected `Exception` from a buggy `deps.verifier`) propagate. Aligns with AGENTS.md "explicit failure" + EMSE methodology integrity. Smaller change than full removal but eliminates the silent-degrade path identified by Codex. |
  | **W-6** | `main.py-pipelineerror-claim-wrong`: v1 says "existing `try/except PipelineError` blocks catch it transparently." Reality (`main.py:77,180,194,211,226,410,427,518,533,721`): all bare `except Exception`. No `PipelineError` catch anywhere. | downstream-impact accuracy | **ADOPTED with correction (no scope expansion).** Spec v2 §Downstream-impact restates accurately: bare-Exception handlers at `main.py:533` and `main.py:427` catch `ArchitectGroundingError` transitively via `str(e)` serialization (preserves run-failure surfacing); no explicit `PipelineError` catch needed for WP-CORE-7. **Adding a typed PipelineError handler is out of WP-CORE-7 scope** (smallest correct change). Tracked as F-23 backlog entry instead. |
  | **N-1** | `feedback-per-internal-retry`: v1 doesn't specify whether the feedback block is prepended once per outer architect attempt or per each of the 5 internal JSON-parse retries inside `identify_contexts`. | precision | **ADOPTED.** Spec v2 §D-4 specifies: feedback is prepended **once** per outer architect attempt, then the resulting prompt is reused unchanged across all 5 internal JSON/shape retries (no per-internal-retry re-derivation). Mirrors how the base prompt is built once before `for retry in range(5)` at `architect.py:486`. |
  | **N-2** | `oq-4-d5-unreachable-not-indirect`: v1 OQ-4 said D5 synthesizer check is "covered indirectly." Reality: `ContextHypothesis` has no `allowed_dependencies` field (`pipeline_contracts.py:87-92`), so D5 (`checks_deterministic.py:135-153`) is unreachable in the current wiring. | rationale precision | **ADOPTED.** Spec v2 OQ-4 reworded: "D5 synthesizer-stage check is unreachable today (ContextHypothesis lacks the `allowed_dependencies` field D5 inspects). When a real synthesizer-stage check is added, its dispatch becomes a new WP scope." |
  | **OQ-1** | `a6-srs-path-unlocked`: WP-CORE-6 deferred A6-srs-path OQ with trigger "post-F-22." WP-CORE-7 closes F-22. Should be recorded as unlocked follow-up. | scope hygiene | **ADOPTED as out-of-scope follow-up note.** Spec v2 §Open Questions OQ-6 records that A6-srs-path (adding `srs_path` to `VerifierIssue` schema) is unlocked. **Not bundled** into WP-CORE-7 — keeps blast radius narrow. Backlog entry F-24 to be added in DOC commit. |

  **Codex disposition summary**: 2 CRITICAL all ADOPTED inline; 6 WARN all ADOPTED inline; 2 NIT confirmed and inlined; 1 OQ tracked as F-24 (post-F-22 follow-up). No items deferred without trigger.

---

## Motivation

### The half-fix this closes (WP-CORE-6 D-3 was "honest signal, not enforcement")

WP-CORE-6 added a non-empty clause to `check_d1_supporting_sentence_ids_subset` (`core/verifier/checks_deterministic.py:22-33`):

```python
ids = ctx.get("supporting_sentence_ids", [])
if not ids:
    issues.append(VerifierIssue(stage="architect", ..., severity=ERROR, ...))
    continue
```

This emits a `stage="architect"` ERROR when Architect under-grounds a context. WP-CORE-6 deliberately stopped short of enforcement: per spec §D-3 fold-in, D1 is documented as **"honest signal, not enforcement"**. The reason: Refiner's existing control loop (`core/refiner/loop.py`, `core/orchestration/pipeline.py:70-112`) only re-runs the **Specialist** stage. An architect-stage ERROR can never be corrected by re-running Specialist — Refiner exhausts cycles, raises `RefinementExhaustedError`, and the pipeline degrades to best-effort via `pipeline.py:82-97`.

The empirical impact: on every project run where the Architect produces a context the LLM can't ground in Scout's emitted indices, D1 emits an ERROR which is captured in the run manifest via WP-CORE-6 C-4 degrade-log enrichment, **then the pipeline ships the model anyway**. The EMSE methodology claim "D1 catches contexts citing un-emitted sentences" is currently honored at the verifier level but not at the pipeline level.

### Production reachability (loop discipline — mandatory subsection)

Per loop discipline (lesson from F-11 dormant-scope miss): every WP spec must include a production-reachability check before drafting the fix.

**F-22 status:** **LIVE in production.** The control flow that triggers this defect runs on every `analyze_document` call:

1. `DomainArchitect.analyze_document` builds the closure stack at `core/architect.py:867-880` — `architect_fn` calls `identify_contexts`.
2. `run_pipeline` (`core/orchestration/pipeline.py:60`) invokes `architect_fn` once, then calls `refine_until_clean` with `_re_run_specialist` as the sole stage runner (`pipeline.py:70-81`).
3. If the verifier surfaces ANY architect-stage issue (e.g., D1 `ungrounded_context`), `refine_until_clean` calls `_re_run_specialist` (which only re-runs Specialist), the verifier re-evaluates with the same Architect output, the architect-stage issue persists, and after `max_cycles=2` `RefinementExhaustedError` is raised.
4. `pipeline.py:82-97` catches the exception, emits the degrade-log (now WP-CORE-6 C-4 enriched), and continues with `refined_specialist = specialist_output`.
5. The pipeline ships a `DomainModel` despite a deterministic D1 ERROR.

Contrast with F-11 (dormant; parallel Scout never enabled by default) and F-14 (dormant; SynthesizerEmptyModelError fold-in covered the production path elsewhere). F-22 is comparable to F-21 (LIVE): every project run with a non-trivial SRS is a candidate for triggering this path, and the WP-CORE-6 honest-signal commit already exposes the ERRORs in the run manifest, confirming the path is hit empirically (see WP-CORE-6 dev_doc §"Empirical results").

### Why mode C (hybrid: 1 architect re-run + hard-fail)

Three modes were considered for the architect-stage handler:

- **Mode A** — hard-fail only: detect architect-stage issue post-refine, skip re-run, raise `ArchitectGroundingError`. Smallest correct change. Risk: an Architect that produces under-grounded contexts on the first attempt has zero chance of self-correction, even though the LLM is given the same prompt with different sampling.
- **Mode B** — issue-aware re-prompt only: loop architect_fn with feedback up to `max_cycles`. Genuine closed-loop control. Risk: LLM determinism asymmetry — feedback-injected re-prompt may not consistently fix; complexity creep; potential cost amplification (5 internal retries × N outer re-runs).
- **Mode C** — hybrid (chosen): one architect re-run with issue-aware feedback prompt; if still failing → raise `ArchitectGroundingError`. Bounded cost (max 5 internal retries × 2 attempts = 10 LLM calls worst case). Genuine self-correction shot. Explicit failure on persistent grounding violation. Matches AGENTS.md "explicit failure" + "smallest correct change".

---

## Discovery (audit-text-vs-code-reality)

### D-1. Backlog claim verified

**Claim** (backlog F-22): "Refiner orchestration only re-runs the Specialist stage (`pipeline.py:53-55` `_re_run_specialist`). Architect-stage verifier failures (e.g., D1 `ungrounded_context` ERROR) cannot be auto-corrected; pipeline degrades to best-effort via `RefinementExhaustedError` handler, silently shipping a model."

**Code reality (HEAD `4c8580c`):**

- `core/orchestration/pipeline.py:70-72`:

  ```python
  def _re_run_specialist(_prev, _result) -> List[SpecialistAnalysis]:
      # Phase C ships a simple re-run; Phase D wires issue-aware re-prompting.
      return deps.specialist(arch, scout)
  ```

  Confirms the runner re-runs Specialist only. The `arch` variable (line 60) is bound once; no architect re-run path exists.

- `core/refiner/loop.py:30-37`: `refine_until_clean` is single-stage by signature — takes one `stage_runner`. No dispatch by issue stage.

- `core/orchestration/pipeline.py:82-97`: `RefinementExhaustedError` is caught and the pipeline continues with `refined_specialist = specialist_output`. The exception's issues list is now (post-WP-CORE-6 C-4) logged, but not used for control flow.

- `core/verifier/checks_deterministic.py:24,38,67,91,118,145`: D1-D5 issues set `target=f"{stage}:..."` prefix. D1 + (architect-only): `target=f"architect:..."`. D2/D3/D4 + (specialist-only): `target=f"specialist:..."`. D5: `target=f"synthesizer:..."`. **Stage is derivable from target prefix.**

- `core/architect.py:835-846` `_to_contract_issue`: maps legacy `VerifierIssue` (with `stage` attr) to contract `VerifierIssue` (Pydantic, **no `stage` attr**). The legacy `stage` is dropped, but `target` retains the prefix.

- `core/pipeline_contracts.py:152-163` `VerifierIssue` (Pydantic): `severity` / `check_id` / `target` / `message`. No `stage`.

**Verdict:** backlog claim accurate. Fix scope = (a) extract stage from `target` prefix via helper, (b) dispatch by stage, (c) architect re-run runner with feedback injection, (d) raise hard-fail exception on exhaustion.

### D-2. New helper `_issue_stage` (no schema change to `VerifierIssue`)

```python
# core/orchestration/pipeline.py (new private helper)

from core.pipeline_contracts import VerifierIssue

_KNOWN_STAGE_PREFIXES = ("scout:", "architect:", "specialist:", "synthesizer:")

def _issue_stage(issue: VerifierIssue) -> Optional[str]:
    """Derive stage from the issue's `target` prefix.

    Every Verifier check (checks_deterministic.py D1-D5, checks_semantic_d6_d7_d8.py
    D6-D8, checks_semantic.py S1) populates `target` with a `"{stage}:..."` prefix.
    This helper keeps stage routing without widening the contract schema.

    Returns None if the prefix is unrecognized (forward-compat with future checks).
    """
    target = issue.target or ""
    for prefix in _KNOWN_STAGE_PREFIXES:
        if target.startswith(prefix):
            return prefix.rstrip(":")
    return None
```

**Rationale for not extending `VerifierIssue` schema (Codex C-1 disposition):** adding a `stage: Literal[...]` field would force updates at every `VerifierIssue(...)` call site (5 in `checks_semantic_d6_d7_d8.py`, 2 in `checks_semantic.py`, 4 test files). The target prefix already encodes the routing signal — using a derivation helper is the smallest correct change. **Invariant** documented in §Non-negotiables (DOC commit): "every `VerifierIssue.target` MUST be prefixed with `'{stage}:'`."

### D-3. New exception `ArchitectGroundingError`

`core/orchestration/errors.py` gains:

```python
from typing import Any, List, Optional

class ArchitectGroundingError(PipelineError):
    """Raised when Refiner's architect-stage re-run exhausts budget on
    persistent D1 (ungrounded_context) ERRORs.

    Distinct from ArchitectExtractionError:
      - ArchitectExtractionError = syntactic/extraction failure (no parseable JSON,
        empty contexts list)
      - ArchitectGroundingError = semantic grounding failure (well-formed contexts
        whose supporting_sentence_ids fail D1)

    Carries:
        srs_path: SRS being processed (or "<unknown>" if unset).
        issues: architect-stage VerifierIssue list at exhaustion.
        residual_issues: non-architect-stage issues observed at exhaustion
            (specialist/synthesizer issues that may have also been present).
            Preserves post-mortem visibility per Codex W-4 disposition.
        cycles_attempted: number of architect re-runs (1 in mode C hybrid).
    """

    def __init__(
        self,
        srs_path: str,
        issues: List[Any],
        residual_issues: List[Any],
        cycles_attempted: int,
        message: Optional[str] = None,
    ):
        self.srs_path = srs_path
        self.issues = issues
        self.residual_issues = residual_issues
        self.cycles_attempted = cycles_attempted
        super().__init__(
            message
            or (
                f"Architect re-run exhausted ({cycles_attempted} cycle(s)) "
                f"with {len(issues)} unresolved grounding issue(s) "
                f"({len(residual_issues)} non-architect residual) for srs={srs_path}"
            )
        )
```

LOC: ~30.

### D-4. `identify_contexts` accepts optional `feedback_issues` (feedback prepended ONCE per outer attempt)

`core/architect.py:419-588` `identify_contexts` gains an optional kwarg:

```python
def identify_contexts(
    self,
    domain_sentences: List[str],
    feedback_issues: Optional[List["VerifierIssue"]] = None,
) -> List[Dict[str, Any]]:
```

When `feedback_issues` is provided, prepend a feedback block to the prompt **before** the existing instruction block. Per Codex N-1 disposition: feedback is built **once** before the `for retry in range(5)` loop, then the same prompt is reused across all 5 internal JSON/shape retries (no per-internal-retry re-derivation).

**Exact feedback block format** (Codex W-3 disposition):

```
PREVIOUS ATTEMPT FAILED VERIFICATION:
The previous response was rejected because of the following grounding issues:
- {issue.target}: {issue.message}
- {issue.target}: {issue.message}
...

For this retry, ensure every context cites valid supporting_sentence_id
values that appear in the numbered list below.

```

Implementation: a small private function-level helper `_build_grounding_feedback_block(issues: List[VerifierIssue]) -> str` returning the prepended string or `""`. Prompt construction stays at line 459 with `feedback_block + main_prompt`.

LOC: ~25.

### D-5. Refiner stage dispatch in `pipeline.py` (pre-check architect issues BEFORE specialist refine loop)

`core/orchestration/pipeline.py:39-140` `run_pipeline` is restructured to dispatch by stage **before** entering the specialist refine loop (Codex C-2 disposition). The bare `except Exception` block is **narrowed** to `except RefinementExhaustedError` only (Codex W-5 disposition).

```python
def run_pipeline(*, srs_text, deps, srs_path=None) -> DomainModel:
    scout: ScoutOutput = deps.scout(srs_text)

    architect_attempts = 0
    architect_max_cycles = 1
    architect_feedback: Optional[List[VerifierIssue]] = None

    while True:
        # Stage 2 (with optional feedback on re-run)
        if architect_feedback is None:
            arch: ArchitectOutput = deps.architect(scout)
        else:
            arch = deps.architect_with_feedback(scout, architect_feedback)

        # Stage 3
        specialist_output: List[SpecialistAnalysis] = deps.specialist(arch, scout)

        snapshot: Dict[str, Any] = {
            "scout": scout, "architect": arch, "specialist": specialist_output,
        }

        # Pre-check: any architect-stage issues on the FIRST verify call?
        # Dispatch BEFORE entering specialist refine loop to avoid burning
        # 2 wasted specialist re-runs on architect-stage failures.
        initial_result = deps.verifier(snapshot)
        arch_issues = [i for i in initial_result.issues if _issue_stage(i) == "architect"]
        other_initial_issues = [i for i in initial_result.issues if _issue_stage(i) != "architect"]

        if initial_result.error_count() > 0 and arch_issues:
            if architect_attempts < architect_max_cycles:
                architect_attempts += 1
                architect_feedback = arch_issues
                _log_architect_rerun(arch_issues, architect_attempts)
                continue  # restart outer loop with new arch
            _log_architect_fail(arch_issues, architect_attempts, srs_path)
            raise ArchitectGroundingError(
                srs_path=srs_path or "<unknown>",
                issues=arch_issues,
                residual_issues=other_initial_issues,
                cycles_attempted=architect_attempts,
            )

        # No architect-stage issues → enter specialist refine loop.
        def _re_run_specialist(_prev, _result):
            return deps.specialist(arch, scout)

        try:
            refined_specialist, _cycles = refine_until_clean(
                stage_name="specialist",
                initial_output=specialist_output,
                stage_runner=_re_run_specialist,
                verifier=lambda s: deps.verifier({**snapshot, "specialist": s}),
                max_cycles=2,
            )
        except RefinementExhaustedError as exc:
            # Architect issues can surface AFTER specialist re-runs too
            # (e.g., re-evaluation reveals architect drift).
            late_arch_issues = [i for i in exc.issues if _issue_stage(i) == "architect"]
            late_other_issues = [i for i in exc.issues if _issue_stage(i) != "architect"]

            if late_arch_issues:
                if architect_attempts < architect_max_cycles:
                    architect_attempts += 1
                    architect_feedback = late_arch_issues
                    _log_architect_rerun(late_arch_issues, architect_attempts)
                    continue
                _log_architect_fail(late_arch_issues, architect_attempts, srs_path)
                raise ArchitectGroundingError(
                    srs_path=srs_path or "<unknown>",
                    issues=late_arch_issues,
                    residual_issues=late_other_issues,
                    cycles_attempted=architect_attempts,
                )

            # Specialist-only exhaustion: existing degrade-log path (preserves WP-CORE-6 C-4).
            _log_specialist_degrade(exc.issues)
            refined_specialist = specialist_output

        break  # exit outer while loop

    # Existing post-loop pre-call guard + synthesizer + post-call check (unchanged from WP-CORE-5b/-6).
    if not refined_specialist:
        raise SynthesizerEmptyModelError(
            input_summary="0 SpecialistAnalysis from upstream pipeline",
            srs_path=srs_path or "<unknown>",
        )
    model = deps.synthesizer(refined_specialist)
    if not model.bounded_contexts:
        raise SynthesizerEmptyModelError(
            input_summary="synthesizer returned 0 bounded contexts (bypassed Pydantic)",
            srs_path=srs_path or "<unknown>",
        )
    return model
```

**Removed: bare `except Exception` block at v1 pipeline.py:98-112.** Codex W-5 disposition. Only `RefinementExhaustedError` is caught; other exceptions propagate to the caller (preserves "explicit failure" mandate). This is a tightening of error handling, not a behavior change in normal paths.

**Log helpers (`_log_architect_rerun`, `_log_architect_fail`, `_log_specialist_degrade`)** are 3-5-line module-private functions. They preserve the WP-CORE-6 C-4 issues-list contract (each line emits `severity@stage:target: message`). Acceptable per AGENTS.md: each helper has a single responsibility (one log line shape), not a speculative abstraction.

LOC: ~80 (net; existing degrade-log block is rewritten into the dispatcher).

### D-6. `PipelineDeps` gains `architect_with_feedback`

`PipelineDeps` is widened with an additional callable:

```python
ArchitectWithFeedbackFn = Callable[
    [ScoutOutput, List["VerifierIssue"]], ArchitectOutput
]

@dataclass
class PipelineDeps:
    scout: ScoutFn
    architect: ArchitectFn
    architect_with_feedback: ArchitectWithFeedbackFn   # NEW
    specialist: SpecialistFn
    synthesizer: SynthesizerFn
    verifier: VerifierFn
```

`DomainArchitect.analyze_document` (`architect.py:867-951`) builds:

```python
def architect_with_feedback_fn(
    scout: ScoutOutput, issues: List[VerifierIssue],
) -> ArchitectOutput:
    sentence_texts = [s.text for s in scout.sentences]
    ctx_proposals = self.identify_contexts(
        sentence_texts, feedback_issues=issues,
    )
    contexts = [
        ContextHypothesis(
            context_name=c["name"],
            description=f"{c['name']} context",
            supporting_sentence_ids=c["supporting_sentence_ids"],
        )
        for c in ctx_proposals
    ]
    return ArchitectOutput(contexts=contexts)
```

LOC: ~15 (closure + dataclass field).

**Test fixture impact:** every existing test fixture that constructs `PipelineDeps(...)` must add the new keyword arg or use a `MagicMock`. Inventory (Codex R-3 risk):

- `tests/test_pipeline_orchestration.py:_make_typed_deps()` (line 28-74): add `architect_with_feedback=architect_with_feedback_fn` where the architect_fn is reused with feedback param accepted-but-ignored, OR provide a separate `architect_with_feedback_fn`. **Approach**: update `_make_typed_deps()` to also accept feedback; for tests that don't exercise the feedback path, the same architect_fn is reused.
- `tests/test_pipeline_orchestration.py:test_pipeline_invokes_refiner_when_verifier_finds_issues()` (line 99-162): same.
- `tests/test_pipeline_orchestration.py:test_pipeline_raises_synthesizer_empty_model_error_when_refiner_rerun_returns_empty()` (line 194-249): same.
- `tests/test_pipeline_orchestration.py:test_refiner_exhaustion_log_includes_issues_list` (line 282-307): being REPLACED via D-7.

GREEN commit must update these 4 fixture sites in one atomic move with the production code.

### D-7. Update T-DEGRADE-LOG-1 to expect new behavior

`tests/test_pipeline_orchestration.py:282-307` (T-DEGRADE-LOG-1) currently asserts the pipeline degrades gracefully on persistent architect-stage issues. Post-WP-CORE-7, architect-stage exhaustion **raises** instead. The test must be updated:

```python
def test_refiner_exhaustion_on_architect_stage_raises_grounding_error(capsys):
    """T-DEGRADE-LOG-1 (WP-CORE-7 update, Codex C-1+C-2): architect-stage
    issues no longer degrade silently. After 1 architect re-run with persistent
    issues, pipeline raises ArchitectGroundingError. Log still includes issue
    target + message (preserves WP-CORE-6 C-4 visibility contract on the
    hard-fail path)."""
    from core.orchestration.errors import ArchitectGroundingError
    deps = _make_typed_deps()
    bad_issue = VerifierIssue(
        severity="ERROR",
        check_id="D1",
        target="architect:contexts[OrderMgmt].supporting_sentence_ids",
        message="Context 'OrderMgmt' has no supporting_sentence_ids — cannot verify SRS grounding",
    )
    deps.verifier = lambda snapshot: VerifierResult(ok=False, issues=[bad_issue])

    with pytest.raises(ArchitectGroundingError) as exc_info:
        run_pipeline(srs_text="x", deps=deps)

    assert exc_info.value.cycles_attempted == 1
    assert len(exc_info.value.issues) == 1
    assert len(exc_info.value.residual_issues) == 0

    captured = capsys.readouterr().out
    assert "architect:contexts[OrderMgmt].supporting_sentence_ids" in captured
    assert "no supporting_sentence_ids" in captured
```

LOC: ~30 (test changes).

---

## Test plan

**RED commit expected pytest result:** **355 collected** (was 348; T-DEGRADE-LOG-1 modified counts as 1, 7 new tests added). **347 passed, 8 failed** (T-DEGRADE-LOG-1 flips from pass to fail under new assertion; 7 new RED-by-design tests fail; T-LOG-2 passes from start because it tests the unchanged specialist-degrade path which still works). Codex W-2 disposition: collection-error guard on T-AGE-1 and T-INT-1 import.

| # | name | file | what it asserts | RED expectation |
|---|---|---|---|---|
| T-AGE-1 | `test_architect_grounding_error_carries_srs_path_issues_cycles` | `test_orchestration_errors.py` | `ArchitectGroundingError("srs", issues=[...], residual_issues=[...], cycles_attempted=1)` exposes attrs + readable message | FAIL by assertion (import-guarded) — class doesn't exist; `pytest.importorskip("core.orchestration.errors", reason="...")` skips cleanly if needed, OR wraps in try/except. **Chosen**: import inside test body; `with pytest.raises(ImportError)`-style not used. Instead, the test imports inline `from core.orchestration.errors import ArchitectGroundingError` and pytest records this as a test failure (not collection error) because the import is inside the function body. |
| T-DISPATCH-1 | `test_pipeline_re_runs_architect_on_initial_architect_stage_issue` | `test_pipeline_orchestration.py` | Verifier returns 1 architect-stage issue on first call then ok on second; `architect_with_feedback` invoked exactly once; `architect_fn` invoked once (initial) | FAIL — current pipeline doesn't re-run architect; `architect_with_feedback` doesn't exist |
| T-DISPATCH-2 | `test_pipeline_raises_grounding_error_after_architect_rerun_exhaustion` | `test_pipeline_orchestration.py` | Verifier always returns architect-stage issue; `ArchitectGroundingError` raised with `cycles_attempted == 1`, `issues` len 1, `residual_issues` len 0 | FAIL — current degrades silently |
| T-DISPATCH-3 | `test_pipeline_specialist_stage_issue_does_not_re_run_architect` | `test_pipeline_orchestration.py` | Verifier returns specialist-stage issue twice then ok; specialist called twice; `architect_with_feedback` never invoked | FAIL — `architect_with_feedback` doesn't exist yet; GREEN passes |
| T-DISPATCH-4 | `test_pipeline_mixed_stage_issues_persistent_architect_raises_with_residuals` | `test_pipeline_orchestration.py` | Verifier returns BOTH architect+specialist issues always; pipeline raises `ArchitectGroundingError`; `exc.issues` contains 1 architect issue, `exc.residual_issues` contains 1 specialist issue | FAIL — no dispatch logic + ArchitectGroundingError doesn't exist |
| T-DISPATCH-5 | `test_pipeline_architect_rerun_succeeds_then_specialist_refine_runs` | `test_pipeline_orchestration.py` | Verifier: first call → architect issue; after architect re-run → specialist issue; after specialist re-run → ok. End result: model produced; architect called twice (initial + 1 feedback rerun); specialist called twice (post-rerun-architect + 1 specialist-refine rerun) | FAIL — current can't recover from architect issue |
| T-FEEDBACK-1 | `test_identify_contexts_prepends_feedback_block_when_issues_provided` | `test_architect_identify_contexts.py` | `identify_contexts(sentences, feedback_issues=[issue])` injects feedback block; prompt sent to `client.chat` contains: (a) `"PREVIOUS ATTEMPT FAILED VERIFICATION:"`, (b) `issue.target`, (c) `issue.message` | FAIL — kwarg doesn't exist |
| T-LOG-1 | `test_pipeline_architect_fail_log_includes_full_issue_list` | `test_pipeline_orchestration.py` | `ArchitectGroundingError` raise emits stdout line containing each issue's `target` + `message` (WP-CORE-6 C-4 contract preserved on hard-fail path) | FAIL — no architect-fail log |
| T-LOG-2 | `test_pipeline_specialist_degrade_log_still_includes_issues_list` | `test_pipeline_orchestration.py` | Specialist-only exhaustion still degrades and logs (WP-CORE-6 C-4 contract preserved) — guards specialist path regression | **PASS from start** (existing degrade-log behavior on specialist-only path is unchanged in GREEN) |
| T-INT-1 | `test_analyze_document_e2e_architect_grounding_error_surfaces` | `test_analyze_document_e2e.py` (NEW) | Full `DomainArchitect.analyze_document` with mocked LLM where Architect produces empty `supporting_sentence_ids` twice (initial + feedback rerun); `ArchitectGroundingError` propagates to caller with `srs_path` populated | FAIL — current path degrades |
| **T-DEGRADE-LOG-1** | renamed: `test_refiner_exhaustion_on_architect_stage_raises_grounding_error` (D-7 above) | `test_pipeline_orchestration.py` | Updated to expect `ArchitectGroundingError` and log contract per D-7 | **FLIP** (was passing on degrade; assertion changed to raise) |

**Total**: 8 fail (T-AGE-1, T-DISPATCH-1..5, T-FEEDBACK-1, T-INT-1, T-DEGRADE-LOG-1-flip — wait that's 8 not counting T-LOG-1; recount): T-AGE-1, T-DISPATCH-1, T-DISPATCH-2, T-DISPATCH-3, T-DISPATCH-4, T-DISPATCH-5, T-FEEDBACK-1, T-LOG-1, T-INT-1, T-DEGRADE-LOG-1-flip = **10 failing tests** (T-LOG-2 passes from start). Adjust: RED commit = 355 collected, 345 passed, 10 failed.

**Final RED math (Codex W-2 reconciled): 355 collected, 345 passed, 10 failed, 31 deselected.** GREEN turns all 10 green.

---

## Risks

| # | risk | mitigation |
|---|---|---|
| R-1 | Refactor of `run_pipeline` control flow touches the most-tested orchestration code. | Stage-by-stage RED tests + preserve existing specialist-only tests as regression contract (T-LOG-2). Specialist-degrade path semantics bit-for-bit preserved. |
| R-2 | Architect re-run produces new `arch` object; specialist must re-run with new contexts. | Loop structure naturally re-invokes `deps.specialist(arch, scout)` on next iteration after `architect_attempts += 1`. T-DISPATCH-1 + T-DISPATCH-5 explicitly verify specialist sees the new arch. |
| R-3 | `PipelineDeps` widening with `architect_with_feedback` is a breaking signature change for fixtures. | 4 fixture sites identified in D-6; updated in GREEN commit in lockstep. No production callers other than `architect.py:940-946`. |
| R-4 | LLM determinism: feedback re-prompt may not help if the LLM's grounding failure is systemic. | Bounded by `architect_max_cycles=1`; on persistent failure → `ArchitectGroundingError`. Hard-fail is the intended EMSE methodology behavior for un-groundable SRSs. |
| R-5 | Quota / cost amplification: 5 internal retries × 2 architect attempts = 10 LLM calls worst case. | Bounded by D1. Each architect re-run is rate-limited via existing `_wait_for_rate_limit`. Worst-case 10 calls within 6s rate-limit = ~60s additional latency on persistent-failure runs. Acceptable for methodology integrity. |
| R-6 | `_issue_stage` helper depends on `target` prefix convention. A future check that forgets the prefix routes incorrectly (returns None). | Add invariant to handoff §"Non-negotiables carried forward": "every `VerifierIssue.target` MUST be prefixed with `'{stage}:'`." Audit existing call sites in GREEN commit (12 sites in `checks_deterministic.py`, `checks_semantic.py`, `checks_semantic_d6_d7_d8.py` — all conform). |
| R-7 | Narrowed `except RefinementExhaustedError` (Codex W-5) means previously-silenced bugs now surface. | This is the intended methodology-integrity gain. Any newly-surfacing exception is a latent bug that was being hidden. Document in DOC commit. |
| R-8 | `main.py` bare-Exception handlers serialize `ArchitectGroundingError` via `str(e)`. | Acceptable per Codex W-6 disposition. New exception's `__str__` already includes srs_path + issue counts + cycle count. Caller surfaces meaningful error. Typed `PipelineError` handler tracked as F-23 backlog (out of WP-CORE-7 scope). |

---

## Open questions

| # | question | disposition |
|---|---|---|
| **OQ-1** | Should mixed-stage issues prioritize Architect over Specialist? | **ANSWERED in D-5.** Architect first: an architect-stage grounding failure invalidates the contexts that Specialist analyzed, so re-running Specialist on stale contexts is wasted. Architect failure is upstream-blocking. |
| **OQ-2** | Should `architect_max_cycles` be configurable? | **DEFERRED with rationale.** Hard-code N=1 per AGENTS.md "no speculative generalization." Add as F-25 backlog **only if** empirically single-shot feedback rarely succeeds. |
| **OQ-3** | Should `ArchitectGroundingError` be a subclass of `ArchitectExtractionError` for taxonomic kinship? | **NO.** Different failure modes (syntactic vs semantic). Both inherit from `PipelineError` — right granularity. |
| **OQ-4** | Should Synthesizer-stage issues (D5) get the same dispatch treatment? | **NO.** Per Codex N-2 disposition: D5 (`checks_deterministic.py:135-153`) inspects `context.allowed_dependencies` which doesn't exist on `ContextHypothesis` (`pipeline_contracts.py:87-92`). **D5 is unreachable today.** When a real synthesizer-stage check is added, its dispatch becomes a new WP. |
| **OQ-5** | Should the feedback prompt include the LLM's previous response? | **NO for v1.** Prompt token cost amplification. Verifier issue messages already include context name + reason. Revisit if empirically inadequate. |
| **OQ-6 (NEW)** | Should `srs_path` be threaded through `VerifierIssue`? | **DEFERRED as F-24 backlog (NEW).** Codex OQ-1 disposition: WP-CORE-6's A6-srs-path deferred OQ is now unlocked post-F-22. Adding `srs_path` to `VerifierIssue` requires schema widening + threading at 13 call sites. **Not bundled** into WP-CORE-7 — keeps blast radius narrow. |

---

## Atomic commit sequence

1. **RED commit** — `test(orchestration, architect): WP-CORE-7 red-phase tests for stage-aware Refiner + ArchitectGroundingError`
   - Add 9 new tests (T-AGE-1, T-DISPATCH-1..5, T-FEEDBACK-1, T-LOG-1, T-LOG-2, T-INT-1) + 1 modified (T-DEGRADE-LOG-1 flip).
   - Net pytest: 355 collected, 345 passed, 10 failed, 31 deselected.
   - Files: `tests/test_orchestration_errors.py` (+T-AGE-1), `tests/test_pipeline_orchestration.py` (+T-DISPATCH-1..5, +T-LOG-1, +T-LOG-2, T-DEGRADE-LOG-1 modified), `tests/test_architect_identify_contexts.py` (+T-FEEDBACK-1), `tests/test_analyze_document_e2e.py` (NEW, +T-INT-1)
   - Import-guard: T-AGE-1 + T-INT-1 import `ArchitectGroundingError` inside the test function body (not at module top) so pytest collection succeeds and the tests fail at assertion time, not collection time.
   - LOC: +~210

2. **GREEN commit** — `fix(orchestration, architect): WP-CORE-7 Refiner stage-aware dispatch + ArchitectGroundingError`
   - `core/orchestration/errors.py` — `ArchitectGroundingError` class with `residual_issues` (+~30 LOC)
   - `core/orchestration/pipeline.py` — outer architect-refine loop + `_issue_stage` + dispatcher + log helpers + narrowed exception catch (Codex W-5) (+~80 LOC; ~30 LOC rewrite of existing block)
   - `core/architect.py` — `identify_contexts(feedback_issues=...)` + `_build_grounding_feedback_block` helper + `architect_with_feedback_fn` closure + `PipelineDeps(architect_with_feedback=...)` (+~40 LOC)
   - 4 fixture sites in `tests/test_pipeline_orchestration.py` updated to include `architect_with_feedback=architect_fn` in `PipelineDeps(...)`
   - Pytest: 355 collected, all passing, zero regression vs 348 baseline.
   - LOC: +~150 production / +~30 test fixture updates

3. **DOC commit** — `chore(artifacts): WP-CORE-7 dev_doc + audit state update + F-22 SHIPPED + F-23/24 backlog`
   - `development_docs/WP-CORE-7-refiner-stage-aware.md` (created)
   - `development_docs/INDEX.md` (ACTIVE row #8 added)
   - `.planning/pipeline_audit/CURRENT.md` (iteration 6 SHIPPED status)
   - `.planning/pipeline_audit/improvements_backlog.md` (F-22 → SHIPPED with commit SHA; F-23 NEW = `main.py` typed PipelineError handler; F-24 NEW = `srs_path` in `VerifierIssue`)
   - `.planning/pipeline_audit/decision_log.md` (D-PICK-WP-CORE-7, D-CODEX-REVIEW-WP-CORE-7)
   - `.planning/pipeline_audit/findings/orchestrator.md` (F-22 SHIPPED status added)
   - `.planning/pipeline_audit/handoff-2026-05-23-<time>.md` (iteration 7 handoff)

4. **PLANNING commit** — `chore(planning): WP-CORE-7 spec v2 + plan into git history`
   - `docs/superpowers/specs/2026-05-22-wp-core-7-refiner-stage-aware-design.md` (v2 — this file)
   - `docs/superpowers/plans/2026-05-22-wp-core-7-refiner-stage-aware.md`

---

## Downstream impact

| concern | impact | action |
|---|---|---|
| `main.py:/generate-model` (lines 458-545) + `/validate` exception handlers | Bare `except Exception as e:` at line 533 + 427 catch `ArchitectGroundingError` transitively; `str(e)` serializes srs_path + issue counts + cycle count → run-failure message surfaces meaningfully. **No code change required for WP-CORE-7.** | Tracked as F-23 backlog (NEW): add typed `except PipelineError` handler for richer run-manifest signal. Out of WP-CORE-7 scope. |
| Run manifest (`runs/outputs/*.manifest.json`) | New error type appears in failure runs as `type(exc).__name__ = "ArchitectGroundingError"`. | None — manifest already serializes type name; no schema change. |
| EMSE paper Methods section | Pre-WP-CORE-7 claim "D1 catches contexts citing un-emitted sentences" was true at verifier level but degraded at pipeline level. Post-WP-CORE-7: "D1 ERRORs are enforced via one issue-aware re-prompt of Architect; persistent grounding violations raise `ArchitectGroundingError` and fail the run." | Flag for advisor at next review. Update `paper.tex` when iteration 7+ paper-revision WP runs. |
| WP-CORE-6 invariant compatibility | WP-CORE-6 invariant: "D1 verifier WILL flag empty IDs as `ungrounded_context` ERROR" — preserved. New invariant added: WP-CORE-7 elevates the flag to enforcement after 1 re-run. | Add to handoff §"Non-negotiables carried forward" in DOC commit. |
| F-15 (Refiner exhaustion fallback observability) | Partially closed for architect path (now raises with full issue list). Specialist path retains existing degrade-log. | Backlog: keep F-15 OPEN but downgrade severity to TRIVIAL since the architect-stage portion shipped. |
| `VerifierIssue.target` prefix convention (D-2 derives stage from this) | Becomes a load-bearing invariant for stage-based dispatch. | Add to handoff §"Non-negotiables carried forward": "every `VerifierIssue.target` MUST be prefixed with `'{stage}:'`." Codex C-1 disposition trade-off documented. |
| `except Exception` narrowing in `pipeline.py` (Codex W-5) | Previously-silenced exceptions now propagate. | Acceptable per methodology integrity. If a real bug surfaces post-GREEN, it gets a dedicated follow-up WP. |

---

## Goal-backward verification (spec-level)

| Iteration-6 goal | Evidence at spec-time |
|---|---|
| Pick F-22 per WP-CORE-6 handoff Codex W-4 / A6-f22 | F-22 picked; LIVE-in-production confirmed by Discovery D-1. |
| Spec → Codex xhigh review → plan → SDD → dev_doc → state update | Spec v1 drafted; Codex review returned 2 CRITICAL + 6 WARN + 2 NIT + 1 OQ; v2 incorporates all CRITICAL+WARN inline; 2 NIT inlined; 1 OQ recorded as F-24 follow-up. |
| Each commit gated on pytest ≥ baseline | Atomic commit sequence specifies RED-pytest delta (348 → 345 + 10 fail) and GREEN-pytest delta (→ 355 pass). |
| Production reachability subsection in spec §Motivation | YES — §Motivation includes "Production reachability (loop discipline)" subsection. |
| Honest-signal-to-enforcement upgrade | YES — D-5 dispatcher raises `ArchitectGroundingError` on architect-stage exhaustion; T-DISPATCH-2 + T-INT-1 + T-DEGRADE-LOG-1-flip verify. |
| Closes WP-CORE-6 D-3 half-fix | YES — D-3 documented as "honest signal, not enforcement" in WP-CORE-6; WP-CORE-7 promotes signal to enforcement. |
| Codex CRITICAL findings handled inline | YES — C-1 (stage-derive helper), C-2 (pre-check dispatch order). |
| Codex WARN findings handled inline | YES — W-1 (T-DISPATCH-5 added), W-2 (RED math reconciled to 355/345/10; import guarded), W-3 (T-FEEDBACK-1 exact format), W-4 (`residual_issues` payload), W-5 (narrowed exception catch), W-6 (main.py claim corrected, F-23 backlog). |
| Codex NIT findings handled inline | YES — N-1 (per-attempt feedback), N-2 (OQ-4 reworded). |
| Codex OQ tracked as follow-up | YES — OQ-1 (A6-srs-path) → F-24 backlog entry to be added in DOC commit. |

Spec v2 ready for execution. RED → GREEN → DOC → PLANNING per atomic commit sequence.
