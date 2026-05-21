# WP-CORE-5b — Implementation plan

**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-5b-synthesizer-empty-model-policy-design.md` (v2, post-Codex)
**Status:** READY for execution
**Baseline:** 332 passed, 31 deselected at HEAD `2b8602f`
**Target:** 338 passed, 31 deselected at HEAD `{green-sha}` (+6 tests)

---

## Task breakdown

### Task 1 — RED commit

**Commit type:** `test(orchestration)`
**Summary:** `WP-CORE-5b red-phase tests for SynthesizerEmptyModelError guard placement + srs_path propagation`
**Files touched:**

1. `extension/backend/tests/test_pipeline_orchestration.py` — APPEND 4 tests (T-EMPTY-1..T-EMPTY-4)
2. `extension/backend/tests/test_synthesizer_empty_model_error.py` — APPEND 2 tests (T-EMPTY-5..T-EMPTY-6)

**Test bodies — `test_pipeline_orchestration.py` additions:**

```python
def test_pipeline_raises_synthesizer_empty_model_error_when_specialist_returns_empty():
    """T-EMPTY-1: initial-empty Specialist DI path raises SynthesizerEmptyModelError,
    NOT pydantic.ValidationError. Verifies pre-call guard + PipelineError taxonomy."""
    from core.orchestration.errors import PipelineError, SynthesizerEmptyModelError
    deps = _make_typed_deps()
    deps.specialist = MagicMock(return_value=[])
    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert isinstance(exc_info.value, SynthesizerEmptyModelError)


def test_pipeline_synthesizer_not_invoked_when_specialist_empty():
    """T-EMPTY-2: pre-call guard short-circuits before deps.synthesizer is called."""
    from core.orchestration.errors import SynthesizerEmptyModelError
    deps = _make_typed_deps()
    deps.specialist = MagicMock(return_value=[])
    synth_mock = MagicMock()
    deps.synthesizer = synth_mock
    with pytest.raises(SynthesizerEmptyModelError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert synth_mock.call_count == 0, (
        "Pre-call guard must short-circuit before deps.synthesizer is invoked."
    )


def test_pipeline_raises_synthesizer_empty_model_error_when_refiner_rerun_returns_empty():
    """T-EMPTY-3 (Codex W-1): refiner-success-path edge — first Specialist call
    returns non-empty, verifier fails once, rerun returns [], verifier accepts.
    refined_specialist becomes []; pre-call guard raises SynthesizerEmptyModelError."""
    from core.orchestration.errors import SynthesizerEmptyModelError
    from core.pipeline_contracts import (
        ScoutOutput, ArchitectOutput, ContextHypothesis,
        SpecialistAnalysis, SectionedSentence, ChunkMetadata,
    )
    from core.schemas import Entity
    from core.verifier.types import VerifierResult, VerifierIssue, IssueSeverity

    specialist_calls = [0]

    def architect_fn(scout):
        return ArchitectOutput(contexts=[
            ContextHypothesis(context_name="OrderMgmt", description="x"),
        ])

    def specialist_fn(arch, scout):
        specialist_calls[0] += 1
        if specialist_calls[0] == 1:
            return [SpecialistAnalysis(
                context=arch.contexts[0],
                entities=[Entity(
                    name="Order",
                    description="An order.",
                    confidence=0.9,
                    justification="cited",
                    evidence_sentence_indices=[0],
                )],
            )]
        return []  # Rerun returns empty.

    verifier_calls = [0]

    def verifier_fn(snapshot):
        verifier_calls[0] += 1
        if verifier_calls[0] == 1:
            return VerifierResult(ok=False, issues=[VerifierIssue(
                stage="specialist",
                location="x",
                issue_type="t",
                severity=IssueSeverity.ERROR,
                message="m",
            )])
        return VerifierResult(ok=True, issues=[])

    deps = PipelineDeps(
        scout=lambda text: ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
        ),
        architect=architect_fn,
        specialist=specialist_fn,
        synthesizer=MagicMock(),  # Should never be invoked.
        verifier=verifier_fn,
    )

    with pytest.raises(SynthesizerEmptyModelError):
        run_pipeline(srs_text="x", deps=deps)


def test_pipeline_post_call_check_catches_injected_synthesizer_returning_empty_model():
    """T-EMPTY-4 (Codex W-3): belt-and-suspenders for injected synthesizers that
    bypass Pydantic via DomainModel.model_construct."""
    from core.orchestration.errors import SynthesizerEmptyModelError
    from core.schemas import DomainModel, ProjectMetadata

    deps = _make_typed_deps()

    def injected_synthesizer(analyses):
        # model_construct bypasses Pydantic validation, allowing empty bounded_contexts.
        return DomainModel.model_construct(
            project_name="Test",
            project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
            bounded_contexts=[],
            global_rules=None,
        )

    deps.synthesizer = injected_synthesizer

    with pytest.raises(SynthesizerEmptyModelError) as exc_info:
        run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert "bypassed Pydantic" in str(exc_info.value), (
        "Post-call check should emit a message distinguishing it from the pre-call guard."
    )
```

**Test bodies — `test_synthesizer_empty_model_error.py` additions:**

```python
def test_synthesizer_empty_model_error_carries_srs_path():
    """T-EMPTY-5 (Codex OQ-2): SynthesizerEmptyModelError must carry srs_path
    field and include it in str(err). Default is '<unknown>'."""
    err = SynthesizerEmptyModelError(
        input_summary="0 SpecialistAnalysis from upstream pipeline",
        srs_path="/abs/path/SRS.docx",
    )
    assert err.srs_path == "/abs/path/SRS.docx"
    assert "/abs/path/SRS.docx" in str(err)

    err_default = SynthesizerEmptyModelError(input_summary="x")
    assert err_default.srs_path == "<unknown>"
    assert "<unknown>" in str(err_default)


def test_synthesizer_empty_model_error_message_diagnostic():
    """T-EMPTY-6: error message must be diagnostic enough for support cases."""
    err = SynthesizerEmptyModelError(
        input_summary="0 SpecialistAnalysis from upstream pipeline",
        srs_path="/inputs/SRS.docx",
    )
    msg = str(err)
    assert "empty DomainModel" in msg
    assert "0 SpecialistAnalysis" in msg
    assert "/inputs/SRS.docx" in msg
```

**Imports to add** (top of `test_pipeline_orchestration.py`):
- `from unittest.mock import MagicMock` (already imported at line 4)
- `import pytest` (already imported at line 3)
- `from core.orchestration.errors import SynthesizerEmptyModelError` — append to existing import block at line 7-10

**Imports to add** (top of `test_synthesizer_empty_model_error.py`):
- `from core.orchestration.errors import SynthesizerEmptyModelError` (already imported at line 10)
- No new imports needed.

**Gate before commit:**
```bash
cd extension/backend
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=short -q tests/test_pipeline_orchestration.py tests/test_synthesizer_empty_model_error.py 2>&1 | tail -20
```
Expected: 5 failed (T-EMPTY-1..T-EMPTY-5), 1 passed (T-EMPTY-6 — string contents already correct in v1 constructor for the `"empty DomainModel"` substring; will need to verify post-srs_path-rename whether the assertion holds). If T-EMPTY-6 fails, it's because the v1 constructor doesn't include `srs_path` in the message — that's also valid RED.

Final RED expectation: **5 or 6 failed, the rest passed**. Either count is acceptable as RED signal.

Overall pytest after RED commit: **332 + 6 collected = 338; 332 + 1 passed = 333; 5 failed**.

**Commit message:**
```
test(orchestration): WP-CORE-5b red-phase tests for SynthesizerEmptyModelError guard placement + srs_path propagation

WP-CORE-5b red-phase. 6 new tests across 2 files:
  - tests/test_pipeline_orchestration.py: T-EMPTY-1..T-EMPTY-4 (pre-call
    guard, synthesizer-not-invoked, refiner-shrink-to-empty edge, injected-
    synthesizer post-call belt-and-suspenders)
  - tests/test_synthesizer_empty_model_error.py: T-EMPTY-5..T-EMPTY-6
    (srs_path field propagation, diagnostic message format)

Expected RED: 5 failing on current main (HEAD 2b8602f); GREEN commit
in this WP turns them green.

Codex xhigh review: 0 CRITICAL + 6 WARN + 3 NITS + 3 OQ all handled
inline in spec v2 (zero deferred — third consecutive zero-deferred
iteration matching WP-CORE-3 + WP-CORE-4 standard).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

### Task 2 — GREEN commit

**Commit type:** `fix(orchestration, architect)`
**Summary:** `WP-CORE-5b pre-call SynthesizerEmptyModelError guard + post-call belt-and-suspenders + srs_path propagation`
**Files touched:**

1. `extension/backend/core/orchestration/errors.py` — modify `SynthesizerEmptyModelError.__init__` to accept + carry `srs_path` (default `"<unknown>"`); update default message format
2. `extension/backend/core/orchestration/pipeline.py` — widen `run_pipeline` signature with `srs_path: Optional[str] = None`; add pre-call guard before `deps.synthesizer(...)`; rewrite post-call guard to emit the "bypassed Pydantic" message
3. `extension/backend/core/architect.py` — line 846, pass `srs_path=self._current_srs_path` to `run_pipeline`

**Gate before commit:**
```bash
cd extension/backend
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=short -q 2>&1 | tail -5
```
Expected: **338 passed, 31 deselected** (zero failures, zero regressions).

**Commit message:**
```
fix(orchestration, architect): WP-CORE-5b SynthesizerEmptyModelError pre-call guard + srs_path

Three-file change for F-14:
  - core/orchestration/errors.py: SynthesizerEmptyModelError now carries
    srs_path (default "<unknown>") and includes it in str(err). Matches
    WP-CORE-4 pattern for IntermediateSaveError.
  - core/orchestration/pipeline.py: pre-call guard raises Synthesizer-
    EmptyModelError when refined_specialist == [] (closes Codex W-1/W-2:
    refiner-rerun-to-empty edge + Specialist returns [] silently on empty
    contexts input). Post-call check retained per Codex W-3 as belt-and-
    suspenders for injected synthesizers that bypass Pydantic via
    model_construct. run_pipeline signature widened with srs_path kwarg.
  - core/architect.py: analyze_document forwards self._current_srs_path
    to run_pipeline.

Pre-WP behavior: empty refined_specialist raised pydantic.ValidationError
(escapes PipelineError taxonomy).
Post-WP behavior: raises SynthesizerEmptyModelError (subclass of Pipeline-
Error) with srs_path context — diagnostic for the failing SRS.

Net change: +31 LOC across 3 files (~22 are doc comments; code-only ~9
LOC). Baseline 332 → 338 (+6 tests; all WP-CORE-5b tests green).

Spec: docs/superpowers/specs/2026-05-21-wp-core-5b-synthesizer-empty-model-policy-design.md (v2)
Codex xhigh review: zero deferred (all 6 WARN + 3 NITS + 3 OQ handled inline).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

### Task 3 — DOC commit (artifacts)

**Commit type:** `chore(artifacts)`
**Summary:** `WP-CORE-5b dev_doc + audit state update`
**Files touched:**

1. `development_docs/WP-CORE-5b-synthesizer-empty-model-policy.md` — NEW. Sections:
   - Header: status / branch / commit SHAs (will fill at commit time) / spec + plan paths / TL;DR.
   - Motivation: F-14 dormant in production; taxonomy preservation for paper methodology.
   - Architectural decisions:
     1. Pre-call guard placement (Codex OQ-1).
     2. Post-call belt-and-suspenders retained (Codex W-3).
     3. `srs_path` symmetry with `IntermediateSaveError` (Codex OQ-2).
   - File-level changes: 3-row table.
   - Methodology applied: TDD with genuine-RED-fail (Codex OQ-5).
   - Empirical results: 332 → 338 tests, +6 tests, zero regression.
   - Limitations + follow-ups: F-21 (vacuous D1 verifier) explicitly queued as next iteration per Codex W-8.
   - Cross-references: `[[WP-CORE-4-intermediate-save-observability]]` (sibling pattern), `[[wp-core-5-abandoned]]` (predecessor in this iteration).
2. `development_docs/INDEX.md` — APPEND ACTIVE row #6 for WP-CORE-5b.
3. `.planning/pipeline_audit/CURRENT.md` — update last-action + next-pointer.
4. `.planning/pipeline_audit/improvements_backlog.md` — move F-14 to SHIPPED row with commit SHA + 1-line summary.
5. `.planning/pipeline_audit/findings/architect.md` — append §F-14 status note ("SHIPPED in WP-CORE-5b; pre-call guard + post-call belt-and-suspenders + srs_path; see `decision_log.md` D-CODEX-REVIEW-WP-CORE-5b").
6. `.planning/pipeline_audit/handoff-2026-05-21-{HHMM}.md` — NEW, iteration-4 → iteration-5 handoff.
7. `.planning/pipeline_audit/decision_log.md` — append `D-CODEX-REVIEW-WP-CORE-5b` summary (already partially written in earlier session step; verify + complete).

**Commit message:**
```
chore(artifacts): WP-CORE-5b dev_doc + audit state update

  - development_docs/WP-CORE-5b-synthesizer-empty-model-policy.md (new)
  - development_docs/INDEX.md ACTIVE row #6
  - .planning/pipeline_audit/CURRENT.md next-pointer to iteration 5
  - .planning/pipeline_audit/improvements_backlog.md F-14 → SHIPPED
  - .planning/pipeline_audit/findings/architect.md §F-14 status
  - .planning/pipeline_audit/decision_log.md D-CODEX-REVIEW-WP-CORE-5b
  - .planning/pipeline_audit/handoff-2026-05-21-{HHMM}.md (new)

Iteration-5 recommendation: F-21 (vacuous D1 verifier pass) per Codex
W-8 priority bump — affects every project run, not just dormant DI paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

### Task 4 — PLANNING commit

**Commit type:** `chore(planning)`
**Summary:** `WP-CORE-5b spec v2 + plan into git history`
**Files touched:**

1. `docs/superpowers/specs/2026-05-21-wp-core-5b-synthesizer-empty-model-policy-design.md` — already exists; staged for commit.
2. `docs/superpowers/specs/2026-05-21-wp-core-5-parallel-scout-rate-limit-design.md` — already exists with ABANDONED banner; staged for commit.
3. `docs/superpowers/plans/2026-05-21-wp-core-5b-synthesizer-empty-model-policy.md` — this file; staged for commit.

**Commit message:**
```
chore(planning): WP-CORE-5b spec v2 + plan into git history

WP-CORE-5b artifacts (Codex-reviewed, zero-deferred).
WP-CORE-5 (abandoned) preserved with banner for audit trail.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Goal-backward verification

| Iteration-4 goal | Evidence (anticipated) |
|---|---|
| Pick a MAJOR finding per handoff-2026-05-21-1033 §"Recommended next iteration" — pivot to F-14 after WP-CORE-5 abandon | F-14 chosen; WP-CORE-5b shipped; F-11 marked DORMANT in backlog; pivot rationale recorded in decision_log D-PICK-WP-CORE-5b. |
| Each commit gated on pytest ≥ baseline (sole exception: RED commit accepts known-failing tests by design) | RED expected: 333 passed + 5 failed = 338 collected. GREEN expected: 338 passed. Artifacts commits: 338 passed (no test changes). |
| Spec → Codex xhigh review → plan → SDD → dev_doc → state update | Spec v1 → review → v2 → this plan → SDD via RED/GREEN commits → dev_doc + state in DOC commit. |
| No `git push` | Confirmed; main remains local. |
| Atomic Conventional Commits + Claude trailer | All 4 commits conform per messages above. |
| Zero deferred WARNs (matching iteration-2 + iteration-3's standard) | Confirmed; 6 WARN + 3 NITS + 3 OQ all handled inline in spec v2. Third consecutive zero-deferred iteration. |

---

## Post-iteration handoff outline (to be written at task 3)

**Next iteration target (iteration 5):** F-21 — vacuous D1 verifier pass (Codex W-8 bumped priority; affects every project run methodologically).

**Backlog post-WP-CORE-5b:**
- Ingestion: 2 SHIPPED + 4 MAJOR-OPEN + 3 MINOR-OPEN + 1 TRIVIAL-OPEN = 8 OPEN
- Orchestrator: 2 SHIPPED (F-13 + F-14) + 2 MAJOR-OPEN (F-11 DORMANT, F-21) + 6 MINOR-OPEN + 1 TRIVIAL-OPEN = 9 OPEN
- Total OPEN MAJOR (live): 5 (F-1, F-2, F-4-uncertain, F-21) — F-11 DORMANT counted separately

**Baseline post-iteration:** 338 passed, 31 deselected.
