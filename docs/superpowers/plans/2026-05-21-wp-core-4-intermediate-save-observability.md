# WP-CORE-4 Intermediate-Save Observability — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate F-13 (silent I/O swallow in `_save_intermediate`) + fold-in the `_current_srs_path` anomaly (4 reads, 0 writes — all `ArchitectExtractionError` instances say "<unknown>"). Both are observability fixes binding at the error-message level: new `IntermediateSaveError(PipelineError)` carries `srs_path` populated from `self._current_srs_path` at raise time. Architect's `identify_contexts` retry handler explicitly re-raises `IntermediateSaveError` so save failures aren't silently rewrapped as `ArchitectExtractionError`.

**Architecture:** One new exception class in `core/orchestration/errors.py` (`IntermediateSaveError(PipelineError)`); two edits in `core/architect.py` (`_save_intermediate` rewrite + `identify_contexts` re-raise guard + `__init__` attribute init + `analyze_document` signature widen with unconditional assignment); three call-site updates in `main.py` (lifespan single-file, `/generate-model` batch, `/generate-model-stream` batch). Tests: 5 T-SAVE-* + 4 T-SRS-* + 3 T-WIRE-MAIN-* = **11 new tests** across (new) `tests/test_intermediate_save.py` + `tests/test_architect_helpers.py` (or new sub-file) + `tests/test_main_wiring.py`.

**Tech Stack:** Python 3.13 (local dev), Python 3.12 (CI), `pytest` `-m "not integration"`, `monkeypatch` fixture, `tmp_path` fixture, `unittest.mock` for thread-aware kwarg capture, `pyright` strict on `core/`.

**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md` (v2 — Codex xhigh reviewed; 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ all handled)
**Audit findings:** `.planning/pipeline_audit/findings/architect.md` F-13 (MAJOR) + Anomalies §1 (`_current_srs_path` never set)
**Pre-WP HEAD:** `d7dc188`
**Pre-WP pytest baseline:** 321 passed, 31 deselected
**Target pytest baseline post-GREEN:** 332 passed, 31 deselected (11 new tests)

---

## File Structure

| file | role | change type |
|---|---|---|
| `extension/backend/core/orchestration/errors.py` | P3 pipeline error taxonomy. Add `IntermediateSaveError(PipelineError)`. | MODIFY (~+15 LOC: new class + docstring) |
| `extension/backend/core/architect.py` | DomainArchitect orchestrator. Rewrite `_save_intermediate`; add re-raise guard in `identify_contexts`; init `_current_srs_path` in `__init__`; widen `analyze_document` signature + unconditional assignment. Import `IntermediateSaveError` from `core.orchestration.errors`. | MODIFY (~+20 LOC net; +import, +__init__ line, +1 analyze_document line, +3 lines in _save_intermediate, +2 lines in identify_contexts; -3 lines old swallow body) |
| `extension/backend/main.py` | FastAPI entrypoint. 3 callsites pass srs_path to architect. | MODIFY (3 lines, +~30 chars each) |
| `extension/backend/tests/test_intermediate_save.py` | NEW — F-13 raise-path tests (T-SAVE-1..5). | CREATE (~150 LOC) |
| `extension/backend/tests/test_architect_srs_path.py` | NEW — `_current_srs_path` propagation tests (T-SRS-1..4). | CREATE (~130 LOC) |
| `extension/backend/tests/test_main_wiring.py` | EXISTING — append T-WIRE-MAIN-2 + T-WIRE-MAIN-3 (T-WIRE-MAIN-1 was created in WP-CORE-3; here extended for srs_path forwarding). | MODIFY (~+100 LOC) |
| `development_docs/WP-CORE-4-intermediate-save-observability.md` | Persistent dev memory for this WP. | CREATE |
| `development_docs/INDEX.md` | Dev-doc status board. | MODIFY (new ACTIVE row) |
| `.planning/pipeline_audit/improvements_backlog.md` | Audit backlog. | MODIFY (F-13 OPEN → SHIPPED; add F-21 OPEN; downgrade F-20 to MINOR) |
| `.planning/pipeline_audit/CURRENT.md` | Audit pointer. | MODIFY (last action + next iteration recommendation) |
| `.planning/pipeline_audit/decision_log.md` | Decision history. | ALREADY APPENDED (D-CL2 + D-PICK-WP-CORE-4 + D-CODEX-REVIEW-WP-CORE-4 entries written before plan-phase). |
| `.planning/pipeline_audit/handoff-2026-05-21-{HHMM}.md` | Iteration-3 handoff for next coordinator. | CREATE |

---

## Pre-flight checks

- [ ] Verify HEAD == `d7dc188` (`git rev-parse HEAD`).
- [ ] Verify pytest baseline: `cd extension/backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=no -q 2>&1 | tail -3` → `321 passed, 31 deselected`.
- [ ] Verify spec v2 is current (`grep "Status:" docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md` → "REVISED v2").
- [ ] Verify `IntermediateSaveError` does NOT yet exist (`grep -rn "IntermediateSaveError" extension/backend/` → no results).
- [ ] Verify `analyze_document` signature is current (`grep -n "def analyze_document" core/architect.py` → 1 result with `text: str` only).
- [ ] Verify `_current_srs_path` has zero assignment sites (`grep -nE "_current_srs_path[[:space:]]*=" core/architect.py` → no results).

---

## Step 1 — RED commit: failing tests for F-13 + `_current_srs_path` propagation

**Commit:** `test(architect, orchestration): WP-CORE-4 red-phase tests for save observability + srs_path propagation`

### 1.1 — Create `tests/test_intermediate_save.py` (T-SAVE-1..5)

**T-SAVE-1 — happy path:**
- [ ] Construct `DomainArchitect()` with env var `GEMINI_API_KEY` set (via `tmp_path`-based env patch).
- [ ] Call `arch._save_intermediate("test_stage_happy", {"x": 1, "list": [1, 2, 3]})`.
- [ ] Read the written file from `INTERMEDIATE_DIR` matching `*_test_stage_happy.json`.
- [ ] Assert `json.load(file) == {"x": 1, "list": [1, 2, 3]}`.
- [ ] Cleanup: delete the test file (or rely on test isolation).

**T-SAVE-2 — filesystem failure raises `IntermediateSaveError`:**
- [ ] Use `monkeypatch.setattr("core.architect.open", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("read-only")))` (or `mock.patch` against `builtins.open`). Either approach works; pick the lighter one.
- [ ] Call `_save_intermediate("test_stage_fail", {"x": 1})`.
- [ ] Assert `pytest.raises(IntermediateSaveError) as exc_info`.
- [ ] Assert `exc_info.value.stage == "test_stage_fail"`.
- [ ] Assert `isinstance(exc_info.value.cause, PermissionError)`.
- [ ] Assert `isinstance(exc_info.value, PipelineError)` — taxonomy lock per W-1.
- [ ] Assert `exc_info.value.srs_path == "<unknown>"` (architect was constructed but `analyze_document` was not called — `__init__` default applies).

**T-SAVE-3 — non-serializable data raises `IntermediateSaveError`:**
- [ ] Call `_save_intermediate("test_stage_typeerror", {"obj": object()})`.
- [ ] Assert `pytest.raises(IntermediateSaveError) as exc_info`.
- [ ] Assert `isinstance(exc_info.value.cause, TypeError)`.

**T-SAVE-4 — failure inside `identify_contexts` propagates `IntermediateSaveError`, NOT `ArchitectExtractionError` (per C-2):**
- [ ] Construct `DomainArchitect()`.
- [ ] Patch `arch._save_intermediate` to raise `IntermediateSaveError(stage="2_architect", filepath="/fake", cause=PermissionError("ro"), srs_path="<test>")` directly. (Bypass actual open/dump.)
- [ ] Patch the LLM client (`arch.client.generate_content` or equivalent) to return a valid JSON response containing a `contexts` list. Reuse the mock-LLM pattern from `tests/test_architect_extraction_error.py`.
- [ ] Patch `arch._wait_for_rate_limit` to a no-op.
- [ ] Call `arch.identify_contexts(["sample sentence one.", "sample sentence two."])`.
- [ ] Assert `pytest.raises(IntermediateSaveError)`.
- [ ] Explicit anti-assertion: `not isinstance(exc, ArchitectExtractionError)`.

**T-SAVE-5 — failure inside `extract_per_context_details` propagates `IntermediateSaveError` cleanly (Specialist path):**
- [ ] Same shape as T-SAVE-4 but exercise Specialist via `arch.extract_per_context_details(["CtxA"], ["sentence one.", "sentence two."])` with the LLM client mocked to return valid per-context payloads.
- [ ] Patch `arch._save_intermediate` to raise `IntermediateSaveError(stage="3_specialist", ...)`.
- [ ] Assert `pytest.raises(IntermediateSaveError)`. Since the save call is OUTSIDE the per-context retry loop, no re-raise guard is needed; the assertion confirms clean propagation.

### 1.2 — Create `tests/test_architect_srs_path.py` (T-SRS-1..4)

**T-SRS-1 — `analyze_document(srs_path=...)` assigns attribute:**
- [ ] Construct `DomainArchitect()`. Assert `arch._current_srs_path == "<unknown>"` immediately (from `__init__`).
- [ ] Patch `run_pipeline` (in `core.orchestration.pipeline`) to return a stub `DomainModel()` so `analyze_document` completes without real LLM calls. Pattern: reuse `tests/test_architect_facade.py:11-46`'s mock approach.
- [ ] Call `arch.analyze_document(text="some srs text", srs_path="/path/to/srs.docx")`.
- [ ] Assert `arch._current_srs_path == "/path/to/srs.docx"`.

**T-SRS-2 — `analyze_document(srs_path=None)` resets to `<unknown>`:**
- [ ] Same setup as T-SRS-1.
- [ ] Call `arch.analyze_document(text="...")` (no srs_path).
- [ ] Assert `arch._current_srs_path == "<unknown>"`.

**T-SRS-3 — `ArchitectExtractionError` carries assigned path:**
- [ ] Construct `DomainArchitect()`. Manually set `arch._current_srs_path = "/p/foo.docx"`.
- [ ] Patch LLM client to return a JSON-parse-failure response 5 times in a row.
- [ ] Patch `_wait_for_rate_limit` to no-op.
- [ ] Patch `_save_intermediate` to no-op (so the save step never runs — we're testing the exhaustion path, not save).
- [ ] Call `arch.identify_contexts([...])` and expect `pytest.raises(ArchitectExtractionError) as exc_info`.
- [ ] Assert `exc_info.value.srs_path == "/p/foo.docx"` (NOT `"<unknown>"`).

**T-SRS-4 — instance reuse resets path (per W-2):**
- [ ] Construct `DomainArchitect()`.
- [ ] Patch `run_pipeline` as in T-SRS-1.
- [ ] Call `arch.analyze_document(text="...", srs_path="/p/A.docx")`. Assert `arch._current_srs_path == "/p/A.docx"`.
- [ ] Call `arch.analyze_document(text="...")` (no srs_path this time).
- [ ] Assert `arch._current_srs_path == "<unknown>"`, NOT `"/p/A.docx"`. **This catches the stale-path-on-reuse bug Codex W-2 flagged.**

### 1.3 — Append to `tests/test_main_wiring.py` (T-WIRE-MAIN-2 + T-WIRE-MAIN-3)

**T-WIRE-MAIN-2 — `/generate-model` forwards joined srs_path (per OQ5):**
- [ ] Set up the same monkeypatched `SRSDocumentParser` from existing T-WIRE-* (in `tests/test_main_wiring.py`) that returns non-empty text per path.
- [ ] Monkeypatch `main.DomainArchitect` constructor to return a stub whose `analyze_document(text=..., srs_path=...)` records both kwargs into a list/holder.
- [ ] Monkeypatch `main.ASTModelSignalExtractor` to a no-op shim (just returns the input model).
- [ ] Construct a `GenerateModelRequest(file_paths=["/tmp/a.docx", "/tmp/b.docx"])`.
- [ ] Call the endpoint function directly: `await main.generate_model_endpoint(request)`. **Note:** the endpoint may be sync. If async, use `asyncio.run()` or pytest-async.
- [ ] Assert the captured `srs_path == "/tmp/a.docx; /tmp/b.docx"`.

**T-WIRE-MAIN-3 — `/generate-model-stream` forwards joined srs_path:**
- [ ] Same shape as T-WIRE-MAIN-2 but exercise `generate_model_stream_endpoint`.
- [ ] **Thread synchronization needed:** the streaming endpoint dispatches a worker thread. Use a `threading.Event` (or `queue.Queue`) inside the analyze_document stub to signal capture, then `Event.wait(timeout=5.0)` in the test before asserting.
- [ ] **Fallback if too fragile:** mark the streaming test with `pytest.skip` if thread sync proves unreliable; T-WIRE-MAIN-2's signature-test gives enough coverage of the joining behavior. Confirm with reviewer before skipping.

### 1.4 — Run RED suite, expect 11 failures

- [ ] `cd extension/backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=no -q 2>&1 | tail -8`
- [ ] Expected: `11 failed, 321 passed, 31 deselected` (or close — count drift ≤ 1 acceptable if test refactors merge two assertions; explicitly mark in commit message if so).
- [ ] Verify the failure reasons are the **right** kind: `IntermediateSaveError` is `NameError: not defined`, `TypeError: analyze_document() got an unexpected keyword argument 'srs_path'`, etc. — NOT logic errors in the test setup.

### 1.5 — Commit RED

- [ ] `git add extension/backend/tests/test_intermediate_save.py extension/backend/tests/test_architect_srs_path.py extension/backend/tests/test_main_wiring.py`
- [ ] Commit message:

```
test(architect, orchestration): WP-CORE-4 red-phase tests for save observability + srs_path propagation

Adds 11 failing tests for WP-CORE-4 (F-13 + _current_srs_path anomaly).
All tests expected to fail until GREEN commit lands the production code:
- IntermediateSaveError class in core/orchestration/errors.py
- _save_intermediate raise-on-failure in core/architect.py
- except IntermediateSaveError re-raise in identify_contexts retry handler
- _current_srs_path init in __init__
- analyze_document(srs_path=None) signature + unconditional assignment
- 3 main.py callsites passing srs_path

Spec: docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md (v2)
Codex review: 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ, all handled inline.

Test inventory:
- T-SAVE-1..5 (tests/test_intermediate_save.py) — F-13 raise-path coverage
- T-SRS-1..4 (tests/test_architect_srs_path.py) — _current_srs_path propagation
- T-WIRE-MAIN-2..3 (tests/test_main_wiring.py) — endpoint wiring per Codex OQ5

Baseline: 321 passed pre-RED → 321 passed + 11 failed during RED.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

- [ ] Run `git status` to confirm nothing else staged.
- [ ] `git commit` (no `--no-verify`).
- [ ] Verify commit landed: `git log -1 --oneline`.

---

## Step 2 — GREEN-atomic commit: production code + tests now pass

**Commit:** `fix(architect, main, orchestration): WP-CORE-4 IntermediateSaveError + srs_path propagation`

### 2.1 — Add `IntermediateSaveError` to `core/orchestration/errors.py`

- [ ] Read existing `errors.py` (70 LOC; classes `PipelineError`, `ArchitectExtractionError`, `SpecialistFailureError`, `SynthesizerEmptyModelError`).
- [ ] Append after the last existing class:

```python
class IntermediateSaveError(PipelineError):
    """Raised when stage diagnostic JSON cannot be persisted to INTERMEDIATE_DIR.

    Per AGENTS.md "no silent degradation": stage diagnostic artifacts are
    EMSE reproducibility evidence; silent loss is a methodology gap.

    Carries:
        stage: the pipeline stage label (e.g., "2_architect", "3_specialist").
        filepath: the intended on-disk path.
        cause: the wrapped exception (OSError / TypeError / ValueError).
        srs_path: the SRS being processed (or "<unknown>" if pre-analyze_document).
    """

    def __init__(
        self,
        stage: str,
        filepath: str,
        cause: Exception,
        srs_path: str = "<unknown>",
    ):
        self.stage = stage
        self.filepath = filepath
        self.cause = cause
        self.srs_path = srs_path
        super().__init__(
            f"Failed to save intermediate output for stage='{stage}' "
            f"at '{filepath}' (srs={srs_path}): {type(cause).__name__}: {cause}"
        )
```

### 2.2 — Update `core/architect.py` imports

- [ ] At the existing `from core.orchestration.errors import ...` block, add `IntermediateSaveError` to the import list.

### 2.3 — Rewrite `_save_intermediate` (lines 880-891)

- [ ] Replace body with:

```python
def _save_intermediate(self, stage: str, data: Dict[str, Any]):
    """Save intermediate pipeline output to JSON file.

    Raises IntermediateSaveError on any I/O or serialization failure.
    Per AGENTS.md "no silent degradation": stage diagnostic artifacts are
    EMSE reproducibility evidence; silent loss is a methodology gap.
    """
    filename = f"{self.run_timestamp}_{stage}.json"
    filepath = os.path.join(INTERMEDIATE_DIR, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except (OSError, TypeError, ValueError) as e:
        raise IntermediateSaveError(
            stage=stage,
            filepath=filepath,
            cause=e,
            srs_path=getattr(self, "_current_srs_path", "<unknown>"),
        ) from e
    print(f"  💾 Saved intermediate output: {filename}")
```

### 2.4 — Add `except IntermediateSaveError: raise` to `identify_contexts` retry handler

- [ ] Locate `except ArchitectExtractionError: raise` (around line 483).
- [ ] Insert between it and `except Exception as e:`:

```python
except IntermediateSaveError:
    raise  # WP-CORE-4 C-1: never silently rewrap save failures as ArchitectExtractionError
```

### 2.5 — Initialize `_current_srs_path` in `__init__`

- [ ] Locate the existing `self.run_timestamp = time.strftime(...)` line in `__init__`.
- [ ] Add immediately after (or after the `os.makedirs(INTERMEDIATE_DIR, exist_ok=True)` line):

```python
self._current_srs_path: str = "<unknown>"  # set by analyze_document(srs_path=...)
```

### 2.6 — Widen `analyze_document` signature + unconditional assignment

- [ ] Replace `def analyze_document(self, text: str) -> DomainModel:` (line 709) with:

```python
def analyze_document(
    self,
    text: str,
    srs_path: Optional[str] = None,
) -> DomainModel:
```

- [ ] Update the docstring to document the new kwarg.
- [ ] Add at the function start (after the docstring, before the existing import-from-core.synthesizer line):

```python
# Per W-2: unconditional assignment guards against stale path on instance reuse.
self._current_srs_path = srs_path or "<unknown>"
```

### 2.7 — Update 3 main.py callsites

- [ ] **`main.py:107`** — change `architect.analyze_document(text=raw_text)` → `architect.analyze_document(text=raw_text, srs_path=srs_path)`.
- [ ] **`main.py:362`** — change `architect.analyze_document(text=combined_text)` → `architect.analyze_document(text=combined_text, srs_path="; ".join(str(p) for p in request.file_paths))`.
- [ ] **`main.py:473`** — same as `:362`, inside the streaming endpoint.

### 2.8 — Run full pytest suite

- [ ] `cd extension/backend && /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=short 2>&1 | tail -20`
- [ ] Expected: `332 passed, 31 deselected` (or `≥ 332` if RED's count drift was tolerated).
- [ ] If any tests fail: STOP. Diagnose. Do NOT commit.

### 2.9 — Optional: pyright check

- [ ] `cd extension/backend && pyright core/architect.py core/orchestration/errors.py main.py 2>&1 | tail -10`
- [ ] If new type errors introduced: fix before commit (e.g., missing `Optional` import, type annotation mismatch).
- [ ] Pre-existing type errors are tolerated (CI's pyright step is `continue-on-error: true` per CLAUDE.md).

### 2.10 — Commit GREEN

- [ ] `git add extension/backend/core/orchestration/errors.py extension/backend/core/architect.py extension/backend/main.py`
- [ ] Also re-stage the test files (if any were modified during GREEN debugging).
- [ ] Commit message:

```
fix(architect, main, orchestration): WP-CORE-4 IntermediateSaveError + srs_path propagation

Closes F-13 (MAJOR — silent I/O swallow in _save_intermediate) and folds in
the _current_srs_path anomaly (4 reads, 0 writes — all ArchitectExtractionError
messages said "<unknown>"). Both observability fixes interlock via the new
error message format.

Changes:
- core/orchestration/errors.py: new IntermediateSaveError(PipelineError)
  carrying stage, filepath, cause, srs_path.
- core/architect.py:
  - _save_intermediate raises IntermediateSaveError on (OSError, TypeError,
    ValueError) instead of the prior except-Exception swallow.
  - identify_contexts retry handler adds `except IntermediateSaveError: raise`
    before `except Exception` (Codex C-1: prevents silent rewrap into
    ArchitectExtractionError after 5 fake-retries).
  - __init__ initializes self._current_srs_path = "<unknown>".
  - analyze_document(text, srs_path=None) — unconditional reassignment per W-2
    guards against stale path on instance reuse.
- main.py: 3 callsites pass srs_path to architect (single path for lifespan,
  "; "-joined for /generate-model and /generate-model-stream batch endpoints).

Behavior delta (R-2): a _save_intermediate failure in production now aborts
the run with IntermediateSaveError instead of silently completing with
corrupt diagnostics. Per AGENTS.md "no silent degradation" the prior
behavior was wrong; this surfaces the failure where it always belonged.

Spec: docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md (v2)
Codex review: 2 CRITICAL + 5 WARN + 3 NITS + 7 OQ, all handled inline.
Baseline: 321 → 332 (+11 tests).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

- [ ] Run `git status` to confirm only the intended files are staged + the test files from RED already committed.
- [ ] `git commit`.
- [ ] Verify commit landed: `git log -2 --oneline`.

---

## Step 3 — DOC commit: dev_doc + audit state update

**Commit:** `chore(artifacts): WP-CORE-4 dev_doc + audit state update`

### 3.1 — Write `development_docs/WP-CORE-4-intermediate-save-observability.md`

- [ ] Copy `development_docs/WP-CORE-3-empty-input-contract.md` as a template structure.
- [ ] Sections required (per CLAUDE.md §"Persistent Development Memory"):
  - **Status / Branch / Commit SHAs / Spec + Plan paths / TL;DR**
  - **Motivation** (1-2 paragraphs — F-13 + anomaly bundling rationale)
  - **Architectural decisions** (numbered, with rationale): 5 minimum
    1. New `IntermediateSaveError(PipelineError)` in `errors.py` (W-1 disposition)
    2. Narrow catch list `(OSError, TypeError, ValueError)` excluding `JSONDecodeError` (N-2 disposition)
    3. `except IntermediateSaveError: raise` in `identify_contexts` (C-1 disposition)
    4. Unconditional `self._current_srs_path = srs_path or "<unknown>"` at start of `analyze_document` (W-2 disposition)
    5. Display-only `"; ".join(...)` batch label with no truncation (W-3 disposition)
    6. Kept the 4 `getattr` fallbacks for belt-and-suspenders (OQ4 PARTIALLY)
  - **File-level changes table** (matches §"File Structure" of this plan)
  - **Methodology applied** (RED → GREEN-atomic; Codex xhigh adversarial review with zero deferred WARNs)
  - **Empirical results** (`321 → 332` pytest; `git diff --stat` summary)
  - **Limitations + follow-ups** (F-21 deferred; F-11 + F-14 still open; "typed Scout dump missing" anomaly added)
  - **Cross-references** (link to spec v2, plan, sibling WP docs, audit findings)

### 3.2 — Update `development_docs/INDEX.md`

- [ ] Append a new row to the ACTIVE table for WP-CORE-4 (matches WP-CORE-3's row format).

### 3.3 — Update `.planning/pipeline_audit/improvements_backlog.md`

- [ ] Move F-13 from "Open" to "Shipped" (note SHA of GREEN commit).
- [ ] Update F-20 status to MINOR (downgrade per spec drafting verification).
- [ ] Add F-21 to "Open" (MAJOR, deferred, anchor to architect.py:757-761).
- [ ] Update "Last refresh" date.

### 3.4 — Update `.planning/pipeline_audit/CURRENT.md`

- [ ] Replace "Last action" + "Next" lines to reflect WP-CORE-4 shipped and iteration-4 candidate.

### 3.5 — Write iteration-3 handoff `.planning/pipeline_audit/handoff-2026-05-21-{HHMM}.md`

- [ ] Filename uses commit-time `HHMM` for uniqueness.
- [ ] Follow structure of `handoff-2026-05-21-0937.md` (the iteration-2 handoff):
  - State of the loop (baseline, HEAD, commit summary)
  - WP-CORE-4 SHIPPED summary
  - Backlog state table (refreshed)
  - Component catalog state
  - Recommended next iteration (F-11 or F-14 or F-21 — pick recommendation with rationale)
  - Loop ritual reminder
  - Non-negotiables
  - Open follow-ups
  - Files produced this iteration
  - Goal-backward verification (iteration-3 goal: ship WP-CORE-4 cleanly)

### 3.6 — Commit DOC

- [ ] `git add development_docs/WP-CORE-4-intermediate-save-observability.md development_docs/INDEX.md .planning/pipeline_audit/improvements_backlog.md .planning/pipeline_audit/CURRENT.md .planning/pipeline_audit/handoff-2026-05-21-*.md .planning/pipeline_audit/findings/architect.md .planning/pipeline_audit/component_catalog.md`

  (Note: findings/architect.md + component_catalog.md were created by the Explore subagent + edited for F-21 addition; they should land in this DOC commit.)

- [ ] Commit message:

```
chore(artifacts): WP-CORE-4 dev_doc + audit state update

Adds development_docs/WP-CORE-4-intermediate-save-observability.md, updates
INDEX.md ACTIVE table, and refreshes audit state after WP-CORE-4 shipped:
- improvements_backlog: F-13 OPEN → SHIPPED at <GREEN_SHA>; F-20 downgraded
  to MINOR after spec-time verification; F-21 added OPEN (MAJOR, deferred).
- findings/architect.md: 11 findings catalogued (F-11..F-21); F-21 new from
  spec drafting (vacuous D1 pass via empty supporting_sentence_ids).
- component_catalog: priority-2 row (architect.py + orchestration) moved
  PENDING → DONE.
- handoff-2026-05-21-{HHMM}.md: iteration-3 close, iteration-4 ritual primer.

No code touched in this commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Step 4 — Planning commit: spec + plan into git history

**Commit:** `chore(planning): WP-CORE-4 spec v2 + plan into git history`

### 4.1 — Stage planning artifacts

- [ ] `git add docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md docs/superpowers/plans/2026-05-21-wp-core-4-intermediate-save-observability.md`

### 4.2 — Commit

```
chore(planning): WP-CORE-4 spec v2 + plan into git history

Persists the WP-CORE-4 spec (v2, post-Codex-xhigh review) and the
implementation plan to git history for cross-session traceability,
matching WP-CORE-2 and WP-CORE-3 patterns.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Verification matrix (post-Step 4)

| acceptance criterion (from spec §"Acceptance criteria") | verification command | expected |
|---|---|---|
| pytest ≥ 332 | `pytest -m "not integration" --tb=no -q 2>&1 \| tail -3` | `332 passed, 31 deselected` |
| No new bare-Exception catches | `grep -c "except Exception" core/architect.py` | ≤ pre-WP count |
| `_current_srs_path` has ≥ 2 assignment sites | `grep -nE "self\._current_srs_path[[:space:]]*=" core/architect.py` | 2 results (`__init__` + `analyze_document`) |
| `srs_path=` passed at ≥ 3 main.py sites | `grep -n "srs_path=" main.py` | ≥ 3 results |
| `IntermediateSaveError` defined in errors.py | `grep -n "class IntermediateSaveError" core/orchestration/errors.py` | 1 result |
| `IntermediateSaveError` referenced in architect.py | `grep -c "IntermediateSaveError" core/architect.py` | ≥ 2 (raise in `_save_intermediate` + re-raise in `identify_contexts`) |
| RED test count matches | `grep -c "^def test_" tests/test_intermediate_save.py tests/test_architect_srs_path.py` | 9 (5 + 4) |
| Wiring test extensions | `grep -c "^def test_" tests/test_main_wiring.py` | ≥ 3 (1 pre-existing T-WIRE-MAIN-1 + 2 new T-WIRE-MAIN-2/3) |

---

## Failure modes + rollback

| failure | likely cause | rollback |
|---|---|---|
| RED suite shows fewer than 11 failures | Test setup imports succeeded but functions defined incorrectly. | Inspect each `FAILED` line; fix test stub or assert; recommit RED with `--amend` (only allowed at the RED commit, since GREEN hasn't landed yet — per CLAUDE.md amend policy). |
| GREEN suite shows < 332 passing | Production code missed a step (e.g., forgot to add re-raise guard in identify_contexts). | Re-read §2; verify each step; do NOT amend GREEN — fix forward with a new commit `fix(architect): WP-CORE-4 followup for missed step X`. |
| Test T-WIRE-MAIN-3 (streaming) flakes due to thread sync | Worker thread didn't finish before assertion; `Event.wait(5.0)` timed out. | Increase timeout to 10.0; if still flaky, mark `@pytest.mark.skip("WP-CORE-4 follow-up: stream-test thread sync flakiness")` with a TODO in the handoff. |
| `git commit` triggers hook failure | Likely pyright complaint on the new code. | Fix the type error before commit; do NOT use `--no-verify`. |
| Pyright complains about `Optional[str]` import | Forgot to import `Optional` in `core/architect.py`. | Add `from typing import Optional` to imports. |
| Spec / plan / decision-log inconsistencies discovered late | DOC commit might miss an audit-state update. | Add a follow-up `chore(artifacts):` commit; do not amend. |

---

**Plan complete. Begin Step 1 (RED).**
