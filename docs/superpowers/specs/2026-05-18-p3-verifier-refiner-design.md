# Design Spec — P3 Verifier+Refiner Architecture (DDD-Enforcer)

| Field | Value |
|---|---|
| **Status** | Approved (2026-05-18) |
| **Owner** | Baran Dincoguz |
| **Scope** | `extension/backend/core/architect.py` and adjacent modules |
| **Drives** | EMSE RQ1 (pipeline comparison) — quality of the P3 multi-agent pipeline |
| **Estimate** | 12-20 working days, 23 atomic commits, +720 / -300 LOC |
| **Predecessor** | Prior session locked roadmap (`todos/MASTER_PLAN.md`); this spec supersedes WP-01a as immediate priority. WP-01a deferred until after Phase D. |

---

## 1. Context

### 1.1 What exists today

`core/architect.py` (979 LOC) implements a 4-stage multi-agent pipeline:

```
SRS → Scout → Architect → Specialist → Synthesizer → DomainModel
```

Each stage is a single Gemini LLM call with Pydantic-structured output. Stage outputs are persisted to `core/intermediate/{timestamp}_{stage}.json` for debug replay.

### 1.2 Why this spec exists

A deep architectural audit (2026-05-18) identified **25 failure modes** in the current P3 pipeline, of which **5 are critical** for EMSE submission integrity:

| ID | Failure | File:line | Impact |
|---|---|---|---|
| FM-01 | Architect silently returns `["CoreDomain"]` on retry exhaustion | `architect.py:415-417, 458-459, 464-465, 467` | RQ1 stats collapse fallback runs into success bucket |
| FM-02 | Specialist returns `{"error": "parse_failed"}` shaped dicts; Synthesizer hallucinates from them | `architect.py:554-557, 590-593, 596-599` | Synthesizer fabricates entities from error payloads |
| FM-04 | Synthesizer returns empty `DomainModel(bounded_contexts=[])` on any error | `architect.py:696-697, 770-773` | 0-recall pipeline silently passes Pydantic validation |
| FM-05 | Synthesizer prompt omits `services` + `aggregates` example output | `architect.py:639-657` vs `schemas.py:115-128` | OSS models will follow prompt literally → unfair RQ1 comparison vs Gemini |
| FM-21 | `synthesize_final_model` wraps everything in bare `except Exception:` | `architect.py:758-773` | Pydantic validation errors silenced |

These five make the pipeline appear healthy when it has degraded. Other clusters cover prompt/schema divergence, truncation opacity, missing verification, and weak determinism. Full inventory in Appendix B.

### 1.3 Decision (this spec)

Adopt **Alt-A (Verifier+Refine)** with **Alt-D cherry-pick (evidence citations)** from the audit's 7 alternatives. Phased delivery A→B→C→D, each phase test-driven and CI-green.

---

## 2. Goals & Non-Goals

### 2.1 Goals

1. Every silent fallback in `architect.py` is converted to an explicit raise + structured failure log.
2. Synthesizer prompt aligns with the Pydantic `DomainModel` contract — all 5 building blocks (Entity, ValueObject, Service, Aggregate, DomainEvent) are explicitly demonstrated in the prompt's example output.
3. Each `Entity` in the persisted `domain/model.json` carries direct SRS evidence (`evidence_sentence_indices: List[int]`) and an LLM-emitted `confidence ∈ [0,1]`. No fabricated `InferenceSource = "generated"` survives.
4. A new `Verifier` stage runs deterministic + semantic checks after Synthesizer. Issues feed a `Refiner` that re-prompts the offending upstream stage. Max 2 refine cycles.
5. Section-aware SRS chunking replaces the current 10000-char hard split.
6. Per-context Specialist loop replaces the single omnibus Specialist call.
7. CI stays green after every commit (TDD discipline; mock LLM provider for unit tests, real integration test gated to one per phase boundary).

### 2.2 Non-goals

- **No provider abstraction in this scope.** Stays Gemini-only until WP-01a is run as a follow-up. Phase C's semantic Verifier uses the current `genai.Client`. Migrating to `core/llm/` is deferred.
- **No paper / writing / replication work.** User pivoted: project quality first, paper/docs deferred.
- **No CI/repro hygiene fixes (G14-G21).** Out of immediate scope. Tracked separately.
- **No prompt translation to per-provider variants.** OQ5 deferred. Phase C accepts that Gemini-tuned prompts may bias 6-model RQ1; this is a paper limitation, addressed at writing time.
- **No SDK / dependency upgrades.** Lock file stays as it is.

---

## 3. Architecture

### 3.1 Pipeline topology

5-stage forward pipeline + bounded refine loop.

```
SRS document
 │
 ▼  Stage 1 — Scout
 │     • Input: parsed SRS (text + section structure from document_parser).
 │     • Section-aware chunking: each chunk = one SRS section
 │       (e.g. "3.1 Order Management").
 │     • Output: per-section domain-relevant sentences with sentence_id.
 │     • Shape: {section_id, sentences: [{idx, text, is_domain_relevant}]}
 │
 ▼  Stage 2 — Architect
 │     • Input: aggregated Scout output.
 │     • Output: [{context_name, supporting_sentence_ids: [int]}]
 │     • Constraint (prompt): every supporting_sentence_ids must be a
 │       subset of Scout-extracted sentence indices.
 │
 ▼  Stage 3 — Specialist  (per-context loop, NOT omnibus)
 │     • Loop: one LLM call per bounded context (not one call for all).
 │     • Output (per context):
 │       {entities, value_objects, services, aggregates, domain_events}
 │       Each entity: {name, attributes, confidence ∈ [0,1],
 │                     evidence_sentence_indices: [int], justification: str}
 │       Each aggregate: {name, members: [entity_name], confidence,
 │                        evidence_sentence_indices}
 │
 ▼  Stage 4 — Verifier   (Alt-A core)
 │     • Deterministic checks:
 │       D1: every BC.supporting_sentence_ids ⊆ Scout-emitted indices.
 │       D2: every Entity has ≥1 evidence_sentence_index.
 │       D3: every Entity name appears in exactly one bounded context.
 │       D4: every Aggregate.members entry exists as an Entity in the
 │           same context.
 │       D5: every context.allowed_dependencies references an existing
 │           bounded_context name.
 │     • Semantic check (LLM-based):
 │       S1: For each Entity, does the cited evidence sentence
 │           actually mention this concept? (sample-based at high N to
 │           save tokens; full check at low N).
 │     • Output: VerifierResult = {ok: bool, issues: [Issue]}
 │
 ▼  If issues → Stage 4b — Refiner
 │     • Refiner takes the issue list + the failing stage's output +
 │       the failing stage's prompt template.
 │     • Re-prompts the failing stage with issues as "fix this" tail.
 │     • Bounded retry: max 2 cycles per stage per run.
 │     • If still issues after 2 cycles → raise
 │       RefinementExhaustedError(issues).
 │
 ▼  Stage 5 — Synthesizer
 │     • Merge per-context Specialist outputs into a single DomainModel
 │       (Pydantic strict).
 │     • Empty model → raise (no fallback).
 │     • _cleanup_domain_data is rewritten to coerce only structural
 │       shape issues, never inject content (no "PascalCase" defaults).
 │
DomainModel persisted
```

### 3.2 Why 5 stages, not 4

Verifier is a distinct stage, not a hidden retry inside Specialist or Synthesizer, because:
- It receives the *aggregated* output of every prior stage and must reason across stage boundaries.
- Refiner uses Verifier output as input — separating the agent that detects issues from the agent that fixes them keeps prompts focused.
- The dead `core/intermediate/*_5_verifier.json` files indicate a prior team prototype of exactly this topology (audit FM-22). Reuse over reinvention.

### 3.3 Why per-context Specialist loop

The audit (FM-23) identified context-blending as a structural risk: a single Specialist prompt holding all bounded contexts simultaneously lets the LLM assign the same entity to multiple contexts. Per-context loops force exclusive ownership at the prompt level.

Cost: each context becomes 1 LLM call rather than 1/n of one call. For a typical 4-context SRS, this is 4× Specialist calls — but each prompt is much smaller (one context's slice of Scout output, not all of it), so per-call token cost drops.

---

## 4. Component Map

### 4.1 New modules

```
core/verifier/
  __init__.py
  types.py                        # VerifierIssue, IssueSeverity (enum), VerifierResult
  checks_deterministic.py         # D1-D5: pure-function checks against stage outputs
  checks_semantic.py              # S1: LLM-based grounding spot-check
core/refiner/
  __init__.py
  prompts.py                      # per-stage refinement prompt templates
  loop.py                         # bounded retry orchestration; emits RefinementExhaustedError
core/orchestration/
  __init__.py
  pipeline.py                     # 5-stage driver; replaces DomainArchitect.analyze_document
core/scout/
  __init__.py
  chunking.py                     # section-aware chunker
```

### 4.2 Modified files

| File | Change |
|---|---|
| `core/architect.py` | Stage prompts rewritten (FM-05/06/07/08); bare-except narrowed (FM-21); silent fallbacks raise (FM-01/02/04); per-context Specialist loop (FM-23). `DomainArchitect.analyze_document` becomes a thin caller of `core/orchestration/pipeline.py`. |
| `core/schemas.py` | `Entity.evidence_sentence_indices: List[int]` (Optional in Phase A, required min_items=1 from D1); `Entity.justification: str` (required); `Entity.confidence: float` (required, no default 0.5); `BoundedContext.supporting_sentence_ids: List[int]`; `BoundedContext.business_rules: Optional[List[str]]`; `Aggregate.members: List[str]`. Pydantic validators reject empty `bounded_contexts`. |
| `core/AST/ast_model_signals.py` | `_collect_signals` raises instead of silently swallowing exceptions (G01). |
| `core/AST/ast_signal_enrichment.py` | `_ensure_traceability` drops `"generated"` fabrication; raises `InsufficientGroundingError` (OQ2 = drop+raise). |
| `core/parser.py` | No change (facade preserved). |

### 4.3 Removed code

- `architect.py:707-720` `_create_fallback_model` — deleted outright.
- `architect.py:415-417, 458-459` `["CoreDomain"]` fallback path — deleted.
- `architect.py:758-773` bare `except Exception` in `synthesize_final_model` — narrowed.
- AST `"generated"` `InferenceSource` fallback — deleted.

### 4.4 Exception hierarchy (new)

```python
# core/orchestration/errors.py
class PipelineError(Exception): ...
class ScoutChunkParseError(PipelineError): ...
class ArchitectExtractionError(PipelineError): ...
class SpecialistFailureError(PipelineError): ...
class SynthesizerEmptyModelError(PipelineError): ...
class RefinementExhaustedError(PipelineError): ...
class InsufficientGroundingError(PipelineError): ...
```

All inherit `PipelineError`. Top-level orchestrator catches `PipelineError`, marks run `degraded: True`, writes `runs/<id>/failure_log.json`, and **does not return an empty DomainModel**.

---

## 5. Data Flow & Failure Policy

### 5.1 Per-stage I/O contract

```python
ScoutOutput = {
    "sections": [{
        "section_id": str,
        "section_title": str,
        "sentences": [{"idx": int, "text": str, "is_domain_relevant": bool}]
    }]
}

ArchitectOutput = {
    "contexts": [{
        "name": str,
        "description": str,
        "supporting_sentence_ids": List[int]
    }]
}

SpecialistOutput = {  # one per context
    "context_name": str,
    "entities": [{
        "name": str,
        "attributes": List[str],
        "confidence": float,        # [0.0, 1.0], LLM-emitted (OQ3)
        "evidence_sentence_indices": List[int],   # required, min_items=1
        "justification": str
    }],
    "value_objects": [...],
    "services": [...],
    "aggregates": [{
        "name": str,
        "members": List[str],         # entity names, must exist in entities[]
        "confidence": float,
        "evidence_sentence_indices": List[int]
    }],
    "domain_events": [...]
}

VerifierResult = {
    "ok": bool,
    "issues": [{
        "stage": Literal["scout", "architect", "specialist", "synthesizer"],
        "location": str,              # e.g. "specialist:OrderMgmt.entities[2]"
        "issue_type": Literal["missing_evidence", "duplicate_name",
                              "invalid_member", "ungrounded", ...],
        "severity": Literal["error", "warn"],
        "message": str,
        "suggestion": Optional[str]  # for Refiner
    }]
}

SynthesizerOutput = DomainModel  # Pydantic strict
```

### 5.2 Failure policy table

Replaces every silent fallback in the current pipeline.

| Failure condition | Action |
|---|---|
| Scout chunk parse-fail (after retries) | raise `ScoutChunkParseError(chunk_id, attempts)` |
| Architect returns 0 contexts | raise `ArchitectExtractionError(srs_path)` |
| Specialist returns `{"error": ...}` shaped dict | raise `SpecialistFailureError(context_name)` |
| Synthesizer returns `bounded_contexts=[]` | raise `SynthesizerEmptyModelError(input_summary)` |
| Verifier issues persist after 2 refine cycles | raise `RefinementExhaustedError(issues)` |
| AST entity has no grounding evidence | raise `InsufficientGroundingError(entity_name)` |
| Pydantic validation of any stage output | raise — no `try/except` swallowing |

Top-level pipeline wrapper catches `PipelineError`, marks run `degraded: True` with reason, writes `runs/<id>/failure_log.json` containing partial state, and **propagates** to the orchestrator. The orchestrator (run_pipeline CLI, future WP-01b) decides whether to retry, skip, or count as a failed run for RQ1 metrics.

### 5.3 What "degraded" means

A run is degraded if ANY stage raised a `PipelineError`. The orchestrator's response:
- **Retry budget remaining**: re-run with a fresh seed (paper variance budget per D4).
- **No retry budget**: count as a failed run in RQ1 metrics (excluded from precision/recall numerators; counted in the denominator for `json_failed_rate`).
- **Failure log**: persisted to `runs/<id>/failure_log.json` for post-hoc analysis.

This is the inverse of today's behavior where degraded runs are silently classified as successes.

---

## 6. Testing Strategy

### 6.1 TDD discipline per commit

Every commit follows: failing test → implementation → green test. Each commit must keep the full unit-test suite green (`pytest -m "not integration"`).

### 6.2 Mock LLM provider

A `MockLLMClient` fixture (under `tests/fixtures/mock_llm.py`) returns canned responses keyed by stage name and prompt hash. Used by:
- All Verifier unit tests
- All Refiner unit tests
- Pipeline orchestration tests

Real Gemini calls only happen in integration tests, gated to one per phase boundary.

### 6.3 Golden fixtures per stage

```
tests/fixtures/srs_d1_clean/
  scout_output.json
  architect_output.json
  specialist_output.json        # one file with all contexts inline for test ergonomics
  verifier_output_ok.json
  verifier_output_with_issues.json
  synthesizer_output.json       # final DomainModel
```

Each fixture is generated once from a real Gemini run on `inputs/SRS.docx`, then frozen. Tests assert against these to detect regressions.

### 6.4 Integration test cadence

- End of Phase A: full pipeline on D1 SRS produces non-empty `DomainModel` (smoke).
- End of Phase B: silent-fallback regression suite — for each old fallback path, assert that the right exception now raises.
- End of Phase C: Verifier catches injected issues; Refiner closes them within 2 cycles on the happy path; RefinementExhaustedError raises on adversarial case.
- End of Phase D: every persisted entity has `evidence_sentence_indices` non-empty; AST enrichment never injects `"generated"`.

### 6.5 Determinism test

Two layers:
1. **Unit-level (strict)**: `tests/test_refiner_determinism.py` runs the Refiner with a mock LLM returning identical canned responses; asserts (a) same retry counts, (b) same issue→fix mapping. Catches non-determinism in *our* refine logic.
2. **Integration-level (soft)**: `tests/test_pipeline_determinism_integration.py` runs the full pipeline N=3 against real Gemini with `seed=42`; reports run-to-run variance as a metric (not an assert). Detects unexpected drift but accepts Gemini's "best-effort" seed (FM-17). Variance values are recorded for the eventual paper.

---

## 7. Phasing

23 atomic commits across 4 phases. Every commit ships test-first.

### Phase A — Prompt / Schema alignment (1 day, 6 commits, low risk)

Fixes prompt-schema divergence and the bare `except`. No structural change.

| # | Commit | Touches | Test |
|---|---|---|---|
| A1 | `fix(synthesizer): add services + aggregates + domain_events to prompt example` | `architect.py:619-664` | `test_synthesizer_prompt_aligns_with_schema` |
| A2 | `feat(specialist): emit structured aggregates with members[] instead of aggregate_roots[]` | `architect.py:497-521` + `schemas.py:93-106` | `test_specialist_aggregates_have_members` |
| A3 | `feat(specialist): require domain_events extraction field` | `architect.py:497-521` | `test_specialist_emits_domain_events` |
| A4 | `feat(schema+specialist): emit Entity.confidence ∈ [0,1] + justification (evidence_sentence_indices added as Optional, tightened in D1)` | `schemas.py:40-56` + Specialist prompt | `test_entity_schema_strict_confidence` |
| A5 | `refactor(synthesizer): narrow bare except to specific exceptions` | `architect.py:758-773` | `test_synthesizer_propagates_pydantic_errors` |
| A6 | `refactor(cleanup): _cleanup_domain_data no longer fabricates defaults` | `architect.py:832-868` | `test_cleanup_preserves_llm_output_faithfully` |

### Phase B — Silent fallback removal (1-2 days, 5 commits, medium risk)

Converts every silent fallback to an explicit raise. Adds the exception hierarchy.

| # | Commit | Touches | Test |
|---|---|---|---|
| B1 | `feat(errors): introduce PipelineError hierarchy in core/orchestration/errors.py` | new file | `test_pipeline_error_subclasses` |
| B2 | `refactor(architect): raise ArchitectExtractionError instead of CoreDomain fallback` | `architect.py:415-417, 458-459` | `test_architect_raises_on_empty_contexts` |
| B3 | `refactor(specialist): raise SpecialistFailureError instead of returning error dicts` | `architect.py:554-557, 590-593, 596-599` | `test_specialist_raises_on_parse_failure` |
| B4 | `refactor(synthesizer): remove _create_fallback_model; raise SynthesizerEmptyModelError` | `architect.py:707-720, 770-773` | `test_synthesizer_raises_on_empty_model` |
| B5 | `refactor(ast): _collect_signals raises instead of silently swallowing` | `core/AST/ast_model_signals.py:71-72` (G01) | `test_collect_signals_propagates_errors` |

### Phase C — Verifier + Refiner + section-aware chunking (7-12 days, 8 commits, high risk)

The structural change. Each commit independently testable.

| # | Commit | Touches | Test |
|---|---|---|---|
| C1 | `feat(verifier): types.py + __init__ — interface only, no impl` | new `core/verifier/types.py` | `test_verifier_types_construct` |
| C2 | `feat(verifier): deterministic checks D1-D5` | new `core/verifier/checks_deterministic.py` | 5 unit tests, one per check |
| C3 | `feat(verifier): semantic grounding check S1 with mock LLM` | new `core/verifier/checks_semantic.py` | `test_semantic_check_flags_ungrounded_entity` |
| C4 | `feat(refiner): prompts + bounded retry loop` | new `core/refiner/` | `test_refiner_caps_at_2_cycles` + `test_refiner_raises_on_exhaustion` |
| C5 | `feat(scout): section-aware chunking` | new `core/scout/chunking.py` (OQ1) | `test_section_chunker_respects_token_budget` |
| C6 | `feat(orchestration): 5-stage pipeline driver with refine loop` | new `core/orchestration/pipeline.py` | `test_pipeline_happy_path` + `test_pipeline_invokes_refiner_on_issues` |
| C7 | `refactor(architect): DomainArchitect.analyze_document delegates to orchestration.pipeline` | `architect.py` (thin facade) | `test_analyze_document_calls_pipeline` + existing integration tests stay green |
| C8 | `test(integration): full pipeline on D1 SRS produces valid DomainModel` | new integration test | runs against real Gemini, gated by env var |

### Phase D — Evidence citation + grounding tightening (3-5 days, 4 commits, medium risk)

| # | Commit | Touches | Test |
|---|---|---|---|
| D1 | `feat(specialist): require evidence_sentence_indices per entity (tighten schema to min_items=1, prompt emit references stable C5 sentence_ids)` | `architect.py` prompt update + `schemas.py` tighten | `test_specialist_entity_has_evidence_ids` |
| D2 | `feat(verifier): D2 check rejects entities without ≥1 evidence_sentence_index` | `core/verifier/checks_deterministic.py` | `test_d2_rejects_ungrounded_entity` |
| D3 | `refactor(ast): drop fabricated "generated" InferenceSource; raise InsufficientGroundingError` | `core/AST/ast_signal_enrichment.py:177-197` (OQ2) | `test_ast_enrichment_raises_when_ungrounded` |
| D4 | `test(integration): every persisted entity has real SRS evidence` | new integration test | regression assertion on `domain/model.json` |

### Phase totals

| Phase | LOC delta | Days | Commits | Risk |
|---|---|---|---|---|
| A | +50 / -30 | 1 | 6 | low |
| B | +120 / -180 | 1-2 | 5 | medium |
| C | +450 / -50 | 7-12 | 8 | high |
| D | +100 / -40 | 3-5 | 4 | medium |
| **Total** | **+720 / -300** | **12-20** | **23** | — |

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Phase C blows past 12 days | medium | schedule slip on RQ1 | budget 2 extra buffer days; if exceeded, ship C7-C8 as Phase C2 (delivered after Phase D) |
| Section-aware chunking changes Scout output schema → regression on existing Gemini runs | medium | RQ1 numbers shift mid-experiment | freeze pre-C5 intermediate JSONs as `legacy_intermediate/`; document chunking change as a study parameter |
| Refiner loop introduces non-determinism (different retry counts across runs) | medium | paper variance budget bloats | tight determinism test in C4; cap retries at 2 hard; same-seed → same-retry-count assertion |
| Phase C semantic Verifier (LLM check) becomes the cost bottleneck | medium | RQ run costs blow up | sample-based S1 check at high N; full check at low N; budget tracking via existing TokenTracker |
| Removing silent fallbacks (Phase B) reveals previously-hidden regressions | high | integration tests fail | this is GOOD — that's what Phase B exposes; budget time to fix each unmasked bug; do not re-introduce fallbacks |
| Locked Gemini-only client conflicts with WP-01a when it lands | low | architect.py needs second migration | Phase C uses current `genai.Client` directly; WP-01a (future) wraps it; well-scoped Phase D-adjacent rework |
| Phase D's grounding-tightening rejects too many entities → low recall | medium | RQ1 metrics shift | calibration: in Phase D2, default to `severity: warn` for the first integration test; promote to `severity: error` after measuring rejection rate |

---

## 9. Verification Plan

### 9.1 Per-commit gate

- `pytest -m "not integration"` green
- `pyright` clean (CI continue-on-error currently — promote to blocking gate in a follow-up)
- No new `except Exception:` blocks in changed files

### 9.2 Per-phase gate

- **End A**: integration smoke test runs and produces a non-empty DomainModel with `services` + `aggregates` populated.
- **End B**: regression suite — every removed silent fallback now raises the right typed exception.
- **End C**: Verifier integration test passes; Refiner closes injected issues within budget.
- **End D**: every persisted entity has non-empty `evidence_sentence_indices`; `grep -rn "generated" extension/backend/domain/model.json` returns zero.

### 9.3 Final acceptance

Full pipeline run on `inputs/SRS.docx` produces a `DomainModel` that satisfies:

1. `bounded_contexts` has ≥3 entries (smoke).
2. Every `Entity.evidence_sentence_indices` is non-empty and lies within Scout-emitted indices.
3. Every `Aggregate.members` references entities in the same bounded context.
4. Every `BoundedContext.allowed_dependencies` references an existing context.
5. No `InferenceSource` has `rule = "LLM_SYNTHESIS"` or `file = "generated"`.
6. `n_refiner_cycles = 0` on the happy path; `n_refiner_cycles ≤ 2` always.
7. Same seed → same output across 3 runs (determinism test).

---

## 10. Out of Scope (explicit)

- WP-01a provider abstraction (`core/llm/` package). Stays Gemini-only; deferred to follow-up after Phase D.
- Paper-writing WPs (10, 11, 13, 14, 15, 16). Deferred per user pivot.
- Replication packaging (WP-12, WP-NEW-D). Deferred.
- Hijyen sweep gaps G14-G21 (CI lockfile, README, .vscodeignore, .DS_Store, pyright strict, PYTHONHASHSEED, PyTorch pin). Deferred.
- OQ5 per-provider prompt validation (4 OSS variants). Deferred to writing time as paper limitation.
- WP-NEW-C prompt sensitivity ablation. Deferred — depends on a stable post-Phase-D baseline.
- Audit (WP-08, Fleiss's κ). Deferred — depends on having scoreable runs.

---

## 11. Open Question Tracking

| ID | Status | Resolution |
|---|---|---|
| OQ1 — chunking strategy | **resolved** | Section-aware (this spec §3.1, §7 Phase C5) |
| OQ2 — fabricated InferenceSource | **resolved** | Drop + raise (this spec §4.2, §7 Phase D3) |
| OQ3 — confidence source | **resolved** | LLM emits in Specialist prompt (this spec §3.1, §5.1) |
| OQ4 — lost v2 Specialist | **pending** | Phase D kickoff: `git log --all -p -- extension/backend/core/architect.py` to search for the richer schema seen in `core/intermediate/20260312_222001_3_specialist.json`; if found, port; if not, design fresh. Single-day investigation; logged in commit message. |
| OQ5 — per-provider prompt fairness | **pending — deferred** | Recognized as paper-time concern; documented as a Threat to Validity in §9.3 of the eventual paper. Not addressed in this spec's scope. |

---

## Appendix A — Alternatives Considered

The 2026-05-18 architectural audit evaluated 7 alternatives. The full scorecard:

| Arch | Precision | Recall | Traceability | DDD-correctness | Determinism | JSON-conform | Cost (rel) | Complexity |
|---|---|---|---|---|---|---|---|---|
| P3 baseline | = | = | = | = | = | = | 1.0× | 979 LOC |
| **Alt-A Verifier+Refine** | **+** | **+** | **++** | **+** | − | **+** | 1.3× | +330 |
| Alt-B Self-consistency ensemble | + | = | = | = | ++ | + | 3.0× | +120 |
| Alt-C ReAct tool-using single agent | + | + | ++ | = | −− | = | 2.0× | +400 |
| Alt-D Multi-agent debate + arbiter | ++ | + | + | + | − | = | 2.2× | +350 |
| Alt-E Plan+execute+state | + | ++ | ++ | ++ | + | = | 1.8× | +500 |
| Alt-F Per-element parallel specialists | + | + | = | = | = | + | 2.5× wall ≈ 1× | +300 |
| Alt-G RAG-iterative agent | = | + | + | − | − | = | 1.4× | +250 |

**Decision (this spec)**: Alt-A. Cherry-pick from Alt-D: enforce evidence citation at the Specialist prompt level (Phase D1). Rationale: Alt-A fixes the most critical failure mode clusters (Cluster A silent fallbacks, Cluster D no-verification) with the lowest implementation risk in a 14-week submission window. Cherry-picking the evidence-citation requirement from Alt-D closes FM-12 traceability without the cost of running adversarial agents.

---

## Appendix B — Failure Mode Inventory (Audit Output, 2026-05-18)

25 failure modes identified. Severity breakdown: 5 critical, 10 high, 7 medium, 3 informational.

### Cluster A — Silent fallbacks (5 modes)

| ID | File:line | Fixed by |
|---|---|---|
| FM-01 | `architect.py:415-417, 458-459` | Phase B2 |
| FM-02 | `architect.py:554-557, 590-593, 596-599` | Phase B3 |
| FM-03 | `architect.py:315-317, 336-338` | Phase B (Scout split fallback removed) |
| FM-04 | `architect.py:696-697, 707-720, 770-773` | Phase B4 |
| FM-21 | `architect.py:758-773` | Phase A5 |

### Cluster B — Prompt↔schema divergence (4 modes)

| ID | File:line | Fixed by |
|---|---|---|
| FM-05 | `architect.py:639-657` vs `schemas.py:115-128` | Phase A1 |
| FM-06 | `architect.py:500-519` | Phase A2 |
| FM-07 | `architect.py:497-521` | Phase A3 |
| FM-08 | `architect.py:516` (rules dropped) | Phase A (BoundedContext.business_rules) |

### Cluster C — Truncation/chunking opacity (3 modes)

| ID | File:line | Fixed by |
|---|---|---|
| FM-09 | `architect.py:48-63` (chunk visibility) | Phase C5 (section-aware) |
| FM-10 | `architect.py:178, 360, 492` (hardcoded sizes) | Phase C5 (dynamic from model context_window) |
| FM-25 | `architect.py:231-247` (char-based split) | Phase C5 |

### Cluster D — No verification (5 modes)

| ID | What | Fixed by |
|---|---|---|
| FM-11 | No Architect grounding check | Phase C2 (D1 check) |
| FM-12 | No Entity→Scout-sentence linkback | Phase C2 + D1 (evidence_sentence_indices) |
| FM-13 | No aggregate-context-binding check | Phase C2 (D3, D4 checks) |
| FM-14 | Synonyms unsupervised | Phase C3 (semantic check) |
| FM-15 | No cross-stage feedback loop | Phase C4-C6 (Refiner) |
| FM-24 | No deterministic post-validator | Phase C2 (D-checks run after Synthesizer too) |

### Cluster E — Determinism/confidence (3 modes)

| ID | What | Fixed by |
|---|---|---|
| FM-16 | `Entity.confidence` always 0.5 default | Phase A4 (LLM-emitted) |
| FM-17 | `temperature=0.05, seed=42` is best-effort | Deferred (variance measurement in D4 plan) |
| FM-18 | `json_failed_rate` not tracked | Deferred (TokenTracker enhancement, separate WP) |

### Cluster F — Cross-cutting (5 modes)

| ID | What | Fixed by |
|---|---|---|
| FM-19 | `min_delay=6.0` blocks throughput | Deferred (Pro-tier override exists) |
| FM-20 | `_cleanup_domain_data` injects defaults | Phase A6 |
| FM-22 | Dead `_5_verifier.json` files | Resolved (Verifier resurrected in Phase C) |
| FM-23 | Specialist context-blending | Phase A2 + C7 (per-context loop) |
| OQ4 | Lost v2 specialist (info-only) | Tracked as open question |

---

## Appendix C — Files Cited

`extension/backend/core/architect.py` (specifically lines 38, 48-63, 178, 231-247, 275, 315-317, 336-338, 360, 363, 415-417, 458-459, 467, 473-521, 492, 497-521, 500-519, 516, 554-557, 590-593, 596-599, 619-664, 625, 639-657, 647, 664, 696-697, 707-720, 752-773, 832-868, 870-884) · `extension/backend/core/schemas.py:28-34, 44-49, 93-106, 115-129` · `extension/backend/core/AST/ast_signal_enrichment.py:101-105, 177-197` · `extension/backend/core/AST/ast_model_signals.py:71-72` · `extension/backend/core/token_tracker.py:59-94` · `extension/backend/configs/models.py:99, 113-115`.

---

## Approval Trail

- **2026-05-18**: User confirmed Alt-A direction.
- **2026-05-18**: User confirmed OQ1 (section-aware chunking), OQ2 (drop+raise grounding), OQ3 (LLM-emitted confidence).
- **2026-05-18**: User confirmed scope pivot — Baran solo, paper/docs/replication deferred, focus = project quality.
- **2026-05-18**: User approved this design.

Next: spec self-review → user reviews spec file → invoke `superpowers:writing-plans` skill to produce per-commit implementation plan.
