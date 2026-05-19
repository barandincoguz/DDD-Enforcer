# Typed Pipeline Contracts + Deterministic Synthesizer (WP-CORE-1) — Design (Revised)

**Status:** Approved 2026-05-19 (revised after Codex 5.5-xhigh adversarial review + live re-baseline run)
**WP code:** WP-CORE-1 (new — not in original 23-WP roadmap; "core muscle strengthening" per user pivot)
**Branch:** `feat/typed-pipeline-deterministic-synthesizer` → FF-merge to `main`
**Commits planned:** 8-10 sequential, each green-CI

**Revision history:**
- Draft 1 (commit `be85ca4`, 2026-05-19 22:15) — built on the wrong FM-LOST hypothesis (entity-count silent loss). Used `ctx.get('entities', [])` flat-path inspection but the real entity path is `ctx['ubiquitous_language']['entities']` nested.
- Marked PENDING_REVISION (commit `6ef246a`, 2026-05-19 23:00) — Codex adversarial review caught the diagnosis error.
- Revised (this version) — live pipeline run on D1 SRS failed at `architect.py:692` with `AttributeError: 'list' object has no attribute 'get'`. Real failure mode is **FM-CRASH**: Specialist LLM occasionally returns a JSON array at the top level, and the code path has no shape validation between parse and dict-access. Diagnosis is now evidence-grounded.

---

## 1. TL;DR

The domain-model pipeline **does not work on `main` as of `6ef246a`**. A fresh live run on `inputs/SRS.docx` (16,929 chars) on 2026-05-19 23:09 crashed at the very first Specialist call:

```
Stage 1 Scout: OK (16929 chars read)
Stage 2 Architect: OK (4 contexts identified — UserManagement, ProductCatalog,
                       OfferManagement, CustomerSupport)
Stage 3 Specialist: CRASH on first context (UserManagement)
  architect.py:692: AttributeError: 'list' object has no attribute 'get'
  All 5 retries hit same error.
  SpecialistFailureError raised → pipeline dies.
```

Root cause: at `architect.py:678` Specialist parses the LLM JSON response into `result`. The code then does `result.get("entities", [])` at line 692, assuming `result` is a dict. But Gemini 3.1 Pro occasionally returns a top-level JSON array (e.g. `[{...entities...}]`) instead of the prompted-for object `{entities: [...], ...}`. There is no shape-check between parse and dict access. The error from each retry is identical, so the retry loop is wasted on the same crash.

This is a textbook example of **dict-typed boundary with no shape validation**. WP-CORE-1 closes this by:

- (a) Introducing Pydantic typed contracts at all 5 stage boundaries (Scout / Architect / Specialist / Verifier / Refiner / Synthesizer). `model_validate(...)` raises on shape mismatch → retry can re-prompt the LLM with the schema error verbatim, not a stack trace.
- (b) Replacing the LLM-rewrite Synthesizer at `architect.py:766-837` with deterministic merge + a narrow LLM enrichment call (`synonyms_to_avoid`, cross-context `allowed_dependencies`). Eliminates the cost of one full LLM rewrite per pipeline run and removes the only remaining LLM-translation surface where Specialist data could be lost.
- (c) Adding Verifier semantic invariants D6 (entity-count preservation), D7 (entity-name traceability), D8 (aggregate-member referential integrity) as **hard-fail assertions on the deterministic Synthesizer output** — these are code-bug detectors, not Refiner-retry inputs (corrects the original-spec Refiner-ordering error caught by Codex B2).

Acceptance: the same live re-baseline run that just crashed must succeed — fresh `domain/model.json` with D1-strict-schema entities populated across all Architect-identified contexts, written from a green pipeline.

---

## 2. Goals & non-goals

### Goals

- Pipeline runs to completion on D1 SRS and produces a fresh, D1-strict-schema `domain/model.json` with non-empty entities.
- Every stage boundary is Pydantic-typed. A shape mismatch raises `ValidationError`, which the retry loop converts into a targeted re-prompt (not a stack-trace propagation).
- Synthesizer no longer transforms data through an LLM call. The only LLM use in Synthesizer is the narrow enrichment of `synonyms_to_avoid` + `allowed_dependencies` (data not in Specialist output).
- D6/D7/D8 invariants make any future Synthesizer regression visible as `SynthesizerInvariantError`.
- All existing 237 unit tests still pass; ~15-25 new tests cover the new contracts + deterministic Synthesizer + D6/D7/D8.
- 8 existing tests pinned to the old Synthesizer dict shape are migrated (not deleted) to the new typed contract.

### Non-goals

- File modularization of `architect.py` (1143 LOC) — deferred to WP-01d (the dedicated pipeline-class extraction WP).
- RAG / Validator-pipeline changes — separate WPs.
- ReAct tool-use for Specialist (M4) — deferred.
- CRITIC stage (M2) — explicitly distinct from D6/D7/D8 invariants (those are deterministic-code assertions, not LLM-based semantic critique).
- Cross-context Reflexion (M3) — deferred.
- Markdown-fence stripping for OSS models — user-paused OSS work.

---

## 3. Architectural decisions (and rationale)

### A1. Single WP for typed contracts + deterministic Synthesizer

The two are coupled: a deterministic Synthesizer can only safely passthrough Specialist data when that data is typed at the boundary. Splitting would leave a deterministic Synthesizer consuming dicts (same FM-CRASH surface). Atomic refactor only.

### A2. Per-context narrow LLM enrichment (revised from "one omnibus call")

Codex H3 caught that a single LLM call across N contexts × M entities for `synonyms_to_avoid` is a truncation / cost risk. Revised:

- One LLM call **per bounded context** to enrich `synonyms_to_avoid` per-entity within that context. Token budget is bounded by per-context entity count (~5-10 entities × 3-5 synonyms each = manageable).
- One additional LLM call (or deterministic inference) for cross-context `allowed_dependencies`. Inference rule: scan each context's entities for mentions of names in other contexts; if a mention exists, add the dependency. The LLM is used to disambiguate naming (entity vs incidental term).
- Total Synthesizer LLM cost: `N+1` narrow calls (N = number of bounded contexts, typically 4-6). Compared to the original 1 omnibus call, this is more API-call overhead but smaller payloads, lower truncation risk, and per-context retry granularity.

### A3. Pydantic typed contracts at all 5 boundaries

| Stage | Input type | Output type |
|---|---|---|
| Scout | `str` (raw SRS text) | `ScoutOutput` |
| Architect | `ScoutOutput` | `ArchitectOutput` |
| Specialist (per-context) | `ContextHypothesis` + `ScoutOutput` | `SpecialistAnalysis` |
| Verifier | `List[SpecialistAnalysis]` | `VerifierResult` |
| Refiner | `VerifierResult + List[SpecialistAnalysis]` | `List[SpecialistAnalysis]` (corrected) |
| Synthesizer | `List[SpecialistAnalysis]` | `DomainModel` (existing strict schema) |

The new boundary classes live in `core/pipeline_contracts.py`. Content classes (`Entity`, `ValueObject`, `Aggregate`, `BoundedContext`, `DomainModel` from `core/schemas.py:40-167`) are reused unchanged.

### A4. Refiner only re-runs Specialist; Synthesizer invariants are hard-fail (NEW — addresses Codex B2)

Codex caught that the original-spec routing of D6/D7/D8 failures back through Refiner was meaningless: Refiner re-runs **Specialist**, not Synthesizer. A Synthesizer-caused data integrity violation cannot be fixed by retrying Specialist.

Revised flow:

- Verifier D1/D2/D3/D4/D5 + S1 (existing): run on `List[SpecialistAnalysis]` BEFORE Synthesizer. Failures → Refiner re-prompts Specialist with structured feedback.
- D6/D7/D8 (new): run on the deterministic Synthesizer output. Because Synthesizer is deterministic, a D6/D7/D8 failure is a **code bug**, not an LLM hiccup. Raises `SynthesizerInvariantError` immediately; no retry.

This is honest with the architectural reality: deterministic code that violates an invariant means the code is broken; LLM retry won't help.

### A5. Specialist prompt updated to emit `description` (addresses Codex B1)

The current Specialist per-context prompt at `architect.py:746-752` emits entity fields `{name, attributes, confidence, justification, evidence_sentence_indices}` but does NOT emit `description`. The strict `Entity` Pydantic schema at `core/schemas.py:42-55` requires `description`. As of post-D1 patch, every fresh Specialist response would fail Pydantic validation on the missing field.

Fix: update the Specialist prompt to emit `description` as a 1-2 sentence summary of the entity's role in the domain. Update both the per-context prompt (`architect.py:740-758`) and the legacy omnibus prompt (`architect.py:540-552`) — the latter is even more outdated and only emits `{name, attributes}`.

The legacy omnibus prompt should also be deleted because the per-context flow is the only one used post-P3 refactor (`architect.py:986-987` calls `extract_per_context_details`).

### A6. Boundary parse must handle list-or-dict honestly (addresses the actual crash)

At every stage where the LLM is parsed, the result must be type-checked before any attribute access. Pattern:

```python
parsed = self._parse_json_response(text)
try:
    typed = SpecialistAnalysis.model_validate({
        "context": ctx.model_dump(),
        **(parsed if isinstance(parsed, dict) else _unwrap_singleton_list(parsed)),
    })
except ValidationError as e:
    # convert to retry-able feedback; do NOT crash
    raise SpecialistShapeError(context=ctx, errors=e.errors()) from e
```

The `_unwrap_singleton_list` helper handles the common Gemini-Pro case where the model returns `[{...}]` instead of `{...}` — extract the single dict if the list has length 1, else surface as a validation error.

This is **defensive parsing at the typed boundary**, not silent coercion: shape mismatches are converted to typed errors that the retry loop can act on with model-specific feedback (e.g. "your response was a JSON array; respond with a single JSON object").

### A7. Acceptance is "every Specialist entity present in DomainModel" (revised — addresses Codex B4)

The original spec acceptance ("non-empty entities in ≥4 of 6 contexts") contradicted the goal ("every entity preserved unchanged"). Revised acceptance:

- For each `SpecialistAnalysis a` in the post-Refiner analyses list, and for each `e in a.entities`: there exists exactly one `e' in synthesized_domain_model.bounded_contexts[a.context.context_name].ubiquitous_language.entities` such that `e.name == e'.name` (case-insensitive) and `e.attributes` is a subset of `e'.attributes` (deterministic merge preserves at least every input attribute).

This is the D6/D7 check turned into a runtime assertion. The live run is acceptable iff this invariant holds on the fresh `domain/model.json`.

### A8. Refactor strategy: stage-by-stage commits, single feature branch, FF merge

Same pattern as WP-NEW-B Stage 1 (proven). 8-10 commits, each green-CI, rollback-able per commit.

Planned commit sequence (high-level; writing-plans skill will produce the precise plan):

1. `feat(pipeline_contracts)` — new `core/pipeline_contracts.py` with all stage-envelope classes
2. `fix(architect/specialist)` — update Specialist per-context prompt to emit `description`; delete legacy omnibus method + prompt
3. `fix(architect/specialist)` — add list-or-dict defensive parsing + Pydantic boundary validation, retry on shape errors
4. `refactor(synthesizer)` — extract Synthesizer into `core/synthesizer/` package with deterministic merge + narrow per-context enrichment
5. `feat(verifier)` — add D6/D7/D8 checks; hard-fail (no Refiner path)
6. `refactor(architect)` — replace `synthesize_final_model` and `analyze_document` dict-cast at line 989-995 with typed flow
7. `refactor(orchestration)` — update `core/orchestration/pipeline.py` `SpecialistFn` / `SynthesizerFn` types to use typed contracts; remove `[List[Dict]]` aliases
8. `test(migration)` — migrate 8 existing tests pinned to dict shape (test_architect_prompts, test_pipeline_orchestration, test_synthesizer_empty_model_error, test_synthesize_final_model_errors) to the new typed contracts
9. `test(unit)` — ~15-25 new TDD tests for contracts + deterministic Synthesizer + D6/D7/D8 + boundary defensive parsing
10. `chore(artifacts)` — live re-baseline run on D1 SRS, commit fresh `domain/model.json` + manifest

---

## 4. Component design

### 4.1 New file: `core/pipeline_contracts.py`

```python
"""Typed contracts for stage boundaries in the domain-model pipeline.

Each stage produces and consumes a typed envelope. Boundary validation
is enforced via Pydantic .model_validate() at every transition. A
schema mismatch raises ValidationError that the stage retry loop
converts into targeted LLM feedback — not a stack-trace crash.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from core.schemas import (
    Entity, ValueObject, Service, Aggregate, DomainEvent,
)


class SectionedSentence(BaseModel):
    index: int = Field(ge=0)
    text: str
    section: Optional[str] = None


class ChunkMetadata(BaseModel):
    chunk_count: int
    total_chars: int
    truncated_chunks: int = 0


class ScoutOutput(BaseModel):
    sentences: List[SectionedSentence]
    chunk_metadata: ChunkMetadata


class ContextHypothesis(BaseModel):
    context_name: str
    description: str = ""
    supporting_sentence_ids: List[int] = Field(default_factory=list)


class ArchitectOutput(BaseModel):
    contexts: List[ContextHypothesis]
    open_questions: List[str] = Field(default_factory=list)


class Ambiguity(BaseModel):
    target: str
    reason: str


class SpecialistAnalysis(BaseModel):
    context: ContextHypothesis
    entities: List[Entity] = Field(default_factory=list)
    value_objects: List[ValueObject] = Field(default_factory=list)
    services: List[Service] = Field(default_factory=list)
    aggregates: List[Aggregate] = Field(default_factory=list)
    domain_events: List[DomainEvent] = Field(default_factory=list)
    business_rules: List[str] = Field(default_factory=list)
    ambiguities: List[Ambiguity] = Field(default_factory=list)


class VerifierIssue(BaseModel):
    severity: str  # "ERROR" | "WARN"
    check_id: str  # "D1" .. "D5" | "S1" | "D6" | "D7" | "D8"
    target: str
    message: str


class VerifierResult(BaseModel):
    is_ok: bool
    issues: List[VerifierIssue] = Field(default_factory=list)
```

### 4.2 Specialist boundary defensive parse + Pydantic validation

In `architect.py:extract_per_context_details`, replace the current dict-access block (lines 678-698) with the typed boundary pattern from §A6. New helper `_unwrap_singleton_list(parsed)` handles the Gemini-Pro list-wrapping case.

New exception class `SpecialistShapeError(SpecialistFailureError)` lets the retry loop distinguish "LLM returned wrong shape" (re-prompt with structured feedback) from "LLM returned valid shape but invalid data" (currently raises early).

### 4.3 Deterministic Synthesizer (`core/synthesizer/` package)

Extract the Synthesizer from `architect.py:766-944` into a new package:

- `core/synthesizer/__init__.py` — exports `synthesize_domain_model(analyses, llm_client, project_name) -> DomainModel`
- `core/synthesizer/merge.py` — deterministic merge of `List[SpecialistAnalysis]` → `DomainModel` skeleton. Pure function.
- `core/synthesizer/enrich.py` — per-context narrow LLM calls for `synonyms_to_avoid`; global LLM call (or deterministic + LLM disambiguation) for `allowed_dependencies`.
- `core/synthesizer/metadata.py` — mechanical `ProjectMetadata` + `GlobalRules` defaults.

Old `DomainArchitect.synthesize` and `DomainArchitect.synthesize_final_model` are **deleted** — not kept as wrappers (Codex M1 / AGENTS.md no-shim). `main.py` and other callers migrate to call the new package directly.

### 4.4 Verifier D6 / D7 / D8

Three new functions in `core/verifier/checks_semantic.py`:

```python
def check_d6_entity_count_preservation(
    analyses: List[SpecialistAnalysis],
    domain_model: DomainModel,
) -> List[VerifierIssue]:
    """Total entity count across analyses (POST-Refiner) must equal
    total entity count across DomainModel.bounded_contexts. Synthesizer
    is deterministic; any drop is a code bug."""

def check_d7_entity_name_traceability(...) -> List[VerifierIssue]:
    """Every entity in DomainModel must trace to a Specialist analysis
    entry by name. No fabricated entities."""

def check_d8_aggregate_member_referential_integrity(...) -> List[VerifierIssue]:
    """For every BoundedContext.aggregates[*].members[*], the referenced
    name must exist in the same context's entities list."""
```

These are called from `synthesize_domain_model` IMMEDIATELY after merge, before returning. Any ERROR-severity issue raises `SynthesizerInvariantError`. No Refiner loop.

### 4.5 Orchestration type updates (`core/orchestration/pipeline.py`)

Replace dict-typed callable aliases with typed ones:

```python
SpecialistFn = Callable[[ArchitectOutput, ScoutOutput], List[SpecialistAnalysis]]
SynthesizerFn = Callable[[List[SpecialistAnalysis]], DomainModel]
VerifierFn = Callable[[..., List[SpecialistAnalysis]], VerifierResult]
```

`run_pipeline(...)` consumes typed inputs throughout. `DomainArchitect.analyze_document` (`architect.py:1024`) constructs the typed `PipelineDeps` from the new typed methods.

### 4.6 Test migration (8 tests)

| File | Old expectation | New expectation |
|---|---|---|
| `tests/test_architect_prompts.py:9-63` (4 tests) | Substring-grep the old Synthesizer prompt | Substring-grep the new Specialist prompt that includes `description`. The old Synthesizer prompt is deleted. |
| `tests/test_pipeline_orchestration.py:40-108` (2 tests) | Uses dict-returning `synthesizer` fixture | Uses `DomainModel`-returning typed fixture |
| `tests/test_synthesizer_empty_model_error.py:19-29` | Patches `architect.synthesize_final_model` | Patches `core.synthesizer.synthesize_domain_model` |
| `tests/test_synthesize_final_model_errors.py:41-70` | Same | Same |

These are migrated in commit 8 of the sequence. The test-count delta is net-zero (8 migrated, ~15-25 new).

---

## 5. Data flow (post-refactor)

```
raw SRS text
   │
   ▼  Scout._extract_sentences_from_chunk
   │
   ▼   ScoutOutput (typed)   ←─ Pydantic validate
   │
   ▼  Architect.identify_contexts
   │
   ▼   ArchitectOutput (typed)   ←─ Pydantic validate
   │
   ▼  Specialist.extract_per_context_details (per-context loop)
   │     ↑
   │     └── on ValidationError → re-prompt LLM with schema error feedback
   │
   ▼   List[SpecialistAnalysis] (typed)   ←─ Pydantic validate at boundary
   │
   ▼  Verifier (D1-D5 + S1) on List[SpecialistAnalysis]
   │
   ▼  if !is_ok → Refiner → re-runs Specialist with VerifierResult feedback
   │
   ▼   List[SpecialistAnalysis] (refined)
   │
   ▼  synthesize_domain_model (deterministic merge + per-context enrichment)
   │
   ▼   DomainModel skeleton (typed, strict)
   │
   ▼  Verifier D6 + D7 + D8 (assertions on the merge output)
   │     │
   │     └── if any ERROR → raise SynthesizerInvariantError (no Refiner)
   │
   ▼   DomainModel (final, persisted to domain/model.json)
```

---

## 6. Verification (3 layers)

### 6.1 Unit tests (TDD per commit, ~15-25 new tests)

- `tests/test_pipeline_contracts.py` — every envelope class: construct, round-trip, ValidationError on bad input.
- `tests/test_specialist_boundary_parse.py` — singleton-list unwrap; multi-list rejection; dict passthrough; structured retry feedback on ValidationError.
- `tests/test_synthesizer_deterministic_merge.py` — typed `List[SpecialistAnalysis]` → DomainModel skeleton; entity count preserved exactly; aggregate.members all reference real entities; no fabrication.
- `tests/test_synthesizer_enrich.py` — mocked LLM, narrow per-context call returns synonyms; merged into Entity.synonyms_to_avoid; no other field touched.
- `tests/test_verifier_d6_d7_d8.py` — synthetic mismatch scenarios for each invariant.

### 6.2 Replay against last good intermediate

Load `core/intermediate/20260313_221928_3_specialist.json` (the historical run, schema is pre-D1 — adapter fills `description=name+" entity"`, `confidence=0.5`, `justification="(historical)"`, `evidence_sentence_indices=[0]`). Run the new deterministic Synthesizer. Assert:

- Output `DomainModel` has 6 bounded contexts (matches input).
- User entity in UserManagement preserved (name + attributes from `User: [id, firstName, ..., 13 attrs]`).
- Product entity in ProductCatalog preserved (name + 14 attrs).
- D6 check passes: total Specialist entity count == total DomainModel entity count.

This catches FM-LOST-style regressions without spending live LLM tokens.

### 6.3 Live re-baseline on D1 SRS (acceptance gate)

After unit + replay green:

```bash
cd extension/backend
.venv/bin/python -c "from main import generate_domain_model; generate_domain_model('inputs/SRS.docx')"
```

Expected:
- Pipeline runs to completion (no `AttributeError` crash like the failed 2026-05-19 23:09 run).
- Fresh `domain/model.json` written with D1-strict schema entities (`confidence`, `justification`, `evidence_sentence_indices`, `description` all populated).
- Architect-identified contexts (4-6) all have non-empty entities lists (or, for utility contexts like CustomerSupport that may legitimately be VO-only, at least non-empty value_objects).
- Manifest sidecar at `runs/domain_run-{ts}.{json,manifest.json}` captures git_commit, package_versions, model_id, total Specialist entity count, total DomainModel entity count (must match).

Live run cost: ~$0.50, wall time ~12-15 min (5 stages, per-context Specialist + per-context Synthesizer enrichment).

---

## 7. Risks + mitigations (incorporates all 13 Codex findings)

| # | Codex | Risk | Mitigation |
|---|---|---|---|
| 1 | B1 | Specialist prompt missing `description` → strict Entity validation fails every entity | A5: update prompt to emit `description`. Commit-2 in sequence. |
| 2 | B2 | Refiner re-runs Specialist; can't fix Synthesizer-caused errors | A4: D6/D7/D8 hard-fail; no Refiner loop. Code-bug semantics. |
| 3 | B3 | Original spec used flat `ctx.get('entities')` path; entity loss claim was wrong | This revised spec uses the actual failure (FM-CRASH at L692) caught by the live run. |
| 4 | B4 | Acceptance "≥4 of 6 contexts" contradicts goal of preservation | A7: revised acceptance is exact per-entity preservation, not context-count threshold. |
| 5 | H1 | `analyze_document` at L989-995 still adapts to legacy `{"context", "analysis"}` shape | Commit-6: refactor `analyze_document` to construct typed `PipelineDeps`. |
| 6 | H2 | 8 existing tests pinned to old dict API | §4.6: explicit migration table; commit-8 in sequence migrates them. No tests deleted. |
| 7 | H3 | Single LLM enrich call would truncate / cost-overrun | A2: per-context narrow calls; bounded payload per call. |
| 8 | H4 | Live integration test is skipped in CI | Live run is **manual acceptance gate**, not a CI gate. Documented in spec acceptance section. Future WP may add a CI workflow that runs live tests on a schedule, but not in this WP. |
| 9 | H5 | Pydantic tolerates extras silently; LLM `_unresolved` keys would be swallowed | Add explicit Pydantic validator on the typed envelopes: if any extra field name starts with `_unresolved` or `_refiner_`, raise. Other extras silently tolerated (LLM cosmetic emissions). |
| 10 | M1 | "Thin wrapper" wording ambiguous re: shim policy | A8: delete old methods outright; callers migrate. No wrappers. Codex's no-shim concern resolved. |
| 11 | M2 | architect.py is 1143 LOC, modularization deferred | This WP touches but does not split the file. Stage-by-stage commits each <250 LOC delta to bound diff bleed. File split is WP-01d's territory. |
| 12 | M3 | D6 equality vs Refiner can add entities | A4 revision: D6 compares POST-Refiner `List[SpecialistAnalysis]` total against DomainModel total. Refiner additions are in the POST-Refiner input. Equality holds. |
| 13 | M4 | D6/D7/D8 vs CRITIC stage distinction | Documented in §2 non-goals + A4: D6/D7/D8 are deterministic-code assertions (no LLM). The future CRITIC stage (M2) would be LLM-based semantic critique ("is this entity actually justified by the SRS sentences?"). Distinct contracts. |
| 14 | (live evidence) | Specialist LLM occasionally emits top-level array instead of object | A6: defensive parse with `_unwrap_singleton_list`; `SpecialistShapeError` retry path. |
| 15 | (live evidence) | Retry loop currently spends 5 retries on identical crashes | A6: each retry now feeds the LLM the specific `ValidationError` so the model can correct the shape. Different feedback per iteration. |

---

## 8. Acceptance criteria

- [ ] 8-10 commits land green-CI on `feat/typed-pipeline-deterministic-synthesizer`
- [ ] All existing 237 unit tests still pass; 8 migrated tests pass
- [ ] ~15-25 new tests cover contracts + boundary parse + deterministic Synthesizer + D6/D7/D8
- [ ] Replay test green against `core/intermediate/20260313_221928_3_specialist.json` (entity count + name preservation)
- [ ] Live re-baseline run on D1 SRS COMPLETES without crash and produces fresh `domain/model.json`
- [ ] Every entity in every `SpecialistAnalysis` (post-Refiner) appears in the corresponding `BoundedContext.ubiquitous_language.entities` of the synthesized DomainModel, name-matched and attribute-superset preserved.
- [ ] Every entity has all D1-strict fields populated: `description, confidence, justification, evidence_sentence_indices` (and `synonyms_to_avoid` from the narrow enrich).
- [ ] Manifest sidecar carries the same 16-key shape as WP-NEW-B Stage 1; D6 total entity-count match is recorded in the manifest as a verification line.
- [ ] No `gemini.py`, `ollama.py`, RAG, or Validator code touched.
- [ ] `development_docs/WP-CORE-1-typed-pipeline.md` written + INDEX updated in the artifact commit.

---

## 9. Out of scope (explicit)

- File modularization of `architect.py` → WP-01d.
- ReAct tool-use for Specialist (M4 in architectural menu) → future WP.
- CRITIC stage (M2 — LLM-based semantic critique distinct from D6-D8) → future WP.
- Cross-context Reflexion stage (M3) → future WP.
- Multi-pass Specialist (extraction → classification → aggregation) → future WP.
- Markdown-fence stripping for OSS models → user-paused.
- Token tracker normalization for the new narrow per-context Synthesizer calls → WP-01c.
- Run-spec orchestrator (multi-run YAML manifests) → WP-01b.
- CI workflow for live integration tests → future hygiene WP.

---

## 10. Cross-references

- Commit `be85ca4` — draft 1 spec (preserved as history)
- Commit `6ef246a` — PENDING_REVISION mark after Codex adversarial review
- This revision — built on live re-baseline run evidence (failed run at 2026-05-19 23:09)
- [[WP-NEW-B-Stage-1-schema-probe]] — same Codex-juri-then-spec workflow; `runs/` artifact format established here
- `core/schemas.py:40-167` — existing content schemas (reused)
- `core/architect.py:692` — the literal source of the crash this WP fixes
- `core/orchestration/pipeline.py:16-19` — dict-typed callable aliases that this WP replaces
- `core/intermediate/20260519_230928_2_architect.json` — Architect output of the failed live run; useful for replay-test scaffolding
- `domain/model.PRE-FRESH-RUN-BACKUP.json` — backup of the pre-refactor `domain/model.json` (made before the failed live run; preserved as historical reference)

---

## 11. Live run evidence (this WP's existence is grounded in)

```
2026-05-19 23:09:28
  Stage 1 Scout: 16929 chars parsed ✓
  Stage 2 Architect: 4 contexts identified ✓
    [UserManagement, ProductCatalog, OfferManagement, CustomerSupport]
  Stage 3 Specialist: UserManagement context
    API Request #1 → result type: list
    architect.py:692: result.get("entities", []) → AttributeError
    API Request #2-5 (retries): same error each time
  SpecialistFailureError raised → pipeline died
  Final state: no domain/model.json written, pipeline crashed
```

This is the failure WP-CORE-1 fixes. The acceptance gate is: this same pipeline command must succeed after the WP merges.
