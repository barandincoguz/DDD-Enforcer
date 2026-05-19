# Typed Pipeline Contracts + Deterministic Synthesizer (WP-CORE-1) — Design

**Status:** Draft 2026-05-19 (awaiting user review)
**WP code:** WP-CORE-1 (new — not in original 23-WP roadmap; "core muscle strengthening" per user pivot)
**Branch:** `feat/typed-pipeline-deterministic-synthesizer` → FF-merge to `main`
**Commits planned:** 6-8 sequential, each green-CI

---

## 1. TL;DR

The domain-model pipeline silently loses Specialist's entity output during Synthesizer. **Mar-13 intermediate run shows Specialist correctly extracted `User`(13 attrs) and `Product`(14 attrs) entities; Synthesizer final_model emitted `entities=[]` for every context.** The committed `domain/model.json` mirrors this — 5 bounded contexts with zero entities/VOs/services across all.

Root cause: Synthesizer is a full LLM-rewrite (`architect.py:766-837`). It takes Specialist's `{name, attributes: [str]}` dict via JSON-dump in prompt and asks the LLM to re-emit in the strict Pydantic schema (`{name, description, confidence, justification, evidence_sentence_indices}` per `core/schemas.py:40-63`). The LLM cannot invent `confidence` or `justification` from nothing, so it emits empty arrays — and there is no code-level guarantee the input survives the LLM translation. Verifier passes because the structural check (5 contexts exist) succeeds.

This WP fixes FM-LOST by:
- (a) Typing all 5 stage boundaries with Pydantic models (Scout→Architect→Specialist→Verifier→Refiner→Synthesizer)
- (b) Refactoring Synthesizer from LLM-rewrite to deterministic merge with a single narrow LLM enrichment call for `synonyms_to_avoid` + cross-context `allowed_dependencies`
- (c) Adding Verifier D6/D7/D8 cross-stage referential integrity checks so any future silent-loss regression fails loudly

---

## 2. Goal & non-goals

### Goals

- Fix the entity-count silent-loss bug. After the refactor, every Specialist entity that passes per-context Verifier checks MUST appear unchanged in `domain/model.json`.
- Make stage boundary contracts explicit and machine-validated. A schema mismatch becomes a `ValidationError` at the boundary, not silent data loss at LLM translation time.
- Add semantic invariants (D6/D7/D8) so the regression cannot recur without CI catching it.
- Prove the pipeline produces non-empty entity output on D1 SRS by running it live and committing the resulting `domain/model.json` as an artifact.

### Non-goals

- File modularization of `architect.py` (single 1148-LOC file stays as-is; that's WP-01d's territory).
- RAG / Validator-pipeline changes (separate WPs).
- Adding tool-use to Specialist (M4 in the broader architectural menu — deferred).
- Multi-pass Specialist or Hierarchical Architect (deferred).
- CRITIC stage (M2 — deferred until M0+M1 land).
- Cross-context Reflexion stage (M3 — deferred).
- Markdown-fence stripping for OSS models (deferred; user paused OSS work).

---

## 3. Architecture decisions (and rationale)

### A1. Single WP for M0 + M1 (not two)

Typed contracts and deterministic Synthesizer are coupled: Specialist must emit a typed object before Synthesizer can passthrough that object deterministically. Splitting M0 from M1 would mean M0 ships a deterministic Synthesizer that *still* consumes dicts — same silent-loss surface area, just one layer down. Better to land both atomically.

### A2. LLM-enrich-only Synthesizer

After the refactor, Synthesizer:
- Takes a `List[SpecialistAnalysis]` (Pydantic-typed).
- Builds the final `DomainModel` by deterministically copying every entity / VO / service / aggregate / domain_event into the right `BoundedContext` slot. No LLM involved in data copy.
- Issues ONE narrow LLM call to enrich the otherwise-deterministic skeleton with: `Entity.synonyms_to_avoid` (per entity, since synonyms are creative and context-bound) and `BoundedContext.allowed_dependencies` (inferred from cross-context entity references, with the LLM disambiguating naming).
- Generates `ProjectMetadata` (project_name, generated_at, description) mechanically.
- Generates `GlobalRules` from a static template (banned_global_terms is project-policy, not document-derived).

This drops Synthesizer from one full omnibus LLM call to one narrow enrichment call (token cost ↓ ~80%), and makes silent entity loss impossible by construction.

### A3. Pydantic typed contracts at all 5 boundaries

Each stage emits a typed output that the next stage consumes. The boundary is a `model_validate(...)` call that raises `ValidationError` on mismatch (fail-fast per AGENTS.md). The current dict-shape passing is replaced.

| Stage | Input type | Output type |
|---|---|---|
| Scout | `str` (raw SRS text) | `ScoutOutput` (`sentences: List[SectionedSentence]`, `chunk_metadata`) |
| Architect | `ScoutOutput` | `ArchitectOutput` (`contexts: List[ContextHypothesis]`, `open_questions: List[str]`) |
| Specialist | `ArchitectOutput` | `List[SpecialistAnalysis]` (each: `context, entities, value_objects, services, aggregates, domain_events, business_rules, ambiguities`) |
| Verifier | `List[SpecialistAnalysis]` | `VerifierResult` (`issues: List[VerifierIssue]`, `semantic_checks: List[SemanticCheck]`) |
| Refiner | `VerifierResult + List[SpecialistAnalysis]` | `List[SpecialistAnalysis]` (corrected) |
| Synthesizer | `List[SpecialistAnalysis]` | `DomainModel` (existing strict schema in `core/schemas.py`) |

The existing `Entity`, `ValueObject`, `Service`, `Aggregate`, `DomainEvent`, `BoundedContext`, `DomainModel` Pydantic classes in `core/schemas.py` are reused for content. New classes are stage-boundary envelopes.

### A4. New Verifier semantic invariants (D6/D7/D8)

| Check | Description | Failure mode caught |
|---|---|---|
| **D6** Entity-count preservation | `sum(len(a.entities) for a in specialist_analyses) == sum(len(bc.ubiquitous_language.entities) for bc in domain_model.bounded_contexts)` | FM-LOST: silent loss during Synthesizer |
| **D7** Entity-name traceability | Every `Entity.name` in `DomainModel` must exist (case-insensitive) in some `SpecialistAnalysis.entities[*].name` | Fabrication: Synthesizer invents an entity |
| **D8** Aggregate-member referential integrity | For every `BoundedContext.aggregates[*].members[*]`, the referenced entity name exists in the SAME context's `entities[*].name` list | Dangling reference: aggregate references nonexistent entity |

These are cross-stage invariants enforced AFTER Synthesizer runs. Failure → Refiner retry; if Refiner can't fix → hard `SynthesizerInvariantError` (no silent degradation).

### A5. Strict mode for stage validators, but NOT `extra='forbid'`

Per AGENTS.md "no silent fallbacks", every boundary `.model_validate(...)` raises on type/required-field mismatch. We do **not** use `model_config = ConfigDict(extra='forbid')` because LLMs occasionally emit slightly-extra fields ("metadata", "reasoning") that are harmless — forbid would retry-storm. Extras are silently ignored at the Pydantic layer.

### A6. Same Pydantic schema for content; new envelope schemas only

`Entity`, `ValueObject`, etc. in `core/schemas.py` stay unchanged. New file `core/pipeline_contracts.py` adds the stage-envelope classes (`ScoutOutput`, `ArchitectOutput`, `SpecialistAnalysis`, `VerifierResult`). Refiner consumes Verifier output + Specialist input; no new Refiner-specific envelope needed.

### A7. Stage-by-stage commits, single feature branch, FF merge

Same pattern as WP-NEW-B Stage 1 (proven this session). 6-8 commits, each green-CI, rollback-able per commit. FF merge to main when full chain is green. No push to origin until user explicitly OKs.

---

## 4. Component design

### 4.1 New file: `core/pipeline_contracts.py`

```python
"""Typed contracts for stage boundaries in the domain-model pipeline.

Each stage produces and consumes a typed envelope. Boundary validation
is enforced via Pydantic .model_validate() at every transition. A
schema mismatch raises ValidationError, never silent loss.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from core.schemas import (
    Entity, ValueObject, Service, Aggregate, DomainEvent, BoundedContext,
)


class SectionedSentence(BaseModel):
    """A single Scout-extracted sentence with section provenance."""
    index: int = Field(ge=0)
    text: str
    section: Optional[str] = None


class ChunkMetadata(BaseModel):
    """Scout pass diagnostic info."""
    chunk_count: int
    total_chars: int
    truncated_chunks: int = 0


class ScoutOutput(BaseModel):
    """Output of the Scout stage: numbered domain-relevant sentences."""
    sentences: List[SectionedSentence]
    chunk_metadata: ChunkMetadata


class ContextHypothesis(BaseModel):
    """Architect's per-context proposal."""
    context_name: str
    description: str
    supporting_sentence_ids: List[int] = Field(default_factory=list)


class ArchitectOutput(BaseModel):
    """Output of the Architect stage: identified contexts + uncertainty."""
    contexts: List[ContextHypothesis]
    open_questions: List[str] = Field(
        default_factory=list,
        description="Architect-flagged ambiguities (informational; not a fail-fast)",
    )


class Ambiguity(BaseModel):
    """Specialist-flagged uncertainty about an emission."""
    target: str  # e.g. "entity:Customer"
    reason: str


class SpecialistAnalysis(BaseModel):
    """Per-context Specialist output. Multiple of these come back from
    `extract_per_context_details`."""
    context: ContextHypothesis
    entities: List[Entity] = Field(default_factory=list)
    value_objects: List[ValueObject] = Field(default_factory=list)
    services: List[Service] = Field(default_factory=list)
    aggregates: List[Aggregate] = Field(default_factory=list)
    domain_events: List[DomainEvent] = Field(default_factory=list)
    business_rules: List[str] = Field(default_factory=list)
    ambiguities: List[Ambiguity] = Field(default_factory=list)


class VerifierIssue(BaseModel):
    """One issue surfaced by a Verifier check."""
    severity: str  # "ERROR" | "WARN"
    check_id: str  # "D1" | "D2" | ... | "S1" | "D6" | "D7" | "D8"
    target: str
    message: str


class VerifierResult(BaseModel):
    """Verifier output: deterministic + semantic issues."""
    is_ok: bool
    issues: List[VerifierIssue] = Field(default_factory=list)
```

### 4.2 Synthesizer rewrite (deterministic merge + narrow LLM enrich)

New file: `core/synthesizer/` package (split from `architect.py`'s 70-line `synthesize` method):

- `core/synthesizer/__init__.py` — exports `synthesize_domain_model(analyses, llm_client) -> DomainModel`
- `core/synthesizer/merge.py` — deterministic merge: typed `List[SpecialistAnalysis]` → `DomainModel` skeleton
- `core/synthesizer/enrich.py` — narrow LLM call for `synonyms_to_avoid` + `allowed_dependencies`
- `core/synthesizer/metadata.py` — mechanical `ProjectMetadata` + `GlobalRules` defaults

```python
# core/synthesizer/__init__.py (pseudocode)
def synthesize_domain_model(
    analyses: List[SpecialistAnalysis],
    llm_client: LLMClient,
    project_name: str = "DomainModel",
) -> DomainModel:
    skeleton = build_skeleton_deterministic(analyses)
    enriched = enrich_with_llm(skeleton, llm_client)
    return finalize_with_metadata(enriched, project_name)
```

Old `synthesize` / `synthesize_final_model` methods on `DomainArchitect` become thin wrappers that delegate to the new package (preserves public call sites in `main.py`).

### 4.3 Verifier extensions

Existing `core/verifier/` package gets three new checks. Per existing convention, each check is a function returning `List[VerifierIssue]`:

- `core/verifier/checks_semantic.py` (extend or new module):
  - `check_d6_entity_count_preservation(analyses, domain_model) -> List[VerifierIssue]`
  - `check_d7_entity_name_traceability(analyses, domain_model) -> List[VerifierIssue]`
  - `check_d8_aggregate_member_referential_integrity(domain_model) -> List[VerifierIssue]`

D6, D7, D8 run AFTER Synthesizer, BEFORE Refiner. If any returns ERROR severity, Refiner is invoked. If Refiner cannot fix after retry budget, raise `SynthesizerInvariantError`.

### 4.4 Boundary enforcement in `architect.py`

Each stage method gains a `model_validate(...)` call on its output before passing to the next. Existing retry loops stay (they handle LLM-level parse failures). New validation failures route to the same retry path.

```python
# Pseudocode for Specialist
def extract_per_context_details(self, contexts: ArchitectOutput) -> List[SpecialistAnalysis]:
    results = []
    for ctx in contexts.contexts:
        raw_dict = self._call_llm_for_context(ctx, ...)
        try:
            analysis = SpecialistAnalysis.model_validate({
                "context": ctx.model_dump(),
                **raw_dict,
            })
        except ValidationError as e:
            # Existing retry logic — feed e into Refiner
            ...
        results.append(analysis)
    return results
```

### 4.5 Refiner integration

Refiner's existing retry loop is preserved; the input/output now flows typed objects. New `VerifierResult.is_ok==False` triggers the loop. Refiner has access to the typed `List[SpecialistAnalysis]` to emit corrections.

---

## 5. Data flow (post-refactor)

```
raw SRS text
   │
   ▼  Scout._extract_sentences_from_chunk
   │
   ▼   ScoutOutput (typed)  [boundary validate]
   │
   ▼  Architect.identify_contexts
   │
   ▼   ArchitectOutput (typed)  [boundary validate]
   │
   ▼  Specialist.extract_per_context_details
   │
   ▼   List[SpecialistAnalysis] (typed)  [boundary validate per item]
   │
   ▼  Verifier (D1..D5 + S1 existing, new D6/D7/D8 deferred until after Synthesizer)
   │
   ▼   VerifierResult (typed)
   │
   ▼  Refiner (if !is_ok)  → re-invokes Specialist with feedback
   │
   ▼  Synthesizer (deterministic merge → narrow LLM enrich → metadata)
   │
   ▼   DomainModel (typed, strict)  [boundary validate]
   │
   ▼  Verifier D6/D7/D8 (cross-stage invariants on DomainModel + analyses)
   │
   ▼   if D6/D7/D8 fail → Refiner → re-Synthesize
   │
   ▼  domain/model.json (persisted)
```

---

## 6. Verification (3-layer)

### 6.1 Unit tests (TDD per commit)

- `tests/test_pipeline_contracts.py` — schema construction, round-trip, validation-error on bad input
- `tests/test_synthesizer_deterministic.py` — mock `List[SpecialistAnalysis]` → assert entity count preserved, aggregate members reference real entities, no fabrication
- `tests/test_synthesizer_enrich.py` — mock LLM client, assert enrichment narrowly populates `synonyms_to_avoid` + `allowed_dependencies` without touching entity data
- `tests/test_verifier_d6_d7_d8.py` — synthetic mismatch scenarios, assert each check fires with expected `VerifierIssue` severity

### 6.2 Replay against historical intermediate

Load `core/intermediate/20260313_221928_3_specialist.json` (the run that demonstrated the bug). Build `List[SpecialistAnalysis]` from it. Run the new deterministic Synthesizer. Assert:

- `User` entity in `UserManagement` context: present, attributes preserved
- `Product` entity in `ProductCatalog` context: present, attributes preserved
- 6 bounded contexts in output, matching input
- Total entity count > 0

This catches the FM-LOST regression without spending live LLM tokens.

### 6.3 Live re-baseline on D1 SRS

After unit + replay green:

```bash
cd extension/backend
python -m core.architect --srs inputs/SRS.docx --out domain/model.json
```

Or equivalent invocation. ~$0.50 expected cost, ~10 min wall time (5 stages × Gemini calls + 1 narrow enrich call).

Acceptance: fresh `domain/model.json` shows entity count > 0 in at least 4 of 6 contexts (Search and similar utility contexts may legitimately be VO-only). Manifest captures git_commit, timestamps, package versions, prompts (similar pattern to WP-NEW-B Stage 1).

---

## 7. Risks + mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Specialist LLM output occasionally lacks `evidence_sentence_indices` (strict-required field) | High | Boundary validation fails → retry-storm | Refiner has retry budget (existing); if all retries fail, fail loudly (current behavior preserved). Specialist prompt already emphasizes `evidence_sentence_indices` per D1 patch. |
| Replay test fails because Mar-13 Specialist data doesn't have post-D1 schema fields | Cert. | Replay can't validate strict schema | Replay loader is permissive: fills `confidence=0.5`, `justification="(legacy)"`, `evidence_sentence_indices=[0]` for missing fields. Test asserts entity *count* preservation, not field-by-field fidelity. |
| Live run cost overrun | Low | Wasted $0.50 | Probe cost first with `--trials 1` smoke. |
| `architect.py` is a 1148-LOC monolith; refactoring methods in-place is messy | Med | Bigger diff, harder to review | Stage-by-stage commits each <200 LOC delta. New Synthesizer logic in `core/synthesizer/` package. `architect.py` methods become thin wrappers. |
| Refiner retry semantics interact with new D6/D7/D8 in unexpected ways | Med | Infinite retry loop | Hard cap retries at 5 (existing convention). On exhaustion, raise `SynthesizerInvariantError` — no silent degradation. |
| `extra='forbid'` not used — LLM emits unexpected fields | Low | Silently ignored, no data loss | Pydantic strict on required fields enforces fail-fast for missing; extras are tolerated. Document in spec (this section). |
| Existing intermediate JSON dumps don't match new schemas | Cert. | Replay test against MAR-13 needs adapter | Replay loader has known-good adapter (above). Old dumps stay untouched; new runs use new format. |

---

## 8. Acceptance criteria

- [ ] 8 commits land green-CI on `feat/typed-pipeline-deterministic-synthesizer`
- [ ] All existing 237 unit tests still pass
- [ ] New tests: ~15-20 unit tests covering contracts, deterministic Synthesizer, D6/D7/D8
- [ ] Replay test green against `core/intermediate/20260313_221928_3_specialist.json`
- [ ] Live run on D1 SRS produces `domain/model.json` with non-empty entities in ≥4 of 6 bounded contexts
- [ ] Fresh manifest sidecar in `runs/` carries git_commit, package_versions, stage prompts (mirror WP-NEW-B Stage 1)
- [ ] `domain/model.json` artifact committed in the final chore commit
- [ ] No `gemini.py`, `ollama.py`, RAG, or Validator code touched
- [ ] FF merge to main; no push without explicit user approval
- [ ] `development_docs/WP-CORE-1-typed-pipeline.md` written and indexed

---

## 9. Out of scope (explicit)

- File modularization of `architect.py` into `core/architect/{scout,architect,specialist,...}.py` — defer to WP-01d
- Adding ReAct tool-use to Specialist (M4)
- CRITIC stage between Verifier and Synthesizer (M2 — defer to follow-up WP after this lands)
- Cross-context Reflexion stage (M3)
- Multi-pass Specialist (M5)
- Hierarchical Architect (M7)
- Markdown-fence stripping for OSS models — paused per user's OSS-defer pivot
- Token tracker normalization for the new narrow Synthesizer call (defer to WP-01c)
- Run-spec orchestrator (defer to WP-01b)

---

## 10. Cross-references

- [[WP-NEW-B-Stage-1-schema-probe]] — same Codex-juri-then-spec workflow pattern; `runs/` artifact format established here
- `core/schemas.py:40-167` — existing Entity / ValueObject / etc. Pydantic schemas (reused, not modified)
- `core/architect.py:766-837` — current Synthesizer (will be replaced by `core/synthesizer/` package)
- `core/verifier/` — existing D1-D5 + S1 checks (will be extended with D6/D7/D8)
- `core/intermediate/20260313_221928_3_specialist.json` — replay input
- `todos/MASTER_PLAN.md` — this WP is NEW (WP-CORE-1), not in original 23-WP table; rationale documented above

---

## 11. Open questions for review

None at this draft stage. All 6 strategic decisions (single-WP scope, LLM-enrich-only Synthesizer, all-5-stage typed contracts, D6+D7+D8 invariants, triple-layer verification, stage-by-stage commits) were resolved during the brainstorming dialog before this spec was written.
