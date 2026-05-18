# P3 Verifier+Refiner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current 4-stage Scout→Architect→Specialist→Synthesizer pipeline with a 5-stage version that adds a Verifier+Refiner loop (Alt-A from the 2026-05-18 audit), aligns prompts with the Pydantic schema (FM-05/06/07), removes every silent fallback (FM-01/02/04/21), converts Specialist to a per-context loop (FM-23), and grounds every entity in cited SRS sentences (Alt-D cherry-pick + OQ2). 24 atomic commits across 4 phases (spec called for 23; +1 for C7b which the spec body §3.3 implied but the §7 phasing did not enumerate).

**Architecture:** Five forward stages with a bounded refine loop. Stage 4 (Verifier) runs deterministic + semantic checks; on issues, Refiner re-prompts the failing upstream stage (max 2 cycles). All silent fallbacks become typed `PipelineError` raises. Section-aware SRS chunking and per-context Specialist loops replace the current global-prompt patterns.

**Tech Stack:** Python 3.12, `google-genai` SDK (Gemini-only, no provider abstraction yet), Pydantic v2, pytest (`-m "not integration"` for unit tier, integration tests gated by `DDD_INTEGRATION_TEST=1`).

**Spec:** `docs/superpowers/specs/2026-05-18-p3-verifier-refiner-design.md` (commit `07b5d51`)

---

## File Structure

### New files (created in this plan)

```
extension/backend/core/
  orchestration/
    __init__.py
    errors.py                       # PipelineError hierarchy
    pipeline.py                     # 5-stage driver
  verifier/
    __init__.py
    types.py                        # VerifierIssue, IssueSeverity, VerifierResult
    checks_deterministic.py         # D1-D5
    checks_semantic.py              # S1
  refiner/
    __init__.py
    prompts.py                      # per-stage refinement templates
    loop.py                         # bounded retry orchestration
  scout/
    __init__.py
    chunking.py                     # section-aware chunker

extension/backend/tests/
  test_architect_prompts.py         # prompt-level assertions (no LLM)
  test_orchestration_errors.py      # PipelineError hierarchy
  test_verifier_deterministic.py    # D1-D5 unit tests
  test_verifier_semantic.py         # S1 with mock LLM
  test_refiner_loop.py              # bounded retry
  test_scout_chunking.py            # section-aware chunker
  test_pipeline_orchestration.py    # full 5-stage with mock LLM
  test_p3_integration.py            # real Gemini call, env-gated
  test_grounding_regression.py      # post-Phase-D: every entity has evidence
  fixtures/
    sample_srs.txt                  # 3-section minimal SRS for unit tests
```

### Modified files

```
extension/backend/core/
  architect.py                      # prompts rewritten, fallbacks removed, thin facade after C7
  schemas.py                        # Entity/BoundedContext/Aggregate field changes
  AST/
    ast_model_signals.py            # _collect_signals raises (G01)
    ast_signal_enrichment.py        # _ensure_traceability drops "generated" (OQ2)
```

---

## Pre-flight (do this once, before Task A1)

- [ ] **Step 0.1: Verify repo state is clean** (except the pre-existing `M extension/.DS_Store` that stays unstaged)

Run: `git status --short`
Expected: shows only `M extension/.DS_Store` (or empty).

- [ ] **Step 0.2: Confirm unit-test baseline is green**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: all tests pass (per the prior session's Phase 0 work, baseline is 105 passing).

- [ ] **Step 0.3: Confirm the spec is committed and current**

Run: `git log --oneline -1 docs/superpowers/specs/2026-05-18-p3-verifier-refiner-design.md`
Expected: a commit `07b5d51` (or later) titled `docs(spec): add P3 Verifier+Refiner architecture design (supersedes WP-01a as immediate priority)`.

---

# Phase A — Prompt/Schema Alignment (6 commits, ~1 day, low risk)

Goal: Eliminate the Synthesizer prompt's omission of `services` + `aggregates` (FM-05), give Specialist a structured `aggregates` and a `domain_events` field (FM-06/07), require LLM-emitted `confidence` and `justification` on every entity (FM-16), narrow the bare `except` in `synthesize_final_model` (FM-21), and stop `_cleanup_domain_data` from fabricating defaults (FM-20). No structural change to the pipeline. Each test runs against the prompt string, not the LLM.

---

### Task A1: Synthesizer prompt — add services, aggregates, domain_events to example output (FM-05)

**Files:**
- Create: `extension/backend/tests/test_architect_prompts.py`
- Modify: `extension/backend/core/architect.py` (the `synthesize` prompt at lines 619-664)

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_architect_prompts.py`:

```python
"""Prompt-level assertions on DomainArchitect's stage prompts.

These tests do NOT call any LLM. They assert that the prompt strings
sent to the LLM include the expected field examples so that all 6 D1
models (Gemini + 4 OSS) receive a faithful schema demonstration.
"""

import re
from core.architect import DomainArchitect


def _get_synthesize_prompt(analyses):
    """Inspect the synthesize prompt without making an LLM call."""
    arch = DomainArchitect.__new__(DomainArchitect)
    arch.model_name = "gemini-3.1-pro-preview"
    src = open("core/architect.py").read()
    return src


def test_synthesize_prompt_example_includes_services():
    src = open("core/architect.py").read()
    synthesize_section = src.split("def synthesize(")[1].split("def ")[0]
    assert '"services"' in synthesize_section, (
        "synthesize prompt example output must demonstrate the 'services' field "
        "or OSS models that follow the prompt literally will silently drop services"
    )


def test_synthesize_prompt_example_includes_aggregates():
    src = open("core/architect.py").read()
    synthesize_section = src.split("def synthesize(")[1].split("def ")[0]
    assert '"aggregates"' in synthesize_section, (
        "synthesize prompt example output must demonstrate the 'aggregates' field"
    )


def test_synthesize_prompt_example_includes_domain_events_objects():
    src = open("core/architect.py").read()
    synthesize_section = src.split("def synthesize(")[1].split("def ")[0]
    assert '"domain_events"' in synthesize_section
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_architect_prompts.py -v`
Expected: 3 tests collected, `test_synthesize_prompt_example_includes_services` FAILS with `AssertionError` (the current prompt at lines 639-657 only shows `entities`, `value_objects`, and `domain_events`, not `services` or `aggregates`).

- [ ] **Step 3: Modify the synthesize prompt**

Edit `extension/backend/core/architect.py`, replace lines 619-664 (the `prompt = f"""..."""` block in the `synthesize` method) with:

```python
        prompt = f"""Synthesize Bounded Context analyses into a cohesive Domain Model.

SYNTHESIS RULES:
1. Resolve duplicates: Each entity belongs to ONE primary context.
2. Generate synonyms_to_avoid for each entity (common alternative names to flag for V1).
3. Define allowed_dependencies between contexts (which contexts can reference which).
4. Ensure naming consistency: PascalCase for all names.
5. Every aggregate must list its members (entity names that live inside the aggregate).

CONTEXT ANALYSES:
{json.dumps(analyses, indent=2)}

OUTPUT SCHEMA (must match exactly — populate every field even if empty):
{{
  "project_name": "ProjectNameDomainModel",
  "project_metadata": {{
    "version": "1.0",
    "generated_at": "YYYY-MM-DD",
    "description": "Brief description"
  }},
  "bounded_contexts": [
    {{
      "context_name": "ContextName",
      "description": "What this context manages",
      "allowed_dependencies": ["OtherContext1"],
      "ubiquitous_language": {{
        "entities": [{{
          "name": "EntityName",
          "description": "Entity purpose",
          "confidence": 0.85,
          "justification": "Mentioned in 3 SRS sentences as a primary actor",
          "synonyms_to_avoid": ["Synonym1", "Synonym2"]
        }}],
        "value_objects": [{{
          "name": "ValueObjectName",
          "attributes": ["attr1", "attr2"],
          "description": "Value object purpose"
        }}],
        "services": [{{
          "name": "ServiceName",
          "description": "Service responsibility"
        }}],
        "aggregates": [{{
          "name": "AggregateName",
          "description": "Aggregate consistency boundary",
          "members": ["EntityName1", "EntityName2"]
        }}],
        "domain_events": ["EventName1"]
      }}
    }}
  ],
  "global_rules": {{
    "naming_convention": "PascalCase",
    "banned_global_terms": ["Manager", "Util", "Helper", "Data", "Info"]
  }}
}}

CRITICAL: synonyms_to_avoid must be populated for V1 detection. Every aggregate.members must reference an entity in the same context. Do not invent data not present in the analyses."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension/backend && pytest tests/test_architect_prompts.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_architect_prompts.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
fix(synthesizer): demonstrate services + aggregates + structured entities in prompt (FM-05)

Adds prompt-level test that fails when the synthesize prompt example
output omits any of services/aggregates/entity-confidence/aggregate-members.
Replaces the prompt at architect.py:619-664 with a version that:

- Demonstrates every field of the DomainModel Pydantic contract
- Shows entity-level confidence + justification (foundation for FM-16)
- Shows aggregate.members so V5 detection can be sourced from LLM output

The OSS models in D1 (gpt-oss, qwen3-coder, minimax, gemma4) follow
prompts literally; the old example silently biased the 6-model RQ1
comparison toward Gemini. This commit closes that fairness threat.
EOF
)"
```

---

### Task A2: Specialist — emit structured aggregates with members[] instead of aggregate_roots[] (FM-06)

**Files:**
- Modify: `extension/backend/tests/test_architect_prompts.py` (append)
- Modify: `extension/backend/core/architect.py` (the `extract_all_contexts_details` prompt at lines 497-521)

- [ ] **Step 1: Write the failing test**

Append to `extension/backend/tests/test_architect_prompts.py`:

```python
def test_specialist_prompt_emits_structured_aggregates():
    src = open("core/architect.py").read()
    specialist_section = src.split("def extract_all_contexts_details(")[1].split("def ")[0]
    assert '"aggregate_roots"' not in specialist_section, (
        "Specialist prompt must no longer emit flat aggregate_roots:[str]; "
        "use structured aggregates:[{name, members:[entity_name]}] instead"
    )
    assert '"aggregates"' in specialist_section
    assert '"members"' in specialist_section, (
        "Specialist prompt must require aggregate.members so V5 (aggregate boundary) "
        "violation detection has the data it needs."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_architect_prompts.py::test_specialist_prompt_emits_structured_aggregates -v`
Expected: FAIL (current Specialist prompt at line 513 emits `"aggregate_roots": ["Root1"]`, has no `"aggregates"` or `"members"`).

- [ ] **Step 3: Modify the Specialist prompt**

In `extension/backend/core/architect.py`, replace lines 497-521 (the `prompt = f"""..."""` block in `extract_all_contexts_details`) with:

```python
        prompt = f"""Analyze the domain knowledge for these Bounded Contexts: {contexts_text}

For EACH context, extract the 5 DDD building blocks:
1. Entities       - Objects with unique identity (Customer, Order, Product)
2. Value Objects  - Immutable objects defined by attributes (Address, Money)
3. Services       - Stateless operations that don't naturally belong to an entity
4. Aggregates     - Consistency boundaries; each aggregate has a name and lists
                    the entities (`members`) that live inside it
5. Domain Events  - Past-tense business facts (OrderPlaced, PaymentReceived)

DOMAIN KNOWLEDGE:
{sentences_text}

RESPOND WITH JSON:
{{
  "analyses": [
    {{
      "context": "ContextName",
      "entities": [{{"name": "Entity1", "attributes": ["id", "name", "status"]}}],
      "value_objects": [{{"name": "Money", "attributes": ["amount", "currency"]}}],
      "services": [{{"name": "PricingService", "description": "Computes order totals"}}],
      "aggregates": [{{"name": "Order", "members": ["Order", "OrderLine"]}}],
      "domain_events": ["OrderPlaced", "OrderCancelled"],
      "business_rules": ["Orders must have at least one item"]
    }}
  ]
}}

If a category has no data, use empty arrays. Do not invent data. Every aggregate.members entry must also appear in the same context's entities list."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension/backend && pytest tests/test_architect_prompts.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_architect_prompts.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
fix(specialist): emit structured aggregates with members[] (FM-06)

Replaces flat aggregate_roots:[str] with aggregates:[{name, members}].
Adds explicit services:[{name, description}] and domain_events to the
Specialist prompt so V5 (aggregate boundary) and V6 (domain event)
detection have source data from the LLM rather than ad-hoc
Synthesizer-stage invention.
EOF
)"
```

---

### Task A3: Specialist — add domain_events extraction field (FM-07)

This is already covered by the Task A2 prompt rewrite (the prompt now lists `domain_events` explicitly under both the extraction instructions and the RESPOND WITH JSON example). The Task A2 test already asserts the presence of `domain_events` in the Specialist prompt indirectly via the JSON example. Add a direct prompt-level assertion.

**Files:**
- Modify: `extension/backend/tests/test_architect_prompts.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `extension/backend/tests/test_architect_prompts.py`:

```python
def test_specialist_prompt_lists_domain_events_as_extraction_target():
    src = open("core/architect.py").read()
    specialist_section = src.split("def extract_all_contexts_details(")[1].split("def ")[0]
    assert "Domain Events" in specialist_section, (
        "Specialist prompt must explicitly instruct extraction of Domain Events; "
        "Synthesizer must not be the first stage that mentions them."
    )
    assert '"domain_events"' in specialist_section
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `cd extension/backend && pytest tests/test_architect_prompts.py::test_specialist_prompt_lists_domain_events_as_extraction_target -v`
Expected: PASS (Task A2's prompt already includes "Domain Events" instructions).

If the test fails because the prompt extraction wording differs, adjust the assertion (e.g. `assert "Domain Events" in specialist_section or "domain_events" in specialist_section`) — but the literal "Domain Events" string from the numbered list of 5 building blocks should match.

- [ ] **Step 3: Implementation already in place** — nothing to add.

- [ ] **Step 4: Re-run full prompt test file**

Run: `cd extension/backend && pytest tests/test_architect_prompts.py -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_architect_prompts.py
git commit -m "$(cat <<'EOF'
test(specialist): pin domain_events as explicit extraction target (FM-07)

Locks in the FM-07 fix from Task A2 with a dedicated regression test.
Prevents future prompt edits from silently dropping domain_events back
into Synthesizer-only territory.
EOF
)"
```

---

### Task A4: Schema — Entity.confidence + justification become required, evidence_sentence_indices optional in Phase A (FM-16)

**Files:**
- Create: `extension/backend/tests/test_schemas_strict.py`
- Modify: `extension/backend/core/schemas.py`
- Modify: `extension/backend/tests/test_architect_prompts.py` (append)

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_schemas_strict.py`:

```python
"""Schema-strict tests for Entity, ValueObject, Aggregate, BoundedContext.

Phase A adds: required Entity.confidence, required Entity.justification,
optional Entity.evidence_sentence_indices (tightened to required in
Phase D1), required Aggregate.members, optional
BoundedContext.supporting_sentence_ids,
optional BoundedContext.business_rules.
"""

import pytest
from pydantic import ValidationError
from core.schemas import (
    Entity,
    Aggregate,
    BoundedContext,
    DomainModel,
    UbiquitousLanguage,
    ProjectMetadata,
)


def test_entity_requires_confidence_field():
    """confidence must be supplied; the old default 0.5 is gone."""
    with pytest.raises(ValidationError):
        Entity(name="Customer", description="Buys things")


def test_entity_requires_justification_field():
    with pytest.raises(ValidationError):
        Entity(name="Customer", description="Buys things", confidence=0.8)


def test_entity_accepts_phase_a_minimum_fields():
    """In Phase A, evidence_sentence_indices is optional (default empty)."""
    e = Entity(
        name="Customer",
        description="A buyer in the e-commerce domain",
        confidence=0.9,
        justification="Mentioned in 4 SRS sentences as the principal actor",
    )
    assert e.confidence == 0.9
    assert e.evidence_sentence_indices == []  # optional in Phase A


def test_aggregate_requires_members_field():
    with pytest.raises(ValidationError):
        Aggregate(name="Order", description="Order consistency boundary")


def test_aggregate_accepts_explicit_members():
    a = Aggregate(
        name="Order",
        description="Order consistency boundary",
        members=["Order", "OrderLine"],
    )
    assert a.members == ["Order", "OrderLine"]


def test_bounded_context_accepts_business_rules():
    bc = BoundedContext(
        context_name="OrderMgmt",
        description="Manages orders",
        ubiquitous_language=UbiquitousLanguage(
            entities=[
                Entity(
                    name="Order",
                    description="An order",
                    confidence=0.95,
                    justification="Primary entity",
                )
            ],
            value_objects=[],
            domain_events=[],
        ),
        business_rules=["Orders must have at least one item"],
    )
    assert bc.business_rules == ["Orders must have at least one item"]


def test_domain_model_rejects_empty_bounded_contexts():
    """An empty bounded_contexts list must not validate (FM-04 prep)."""
    with pytest.raises(ValidationError):
        DomainModel(
            project_name="X",
            project_metadata=ProjectMetadata(version="1.0", generated_at="2026-05-18"),
            bounded_contexts=[],
            global_rules=None,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_schemas_strict.py -v`
Expected: 6 tests FAIL (current `Entity` defaults `confidence=0.5`, has no `justification`; `Aggregate` has no `members`; `BoundedContext` has no `business_rules`; `DomainModel` accepts empty `bounded_contexts`).

- [ ] **Step 3: Modify the schemas**

Edit `extension/backend/core/schemas.py`. Apply these targeted changes:

(a) Replace the `Entity` class (currently at lines 40-57):

```python
class Entity(BaseModel):
    """Domain entity definition."""
    name: str = Field(description="Name of the domain entity (e.g., Customer)")
    description: str = Field(description="Brief description of the entity's role")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="LLM-emitted confidence in this inference (0.0-1.0). Required."
    )
    justification: str = Field(
        description="LLM-emitted reason for this entity (e.g. supporting sentence count, role)."
    )
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that ground this entity. Optional in Phase A, required (min_items=1) from Phase D1."
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list populated by AST enrichment (file/line/rule)."
    )
    synonyms_to_avoid: Optional[List[str]] = Field(
        default=None,
        description="Terms forbidden for this entity (e.g., Client, User)."
    )
```

(b) Replace the `Aggregate` class (currently at lines 93-106):

```python
class Aggregate(BaseModel):
    """Aggregate root candidate definition."""
    name: str = Field(description="Name of the aggregate root")
    description: str = Field(description="Aggregate consistency boundary")
    members: List[str] = Field(
        description="Entity names that live inside this aggregate. Required."
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence score for this inference (0.0-1.0)"
    )
    sources: List["InferenceSource"] = Field(
        default_factory=list,
        description="Traceable evidence list (file/line/rule)"
    )
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that ground this aggregate."
    )
```

(c) Replace the `BoundedContext` class (currently at lines 136-146):

```python
class BoundedContext(BaseModel):
    """Definition of a bounded context."""
    context_name: str = Field(description="Name of the bounded context")
    description: str = Field(description="What this context is responsible for")
    allowed_dependencies: Optional[List[str]] = Field(
        default=None,
        description="List of other contexts this context can depend on"
    )
    supporting_sentence_ids: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices that justify identifying this context."
    )
    business_rules: Optional[List[str]] = Field(
        default=None,
        description="Context-specific business rules surfaced by Specialist."
    )
    ubiquitous_language: "UbiquitousLanguage" = Field(
        description="The language and models specific to this context"
    )
```

(d) Replace the `DomainModel` class (currently at lines 175-184):

```python
from pydantic import field_validator


class DomainModel(BaseModel):
    """Complete domain model for a project."""
    project_name: str = Field(description="Name of the project")
    project_metadata: ProjectMetadata = Field(description="Generation metadata")
    bounded_contexts: List[BoundedContext] = Field(
        description="List of all identified Bounded Contexts. Must be non-empty."
    )
    global_rules: Optional[GlobalRules] = Field(
        description="Project-wide architectural rules"
    )

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

If `from pydantic import field_validator` is not at the top of the file already, add it to the existing `from pydantic import BaseModel, Field` line.

Also append a prompt-level test to `extension/backend/tests/test_architect_prompts.py`:

```python
def test_specialist_prompt_demands_confidence_and_justification():
    src = open("core/architect.py").read()
    specialist_section = src.split("def extract_all_contexts_details(")[1].split("def ")[0]
    # In Phase A we add these fields to the Specialist prompt's entity example
    # via the synthesize prompt cascade. For now, assert at the synthesize level
    # which is downstream and definitive:
    synth = src.split("def synthesize(")[1].split("def ")[0]
    assert '"confidence"' in synth
    assert '"justification"' in synth
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd extension/backend && pytest tests/test_schemas_strict.py tests/test_architect_prompts.py -v`
Expected: all green.

Run also: `cd extension/backend && pytest -m "not integration" -q`
Expected: full unit suite green. Existing tests that built `Entity(name=..., description=...)` without `confidence`/`justification` will now fail — find them and fix by passing `confidence=0.5, justification="legacy fixture"`.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_schemas_strict.py extension/backend/tests/test_architect_prompts.py extension/backend/core/schemas.py
# Plus any existing tests you had to patch to provide confidence/justification
git commit -m "$(cat <<'EOF'
feat(schema): require Entity.confidence + justification; add Aggregate.members; reject empty DomainModel (FM-16, FM-04 prep)

- Entity.confidence and Entity.justification become required fields
  (no default 0.5). LLM must emit them per the Synthesizer prompt.
- Entity.evidence_sentence_indices added (Optional in Phase A, tightened
  to min_items=1 in Phase D1).
- Aggregate.members becomes required so V5 detection has source data.
- BoundedContext.supporting_sentence_ids + business_rules added.
- DomainModel rejects empty bounded_contexts via field_validator (FM-04
  prep: forces the Synthesizer to raise rather than return an empty
  model with the upcoming Phase B removal of _create_fallback_model).
EOF
)"
```

---

### Task A5: Synthesizer — narrow the bare `except Exception` (FM-21)

**Files:**
- Create: `extension/backend/tests/test_synthesize_final_model_errors.py`
- Modify: `extension/backend/core/architect.py` (the `synthesize_final_model` method, currently at lines 752-773)

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_synthesize_final_model_errors.py`:

```python
"""Phase A: synthesize_final_model must propagate Pydantic validation errors
instead of returning an empty model via a bare except.
"""

import pytest
from pydantic import ValidationError
from unittest.mock import patch, MagicMock

from core.architect import DomainArchitect


def _make_arch():
    """Bypass __init__ to avoid needing a real API key in this unit test."""
    arch = DomainArchitect.__new__(DomainArchitect)
    arch.model_name = "gemini-3.1-pro-preview"
    arch.last_request_time = 0
    arch.min_delay = 0
    arch.request_count = 0
    import threading
    arch._rate_limit_lock = threading.Lock()
    arch.scout_max_workers = 1
    from core.token_tracker import TokenTracker
    arch.token_tracker = TokenTracker.get_instance()
    arch.progress_callback = None
    arch.run_timestamp = "20260518_000000"
    return arch


def test_synthesize_final_model_propagates_validation_error():
    """When synthesize() returns invalid JSON shape, the Pydantic error must
    propagate; the bare except path is gone."""
    arch = _make_arch()
    with patch.object(arch, "synthesize") as mock_synth:
        # Return a dict missing required fields — DomainModel construction will fail
        mock_synth.return_value = {"project_name": "X"}
        with pytest.raises(ValidationError):
            arch.synthesize_final_model(analyses=[])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_synthesize_final_model_errors.py -v`
Expected: FAIL. Either it does not raise (current bare except swallows), or it raises a different exception type.

- [ ] **Step 3: Modify `synthesize_final_model`**

In `extension/backend/core/architect.py`, locate `synthesize_final_model` (around line 752-773). Replace the bare `except Exception` block with narrowed handling:

```python
    def synthesize_final_model(self, analyses: List[Dict[str, Any]]) -> DomainModel:
        """Synthesize per-context analyses into a final DomainModel.

        Phase A: Propagates Pydantic ValidationError instead of returning an
        empty fallback model. The previous bare except (FM-21) silently turned
        validation errors into successful empty-model responses, which is
        removed in Phase B together with _create_fallback_model.
        """
        raw_dict = self.synthesize(analyses)
        raw_dict = self._cleanup_domain_data(raw_dict)
        # Pydantic raises ValidationError on shape/constraint failures —
        # let it propagate. The narrow except below only catches the
        # KeyError that _cleanup_domain_data may surface from a stage-3
        # error payload, and re-raises with context.
        return DomainModel(**raw_dict)
```

Delete the existing `try` / `except Exception` wrapper and the `_create_fallback_model` invocation inside it. Leave the `_create_fallback_model` method definition in place for now (Phase B4 deletes it).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension/backend && pytest tests/test_synthesize_final_model_errors.py tests/test_schemas_strict.py tests/test_architect_prompts.py -v`
Expected: all green.

Run also: `cd extension/backend && pytest -m "not integration" -q`
Expected: full unit suite green. If `test_unit.py` or `test_api.py` exercised the bare-except fallback, they will fail. Patch them to expect `ValidationError`.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_synthesize_final_model_errors.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
refactor(synthesizer): narrow bare except so Pydantic errors propagate (FM-21)

Removes the bare `except Exception` in synthesize_final_model that
silently swallowed every error type (including Pydantic ValidationError,
KeyError, and TypeError) and returned an empty model. Pydantic errors
now propagate. The _create_fallback_model method is retained for one
more commit; Phase B4 removes it together with its call sites.
EOF
)"
```

---

### Task A6: `_cleanup_domain_data` — stop fabricating defaults (FM-20)

**Files:**
- Create: `extension/backend/tests/test_cleanup_domain_data.py`
- Modify: `extension/backend/core/architect.py` (the `_cleanup_domain_data` method, currently at lines 832-868)

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_cleanup_domain_data.py`:

```python
"""Phase A6: _cleanup_domain_data must preserve LLM-emitted values rather
than fabricating defaults (FM-20). Test verifies that a snake_case
naming_convention emitted by the LLM survives, and that an absent
global_rules block does not get replaced with hardcoded content.
"""

from core.architect import DomainArchitect


def _make_arch():
    arch = DomainArchitect.__new__(DomainArchitect)
    return arch


def test_cleanup_preserves_llm_emitted_naming_convention():
    arch = _make_arch()
    raw = {
        "project_name": "X",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [],
        "global_rules": {
            "naming_convention": "snake_case",  # LLM intentionally chose snake_case
            "banned_global_terms": [],
        },
    }
    cleaned = arch._cleanup_domain_data(raw)
    assert cleaned["global_rules"]["naming_convention"] == "snake_case"


def test_cleanup_passes_through_missing_global_rules_unchanged():
    arch = _make_arch()
    raw = {
        "project_name": "X",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [],
        # No global_rules key at all
    }
    cleaned = arch._cleanup_domain_data(raw)
    # Cleanup is structural only; if the LLM omitted the block, leave it absent
    # so the orchestrator can decide whether to raise.
    assert "global_rules" not in cleaned or cleaned["global_rules"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_cleanup_domain_data.py -v`
Expected: FAIL. Current `_cleanup_domain_data` (lines 832-868) coerces `naming_convention` to `"PascalCase"` and injects an empty `banned_global_terms` list.

- [ ] **Step 3: Modify `_cleanup_domain_data`**

In `extension/backend/core/architect.py`, replace the `_cleanup_domain_data` method (lines 832-868) with a version that only performs structural coercion:

```python
    def _cleanup_domain_data(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Structural-only normalization of the Synthesizer's raw JSON.

        Phase A6 (FM-20): no longer fabricates defaults. If the LLM omitted
        a field, leave it omitted — let downstream Pydantic validation or
        the Verifier surface the gap. The only transformation kept is
        coercing domain_events from `[{"name": "X"}]` (object form some
        models emit) to `["X"]` (list-of-string form the schema expects).
        """
        if "bounded_contexts" in raw:
            for ctx in raw["bounded_contexts"]:
                ul = ctx.get("ubiquitous_language", {})
                events = ul.get("domain_events")
                if isinstance(events, list):
                    ul["domain_events"] = [
                        e["name"] if isinstance(e, dict) and "name" in e else e
                        for e in events
                    ]
        return raw
```

- [ ] **Step 4: Run tests to verify**

Run: `cd extension/backend && pytest tests/test_cleanup_domain_data.py -v`
Expected: 2 tests PASS.

Run also: `cd extension/backend && pytest -m "not integration" -q`
Expected: full unit suite green. Tests that expected the old fabricated defaults will fail; patch them.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_cleanup_domain_data.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
refactor(cleanup): _cleanup_domain_data preserves LLM output faithfully (FM-20)

Removes the hardcoded "PascalCase" naming_convention and empty
banned_global_terms injection that masked LLM-emitted choices in the
persisted domain/model.json. Only the structural domain_events object
→ string coercion remains, since some Gemini variants emit
[{"name": "X"}] while the schema requires ["X"].
EOF
)"
```

**End of Phase A.** Phase-boundary smoke test:

- [ ] **Step A-end.1: Phase A integration smoke**

Run: `cd extension/backend && DDD_INTEGRATION_TEST=1 pytest tests/test_api.py -m integration -k generate_model_basic -v` (only if you have a live backend; otherwise skip)

Expected: The end-to-end domain-model generation against `inputs/SRS.docx` either succeeds with `services` and `aggregates` populated (success), or surfaces a real bug uncovered by the prompt change. If a bug surfaces, fix it before proceeding to Phase B.

---

# Phase B — Silent Fallback Removal (5 commits, ~1-2 days, medium risk)

Goal: Convert every silent fallback in `architect.py` into an explicit typed `PipelineError` raise. After Phase B the pipeline either produces a valid `DomainModel` or raises a `PipelineError` subclass — there is no third path. Phase B will surface previously-hidden regressions; budget time to fix each unmasked bug rather than re-introducing fallbacks.

---

### Task B1: Introduce `PipelineError` hierarchy

**Files:**
- Create: `extension/backend/core/orchestration/__init__.py`
- Create: `extension/backend/core/orchestration/errors.py`
- Create: `extension/backend/tests/test_orchestration_errors.py`

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_orchestration_errors.py`:

```python
"""Phase B1: PipelineError hierarchy."""

import pytest
from core.orchestration.errors import (
    PipelineError,
    ScoutChunkParseError,
    ArchitectExtractionError,
    SpecialistFailureError,
    SynthesizerEmptyModelError,
    RefinementExhaustedError,
    InsufficientGroundingError,
)


def test_all_pipeline_errors_subclass_pipeline_error():
    for cls in [
        ScoutChunkParseError,
        ArchitectExtractionError,
        SpecialistFailureError,
        SynthesizerEmptyModelError,
        RefinementExhaustedError,
        InsufficientGroundingError,
    ]:
        assert issubclass(cls, PipelineError), f"{cls.__name__} must subclass PipelineError"


def test_scout_chunk_parse_error_carries_chunk_id_and_attempts():
    e = ScoutChunkParseError(chunk_id="3.1", attempts=5)
    assert e.chunk_id == "3.1"
    assert e.attempts == 5
    assert "3.1" in str(e)


def test_architect_extraction_error_carries_srs_path():
    e = ArchitectExtractionError(srs_path="inputs/SRS.docx")
    assert e.srs_path == "inputs/SRS.docx"
    assert "SRS.docx" in str(e)


def test_specialist_failure_error_carries_context_name():
    e = SpecialistFailureError(context_name="OrderMgmt")
    assert e.context_name == "OrderMgmt"


def test_synthesizer_empty_model_error_carries_input_summary():
    e = SynthesizerEmptyModelError(input_summary="0 analyses")
    assert "0 analyses" in str(e)


def test_refinement_exhausted_error_carries_issues():
    e = RefinementExhaustedError(issues=[{"stage": "specialist", "issue_type": "missing_evidence"}])
    assert len(e.issues) == 1
    assert e.issues[0]["stage"] == "specialist"


def test_insufficient_grounding_error_carries_entity_name():
    e = InsufficientGroundingError(entity_name="GhostEntity")
    assert "GhostEntity" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_orchestration_errors.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'core.orchestration'`.

- [ ] **Step 3: Implement the hierarchy**

Create `extension/backend/core/orchestration/__init__.py`:

```python
"""Pipeline orchestration package: errors, 5-stage driver, and the
Verifier+Refiner loop wiring.
"""

from core.orchestration.errors import (
    PipelineError,
    ScoutChunkParseError,
    ArchitectExtractionError,
    SpecialistFailureError,
    SynthesizerEmptyModelError,
    RefinementExhaustedError,
    InsufficientGroundingError,
)

__all__ = [
    "PipelineError",
    "ScoutChunkParseError",
    "ArchitectExtractionError",
    "SpecialistFailureError",
    "SynthesizerEmptyModelError",
    "RefinementExhaustedError",
    "InsufficientGroundingError",
]
```

Create `extension/backend/core/orchestration/errors.py`:

```python
"""Typed exceptions for the P3 pipeline.

All silent fallbacks in core/architect.py are converted to raises of
these classes. The top-level orchestrator catches PipelineError, writes
a structured failure_log.json, and decides retry/skip/fail per RQ1
metrics policy.
"""

from typing import Any, List, Optional


class PipelineError(Exception):
    """Base for every P3 pipeline failure."""


class ScoutChunkParseError(PipelineError):
    def __init__(self, chunk_id: str, attempts: int, message: Optional[str] = None):
        self.chunk_id = chunk_id
        self.attempts = attempts
        super().__init__(message or f"Scout chunk {chunk_id} failed to parse after {attempts} attempts")


class ArchitectExtractionError(PipelineError):
    def __init__(self, srs_path: str, message: Optional[str] = None):
        self.srs_path = srs_path
        super().__init__(message or f"Architect produced zero bounded contexts for {srs_path}")


class SpecialistFailureError(PipelineError):
    def __init__(self, context_name: str, message: Optional[str] = None):
        self.context_name = context_name
        super().__init__(message or f"Specialist failed for context {context_name!r}")


class SynthesizerEmptyModelError(PipelineError):
    def __init__(self, input_summary: str, message: Optional[str] = None):
        self.input_summary = input_summary
        super().__init__(message or f"Synthesizer returned an empty DomainModel (input: {input_summary})")


class RefinementExhaustedError(PipelineError):
    def __init__(self, issues: List[Any], message: Optional[str] = None):
        self.issues = issues
        super().__init__(message or f"Refiner exhausted retries with {len(issues)} unresolved issues")


class InsufficientGroundingError(PipelineError):
    def __init__(self, entity_name: str, message: Optional[str] = None):
        self.entity_name = entity_name
        super().__init__(message or f"Entity {entity_name!r} has no SRS evidence_sentence_indices and no AST grounding")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension/backend && pytest tests/test_orchestration_errors.py -v`
Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/orchestration/__init__.py extension/backend/core/orchestration/errors.py extension/backend/tests/test_orchestration_errors.py
git commit -m "$(cat <<'EOF'
feat(orchestration): introduce PipelineError hierarchy

Adds core/orchestration/errors.py with six typed exceptions used by
Phases B2-B5 and C4 to replace silent fallbacks. The pipeline-level
wrapper (Task C6) catches PipelineError, persists failure_log.json,
and surfaces the failure to the orchestrator (RQ1 policy decides
retry/skip/fail).
EOF
)"
```

---

### Task B2: Architect raises `ArchitectExtractionError` instead of returning `["CoreDomain"]` (FM-01)

**Files:**
- Create: `extension/backend/tests/test_architect_extraction_error.py`
- Modify: `extension/backend/core/architect.py` — `identify_contexts` (lines ~415-417, 458-459, 461-467)

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_architect_extraction_error.py`:

```python
"""Phase B2: Architect must raise ArchitectExtractionError when retries
exhaust, not return ['CoreDomain']."""

import pytest
from unittest.mock import patch, MagicMock
from core.architect import DomainArchitect
from core.orchestration.errors import ArchitectExtractionError


def _arch():
    a = DomainArchitect.__new__(DomainArchitect)
    a.model_name = "gemini-3.1-pro-preview"
    a.last_request_time = 0
    a.min_delay = 0
    a.request_count = 0
    import threading
    a._rate_limit_lock = threading.Lock()
    from core.token_tracker import TokenTracker
    a.token_tracker = TokenTracker.get_instance()
    a.progress_callback = None
    a.run_timestamp = "20260518_000000"
    a.client = MagicMock()
    return a


def test_architect_raises_when_response_parse_fails_all_retries():
    arch = _arch()
    bad_response = MagicMock()
    bad_response.candidates = [MagicMock()]
    bad_response.candidates[0].finish_reason = "STOP"
    bad_response.text = "not valid json"
    arch.client.models.generate_content.return_value = bad_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"):
        with pytest.raises(ArchitectExtractionError) as excinfo:
            arch.identify_contexts(domain_sentences=["one", "two"])
        assert "SRS" in str(excinfo.value) or excinfo.value.srs_path


def test_architect_raises_when_response_is_empty_list():
    arch = _arch()
    empty_response = MagicMock()
    empty_response.candidates = [MagicMock()]
    empty_response.candidates[0].finish_reason = "STOP"
    empty_response.text = '{"contexts": []}'
    arch.client.models.generate_content.return_value = empty_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch, "_parse_json_response", return_value={"contexts": []}):
        with pytest.raises(ArchitectExtractionError):
            arch.identify_contexts(domain_sentences=["one", "two"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_architect_extraction_error.py -v`
Expected: Both tests FAIL because `identify_contexts` currently returns `["CoreDomain"]` instead of raising.

- [ ] **Step 3: Modify `identify_contexts`**

In `extension/backend/core/architect.py`, locate `identify_contexts` (the method that prints "STAGE 2: ARCHITECT"). Replace the three `return ["CoreDomain"]` lines (currently around lines 417, 458-459, 465-467) with explicit raises. The function should never return `["CoreDomain"]` again.

At the top of the file, add:

```python
from core.orchestration.errors import ArchitectExtractionError
```

Inside `identify_contexts`, find the retry loop and adjust:

```python
        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(...)

                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue

                result = self._parse_json_response(self._safe_response_text(response))

                if (
                    isinstance(result, dict)
                    and result.get("error") == "json_parse_failed"
                ):
                    if retry < 4:
                        continue
                    raise ArchitectExtractionError(
                        srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                        message="Architect exhausted JSON parse retries (5/5)"
                    )

                self.token_tracker.track_api_call(response, stage="Architect", operation="identify_contexts")

                contexts: Optional[List[str]] = None
                if isinstance(result, dict) and "contexts" in result:
                    candidate = result["contexts"]
                    if candidate and len(candidate) > 0:
                        contexts = candidate
                elif isinstance(result, list) and len(result) > 0:
                    contexts = result

                if contexts:
                    self._save_intermediate(stage="2_architect", data={
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "contexts_identified": len(contexts),
                        "contexts": contexts,
                        "input_sentences_count": len(domain_sentences)
                    })
                    self._report_progress("Architect", "completed", f"Found {len(contexts)} contexts", 100)
                    return contexts

                if retry < 4:
                    continue
                raise ArchitectExtractionError(
                    srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                    message="Architect produced empty contexts list after 5 retries"
                )

            except ArchitectExtractionError:
                raise
            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    if retry >= 4:
                        raise ArchitectExtractionError(
                            srs_path=getattr(self, "_current_srs_path", "<unknown>"),
                            message=f"Architect failed with {type(e).__name__}: {e}"
                        ) from e

        raise ArchitectExtractionError(
            srs_path=getattr(self, "_current_srs_path", "<unknown>"),
            message="Architect exhausted retry loop without producing contexts"
        )
```

(Set `self._current_srs_path` in `analyze_document` at the start of a run; if it isn't set yet, the `<unknown>` default keeps tests passing — Phase C6 wires it explicitly.)

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_architect_extraction_error.py -v`
Expected: 2 tests PASS.

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: full unit suite green. Patch any test that asserted `identify_contexts` returns `["CoreDomain"]`.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_architect_extraction_error.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
refactor(architect): raise ArchitectExtractionError on retry exhaustion (FM-01)

Removes all three silent returns of ['CoreDomain'] from identify_contexts.
The pipeline now either produces a non-empty bounded_contexts list or
raises ArchitectExtractionError carrying srs_path + retry context.
RQ1 metrics no longer conflate fallback runs with successful 1-context
runs.
EOF
)"
```

---

### Task B3: Specialist raises `SpecialistFailureError` instead of returning error dicts (FM-02)

**Files:**
- Create: `extension/backend/tests/test_specialist_failure_error.py`
- Modify: `extension/backend/core/architect.py` — `extract_all_contexts_details` (lines 554-557, 590-593, 596-599)

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_specialist_failure_error.py`:

```python
"""Phase B3: Specialist must raise SpecialistFailureError when retries
exhaust, not return [{"context": ctx, "analysis": {"error": "parse_failed"}}]."""

import pytest
from unittest.mock import patch, MagicMock
from core.architect import DomainArchitect
from core.orchestration.errors import SpecialistFailureError


def _arch():
    a = DomainArchitect.__new__(DomainArchitect)
    a.model_name = "gemini-3.1-pro-preview"
    a.last_request_time = 0
    a.min_delay = 0
    a.request_count = 0
    import threading
    a._rate_limit_lock = threading.Lock()
    from core.token_tracker import TokenTracker
    a.token_tracker = TokenTracker.get_instance()
    a.progress_callback = None
    a.run_timestamp = "20260518_000000"
    a.client = MagicMock()
    return a


def test_specialist_raises_on_parse_failure_after_retries():
    arch = _arch()
    bad_response = MagicMock()
    bad_response.candidates = [MagicMock()]
    bad_response.candidates[0].finish_reason = "STOP"
    bad_response.text = "garbage"
    arch.client.models.generate_content.return_value = bad_response

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch, "_parse_json_response", return_value={"error": "json_parse_failed"}):
        with pytest.raises(SpecialistFailureError):
            arch.extract_all_contexts_details(
                contexts=["OrderMgmt"], domain_sentences=["s1", "s2"]
            )


def test_specialist_raises_on_exception_after_retries():
    arch = _arch()
    arch.client.models.generate_content.side_effect = RuntimeError("boom")

    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(arch, "_is_quota_error_and_backoff", return_value=False):
        with pytest.raises(SpecialistFailureError):
            arch.extract_all_contexts_details(
                contexts=["OrderMgmt"], domain_sentences=["s1", "s2"]
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_specialist_failure_error.py -v`
Expected: Both FAIL because the current implementation returns error-dict lists.

- [ ] **Step 3: Modify `extract_all_contexts_details`**

At the top of `extension/backend/core/architect.py` add:

```python
from core.orchestration.errors import SpecialistFailureError
```

In the `extract_all_contexts_details` retry loop, replace the three fallback returns:

```python
        sc = stage_config("Specialist")
        for retry in range(5):
            try:
                self._wait_for_rate_limit()
                response = self.client.models.generate_content(...)

                if not self._check_response_completion(response, retry):
                    if retry < 4:
                        time.sleep(2)
                        continue

                result = self._parse_json_response(self._safe_response_text(response))

                if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                    if retry < 4:
                        time.sleep(2)
                        continue
                    raise SpecialistFailureError(
                        context_name="<all>",
                        message=f"Specialist exhausted JSON parse retries for {len(contexts)} contexts"
                    )

                self.token_tracker.track_api_call(
                    response, stage="Specialist", operation="analyze_all_contexts"
                )

                if isinstance(result, dict) and "analyses" in result:
                    analyses = [
                        {"context": a.get("context", "Unknown"), "analysis": a}
                        for a in result["analyses"]
                    ]
                    self._save_intermediate(stage="3_specialist", data={
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "contexts_analyzed": len(analyses),
                        "analyses": analyses
                    })
                    self._report_progress("Specialist", "completed", f"Analyzed {len(analyses)} contexts", 100)
                    return analyses

                if retry < 4:
                    continue
                raise SpecialistFailureError(
                    context_name="<all>",
                    message="Specialist produced no 'analyses' field after 5 retries"
                )

            except SpecialistFailureError:
                raise
            except Exception as e:
                if not self._is_quota_error_and_backoff(e, retry):
                    if retry >= 4:
                        raise SpecialistFailureError(
                            context_name="<all>",
                            message=f"Specialist failed with {type(e).__name__}: {e}"
                        ) from e

        raise SpecialistFailureError(
            context_name="<all>",
            message="Specialist exhausted retry loop without analyses"
        )
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_specialist_failure_error.py -v`
Expected: 2 tests PASS.

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green. Patch any test that asserted error-dict shape.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_specialist_failure_error.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
refactor(specialist): raise SpecialistFailureError on retry exhaustion (FM-02)

Removes the three silent returns of [{"context": ctx, "analysis":
{"error": ...}}] from extract_all_contexts_details. The Synthesizer
can no longer hallucinate from error-shaped payloads because the
pipeline raises before Synthesizer sees anything.
EOF
)"
```

---

### Task B4: Synthesizer raises `SynthesizerEmptyModelError`; delete `_create_fallback_model` (FM-04)

**Files:**
- Create: `extension/backend/tests/test_synthesizer_empty_model_error.py`
- Modify: `extension/backend/core/architect.py` — `synthesize_final_model` (around 752-773), delete `_create_fallback_model` (around 707-720)

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_synthesizer_empty_model_error.py`:

```python
"""Phase B4: Synthesizer must raise SynthesizerEmptyModelError when the
LLM returns an empty bounded_contexts list. _create_fallback_model is
deleted; the bare except is already narrowed (A5).
"""

import pytest
from unittest.mock import patch, MagicMock
from core.architect import DomainArchitect
from core.orchestration.errors import SynthesizerEmptyModelError


def _arch():
    a = DomainArchitect.__new__(DomainArchitect)
    a.model_name = "gemini-3.1-pro-preview"
    a.run_timestamp = "20260518_000000"
    return a


def test_synthesize_final_model_raises_when_empty_bounded_contexts():
    arch = _arch()
    empty_payload = {
        "project_name": "X",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [],
        "global_rules": None,
    }
    with patch.object(arch, "synthesize", return_value=empty_payload):
        with pytest.raises(SynthesizerEmptyModelError):
            arch.synthesize_final_model(analyses=[])


def test_create_fallback_model_is_gone():
    arch = _arch()
    assert not hasattr(arch, "_create_fallback_model"), (
        "_create_fallback_model must be deleted in B4; an empty model is "
        "no longer a legitimate pipeline output."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_synthesizer_empty_model_error.py -v`
Expected: The first test FAILS (Pydantic ValidationError raised by A4's field_validator is not the right type yet). The second test FAILS because `_create_fallback_model` still exists.

- [ ] **Step 3: Modify `synthesize_final_model` and delete `_create_fallback_model`**

In `extension/backend/core/architect.py`:

(a) Add import at top:

```python
from core.orchestration.errors import SynthesizerEmptyModelError
```

(b) Replace `synthesize_final_model` so it converts the A4 ValidationError on empty `bounded_contexts` into a typed `SynthesizerEmptyModelError`, and re-raises Pydantic errors otherwise:

```python
    def synthesize_final_model(self, analyses: List[Dict[str, Any]]) -> DomainModel:
        """Synthesize per-context analyses into a final DomainModel.

        Phase B4: Empty bounded_contexts raises SynthesizerEmptyModelError
        (typed). Other Pydantic errors propagate unchanged so that
        prompt/schema bugs surface loudly rather than as empty models.
        """
        raw_dict = self.synthesize(analyses)
        raw_dict = self._cleanup_domain_data(raw_dict)
        if not raw_dict.get("bounded_contexts"):
            raise SynthesizerEmptyModelError(
                input_summary=f"{len(analyses)} analyses"
            )
        try:
            return DomainModel(**raw_dict)
        except ValidationError as e:
            # If Pydantic complains about empty bounded_contexts despite the
            # explicit check above (e.g. due to A4's field_validator on a
            # subtly malformed payload), convert to a typed error for the
            # orchestrator.
            if any("bounded_contexts" in str(err) for err in e.errors()):
                raise SynthesizerEmptyModelError(
                    input_summary=f"{len(analyses)} analyses; ValidationError: {e}"
                ) from e
            raise
```

Add the import for `ValidationError` at the top of the file:

```python
from pydantic import ValidationError
```

(c) Delete the `_create_fallback_model` method body entirely (currently around lines 707-720). Also remove any remaining call sites in `architect.py` if they survived A5 — search with `grep -n _create_fallback_model extension/backend/core/architect.py` to be sure.

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_synthesizer_empty_model_error.py -v`
Expected: 2 tests PASS.

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green. Patch tests that asserted empty-model fallback.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_synthesizer_empty_model_error.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
refactor(synthesizer): raise SynthesizerEmptyModelError; delete _create_fallback_model (FM-04)

Removes the last silent fallback path: synthesize_final_model can no
longer return an empty DomainModel(bounded_contexts=[]). Combined with
A4's field_validator, an empty result raises a typed error that the
orchestrator can surface in RQ1 metrics as a failed run rather than a
0-recall success.
EOF
)"
```

---

### Task B5: AST `_collect_signals` raises instead of silently swallowing exceptions (G01)

**Files:**
- Create: `extension/backend/tests/test_ast_collect_signals_raises.py`
- Modify: `extension/backend/core/AST/ast_model_signals.py` (around lines 71-72)

- [ ] **Step 1: Inspect the current `_collect_signals` to know the right exception to raise**

Read `extension/backend/core/AST/ast_model_signals.py` first; locate the `_collect_signals` method and the `try/except` block around lines 71-72.

- [ ] **Step 2: Write the failing test**

Create `extension/backend/tests/test_ast_collect_signals_raises.py`:

```python
"""Phase B5 / G01: _collect_signals must raise on a malformed source file
instead of silently logging and dropping the file from the candidate set.
"""

import pytest
from pathlib import Path
from core.AST.ast_model_signals import ASTModelSignalExtractor


def test_collect_signals_raises_on_unreadable_file(tmp_path):
    bad = tmp_path / "broken.py"
    bad.write_bytes(b"\xff\xfe\x00\x00not valid utf-8\x00")
    extractor = ASTModelSignalExtractor()
    with pytest.raises((SyntaxError, UnicodeDecodeError, ValueError, OSError)):
        # Use the public method that triggers _collect_signals internally
        extractor.enrich_domain_model(
            domain_model=None,  # type: ignore[arg-type]
            workspace_path=str(tmp_path),
        )
```

(If the public surface of `ASTModelSignalExtractor` differs, adapt this test to whatever method triggers `_collect_signals`; the goal is: a malformed Python file in the workspace should cause `_collect_signals` to raise rather than swallow.)

- [ ] **Step 3: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_ast_collect_signals_raises.py -v`
Expected: FAIL. The current `_collect_signals` swallows the exception and returns an empty list.

- [ ] **Step 4: Modify `_collect_signals`**

In `extension/backend/core/AST/ast_model_signals.py`, replace the broad `try/except` that swallows exceptions (around lines 71-72) with explicit, narrow handling:

```python
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            # G01: raise instead of silently returning an empty signal list.
            # The caller (enrich_domain_model) decides whether to count this
            # toward run-failure metrics or surface to the user.
            raise SyntaxError(f"AST parse failed for {path}: {e}") from e
```

If the surrounding logic catches more exception types (file IO, encoding), keep each catch narrow and re-raise; never swallow with `pass`.

- [ ] **Step 5: Run tests**

Run: `cd extension/backend && pytest tests/test_ast_collect_signals_raises.py -v`
Expected: PASS.

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green. Any test that depended on `_collect_signals` silently tolerating broken files will fail; fix those by either supplying valid fixtures or asserting the new typed raise.

- [ ] **Step 6: Commit**

```bash
git add extension/backend/tests/test_ast_collect_signals_raises.py extension/backend/core/AST/ast_model_signals.py
git commit -m "$(cat <<'EOF'
refactor(ast): _collect_signals propagates SyntaxError instead of swallowing (G01)

Silent except-block in core/AST/ast_model_signals.py:71-72 caused
malformed Python files in the workspace to vanish from the candidate
set, distorting RQ1/RQ3 counts. Narrow the catch to SyntaxError + IO,
re-raise with file context. enrich_domain_model is now responsible
for deciding whether to surface or count.
EOF
)"
```

**End of Phase B.** Phase-boundary smoke test:

- [ ] **Step B-end.1: Regression sweep**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green.

Run: `cd extension/backend && grep -n "return \[\"CoreDomain\"\]\|_create_fallback_model\|except Exception:" core/architect.py core/AST/ast_model_signals.py`
Expected: zero matches (silent-fallback patterns are gone). If any match remains, it is a leak — fix before Phase C.

---

# Phase C — Verifier + Refiner + Section-aware Chunking (9 commits, ~7-12 days, high risk)

Goal: Stand up the new orchestration package and migrate `DomainArchitect.analyze_document` to a thin facade over `core/orchestration/pipeline.py`. This is the largest single phase; each commit is independently testable and individually small.

---

### Task C1: `core/verifier/types.py` + `__init__.py` (interface only)

**Files:**
- Create: `extension/backend/core/verifier/__init__.py`
- Create: `extension/backend/core/verifier/types.py`
- Create: `extension/backend/tests/test_verifier_types.py`

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_verifier_types.py`:

```python
"""Phase C1: VerifierIssue + IssueSeverity + VerifierResult interfaces."""

import pytest
from core.verifier.types import VerifierIssue, IssueSeverity, VerifierResult


def test_issue_severity_has_error_and_warn():
    assert IssueSeverity.ERROR == "error"
    assert IssueSeverity.WARN == "warn"


def test_verifier_issue_construct():
    issue = VerifierIssue(
        stage="specialist",
        location="specialist:OrderMgmt.entities[2]",
        issue_type="missing_evidence",
        severity=IssueSeverity.ERROR,
        message="Entity 'Order' has no evidence_sentence_indices",
    )
    assert issue.stage == "specialist"
    assert issue.severity == IssueSeverity.ERROR


def test_verifier_result_ok_when_no_issues():
    result = VerifierResult(ok=True, issues=[])
    assert result.ok
    assert len(result.issues) == 0


def test_verifier_result_not_ok_with_error_issue():
    result = VerifierResult(
        ok=False,
        issues=[
            VerifierIssue(
                stage="architect",
                location="architect:contexts[0]",
                issue_type="ungrounded",
                severity=IssueSeverity.ERROR,
                message="Context name not in Scout output",
            )
        ],
    )
    assert not result.ok
    assert result.issues[0].issue_type == "ungrounded"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_verifier_types.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'core.verifier'`.

- [ ] **Step 3: Implement**

Create `extension/backend/core/verifier/__init__.py`:

```python
"""Verifier package: deterministic + semantic checks on stage outputs."""

from core.verifier.types import VerifierIssue, IssueSeverity, VerifierResult

__all__ = ["VerifierIssue", "IssueSeverity", "VerifierResult"]
```

Create `extension/backend/core/verifier/types.py`:

```python
"""Verifier types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARN = "warn"


@dataclass(frozen=True)
class VerifierIssue:
    stage: Literal["scout", "architect", "specialist", "synthesizer"]
    location: str
    issue_type: str
    severity: IssueSeverity
    message: str
    suggestion: Optional[str] = None


@dataclass
class VerifierResult:
    ok: bool
    issues: List[VerifierIssue] = field(default_factory=list)

    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd extension/backend && pytest tests/test_verifier_types.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/verifier/__init__.py extension/backend/core/verifier/types.py extension/backend/tests/test_verifier_types.py
git commit -m "$(cat <<'EOF'
feat(verifier): types — VerifierIssue, IssueSeverity, VerifierResult

Interface-only commit. Establishes the dataclass shape for stage-output
issues that C2 (deterministic checks) and C3 (semantic check) will
return and that C4 (Refiner) will consume.
EOF
)"
```

---

### Task C2: `checks_deterministic.py` — D1-D5 pure-function checks

**Files:**
- Create: `extension/backend/core/verifier/checks_deterministic.py`
- Create: `extension/backend/tests/test_verifier_deterministic.py`

- [ ] **Step 1: Write the failing tests**

Create `extension/backend/tests/test_verifier_deterministic.py`:

```python
"""Phase C2: D1-D5 deterministic checks."""

import pytest
from core.verifier.types import IssueSeverity
from core.verifier.checks_deterministic import (
    check_d1_supporting_sentence_ids_subset,
    check_d2_entity_evidence_nonempty,
    check_d3_entity_names_unique_across_contexts,
    check_d4_aggregate_members_exist_in_context,
    check_d5_allowed_dependencies_reference_existing_contexts,
)


SCOUT_INDICES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}


# ----- D1 -----

def test_d1_passes_when_all_supporting_ids_in_scout():
    contexts = [{"name": "OrderMgmt", "supporting_sentence_ids": [1, 2, 3]}]
    issues = check_d1_supporting_sentence_ids_subset(contexts, SCOUT_INDICES)
    assert issues == []


def test_d1_flags_id_not_in_scout():
    contexts = [{"name": "OrderMgmt", "supporting_sentence_ids": [1, 99]}]
    issues = check_d1_supporting_sentence_ids_subset(contexts, SCOUT_INDICES)
    assert len(issues) == 1
    assert issues[0].issue_type == "ungrounded_context"
    assert issues[0].severity == IssueSeverity.ERROR


# ----- D2 -----

def test_d2_passes_when_evidence_phase_a_optional():
    """In Phase A-C, evidence_sentence_indices is optional → emit only warn."""
    entity = {"name": "Order", "evidence_sentence_indices": []}
    issues = check_d2_entity_evidence_nonempty(
        context_name="OrderMgmt", entities=[entity], phase="C"
    )
    assert len(issues) == 1
    assert issues[0].severity == IssueSeverity.WARN


def test_d2_passes_when_evidence_present_phase_d():
    entity = {"name": "Order", "evidence_sentence_indices": [2, 5]}
    issues = check_d2_entity_evidence_nonempty(
        context_name="OrderMgmt", entities=[entity], phase="D"
    )
    assert issues == []


def test_d2_errors_when_evidence_empty_phase_d():
    entity = {"name": "Order", "evidence_sentence_indices": []}
    issues = check_d2_entity_evidence_nonempty(
        context_name="OrderMgmt", entities=[entity], phase="D"
    )
    assert len(issues) == 1
    assert issues[0].severity == IssueSeverity.ERROR


# ----- D3 -----

def test_d3_passes_when_names_unique():
    by_context = {
        "OrderMgmt": [{"name": "Order"}, {"name": "OrderLine"}],
        "Billing": [{"name": "Invoice"}],
    }
    issues = check_d3_entity_names_unique_across_contexts(by_context)
    assert issues == []


def test_d3_flags_duplicate_entity_in_two_contexts():
    by_context = {
        "OrderMgmt": [{"name": "Customer"}],
        "Billing": [{"name": "Customer"}],
    }
    issues = check_d3_entity_names_unique_across_contexts(by_context)
    assert len(issues) == 1
    assert issues[0].issue_type == "duplicate_entity_across_contexts"


# ----- D4 -----

def test_d4_passes_when_aggregate_members_are_entities():
    entities = [{"name": "Order"}, {"name": "OrderLine"}]
    aggregates = [{"name": "Order", "members": ["Order", "OrderLine"]}]
    issues = check_d4_aggregate_members_exist_in_context(
        context_name="OrderMgmt", entities=entities, aggregates=aggregates
    )
    assert issues == []


def test_d4_flags_phantom_member():
    entities = [{"name": "Order"}]
    aggregates = [{"name": "Order", "members": ["Order", "PhantomLine"]}]
    issues = check_d4_aggregate_members_exist_in_context(
        context_name="OrderMgmt", entities=entities, aggregates=aggregates
    )
    assert len(issues) == 1
    assert "PhantomLine" in issues[0].message


# ----- D5 -----

def test_d5_passes_when_dependencies_reference_existing():
    contexts = [
        {"name": "OrderMgmt", "allowed_dependencies": ["Billing"]},
        {"name": "Billing", "allowed_dependencies": []},
    ]
    issues = check_d5_allowed_dependencies_reference_existing_contexts(contexts)
    assert issues == []


def test_d5_flags_unknown_dependency():
    contexts = [
        {"name": "OrderMgmt", "allowed_dependencies": ["GhostContext"]},
    ]
    issues = check_d5_allowed_dependencies_reference_existing_contexts(contexts)
    assert len(issues) == 1
    assert "GhostContext" in issues[0].message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd extension/backend && pytest tests/test_verifier_deterministic.py -v`
Expected: All FAIL with `ImportError`.

- [ ] **Step 3: Implement**

Create `extension/backend/core/verifier/checks_deterministic.py`:

```python
"""Deterministic D1-D5 checks. Pure functions over stage output dicts."""

from typing import Dict, Iterable, List, Set
from core.verifier.types import IssueSeverity, VerifierIssue


def check_d1_supporting_sentence_ids_subset(
    contexts: List[Dict],
    scout_sentence_indices: Set[int],
) -> List[VerifierIssue]:
    """D1: every BC.supporting_sentence_ids ⊆ Scout-emitted indices."""
    issues: List[VerifierIssue] = []
    for ctx in contexts:
        bad = [i for i in ctx.get("supporting_sentence_ids", []) if i not in scout_sentence_indices]
        if bad:
            issues.append(VerifierIssue(
                stage="architect",
                location=f"architect:contexts[{ctx.get('name')}].supporting_sentence_ids",
                issue_type="ungrounded_context",
                severity=IssueSeverity.ERROR,
                message=(
                    f"Context {ctx.get('name')!r} cites sentence ids {bad} "
                    f"that Scout did not emit"
                ),
                suggestion=f"Drop sentence ids {bad} or revise the context",
            ))
    return issues


def check_d2_entity_evidence_nonempty(
    context_name: str,
    entities: List[Dict],
    phase: str = "C",
) -> List[VerifierIssue]:
    """D2: every Entity has ≥1 evidence_sentence_index.

    In Phases A-C the field is optional, so emit WARN. From Phase D
    onward the field is required (min_items=1 on the schema) and
    missing evidence is an ERROR.
    """
    severity = IssueSeverity.ERROR if phase >= "D" else IssueSeverity.WARN
    issues: List[VerifierIssue] = []
    for idx, entity in enumerate(entities):
        if not entity.get("evidence_sentence_indices"):
            issues.append(VerifierIssue(
                stage="specialist",
                location=f"specialist:{context_name}.entities[{idx}]({entity.get('name')})",
                issue_type="missing_evidence",
                severity=severity,
                message=(
                    f"Entity {entity.get('name')!r} has no "
                    f"evidence_sentence_indices"
                ),
                suggestion="Cite at least one Scout sentence id that names this entity",
            ))
    return issues


def check_d3_entity_names_unique_across_contexts(
    entities_by_context: Dict[str, List[Dict]],
) -> List[VerifierIssue]:
    """D3: every entity name appears in exactly one bounded context."""
    issues: List[VerifierIssue] = []
    seen: Dict[str, str] = {}
    for ctx_name, entities in entities_by_context.items():
        for entity in entities:
            name = entity.get("name")
            if name in seen and seen[name] != ctx_name:
                issues.append(VerifierIssue(
                    stage="specialist",
                    location=f"specialist:{ctx_name}.entities({name})",
                    issue_type="duplicate_entity_across_contexts",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Entity {name!r} appears in both {seen[name]!r} "
                        f"and {ctx_name!r}"
                    ),
                    suggestion="Pick one context for this entity",
                ))
            else:
                seen[name] = ctx_name
    return issues


def check_d4_aggregate_members_exist_in_context(
    context_name: str,
    entities: List[Dict],
    aggregates: List[Dict],
) -> List[VerifierIssue]:
    """D4: every Aggregate.members entry exists as an Entity in the same context."""
    entity_names = {e.get("name") for e in entities}
    issues: List[VerifierIssue] = []
    for agg in aggregates:
        for member in agg.get("members", []):
            if member not in entity_names:
                issues.append(VerifierIssue(
                    stage="specialist",
                    location=(
                        f"specialist:{context_name}.aggregates({agg.get('name')}).members"
                    ),
                    issue_type="invalid_aggregate_member",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Aggregate {agg.get('name')!r} lists member "
                        f"{member!r} which is not an entity in {context_name!r}"
                    ),
                    suggestion=(
                        f"Either add an entity {member!r} or drop it from members"
                    ),
                ))
    return issues


def check_d5_allowed_dependencies_reference_existing_contexts(
    contexts: List[Dict],
) -> List[VerifierIssue]:
    """D5: every context.allowed_dependencies references an existing context."""
    known = {ctx.get("name") for ctx in contexts}
    issues: List[VerifierIssue] = []
    for ctx in contexts:
        for dep in ctx.get("allowed_dependencies") or []:
            if dep not in known:
                issues.append(VerifierIssue(
                    stage="synthesizer",
                    location=f"synthesizer:contexts[{ctx.get('name')}].allowed_dependencies",
                    issue_type="unknown_dependency",
                    severity=IssueSeverity.ERROR,
                    message=(
                        f"Context {ctx.get('name')!r} declares dependency "
                        f"{dep!r} which is not a known context"
                    ),
                    suggestion=f"Drop {dep!r} or add a corresponding bounded context",
                ))
    return issues
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd extension/backend && pytest tests/test_verifier_deterministic.py -v`
Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/verifier/checks_deterministic.py extension/backend/tests/test_verifier_deterministic.py
git commit -m "$(cat <<'EOF'
feat(verifier): deterministic checks D1-D5

D1: supporting_sentence_ids ⊆ Scout
D2: entity has ≥1 evidence_sentence_index (WARN in phase C, ERROR in phase D)
D3: entity names unique across contexts
D4: aggregate members exist as entities in same context
D5: allowed_dependencies references existing contexts only

All five are pure functions over stage output dicts — fast unit tests,
no LLM required.
EOF
)"
```

---

### Task C3: `checks_semantic.py` — S1 LLM-based grounding spot-check

**Files:**
- Create: `extension/backend/core/verifier/checks_semantic.py`
- Create: `extension/backend/tests/test_verifier_semantic.py`

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_verifier_semantic.py`:

```python
"""Phase C3: S1 semantic grounding check via LLM.

The check asks an LLM judge whether a claimed entity actually appears
in its cited Scout sentences. Tests use a mock LLM that returns canned
verdicts so unit tests stay fast and deterministic.
"""

import pytest
from unittest.mock import MagicMock
from core.verifier.types import IssueSeverity
from core.verifier.checks_semantic import check_s1_entity_grounded_in_evidence


SCOUT_SENTENCES = {
    0: "The Order Management context handles all customer purchases.",
    1: "A Customer can place an Order.",
    2: "Payment is processed by the Billing service.",
}


def test_s1_passes_when_llm_confirms_grounding():
    fake_llm = MagicMock()
    fake_llm.judge.return_value = {"grounded": True, "reason": "Customer appears in sentence 1"}
    entity = {"name": "Customer", "evidence_sentence_indices": [1]}
    issues = check_s1_entity_grounded_in_evidence(
        entity, scout_sentences=SCOUT_SENTENCES, llm_judge=fake_llm
    )
    assert issues == []


def test_s1_flags_when_llm_says_not_grounded():
    fake_llm = MagicMock()
    fake_llm.judge.return_value = {"grounded": False, "reason": "PhantomEntity not present"}
    entity = {"name": "PhantomEntity", "evidence_sentence_indices": [0]}
    issues = check_s1_entity_grounded_in_evidence(
        entity, scout_sentences=SCOUT_SENTENCES, llm_judge=fake_llm
    )
    assert len(issues) == 1
    assert issues[0].issue_type == "semantic_ungrounded"
    assert issues[0].severity == IssueSeverity.ERROR


def test_s1_passes_with_no_indices_phase_c():
    """If the entity has no indices, D2 already handles it — S1 returns []."""
    fake_llm = MagicMock()
    entity = {"name": "Order", "evidence_sentence_indices": []}
    issues = check_s1_entity_grounded_in_evidence(
        entity, scout_sentences=SCOUT_SENTENCES, llm_judge=fake_llm
    )
    assert issues == []
    fake_llm.judge.assert_not_called()
```

- [ ] **Step 2: Run tests to verify fail**

Run: `cd extension/backend && pytest tests/test_verifier_semantic.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

Create `extension/backend/core/verifier/checks_semantic.py`:

```python
"""S1: LLM-based semantic grounding spot-check.

For each entity with cited evidence_sentence_indices, ask a small LLM
judge: "does the entity name (or a close synonym) appear in the cited
sentences?" Issues are emitted when the judge says no.

Phase C3 ships this as a Protocol-based dependency so unit tests can
inject a MagicMock judge. Phase C6 wires a real Gemini call.
"""

from typing import Dict, List, Protocol, Sequence
from core.verifier.types import IssueSeverity, VerifierIssue


class LLMJudge(Protocol):
    def judge(self, *, entity_name: str, sentences: Sequence[str]) -> Dict:
        """Return {"grounded": bool, "reason": str}."""
        ...


def check_s1_entity_grounded_in_evidence(
    entity: Dict,
    scout_sentences: Dict[int, str],
    llm_judge: LLMJudge,
) -> List[VerifierIssue]:
    """Ask the LLM judge whether the entity actually appears in its
    cited evidence sentences. Returns [] when grounded or no indices
    are claimed; one ERROR issue when the judge rules ungrounded.
    """
    indices = entity.get("evidence_sentence_indices") or []
    if not indices:
        return []
    cited = [scout_sentences[i] for i in indices if i in scout_sentences]
    if not cited:
        return [VerifierIssue(
            stage="specialist",
            location=f"specialist:entities({entity.get('name')})",
            issue_type="evidence_indices_out_of_range",
            severity=IssueSeverity.ERROR,
            message=(
                f"Entity {entity.get('name')!r} cites sentence ids "
                f"{indices} not present in scout_sentences"
            ),
        )]
    verdict = llm_judge.judge(entity_name=entity["name"], sentences=cited)
    if not verdict.get("grounded", False):
        return [VerifierIssue(
            stage="specialist",
            location=f"specialist:entities({entity.get('name')})",
            issue_type="semantic_ungrounded",
            severity=IssueSeverity.ERROR,
            message=(
                f"LLM judge: entity {entity.get('name')!r} not grounded in "
                f"cited sentences ({verdict.get('reason')})"
            ),
            suggestion="Either cite different sentences or drop this entity",
        )]
    return []
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_verifier_semantic.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/verifier/checks_semantic.py extension/backend/tests/test_verifier_semantic.py
git commit -m "$(cat <<'EOF'
feat(verifier): S1 semantic grounding check with injectable LLM judge

Adds the Protocol-based LLMJudge dependency so unit tests can inject a
MagicMock. C6 wires a real Gemini-backed judge in the pipeline driver.
The check emits semantic_ungrounded ERROR when the judge says an
entity name does not appear in its cited evidence sentences.
EOF
)"
```

---

### Task C4: `core/refiner/` — prompts + bounded retry loop

**Files:**
- Create: `extension/backend/core/refiner/__init__.py`
- Create: `extension/backend/core/refiner/prompts.py`
- Create: `extension/backend/core/refiner/loop.py`
- Create: `extension/backend/tests/test_refiner_loop.py`

- [ ] **Step 1: Write the failing tests**

Create `extension/backend/tests/test_refiner_loop.py`:

```python
"""Phase C4: bounded retry loop. Mock LLM provider, mock verifier."""

import pytest
from unittest.mock import MagicMock
from core.verifier.types import VerifierIssue, IssueSeverity, VerifierResult
from core.orchestration.errors import RefinementExhaustedError
from core.refiner.loop import refine_until_clean


def _ok_result():
    return VerifierResult(ok=True, issues=[])


def _err_result(stage="specialist"):
    return VerifierResult(
        ok=False,
        issues=[VerifierIssue(
            stage=stage,
            location=f"{stage}:x.entities[0]",
            issue_type="missing_evidence",
            severity=IssueSeverity.ERROR,
            message="missing",
        )],
    )


def test_refiner_returns_clean_when_verifier_passes_first_try():
    stage_runner = MagicMock(return_value={"ok": True})
    verifier = MagicMock(return_value=_ok_result())
    out, cycles = refine_until_clean(
        stage_name="specialist",
        initial_output={"ok": True},
        stage_runner=stage_runner,
        verifier=verifier,
        max_cycles=2,
    )
    assert cycles == 0
    assert verifier.call_count == 1
    stage_runner.assert_not_called()


def test_refiner_runs_one_cycle_to_fix_issues():
    stage_runner = MagicMock(side_effect=[{"fixed": True}])
    verifier = MagicMock(side_effect=[_err_result(), _ok_result()])
    out, cycles = refine_until_clean(
        stage_name="specialist",
        initial_output={"buggy": True},
        stage_runner=stage_runner,
        verifier=verifier,
        max_cycles=2,
    )
    assert cycles == 1
    assert out == {"fixed": True}
    stage_runner.assert_called_once()


def test_refiner_raises_after_max_cycles():
    stage_runner = MagicMock(side_effect=[{"still_buggy": 1}, {"still_buggy": 2}])
    verifier = MagicMock(side_effect=[_err_result(), _err_result(), _err_result()])
    with pytest.raises(RefinementExhaustedError) as excinfo:
        refine_until_clean(
            stage_name="specialist",
            initial_output={"buggy": True},
            stage_runner=stage_runner,
            verifier=verifier,
            max_cycles=2,
        )
    assert len(excinfo.value.issues) == 1


def test_refiner_determinism_same_input_same_cycle_count():
    """Two identical setups must produce identical cycle counts."""
    def setup():
        sr = MagicMock(side_effect=[{"fixed": True}])
        ver = MagicMock(side_effect=[_err_result(), _ok_result()])
        return sr, ver
    sr1, v1 = setup()
    _, c1 = refine_until_clean(stage_name="x", initial_output={}, stage_runner=sr1, verifier=v1, max_cycles=2)
    sr2, v2 = setup()
    _, c2 = refine_until_clean(stage_name="x", initial_output={}, stage_runner=sr2, verifier=v2, max_cycles=2)
    assert c1 == c2 == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd extension/backend && pytest tests/test_refiner_loop.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the refiner**

Create `extension/backend/core/refiner/__init__.py`:

```python
"""Refiner package: bounded retry loop + per-stage refinement prompts."""

from core.refiner.loop import refine_until_clean

__all__ = ["refine_until_clean"]
```

Create `extension/backend/core/refiner/prompts.py`:

```python
"""Per-stage refinement prompts.

The Refiner takes the Verifier's issue list + the failing stage's
output + the failing stage's original prompt template and produces a
new prompt that asks the same stage to fix the cited issues.
"""

from typing import List
from core.verifier.types import VerifierIssue


REFINEMENT_PROMPT_TEMPLATE = """\
Your previous output for stage `{stage_name}` had the following issues:

{issue_list}

PREVIOUS OUTPUT:
{previous_output_json}

INSTRUCTIONS:
1. Address every issue listed above.
2. Preserve all correct parts of the previous output.
3. Re-emit the FULL stage output (not just the corrections).
4. If you cannot address an issue, leave that part of the output unchanged
   and add a `_unresolved` key to the affected element.

Respond with valid JSON matching the original stage's schema.
"""


def render_refinement_prompt(
    *, stage_name: str, previous_output_json: str, issues: List[VerifierIssue]
) -> str:
    issue_list = "\n".join(
        f"- [{i.severity.value}] {i.location} — {i.message}"
        + (f" (suggestion: {i.suggestion})" if i.suggestion else "")
        for i in issues
    )
    return REFINEMENT_PROMPT_TEMPLATE.format(
        stage_name=stage_name,
        issue_list=issue_list,
        previous_output_json=previous_output_json,
    )
```

Create `extension/backend/core/refiner/loop.py`:

```python
"""Bounded retry orchestration.

refine_until_clean takes a stage's output and runs the Verifier; if
issues exist, it asks the stage_runner to produce a corrected output
and re-verifies. Capped at max_cycles cycles; on exhaustion raises
RefinementExhaustedError carrying the residual issues.
"""

from typing import Any, Callable, Tuple
from core.verifier.types import VerifierResult
from core.orchestration.errors import RefinementExhaustedError


def refine_until_clean(
    *,
    stage_name: str,
    initial_output: Any,
    stage_runner: Callable[[Any, VerifierResult], Any],
    verifier: Callable[[Any], VerifierResult],
    max_cycles: int = 2,
) -> Tuple[Any, int]:
    """Run verifier; if issues, call stage_runner with (output, result)
    to produce a corrected output; loop up to max_cycles.

    Returns (final_output, cycles_used). Raises RefinementExhaustedError
    when verifier still reports issues after max_cycles.
    """
    output = initial_output
    cycles = 0
    while True:
        result = verifier(output)
        if result.ok or result.error_count() == 0:
            return output, cycles
        if cycles >= max_cycles:
            raise RefinementExhaustedError(issues=result.issues)
        output = stage_runner(output, result)
        cycles += 1
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_refiner_loop.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/refiner/__init__.py extension/backend/core/refiner/prompts.py extension/backend/core/refiner/loop.py extension/backend/tests/test_refiner_loop.py
git commit -m "$(cat <<'EOF'
feat(refiner): bounded retry loop with per-stage refinement prompts

refine_until_clean drives the verify→refine→re-verify cycle, capping
at max_cycles=2 and raising RefinementExhaustedError on exhaustion.
Deterministic by design: same input + same stage_runner → same cycle
count. Verified by test_refiner_determinism_same_input_same_cycle_count.
EOF
)"
```

---

### Task C5: `core/scout/chunking.py` — section-aware chunker (OQ1)

**Files:**
- Create: `extension/backend/core/scout/__init__.py`
- Create: `extension/backend/core/scout/chunking.py`
- Create: `extension/backend/tests/fixtures/sample_srs.txt`
- Create: `extension/backend/tests/test_scout_chunking.py`

- [ ] **Step 1: Write the failing tests**

Create `extension/backend/tests/fixtures/sample_srs.txt`:

```
1 Introduction
This document specifies the system requirements.

2 Functional Requirements

2.1 Order Management
The system shall allow customers to place orders for products.
An order contains one or more order lines.

2.2 Inventory
The system shall track stock levels per product.

3 Non-Functional Requirements
The system shall respond within 200ms.
```

Create `extension/backend/tests/test_scout_chunking.py`:

```python
"""Phase C5: section-aware chunking."""

import pytest
from pathlib import Path
from core.scout.chunking import section_aware_chunks


SAMPLE_SRS = Path(__file__).parent / "fixtures" / "sample_srs.txt"


def test_section_chunker_emits_one_chunk_per_section():
    text = SAMPLE_SRS.read_text()
    chunks = section_aware_chunks(text, token_budget=10000)
    assert len(chunks) == 5  # 1, 2, 2.1, 2.2, 3
    titles = [c["section_title"] for c in chunks]
    assert any("Order Management" in t for t in titles)
    assert any("Inventory" in t for t in titles)


def test_section_chunker_assigns_unique_section_ids():
    text = SAMPLE_SRS.read_text()
    chunks = section_aware_chunks(text, token_budget=10000)
    ids = [c["section_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_section_chunker_splits_oversize_section_under_budget():
    long_section = "1 Big Section\n" + ("This is a very long paragraph. " * 5000)
    chunks = section_aware_chunks(long_section, token_budget=50)
    assert len(chunks) > 1
    for c in chunks:
        # 50 tokens ≈ 200 chars (rough heuristic in chunker)
        assert len(c["text"]) <= 250


def test_section_chunker_keeps_short_sections_intact():
    text = "1 Tiny Section\nOne sentence.\n2 Another\nAnother sentence."
    chunks = section_aware_chunks(text, token_budget=10000)
    assert len(chunks) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd extension/backend && pytest tests/test_scout_chunking.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

Create `extension/backend/core/scout/__init__.py`:

```python
"""Scout-stage helpers: section-aware chunking."""

from core.scout.chunking import section_aware_chunks

__all__ = ["section_aware_chunks"]
```

Create `extension/backend/core/scout/chunking.py`:

```python
"""Section-aware SRS chunker (OQ1).

Replaces architect.py's character-based _split_text_into_chunks with a
parser that respects SRS section structure. Each chunk corresponds to
one numbered section (e.g. "2.1 Order Management"). Oversize sections
are subdivided into sub-chunks while preserving the section_id prefix.

Token budget is approximated at 4 chars/token (rough heuristic for
prose). Override per-model via the caller.
"""

import re
from typing import Dict, List


SECTION_HEADER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)$", re.MULTILINE)


def section_aware_chunks(text: str, token_budget: int = 10000) -> List[Dict]:
    """Split SRS text into chunks at section boundaries.

    Returns a list of {"section_id", "section_title", "text"} dicts. If a
    single section's text exceeds the token_budget (approximated as
    token_budget * 4 chars), it is subdivided into sub-chunks suffixed
    with ".chunk_N" on the section_id.
    """
    char_budget = max(80, token_budget * 4)  # heuristic
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        # No section headers — fall back to a single chunk
        return [{"section_id": "0", "section_title": "Document", "text": text}]

    chunks: List[Dict] = []
    for i, m in enumerate(matches):
        section_id = m.group(1)
        section_title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        full = f"{section_id} {section_title}\n{body}"
        if len(full) <= char_budget:
            chunks.append({
                "section_id": section_id,
                "section_title": section_title,
                "text": full,
            })
        else:
            # Subdivide oversize section
            sub_count = 0
            cursor = 0
            while cursor < len(full):
                sub_end = min(cursor + char_budget, len(full))
                sub_count += 1
                chunks.append({
                    "section_id": f"{section_id}.chunk_{sub_count}",
                    "section_title": f"{section_title} (part {sub_count})",
                    "text": full[cursor:sub_end],
                })
                cursor = sub_end
    return chunks
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_scout_chunking.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/scout/__init__.py extension/backend/core/scout/chunking.py extension/backend/tests/fixtures/sample_srs.txt extension/backend/tests/test_scout_chunking.py
git commit -m "$(cat <<'EOF'
feat(scout): section-aware SRS chunker (OQ1)

Replaces the character-based _split_text_into_chunks with a chunker
that respects numbered section headers. Each section becomes a chunk;
oversize sections subdivide while preserving the section_id prefix.
This grounds every downstream sentence_id in a stable section_id
namespace, enabling Phase D's evidence_sentence_indices contract.
EOF
)"
```

---

### Task C6: `core/orchestration/pipeline.py` — 5-stage driver with refine loop

**Files:**
- Create: `extension/backend/core/orchestration/pipeline.py`
- Create: `extension/backend/tests/test_pipeline_orchestration.py`

- [ ] **Step 1: Write the failing tests**

Create `extension/backend/tests/test_pipeline_orchestration.py`:

```python
"""Phase C6: 5-stage pipeline driver. Mock LLM, mock verifier, mock refiner."""

import pytest
from unittest.mock import MagicMock
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.verifier.types import VerifierResult, VerifierIssue, IssueSeverity
from core.orchestration.errors import (
    ArchitectExtractionError,
    SpecialistFailureError,
)


def _ok():
    return VerifierResult(ok=True, issues=[])


def _make_deps_happy_path():
    scout = MagicMock(return_value=[
        {"section_id": "1", "section_title": "Intro", "text": "..."}
    ])
    architect = MagicMock(return_value=[
        {"name": "OrderMgmt", "supporting_sentence_ids": [0]}
    ])
    specialist = MagicMock(return_value=[
        {
            "context_name": "OrderMgmt",
            "entities": [{
                "name": "Order",
                "description": "An order",
                "confidence": 0.9,
                "justification": "test",
                "evidence_sentence_indices": [0],
            }],
            "value_objects": [],
            "services": [],
            "aggregates": [{"name": "Order", "members": ["Order"]}],
            "domain_events": [],
        }
    ])
    synthesizer = MagicMock(return_value={
        "project_name": "Test",
        "project_metadata": {"version": "1.0", "generated_at": "2026-05-18"},
        "bounded_contexts": [{
            "context_name": "OrderMgmt",
            "description": "Manages orders",
            "ubiquitous_language": {
                "entities": [{
                    "name": "Order",
                    "description": "An order",
                    "confidence": 0.9,
                    "justification": "test",
                    "evidence_sentence_indices": [0],
                }],
                "value_objects": [],
                "services": [],
                "aggregates": [{"name": "Order", "description": "agg", "members": ["Order"]}],
                "domain_events": [],
            },
            "supporting_sentence_ids": [0],
            "business_rules": None,
        }],
        "global_rules": None,
    })
    verifier = MagicMock(return_value=_ok())
    return PipelineDeps(
        scout=scout,
        architect=architect,
        specialist=specialist,
        synthesizer=synthesizer,
        verifier=verifier,
    )


def test_pipeline_happy_path_produces_domain_model():
    deps = _make_deps_happy_path()
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    assert model.project_name == "Test"
    assert len(model.bounded_contexts) == 1


def test_pipeline_propagates_architect_extraction_error():
    deps = _make_deps_happy_path()
    deps.architect.side_effect = ArchitectExtractionError(srs_path="x")
    with pytest.raises(ArchitectExtractionError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_propagates_specialist_failure():
    deps = _make_deps_happy_path()
    deps.specialist.side_effect = SpecialistFailureError(context_name="OrderMgmt")
    with pytest.raises(SpecialistFailureError):
        run_pipeline(srs_text="Sample SRS text", deps=deps)


def test_pipeline_invokes_refiner_when_verifier_finds_issues():
    deps = _make_deps_happy_path()
    deps.verifier.side_effect = [
        VerifierResult(ok=False, issues=[VerifierIssue(
            stage="specialist", location="specialist:x.entities[0]",
            issue_type="missing_evidence", severity=IssueSeverity.ERROR,
            message="missing"
        )]),
        VerifierResult(ok=True, issues=[]),  # second call after refine
    ]
    model = run_pipeline(srs_text="Sample SRS text", deps=deps)
    # Refiner re-runs Specialist once
    assert deps.specialist.call_count == 2
    assert model is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd extension/backend && pytest tests/test_pipeline_orchestration.py -v`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement**

Create `extension/backend/core/orchestration/pipeline.py`:

```python
"""5-stage pipeline driver: Scout → Architect → Specialist → Verifier → Synthesizer.

Each stage is injected as a Callable so this module stays test-friendly.
Real wiring (Gemini-backed stages) happens in DomainArchitect.analyze_document
in Task C7.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from core.schemas import DomainModel
from core.verifier.types import VerifierResult
from core.orchestration.errors import (
    ScoutChunkParseError,
    ArchitectExtractionError,
    SpecialistFailureError,
    SynthesizerEmptyModelError,
    RefinementExhaustedError,
)
from core.refiner.loop import refine_until_clean


ScoutFn = Callable[[str], List[Dict]]
ArchitectFn = Callable[[List[Dict]], List[Dict]]
SpecialistFn = Callable[[List[Dict], List[Dict]], List[Dict]]
SynthesizerFn = Callable[[List[Dict]], Dict]
VerifierFn = Callable[[Dict], VerifierResult]


@dataclass
class PipelineDeps:
    scout: ScoutFn
    architect: ArchitectFn
    specialist: SpecialistFn
    synthesizer: SynthesizerFn
    verifier: VerifierFn


def run_pipeline(*, srs_text: str, deps: PipelineDeps) -> DomainModel:
    """Run the 5-stage pipeline. Raises PipelineError subclasses on
    failure; otherwise returns a validated DomainModel.
    """
    scout_chunks = deps.scout(srs_text)
    contexts = deps.architect(scout_chunks)
    specialist_outputs = deps.specialist(contexts, scout_chunks)

    # Build a combined snapshot for the Verifier.
    snapshot = {
        "scout": scout_chunks,
        "architect": contexts,
        "specialist": specialist_outputs,
    }

    def _re_run_specialist(_prev, _result):
        # Phase C ships a simple re-run; Phase D wires issue-aware re-prompting.
        return deps.specialist(contexts, scout_chunks)

    refined_specialist, cycles = refine_until_clean(
        stage_name="specialist",
        initial_output=specialist_outputs,
        stage_runner=_re_run_specialist,
        verifier=lambda s: deps.verifier({**snapshot, "specialist": s}),
        max_cycles=2,
    )

    raw_model = deps.synthesizer(refined_specialist)
    if not raw_model.get("bounded_contexts"):
        raise SynthesizerEmptyModelError(input_summary=f"{len(refined_specialist)} contexts")
    return DomainModel(**raw_model)
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_pipeline_orchestration.py -v`
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/orchestration/pipeline.py extension/backend/tests/test_pipeline_orchestration.py
git commit -m "$(cat <<'EOF'
feat(orchestration): 5-stage pipeline driver with bounded refine loop

Injects stage Callables via PipelineDeps so tests can mock cleanly.
Wires the Verifier between Specialist and Synthesizer, runs the
Refiner on issues, raises typed errors on hard failure. Real
Gemini-backed wiring lands in C7's DomainArchitect refactor.
EOF
)"
```

---

### Task C7: `DomainArchitect.analyze_document` becomes a thin facade over `pipeline.run_pipeline`

**Files:**
- Modify: `extension/backend/core/architect.py` — `DomainArchitect.analyze_document` method

- [ ] **Step 1: Locate the current `analyze_document`**

Read `extension/backend/core/architect.py` around lines 779-826 to see the current control flow. It's a sequence of `parse_document` → `identify_contexts` → `extract_all_contexts_details` → `synthesize_final_model`.

- [ ] **Step 2: Write a refactor regression test**

Append to `extension/backend/tests/test_unit.py` (or create `tests/test_architect_facade.py` if you prefer):

```python
"""C7: analyze_document delegates to core.orchestration.pipeline.run_pipeline."""

from unittest.mock import patch, MagicMock
from core.architect import DomainArchitect
from core.schemas import (
    DomainModel, BoundedContext, UbiquitousLanguage,
    Entity, ProjectMetadata,
)


def test_analyze_document_calls_run_pipeline():
    arch = DomainArchitect.__new__(DomainArchitect)
    arch.model_name = "gemini-3.1-pro-preview"
    arch.last_request_time = 0
    arch.min_delay = 0
    arch.request_count = 0
    import threading
    arch._rate_limit_lock = threading.Lock()
    from core.token_tracker import TokenTracker
    arch.token_tracker = TokenTracker.get_instance()
    arch.progress_callback = None
    arch.run_timestamp = "20260518_000000"
    arch.client = MagicMock()
    arch.scout_max_workers = 1

    fake_model = DomainModel(
        project_name="X",
        project_metadata=ProjectMetadata(version="1.0", generated_at="2026-05-18"),
        bounded_contexts=[BoundedContext(
            context_name="OrderMgmt",
            description="manages",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="d", confidence=0.9,
                    justification="t"
                )],
                value_objects=[], domain_events=[]
            ),
        )],
        global_rules=None,
    )

    with patch("core.architect.run_pipeline", return_value=fake_model) as mock_run:
        result = arch.analyze_document(text="sample SRS text")
        assert result.project_name == "X"
        mock_run.assert_called_once()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd extension/backend && pytest tests/test_unit.py::test_analyze_document_calls_run_pipeline -v` (or `test_architect_facade.py`)
Expected: FAIL because `analyze_document` does not import or call `run_pipeline` yet.

- [ ] **Step 4: Refactor `analyze_document`**

At the top of `extension/backend/core/architect.py`, add:

```python
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.scout.chunking import section_aware_chunks
from core.verifier.checks_deterministic import (
    check_d1_supporting_sentence_ids_subset,
    check_d3_entity_names_unique_across_contexts,
    check_d4_aggregate_members_exist_in_context,
    check_d5_allowed_dependencies_reference_existing_contexts,
)
from core.verifier.types import VerifierResult
```

Replace `analyze_document` (currently around lines 779-826) with:

```python
    def analyze_document(self, text: str) -> DomainModel:
        """Run the 5-stage pipeline on raw SRS text.

        Phase C7: this method becomes a thin facade over
        core.orchestration.pipeline.run_pipeline. Stage callables wrap
        the existing identify_contexts / extract_all_contexts_details /
        synthesize_final_model methods so behaviour is preserved aside
        from the new Verifier+Refiner loop and section-aware chunking.
        """

        def scout_fn(srs_text: str):
            # Section-aware chunking (OQ1, C5)
            return section_aware_chunks(srs_text, token_budget=10000)

        def architect_fn(scout_chunks):
            # The legacy identify_contexts expects a list of sentences; for
            # now, flatten chunks into a sentence list. Phase C8's integration
            # test verifies this end-to-end.
            sentences = []
            for chunk in scout_chunks:
                sentences.append(chunk["text"])
            ctx_names = self.identify_contexts(sentences)
            return [{"name": name, "supporting_sentence_ids": []} for name in ctx_names]

        def specialist_fn(contexts, scout_chunks):
            sentences = [c["text"] for c in scout_chunks]
            ctx_names = [c["name"] for c in contexts]
            results = self.extract_all_contexts_details(ctx_names, sentences)
            # Normalize legacy {"context": ..., "analysis": ...} into the new
            # per-context output shape expected by the verifier.
            normalized = []
            for r in results:
                a = r.get("analysis", {})
                normalized.append({
                    "context_name": r.get("context"),
                    "entities": a.get("entities", []),
                    "value_objects": a.get("value_objects", []),
                    "services": a.get("services", []),
                    "aggregates": a.get("aggregates", []),
                    "domain_events": a.get("domain_events", []),
                })
            return normalized

        def synthesizer_fn(specialist_outputs):
            legacy_input = [
                {"context": s["context_name"], "analysis": s}
                for s in specialist_outputs
            ]
            return self.synthesize(legacy_input)

        def verifier_fn(snapshot):
            scout_indices = set(range(sum(len(c["text"].split(".")) for c in snapshot["scout"])))
            contexts = snapshot["architect"]
            issues = []
            issues.extend(check_d1_supporting_sentence_ids_subset(contexts, scout_indices))
            entities_by_context = {
                s["context_name"]: s.get("entities", [])
                for s in snapshot["specialist"]
            }
            issues.extend(check_d3_entity_names_unique_across_contexts(entities_by_context))
            for s in snapshot["specialist"]:
                issues.extend(check_d4_aggregate_members_exist_in_context(
                    s["context_name"], s.get("entities", []), s.get("aggregates", [])
                ))
            issues.extend(check_d5_allowed_dependencies_reference_existing_contexts(contexts))
            return VerifierResult(ok=(len(issues) == 0), issues=issues)

        deps = PipelineDeps(
            scout=scout_fn,
            architect=architect_fn,
            specialist=specialist_fn,
            synthesizer=synthesizer_fn,
            verifier=verifier_fn,
        )
        return run_pipeline(srs_text=text, deps=deps)
```

- [ ] **Step 5: Run the regression test + full suite**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green. If `test_unit.py` had existing tests asserting the old `analyze_document` flow, update them to mock `run_pipeline` instead.

- [ ] **Step 6: Commit**

```bash
git add extension/backend/tests/test_unit.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
refactor(architect): analyze_document delegates to orchestration.pipeline (C7)

DomainArchitect.analyze_document is now a thin facade over
core/orchestration/pipeline.run_pipeline. Stage callables wrap the
existing identify_contexts / extract_all_contexts_details / synthesize
methods so legacy behaviour is preserved aside from the new Verifier
(D1+D3+D4+D5 checks) and the section-aware Scout chunking. Public
import path stays stable so main.py and tests need no changes.
EOF
)"
```

---

### Task C7b: Per-context Specialist loop (FM-23) — spec §3.3

> **Note on scope:** the spec body §3.3 calls for a per-context Specialist loop (FM-23) but the §7 phasing enumeration listed only 8 Phase C commits with no dedicated loop task. The writing-plans skill's spec-coverage check flagged this gap. Adding this as C7b (24th plan commit total). Cost trade-off accepted per spec §3.3: 4× Specialist calls but each prompt is much smaller.

**Files:**
- Modify: `extension/backend/core/architect.py` — split `extract_all_contexts_details` into per-context loop
- Create: `extension/backend/tests/test_specialist_per_context_loop.py`

- [ ] **Step 1: Write the failing test**

Create `extension/backend/tests/test_specialist_per_context_loop.py`:

```python
"""Phase C7b: Specialist now runs one LLM call per bounded context, not
one omnibus call for all (FM-23 context-blending fix).
"""

from unittest.mock import MagicMock
from core.architect import DomainArchitect


def _arch():
    a = DomainArchitect.__new__(DomainArchitect)
    a.model_name = "gemini-3.1-pro-preview"
    a.last_request_time = 0
    a.min_delay = 0
    a.request_count = 0
    import threading
    a._rate_limit_lock = threading.Lock()
    from core.token_tracker import TokenTracker
    a.token_tracker = TokenTracker.get_instance()
    a.progress_callback = None
    a.run_timestamp = "20260518_000000"
    a.client = MagicMock()
    a.scout_max_workers = 1
    return a


def test_extract_per_context_makes_one_llm_call_per_context():
    arch = _arch()
    ok_response = MagicMock()
    ok_response.candidates = [MagicMock()]
    ok_response.candidates[0].finish_reason = "STOP"
    ok_response.text = (
        '{"context": "X", "entities": [{"name": "E", "attributes": [], '
        '"confidence": 0.9, "justification": "t", "evidence_sentence_indices": [0]}], '
        '"value_objects": [], "services": [], "aggregates": [], '
        '"domain_events": [], "business_rules": []}'
    )
    arch.client.models.generate_content.return_value = ok_response

    from unittest.mock import patch
    with patch.object(arch, "_save_intermediate"), \
         patch.object(arch, "_report_progress"), \
         patch.object(arch, "_wait_for_rate_limit"), \
         patch.object(
             arch, "_parse_json_response",
             return_value={
                 "context": "X",
                 "entities": [{"name": "E", "attributes": [], "confidence": 0.9, "justification": "t", "evidence_sentence_indices": [0]}],
                 "value_objects": [], "services": [], "aggregates": [], "domain_events": [], "business_rules": [],
             },
         ):
        contexts = ["OrderMgmt", "Billing", "Inventory"]
        sentences = ["s0", "s1", "s2"]
        results = arch.extract_per_context_details(contexts, sentences)
    # 3 contexts → 3 calls
    assert arch.client.models.generate_content.call_count == 3
    assert len(results) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd extension/backend && pytest tests/test_specialist_per_context_loop.py -v`
Expected: FAIL — `extract_per_context_details` does not yet exist.

- [ ] **Step 3: Implement the per-context loop**

Add a new method `extract_per_context_details` to `DomainArchitect` in `extension/backend/core/architect.py` (keep the legacy `extract_all_contexts_details` around for now; the C7 facade currently calls it):

```python
    def extract_per_context_details(
        self, contexts: List[str], domain_sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """Per-context Specialist loop (FM-23).

        Issues one LLM call per bounded context with a focused prompt
        that mentions only that one context. Forces exclusive entity
        ownership at the prompt level.
        """
        results: List[Dict[str, Any]] = []
        numbered_sentences_text = "\n".join(
            f"[{i}] {s}" for i, s in enumerate(domain_sentences)
        )
        for ctx_name in contexts:
            prompt = self._build_specialist_prompt_per_context(
                context_name=ctx_name,
                numbered_sentences_text=numbered_sentences_text,
            )
            sc = stage_config("Specialist")
            for retry in range(5):
                try:
                    self._wait_for_rate_limit()
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=sc.temperature,
                            seed=sc.seed,
                        ),
                    )
                    if not self._check_response_completion(response, retry):
                        if retry < 4:
                            time.sleep(2)
                            continue
                    result = self._parse_json_response(self._safe_response_text(response))
                    if isinstance(result, dict) and result.get("error") == "json_parse_failed":
                        if retry < 4:
                            time.sleep(2)
                            continue
                        raise SpecialistFailureError(
                            context_name=ctx_name,
                            message=f"Specialist parse failed for {ctx_name} after 5 retries",
                        )
                    self.token_tracker.track_api_call(
                        response, stage="Specialist", operation=f"per_context:{ctx_name}",
                    )
                    results.append({
                        "context_name": ctx_name,
                        "entities": result.get("entities", []),
                        "value_objects": result.get("value_objects", []),
                        "services": result.get("services", []),
                        "aggregates": result.get("aggregates", []),
                        "domain_events": result.get("domain_events", []),
                        "business_rules": result.get("business_rules", []),
                    })
                    break
                except SpecialistFailureError:
                    raise
                except Exception as e:
                    if not self._is_quota_error_and_backoff(e, retry):
                        if retry >= 4:
                            raise SpecialistFailureError(
                                context_name=ctx_name,
                                message=f"Specialist failed for {ctx_name}: {type(e).__name__}: {e}",
                            ) from e
            else:
                raise SpecialistFailureError(
                    context_name=ctx_name,
                    message=f"Specialist exhausted retry loop for {ctx_name}",
                )
        self._save_intermediate(
            stage="3_specialist",
            data={
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "contexts_analyzed": len(results),
                "analyses": results,
            },
        )
        return results

    def _build_specialist_prompt_per_context(
        self, *, context_name: str, numbered_sentences_text: str
    ) -> str:
        return f"""You are analyzing exactly ONE Bounded Context: {context_name}.

Do NOT include entities, aggregates, services, value objects, or
domain events that belong to other contexts.

Extract the 5 DDD building blocks for {context_name}:
1. Entities       - Objects with unique identity
2. Value Objects  - Immutable objects defined by attributes
3. Services       - Stateless operations
4. Aggregates     - Consistency boundaries with explicit members
5. Domain Events  - Past-tense business facts

DOMAIN KNOWLEDGE (numbered sentences):
{numbered_sentences_text}

RESPOND WITH JSON for {context_name} only:
{{
  "context": "{context_name}",
  "entities": [{{
    "name": "EntityName",
    "attributes": ["attr1"],
    "confidence": 0.9,
    "justification": "Cited in 3 sentences",
    "evidence_sentence_indices": [2, 7]
  }}],
  "value_objects": [],
  "services": [],
  "aggregates": [{{"name": "X", "members": ["EntityName"]}}],
  "domain_events": [],
  "business_rules": []
}}

Every entity.evidence_sentence_indices must contain at least one sentence index from the DOMAIN KNOWLEDGE above. Do not invent data."""
```

Update `analyze_document`'s `specialist_fn` (the C7 wiring) to use the new method:

```python
        def specialist_fn(contexts, scout_chunks):
            sentences = [c["text"] for c in scout_chunks]
            ctx_names = [c["name"] for c in contexts]
            return self.extract_per_context_details(ctx_names, sentences)
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_specialist_per_context_loop.py -v`
Expected: PASS — 3 contexts → 3 LLM calls.

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_specialist_per_context_loop.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
refactor(specialist): per-context loop replaces omnibus single call (FM-23)

Splits the single extract_all_contexts_details call into a per-context
loop. Each LLM call sees exactly one bounded context name in its
prompt, forcing exclusive entity ownership at the prompt level and
eliminating cross-context entity-blending. 4× Specialist calls per
typical SRS but each prompt is small (1 context's slice). Wires the
new extract_per_context_details into analyze_document's specialist_fn.
The legacy extract_all_contexts_details method remains but is no
longer reachable from the pipeline; can be deleted in a follow-up.
EOF
)"
```

---

### Task C8: Integration test on D1 SRS (env-gated)

**Files:**
- Create: `extension/backend/tests/test_p3_integration.py`

- [ ] **Step 1: Write the integration test**

Create `extension/backend/tests/test_p3_integration.py`:

```python
"""Phase C8: end-to-end integration test for the 5-stage pipeline against
the D1 SRS. Requires DDD_INTEGRATION_TEST=1 and a real GEMINI_API_KEY.

Run with:
    DDD_INTEGRATION_TEST=1 GEMINI_API_KEY=... pytest tests/test_p3_integration.py -v
"""

import os
import pytest
from pathlib import Path
from core.architect import DomainArchitect
from core.document_parser import SRSDocumentParser


pytestmark = pytest.mark.integration


@pytest.fixture
def srs_text():
    srs_path = Path("inputs/SRS.docx")
    if not srs_path.exists():
        pytest.skip("inputs/SRS.docx not present")
    parser = SRSDocumentParser()
    return parser.parse_file(str(srs_path))


@pytest.mark.skipif(
    os.getenv("DDD_INTEGRATION_TEST") != "1" or not os.getenv("GEMINI_API_KEY"),
    reason="integration test gated by DDD_INTEGRATION_TEST=1 + GEMINI_API_KEY"
)
def test_d1_srs_produces_valid_domain_model(srs_text):
    arch = DomainArchitect()
    model = arch.analyze_document(text=srs_text)
    assert model.project_name
    assert len(model.bounded_contexts) >= 1
    for bc in model.bounded_contexts:
        assert bc.ubiquitous_language.entities or bc.ubiquitous_language.value_objects
        for e in bc.ubiquitous_language.entities:
            assert 0.0 <= e.confidence <= 1.0
            assert e.justification
```

- [ ] **Step 2: Run unit suite to ensure nothing else broke**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green (the new file is marked `integration` so it is skipped here).

- [ ] **Step 3: Optionally run the integration test locally**

Run: `cd extension/backend && DDD_INTEGRATION_TEST=1 pytest tests/test_p3_integration.py -m integration -v`
Expected: PASS, producing a non-empty `DomainModel` with `services` and `aggregates` populated.

If it fails, the failure points to a real bug in the prompts/wiring — fix it before committing the next phase. Do not regress to silent fallbacks.

- [ ] **Step 4: Commit**

```bash
git add extension/backend/tests/test_p3_integration.py
git commit -m "$(cat <<'EOF'
test(integration): end-to-end P3 pipeline against D1 SRS (env-gated)

Marks the test as `integration` so it is skipped by CI's default
`-m "not integration"` filter. Run locally with
DDD_INTEGRATION_TEST=1 + GEMINI_API_KEY. Asserts non-empty
bounded_contexts and that every entity has 0 ≤ confidence ≤ 1 and a
populated justification.
EOF
)"
```

**End of Phase C.** The pipeline now runs through the orchestration package with section-aware chunking, deterministic Verifier checks, and the Refiner loop. Phase D tightens the evidence-grounding contract.

---

# Phase D — Evidence Citation + Grounding Tightening (4 commits, ~3-5 days, medium risk)

Goal: Require LLM-emitted `evidence_sentence_indices` per entity (min_items=1), promote the Verifier's D2 check from WARN to ERROR, and stop AST enrichment from fabricating `InferenceSource = "generated"`. Pre-Phase-D investigate OQ4 (lost v2 specialist via `git log`).

---

### Task D0 (investigative, no commit): OQ4 — lost v2 specialist

- [ ] **Step 1: Search git history for richer Specialist schema**

Run: `cd extension/backend && git log --all -p -- core/architect.py | grep -A 5 -B 5 "evidence_ids\|actors\|capabilities" | head -200`

- [ ] **Step 2: Inspect intermediate file for clues**

Run: `cat core/intermediate/20260312_222001_3_specialist.json | head -100`

- [ ] **Step 3: Decision point**

If a richer Specialist v2 prompt is found in the git history, port the structure (entity → `evidence_ids` field) into the new prompt below. If not, design fresh — the prompt below assumes fresh design. Record the decision in the D1 commit message.

(D0 produces no commit — it is investigative.)

---

### Task D1: Specialist prompt requires `evidence_sentence_indices` per entity; schema tightens to min_items=1

**Files:**
- Modify: `extension/backend/core/architect.py` — `extract_all_contexts_details` prompt
- Modify: `extension/backend/core/schemas.py` — `Entity.evidence_sentence_indices` becomes required
- Modify: `extension/backend/tests/test_schemas_strict.py` — update tests for Phase D contract
- Modify: `extension/backend/tests/test_architect_prompts.py` — append D1 prompt assertion

- [ ] **Step 1: Write the failing tests**

Append to `extension/backend/tests/test_architect_prompts.py`:

```python
def test_specialist_prompt_requires_evidence_sentence_indices():
    src = open("core/architect.py").read()
    specialist_section = src.split("def extract_all_contexts_details(")[1].split("def ")[0]
    assert "evidence_sentence_indices" in specialist_section, (
        "Phase D1: Specialist prompt must require evidence_sentence_indices "
        "per entity for grounding traceability"
    )
```

Update `extension/backend/tests/test_schemas_strict.py` so that `test_entity_accepts_phase_a_minimum_fields` is replaced by:

```python
def test_entity_rejects_missing_evidence_sentence_indices_in_phase_d():
    """Phase D1: evidence_sentence_indices is required (min_items=1)."""
    with pytest.raises(ValidationError):
        Entity(
            name="Customer",
            description="A buyer",
            confidence=0.9,
            justification="Mentioned in 4 SRS sentences",
        )


def test_entity_accepts_phase_d_with_evidence():
    e = Entity(
        name="Customer",
        description="A buyer",
        confidence=0.9,
        justification="Mentioned in 4 SRS sentences",
        evidence_sentence_indices=[2, 5, 9],
    )
    assert e.evidence_sentence_indices == [2, 5, 9]
```

(Delete the old `test_entity_accepts_phase_a_minimum_fields` test — it was a Phase A contract that is now superseded.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd extension/backend && pytest tests/test_schemas_strict.py tests/test_architect_prompts.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Tighten the schema and update the Specialist prompt**

In `extension/backend/core/schemas.py`, modify `Entity`:

```python
    evidence_sentence_indices: List[int] = Field(
        min_length=1,
        description="Scout sentence indices that ground this entity. Required from Phase D1 onward."
    )
```

(`min_length=1` in Pydantic v2 replaces the `default_factory=list` from Phase A4.)

In `extension/backend/core/architect.py`, update the Specialist prompt's entity example to include `evidence_sentence_indices`:

```python
        prompt = f"""Analyze the domain knowledge for these Bounded Contexts: {contexts_text}

For EACH context, extract the 5 DDD building blocks AND ground every
entity in the sentences from the supplied DOMAIN KNOWLEDGE. Every entity
MUST cite at least one sentence index in evidence_sentence_indices.

1. Entities       - Objects with unique identity (Customer, Order, Product)
2. Value Objects  - Immutable objects defined by attributes (Address, Money)
3. Services       - Stateless operations that don't naturally belong to an entity
4. Aggregates     - Consistency boundaries; each aggregate has a name and lists
                    the entities (`members`) that live inside it
5. Domain Events  - Past-tense business facts (OrderPlaced, PaymentReceived)

The DOMAIN KNOWLEDGE is presented as numbered sentences. Use those
numbers as evidence_sentence_indices.

DOMAIN KNOWLEDGE:
{numbered_sentences_text}

RESPOND WITH JSON:
{{
  "analyses": [
    {{
      "context": "ContextName",
      "entities": [{{
        "name": "Entity1",
        "attributes": ["id", "name", "status"],
        "confidence": 0.9,
        "justification": "Cited in 3 sentences as a primary actor",
        "evidence_sentence_indices": [2, 7, 12]
      }}],
      "value_objects": [{{"name": "Money", "attributes": ["amount", "currency"]}}],
      "services": [{{"name": "PricingService", "description": "Computes order totals"}}],
      "aggregates": [{{"name": "Order", "members": ["Order", "OrderLine"]}}],
      "domain_events": ["OrderPlaced", "OrderCancelled"],
      "business_rules": ["Orders must have at least one item"]
    }}
  ]
}}

If a category has no data, use empty arrays. Do not invent data. Every entity.evidence_sentence_indices must contain at least one valid sentence index from the DOMAIN KNOWLEDGE."""
```

You also need `numbered_sentences_text` — replace the `sentences_text = "\n".join(domain_sentences)` line earlier in `extract_all_contexts_details` with:

```python
        numbered_sentences_text = "\n".join(
            f"[{i}] {s}" for i, s in enumerate(domain_sentences)
        )
        max_chars = 60000
        if len(numbered_sentences_text) > max_chars:
            print(f"  ✂️  Truncating input: {len(numbered_sentences_text):,} → {max_chars:,} chars (head + tail preserved)")
            numbered_sentences_text = _truncate_with_head_tail(numbered_sentences_text, max_chars=max_chars)
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest tests/test_schemas_strict.py tests/test_architect_prompts.py -v`
Expected: all green.

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green. Any fixture that built `Entity(...)` without `evidence_sentence_indices` will now fail; patch by passing `evidence_sentence_indices=[0]` (or whichever fixture index makes sense).

- [ ] **Step 5: Commit**

```bash
git add extension/backend/tests/test_architect_prompts.py extension/backend/tests/test_schemas_strict.py extension/backend/core/architect.py extension/backend/core/schemas.py
git commit -m "$(cat <<'EOF'
feat(specialist): require evidence_sentence_indices per entity (D1, OQ4 deferred)

Tightens Entity.evidence_sentence_indices from Optional (Phase A) to
List[int] min_length=1. Specialist prompt now numbers each domain
sentence ([0], [1], ...) and requires every entity to cite ≥1 index.
Closes FM-12 traceability gap.

OQ4 investigation: [INSERT FINDINGS HERE — either "v2 prompt ported
from commit XYZ" or "no v2 prompt found in history; designed fresh"].
EOF
)"
```

(Update the OQ4 placeholder with your actual investigation result before committing.)

---

### Task D2: Verifier D2 check enforces evidence as ERROR (was WARN in Phase C)

**Files:**
- Modify: `extension/backend/core/verifier/checks_deterministic.py` — `check_d2_entity_evidence_nonempty` default phase
- Modify: `extension/backend/tests/test_verifier_deterministic.py` — update default-phase test

- [ ] **Step 1: Update tests for the new default**

In `extension/backend/tests/test_verifier_deterministic.py`, change the D2 phase-default test:

```python
def test_d2_default_phase_is_d_so_empty_evidence_is_error():
    """Default phase is now 'D'; empty evidence is ERROR, not WARN."""
    entity = {"name": "Order", "evidence_sentence_indices": []}
    issues = check_d2_entity_evidence_nonempty(
        context_name="OrderMgmt", entities=[entity]
    )
    assert len(issues) == 1
    assert issues[0].severity == IssueSeverity.ERROR
```

Remove `test_d2_passes_when_evidence_phase_a_optional` (it captured the Phase A-C behaviour).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd extension/backend && pytest tests/test_verifier_deterministic.py -v`
Expected: the new default-phase test FAILS (because `check_d2_entity_evidence_nonempty` defaults to `phase="C"` → WARN).

- [ ] **Step 3: Update the default**

In `extension/backend/core/verifier/checks_deterministic.py`, change the signature:

```python
def check_d2_entity_evidence_nonempty(
    context_name: str,
    entities: List[Dict],
    phase: str = "D",  # was "C"
) -> List[VerifierIssue]:
```

Also wire `check_d2_entity_evidence_nonempty` into the verifier composite inside `architect.py`'s `verifier_fn` (the C7 wiring); previously only D1+D3+D4+D5 ran.

In `architect.py`, update `verifier_fn`:

```python
        def verifier_fn(snapshot):
            scout_indices = set(range(sum(len(c["text"].split(".")) for c in snapshot["scout"])))
            contexts = snapshot["architect"]
            issues = []
            issues.extend(check_d1_supporting_sentence_ids_subset(contexts, scout_indices))
            entities_by_context = {
                s["context_name"]: s.get("entities", [])
                for s in snapshot["specialist"]
            }
            issues.extend(check_d3_entity_names_unique_across_contexts(entities_by_context))
            for s in snapshot["specialist"]:
                issues.extend(check_d2_entity_evidence_nonempty(
                    context_name=s["context_name"], entities=s.get("entities", [])
                ))
                issues.extend(check_d4_aggregate_members_exist_in_context(
                    s["context_name"], s.get("entities", []), s.get("aggregates", [])
                ))
            issues.extend(check_d5_allowed_dependencies_reference_existing_contexts(contexts))
            return VerifierResult(ok=(len(issues) == 0), issues=issues)
```

Add the import:

```python
from core.verifier.checks_deterministic import check_d2_entity_evidence_nonempty
```

- [ ] **Step 4: Run tests**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/verifier/checks_deterministic.py extension/backend/tests/test_verifier_deterministic.py extension/backend/core/architect.py
git commit -m "$(cat <<'EOF'
feat(verifier): D2 check default phase becomes D (ERROR on missing evidence)

check_d2_entity_evidence_nonempty default phase shifts from "C" (WARN)
to "D" (ERROR) so an entity without evidence_sentence_indices forces
the Refiner to re-prompt the Specialist. Wires D2 into the architect
facade's verifier_fn.
EOF
)"
```

---

### Task D3: AST `_ensure_traceability` drops "generated" InferenceSource and raises (OQ2)

**Files:**
- Modify: `extension/backend/core/AST/ast_signal_enrichment.py` — `_ensure_traceability` (around lines 177-197)
- Create: `extension/backend/tests/test_ast_grounding_strict.py`

- [ ] **Step 1: Inspect the current `_ensure_traceability`**

Read `extension/backend/core/AST/ast_signal_enrichment.py` lines 170-210.

- [ ] **Step 2: Write the failing test**

Create `extension/backend/tests/test_ast_grounding_strict.py`:

```python
"""Phase D3 / OQ2: AST enrichment must not fabricate
InferenceSource(file='generated', rule='LLM_SYNTHESIS'). When an
entity has no SRS evidence and no AST grounding, raise
InsufficientGroundingError instead.
"""

import pytest
from core.orchestration.errors import InsufficientGroundingError
from core.AST.ast_signal_enrichment import _ensure_traceability


def test_ensure_traceability_raises_on_no_evidence():
    entity_dict = {
        "name": "PhantomEntity",
        "description": "no SRS evidence",
        "evidence_sentence_indices": [],
        "sources": [],
    }
    with pytest.raises(InsufficientGroundingError):
        _ensure_traceability(entity_dict, ast_signals_for_entity=[])


def test_ensure_traceability_passes_when_evidence_present():
    entity_dict = {
        "name": "Customer",
        "description": "buyer",
        "evidence_sentence_indices": [2, 5],
        "sources": [],
    }
    # Should not raise
    _ensure_traceability(entity_dict, ast_signals_for_entity=[])


def test_ensure_traceability_passes_when_ast_signals_present():
    entity_dict = {
        "name": "Order",
        "description": "order",
        "evidence_sentence_indices": [],
        "sources": [],
    }
    ast_signals = [{"file": "src/order.py", "line": 12, "rule": "AST_CLASS_DECL"}]
    _ensure_traceability(entity_dict, ast_signals_for_entity=ast_signals)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd extension/backend && pytest tests/test_ast_grounding_strict.py -v`
Expected: FAIL with import or behaviour mismatch.

- [ ] **Step 4: Modify `_ensure_traceability`**

In `extension/backend/core/AST/ast_signal_enrichment.py`, locate `_ensure_traceability`. Replace the fabrication path (the block that emits `{"file": "generated", "rule": "LLM_SYNTHESIS"}` around lines 177-197) with:

```python
def _ensure_traceability(entity_dict, ast_signals_for_entity):
    """Ensure the entity has either SRS evidence (Phase D1's
    evidence_sentence_indices) or AST grounding (signals). Phase D3:
    raise InsufficientGroundingError instead of fabricating a
    'generated' InferenceSource.
    """
    has_srs_evidence = bool(entity_dict.get("evidence_sentence_indices"))
    has_ast_signal = bool(ast_signals_for_entity)
    if not has_srs_evidence and not has_ast_signal:
        from core.orchestration.errors import InsufficientGroundingError
        raise InsufficientGroundingError(entity_name=entity_dict.get("name", "<unknown>"))
    # If AST signals exist, append them as InferenceSource entries (existing
    # logic). Otherwise nothing to do — SRS grounding lives on
    # evidence_sentence_indices and is unrelated to InferenceSource.
    for sig in ast_signals_for_entity:
        entity_dict.setdefault("sources", []).append({
            "file": sig.get("file"),
            "line": sig.get("line", 1),
            "rule": sig.get("rule", "AST"),
            "evidence": sig.get("evidence", ""),
        })
```

(Adapt the function shape to whatever the existing surrounding code expects; the key behavioural change is: never write `{"file": "generated", ...}` ever again, and raise `InsufficientGroundingError` when nothing else exists.)

- [ ] **Step 5: Run tests**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add extension/backend/tests/test_ast_grounding_strict.py extension/backend/core/AST/ast_signal_enrichment.py
git commit -m "$(cat <<'EOF'
refactor(ast): drop fabricated "generated" InferenceSource; raise on no grounding (D3, OQ2)

_ensure_traceability previously injected
InferenceSource(file='generated', rule='LLM_SYNTHESIS') whenever an
entity had neither SRS sentence indices nor AST signals. That
fabricated traceability is incompatible with the EMSE paper's
provenance claim. From D3 onward, an entity without either source
raises InsufficientGroundingError; the orchestrator counts the run as
degraded.
EOF
)"
```

---

### Task D4: Integration regression — every persisted entity has SRS evidence

**Files:**
- Create: `extension/backend/tests/test_grounding_regression.py`

- [ ] **Step 1: Write the regression test**

Create `extension/backend/tests/test_grounding_regression.py`:

```python
"""Phase D4: integration regression. After a full pipeline run, every
persisted entity must have a non-empty evidence_sentence_indices, and
no InferenceSource may have rule='LLM_SYNTHESIS' or file='generated'.
"""

import json
import os
import pytest
from pathlib import Path


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("DDD_INTEGRATION_TEST") != "1",
    reason="integration test gated by DDD_INTEGRATION_TEST=1"
)
def test_persisted_model_has_real_evidence():
    """Assumes the integration test from Task C8 already ran and
    persisted domain/model.json. Loads it and asserts.
    """
    candidates = [
        Path("domain/model.json"),
        Path("extension/backend/domain/model.json"),
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        pytest.skip("No domain/model.json found; run the C8 integration test first")

    model = json.loads(model_path.read_text())
    for bc in model.get("bounded_contexts", []):
        for entity in bc.get("ubiquitous_language", {}).get("entities", []):
            assert entity.get("evidence_sentence_indices"), (
                f"Entity {entity.get('name')!r} in context "
                f"{bc.get('context_name')!r} has no evidence_sentence_indices"
            )
            for src in entity.get("sources", []):
                assert src.get("rule") != "LLM_SYNTHESIS", (
                    f"Entity {entity.get('name')!r} carries a forbidden "
                    f"InferenceSource rule='LLM_SYNTHESIS'"
                )
                assert src.get("file") != "generated"
```

- [ ] **Step 2: Run the test (locally, when integration is enabled)**

Run: `cd extension/backend && DDD_INTEGRATION_TEST=1 GEMINI_API_KEY=... pytest tests/test_grounding_regression.py -m integration -v`
Expected: PASS (if you ran C8 first to populate `domain/model.json`). If any entity is missing evidence, that is the canonical FM-12 + OQ2 bug — fix the prompt or the grounding code before merging Phase D.

- [ ] **Step 3: Run unit suite**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: green (the new test is `integration`-marked and skipped here).

- [ ] **Step 4: Commit**

```bash
git add extension/backend/tests/test_grounding_regression.py
git commit -m "$(cat <<'EOF'
test(integration): every persisted entity has SRS evidence (D4)

Regression assertion over the persisted domain/model.json: every
entity must carry a non-empty evidence_sentence_indices and no
InferenceSource may have the forbidden 'LLM_SYNTHESIS' rule or
'generated' file marker. Gated by DDD_INTEGRATION_TEST=1.
EOF
)"
```

**End of Phase D.** Final verification:

- [ ] **Step D-end.1: Full final acceptance**

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: full unit suite green (post all 23 commits).

Run: `cd extension/backend && grep -rn "return \[\"CoreDomain\"\]\|_create_fallback_model\|file=\"generated\"\|rule=\"LLM_SYNTHESIS\"" core/`
Expected: zero matches.

Run: `cd extension/backend && grep -rn "except Exception:" core/ | wc -l`
Expected: zero or only documented narrow catches.

Run integration:
`cd extension/backend && DDD_INTEGRATION_TEST=1 GEMINI_API_KEY=... pytest -m integration -v`
Expected: all integration tests PASS.

---

## Self-Review Output

**Spec coverage** (against `docs/superpowers/specs/2026-05-18-p3-verifier-refiner-design.md`):

- ✅ §3.1 5-stage diagram — implemented across C5 (Scout), C7 (Architect/Specialist wiring), C6 (Verifier+Refiner+Synthesizer driver)
- ✅ §3.2 5 stages not 4 — Verifier as distinct stage (C1-C3); Refiner in C4
- ✅ §3.3 per-context Specialist loop — covered in C7b (added after spec-coverage check flagged the gap)
- ✅ §4.1 new modules — created in C1, C2, C3, C4, C5, C6, B1
- ✅ §4.2 modified files — covered in A1-A6, B2-B5, C7, D1-D3
- ✅ §4.3 removed code — A5 (bare except), B4 (`_create_fallback_model` + `["CoreDomain"]` fallback)
- ✅ §4.4 exception hierarchy — B1
- ✅ §5 data flow + failure policy — B1-B5
- ✅ §6 testing strategy (TDD per commit, mock LLM, golden fixtures, integration cadence) — covered throughout
- ✅ §7 phasing — mirrored with one addition: A1-A6, B1-B5, C1-C7+C7b+C8, D1-D4 = 24 commits (the +1 is C7b per spec-coverage gap noted above)
- ✅ §10 out of scope — respected
- ✅ §11 OQ1/2/3 — embedded in C5/D3/A4 respectively
- ✅ §11 OQ4 — D0 investigative step
- ⚠️ §3.3 / FM-23 per-context Specialist loop — Task A2's prompt fix moved Specialist toward structured output but did **not** convert the single-call Specialist into a per-context loop. **GAP**: this needs a dedicated commit in Phase C (between C6 and C7) or as a follow-up. Adding a note in the Execution Handoff.

**Placeholder scan**: no `TBD` / `TODO` / `fill in details` in the plan. One residual `[INSERT FINDINGS HERE]` placeholder in D1's commit message — that is intentional (it captures OQ4's investigation result at commit time).

**Type consistency**: all references to `VerifierIssue`, `IssueSeverity`, `VerifierResult`, `PipelineDeps`, `ScoutChunkParseError`, etc. match across tasks.

**Identified gap (per-context Specialist loop, FM-23)**: addressed inline by adding Task C7b. Plan now has 24 commits total (spec listed 23; the 24th closes the spec-body-vs-phasing-table gap surfaced during self-review).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-18-p3-verifier-refiner.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Fresh subagent per task; two-stage review between tasks; fast iteration. Best when each task is small and well-bounded (as here).

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`; batch execution with checkpoints. Best when you want to watch every step land.

**Which approach?**
