# Holistic Critic — Active Critique Loop (Topology A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded critique→revise loop where an LLM Critic judges DDD design quality of the synthesized `DomainModel` each cycle and routes findings back to the existing producer stages (Architect / Specialist) for re-derivation, returning the best model — all behind `DDD_CRITIC_LOOP`, with `critic=None` preserving today's exact behavior.

**Architecture:** A new `core/critic/` package holds a pure evaluator (`run_critic`), finding routing, and the loop driver (`run_critique_loop`). `run_pipeline` is refactored to extract a single-pass `_generate_once` and dispatch to the loop when a `critic` dep is present. Revision flows through `architect_with_feedback` / `specialist_with_feedback` + re-synthesis, so grounding + D6/D7/D8 invariants stay valid by construction.

**Tech Stack:** Python 3.12, Pydantic v2, pytest (`-m "not integration"`), the existing `core/llm` `structured_output` contract, `core/orchestration` injected-deps pattern.

**Spec:** `docs/superpowers/specs/2026-05-25-holistic-critic-design.md`

**Conventions:** TDD (failing test first). Atomic Conventional Commits with trailer:
```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```
Gate after each task: `cd extension/backend && pytest -m "not integration" -q` + `pyright` (blocking). All paths below are relative to `extension/backend/`.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `core/schemas.py` | modify | Add `CritiqueFinding`, `CriticReport`, `CriticLoopTrace`; add `DomainModel.critic_report`. |
| `core/critic/__init__.py` | create | Package exports. |
| `core/critic/errors.py` | create | `CriticError`. |
| `core/critic/types.py` | create | LLM-facing `ProposedFinding`, `CriticResponse`; `CritiqueCycleMemory`. |
| `core/critic/routing.py` | create | `partition_findings`, `adapt_findings_to_issues`, `model_diff_summary`. |
| `core/critic/prompt.py` | create | `build_critique_prompt`. |
| `core/critic/critic.py` | create | `run_critic(...) -> CriticReport`. |
| `core/critic/loop.py` | create | `run_critique_loop`, `critique_score`, `findings_signature`. |
| `core/orchestration/pipeline.py` | modify | Extract `_generate_once`; add `PipelineDeps.critic`; dispatch. |
| `configs/models.py` | modify | Add `"Critic": "domain_extraction"` to `STAGE_TO_GROUP`. |
| `core/architect.py` | modify | Wire `critic_fn` into `analyze_document` (gated on `DDD_CRITIC_LOOP`). |
| `tests/test_critic_*.py` | create | Unit + integration tests per task. |

---

## Task 1: Persisted critic schema in `core/schemas.py`

**Files:**
- Modify: `core/schemas.py` (append new models before `DomainModel`; add one field to `DomainModel`)
- Test: `tests/test_critic_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_schema.py
"""Persisted critic schema: additive, backward-compatible."""
import pytest
from pydantic import ValidationError
from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
    CritiqueFinding, CriticReport, CriticLoopTrace,
)


def _minimal_model() -> DomainModel:
    return DomainModel(
        project_name="P",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name="Ctx",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="An order.", confidence=0.9,
                    justification="cited", evidence_sentence_indices=[0],
                )],
                value_objects=None, domain_events=None,
            ),
        )],
        global_rules=None,
    )


def test_domain_model_critic_report_defaults_none():
    assert _minimal_model().critic_report is None


def test_critique_finding_requires_known_type_and_priority():
    f = CritiqueFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Ctx.Order", rationale="no behavior",
        proposed_revision="add methods",
    )
    assert f.evidence_sentence_indices == []
    with pytest.raises(ValidationError):
        CritiqueFinding(
            finding_type="NONSENSE", priority="high",
            target_ref="x", rationale="y", proposed_revision="z",
        )


def test_critic_report_attaches_to_model():
    m = _minimal_model()
    m.critic_report = CriticReport(
        model_id="gemini-3.1-pro-preview",
        findings=[],
        loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"),
    )
    assert m.critic_report.loop.outcome == "converged"
    assert m.critic_report.score == 0.0


def test_old_model_json_without_critic_report_deserializes():
    payload = _minimal_model().model_dump()
    payload.pop("critic_report", None)
    restored = DomainModel.model_validate(payload)
    assert restored.critic_report is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_schema.py -q`
Expected: FAIL with `ImportError: cannot import name 'CritiqueFinding'`.

- [ ] **Step 3: Implement the schema**

In `core/schemas.py`, add these classes immediately before `class DomainModel`:

```python
class CritiqueFinding(BaseModel):
    """One DDD design-quality finding emitted by the Critic (persisted form)."""
    finding_type: Literal[
        "CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "BOUNDARY_SMELL",
        "ANEMIC_ENTITY", "ANEMIC_MODEL", "MISSING_AGGREGATE",
        "MISPLACED_ENTITY", "NAMING_SMELL", "LOW_CONFIDENCE", "OTHER",
    ]
    priority: Literal["high", "medium", "low"]
    target_ref: str = Field(description="e.g. 'context:Ordering' | 'entity:Ordering.Order'")
    rationale: str
    proposed_revision: str
    evidence_sentence_indices: List[int] = Field(default_factory=list)


class CriticLoopTrace(BaseModel):
    """Per-document trace of the critique loop."""
    cycles_used: int
    best_cycle: int
    outcome: Literal["converged", "exhausted", "flapped", "failed"]
    score_per_cycle: List[float] = Field(default_factory=list)
    findings_count_per_cycle: List[int] = Field(default_factory=list)


class CriticReport(BaseModel):
    """Best cycle's critique + loop trace, attached to the DomainModel."""
    model_id: str
    findings: List[CritiqueFinding] = Field(default_factory=list)
    score: float = 0.0
    malformed_findings: int = 0
    loop: CriticLoopTrace
    error: Optional[str] = None
```

Then add one field to `DomainModel` (after `global_rules`):

```python
    critic_report: Optional["CriticReport"] = Field(
        default=None,
        description="Holistic Critic loop output (best cycle). None when the loop did not run.",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_schema.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add core/schemas.py tests/test_critic_schema.py
git commit -m "feat(critic): persisted critic schema (CritiqueFinding/CriticReport/CriticLoopTrace)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `CriticError` + package skeleton

**Files:**
- Create: `core/critic/__init__.py`, `core/critic/errors.py`
- Test: `tests/test_critic_errors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_errors.py
from core.critic.errors import CriticError


def test_critic_error_carries_reason_and_cycle():
    err = CriticError(reason="json_failed: schema_mismatch", cycle=2)
    assert err.cycle == 2
    assert "json_failed" in str(err)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_errors.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.critic'`.

- [ ] **Step 3: Implement**

```python
# core/critic/__init__.py
"""Holistic Critic package: pure evaluator + finding routing + bounded loop."""
```

```python
# core/critic/errors.py
"""Critic-stage error taxonomy."""
from typing import Optional


class CriticError(Exception):
    """Raised when the Critic LLM call is unrecoverable (json_failed after
    retries, or empty parse). Caught at the loop boundary and recorded;
    never silently swallowed."""

    def __init__(self, reason: str, cycle: Optional[int] = None):
        self.reason = reason
        self.cycle = cycle
        super().__init__(f"CriticError(cycle={cycle}): {reason}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_errors.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add core/critic/__init__.py core/critic/errors.py tests/test_critic_errors.py
git commit -m "feat(critic): package skeleton + CriticError

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: LLM-facing types (`core/critic/types.py`)

**Files:**
- Create: `core/critic/types.py`
- Test: `tests/test_critic_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_types.py
from core.critic.types import ProposedFinding, CriticResponse, CritiqueCycleMemory


def test_critic_response_parses_analysis_and_findings():
    resp = CriticResponse.model_validate({
        "analysis": "Ordering context is cohesive; Order entity is anemic.",
        "findings": [{
            "finding_type": "ANEMIC_ENTITY", "priority": "high",
            "target_ref": "entity:Ordering.Order", "rationale": "no behavior",
            "proposed_revision": "add place()/cancel()",
            "evidence_sentence_indices": [3],
        }],
    })
    assert resp.analysis.startswith("Ordering")
    assert resp.findings[0].finding_type == "ANEMIC_ENTITY"


def test_cycle_memory_holds_prior_findings_and_diff():
    mem = CritiqueCycleMemory(
        cycle=0,
        findings_summary=["high ANEMIC_ENTITY entity:Ordering.Order"],
        diff_summary="cycle 0: initial model",
    )
    assert mem.cycle == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_types.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.critic.types'`.

- [ ] **Step 3: Implement**

```python
# core/critic/types.py
"""LLM-facing critic schema + loop memory.

ProposedFinding/CriticResponse are what the LLM emits (no provenance).
run_critic maps them to the persisted core.schemas.CritiqueFinding/CriticReport.
"""
from typing import List, Literal
from pydantic import BaseModel, Field


class ProposedFinding(BaseModel):
    finding_type: Literal[
        "CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "BOUNDARY_SMELL",
        "ANEMIC_ENTITY", "ANEMIC_MODEL", "MISSING_AGGREGATE",
        "MISPLACED_ENTITY", "NAMING_SMELL", "LOW_CONFIDENCE", "OTHER",
    ]
    priority: Literal["high", "medium", "low"]
    target_ref: str
    rationale: str
    proposed_revision: str
    evidence_sentence_indices: List[int] = Field(default_factory=list)


class CriticResponse(BaseModel):
    """Schema-enforced critic output. `analysis` is the CoT scratchpad the
    model fills before listing findings (kept short, not persisted verbatim)."""
    analysis: str = Field(default="", description="Step-by-step DDD reasoning before findings.")
    findings: List[ProposedFinding] = Field(default_factory=list)


class CritiqueCycleMemory(BaseModel):
    """Reflexion memory for one prior cycle, fed back into the next critique."""
    cycle: int
    findings_summary: List[str] = Field(default_factory=list)
    diff_summary: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_types.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add core/critic/types.py tests/test_critic_types.py
git commit -m "feat(critic): LLM-facing CriticResponse/ProposedFinding + cycle memory

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Finding routing + model diff (`core/critic/routing.py`)

**Files:**
- Create: `core/critic/routing.py`
- Test: `tests/test_critic_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_routing.py
from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
    CritiqueFinding,
)
from core.critic.routing import (
    partition_findings, adapt_structural_to_issues, adapt_content_to_issues,
    model_diff_summary,
)


def _finding(ft, pri="high", target="context:Ctx"):
    return CritiqueFinding(
        finding_type=ft, priority=pri, target_ref=target,
        rationale="r", proposed_revision="p",
    )


def _model(ctx_names, entities_by_ctx):
    return DomainModel(
        project_name="P",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name=c,
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name=e, description="d", confidence=0.9,
                    justification="j", evidence_sentence_indices=[0],
                ) for e in entities_by_ctx.get(c, [])],
                value_objects=None, domain_events=None,
            ),
        ) for c in ctx_names],
        global_rules=None,
    )


def test_partition_splits_structural_content_advisory():
    findings = [
        _finding("CONTEXT_SHOULD_MERGE", "high"),     # structural
        _finding("ANEMIC_ENTITY", "medium", "entity:Ctx.Order"),  # content
        _finding("NAMING_SMELL", "low", "entity:Ctx.Order"),      # low → advisory
    ]
    structural, content, advisory = partition_findings(findings)
    assert [f.finding_type for f in structural] == ["CONTEXT_SHOULD_MERGE"]
    assert [f.finding_type for f in content] == ["ANEMIC_ENTITY"]
    assert [f.finding_type for f in advisory] == ["NAMING_SMELL"]


def test_misplaced_entity_is_content_not_structural():
    structural, content, _ = partition_findings([_finding("MISPLACED_ENTITY", "high", "entity:A.X")])
    assert structural == []
    assert len(content) == 1


def test_adapt_structural_keeps_generic_target():
    issues = adapt_structural_to_issues([_finding("CONTEXT_SHOULD_MERGE", "high", "context:Ctx")])
    assert issues[0].target == "context:Ctx"
    assert issues[0].severity == "ERROR"          # high → ERROR
    assert "r" in issues[0].message and "p" in issues[0].suggestion


def test_adapt_content_emits_specialist_prefix_for_affected_context():
    # _specialist_with_feedback derives the affected context from a
    # "specialist:<ctx>" prefix (architect.py:_parse_target_ctx); the adapter
    # MUST emit that prefix or no context is re-extracted.
    issues = adapt_content_to_issues([_finding("ANEMIC_ENTITY", "high", "entity:Ctx.Order")])
    assert issues[0].location == "specialist:Ctx"
    assert issues[0].severity == "ERROR"


def test_model_diff_summary_reports_context_and_entity_deltas():
    before = _model(["A", "B"], {"A": ["X"]})
    after = _model(["A"], {"A": ["X", "Y"]})
    summary = model_diff_summary(before, after)
    assert "B" in summary           # removed context mentioned
    assert "Y" in summary           # added entity mentioned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_routing.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.critic.routing'`.

- [ ] **Step 3: Implement**

```python
# core/critic/routing.py
"""Route critic findings to producer stages + summarize model diffs.

Structural findings re-derive the architecture (Architect); content findings
re-extract within existing contexts (Specialist). Low-priority findings are
advisory only (never trigger a regeneration).
"""
from dataclasses import dataclass
from typing import List, Tuple
from core.schemas import CritiqueFinding, DomainModel

_STRUCTURAL = {"CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "BOUNDARY_SMELL"}
# everything else routable is content; OTHER is advisory-only.
_CONTENT = {
    "ANEMIC_ENTITY", "ANEMIC_MODEL", "MISSING_AGGREGATE",
    "MISPLACED_ENTITY", "NAMING_SMELL", "LOW_CONFIDENCE",
}


def partition_findings(
    findings: List[CritiqueFinding],
) -> Tuple[List[CritiqueFinding], List[CritiqueFinding], List[CritiqueFinding]]:
    """Return (structural, content, advisory). Only high/medium findings are
    routable; low priority and OTHER are advisory."""
    structural: List[CritiqueFinding] = []
    content: List[CritiqueFinding] = []
    advisory: List[CritiqueFinding] = []
    for f in findings:
        if f.priority == "low" or f.finding_type == "OTHER":
            advisory.append(f)
        elif f.finding_type in _STRUCTURAL:
            structural.append(f)
        elif f.finding_type in _CONTENT:
            content.append(f)
        else:
            advisory.append(f)
    return structural, content, advisory


@dataclass
class _CritiqueIssue:
    """Adapter so a CritiqueFinding is consumable by the producer feedback
    paths. Architect feedback (_build_grounding_feedback_block) reads
    .target/.location/.message generically. Specialist feedback requires the
    location to start with 'specialist:<ctx>' so _parse_target_ctx
    (architect.py) flags the affected context, and render_refinement_prompt
    reads .suggestion. All fields are populated to satisfy both consumers."""
    severity: str
    target: str
    location: str
    message: str
    suggestion: str


def _context_of(target_ref: str) -> str:
    """'context:Ordering' -> 'Ordering'; 'entity:Ordering.Order' -> 'Ordering'."""
    body = target_ref.split(":", 1)[-1]
    return body.split(".")[0].strip()


def adapt_structural_to_issues(findings: List[CritiqueFinding]) -> List[_CritiqueIssue]:
    """Architect-bound feedback: generic target/message (no stage prefix needed)."""
    out: List[_CritiqueIssue] = []
    for f in findings:
        out.append(_CritiqueIssue(
            severity="ERROR" if f.priority == "high" else "WARN",
            target=f.target_ref, location=f.target_ref,
            message=f"{f.rationale} | suggested: {f.proposed_revision}",
            suggestion=f.proposed_revision,
        ))
    return out


def adapt_content_to_issues(findings: List[CritiqueFinding]) -> List[_CritiqueIssue]:
    """Specialist-bound feedback: target/location MUST carry the
    'specialist:<ctx>' prefix so _specialist_with_feedback flags the right
    context as affected (architect.py:_parse_target_ctx splits on ':' then '.')."""
    out: List[_CritiqueIssue] = []
    for f in findings:
        ctx = _context_of(f.target_ref)
        out.append(_CritiqueIssue(
            severity="ERROR" if f.priority == "high" else "WARN",
            target=f"specialist:{ctx}", location=f"specialist:{ctx}",
            message=f"{f.rationale} | suggested: {f.proposed_revision}",
            suggestion=f.proposed_revision,
        ))
    return out


def _entity_names(model: DomainModel) -> dict:
    return {
        bc.context_name: {e.name for e in bc.ubiquitous_language.entities}
        for bc in model.bounded_contexts
    }


def model_diff_summary(before: DomainModel, after: DomainModel) -> str:
    """Compact deterministic diff for Reflexion memory."""
    before_ctx = {bc.context_name for bc in before.bounded_contexts}
    after_ctx = {bc.context_name for bc in after.bounded_contexts}
    parts: List[str] = []
    added_ctx = sorted(after_ctx - before_ctx)
    removed_ctx = sorted(before_ctx - after_ctx)
    if added_ctx:
        parts.append(f"contexts added: {', '.join(added_ctx)}")
    if removed_ctx:
        parts.append(f"contexts removed: {', '.join(removed_ctx)}")
    be, ae = _entity_names(before), _entity_names(after)
    for ctx in sorted(after_ctx & before_ctx):
        added_e = sorted(ae.get(ctx, set()) - be.get(ctx, set()))
        removed_e = sorted(be.get(ctx, set()) - ae.get(ctx, set()))
        if added_e:
            parts.append(f"{ctx}: entities added: {', '.join(added_e)}")
        if removed_e:
            parts.append(f"{ctx}: entities removed: {', '.join(removed_e)}")
    return "; ".join(parts) if parts else "no structural change"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_routing.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add core/critic/routing.py tests/test_critic_routing.py
git commit -m "feat(critic): finding routing (structural/content/advisory) + model diff

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Critique prompt builder (`core/critic/prompt.py`)

**Files:**
- Create: `core/critic/prompt.py`
- Test: `tests/test_critic_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_prompt.py
from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata
from core.critic.types import CritiqueCycleMemory
from core.critic.prompt import build_critique_prompt


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name="Ordering",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="An order.", confidence=0.9,
                    justification="j", evidence_sentence_indices=[0],
                )],
                value_objects=None, domain_events=None,
            ),
        )],
        global_rules=None,
    )


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="A customer places an order.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=27),
    )


def test_prompt_includes_model_scout_and_schema_directive():
    prompt = build_critique_prompt(_model(), _scout(), history=[])
    assert "Ordering" in prompt
    assert "A customer places an order." in prompt        # scout grounding
    assert "0" in prompt                                  # sentence index
    assert "findings" in prompt                           # schema directive
    assert "step" in prompt.lower()                       # CoT instruction


def test_prompt_includes_reflexion_history_when_present():
    history = [CritiqueCycleMemory(
        cycle=0, findings_summary=["high ANEMIC_ENTITY entity:Ordering.Order"],
        diff_summary="Ordering: entities added: Payment",
    )]
    prompt = build_critique_prompt(_model(), _scout(), history=history)
    assert "PREVIOUS CYCLES" in prompt
    assert "ANEMIC_ENTITY" in prompt
    assert "entities added: Payment" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_prompt.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.critic.prompt'`.

- [ ] **Step 3: Implement**

```python
# core/critic/prompt.py
"""Build the holistic-critique prompt: model + Scout grounding + Reflexion
history, instructing CoT reasoning before structured findings."""
import json
from typing import List
from core.schemas import DomainModel
from core.pipeline_contracts import ScoutOutput
from core.critic.types import CritiqueCycleMemory

_INSTRUCTIONS = """You are a senior Domain-Driven Design reviewer. You are given a
domain model assembled from a requirements document, plus the numbered source
sentences it was derived from.

Think step by step (this is your `analysis` field), reviewing in order:
1. Bounded contexts: cohesion, over-splitting, god-contexts, boundary smells.
2. Entities: anemic (no behavior implied), misplaced (belong in another context).
3. Aggregates: missing consistency boundaries.
4. Naming: ubiquitous-language smells.
Then emit `findings`. For each finding set:
- finding_type (one of the allowed values),
- priority: high (serious structural/design flaw), medium (worth fixing),
  low (cosmetic / advisory only),
- target_ref: "context:<Name>" or "entity:<Context>.<Entity>",
- rationale (why it is a problem), proposed_revision (what to change),
- evidence_sentence_indices: source sentence indices that justify the finding
  (use the numbered list; [] if none).
Do NOT rewrite the model. Only critique it. Be specific and grounded.
"""


def _serialize_model(model: DomainModel) -> str:
    compact = {
        "project_name": model.project_name,
        "bounded_contexts": [
            {
                "context_name": bc.context_name,
                "description": bc.description,
                "allowed_dependencies": bc.allowed_dependencies,
                "entities": [
                    {"name": e.name, "description": e.description,
                     "confidence": e.confidence}
                    for e in bc.ubiquitous_language.entities
                ],
                "aggregates": [
                    {"name": a.name, "members": a.members}
                    for a in (bc.ubiquitous_language.aggregates or [])
                ],
            }
            for bc in model.bounded_contexts
        ],
    }
    return json.dumps(compact, indent=2, ensure_ascii=False)


def _serialize_scout(scout: ScoutOutput) -> str:
    return "\n".join(f"[{s.index}] {s.text}" for s in scout.sentences)


def _serialize_history(history: List[CritiqueCycleMemory]) -> str:
    if not history:
        return ""
    lines = ["PREVIOUS CYCLES (do not re-report already-addressed issues):"]
    for mem in history:
        lines.append(f"- cycle {mem.cycle} findings: " + "; ".join(mem.findings_summary))
        lines.append(f"  producer changes since: {mem.diff_summary}")
    return "\n".join(lines) + "\n\n"


def build_critique_prompt(
    model: DomainModel, scout: ScoutOutput, history: List[CritiqueCycleMemory],
) -> str:
    return (
        _INSTRUCTIONS
        + "\n\n"
        + _serialize_history(history)
        + "DOMAIN MODEL UNDER REVIEW:\n"
        + _serialize_model(model)
        + "\n\nNUMBERED SOURCE SENTENCES:\n"
        + _serialize_scout(scout)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_prompt.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add core/critic/prompt.py tests/test_critic_prompt.py
git commit -m "feat(critic): critique prompt builder (CoT + scout grounding + Reflexion)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `run_critic` evaluator (`core/critic/critic.py`)

**Files:**
- Create: `core/critic/critic.py`
- Test: `tests/test_critic_run.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_run.py
import pytest
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
from core.llm.base import LLMResponse, TokenUsage
from core.critic.types import CriticResponse, ProposedFinding
from core.critic.critic import run_critic
from core.critic.errors import CriticError
from core.schemas import (
    DomainModel, ProjectMetadata, BoundedContext, UbiquitousLanguage, Entity,
)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


@dataclass
class _StageCfg:
    model_id: str = "gemini-3.1-pro-preview"
    temperature: float = 0.05
    seed: Optional[int] = 42


class _FakeClient:
    """Returns a canned LLMResponse for structured_output."""
    def __init__(self, parsed: Optional[BaseModel], json_failed: bool = False):
        self._parsed = parsed
        self._json_failed = json_failed
        self.calls = 0

    def structured_output(self, messages, schema, model, **kwargs) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            content="{}", parsed=self._parsed,
            usage=TokenUsage(1, 1, 2), model_id=model, provider="fake",
            json_failed=self._json_failed,
            json_fail_reason="schema_mismatch" if self._json_failed else None,
        )


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0", generated_at="now"),
        bounded_contexts=[BoundedContext(
            context_name="Ordering",
            ubiquitous_language=UbiquitousLanguage(
                entities=[Entity(
                    name="Order", description="An order.", confidence=0.9,
                    justification="j", evidence_sentence_indices=[0],
                )],
                value_objects=None, domain_events=None,
            ),
        )],
        global_rules=None,
    )


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="A customer places an order.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=27),
    )


def test_run_critic_maps_valid_findings():
    resp = CriticResponse(analysis="a", findings=[ProposedFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Ordering.Order", rationale="no behavior",
        proposed_revision="add place()", evidence_sentence_indices=[0],
    )])
    report = run_critic(_model(), _scout(), [], client=_FakeClient(resp), stage_cfg=_StageCfg())
    assert report.model_id == "gemini-3.1-pro-preview"
    assert len(report.findings) == 1
    assert report.malformed_findings == 0


def test_run_critic_drops_unresolvable_target_ref():
    resp = CriticResponse(analysis="a", findings=[ProposedFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Nope.Ghost", rationale="x", proposed_revision="y",
    )])
    report = run_critic(_model(), _scout(), [], client=_FakeClient(resp), stage_cfg=_StageCfg())
    assert report.findings == []
    assert report.malformed_findings == 1


def test_run_critic_drops_out_of_range_evidence_but_keeps_finding():
    resp = CriticResponse(analysis="a", findings=[ProposedFinding(
        finding_type="ANEMIC_ENTITY", priority="high",
        target_ref="entity:Ordering.Order", rationale="x", proposed_revision="y",
        evidence_sentence_indices=[0, 99],
    )])
    report = run_critic(_model(), _scout(), [], client=_FakeClient(resp), stage_cfg=_StageCfg())
    assert report.findings[0].evidence_sentence_indices == [0]


def test_run_critic_raises_on_json_failed():
    with pytest.raises(CriticError):
        run_critic(_model(), _scout(), [], client=_FakeClient(None, json_failed=True), stage_cfg=_StageCfg())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_run.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.critic.critic'`.

- [ ] **Step 3: Implement**

```python
# core/critic/critic.py
"""run_critic: one schema-enforced LLM call that judges DDD design quality.

Pure evaluator — never mutates the model. Maps the LLM's ProposedFindings to
persisted CritiqueFindings, dropping ones whose target_ref does not resolve and
trimming out-of-range evidence indices. Raises CriticError on json_failed."""
from typing import Any, List, Optional, Set
from core.schemas import CritiqueFinding, CriticReport, CriticLoopTrace, DomainModel
from core.pipeline_contracts import ScoutOutput
from core.critic.types import CriticResponse, ProposedFinding, CritiqueCycleMemory
from core.critic.prompt import build_critique_prompt
from core.critic.errors import CriticError


def _valid_targets(model: DomainModel) -> Set[str]:
    targets: Set[str] = set()
    for bc in model.bounded_contexts:
        targets.add(f"context:{bc.context_name}")
        for e in bc.ubiquitous_language.entities:
            targets.add(f"entity:{bc.context_name}.{e.name}")
    return targets


def _map_finding(
    pf: ProposedFinding, valid_targets: Set[str], scout_indices: Set[int],
) -> Optional[CritiqueFinding]:
    if pf.target_ref not in valid_targets:
        return None
    evidence = [i for i in pf.evidence_sentence_indices if i in scout_indices]
    return CritiqueFinding(
        finding_type=pf.finding_type, priority=pf.priority,
        target_ref=pf.target_ref, rationale=pf.rationale,
        proposed_revision=pf.proposed_revision,
        evidence_sentence_indices=evidence,
    )


def run_critic(
    model: DomainModel,
    scout: ScoutOutput,
    history: List[CritiqueCycleMemory],
    *,
    client: Any,
    stage_cfg: Any,
) -> CriticReport:
    """Run one critique pass. `client` exposes structured_output(...);
    `stage_cfg` exposes .model_id/.temperature/.seed.

    Note: `loop` is a placeholder single-cycle trace; the loop driver
    (run_critique_loop) overwrites it with the real multi-cycle trace before
    persisting. score is left 0.0 here and computed by the loop driver."""
    prompt = build_critique_prompt(model, scout, history)
    response = client.structured_output(
        messages=[{"role": "user", "content": prompt}],
        schema=CriticResponse,
        model=stage_cfg.model_id,
        temperature=stage_cfg.temperature,
        seed=stage_cfg.seed,
    )
    if response.json_failed or not isinstance(response.parsed, CriticResponse):
        raise CriticError(reason=response.json_fail_reason or "empty_parse")

    parsed: CriticResponse = response.parsed
    valid_targets = _valid_targets(model)
    scout_indices = {s.index for s in scout.sentences}
    findings: List[CritiqueFinding] = []
    malformed = 0
    for pf in parsed.findings:
        mapped = _map_finding(pf, valid_targets, scout_indices)
        if mapped is None:
            malformed += 1
        else:
            findings.append(mapped)

    return CriticReport(
        model_id=stage_cfg.model_id,
        findings=findings,
        malformed_findings=malformed,
        loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_run.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add core/critic/critic.py tests/test_critic_run.py
git commit -m "feat(critic): run_critic evaluator (schema-enforced, grounded, pure)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Refactor `run_pipeline` → extract `_generate_once` + add `critic` dep

**Files:**
- Modify: `core/orchestration/pipeline.py`
- Test: `tests/test_pipeline_generate_once.py` (new) + existing `tests/test_pipeline_orchestration.py` must still pass.

**Context:** Today `run_pipeline` computes `scout` then runs the architect-rerun/specialist-refine/synthesizer body. We extract that body (everything after `scout = deps.scout(srs_text)`) into `_generate_once(scout, deps, srs_path, *, architect_feedback=None)` returning `(model, arch, refined_specialist)`. `run_pipeline` becomes a dispatcher. Behavior with `architect_feedback=None` and `critic=None` must be **byte-identical** to today.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_generate_once.py
"""_generate_once returns the model plus intermediates; critic=None unchanged."""
from core.orchestration.pipeline import run_pipeline, _generate_once, PipelineDeps
from core.pipeline_contracts import (
    ScoutOutput, ArchitectOutput, ContextHypothesis, SpecialistAnalysis,
    SectionedSentence, ChunkMetadata,
)
from core.schemas import DomainModel, Entity
from core.verifier.types import VerifierResult
from unittest.mock import MagicMock


def _deps():
    def scout_fn(t):
        return ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
        )

    def architect_fn(scout):
        return ArchitectOutput(contexts=[ContextHypothesis(context_name="Ord", description="x")])

    def specialist_fn(arch, scout):
        return [SpecialistAnalysis(
            context=arch.contexts[0],
            entities=[Entity(name="Order", description="An order.", confidence=0.9,
                             justification="c", evidence_sentence_indices=[0])],
        )]

    def synthesizer_fn(analyses):
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(analyses, llm_client=MagicMock(),
                                       project_name="T", skip_enrich=True)

    return PipelineDeps(
        scout=scout_fn, architect=architect_fn,
        architect_with_feedback=lambda s, i: architect_fn(s),
        specialist=specialist_fn, synthesizer=synthesizer_fn,
        verifier=lambda snap: VerifierResult(ok=True, issues=[]),
    )


def test_generate_once_returns_model_and_intermediates():
    deps = _deps()
    scout = deps.scout("x")
    model, arch, specialist = _generate_once(scout, deps, srs_path="x")
    assert isinstance(model, DomainModel)
    assert arch.contexts[0].context_name == "Ord"
    assert specialist[0].entities[0].name == "Order"


def test_critic_none_pipeline_unchanged():
    deps = _deps()                       # no critic field set → None
    model = run_pipeline(srs_text="x", deps=deps)
    assert model.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline_generate_once.py -q`
Expected: FAIL with `ImportError: cannot import name '_generate_once'`.

- [ ] **Step 3: Implement the refactor**

In `core/orchestration/pipeline.py`:

(a) Add `critic` to `PipelineDeps` (after `specialist_with_feedback`):

```python
    # WP — Holistic Critic: optional per-cycle critique callable.
    # When set (and DDD_CRITIC_LOOP enabled at wiring time), run_pipeline
    # dispatches to run_critique_loop. None → today's single-pass behavior.
    critic: Optional["CriticFn"] = None
```

And add the type alias near the other `*Fn` aliases:

```python
# Imported lazily inside run_pipeline to avoid a core.critic import cycle.
CriticFn = Callable[[DomainModel, ScoutOutput, list], "CriticReport"]
```

Add to the imports at top: `from core.schemas import DomainModel` is already present; add `from typing import ... ` already has needed names. (No new top-level import of CriticReport — it is only referenced in a string annotation.)

(b) Replace the current `run_pipeline` body. Rename everything from the line `architect_attempts = 0` down through the final `return model` into a new function `_generate_once`, with two changes: it takes `scout` as a parameter (not computed inside), accepts `architect_feedback`, and returns the tuple.

```python
def run_pipeline(
    *,
    srs_text: str,
    deps: PipelineDeps,
    srs_path: Optional[str] = None,
) -> DomainModel:
    """Run the pipeline. With deps.critic set, drive the bounded critique loop;
    otherwise a single generation pass (historical behavior)."""
    with _optional_stage("scout"):
        scout: ScoutOutput = deps.scout(srs_text)

    if deps.critic is None:
        model, _arch, _specialist = _generate_once(scout, deps, srs_path)
        return model

    from core.critic.loop import run_critique_loop
    return run_critique_loop(scout, deps, srs_path)


def _generate_once(
    scout: ScoutOutput,
    deps: PipelineDeps,
    srs_path: Optional[str],
    *,
    architect_feedback: Optional[List[Any]] = None,
) -> tuple[DomainModel, ArchitectOutput, List[SpecialistAnalysis]]:
    """One full generation pass: Architect (rerun loop) → Specialist (refine)
    → Synthesizer. Returns the model plus the final ArchitectOutput and
    refined SpecialistAnalysis list (the critique loop reuses these for the
    content-only path). `architect_feedback`, when provided, seeds the FIRST
    architect call (critic-driven structural revision)."""
    architect_attempts = 0
    architect_max_cycles = 1
    # Critic seed → first architect call uses architect_with_feedback. The
    # internal verifier-driven rerun budget (architect_attempts) is separate.
    architect_feedback_local: Optional[List[Any]] = architect_feedback
    refined_specialist: Optional[List[SpecialistAnalysis]] = None
    arch: Optional[ArchitectOutput] = None

    while True:
        with _optional_stage("architect", extend=(architect_attempts > 0)):
            if architect_feedback_local is None:
                arch = deps.architect(scout)
            else:
                arch = deps.architect_with_feedback(scout, architect_feedback_local)
        # ... (unchanged specialist call, verifier pre-check, refine loop, and
        #      the architect-rerun continue/raise logic from the current body,
        #      but every assignment to `architect_feedback` becomes
        #      `architect_feedback_local`) ...
        break

    if not refined_specialist:
        raise SynthesizerEmptyModelError(
            input_summary="0 SpecialistAnalysis from upstream pipeline",
            srs_path=srs_path or "<unknown>",
        )
    with _optional_stage("synthesizer"):
        model: DomainModel = deps.synthesizer(refined_specialist)
    if not model.bounded_contexts:
        raise SynthesizerEmptyModelError(
            input_summary="synthesizer returned 0 bounded contexts (bypassed Pydantic)",
            srs_path=srs_path or "<unknown>",
        )
    return model, arch, refined_specialist
```

> **Mechanical detail for the implementer:** copy the existing `run_pipeline` body from the current `architect_attempts = 0` line through the `return model` line into `_generate_once`. Apply exactly these edits: (1) delete the local `architect_feedback: Optional[List[Any]] = None` initializer (replaced by the `architect_feedback_local = architect_feedback` parameter seed above); (2) rename every remaining `architect_feedback` reference to `architect_feedback_local`; (3) change `_optional_stage("architect", extend=(architect_feedback is not None))` to `extend=(architect_attempts > 0)`; (4) change the final `return model` to `return model, arch, refined_specialist`. Do not alter the specialist refine loop, the `RefinementExhaustedError` handling, the `_record_refiner_metrics_safely` calls, or the architect rerun `continue`/`raise` logic.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pipeline_generate_once.py tests/test_pipeline_orchestration.py tests/test_pipeline_observability_e2e.py tests/test_stage_extend_mode.py -q`
Expected: PASS (all — existing orchestration tests are the regression guard that the extract is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add core/orchestration/pipeline.py tests/test_pipeline_generate_once.py
git commit -m "refactor(pipeline): extract _generate_once + add optional critic dep

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: The critique loop (`core/critic/loop.py`)

**Files:**
- Create: `core/critic/loop.py`
- Test: `tests/test_critic_loop.py`

**Behavior:** cycle 0 generates + critiques; subsequent cycles regenerate from routed feedback, re-critique, keep-best; stop on converged / flap / exhausted / failed. Returns the best model with `critic_report` attached.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_loop.py
import os
from unittest.mock import MagicMock
from core.critic.loop import run_critique_loop, critique_score, findings_signature
from core.orchestration.pipeline import PipelineDeps
from core.pipeline_contracts import (
    ScoutOutput, ArchitectOutput, ContextHypothesis, SpecialistAnalysis,
    SectionedSentence, ChunkMetadata,
)
from core.schemas import DomainModel, Entity, CritiqueFinding, CriticReport, CriticLoopTrace
from core.verifier.types import VerifierResult


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="An order.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8),
    )


def _base_deps():
    def architect_fn(scout):
        return ArchitectOutput(contexts=[ContextHypothesis(context_name="Ord", description="x")])

    def specialist_fn(arch, scout):
        return [SpecialistAnalysis(
            context=arch.contexts[0],
            entities=[Entity(name="Order", description="An order.", confidence=0.9,
                             justification="c", evidence_sentence_indices=[0])],
        )]

    def synthesizer_fn(analyses):
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(analyses, llm_client=MagicMock(),
                                       project_name="T", skip_enrich=True)

    return PipelineDeps(
        scout=lambda t: _scout(), architect=architect_fn,
        architect_with_feedback=lambda s, i: architect_fn(s),
        specialist=specialist_fn,
        specialist_with_feedback=lambda a, s, prev, issues: specialist_fn(a, s),
        synthesizer=synthesizer_fn,
        verifier=lambda snap: VerifierResult(ok=True, issues=[]),
    )


def _high_finding():
    return CritiqueFinding(finding_type="ANEMIC_ENTITY", priority="high",
                           target_ref="entity:Ord.Order", rationale="r", proposed_revision="p")


def _report(findings, score=0.0):
    return CriticReport(model_id="m", findings=findings, score=score,
                        loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))


def test_score_orders_by_severity():
    assert critique_score([_high_finding()]) == 3.0
    assert critique_score([]) == 0.0


def test_converged_when_cycle0_clean(monkeypatch):
    deps = _base_deps()
    deps.critic = lambda model, scout, history: _report([])      # no findings
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.outcome == "converged"
    assert model.critic_report.loop.cycles_used == 1


def test_keep_best_returns_lowest_score_cycle():
    deps = _base_deps()
    calls = [0]

    def critic(model, scout, history):
        calls[0] += 1
        # cycle 0: one high finding (score 3); cycle 1: two highs (score 6 → worse)
        return _report([_high_finding()]) if calls[0] == 1 else _report([_high_finding(), _high_finding()])

    deps.critic = critic
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.best_cycle == 0
    assert model.critic_report.score == 3.0


def test_flap_stops_loop():
    deps = _base_deps()

    def critic(model, scout, history):
        return _report([_high_finding()])     # identical signature every cycle

    deps.critic = critic
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.outcome == "flapped"


def test_failure_is_non_fatal_returns_best_so_far():
    from core.critic.errors import CriticError
    deps = _base_deps()
    calls = [0]

    def critic(model, scout, history):
        calls[0] += 1
        if calls[0] == 1:
            return _report([_high_finding()])
        raise CriticError(reason="boom", cycle=calls[0])

    deps.critic = critic
    model = run_critique_loop(_scout(), deps, srs_path="x")
    assert model.critic_report.loop.outcome == "failed"
    assert model.critic_report.error is not None
    assert isinstance(model, DomainModel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_loop.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.critic.loop'`.

- [ ] **Step 3: Implement**

```python
# core/critic/loop.py
"""Bounded critique→revise loop (Topology A).

Cycle 0 generates + critiques. Each subsequent cycle routes the prior cycle's
high/medium findings back to the producers, regenerates, and re-critiques.
Returns the best (lowest-score) model with its CriticReport attached.

Stops on: converged (no high/med), flapped (repeated finding signature),
exhausted (cycle cap), or failed (CriticError / regeneration error → non-fatal,
returns best-so-far)."""
import os
from typing import Any, List, Optional, Tuple
from core.schemas import CritiqueFinding, CriticReport, CriticLoopTrace, DomainModel
from core.pipeline_contracts import ScoutOutput, ArchitectOutput, SpecialistAnalysis
from core.critic.errors import CriticError
from core.critic.types import CritiqueCycleMemory
from core.critic.routing import (
    partition_findings, adapt_structural_to_issues, adapt_content_to_issues,
    model_diff_summary,
)

_PRIORITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}


def critique_score(findings: List[CritiqueFinding]) -> float:
    return sum(_PRIORITY_WEIGHT[f.priority] for f in findings)


def findings_signature(findings: List[CritiqueFinding]) -> Tuple:
    return tuple(sorted(
        (f.finding_type, f.target_ref)
        for f in findings if f.priority in ("high", "medium")
    ))


def _has_high_or_medium(report: CriticReport) -> bool:
    return any(f.priority in ("high", "medium") for f in report.findings)


def _max_cycles() -> int:
    try:
        return max(1, int(os.getenv("DDD_CRITIC_MAX_CYCLES", "3")))
    except ValueError:
        return 3


def _findings_summary(report: CriticReport) -> List[str]:
    return [f"{f.priority} {f.finding_type} {f.target_ref}" for f in report.findings]


def run_critique_loop(
    scout: ScoutOutput, deps: Any, srs_path: Optional[str],
) -> DomainModel:
    from core.orchestration.pipeline import _generate_once

    max_cycles = _max_cycles()
    history: List[CritiqueCycleMemory] = []
    score_trace: List[float] = []
    count_trace: List[int] = []

    # --- cycle 0 -----------------------------------------------------------
    model, arch, specialist = _generate_once(scout, deps, srs_path)
    try:
        report = deps.critic(model, scout, history)
    except CriticError as exc:
        return _finalize_failed(model, exc, cycles_used=1,
                                score_trace=[], count_trace=[])

    best_model, best_report, best_cycle = model, report, 0
    score_trace.append(critique_score(report.findings))
    count_trace.append(len(report.findings))
    history.append(CritiqueCycleMemory(
        cycle=0, findings_summary=_findings_summary(report),
        diff_summary="initial model",
    ))

    outcome = "converged"
    prev_signature = findings_signature(report.findings)

    # --- revision cycles ---------------------------------------------------
    for cycle in range(1, max_cycles):
        if not _has_high_or_medium(report):
            outcome = "converged"
            break
        structural, content, _advisory = partition_findings(report.findings)
        try:
            if structural:
                new_model, arch, specialist = _generate_once(
                    scout, deps, srs_path,
                    architect_feedback=adapt_structural_to_issues(structural),
                )
            else:  # content-only → reuse architecture, targeted specialist rerun
                specialist = deps.specialist_with_feedback(
                    arch, scout, specialist, adapt_content_to_issues(content),
                )
                new_model = deps.synthesizer(specialist)
            new_report = deps.critic(new_model, scout, history)
        except CriticError as exc:
            return _finalize_failed(best_model, exc, cycles_used=cycle + 1,
                                    score_trace=score_trace, count_trace=count_trace,
                                    best_report=best_report, best_cycle=best_cycle)

        score_trace.append(critique_score(new_report.findings))
        count_trace.append(len(new_report.findings))
        history.append(CritiqueCycleMemory(
            cycle=cycle, findings_summary=_findings_summary(new_report),
            diff_summary=model_diff_summary(model, new_model),
        ))

        if critique_score(new_report.findings) < critique_score(best_report.findings):
            best_model, best_report, best_cycle = new_model, new_report, cycle

        sig = findings_signature(new_report.findings)
        if sig == prev_signature:
            outcome = "flapped"
            model, report = new_model, new_report
            break
        prev_signature = sig
        model, report = new_model, new_report
    else:
        outcome = "exhausted" if _has_high_or_medium(report) else "converged"

    best_report.score = critique_score(best_report.findings)
    best_report.loop = CriticLoopTrace(
        cycles_used=len(score_trace), best_cycle=best_cycle, outcome=outcome,
        score_per_cycle=score_trace, findings_count_per_cycle=count_trace,
    )
    best_model.critic_report = best_report
    return best_model


def _finalize_failed(
    model: DomainModel, exc: CriticError, *, cycles_used: int,
    score_trace: List[float], count_trace: List[int],
    best_report: Optional[CriticReport] = None, best_cycle: int = 0,
) -> DomainModel:
    report = best_report or CriticReport(
        model_id="unknown", findings=[],
        loop=CriticLoopTrace(cycles_used=cycles_used, best_cycle=best_cycle, outcome="failed"),
    )
    report.score = critique_score(report.findings)
    report.error = str(exc)
    report.loop = CriticLoopTrace(
        cycles_used=cycles_used, best_cycle=best_cycle, outcome="failed",
        score_per_cycle=score_trace, findings_count_per_cycle=count_trace,
    )
    model.critic_report = report
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_loop.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add core/critic/loop.py tests/test_critic_loop.py
git commit -m "feat(critic): bounded critique->revise loop (keep-best, flap, Reflexion, non-fatal)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Register the Critic stage in `configs/models.py`

**Files:**
- Modify: `configs/models.py`
- Test: `tests/test_critic_stage_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_stage_config.py
from configs.models import stage_config


def test_critic_stage_uses_generation_group():
    arch = stage_config("Architect")
    critic = stage_config("Critic")
    assert critic.model_id == arch.model_id          # same generation model (G1)
    assert critic.temperature == arch.temperature
    assert critic.seed == arch.seed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_stage_config.py -q`
Expected: FAIL with `KeyError: 'Critic'`.

- [ ] **Step 3: Implement**

In `configs/models.py`, add the `Critic` entry to `STAGE_TO_GROUP`:

```python
STAGE_TO_GROUP: Dict[str, str] = {
    "Scout":       "domain_extraction",
    "Architect":   "domain_extraction",
    "Specialist":  "domain_extraction",
    "Synthesizer": "domain_extraction",
    "Critic":      "domain_extraction",
    "Validator":   "validation",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_stage_config.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add configs/models.py tests/test_critic_stage_config.py
git commit -m "feat(critic): register Critic stage in STAGE_TO_GROUP (generation group)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Wire `critic_fn` into `analyze_document` (gated on `DDD_CRITIC_LOOP`)

**Files:**
- Modify: `core/architect.py` (`analyze_document`, near the `deps = PipelineDeps(...)` block ~1181)
- Test: `tests/test_critic_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_wiring.py
import os
import pytest
from core.architect import DomainArchitect


@pytest.fixture(autouse=True)
def _gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_build_critic_fn_none_when_flag_off(monkeypatch):
    monkeypatch.delenv("DDD_CRITIC_LOOP", raising=False)
    arch = DomainArchitect()
    assert arch._build_critic_fn() is None


def test_build_critic_fn_present_when_flag_on(monkeypatch):
    monkeypatch.setenv("DDD_CRITIC_LOOP", "1")
    arch = DomainArchitect()
    fn = arch._build_critic_fn()
    assert callable(fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_critic_wiring.py -q`
Expected: FAIL with `AttributeError: 'DomainArchitect' object has no attribute '_build_critic_fn'`.

- [ ] **Step 3: Implement**

In `core/architect.py`, add a method on `DomainArchitect` (near `analyze_document`):

```python
    def _build_critic_fn(self):
        """Return a per-cycle critic callable for the critique loop, or None
        when DDD_CRITIC_LOOP is not enabled. Uses the generation client +
        the Critic stage config (G1)."""
        if os.getenv("DDD_CRITIC_LOOP", "") not in ("1", "true", "True"):
            return None

        from core.critic.critic import run_critic
        critic_cfg = stage_config("Critic")

        def critic_fn(model, scout, history):
            return run_critic(
                model, scout, history,
                client=self.client, stage_cfg=critic_cfg,
            )

        return critic_fn
```

Then in `analyze_document`, pass it into `PipelineDeps` (add the kwarg to the existing `deps = PipelineDeps(...)` construction):

```python
        deps = PipelineDeps(
            scout=scout_fn,
            architect=architect_fn,
            architect_with_feedback=architect_with_feedback_fn,
            specialist=specialist_fn,
            synthesizer=synthesizer_fn,
            verifier=verifier_fn,
            specialist_with_feedback=specialist_with_feedback_fn,
            critic=self._build_critic_fn(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_critic_wiring.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add core/architect.py tests/test_critic_wiring.py
git commit -m "feat(critic): wire critic_fn into analyze_document behind DDD_CRITIC_LOOP

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: End-to-end loop integration test (mocked client)

**Files:**
- Test: `tests/test_critic_integration.py`

**Goal:** prove the whole loop runs through `run_pipeline` when `critic` is set, attaching a `critic_report`, with no live API.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_integration.py
"""Full pipeline → critique loop integration with a fake critic dep."""
from unittest.mock import MagicMock
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.pipeline_contracts import (
    ScoutOutput, ArchitectOutput, ContextHypothesis, SpecialistAnalysis,
    SectionedSentence, ChunkMetadata,
)
from core.schemas import (
    DomainModel, Entity, CritiqueFinding, CriticReport, CriticLoopTrace,
)
from core.verifier.types import VerifierResult


def _deps_with_critic(critic):
    def architect_fn(scout):
        return ArchitectOutput(contexts=[ContextHypothesis(context_name="Ord", description="x")])

    def specialist_fn(arch, scout):
        return [SpecialistAnalysis(
            context=arch.contexts[0],
            entities=[Entity(name="Order", description="An order.", confidence=0.9,
                             justification="c", evidence_sentence_indices=[0])],
        )]

    def synthesizer_fn(analyses):
        from core.synthesizer import synthesize_domain_model
        return synthesize_domain_model(analyses, llm_client=MagicMock(),
                                       project_name="T", skip_enrich=True)

    return PipelineDeps(
        scout=lambda t: ScoutOutput(
            sentences=[SectionedSentence(index=0, text="An order.")],
            chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=8)),
        architect=architect_fn,
        architect_with_feedback=lambda s, i: architect_fn(s),
        specialist=specialist_fn,
        specialist_with_feedback=lambda a, s, prev, issues: specialist_fn(a, s),
        synthesizer=synthesizer_fn,
        verifier=lambda snap: VerifierResult(ok=True, issues=[]),
        critic=critic,
    )


def test_pipeline_with_clean_critic_attaches_converged_report():
    def critic(model, scout, history):
        return CriticReport(model_id="m", findings=[],
                            loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))

    model = run_pipeline(srs_text="x", deps=_deps_with_critic(critic))
    assert isinstance(model, DomainModel)
    assert model.critic_report is not None
    assert model.critic_report.loop.outcome == "converged"
    # Model content is unchanged by the (clean) critic — pure evaluator.
    assert model.bounded_contexts[0].ubiquitous_language.entities[0].name == "Order"


def test_pipeline_content_finding_drives_one_revision_then_converges():
    calls = [0]

    def critic(model, scout, history):
        calls[0] += 1
        if calls[0] == 1:
            return CriticReport(model_id="m", findings=[CritiqueFinding(
                finding_type="ANEMIC_ENTITY", priority="high",
                target_ref="entity:Ord.Order", rationale="r", proposed_revision="p")],
                loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))
        return CriticReport(model_id="m", findings=[],
                            loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged"))

    model = run_pipeline(srs_text="x", deps=_deps_with_critic(critic))
    assert calls[0] == 2                                   # cycle0 + 1 revision
    assert model.critic_report.loop.outcome == "converged"
    assert model.critic_report.loop.best_cycle == 1
```

- [ ] **Step 2: Run test to verify it fails (then passes)**

Run: `pytest tests/test_critic_integration.py -q`
Expected: PASS if Tasks 7–8 are correct. (If it FAILs, fix the loop/dispatch wiring before continuing — this is the acceptance gate for the feature.)

- [ ] **Step 3: Full gate**

Run: `pytest -m "not integration" -q`
Expected: PASS (entire unit suite, including the pre-existing orchestration regression tests).

Run: `pyright`
Expected: 0 production errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_critic_integration.py
git commit -m "test(critic): end-to-end critique-loop integration through run_pipeline

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Observability (spec §11) is deliberately deferred** to keep tasks bite-sized: `run_critic` already returns token usage via the LLM client; per-cycle `StageEmitter` records + manifest `CriticLoopTrace` persistence can be a follow-up task once the loop is proven. If you want it in-scope, add a Task 8b wrapping the per-cycle critic call in `_optional_stage("critic")` and writing the trace to the manifest — mirror `_record_refiner_metrics_safely`.
- **Intermediate dumps (spec §7.7)** likewise are a follow-up; not required for correctness.
- **Content-path inner refine:** for simplicity the content-only path re-synthesizes without re-running `refine_until_clean`; the synthesizer's D6/D7/D8 invariants still apply, and the next critic cycle catches residual issues. If grounding drift appears in practice, add `refine_until_clean` to the content path (spec §5).
- Keep `DDD_CRITIC_LOOP` **OFF** by default; validate on a real SRS before flipping.
```
