# Context-Mapper (A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Context-Mapper agent that produces a typed DDD strategic context map (`DomainModel.context_map`), derives `allowed_dependencies` from it, and participates in the Critic loop as a Critic-driven producer.

**Architecture:** New `core/context_mapper/` package (LLM call + pure derivation), wired into `_generate_once` (so it runs every generation pass) as a pure deep-copying step, plus Critic relationship-awareness (3 new finding types routed back to A's feedback re-map). AST import-graph becomes diagnostics-only when A's map is authoritative.

**Tech Stack:** Python 3.12/3.13, Pydantic v2, pytest. LLM via `core.llm` `structured_output`. Mirrors the shipped `core/critic/` package.

**Spec:** `docs/superpowers/specs/2026-05-25-context-mapper-design.md`

**Run gate everywhere:** from `extension/backend/`, use `.venv/bin/python -m pytest` (or bare `pytest`). `python3` is Homebrew 3.14 without pytest — do NOT use it. Pyright: `pyright` (0 prod errors required; `tests/` excluded from the gate).

**Model tiers for SDD dispatch:** opus = Tasks 2, 5, 7, 12, 13 (correctness-critical); sonnet = Tasks 1, 4, 8, 9, 10, 11, 14; sonnet/haiku = Tasks 3, 6.

---

## File Structure

**Create:**
- `core/context_mapper/__init__.py` — facade exports
- `core/context_mapper/errors.py` — `ContextMapperError`
- `core/context_mapper/types.py` — LLM-facing `ProposedRelationship`, `ContextMapResponse`
- `core/context_mapper/derive.py` — pure `derive_allowed_dependencies`
- `core/context_mapper/prompt.py` — `build_map_prompt`, `build_remap_prompt`
- `core/context_mapper/mapper.py` — `run_context_mapper`
- `tests/test_context_mapper_schema.py`, `tests/test_context_mapper_derive.py`, `tests/test_context_mapper_mapper.py`, `tests/test_context_mapper_pipeline.py`, `tests/test_critic_relationship.py`, `tests/test_import_graph_context_map.py`, `tests/test_context_mapper_e2e.py`

**Modify:**
- `core/schemas.py` — `ContextRelationship`, `ContextMap`, `DomainModel.context_map`; `CritiqueFinding.finding_type` += 3
- `core/critic/types.py` — `ProposedFinding.finding_type` += 3
- `core/critic/prompt.py` — serialize `context_map` + relationship instructions
- `core/critic/critic.py` — `_map_finding` relationship handling
- `core/critic/routing.py` — `_RELATIONSHIP`, `partition_findings` 4-tuple, `model_diff_summary` deltas
- `core/critic/loop.py` — relationship branch, every-cycle feedback, signature canonicalization
- `core/orchestration/pipeline.py` — `ContextMapperFn`, `PipelineDeps.context_mapper`, `_apply_context_map`, call in `_generate_once`
- `core/architect.py` — `_build_context_mapper_fn`, wire into `PipelineDeps`
- `configs/models.py` — `STAGE_TO_GROUP["ContextMapper"]`
- `core/AST/import_graph.py` — diagnostics-only when `context_map` authoritative

---

## Task 1: Schema — ContextRelationship + ContextMap + DomainModel.context_map  [sonnet]

**Files:**
- Modify: `core/schemas.py` (add classes before `DomainModel`; add field to `DomainModel`)
- Test: `tests/test_context_mapper_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_mapper_schema.py
import pytest
from pydantic import ValidationError
from core.schemas import ContextRelationship, ContextMap, DomainModel


def _rel(**kw):
    base = dict(context_a="Ordering", context_b="Inventory",
                relationship_type="CUSTOMER_SUPPLIER", upstream="Inventory",
                rationale="Ordering consumes stock levels from Inventory.")
    base.update(kw)
    return ContextRelationship(**base)


def test_directional_requires_upstream_member():
    r = _rel()
    assert r.upstream == "Inventory"
    with pytest.raises(ValidationError):
        _rel(upstream=None)
    with pytest.raises(ValidationError):
        _rel(upstream="Nonexistent")


def test_mutual_rejects_upstream():
    r = _rel(relationship_type="PARTNERSHIP", upstream=None)
    assert r.relationship_type == "PARTNERSHIP"
    with pytest.raises(ValidationError):
        _rel(relationship_type="PARTNERSHIP", upstream="Ordering")


def test_separate_ways_and_bbom_reject_upstream():
    assert _rel(relationship_type="SEPARATE_WAYS", upstream=None)
    assert _rel(relationship_type="BIG_BALL_OF_MUD", upstream=None)
    with pytest.raises(ValidationError):
        _rel(relationship_type="SEPARATE_WAYS", upstream="Ordering")


def test_distinct_contexts_required():
    with pytest.raises(ValidationError):
        _rel(context_a="X", context_b="X", relationship_type="PARTNERSHIP", upstream=None)


def test_context_map_defaults_and_domain_field():
    cm = ContextMap(model_id="gemini-3.1-pro-preview")
    assert cm.relationships == [] and cm.warnings == [] and cm.error is None


def test_domain_model_context_map_optional_backward_compat():
    # A model built WITHOUT context_map must still validate (existing model.json).
    from core.schemas import BoundedContext, UbiquitousLanguage, ProjectMetadata
    m = DomainModel(
        project_name="P",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[BoundedContext(context_name="Ordering",
                          ubiquitous_language=UbiquitousLanguage())],
        global_rules=None,
    )
    assert m.context_map is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_schema.py -q`
Expected: FAIL (ImportError: cannot import name 'ContextRelationship').

- [ ] **Step 3: Implement schema in `core/schemas.py`**

Add `model_validator` to the pydantic import line if not present (`from pydantic import ..., model_validator`). Insert these classes immediately BEFORE `class DomainModel` (the `CritiqueFinding`/`CriticReport` block is a good neighbor):

```python
class ContextRelationship(BaseModel):
    """One typed DDD strategic relationship between two bounded contexts."""
    context_a: str = Field(description="First context_name in the pair.")
    context_b: str = Field(description="Second context_name in the pair.")
    relationship_type: Literal[
        "PARTNERSHIP", "SHARED_KERNEL", "CUSTOMER_SUPPLIER", "CONFORMIST",
        "ANTI_CORRUPTION_LAYER", "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE",
        "SEPARATE_WAYS", "BIG_BALL_OF_MUD",
    ]
    upstream: Optional[str] = Field(
        default=None,
        description="context_a or context_b — the upstream/supplier side. None "
                    "for mutual (PARTNERSHIP, SHARED_KERNEL) and non-integration "
                    "(SEPARATE_WAYS, BIG_BALL_OF_MUD) types.",
    )
    rationale: str = Field(description="Why this pattern + direction, in DDD terms.")
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices grounding this relationship; [-1] if "
                    "inference-only with no single supporting sentence.",
    )

    _DIRECTIONAL = {"CUSTOMER_SUPPLIER", "CONFORMIST", "ANTI_CORRUPTION_LAYER",
                    "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE"}

    @model_validator(mode="after")
    def _check_upstream_consistency(self) -> "ContextRelationship":
        if self.context_a == self.context_b:
            raise ValueError("context_a and context_b must differ")
        if self.relationship_type in self._DIRECTIONAL:
            if self.upstream not in (self.context_a, self.context_b):
                raise ValueError(
                    f"{self.relationship_type} requires upstream to be one of "
                    f"context_a/context_b; got {self.upstream!r}")
        elif self.upstream is not None:
            raise ValueError(
                f"{self.relationship_type} is non-directional; upstream must be None")
        return self


class ContextMap(BaseModel):
    """Strategic context map produced by the Context-Mapper (A)."""
    relationships: List["ContextRelationship"] = Field(default_factory=list)
    model_id: str
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
```

Add the field to `DomainModel` (after `critic_report`):

```python
    context_map: Optional["ContextMap"] = Field(
        default=None,
        description="Strategic DDD context map (Context-Mapper output). None when "
                    "DDD_CONTEXT_MAP is disabled or no map was produced.",
    )
```

(`Literal`, `Optional`, `List`, `Field`, `BaseModel` are already imported in this file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_schema.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Pyright + commit**

Run: `pyright core/schemas.py` → 0 errors.
```bash
git add core/schemas.py tests/test_context_mapper_schema.py
git commit -m "feat(context-mapper): ContextRelationship + ContextMap schema + DomainModel.context_map"
```

---

## Task 2: derive.py — pure allowed_dependencies derivation  [opus]

**Files:**
- Create: `core/context_mapper/derive.py`
- Create: `core/context_mapper/__init__.py` (minimal, expanded in Task 6)
- Test: `tests/test_context_mapper_derive.py`

**Contract:** `derive_allowed_dependencies(cmap: ContextMap, valid_names: set[str]) -> tuple[dict[str, list[str]], list[str]]`. Returns (deps-by-context, warnings). Directional → downstream depends on upstream. Mutual → both depend on each other. Separate Ways / BBoM → no edges. Relationships referencing unknown context names are dropped (warning each). A directional cycle (mutual edges excluded) is reported as a warning; edges are kept.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_mapper_derive.py
from core.schemas import ContextRelationship, ContextMap
from core.context_mapper.derive import derive_allowed_dependencies

NAMES = {"Ordering", "Inventory", "Billing", "Shipping"}

def _cm(*rels):
    return ContextMap(model_id="m", relationships=list(rels))

def _r(a, b, t, up=None):
    return ContextRelationship(context_a=a, context_b=b, relationship_type=t,
                               upstream=up, rationale="x")

def test_directional_downstream_depends_on_upstream():
    deps, warns = derive_allowed_dependencies(
        _cm(_r("Ordering", "Inventory", "CUSTOMER_SUPPLIER", up="Inventory")), NAMES)
    assert deps["Ordering"] == ["Inventory"]
    assert deps.get("Inventory", []) == []
    assert warns == []

def test_acl_conformist_ohs_pl_are_directional():
    for t in ("CONFORMIST", "ANTI_CORRUPTION_LAYER", "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE"):
        deps, _ = derive_allowed_dependencies(
            _cm(_r("Ordering", "Inventory", t, up="Inventory")), NAMES)
        assert deps["Ordering"] == ["Inventory"], t

def test_mutual_both_directions():
    for t in ("PARTNERSHIP", "SHARED_KERNEL"):
        deps, warns = derive_allowed_dependencies(
            _cm(_r("Ordering", "Billing", t)), NAMES)
        assert deps["Ordering"] == ["Billing"] and deps["Billing"] == ["Ordering"], t
        assert warns == []  # mutual 2-cycle is NOT flagged

def test_separate_ways_and_bbom_no_edges():
    for t in ("SEPARATE_WAYS", "BIG_BALL_OF_MUD"):
        deps, warns = derive_allowed_dependencies(
            _cm(_r("Ordering", "Shipping", t)), NAMES)
        assert deps.get("Ordering", []) == [] and deps.get("Shipping", []) == []

def test_unknown_context_dropped_with_warning():
    deps, warns = derive_allowed_dependencies(
        _cm(_r("Ordering", "Ghost", "CUSTOMER_SUPPLIER", up="Ghost")), NAMES)
    assert deps.get("Ordering", []) == []
    assert any("Ghost" in w for w in warns)

def test_directional_cycle_warns_but_keeps_edges():
    deps, warns = derive_allowed_dependencies(_cm(
        _r("Ordering", "Inventory", "CUSTOMER_SUPPLIER", up="Inventory"),
        _r("Inventory", "Billing", "CUSTOMER_SUPPLIER", up="Billing"),
        _r("Billing", "Ordering", "CUSTOMER_SUPPLIER", up="Ordering"),
    ), NAMES)
    # edges kept: Ordering->Inventory, Inventory->Billing, Billing->Ordering
    assert deps["Ordering"] == ["Inventory"]
    assert deps["Inventory"] == ["Billing"]
    assert deps["Billing"] == ["Ordering"]
    assert any("cycle" in w.lower() for w in warns)

def test_mutual_pair_excluded_from_cycle_detection():
    # Partnership 2-cycle must NOT be reported as a cycle.
    deps, warns = derive_allowed_dependencies(
        _cm(_r("Ordering", "Billing", "PARTNERSHIP")), NAMES)
    assert not any("cycle" in w.lower() for w in warns)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_derive.py -q`
Expected: FAIL (ModuleNotFoundError: core.context_mapper.derive).

- [ ] **Step 3: Implement `core/context_mapper/derive.py`**

```python
"""Pure derivation of flat allowed_dependencies from a typed ContextMap.

Directional relationships add a downstream→upstream edge; mutual ones add
both directions; non-integration ones add nothing. Relationships referencing
unknown contexts are dropped. Non-mutual cycles are reported (warning) but
edges are kept — production does not hard-gate on D11 (it is unwired), so the
correct treatment is loud-but-non-fatal.
"""
from typing import Dict, List, Set, Tuple
from core.schemas import ContextMap

_DIRECTIONAL = {"CUSTOMER_SUPPLIER", "CONFORMIST", "ANTI_CORRUPTION_LAYER",
                "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE"}
_MUTUAL = {"PARTNERSHIP", "SHARED_KERNEL"}


def derive_allowed_dependencies(
    cmap: ContextMap, valid_names: Set[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    warnings: List[str] = []
    deps: Dict[str, Set[str]] = {}
    mutual_edges: Set[Tuple[str, str]] = set()

    for r in cmap.relationships:
        if r.context_a not in valid_names or r.context_b not in valid_names:
            warnings.append(
                f"dropped relationship {r.context_a}/{r.context_b} "
                f"({r.relationship_type}): unknown context name")
            continue
        if r.relationship_type in _DIRECTIONAL:
            upstream = r.upstream
            downstream = r.context_b if upstream == r.context_a else r.context_a
            deps.setdefault(downstream, set()).add(upstream)
        elif r.relationship_type in _MUTUAL:
            deps.setdefault(r.context_a, set()).add(r.context_b)
            deps.setdefault(r.context_b, set()).add(r.context_a)
            mutual_edges.add((r.context_a, r.context_b))
            mutual_edges.add((r.context_b, r.context_a))
        # SEPARATE_WAYS / BIG_BALL_OF_MUD → no edges

    cycle = _find_cycle({k: v for k, v in deps.items()}, mutual_edges)
    if cycle:
        warnings.append(f"non-mutual dependency cycle: {' -> '.join(cycle)}")

    return {k: sorted(v) for k, v in deps.items()}, warnings


def _find_cycle(
    graph: Dict[str, Set[str]], mutual_edges: Set[Tuple[str, str]],
) -> List[str]:
    """DFS (WHITE/GRAY/BLACK) cycle detection, EXCLUDING mutual edges (which are
    intentional 2-cycles). Returns the first cycle path found, or []."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in graph}
    parent: Dict[str, str] = {}

    def neighbors(n: str) -> List[str]:
        return [m for m in graph.get(n, ()) if (n, m) not in mutual_edges]

    def visit(n: str) -> List[str]:
        color[n] = GRAY
        for m in neighbors(n):
            if color.get(m, WHITE) == GRAY:
                path = [m, n]
                cur = parent.get(n)
                while cur is not None and cur != m:
                    path.append(cur)
                    cur = parent.get(cur)
                path.append(m)
                return list(reversed(path))
            if color.get(m, WHITE) == WHITE:
                parent[m] = n
                found = visit(m)
                if found:
                    return found
        color[n] = BLACK
        return []

    for node in list(graph):
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return []
```

Create `core/context_mapper/__init__.py`:

```python
"""Context-Mapper (A): strategic-DDD context map producer + pure derivation."""
from core.context_mapper.derive import derive_allowed_dependencies

__all__ = ["derive_allowed_dependencies"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_derive.py -q`
Expected: PASS (7 tests). If the cycle-path assertion is brittle, assert only the warning substring `"cycle"` and the kept edges (already done).

- [ ] **Step 5: Pyright + commit**

Run: `pyright core/context_mapper/derive.py` → 0 errors.
```bash
git add core/context_mapper/ tests/test_context_mapper_derive.py
git commit -m "feat(context-mapper): pure allowed_dependencies derivation (mutual-exempt cycle check)"
```

---

## Task 3: types.py + errors.py — LLM-facing schema  [haiku/sonnet]

**Files:**
- Create: `core/context_mapper/types.py`, `core/context_mapper/errors.py`
- Test: `tests/test_context_mapper_mapper.py` (start the file; Task 5 extends it)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_mapper_mapper.py
from core.context_mapper.types import ProposedRelationship, ContextMapResponse
from core.context_mapper.errors import ContextMapperError

def test_proposed_relationship_minimal():
    pr = ProposedRelationship(context_a="A", context_b="B",
                              relationship_type="SEPARATE_WAYS", rationale="r")
    assert pr.upstream is None and pr.evidence_sentence_indices == []

def test_response_defaults():
    resp = ContextMapResponse()
    assert resp.analysis == "" and resp.relationships == []

def test_error_is_exception():
    e = ContextMapperError(reason="json_failed")
    assert "json_failed" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_mapper.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement**

`core/context_mapper/errors.py`:
```python
"""Context-Mapper errors."""


class ContextMapperError(Exception):
    """Raised when the mapping LLM call fails (json_failed) after retries."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"context-mapper failed: {reason}")
```

`core/context_mapper/types.py`:
```python
"""LLM-facing context-mapper schema. ProposedRelationship is what the LLM emits;
mapper.run_context_mapper maps it to core.schemas.ContextRelationship."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ProposedRelationship(BaseModel):
    context_a: str
    context_b: str
    relationship_type: Literal[
        "PARTNERSHIP", "SHARED_KERNEL", "CUSTOMER_SUPPLIER", "CONFORMIST",
        "ANTI_CORRUPTION_LAYER", "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE",
        "SEPARATE_WAYS", "BIG_BALL_OF_MUD",
    ]
    upstream: Optional[str] = None
    rationale: str = ""
    evidence_sentence_indices: List[int] = Field(default_factory=list)


class ContextMapResponse(BaseModel):
    """Schema-enforced mapper output. `analysis` is the CoT scratchpad."""
    analysis: str = Field(default="", description="Step-by-step DDD reasoning.")
    relationships: List[ProposedRelationship] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_mapper.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add core/context_mapper/types.py core/context_mapper/errors.py tests/test_context_mapper_mapper.py
git commit -m "feat(context-mapper): LLM-facing types + ContextMapperError"
```

---

## Task 4: prompt.py — build_map_prompt + build_remap_prompt  [sonnet]

**Files:**
- Create: `core/context_mapper/prompt.py`
- Test: `tests/test_context_mapper_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_mapper_prompt.py
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage,
                          ProjectMetadata, CritiqueFinding)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata
from core.context_mapper.prompt import build_map_prompt, build_remap_prompt
from core.schemas import ContextMap, ContextRelationship


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", description="orders",
                           ubiquitous_language=UbiquitousLanguage()),
            BoundedContext(context_name="Inventory", description="stock",
                           ubiquitous_language=UbiquitousLanguage()),
        ],
        global_rules=None,
    )


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="Orders reduce stock.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=10, truncated_chunks=0),
    )


def test_map_prompt_lists_contexts_and_taxonomy():
    p = build_map_prompt(_model(), _scout())
    assert "Ordering" in p and "Inventory" in p
    assert "CUSTOMER_SUPPLIER" in p and "ANTI_CORRUPTION_LAYER" in p and "SEPARATE_WAYS" in p
    assert "[0]" in p  # numbered scout sentence for grounding
    assert "import" not in p.lower()  # AST topology de-scoped (fix #2)


def test_remap_prompt_includes_feedback_and_prior_map():
    prev = ContextMap(model_id="m", relationships=[ContextRelationship(
        context_a="Ordering", context_b="Inventory",
        relationship_type="CONFORMIST", upstream="Inventory", rationale="x")])
    finding = CritiqueFinding(
        finding_type="WRONG_RELATIONSHIP_TYPE", priority="high",
        target_ref="relationship:Ordering->Inventory",
        rationale="Ordering translates Inventory's model; this is ACL not Conformist.",
        proposed_revision="Use ANTI_CORRUPTION_LAYER.")
    p = build_remap_prompt(_model(), _scout(), prev, [finding])
    assert "CONFORMIST" in p and "ANTI_CORRUPTION_LAYER" in p
    assert "WRONG_RELATIONSHIP_TYPE" in p or "ACL" in p or "translates" in p
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_prompt.py -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `core/context_mapper/prompt.py`**

```python
"""Build the context-mapping prompt: contexts + ubiquitous language + numbered
Scout sentences (grounding) + the 9-pattern taxonomy. Intent-level only — NO
AST import topology (it does not exist at generation time)."""
from typing import List
from core.schemas import DomainModel, ContextMap, CritiqueFinding
from core.pipeline_contracts import ScoutOutput

_TAXONOMY = """STRATEGIC DDD RELATIONSHIP PATTERNS (choose exactly one per related pair):
- CUSTOMER_SUPPLIER: downstream's needs drive an upstream supplier (set `upstream`).
- CONFORMIST: downstream conforms to upstream's model with no translation (set `upstream`).
- ANTI_CORRUPTION_LAYER: downstream translates upstream's model behind an ACL (set `upstream`).
- OPEN_HOST_SERVICE: upstream exposes a defined protocol for consumers (set `upstream`).
- PUBLISHED_LANGUAGE: integration via a shared well-documented language (set `upstream` = definer).
- PARTNERSHIP: two contexts succeed/fail together, co-evolved (upstream=null).
- SHARED_KERNEL: a shared subset of the model (upstream=null).
- SEPARATE_WAYS: no integration; they must NOT depend on each other (upstream=null).
- BIG_BALL_OF_MUD: tangled, unclear boundaries — a smell to flag (upstream=null).
Only emit a relationship for pairs that are genuinely related. Omit unrelated pairs."""

_INSTRUCTIONS = """You are a senior Domain-Driven Design strategist. Given the bounded
contexts of a system (with their ubiquitous language) and the numbered requirement
sentences they came from, identify the strategic relationships between context pairs.

Think step by step in `analysis`, then emit `relationships`. For each:
- context_a, context_b: the two context names (must be from the list below),
- relationship_type: one taxonomy value,
- upstream: context_a or context_b for directional types; null otherwise,
- rationale: why this pattern + direction, in DDD terms,
- evidence_sentence_indices: supporting numbered sentence indices ([] or [-1] if none).
"""


def _serialize_contexts(model: DomainModel) -> str:
    lines: List[str] = []
    for bc in model.bounded_contexts:
        ul = bc.ubiquitous_language
        ents = ", ".join(e.name for e in ul.entities) or "(none)"
        lines.append(f"- {bc.context_name}: {bc.description or '(no description)'} "
                     f"| entities: {ents}")
    return "\n".join(lines)


def _serialize_scout(scout: ScoutOutput) -> str:
    return "\n".join(f"[{s.index}] {s.text}" for s in scout.sentences)


def build_map_prompt(model: DomainModel, scout: ScoutOutput) -> str:
    return (
        _INSTRUCTIONS + "\n\n" + _TAXONOMY + "\n\nBOUNDED CONTEXTS:\n"
        + _serialize_contexts(model)
        + "\n\nNUMBERED SOURCE SENTENCES:\n" + _serialize_scout(scout)
    )


def _serialize_prev_map(prev: ContextMap) -> str:
    if not prev.relationships:
        return "(none)"
    return "\n".join(
        f"- {r.context_a}/{r.context_b}: {r.relationship_type} "
        f"(upstream={r.upstream})" for r in prev.relationships)


def _serialize_feedback(findings: List[CritiqueFinding]) -> str:
    return "\n".join(
        f"- {f.target_ref}: {f.rationale} | suggested: {f.proposed_revision}"
        for f in findings)


def build_remap_prompt(
    model: DomainModel, scout: ScoutOutput, prev: ContextMap,
    findings: List[CritiqueFinding],
) -> str:
    return (
        build_map_prompt(model, scout)
        + "\n\nYOUR PREVIOUS MAP:\n" + _serialize_prev_map(prev)
        + "\n\nCRITIC FEEDBACK (revise the affected relationships, keep the rest):\n"
        + _serialize_feedback(findings)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_prompt.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add core/context_mapper/prompt.py tests/test_context_mapper_prompt.py
git commit -m "feat(context-mapper): map + remap prompt builders (intent-level, no AST)"
```

---

## Task 5: mapper.py — run_context_mapper  [opus]

**Files:**
- Create: `core/context_mapper/mapper.py`
- Test: extend `tests/test_context_mapper_mapper.py`

**Contract:** `run_context_mapper(model, scout, feedback, *, client, stage_cfg) -> ContextMap`. One `structured_output` call (schema `ContextMapResponse`). Maps each `ProposedRelationship` → `ContextRelationship`, **trimming evidence** to indices in `scout` ∪ `{-1}` and de-duping; per-relationship schema-validation failures are dropped (counted, not fatal). `feedback is None` → `build_map_prompt`; else `build_remap_prompt`. Raises `ContextMapperError` on `json_failed`.

- [ ] **Step 1: Write the failing test (append to `tests/test_context_mapper_mapper.py`)**

```python
import pytest
from core.context_mapper.mapper import run_context_mapper
from core.context_mapper.errors import ContextMapperError
from core.context_mapper.types import ContextMapResponse, ProposedRelationship
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


class _Resp:
    def __init__(self, parsed=None, failed=False, reason=None):
        self.parsed = parsed
        self.json_failed = failed
        self.json_fail_reason = reason


class _Client:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []
    def structured_output(self, **kw):
        self.calls.append(kw)
        return self._resp


class _Cfg:
    model_id = "gemini-3.1-pro-preview"
    temperature = 0.05
    seed = 42


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", ubiquitous_language=UbiquitousLanguage()),
            BoundedContext(context_name="Inventory", ubiquitous_language=UbiquitousLanguage()),
        ],
        global_rules=None)


def _scout():
    return ScoutOutput(
        sentences=[SectionedSentence(index=0, text="s0"), SectionedSentence(index=1, text="s1")],
        chunk_metadata=ChunkMetadata(chunk_count=2, total_chars=4, truncated_chunks=0))


def test_maps_relationships_and_sets_model_id():
    resp = _Resp(parsed=ContextMapResponse(relationships=[ProposedRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type="CUSTOMER_SUPPLIER",
        upstream="Inventory", rationale="r", evidence_sentence_indices=[0])]))
    cm = run_context_mapper(_model(), _scout(), None, client=_Client(resp), stage_cfg=_Cfg())
    assert cm.model_id == "gemini-3.1-pro-preview"
    assert len(cm.relationships) == 1
    assert cm.relationships[0].relationship_type == "CUSTOMER_SUPPLIER"


def test_evidence_trimmed_to_scout_or_minus_one():
    resp = _Resp(parsed=ContextMapResponse(relationships=[ProposedRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type="PARTNERSHIP",
        rationale="r", evidence_sentence_indices=[0, 1, 99, -1, 0])]))
    cm = run_context_mapper(_model(), _scout(), None, client=_Client(resp), stage_cfg=_Cfg())
    assert sorted(cm.relationships[0].evidence_sentence_indices) == [-1, 0, 1]  # 99 dropped, deduped


def test_invalid_relationship_dropped_not_fatal():
    # directional type with upstream not in pair → ContextRelationship validation fails → dropped
    resp = _Resp(parsed=ContextMapResponse(relationships=[
        ProposedRelationship(context_a="Ordering", context_b="Inventory",
            relationship_type="CONFORMIST", upstream="Ghost", rationale="bad"),
        ProposedRelationship(context_a="Ordering", context_b="Inventory",
            relationship_type="PARTNERSHIP", rationale="ok"),
    ]))
    cm = run_context_mapper(_model(), _scout(), None, client=_Client(resp), stage_cfg=_Cfg())
    assert len(cm.relationships) == 1 and cm.relationships[0].relationship_type == "PARTNERSHIP"


def test_json_failed_raises():
    with pytest.raises(ContextMapperError):
        run_context_mapper(_model(), _scout(), None,
                           client=_Client(_Resp(failed=True, reason="schema")), stage_cfg=_Cfg())


def test_feedback_path_uses_remap(monkeypatch):
    from core.schemas import ContextMap
    import core.context_mapper.mapper as m
    seen = {}
    monkeypatch.setattr(m, "build_remap_prompt", lambda *a, **k: seen.setdefault("remap", True) or "REMAP")
    resp = _Resp(parsed=ContextMapResponse(relationships=[]))
    run_context_mapper(_model(), _scout(), [object()], client=_Client(resp), stage_cfg=_Cfg(),
                       _prev_map=ContextMap(model_id="m"))
    assert seen.get("remap")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_mapper.py -q`
Expected: FAIL (cannot import run_context_mapper).

- [ ] **Step 3: Implement `core/context_mapper/mapper.py`**

```python
"""run_context_mapper: one schema-enforced LLM call producing a typed ContextMap.

Pure producer — maps ProposedRelationships to ContextRelationships, trimming
evidence to valid Scout indices (or the -1 inference sentinel) and dropping
relationships that fail schema validation (counted, non-fatal). Raises
ContextMapperError on json_failed."""
from typing import Any, List, Optional
from pydantic import ValidationError
from core.schemas import ContextMap, ContextRelationship, CritiqueFinding
from core.pipeline_contracts import ScoutOutput
from core.context_mapper.types import ContextMapResponse, ProposedRelationship
from core.context_mapper.prompt import build_map_prompt, build_remap_prompt
from core.context_mapper.errors import ContextMapperError


def _map_one(
    pr: ProposedRelationship, scout_indices: set,
) -> Optional[ContextRelationship]:
    seen = set()
    evidence: List[int] = []
    for i in pr.evidence_sentence_indices:
        if (i in scout_indices or i == -1) and i not in seen:
            seen.add(i)
            evidence.append(i)
    try:
        return ContextRelationship(
            context_a=pr.context_a, context_b=pr.context_b,
            relationship_type=pr.relationship_type, upstream=pr.upstream,
            rationale=pr.rationale, evidence_sentence_indices=evidence,
        )
    except ValidationError:
        return None


def run_context_mapper(
    model: Any,
    scout: ScoutOutput,
    feedback: Optional[List[CritiqueFinding]],
    *,
    client: Any,
    stage_cfg: Any,
    _prev_map: Optional[ContextMap] = None,
) -> ContextMap:
    """`feedback is None` → fresh map; else revise via build_remap_prompt using
    `_prev_map` (the model's existing context_map; defaults to empty)."""
    if feedback:
        prev = _prev_map or getattr(model, "context_map", None) or ContextMap(model_id=stage_cfg.model_id)
        prompt = build_remap_prompt(model, scout, prev, list(feedback))
    else:
        prompt = build_map_prompt(model, scout)

    response = client.structured_output(
        messages=[{"role": "user", "content": prompt}],
        schema=ContextMapResponse,
        model=stage_cfg.model_id,
        temperature=stage_cfg.temperature,
        seed=stage_cfg.seed,
    )
    if response.json_failed or not isinstance(response.parsed, ContextMapResponse):
        raise ContextMapperError(reason=response.json_fail_reason or "empty_parse")

    parsed: ContextMapResponse = response.parsed
    scout_indices = {s.index for s in scout.sentences}
    relationships: List[ContextRelationship] = []
    for pr in parsed.relationships:
        mapped = _map_one(pr, scout_indices)
        if mapped is not None:
            relationships.append(mapped)

    return ContextMap(model_id=stage_cfg.model_id, relationships=relationships)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_mapper.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Pyright + commit**

Run: `pyright core/context_mapper/mapper.py` → 0 errors.
```bash
git add core/context_mapper/mapper.py tests/test_context_mapper_mapper.py
git commit -m "feat(context-mapper): run_context_mapper (evidence trim, drop-invalid, remap path)"
```

---

## Task 6: __init__ facade + STAGE_TO_GROUP registration  [haiku/sonnet]

**Files:**
- Modify: `core/context_mapper/__init__.py`, `configs/models.py`
- Test: `tests/test_context_mapper_stage.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_mapper_stage.py
def test_facade_exports():
    from core.context_mapper import run_context_mapper, derive_allowed_dependencies, ContextMapperError
    assert callable(run_context_mapper) and callable(derive_allowed_dependencies)

def test_stage_registered():
    from configs.models import STAGE_TO_GROUP, stage_config
    assert STAGE_TO_GROUP["ContextMapper"] == "domain_extraction"
    cfg = stage_config("ContextMapper")
    assert cfg.model_id  # resolves to the domain_extraction group model
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_stage.py -q`
Expected: FAIL (ImportError / KeyError 'ContextMapper').

- [ ] **Step 3: Implement**

`core/context_mapper/__init__.py` (replace):
```python
"""Context-Mapper (A): strategic-DDD context map producer + pure derivation."""
from core.context_mapper.derive import derive_allowed_dependencies
from core.context_mapper.mapper import run_context_mapper
from core.context_mapper.errors import ContextMapperError

__all__ = ["derive_allowed_dependencies", "run_context_mapper", "ContextMapperError"]
```

`configs/models.py` — add to the `STAGE_TO_GROUP` dict (after `"Critic"`):
```python
    "ContextMapper": "domain_extraction",
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_stage.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add core/context_mapper/__init__.py configs/models.py tests/test_context_mapper_stage.py
git commit -m "feat(context-mapper): facade exports + ContextMapper stage registration"
```

---

## Task 7: pipeline — _apply_context_map + PipelineDeps.context_mapper + _generate_once slot  [opus]

**Files:**
- Modify: `core/orchestration/pipeline.py`
- Test: `tests/test_context_mapper_pipeline.py`

**Contract:** `_apply_context_map(model, deps, scout, *, feedback=None) -> DomainModel` — pure (deep-copies), no-op when `deps.context_mapper is None`, wraps the call in `_optional_stage("context_mapper")`, on `ContextMapperError` records `context_map.error` + keeps baseline deps, else derives + overwrites `allowed_dependencies`. Called at the end of `_generate_once`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_mapper_pipeline.py
import copy
from core.orchestration.pipeline import _apply_context_map, PipelineDeps
from core.context_mapper.errors import ContextMapperError
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage,
                          ProjectMetadata, ContextMap, ContextRelationship)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", ubiquitous_language=UbiquitousLanguage()),
            BoundedContext(context_name="Inventory", ubiquitous_language=UbiquitousLanguage()),
        ],
        global_rules=None)


def _scout():
    return ScoutOutput(sentences=[SectionedSentence(index=0, text="s")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=1, truncated_chunks=0))


def _deps(context_mapper):
    # only context_mapper matters here; other fields unused by _apply_context_map
    return PipelineDeps(scout=lambda t: None, architect=lambda s: None,
        architect_with_feedback=lambda s, i: None, specialist=lambda a, s: None,
        synthesizer=lambda x: None, verifier=lambda s: None, context_mapper=context_mapper)


def test_noop_when_mapper_none():
    m = _model()
    out = _apply_context_map(m, _deps(None), _scout())
    assert out.context_map is None and out is m


def test_purity_input_unchanged_and_deps_derived():
    def mapper(model, scout, feedback):
        return ContextMap(model_id="m", relationships=[ContextRelationship(
            context_a="Ordering", context_b="Inventory",
            relationship_type="CUSTOMER_SUPPLIER", upstream="Inventory", rationale="r")])
    m = _model()
    out = _apply_context_map(m, _deps(mapper), _scout())
    assert m.context_map is None  # input untouched (deep copy)
    assert out.context_map is not None
    ord_ctx = next(b for b in out.bounded_contexts if b.context_name == "Ordering")
    inv_ctx = next(b for b in out.bounded_contexts if b.context_name == "Inventory")
    assert ord_ctx.allowed_dependencies == ["Inventory"]
    assert inv_ctx.allowed_dependencies is None  # empty → None


def test_failure_keeps_baseline_and_records_error():
    def mapper(model, scout, feedback):
        raise ContextMapperError(reason="json_failed")
    m = _model()
    m.bounded_contexts[0].allowed_dependencies = ["Inventory"]  # text-scan baseline
    out = _apply_context_map(m, _deps(mapper), _scout())
    assert out.context_map is not None and out.context_map.error == "json_failed"
    assert out.bounded_contexts[0].allowed_dependencies == ["Inventory"]  # baseline kept


def test_feedback_forwarded():
    captured = {}
    def mapper(model, scout, feedback):
        captured["fb"] = feedback
        return ContextMap(model_id="m")
    _apply_context_map(_model(), _deps(mapper), _scout(), feedback=["finding"])
    assert captured["fb"] == ["finding"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_pipeline.py -q`
Expected: FAIL (cannot import _apply_context_map).

- [ ] **Step 3: Implement in `core/orchestration/pipeline.py`**

Under the `TYPE_CHECKING` block (where `CriticReport` is imported), add `ContextMap`:
```python
if TYPE_CHECKING:
    from core.schemas import CriticReport, ContextMap
```
After the `CriticFn` alias, add:
```python
# Context-Mapper: produce a strategic context map. feedback=None → fresh map;
# a list of CritiqueFindings → Critic-driven re-map.
ContextMapperFn = Callable[[DomainModel, ScoutOutput, Optional[list]], "ContextMap"]
```
Add the field to `PipelineDeps` (after `critic`):
```python
    context_mapper: Optional["ContextMapperFn"] = None
```
Add the helper (place it just above `run_pipeline`):
```python
def _apply_context_map(
    model: DomainModel,
    deps: "PipelineDeps",
    scout: ScoutOutput,
    *,
    feedback: Optional[List[Any]] = None,
) -> DomainModel:
    """Attach a strategic context map + re-derive allowed_dependencies.

    PURE: returns a deep copy; never mutates `model` (the critique loop's
    best_model may alias it). No-op (returns `model` unchanged) when no
    context_mapper is wired. On ContextMapperError the text-scan baseline
    allowed_dependencies is kept and the failure is recorded on context_map."""
    if deps.context_mapper is None:
        return model
    from core.context_mapper import derive_allowed_dependencies
    from core.context_mapper.errors import ContextMapperError
    from core.schemas import ContextMap

    new_model = model.model_copy(deep=True)
    with _optional_stage("context_mapper"):
        try:
            cmap = deps.context_mapper(new_model, scout, feedback)
        except ContextMapperError as exc:
            print(f"  ⚠️  context-mapper failed: {exc}; keeping baseline allowed_dependencies")
            new_model.context_map = ContextMap(model_id="unknown", error=exc.reason)
            return new_model

    valid_names = {bc.context_name for bc in new_model.bounded_contexts}
    derived, warnings = derive_allowed_dependencies(cmap, valid_names)
    cmap.warnings.extend(warnings)
    new_model.context_map = cmap
    for bc in new_model.bounded_contexts:
        dep_list = derived.get(bc.context_name)
        if dep_list is not None:
            bc.allowed_dependencies = sorted(dep_list) if dep_list else None
    return new_model
```
In `_generate_once`, change the final return (lines ~477–480):
```python
    model = _apply_context_map(model, deps, scout)
    return model, arch, refined_specialist
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_pipeline.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Full suite + pyright + commit**

Run: `.venv/bin/python -m pytest -m "not integration" -q` (existing pipeline/critic tests must still pass — `_apply_context_map` is a no-op when `context_mapper` unset).
Run: `pyright core/orchestration/pipeline.py` → 0 errors.
```bash
git add core/orchestration/pipeline.py tests/test_context_mapper_pipeline.py
git commit -m "feat(context-mapper): _apply_context_map (pure) + PipelineDeps wiring + _generate_once slot"
```

---

## Task 8: architect.py — _build_context_mapper_fn + wire PipelineDeps  [sonnet]

**Files:**
- Modify: `core/architect.py` (add factory near `_build_critic_fn` at ~1002; add dep at ~1200)
- Test: `tests/test_context_mapper_wiring.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_mapper_wiring.py
import os
from core.architect import DomainArchitect


def test_context_mapper_fn_default_on(monkeypatch):
    monkeypatch.delenv("DDD_CONTEXT_MAP", raising=False)
    da = DomainArchitect()
    assert da._build_context_mapper_fn() is not None


def test_context_mapper_fn_opt_out(monkeypatch):
    for v in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DDD_CONTEXT_MAP", v)
        assert DomainArchitect()._build_context_mapper_fn() is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_wiring.py -q`
Expected: FAIL (AttributeError: _build_context_mapper_fn).

- [ ] **Step 3: Implement**

Add method after `_build_critic_fn` (architect.py:1019):
```python
    def _build_context_mapper_fn(self):
        """Return a context-mapper callable, or None when DDD_CONTEXT_MAP is
        disabled. ON by default; set DDD_CONTEXT_MAP to 0/false/no/off to opt
        out. Uses the generation client + the ContextMapper stage config (G1)."""
        if os.getenv("DDD_CONTEXT_MAP", "1").strip().lower() in ("0", "false", "no", "off"):
            return None

        from core.context_mapper import run_context_mapper
        cm_cfg = stage_config("ContextMapper")

        def context_mapper_fn(model, scout, feedback):
            return run_context_mapper(
                model, scout, feedback, client=self.client, stage_cfg=cm_cfg,
            )

        return context_mapper_fn
```
Add to the `PipelineDeps(...)` constructor (architect.py:1200, after `critic=...`):
```python
            context_mapper=self._build_context_mapper_fn(),
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_wiring.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add core/architect.py tests/test_context_mapper_wiring.py
git commit -m "feat(context-mapper): wire context_mapper into PipelineDeps (DDD_CONTEXT_MAP, default ON)"
```

---

## Task 9: Critic finding types + critic.py relationship target validation  [sonnet]

**Files:**
- Modify: `core/schemas.py` (`CritiqueFinding.finding_type`), `core/critic/types.py` (`ProposedFinding.finding_type`), `core/critic/critic.py` (`_map_finding`)
- Test: `tests/test_critic_relationship.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_relationship.py
from core.critic.critic import _map_finding, _valid_targets
from core.critic.types import ProposedFinding
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata)


def _model():
    return DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", ubiquitous_language=UbiquitousLanguage()),
            BoundedContext(context_name="Inventory", ubiquitous_language=UbiquitousLanguage()),
        ],
        global_rules=None)


def test_relationship_finding_with_valid_contexts_kept():
    m = _model()
    pf = ProposedFinding(finding_type="WRONG_RELATIONSHIP_TYPE", priority="high",
        target_ref="relationship:Ordering->Inventory", rationale="r", proposed_revision="ACL")
    out = _map_finding(pf, _valid_targets(m), {0}, model=m)
    assert out is not None and out.finding_type == "WRONG_RELATIONSHIP_TYPE"


def test_missing_relationship_on_absent_pair_survives():
    m = _model()
    pf = ProposedFinding(finding_type="MISSING_RELATIONSHIP", priority="medium",
        target_ref="relationship:Ordering->Inventory", rationale="should relate",
        proposed_revision="add Customer-Supplier")
    assert _map_finding(pf, _valid_targets(m), {0}, model=m) is not None


def test_relationship_with_unknown_context_dropped():
    m = _model()
    pf = ProposedFinding(finding_type="ILLEGAL_DEPENDENCY", priority="high",
        target_ref="relationship:Ordering->Ghost", rationale="r", proposed_revision="x")
    assert _map_finding(pf, _valid_targets(m), {0}, model=m) is None


def test_existing_context_finding_still_works():
    m = _model()
    pf = ProposedFinding(finding_type="ANEMIC_ENTITY", priority="low",
        target_ref="context:Ordering", rationale="r", proposed_revision="x")
    assert _map_finding(pf, _valid_targets(m), {0}, model=m) is not None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_critic_relationship.py -q`
Expected: FAIL (TypeError: _map_finding got unexpected 'model', or ValidationError on new finding_type).

- [ ] **Step 3: Implement**

In `core/schemas.py` `CritiqueFinding.finding_type` Literal AND `core/critic/types.py` `ProposedFinding.finding_type` Literal, append the three values:
```python
        "WRONG_RELATIONSHIP_TYPE", "ILLEGAL_DEPENDENCY", "MISSING_RELATIONSHIP",
```
(Place before the trailing `"OTHER",` in each.)

In `core/critic/critic.py`, add a module constant + update `_map_finding` to accept `model` and validate relationship targets by context name:
```python
_RELATIONSHIP_TYPES = {"WRONG_RELATIONSHIP_TYPE", "ILLEGAL_DEPENDENCY", "MISSING_RELATIONSHIP"}


def _relationship_contexts_valid(target_ref: str, context_names: Set[str]) -> bool:
    """'relationship:A->B' → True iff both A and B are existing context names."""
    body = target_ref.split(":", 1)[-1]
    if "->" not in body:
        return False
    a, b = (p.strip() for p in body.split("->", 1))
    return a in context_names and b in context_names
```
Change the `_map_finding` signature + the target check:
```python
def _map_finding(
    pf: ProposedFinding, valid_targets: Set[str], scout_indices: Set[int],
    *, model: DomainModel,
) -> Optional[CritiqueFinding]:
    if pf.finding_type in _RELATIONSHIP_TYPES:
        context_names = {bc.context_name for bc in model.bounded_contexts}
        if not _relationship_contexts_valid(pf.target_ref, context_names):
            return None
    elif pf.target_ref not in valid_targets:
        return None
    evidence = [i for i in pf.evidence_sentence_indices if i in scout_indices]
    return CritiqueFinding(
        finding_type=pf.finding_type, priority=pf.priority,
        target_ref=pf.target_ref, rationale=pf.rationale,
        proposed_revision=pf.proposed_revision,
        evidence_sentence_indices=evidence,
    )
```
Update the single call site in `run_critic` (critic.py:68): `mapped = _map_finding(pf, valid_targets, scout_indices, model=model)`.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_critic_relationship.py tests/test_context_mapper_schema.py -q`
Expected: PASS. Also run existing critic tests: `.venv/bin/python -m pytest -k critic -q` → still green.

- [ ] **Step 5: Pyright + commit**

Run: `pyright core/critic/critic.py core/schemas.py` → 0 errors.
```bash
git add core/schemas.py core/critic/types.py core/critic/critic.py tests/test_critic_relationship.py
git commit -m "feat(critic): relationship finding types + context-name target validation"
```

---

## Task 10: Critic prompt — serialize context_map + relationship review step  [sonnet]

**Files:**
- Modify: `core/critic/prompt.py`
- Test: `tests/test_critic_prompt_relationship.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_prompt_relationship.py
from core.critic.prompt import build_critique_prompt
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata,
                          ContextMap, ContextRelationship)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


def _model_with_map():
    m = DomainModel(
        project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[
            BoundedContext(context_name="Ordering", ubiquitous_language=UbiquitousLanguage()),
            BoundedContext(context_name="Inventory", ubiquitous_language=UbiquitousLanguage())],
        global_rules=None)
    m.context_map = ContextMap(model_id="m", relationships=[ContextRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type="CONFORMIST",
        upstream="Inventory", rationale="r")])
    return m


def _scout():
    return ScoutOutput(sentences=[SectionedSentence(index=0, text="s")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=1, truncated_chunks=0))


def test_prompt_includes_context_map_and_relationship_instruction():
    p = build_critique_prompt(_model_with_map(), _scout(), [])
    assert "CONFORMIST" in p
    assert "relationship" in p.lower()
    assert "WRONG_RELATIONSHIP_TYPE" in p or "relationship:" in p


def test_prompt_without_map_still_builds():
    m = _model_with_map(); m.context_map = None
    p = build_critique_prompt(m, _scout(), [])
    assert "Ordering" in p  # no crash, no context_map block
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_critic_prompt_relationship.py -q`
Expected: FAIL (CONFORMIST / relationship instruction absent).

- [ ] **Step 3: Implement in `core/critic/prompt.py`**

In `_INSTRUCTIONS`, add review step 5 + the relationship finding types to the list (after step 4 "Naming"):
```python
5. Context map: for each relationship, is the strategic pattern correct
   (e.g. is it really Conformist, or should it be ACL?), is the direction
   right, and are any required relationships MISSING or any ILLEGAL?
```
Append to the finding_type guidance line:
```
  (relationship findings use finding_type WRONG_RELATIONSHIP_TYPE /
   ILLEGAL_DEPENDENCY / MISSING_RELATIONSHIP and target_ref
   "relationship:<A>-><B>")
```
In `_serialize_model`, after the `bounded_contexts` block, include the map when present:
```python
    if model.context_map and model.context_map.relationships:
        compact["context_map"] = [
            {"context_a": r.context_a, "context_b": r.context_b,
             "relationship_type": r.relationship_type, "upstream": r.upstream}
            for r in model.context_map.relationships
        ]
```
(`compact` is the dict already built in `_serialize_model`; add this before the `return json.dumps(...)`.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_critic_prompt_relationship.py -q`
Expected: PASS (2 tests). Existing critic-prompt tests still green: `.venv/bin/python -m pytest -k "critic and prompt" -q`.

- [ ] **Step 5: Commit**

```bash
git add core/critic/prompt.py tests/test_critic_prompt_relationship.py
git commit -m "feat(critic): surface context_map + relationship review in critique prompt"
```

---

## Task 11: routing — _RELATIONSHIP partition + model_diff_summary deltas  [sonnet]

**Files:**
- Modify: `core/critic/routing.py`
- Test: `tests/test_critic_routing_relationship.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_routing_relationship.py
from core.critic.routing import partition_findings, model_diff_summary
from core.schemas import (CritiqueFinding, DomainModel, BoundedContext,
                          UbiquitousLanguage, ProjectMetadata, ContextMap, ContextRelationship)


def _f(ft, tr, pr="high"):
    return CritiqueFinding(finding_type=ft, priority=pr, target_ref=tr,
                           rationale="r", proposed_revision="x")


def test_partition_returns_four_buckets():
    findings = [
        _f("CONTEXT_SHOULD_MERGE", "context:A"),
        _f("ANEMIC_ENTITY", "entity:A.E"),
        _f("WRONG_RELATIONSHIP_TYPE", "relationship:A->B"),
        _f("OTHER", "context:A", pr="low"),
    ]
    structural, content, relationship, advisory = partition_findings(findings)
    assert [x.finding_type for x in structural] == ["CONTEXT_SHOULD_MERGE"]
    assert [x.finding_type for x in content] == ["ANEMIC_ENTITY"]
    assert [x.finding_type for x in relationship] == ["WRONG_RELATIONSHIP_TYPE"]
    assert len(advisory) == 1


def _model(rels=None):
    m = DomainModel(project_name="P",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[BoundedContext(context_name="A", ubiquitous_language=UbiquitousLanguage()),
                          BoundedContext(context_name="B", ubiquitous_language=UbiquitousLanguage())],
        global_rules=None)
    if rels is not None:
        m.context_map = ContextMap(model_id="m", relationships=rels)
    return m


def test_diff_summary_reports_relationship_change():
    before = _model(rels=[ContextRelationship(context_a="A", context_b="B",
        relationship_type="CONFORMIST", upstream="B", rationale="r")])
    after = _model(rels=[ContextRelationship(context_a="A", context_b="B",
        relationship_type="ANTI_CORRUPTION_LAYER", upstream="B", rationale="r")])
    summary = model_diff_summary(before, after)
    assert "A" in summary and "B" in summary
    assert "CONFORMIST" in summary or "ANTI_CORRUPTION_LAYER" in summary or "relationship" in summary.lower()
    assert summary != "no structural change"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_critic_routing_relationship.py -q`
Expected: FAIL (partition returns 3-tuple → ValueError unpacking 4).

- [ ] **Step 3: Implement in `core/critic/routing.py`**

Add the relationship set + make partition 4-way:
```python
_RELATIONSHIP = {"WRONG_RELATIONSHIP_TYPE", "ILLEGAL_DEPENDENCY", "MISSING_RELATIONSHIP"}
```
Rewrite `partition_findings`:
```python
def partition_findings(
    findings: List[CritiqueFinding],
) -> Tuple[List[CritiqueFinding], List[CritiqueFinding], List[CritiqueFinding], List[CritiqueFinding]]:
    """Return (structural, content, relationship, advisory). Only high/medium
    findings are routable; low priority and OTHER are advisory."""
    structural: List[CritiqueFinding] = []
    content: List[CritiqueFinding] = []
    relationship: List[CritiqueFinding] = []
    advisory: List[CritiqueFinding] = []
    for f in findings:
        if f.priority == "low" or f.finding_type == "OTHER":
            advisory.append(f)
        elif f.finding_type in _STRUCTURAL:
            structural.append(f)
        elif f.finding_type in _CONTENT:
            content.append(f)
        elif f.finding_type in _RELATIONSHIP:
            relationship.append(f)
        else:
            advisory.append(f)
    return structural, content, relationship, advisory
```
Extend `model_diff_summary` — before the final `return`, add context-map deltas:
```python
    def _rels(m):
        cm = getattr(m, "context_map", None)
        if not cm:
            return {}
        return {tuple(sorted((r.context_a, r.context_b))): (r.relationship_type, r.upstream)
                for r in cm.relationships}
    rb, ra = _rels(before), _rels(after)
    for pair in sorted(set(ra) - set(rb)):
        parts.append(f"relationship added: {pair[0]}/{pair[1]} = {ra[pair][0]}")
    for pair in sorted(set(rb) - set(ra)):
        parts.append(f"relationship removed: {pair[0]}/{pair[1]}")
    for pair in sorted(set(ra) & set(rb)):
        if ra[pair] != rb[pair]:
            parts.append(f"relationship changed: {pair[0]}/{pair[1]} {rb[pair][0]}→{ra[pair][0]}")
```

**Keep the suite green:** this 4-tuple change breaks the `partition_findings` caller in `loop.py` (currently unpacks 3). Apply a MINIMAL caller update in the same commit (the real relationship branch lands in Task 12). In `core/critic/loop.py`, change:
```python
        structural, content, _advisory = partition_findings(report.findings)
```
to:
```python
        structural, content, _relationship, _advisory = partition_findings(report.findings)
```
(Relationship findings are ignored until Task 12 — behavior is unchanged, suite stays green.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_critic_routing_relationship.py -q` → PASS (2 tests).
Run: `.venv/bin/python -m pytest -m "not integration" -q` → full suite still green (minimal loop.py caller update keeps it passing).

- [ ] **Step 5: Commit**

```bash
git add core/critic/routing.py core/critic/loop.py tests/test_critic_routing_relationship.py
git commit -m "feat(critic): 4-way finding partition (+relationship) + context-map diff deltas"
```

---

## Task 12: loop — relationship branch + every-cycle feedback + signature canonicalization  [opus]

**Files:**
- Modify: `core/critic/loop.py`
- Test: `tests/test_critic_loop_relationship.py`

**Contract:** revision cycle picks structural→full regen / content→specialist+synth+freshmap / else relationship base; then applies relationship feedback every cycle (filtered to surviving contexts) via `_apply_context_map`. `findings_signature` canonicalizes `relationship:` pairs. Imports `_apply_context_map` lazily (like `_generate_once`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_critic_loop_relationship.py
from core.critic.loop import findings_signature
from core.schemas import CritiqueFinding


def _f(ft, tr):
    return CritiqueFinding(finding_type=ft, priority="high", target_ref=tr,
                           rationale="r", proposed_revision="x")


def test_signature_canonicalizes_reversed_relationship_pairs():
    s1 = findings_signature([_f("WRONG_RELATIONSHIP_TYPE", "relationship:Ordering->Inventory")])
    s2 = findings_signature([_f("WRONG_RELATIONSHIP_TYPE", "relationship:Inventory->Ordering")])
    assert s1 == s2  # reversed pair must produce the same signature (fix #7)


def test_signature_unchanged_for_context_targets():
    s = findings_signature([_f("ANEMIC_ENTITY", "entity:A.E")])
    assert s == (("ANEMIC_ENTITY", "entity:A.E"),)
```

For the loop control-flow, add an integration-style test with fakes:

```python
# (same file) — relationship-only cycle re-maps without architect/specialist rerun
from core.orchestration.pipeline import PipelineDeps
from core.critic.loop import run_critique_loop
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata,
                          ContextMap, ContextRelationship, CriticReport, CriticLoopTrace)
from core.pipeline_contracts import ScoutOutput, SectionedSentence, ChunkMetadata


def _model(rel_type="CONFORMIST"):
    m = DomainModel(project_name="P",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[BoundedContext(context_name="Ordering", ubiquitous_language=UbiquitousLanguage()),
                          BoundedContext(context_name="Inventory", ubiquitous_language=UbiquitousLanguage())],
        global_rules=None)
    m.context_map = ContextMap(model_id="m", relationships=[ContextRelationship(
        context_a="Ordering", context_b="Inventory", relationship_type=rel_type,
        upstream="Inventory", rationale="r")])
    return m


def test_relationship_only_cycle_remaps_without_producer_rerun(monkeypatch):
    scout = ScoutOutput(sentences=[SectionedSentence(index=0, text="s")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=1, truncated_chunks=0))
    calls = {"architect": 0, "specialist": 0, "mapper": 0}

    def fake_generate_once(scout_, deps_, srs_path, *, architect_feedback=None):
        calls["architect"] += 1
        return _model(), object(), [object()]

    def fake_apply_map(model, deps, scout_, *, feedback=None):
        calls["mapper"] += 1
        out = model.model_copy(deep=True)
        out.context_map = ContextMap(model_id="m", relationships=[ContextRelationship(
            context_a="Ordering", context_b="Inventory",
            relationship_type="ANTI_CORRUPTION_LAYER", upstream="Inventory", rationale="fixed")])
        return out

    import core.critic.loop as loopmod
    monkeypatch.setattr(loopmod, "_generate_once", fake_generate_once, raising=False)
    # _apply_context_map is imported lazily inside loop; patch at source
    import core.orchestration.pipeline as pl
    monkeypatch.setattr(pl, "_apply_context_map", fake_apply_map)

    # critic: cycle 0 emits a relationship finding, then converges
    seq = [
        CriticReport(model_id="m", findings=[CritiqueFinding(
            finding_type="WRONG_RELATIONSHIP_TYPE", priority="high",
            target_ref="relationship:Ordering->Inventory", rationale="ACL not Conformist",
            proposed_revision="ACL")], loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged")),
        CriticReport(model_id="m", findings=[], loop=CriticLoopTrace(cycles_used=1, best_cycle=0, outcome="converged")),
    ]
    def fake_critic(model, scout_, history):
        return seq.pop(0)

    deps = PipelineDeps(scout=lambda t: scout, architect=lambda s: None,
        architect_with_feedback=lambda s, i: None, specialist=lambda a, s: None,
        synthesizer=lambda x: None, verifier=lambda s: None,
        specialist_with_feedback=lambda a, s, p, i: p, critic=fake_critic,
        context_mapper=lambda m, s, fb: ContextMap(model_id="m"))

    result = run_critique_loop(scout, deps, "srs")
    assert calls["architect"] == 1  # only cycle-0 generate; relationship-only cycle did NOT re-run architect
    assert calls["mapper"] >= 1     # relationship feedback re-map happened
    assert result.context_map.relationships[0].relationship_type == "ANTI_CORRUPTION_LAYER"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_critic_loop_relationship.py -q`
Expected: FAIL (signature not canonical; loop unpacks 3 from partition).

- [ ] **Step 3: Implement in `core/critic/loop.py`**

Update imports:
```python
from core.critic.routing import (
    partition_findings, adapt_structural_to_issues, adapt_content_to_issues,
    model_diff_summary,
)
```
Add a canonicalizer + use it in `findings_signature`:
```python
def _canonical_target(target_ref: str) -> str:
    """Canonicalize relationship pairs so A->B == B->A for flap detection."""
    if target_ref.startswith("relationship:") and "->" in target_ref:
        body = target_ref.split(":", 1)[1]
        a, b = (p.strip() for p in body.split("->", 1))
        return "relationship:" + "->".join(sorted((a, b)))
    return target_ref


def findings_signature(findings: List[CritiqueFinding]) -> Tuple:
    return tuple(sorted(
        (f.finding_type, _canonical_target(f.target_ref))
        for f in findings if f.priority in ("high", "medium")
    ))
```
Add a relationship-pair filter helper:
```python
def _relationship_pair_in(f: CritiqueFinding, names: set) -> bool:
    body = f.target_ref.split(":", 1)[-1]
    if "->" not in body:
        return False
    a, b = (p.strip() for p in body.split("->", 1))
    return a in names and b in names
```
Rewrite the revision-cycle body (loop.py lines ~85–97) inside the `for cycle` loop:
```python
        structural, content, relationship, _advisory = partition_findings(report.findings)
        try:
            from core.orchestration.pipeline import _apply_context_map
            if structural:
                new_model, arch, specialist = _generate_once(
                    scout, deps, srs_path,
                    architect_feedback=adapt_structural_to_issues(structural),
                )
            elif content:
                specialist = deps.specialist_with_feedback(
                    arch, scout, specialist, adapt_content_to_issues(content),
                )
                new_model = deps.synthesizer(specialist)
                new_model = _apply_context_map(new_model, deps, scout)
            else:
                new_model = model
            # fix #5: apply relationship feedback every cycle, filtered to live contexts
            if relationship:
                survivors = {bc.context_name for bc in new_model.bounded_contexts}
                rel_live = [f for f in relationship if _relationship_pair_in(f, survivors)]
                if rel_live:
                    new_model = _apply_context_map(new_model, deps, scout, feedback=rel_live)
            new_report = deps.critic(new_model, scout, history)
        except (CriticError, PipelineError) as exc:
            return _finalize_failed(best_model, exc, cycles_used=cycle + 1,
                                    score_trace=score_trace, count_trace=count_trace,
                                    best_report=best_report, best_cycle=best_cycle)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_critic_loop_relationship.py -q`
Expected: PASS. Then the FULL suite (Task 11's break is now resolved): `.venv/bin/python -m pytest -m "not integration" -q` → green.

- [ ] **Step 5: Pyright + commit**

Run: `pyright core/critic/loop.py` → 0 errors.
```bash
git add core/critic/loop.py tests/test_critic_loop_relationship.py
git commit -m "feat(critic): relationship-only loop branch + every-cycle remap + pair-canonical flap signature"
```

---

## Task 13: AST import_graph — diagnostics-only when context_map authoritative  [opus]

**Files:**
- Modify: `core/AST/import_graph.py` (`apply_import_topology_to_model`)
- Test: `tests/test_import_graph_context_map.py`

**Contract:** when `model_data["context_map"]` is present AND `error is None`, never auto-fill `allowed_dependencies` (record `cross_check_diff` for all contexts, `auto_populated == []`). Otherwise legacy behavior.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import_graph_context_map.py
from core.AST.import_graph import apply_import_topology_to_model
import textwrap, os


def _write(tmp_path, rel, body):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_authoritative_map_blocks_autofill(tmp_path):
    # Ordering imports Inventory in code, but A declared SEPARATE_WAYS (empty deps).
    _write(tmp_path, "ordering/svc.py", "import inventory.models\n")
    _write(tmp_path, "inventory/models.py", "X = 1\n")
    model_data = {
        "context_map": {"model_id": "m", "error": None, "warnings": [],
                        "relationships": [{"context_a": "ordering", "context_b": "inventory",
                                           "relationship_type": "SEPARATE_WAYS", "upstream": None,
                                           "rationale": "r", "evidence_sentence_indices": []}]},
        "bounded_contexts": [
            {"context_name": "ordering", "allowed_dependencies": None},
            {"context_name": "inventory", "allowed_dependencies": None}],
    }
    diags = apply_import_topology_to_model(model_data,
        python_files=[str(tmp_path / "ordering/svc.py"), str(tmp_path / "inventory/models.py")],
        workspace_root=str(tmp_path))
    # NOT auto-filled (Separate Ways enforcement preserved)
    assert model_data["bounded_contexts"][0]["allowed_dependencies"] is None
    assert diags["auto_populated"] == []
    # but the drift IS recorded for review
    assert "ordering" in diags["cross_check_diff"]


def test_failed_map_keeps_legacy_autofill(tmp_path):
    _write(tmp_path, "ordering/svc.py", "import inventory.models\n")
    _write(tmp_path, "inventory/models.py", "X = 1\n")
    model_data = {
        "context_map": {"model_id": "unknown", "error": "json_failed", "warnings": [], "relationships": []},
        "bounded_contexts": [
            {"context_name": "ordering", "allowed_dependencies": None},
            {"context_name": "inventory", "allowed_dependencies": None}],
    }
    diags = apply_import_topology_to_model(model_data,
        python_files=[str(tmp_path / "ordering/svc.py"), str(tmp_path / "inventory/models.py")],
        workspace_root=str(tmp_path))
    assert model_data["bounded_contexts"][0]["allowed_dependencies"] == ["inventory"]  # legacy fill
    assert "ordering" in diags["auto_populated"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_import_graph_context_map.py -q`
Expected: FAIL (first test: deps got auto-filled to ["inventory"]).

- [ ] **Step 3: Implement in `core/AST/import_graph.py`**

Inside `apply_import_topology_to_model`, after `contexts = model_data.get("bounded_contexts", []) or []` and the empty guard, compute the authoritative flag and use it to gate the auto-fill branch:
```python
    cm = model_data.get("context_map")
    map_authoritative = bool(cm) and cm.get("error") is None
```
In the per-context loop, replace the existing fill block:
```python
        existing = context.get("allowed_dependencies")
        if not existing:
            if map_authoritative:
                # A authoritatively declared deps (possibly empty, e.g. Separate
                # Ways). Do NOT repopulate — only record what code imports show.
                cross_check_diff[ctx_name] = {
                    "extra_in_llm": [],
                    "extra_in_derived": sorted(derived_set),
                }
                continue
            context["allowed_dependencies"] = sorted(derived_set)
            auto_populated.append(ctx_name)
            continue
```
(The `existing`-non-empty branch that computes `extra_llm`/`extra_derived` stays unchanged.)

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_import_graph_context_map.py -q`
Expected: PASS (2 tests). Existing import-graph tests still green: `.venv/bin/python -m pytest -k import_graph -q`.

- [ ] **Step 5: Pyright + commit**

Run: `pyright core/AST/import_graph.py` → 0 errors.
```bash
git add core/AST/import_graph.py tests/test_import_graph_context_map.py
git commit -m "fix(ast): import-topology is diagnostics-only when context_map is authoritative"
```

---

## Task 14: e2e — pipeline produces context_map + env toggles  [sonnet]

**Files:**
- Test: `tests/test_context_mapper_e2e.py`

- [ ] **Step 1: Write the test (TDD: should pass once Tasks 1–13 are in)**

```python
# tests/test_context_mapper_e2e.py
from core.orchestration.pipeline import run_pipeline, PipelineDeps
from core.schemas import (DomainModel, BoundedContext, UbiquitousLanguage, ProjectMetadata,
                          ContextMap, ContextRelationship)
from core.pipeline_contracts import (ScoutOutput, SectionedSentence, ChunkMetadata,
                                      ArchitectOutput, VerifierResult)


def _scout_fn(text):
    return ScoutOutput(sentences=[SectionedSentence(index=0, text="Orders reduce stock.")],
        chunk_metadata=ChunkMetadata(chunk_count=1, total_chars=20, truncated_chunks=0))


def _model():
    return DomainModel(project_name="Shop",
        project_metadata=ProjectMetadata(version="1.0.0", generated_at="now"),
        bounded_contexts=[BoundedContext(context_name="Ordering", ubiquitous_language=UbiquitousLanguage()),
                          BoundedContext(context_name="Inventory", ubiquitous_language=UbiquitousLanguage())],
        global_rules=None)


def _base_deps(context_mapper=None, critic=None):
    return PipelineDeps(
        scout=_scout_fn,
        architect=lambda s: ArchitectOutput(contexts=[]),
        architect_with_feedback=lambda s, i: ArchitectOutput(contexts=[]),
        specialist=lambda a, s: [object()],
        synthesizer=lambda x: _model(),
        verifier=lambda snap: VerifierResult(ok=True, issues=[]),
        context_mapper=context_mapper, critic=critic)


def test_non_critic_path_still_maps():
    def mapper(model, scout, feedback):
        return ContextMap(model_id="m", relationships=[ContextRelationship(
            context_a="Ordering", context_b="Inventory", relationship_type="CUSTOMER_SUPPLIER",
            upstream="Inventory", rationale="r")])
    out = run_pipeline(srs_text="x", deps=_base_deps(context_mapper=mapper), srs_path="s")
    assert out.context_map is not None
    ord_ctx = next(b for b in out.bounded_contexts if b.context_name == "Ordering")
    assert ord_ctx.allowed_dependencies == ["Inventory"]


def test_context_mapper_none_leaves_map_none():
    out = run_pipeline(srs_text="x", deps=_base_deps(context_mapper=None), srs_path="s")
    assert out.context_map is None
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest tests/test_context_mapper_e2e.py -q`
Expected: PASS (2 tests). If `VerifierResult`/`ArchitectOutput` constructor args differ, align with `core/pipeline_contracts.py` (read it; do not guess).

- [ ] **Step 3: Full gate**

Run: `.venv/bin/python -m pytest -m "not integration" -q` → all green (expect ~763 + new tests).
Run: `pyright` → 0 prod errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_context_mapper_e2e.py
git commit -m "test(context-mapper): e2e pipeline map production + env-toggle coverage"
```

---

## Final verification (after all tasks)

- [ ] `.venv/bin/python -m pytest -m "not integration" -q` — full suite green.
- [ ] `pyright` — 0 production errors.
- [ ] Manual sanity: `DDD_CONTEXT_MAP=0 .venv/bin/python -m pytest -k context_mapper -q` semantics hold (map None path).
- [ ] Update `development_docs/INDEX.md` + write `development_docs/A-context-mapper.md` (TL;DR, decisions, file changes, the 10 Codex fixes, limitations/follow-ups).
- [ ] Merge `feat/context-mapper` → `main`.
- [ ] Update memory `project_llm_augmentation` (A shipped).
