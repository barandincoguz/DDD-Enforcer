# Spec — A: Context-Mapper (LLM strategic-DDD context map)

**Status:** APPROVED (design forks resolved with user; adversarial-reviewed by Codex gpt-5.5 xhigh → SHIP-WITH-FIXES, all 10 findings integrated below)
**Date:** 2026-05-25
**Branch:** `feat/context-mapper`
**Predecessor:** C — Holistic Critic (`docs/superpowers/specs/2026-05-25-holistic-critic-design.md`), shipped + default ON.
**Handoff:** `.planning/HANDOFF-2026-05-25-context-mapper.md`
**Goal:** improve `domain_model.json` quality (paper de-scoped; accuracy over cost).

---

## TL;DR

A second LLM design-judgment agent ("Context-Mapper", stage `ContextMapper`) produces a typed **DDD strategic context map** — for each related bounded-context pair, a relationship type (Partnership, Shared Kernel, Customer-Supplier, Conformist, ACL, OHS, Published Language, Separate Ways, Big Ball of Mud) + direction + rationale + grounding. The map is persisted as a new `DomainModel.context_map`, and the flat `allowed_dependencies` (consumed by V4 boundary enforcement + D5) is **derived** from it. A runs inside the single-pass generation body and participates in the existing Critic critique→revise loop as a **Critic-driven producer**: the Critic critiques relationship correctness, and relationship findings route back to A's feedback re-map (the only way to fix a mislabel, since A is near-deterministic at temp 0.05 + seed 42).

---

## Motivation

Strategic DDD is currently absent from the model. `BoundedContext.allowed_dependencies` is a flat `List[str]` with no relationship semantics, populated by a **text-scan stub** (`core/synthesizer/enrich.py:_infer_and_enrich_dependencies`, lines 77–90) whose own docstring promises *"ONE additional LLM call to disambiguate allowed_dependencies across contexts"* — never built. A is that step, elevated to typed relationships. Result: V4 enforcement reasons about a real, intent-level context map instead of a regex heuristic, and the model carries the strategic layer a DDD practitioner expects.

---

## Locked decisions (resolved with user)

- **D-A1 — schema:** new `DomainModel.context_map` field (mirrors how `critic_report` was added, `schemas.py:419`). `allowed_dependencies` becomes a **derived projection**; its `Optional[List[str]]` type is unchanged (V4 `validator.py:562`, D5, `import_graph.py` all assume `List[str]`).
- **D-A2 — run-mode:** A is a **Critic-driven producer**, not one-shot. A near-deterministic single pass cannot self-correct a relationship mislabel; only Critic feedback in the prompt can. So the Critic sees + critiques `context_map`; relationship findings route to A's `_with_feedback` re-map.
- **D-A3 — taxonomy:** full 9 strategic patterns incl. Separate Ways + Big Ball of Mud.

---

## Adversarial review (Codex gpt-5.5 xhigh) — findings integrated

Verdict: **SHIP-WITH-FIXES**. Full trace archived in session tool-results. All integrated:

| # | Sev | Finding | Resolution in this spec |
|---|-----|---------|------------------------|
| 1 | 🔴 | Separate Ways/BBoM enforcement defeated: post-loop AST `import_graph.py:185 (if not existing)` auto-fills A's intentional-empty deps from real imports → V4 stops flagging forbidden imports. | §6.1 — AST topology becomes **diagnostics-only when `context_map` is present** (record `cross_check_diff`, never auto-fill). |
| 2 | 🔴 | A cannot see AST import-topology at gen-time (AST enrichment runs after `analyze_document`). | §2 (`prompt.py`) — **A is intent-level only**; input = contexts + ubiquitous language + Scout sentences. AST de-scoped from A's prompt. |
| 3 | 🔴 | A-derived deps unverified: `verifier_fn` (architect.py:1166) checks name-only dicts; D11 has no production caller. | §4.4 — validate **inside `_apply_context_map`**: drop relationships with non-existent context names (→ `warnings`). |
| 4 | 🔴 | Mutual (Partnership/Shared Kernel) → both-edges = 2-cycle → D11 false-positive. | §4.4 — cycle detection in `_apply_context_map` is **mutual-exempt**; D11 is NOT wired into `verifier_fn`. |
| 5 | 🟠 | `structural>content>relationship` branch priority starves relationship feedback under cycle cap 3. | §5.3 — apply relationship feedback **every cycle**, filtered to surviving contexts, after the structural/content path. |
| 6 | 🟠 | In-place remap corrupts keep-best (`best_model` aliases `model`). | §4.3 — `_apply_context_map` is **pure**: `model.model_copy(deep=True)`, mutate copy, return. |
| 7 | 🟠 | Flap signature `(finding_type, target_ref)` mis-detects reversed pairs `A->B`/`B->A`. | §5.4 — `findings_signature` **canonicalizes** relationship target_refs (sorted pair). |
| 8 | 🟠 | `model_diff_summary` ignores context_map → Reflexion history lies on relationship-only cycles. | §5.5 — extend `model_diff_summary` with context-map deltas. |
| 9 | 🟠 | A evidence not trimmed like Critic; `-1` sentinel unhandled. | §2 (`mapper.py`) — trims evidence to `i in scout_indices or i == -1`, de-dupes. |
| 10 | 🟡 | A LLM calls lose telemetry without a stage wrapper. | §4.2 — wrap A's call in `_optional_stage("context_mapper")`. |

---

## Architecture

### §1 Schema (`core/schemas.py`)

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
        description="context_a or context_b — the upstream/supplier side. "
                    "None for mutual (PARTNERSHIP, SHARED_KERNEL) and "
                    "non-integration (SEPARATE_WAYS, BIG_BALL_OF_MUD) types.",
    )
    rationale: str = Field(description="Why this pattern + direction, in DDD terms.")
    evidence_sentence_indices: List[int] = Field(
        default_factory=list,
        description="Scout sentence indices grounding this relationship; "
                    "[-1] if inference-only with no single supporting sentence.",
    )

    @model_validator(mode="after")
    def _check_upstream_consistency(self) -> "ContextRelationship":
        directional = {"CUSTOMER_SUPPLIER", "CONFORMIST", "ANTI_CORRUPTION_LAYER",
                       "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE"}
        if self.relationship_type in directional:
            if self.upstream not in (self.context_a, self.context_b):
                raise ValueError(
                    f"{self.relationship_type} requires upstream to be one of "
                    f"context_a/context_b; got {self.upstream!r}")
        else:  # mutual / non-integration
            if self.upstream is not None:
                raise ValueError(
                    f"{self.relationship_type} is non-directional; upstream must be None")
        if self.context_a == self.context_b:
            raise ValueError("context_a and context_b must differ")
        return self


class ContextMap(BaseModel):
    """Strategic context map produced by the Context-Mapper (A)."""
    relationships: List[ContextRelationship] = Field(default_factory=list)
    model_id: str
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues recorded during validation/derivation "
                    "(dropped invalid-name pairs, non-mutual cycles).")
    error: Optional[str] = Field(
        default=None,
        description="Set when the mapping LLM call failed (json_failed) after "
                    "retries; allowed_dependencies then falls back to the "
                    "text-scan baseline.")


# DomainModel gains (forward-ref string, exactly like critic_report):
context_map: Optional["ContextMap"] = Field(
    default=None,
    description="Strategic DDD context map (Context-Mapper output). None when "
                "DDD_CONTEXT_MAP is disabled or no map was produced.",
)
```

A `ValidationError` from the model_validator surfaces as `structured_output(...).json_failed` (the LLM client does not raise; it sets the flag), so A retries via its normal retry path, then falls back (§ failure modes).

### §2 Context-Mapper package (`core/context_mapper/`, mirrors `core/critic/`)

| File | Responsibility |
|---|---|
| `errors.py` | `ContextMapperError(reason: str)` (mirrors `CriticError`). |
| `types.py` | LLM-facing schema: `ProposedRelationship` + `ContextMapResponse{analysis: str, relationships: [...]}` (CoT scratchpad + relationships, mirrors `CriticResponse`). |
| `prompt.py` | `build_map_prompt(model, scout)` + `build_remap_prompt(model, scout, prev_map, findings)`. Includes the 9-pattern taxonomy definitions + 1–2 few-shot examples + the contexts & their ubiquitous language + numbered Scout sentences for grounding. **No AST topology** (fix #2). |
| `mapper.py` | `run_context_mapper(model, scout, feedback, *, client, stage_cfg) -> ContextMap`: one `structured_output` call → map `ProposedRelationship`→`ContextRelationship`, **trim evidence** to `i in scout_indices or i == -1` + de-dupe (fix #9), drop pairs that fail schema validation (count malformed). `feedback is None` → `build_map_prompt`; else `build_remap_prompt`. Raises `ContextMapperError` on `json_failed`. |
| `derive.py` | Pure functions (§4.4): `derive_allowed_dependencies(ContextMap, valid_context_names) -> (Dict[str,List[str]], warnings: List[str])`. |
| `__init__.py` | Facade exports. |

#### §2.1 LLM config
`stage_config("ContextMapper")` → group `domain_extraction` → G1 `gemini-3.1-pro-preview`, temp 0.05, seed 42 (same as generation). Register `STAGE_TO_GROUP["ContextMapper"] = "domain_extraction"` in `configs/models.py`.

### §4 Pipeline wiring (`core/orchestration/pipeline.py`)

#### §4.1 Dep
```python
ContextMapperFn = Callable[[DomainModel, ScoutOutput, Optional[list]], "ContextMap"]
# PipelineDeps gains:
context_mapper: Optional["ContextMapperFn"] = None
```
(forward-ref string `ContextMap` under `TYPE_CHECKING`, like `CriticFn`/`CriticReport`.)

#### §4.2 `_apply_context_map`
```python
def _apply_context_map(model, deps, scout, *, feedback=None) -> DomainModel:
    """Pure: returns a deep copy with context_map attached + allowed_dependencies
    re-derived. No-op (returns model unchanged) when deps.context_mapper is None."""
    if deps.context_mapper is None:
        return model
    new_model = model.model_copy(deep=True)                       # fix #6 (purity)
    with _optional_stage("context_mapper"):                        # fix #10 (telemetry)
        try:
            cmap = deps.context_mapper(new_model, scout, feedback)
        except ContextMapperError as exc:
            # fix: non-fatal-but-loud; keep text-scan baseline deps
            print(f"  ⚠️  context-mapper failed: {exc}; keeping baseline allowed_dependencies")
            new_model.context_map = ContextMap(model_id="unknown", error=str(exc))
            return new_model
    valid_names = {bc.context_name for bc in new_model.bounded_contexts}
    derived, warnings = derive_allowed_dependencies(cmap, valid_names)  # fix #3/#4
    cmap.warnings.extend(warnings)
    new_model.context_map = cmap
    # A is AUTHORITATIVE on success: every context's allowed_dependencies is the
    # derived projection (None when it has no edges). Do NOT guard on `is not None`
    # — that would let SEPARATE_WAYS / upstream-only / unmapped contexts keep a
    # stale text-scan baseline, contradicting the map and defeating Separate Ways.
    for bc in new_model.bounded_contexts:
        deps_for_ctx = derived.get(bc.context_name)
        bc.allowed_dependencies = sorted(deps_for_ctx) if deps_for_ctx else None
    return new_model
```

Call site: inside `_generate_once`, after `model = deps.synthesizer(...)` (line 465) and the empty-model guard (lines 472–476), before `return model, arch, refined_specialist`:
```python
    model = _apply_context_map(model, deps, scout)
    return model, arch, refined_specialist
```
This means A runs on **every** generation pass — cycle 0, the structural-regen path, AND the non-critic single-pass path (`run_pipeline` when `deps.critic is None`). A is therefore valuable independently of the Critic loop.

#### §4.3 Purity contract
`_apply_context_map` never mutates its input `model`; it deep-copies first. This is mandatory because the loop's `best_model` may alias the current `model` (loop.py:69) (fix #6).

#### §4.4 Derivation + validation (`derive.py`, pure)
```
derive_allowed_dependencies(cmap, valid_names) -> (deps: Dict[str,List[str]], warnings):
  1. Drop relationships where context_a or context_b ∉ valid_names → warning per drop (fix #3).
  2. For each surviving relationship, add edges:
       directional (CUSTOMER_SUPPLIER/CONFORMIST/ACL/OHS/PUBLISHED_LANGUAGE):
            downstream → upstream     (downstream = the non-upstream member)
       mutual (PARTNERSHIP/SHARED_KERNEL):
            a → b AND b → a
       SEPARATE_WAYS / BIG_BALL_OF_MUD:
            no edges
  3. Cycle detection, mutual-exempt (fix #4): build the directional-only graph
     (exclude edges contributed by mutual relationships). Detect cycles (same
     WHITE/GRAY/BLACK DFS as check_d11). Any cycle found → warning (NOT a hard
     fail; edges are kept). Mutual 2-cycles never reach this graph.
  4. Return (deps, warnings).
```
Rationale for warn-not-fail: production `verifier_fn` never runs D11 on derived deps (D11 is unwired), so there is no hard gate to satisfy; surfacing a non-mutual cycle loudly (warnings + Critic's BBoM finding + AST cross_check_diff) is the correct "explicit but non-fatal" treatment per AGENTS.md.

#### §4.5 `architect.py` factory + wiring
Add `_build_context_mapper_fn` (mirror `_build_critic_fn`, architect.py:1002), gated by `DDD_CONTEXT_MAP` (default ON; `0/false/no/off` to opt out):
```python
def _build_context_mapper_fn(self):
    if os.getenv("DDD_CONTEXT_MAP", "1").strip().lower() in ("0","false","no","off"):
        return None
    from core.context_mapper import run_context_mapper
    cm_cfg = stage_config("ContextMapper")
    def context_mapper_fn(model, scout, feedback):
        return run_context_mapper(model, scout, feedback, client=self.client, stage_cfg=cm_cfg)
    return context_mapper_fn
```
Wire `context_mapper=self._build_context_mapper_fn()` into `PipelineDeps(...)` (architect.py:1200).

### §5 Critic relationship-awareness (`core/critic/`)

#### §5.1 New finding types (BOTH files)
Add to `types.py:ProposedFinding.finding_type` AND `schemas.py:CritiqueFinding.finding_type`:
`WRONG_RELATIONSHIP_TYPE`, `ILLEGAL_DEPENDENCY`, `MISSING_RELATIONSHIP`. `target_ref` convention: `relationship:<A>-><B>`.

#### §5.2 Critic input + targets
- `prompt.py:_serialize_model` — add a `context_map` block (relationships: a/b/type/upstream) so the Critic can judge it; add a relationship-review step to `_INSTRUCTIONS`.
- `critic.py:_map_finding` — for relationship finding types, validate `target_ref` by **parsing the pair and checking both names are valid context names** (NOT membership in a `context:`/`entity:` set). This lets `MISSING_RELATIONSHIP` (pair not yet in the map) survive instead of being dropped as malformed. Evidence trim unchanged.

#### §5.3 Routing (`routing.py`)
- Add `_RELATIONSHIP = {"WRONG_RELATIONSHIP_TYPE","ILLEGAL_DEPENDENCY","MISSING_RELATIONSHIP"}`.
- `partition_findings` returns a **4-tuple** `(structural, content, relationship, advisory)`. Update its only caller (loop.py:85).
- Relationship findings need no `_CritiqueIssue` adapter — they pass straight into `run_context_mapper(..., feedback=relationship_findings)`.

#### §5.4 Loop (`loop.py`) revision-cycle control flow
```python
structural, content, relationship, _adv = partition_findings(report.findings)
if structural:
    new_model, arch, specialist = _generate_once(scout, deps, srs_path,
        architect_feedback=adapt_structural_to_issues(structural))
    # _generate_once already applied a fresh context_map (feedback=None)
elif content:
    specialist = deps.specialist_with_feedback(arch, scout, specialist,
        adapt_content_to_issues(content))
    new_model = deps.synthesizer(specialist)
    new_model = _apply_context_map(new_model, deps, scout)     # fresh map
else:
    new_model = model                                          # relationship-only base
# fix #5 — apply relationship feedback EVERY cycle, filtered to surviving contexts:
if relationship:
    survivors = {bc.context_name for bc in new_model.bounded_contexts}
    rel_live = [f for f in relationship if _relationship_pair_in(f, survivors)]
    if rel_live:
        new_model = _apply_context_map(new_model, deps, scout, feedback=rel_live)
new_report = deps.critic(new_model, scout, history)
```
`_generate_once` / `_apply_context_map` imported lazily in loop.py (as `_generate_once` already is).

#### §5.5 Flap signature + Reflexion (fixes #7, #8)
- `findings_signature` (loop.py:29): canonicalize relationship target_refs — `relationship:B->A` ≡ `relationship:A->B` (sort the pair) so reversed pairs don't evade flap detection. `context:`/`entity:` targets unchanged (no behavior change for C).
- `model_diff_summary` (routing.py:95): append context-map deltas — relationships added/removed (canonical pair), type/upstream changes — so the relationship-only cycle no longer reports "no structural change" and the Reflexion history stays truthful.

### §6 AST reconciliation (`core/AST/import_graph.py`) — fix #1

`apply_import_topology_to_model(model_data, ...)`: when A **authoritatively produced** the deps — i.e. `model_data.get("context_map")` is present **and its `error` is None** — switch to **diagnostics-only**:
- Never execute the `if not existing: context["allowed_dependencies"] = sorted(derived_set)` auto-fill branch.
- Still compute `derived` + `cross_check_diff` for **all** contexts (including those where A produced empty deps — record the AST-observed imports as `extra_in_derived` so a missed relationship is visible).
- Return the same diagnostics dict; `auto_populated` stays `[]` in this mode.

This preserves Separate Ways / Big Ball of Mud enforcement (intentional-empty deps are not silently repopulated) while keeping the AST cross-check as a review signal. **When A failed (`context_map.error` set) or was disabled (`context_map is None`), legacy auto-fill behavior is retained** — the text-scan baseline gaps are filled from AST as before (no regression). `context_map` is already serialized into `model_data` via `model.model_dump(...)`, so no new plumbing is needed (read `model_data["context_map"]["error"]`).

### §7 Failure modes (AGENTS.md explicit-failure)
- A `json_failed` after retries → `ContextMapperError` → `_apply_context_map` records `context_map.error`, logs loudly, **keeps the text-scan baseline** `allowed_dependencies` (no regression, no silent None). Pipeline continues; model still valid.
- Invalid-name relationships / non-mutual cycles → recorded in `context_map.warnings` (loud), edges handled per §4.4.
- Per-relationship schema-validation failure → dropped + counted (malformed), like the Critic.

### §8 Backward compatibility
`context_map` is `Optional`, default `None`; existing `model.json` deserializes unchanged. New finding types are additive enum members. `allowed_dependencies` keeps its `List[str]` shape.

---

## Scope

**IN:** schema (§1), `core/context_mapper/` package (§2), derivation+validation (§4.4), pipeline `_apply_context_map` + `_generate_once` slot + `PipelineDeps` (§4), `architect.py` factory+wiring (§4.5), `configs/models.py` stage (§3.2), Critic relationship-awareness (§5), AST diagnostics-only mode (§6), all 10 review fixes, tests.

**OUT (follow-up WPs):**
- Teaching V4 `validator.py` to reason by relationship *type* (e.g. ACL-mediated imports legal, Conformist one-way). This WP only improves the *flat* `allowed_dependencies` V4 already reads (correct directionality + Separate Ways→empty is already a real gain).
- Wiring D11 into the production `verifier_fn` (currently unwired; would need mutual-exemption first).
- UI/UX surfacing of the context map (separate WP per memory `project-ui-ux-agent-followup`).
- Tracking the full flap-history (not just previous cycle) — affects C equally; out of A's scope.

---

## Testing strategy (TDD; subagents write tests first)

- **schema:** model_validator accepts/rejects each type↔upstream combination; a==b rejected; round-trip serialize/deserialize; old model.json (no context_map) loads.
- **derive.py (pure, high-value):** each of the 9 types → correct edges; mutual → both edges + cycle-exempt; directional chain A→B→C→A → warning, edges kept; invalid context name → dropped + warning; Separate Ways → empty.
- **mapper.py:** maps ProposedRelationship→ContextRelationship; evidence trimmed to scout indices ∪ {-1} + de-duped; `json_failed` → `ContextMapperError`; feedback path builds remap prompt.
- **_apply_context_map:** purity (input model unchanged — assert `model.context_map is None` after); no-op when `context_mapper is None`; overwrites baseline deps; failure → baseline kept + error recorded; runs inside stage scope (telemetry recorded).
- **critic relationship findings:** `_map_finding` keeps `relationship:` findings with valid context names; drops ones with unknown names; `MISSING_RELATIONSHIP` on absent pair survives.
- **routing:** `partition_findings` 4-tuple buckets relationship types correctly; `findings_signature` canonicalizes reversed pairs; `model_diff_summary` reports relationship deltas.
- **loop:** relationship-only cycle re-maps with feedback (no architect/specialist rerun); structural+relationship cycle applies both; keep-best uses deep-copied models; flap on oscillating relationship terminates.
- **AST:** `apply_import_topology_to_model` with `context_map` present → no auto-fill, `auto_populated == []`, `cross_check_diff` still populated; without context_map → unchanged behavior.
- **e2e:** `run_pipeline` with critic+context_mapper deps → model carries `context_map` + derived deps; `DDD_CONTEXT_MAP=0` → `context_map is None`, baseline deps; `DDD_CRITIC_LOOP=0` + context_mapper on → map present, no loop.

Gate: `pytest -m "not integration" -q` + `pyright` (0 prod errors). Run pytest via `.venv/bin/python -m pytest` or `pytest` directly (system Python 3.13; `python3` is Homebrew 3.14 without pytest).

---

## Cross-references
- C — Holistic Critic: `[[WP-critic-holistic]]`, spec `docs/superpowers/specs/2026-05-25-holistic-critic-design.md`.
- Handoff: `.planning/HANDOFF-2026-05-25-context-mapper.md`.
- AGENTS.md engineering charter; D1 6-model lock (`core/llm/registry.py`).
