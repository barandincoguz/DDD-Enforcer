# A — Context-Mapper (LLM strategic-DDD context map)

**Status:** SHIPPED 2026-05-26 (merged to `main`, fast-forward)
**Branch:** `feat/context-mapper` (merged + deleted)
**Commit range:** `0d2f542`..`deccd26` (19 commits: spec + plan + 14 TDD tasks + 3 in-flight fixes)
**Spec:** `docs/superpowers/specs/2026-05-25-context-mapper-design.md`
**Plan:** `docs/superpowers/plans/2026-05-25-context-mapper.md`
**Predecessor:** C — Holistic Critic [[WP-critic-holistic]] (placeholder; C shipped `eea2cb2`..`eca34de`, default ON)

## TL;DR

Adds the second LLM design-judgment agent, **Context-Mapper (A)**, which produces a typed **DDD strategic context map** (`DomainModel.context_map`) — for related bounded-context pairs, one of 9 strategic patterns (Partnership, Shared Kernel, Customer-Supplier, Conformist, ACL, OHS, Published Language, Separate Ways, Big Ball of Mud) + direction + rationale + Scout grounding. The flat `allowed_dependencies` (consumed by V4 boundary enforcement + D5/D11) becomes a **derived projection** of the map, authoritative on success. A runs inside `_generate_once` (so on every generation pass, critic-loop or not) and participates in the Critic loop as a **Critic-driven producer**: three new relationship finding types route back to A's feedback re-map, which is the only way to correct a mislabel given A's near-deterministic decoding (temp 0.05, seed 42). Goal was **product quality** (`domain_model.json` richness), not paper data — the EMSE paper is de-scoped for this work.

50 new tests; suite 763 → 813; pyright 0 prod errors throughout.

## Motivation

Strategic DDD was absent from the model. `BoundedContext.allowed_dependencies` was a flat `List[str]` with no relationship semantics, populated by a **text-scan stub** (`core/synthesizer/enrich.py:_infer_and_enrich_dependencies`) whose own docstring promised *"ONE additional LLM call to disambiguate allowed_dependencies across contexts"* that was never built. A is that step, elevated to typed relationships. The product gain: V4 reasons about an intent-level context map instead of a regex heuristic, and the model carries the strategic layer a DDD practitioner expects.

## Architectural decisions

### D-A1 — new `context_map` field; `allowed_dependencies` derived from it
Chosen over enriching `allowed_dependencies` into richer objects in place. Rationale: V4 (`validator.py:562`), D5, and `import_graph.py` all assume `List[str]`; changing that type is a high-blast-radius break. A new `Optional[ContextMap]` field mirrors exactly how C added `critic_report` — backward-compatible (existing `model.json` deserializes), and the flat list is regenerated as a projection so all existing consumers keep working at higher quality. **Authoritative on success:** every context's `allowed_dependencies` is overwritten from the projection (None when it has no edges); the text-scan baseline survives only when A fails or is disabled.

### D-A2 — Critic-driven producer, not one-shot
A decodes near-deterministically (temp 0.05, seed 42). Re-running it on unchanged contexts yields the *same* relationship label — so a one-shot or "re-run-each-cycle" design **cannot self-correct a mislabel** (the single most likely error, e.g. tagging Conformist what is really ACL). The only mechanism in the repo that catches LLM design-quality errors is the Critic. So the Critic was made relationship-aware, and relationship findings route back to A's `_with_feedback` re-map — feedback in the prompt changes the input, breaking determinism productively. This is strictly richer than one-shot and was the user's explicit "make the model best, don't be lazy" call.

### D-A3 — full 9-pattern taxonomy incl. anti-patterns
Separate Ways is genuinely useful for enforcement (→ empty deps → V4 flags any cross-import); Big Ball of Mud is a flaggable smell that ties into the Critic relationship-awareness. Chosen over a constructive-only or minimal-directional set.

### Slot + reconcile
A runs in `_generate_once` after the Synthesizer assembles contexts, as a **pure** step (`_apply_context_map` deep-copies — the critic loop's `best_model` may alias the input). The existing AST `import_graph` cross-check is the reconciliation: A's LLM map is the authoritative side; AST records drift in `cross_check_diff` and — post-fix #1 — never auto-fills when the map is authoritative, so Separate Ways enforcement holds.

### Adversarial review (Codex gpt-5.5 xhigh) shaped the design before code
The design was reviewed by Codex before any implementation (verdict SHIP-WITH-FIXES). It caught 4 blockers the author missed: (#1) AST auto-fill silently re-populating Separate Ways empties → V4 bypass; (#2) A cannot see AST import-topology at generation time (enrichment runs after `analyze_document`) → A is intent-level only; (#3) production `verifier_fn` never validates the *derived* deps (D11 is unwired) → validation moved inside `_apply_context_map`; (#4) mutual relationships → 2-cycle → false D11 positive → mutual-exempt cycle detection. Plus 6 majors/minors (loop purity, feedback starvation, flap pair-canonicalization, Reflexion diff, evidence trim, stage telemetry). All 10 integrated into the spec before SDD.

## File-level changes

| File | Change |
|---|---|
| `core/schemas.py` | `ContextRelationship` (9-type enum + `@model_validator` for type↔upstream consistency), `ContextMap` (relationships/model_id/warnings/error), `DomainModel.context_map`; `CritiqueFinding.finding_type` += 3 relationship types |
| `core/context_mapper/derive.py` | Pure `derive_allowed_dependencies` — directional→downstream, mutual→both, none→nothing; drops unknown-name relationships (warn); mutual-exempt DFS cycle detection (warn, edges kept) |
| `core/context_mapper/mapper.py` | `run_context_mapper` — one `structured_output` call; maps `ProposedRelationship`→`ContextRelationship` with evidence trim (scout ∪ {-1}, dedupe) + drop-invalid; remap path on feedback; raises `ContextMapperError` on json_failed |
| `core/context_mapper/{types,errors,prompt,__init__}.py` | LLM-facing schema; `ContextMapperError` (str-coerced reason); intent-level prompt builders (taxonomy + contexts + Scout, **no AST**); facade |
| `core/orchestration/pipeline.py` | `ContextMapperFn` + `PipelineDeps.context_mapper`; pure `_apply_context_map` (deep-copy, stage-wrapped, derive+overwrite, failure→baseline+error); called at end of `_generate_once` |
| `core/architect.py` | `_build_context_mapper_fn` (gated by `DDD_CONTEXT_MAP`, default ON) wired into `PipelineDeps` |
| `configs/models.py` | `STAGE_TO_GROUP["ContextMapper"] = "domain_extraction"` (G1, temp 0.05, seed 42) |
| `core/critic/{types,critic,prompt,routing,loop}.py` | `ProposedFinding` += 3 types; `_map_finding` validates `relationship:A->B` by context name; prompt serializes `context_map` + relationship review step; `partition_findings` 4-way + `model_diff_summary` map deltas; loop relationship branch + every-cycle feedback + pair-canonical flap signature |
| `core/AST/import_graph.py` | `apply_import_topology_to_model` diagnostics-only when `context_map` present AND `error is None` (preserves Separate Ways enforcement); legacy auto-fill retained when map absent/failed |

## Methodology applied

Full `brainstorming → spec → Codex adversarial review → writing-plans → subagent-driven-development` cycle (same flow that shipped C). 14 TDD tasks, fresh implementer subagent per task (opus for correctness-critical: derive, mapper, pipeline slot, loop, AST; sonnet for the rest), each followed by a fresh combined spec+quality reviewer that re-ran tests/pyright and verified against the spec without trusting the implementer's report. The hardest tasks (loop, AST) got opus reviewers that empirically probed the integrated behavior.

**Two defects the SDD process caught at the seams** (not in the original 10):
1. **Authoritative-overwrite bug** (the orchestrator caught it overriding a passing review): the spec's `if dep_list is not None` guard let Separate Ways / upstream-only / unmapped contexts keep a stale text-scan baseline, contradicting the map and silently re-defeating Separate Ways at the derivation level. Fixed: success path overwrites *all* contexts. Spec §4.2 corrected to match.
2. **Failure-path crash** (surfaced as a "regression" when A went default-ON): the architect e2e tests use a MagicMock client; `ContextMapperError.reason` was a MagicMock → `ContextMap(error=...)` raised `ValidationError`, crashing the failure handler itself. Fixed: `ContextMapperError` coerces `reason` to `str`. This was a genuine robustness gap (the error path must never crash), not just a test artifact.

## Empirical results

- Tests: 763 → **813** (+50), 0 failed; integration suite deselected as usual.
- pyright: **0 prod errors / 0 warnings** whole-project throughout (tests/ excluded from the gate; editor import squiggles are the known-false IDE noise).
- No live LLM E2E run was performed (deferred — see limitations). All verification is unit/integration with injected fakes.

## Limitations + follow-ups

- **V4 type-awareness is OUT.** This WP only improves the *flat* `allowed_dependencies` V4 already reads (correct directionality + Separate Ways→empty). Teaching `validator.py` to reason by relationship *type* (e.g. ACL-mediated imports legal one-way) is a follow-up WP.
- **D11 is still unwired in production `verifier_fn`.** A's derived deps are validated inside `_apply_context_map` (name-drop + mutual-exempt cycle warning, non-fatal). Wiring D11 into the production verifier would need the mutual-exemption first.
- **Live E2E not run.** The user has ~20 SRS; recommended check: drop an SRS in `inputs/`, run the pipeline (critic+mapper default ON), inspect `domain/model.json` `context_map` + derived `allowed_dependencies` + `core/intermediate/` + `runs/`. `DDD_CONTEXT_MAP=0` and/or `DDD_CRITIC_LOOP=0` to compare.
- **Flap history is previous-cycle-only.** Tracking all previously-seen signatures (affects C equally) was left out of A's scope.
- **UI/UX surfacing** of the context map is a separate WP (memory `project-ui-ux-agent-followup`).
- **Cost:** A adds ~1 G1 call per generation pass + 1 per relationship-feedback critic cycle. Accuracy-over-cost holds; no tier downgrade considered.

## Cross-references
- C — Holistic Critic: [[WP-critic-holistic]] (spec `docs/superpowers/specs/2026-05-25-holistic-critic-design.md`); A reuses C's loop, routing-adapter pattern, and Reflexion memory.
- Handoff: `.planning/HANDOFF-2026-05-25-context-mapper.md`.
- Pipeline orchestration: [[WP-CORE-7-refiner-stage-aware]], [[WP-CORE-1-typed-pipeline]].
- AST import topology: WP-CORE-31b (`core/AST/import_graph.py`).
