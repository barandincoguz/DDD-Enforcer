# HANDOFF — A: Context-Mapper (next LLM-augmentation agent)

**Date:** 2026-05-25
**Repo HEAD at handoff:** `eca34de` (branch `main`, C shipped + merged)
**Why this doc:** resume after `/clear` with zero re-exploration. Everything to execute A is here.
**Predecessor:** `.planning/HANDOFF-2026-05-25-llm-augmentation.md` (the C+A handoff; C is now done — this doc supersedes it for A).

---

## TL;DR — what's next

Build **A — Context-Mapper**, the second LLM design-judgment agent, via the full `brainstorming → spec → plan → SDD` cycle (same flow that shipped C). Goal: **improve the product** (paper de-scoped for this work, per the user 2026-05-25).

**C — Holistic Critic is DONE** (merged `main`, default ON). The generation pipeline now is:
```
Scout → Architect → Specialist → Verifier → Refiner → Synthesizer → [Critic critique→revise loop ×≤N] → DomainModel
```
A is the next layer.

---

## ✅ C status (so you don't re-investigate)

- Shipped 2026-05-25: 12 commits (`eea2cb2`..`d5464a6`) + default-ON flip (`eca34de`). All via SDD, two-stage-reviewed, final review READY TO MERGE. **763 unit tests pass, pyright 0.**
- C = a **pure evaluator** inside a **bounded critique→revise loop** (Topology A). It judges DDD design quality; high/med findings route back to producers (`architect_with_feedback` / `specialist_with_feedback`) → re-synthesize → keep-best/flap/Reflexion. Never mutates the model directly.
- Package: `core/critic/{errors,types,routing,prompt,critic,loop}.py`. Schema: `core/schemas.py` `CritiqueFinding`/`CriticReport`/`CriticLoopTrace` + `DomainModel.critic_report`. Loop driver: `core/critic/loop.py:run_critique_loop`. Dispatch: `core/orchestration/pipeline.py:run_pipeline` → `_generate_once` (single pass) vs `run_critique_loop` (when `deps.critic` set).
- Flags: `DDD_CRITIC_LOOP` (default ON; `0/false/off` to disable), `DDD_CRITIC_MAX_CYCLES` (default 3).
- Spec: `docs/superpowers/specs/2026-05-25-holistic-critic-design.md`. Plan: `docs/superpowers/plans/2026-05-25-holistic-critic-loop.md`.

---

## ⛔ DECISION GATE for A — resolve in brainstorming before code

1. **Run mode:** does A run as a **one-shot enrichment** (after Architect+Specialist, before/inside Synthesizer, computes the context map once) OR as **another producer inside the C critique loop** (so the Critic can flag a bad relationship type and trigger A to re-map)? One-shot is simpler and likely right for v1; loop-participation is the richer option. Decide in brainstorming.
2. **Schema shape:** new top-level `DomainModel.context_map` (list of typed relationship objects) vs enriching the existing flat `BoundedContext.allowed_dependencies: Optional[List[str]]` into richer objects. New field is cleaner + backward-compatible (mirrors how C added `critic_report`); enriching `allowed_dependencies` risks breaking V4's current consumer. **Recommend a new `context_map` field that V4 reads, leaving `allowed_dependencies` intact (or derived from it).**
3. **Reconcile LLM vs code-derived deps:** `allowed_dependencies` today is text-scan + AST import graph. A's LLM map is intent-level. Diff or merge? (Brainstorm fork.)

Product-not-paper framing is already settled (same as C). Accuracy-over-cost holds.

---

## What A is — Context-Mapper

- **Problem it fixes:** strategic DDD is missing. Today contexts only carry a flat `allowed_dependencies` string list (no relationship semantics). A adds a real **context map**.
- **Slot:** after Architect+Specialist (contexts + ubiquitous language known); can run pre- or post-Synthesizer. (See decision gate #1.)
- **Input:** all bounded contexts + their ubiquitous language (+ optionally the AST import topology that currently feeds `allowed_dependencies`).
- **Output:** for each related context **pair**: relationship **type** + integration **direction** + **rationale** (+ grounding). Standard DDD strategic patterns to choose from:
  - **Partnership**, **Shared Kernel**, **Customer-Supplier**, **Conformist**, **Anti-Corruption Layer (ACL)**, **Open Host Service (OHS)**, **Published Language**, **Separate Ways**, **Big Ball of Mud**.
- **Feeds:** **V4 context-boundary enforcement** — `core/llm/validator.py:522` (`ContextBoundaryViolation` checks code imports against `allowed_dependencies`, populated at `:562` via `ctx.get("allowed_dependencies", [])`). A richer typed map lets V4 reason about *legal* vs *illegal* cross-context calls by relationship type (e.g. a Conformist may import upstream; Separate Ways may not).
- **Cost:** ~1 whole-model LLM call (one-shot) or per-loop-cycle (if loop-participant).
- **LLM:** use `get_client_for_model` / the generation stage config (G1 `gemini-3.1-pro-preview`, temp 0.05, seed 42), `structured_output` with a Pydantic response schema. Register a `"ContextMapper"` stage in `configs/models.py:STAGE_TO_GROUP` → `"domain_extraction"` (mirror how C added `"Critic"`).

---

## Lessons from shipping C (apply these)

1. **Adversarial peer review pays.** Codex (`codex exec`, model gpt-5.5, xhigh) reviewing the C design caught: (a) a real routing bug (the `specialist:<ctx>` feedback-prefix requirement), (b) factual errors (D8 auto-heals not hard-fails; wired Verifier is D1–D5 only; `confidence` already double-written by AST). Run the A design past Codex the same way. **agent-relay panel is NOT active** (`AGENT_RELAY_NAME` unset) → `agent-ask`/`/ask-codex` fail; use the **codex-plugin-cc** path / `codex exec --sandbox read-only -` with the prompt on stdin.
2. **The brainstorming forks were the value.** For C: apply-semantics, auto-apply line, schema placement, slot, failure mode. For A expect: run-mode, schema shape, relationship taxonomy scope, LLM-vs-AST reconciliation, V4 wiring.
3. **SDD with opus subagents worked well.** 11 TDD tasks, fresh implementer + spec reviewer + quality reviewer per task, fix loop. The quality review caught a genuine non-fatal-failure bug in the loop. User authorized **opus for every subagent + token spend** for this work — keep doing that.
4. **Pure-evaluator / mutation-via-existing-path** is the winning pattern (Codex endorsed it). If A needs to *change* deps, prefer feeding the existing synthesizer/validator path over a new patch subsystem.
5. **IDE pyright squiggles are false** (env can't resolve pydantic/pytest/core.*; `tests/` excluded from the gate). Trust the subagent's `pyright <prodfile>` run + passing tests, per memory `reference_extension_tsserver_noise`.

---

## Architecture quick-reference (distilled — no re-exploration)

**Two halves:** `extension/src/extension.ts` (VS Code, spawns backend, SSE) ↔ `extension/backend/main.py` (FastAPI). Backend writes `domain/model.json`.

**Pipeline** — orchestrator `core/orchestration/pipeline.py`, agents `core/architect.py`:

| # | Stage | LLM? | Where |
|---|---|---|---|
| 1 | Scout | ❌ chunker | `core/scout/chunking.py` |
| 2 | Architect | ✅ | `architect.py:identify_contexts` (+`..._with_feedback`) |
| 3 | Specialist | ✅ N | `architect.py:extract_per_context_details` (+`_specialist_with_feedback`) |
| 4 | Verifier | ❌ D1–D5 | `architect.py:verifier_fn` (D9/S3/D11 exist but UNWIRED) |
| 5 | Refiner | ❌ ≤2 | `core/refiner/loop.py:refine_until_clean` |
| 6 | Synthesizer | ✅ narrow + det merge | `core/synthesizer/` (D6/D7 hard-fail, **D8 auto-heals**) |
| 7 | **Critic loop** | ✅ ≤N | `core/critic/loop.py` (NEW, C) |

**`_generate_once`** (`pipeline.py`) = the single-pass body (Architect-rerun + Specialist-refine + Synthesizer), returns `(model, arch, refined_specialist)`. `run_pipeline` dispatches: `critic=None` → one pass; else `run_critique_loop`. **A likely adds another `PipelineDeps.*` callable + a slot in `_generate_once` or the loop** (mirror the critic wiring).

**Schemas** (`core/schemas.py`): `DomainModel{project_name, project_metadata, bounded_contexts[], global_rules, critic_report?}`; `BoundedContext{context_name, description, allowed_dependencies: Optional[List[str]], supporting_sentence_ids, business_rules, ubiquitous_language}`; `ubiquitous_language{entities, value_objects, services, aggregates, domain_events, repositories, factories, anti_corruption_layers, specifications}`.

**Grounding thread (keep intact):** Scout `index` → Architect `supporting_sentence_ids` → Specialist `evidence_sentence_indices` → `BoundedContext.supporting_sentence_ids`. Any A additions carry grounding or an AST/`-1` sentinel.

**LLM layer:** `core/llm/` — `get_client_for_model(model_id)`, `structured_output(messages, schema, model, temperature=, seed=) -> LLMResponse` (sets `.json_failed`, doesn't raise; `.parsed` on success). Stage→model via `configs/models.py:STAGE_TO_GROUP` + `stage_config(stage)`. Never import `google.genai`/`openai` outside `core/llm/gemini.py`/`ollama.py`.

**V4 (A's downstream consumer):** `core/llm/validator.py:522` ContextBoundaryViolation vs `allowed_dependencies` (`:562`).

---

## Constraints (do not violate)

- AGENTS.md: smallest correct change, no speculative generalization, **explicit failure** (no bare except / no silent degradation — record + raise or non-fatal-but-loud), facade pattern (import `core.parser`/`core.llm`, not internals), modularity ~500 effective LOC.
- D1 6-model lock + `_RUNTIME_FALLBACKS={}` empty (paper hygiene; still respected even though paper is de-scoped now).
- Communication Turkish; code/comments English. SDD default (memory `feedback-sdd-default`); opus subagents + token spend authorized.
- Backward-compat: new schema fields Optional/`default_factory`; existing `model.json` must still deserialize.

---

## Repo state + how to verify

- HEAD `eca34de` on `main`. Working tree: only runtime artifacts dirty (`validation_metrics_report.json`, `core/AST/intermediate/`, untracked `core/intermediate/`, `domain/model.PRE-FRESH-RUN-BACKUP.json`, untracked `AGENTS.md`) — leave them; never stage them with code.
- Gate (from `extension/backend/`): `pytest -m "not integration" -q` (763 pass) + `pyright` (0 prod errors). **Run `pytest` directly — Python 3.13; `python3` is Homebrew 3.14 with no pytest. `.venv/bin/python` is 3.13.**
- **E2E recipe** (user has ~20 SRS): `cd extension/backend` → put SRS in `inputs/` → `uvicorn main:app` (lifespan auto-generates) OR `POST /generate-model-stream {file_paths, output_path}`. Critic loop now runs by default; inspect `domain/model.json` `critic_report` + `core/intermediate/{ts}_*` + `runs/`. Set `DDD_CRITIC_LOOP=0` to compare without the loop.

## Pointers
- C spec/plan: `docs/superpowers/specs/2026-05-25-holistic-critic-design.md`, `docs/superpowers/plans/2026-05-25-holistic-critic-loop.md`.
- Prior C+A handoff (background): `.planning/HANDOFF-2026-05-25-llm-augmentation.md`.
- Orientation: `todos/AGENT_QUICKSTART.md`, `todos/MASTER_PLAN.md` (D1–D7), `development_docs/INDEX.md`.
- UI/UX follow-up (after A): memory `project-ui-ux-agent-followup`.

## First actions after /clear
1. Read this file.
2. Confirm with the user the A **run-mode** (one-shot vs loop-participant) + **schema** (new `context_map` vs enrich `allowed_dependencies`) — the decision gate above.
3. `superpowers:brainstorming` for **A — Context-Mapper** (forks: run-mode, schema, relationship taxonomy, LLM-vs-AST reconciliation, V4 wiring). Run the resulting design past **Codex** (`codex exec`, gpt-5.5 xhigh) for adversarial review before finalizing the spec.
4. spec → plan → SDD (opus subagents, two-stage review), then merge + memory update + a `development_docs/` doc.
