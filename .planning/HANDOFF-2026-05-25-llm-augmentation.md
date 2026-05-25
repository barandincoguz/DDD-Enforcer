# HANDOFF — LLM-Augmentation of the Domain-Model Pipeline (C → A)

**Date:** 2026-05-25
**Repo HEAD at handoff:** `bd8d5eb` (branch `main`, synced with `origin/main`)
**Why this doc:** resume after `/clear` with zero re-exploration. Everything needed to execute is here.

---

## TL;DR — what we're doing next

Add **LLM design-judgment** to the domain-model pipeline, in this order:
1. **C — Holistic Critic** (FIRST, via brainstorming→spec→plan→SDD)
2. **A — Context-Mapper** (SECOND, same flow)

Goal reframing (the user's words): the domain model is a **DDD rule-book for enforcement, NOT a hard-error generator**. So new LLM stages should *reason and propose*, not rigidly gate. Deterministic checks stay as cheap grounding/shape guardrails; DDD *quality judgment* moves to LLM.

---

## ⛔ DECISION GATE — resolve BEFORE any code

These proposals change the architecture the EMSE paper measures (D1–D7 locked, N=10 baseline, RQ1–RQ4 describe the *current* pipeline). Must pick:

- **(i) Part of the EMSE-submitted system** → adding C/A means re-running the whole experimental matrix (6 models × N runs). Expensive but the paper studies the improved system.
- **(ii) Post-submission v2 / enhanced mode** → build now, keep out of the studied pipeline (e.g. behind a flag / separate mode), paper unchanged.

**Do not start brainstorming C until the user answers (i) vs (ii).** It changes scope, where C/A slot (studied path vs optional mode), and whether reproducibility (seed=42, single-model-per-run) must hold.

---

## Background — why C and A (the analysis already done)

The pipeline does **extraction** (LLM) + **validation** (deterministic) but lacks the middle layer: **design reasoning** (LLM). No agent reads the assembled model and asks "is this good DDD?" The rule-book quality lives in that missing layer.

Missing DDD richness in `model.json` today:
- **Context map** (strategic DDD): only a flat `allowed_dependencies` list (text-scan + AST import graph). No relationship *type* (ACL / Shared Kernel / Conformist / Customer-Supplier / OHS / Published Language). → **A fixes this.**
- **Aggregate invariants**: aggregates are just name+members (often vacuous → S3 WARN). No consistency-boundary/invariant reasoning. (Future "B — Aggregate Designer", not in this handoff's scope.)
- **No holistic quality pass**: Verifier is purely deterministic D1–D11, some HARD-FAIL (`ArchitectGroundingError` can crash). → **C fixes the philosophy: advisor, not gate.**

### C — Holistic Critic (build first)
- **Slot:** after Synthesizer produces the full `DomainModel`, before final persist.
- **Input:** the complete assembled `DomainModel` (+ optionally Scout sentences for grounding).
- **Output:** a critique with *proposed revisions* — misplaced entities, contexts that should merge/split, anemic models, missing aggregates, naming smells — each with rationale. NOT pass/fail.
- **Apply path:** Critic *suggests* → a deterministic guard checks suggestions don't break grounding/shape → Refiner applies. Keep deterministic Verifier as guardrail underneath.
- **Feeds:** overall rule-book coherence.
- **Cost:** 1 whole-model LLM call. Reproducible with seed=42, single model (G1 `gemini-3.1-pro-preview`).

### A — Context-Mapper (build second)
- **Slot:** after Architect+Specialist (contexts + entities known); can run pre- or post-Synthesizer.
- **Input:** all bounded contexts + their ubiquitous language.
- **Output:** a **context map** — for each related context pair: relationship type + integration direction + rationale. Replaces/enriches flat `allowed_dependencies`.
- **Feeds:** V4 boundary-enforcement rules.
- **Cost:** 1 whole-model LLM call.
- **Schema impact:** likely a new `context_map` field on `DomainModel` (or richer `allowed_dependencies` objects) — needs schema design in its spec.

---

## Process to follow (per the user's locked workflow)

For **each** of C then A, run the full superpowers cycle (the user explicitly wants this):
1. `superpowers:brainstorming` → design + AskUserQuestion on the real forks → write spec to `docs/superpowers/specs/`.
2. User reviews spec.
3. `superpowers:writing-plans` → bite-sized TDD plan to `docs/superpowers/plans/`.
4. `superpowers:subagent-driven-development` → fresh implementer subagent per task + **two-stage review** (spec reviewer THEN code-quality reviewer) + fix loops. Final holistic review at end.

TDD/commit conventions (repo): tests in `extension/backend/tests/` (pytest, `pytest -m "not integration"`), atomic Conventional Commits, trailer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`. Backend gate: `cd extension/backend && pytest -m "not integration"` + `pyright` (blocking). Python 3.12 venv.

Brainstorming forks to anticipate for C: (1) does Critic *edit* the model directly or only *emit suggestions* a deterministic step applies? (2) hard-fail vs advisory on critique findings; (3) how to bound LLM over-editing (grounding guard); (4) new schema fields for advisory/uncertain rules.
For A: (1) context_map schema shape; (2) which DDD relationship taxonomy (the 7–8 standard patterns); (3) reconcile with existing AST import-topology `allowed_dependencies` (LLM vs code-derived — diff or merge?).

---

## Architecture quick-reference (distilled — no re-exploration needed)

**Two halves:** `extension/src/extension.ts` (VS Code, spawns backend, SSE progress) ↔ `extension/backend/main.py` (FastAPI). Backend writes `domain/model.json`.

**The multi-agent pipeline** — orchestrator `extension/backend/core/orchestration/pipeline.py`, agents in `core/architect.py`. 6 components, only 3 make LLM calls:

| # | Component | LLM? | File | Output type (pipeline_contracts.py) |
|---|---|---|---|---|
| 1 | Scout | ❌ deterministic chunker | `core/scout/chunking.py` | `ScoutOutput{sentences:[{index,text,section}]}` |
| 2 | Architect | ✅ 1 call | `core/architect.py:identify_contexts` | `ArchitectOutput{contexts:[{context_name, supporting_sentence_ids}]}` |
| 3 | Specialist | ✅ N calls (per context) | `core/architect.py:extract_per_context_details` | `List[SpecialistAnalysis]` |
| 4 | Verifier | ❌ deterministic D1–D5 (+D9/S3/D11) | `core/verifier/checks_deterministic.py` | `VerifierResult{ok, issues[]}` |
| 5 | Refiner | ❌ loop (≤2 cycles, flap-detect) | `core/refiner/loop.py` | re-runs Architect(max1)/Specialist(targeted) |
| 6 | Synthesizer | ✅ narrow (synonyms only) + det merge | `core/synthesizer/{merge,enrich}.py` | final `DomainModel` |

Post-synthesis hard-fail invariants D6/D7/D8 in `core/verifier/checks_semantic_d6_d7_d8.py` (merge-bug detectors, NOT refiner-loopable).

**Where new agents slot:** C = after step 6 (reads `DomainModel`). A = after step 3 or after step 6.

**Grounding thread (keep intact):** Scout `index` → Architect `supporting_sentence_ids` → Specialist `evidence_sentence_indices` → Synthesizer `BoundedContext.supporting_sentence_ids`. Verifier D1 enforces subset. Any new agent's additions MUST carry grounding or be marked AST/`-1` sentinel.

**LLM layer:** `core/llm/` — `get_client_for_model(model_id)` routes to GeminiClient/OllamaClient. Generation model = G1 `gemini-3.1-pro-preview`, temp 0.05, seed 42 (`configs/models.py` STAGE_TO_GROUP). `structured_output(messages, schema, model)` enforces Pydantic schema. New LLM agents should use `get_client_for_model` + a Pydantic response schema, never raw `google.genai`/`openai`.

**Schemas:** `core/schemas.py` — `DomainModel{project_name, project_metadata, bounded_contexts[], global_rules}`; `BoundedContext{context_name, description, allowed_dependencies, supporting_sentence_ids, business_rules, ubiquitous_language}`; `ubiquitous_language{entities, value_objects, services, aggregates, domain_events, repositories, factories, anti_corruption_layers, specifications}`.

**Validator rules these feed (enforcement side, `core/llm/validator.py`):** V1/V2 naming+synonym, V4 context boundary, V5 aggregate boundary, V6 domain event. A → V4; C → overall coherence; (future B → V5).

**Rate limiting:** `DomainArchitect.min_delay` 6s default (`DDD_MIN_DELAY_SECONDS`). Each LLM stage dumps `core/intermediate/{ts}_{stage}.json`.

---

## Constraints (do not violate)

- **D1 6-model lock**, **D6 RQ5 dropped**, accuracy-over-cost (no tier downgrades). `_RUNTIME_FALLBACKS={}` must stay empty.
- Reproducibility: seed=42, single-model-per-run for paper paths; `run_manifest` SHA-256 provenance.
- AGENTS.md: smallest correct change, no speculative generalization, explicit failure (no silent fallback), facade pattern (import `core.parser`/`core.llm`, not internals).
- Communication Turkish; code/comments English. SDD default for cross-stage work (memory `feedback-sdd-default`).

---

## Repo state + how to verify

- HEAD `bd8d5eb`, `main` == `origin/main`. Working tree: only runtime artifacts dirty (`validation_metrics_report.json`, `core/intermediate/`, `core/AST/intermediate/`, `model.PRE-FRESH-RUN-BACKUP.json`) — leave them.
- Faz0 remediation **T1–T15 COMPLETE** (incl. T10/T11 fingerprint two-phase rewrite). Extension suite 147 passing. Backend: `cd extension/backend && pytest -m "not integration"`.
- **E2E test recipe** (user has 20 SRS, wants to run one): `cd extension/backend && source .venv/bin/activate` (3.12; repair per CLAUDE.md if broken) → put SRS in `inputs/` → `uvicorn main:app` (lifespan auto-generates if no model) OR `POST /generate-model-stream {file_paths, output_path}`. Inspect `domain/model.json` + `core/intermediate/{ts}_2_architect.json`/`_3_specialist.json` + `runs/`. Note: AST `sources` only fill if `WORKSPACE_PATH` set + workspace has `.py`.

## Pointers
- Fingerprint work: `docs/superpowers/specs/2026-05-25-fingerprint-indent-string-hardening-design.md` + `docs/superpowers/plans/2026-05-25-fingerprint-indent-string-hardening.md`.
- Faz0 remediation: `.planning/plans/2026-05-25-faz0-refactor-remediation.md` (completion banner at top).
- Project orientation: `todos/AGENT_QUICKSTART.md`, `todos/MASTER_PLAN.md` (D1–D7), `development_docs/INDEX.md`.

## First actions after /clear
1. Read this file.
2. Ask the user the **(i) vs (ii) decision** (EMSE-submitted vs v2). Do not skip.
3. On answer → `superpowers:brainstorming` for **C — Holistic Critic**.
4. After C ships → repeat for **A — Context-Mapper**.
