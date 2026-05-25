# Design Spec — C: Holistic Critic + Active Critique Loop (Topology A)

**Date:** 2026-05-25
**Status:** DESIGN — approved in brainstorming; awaiting spec review before plan
**Feature:** A bounded **critique→revise loop** that improves the generated `DomainModel` by having an LLM **Critic (C)** judge DDD design quality each cycle and routing its findings back to the existing producer stages (Architect / Specialist) for revision, until convergence or a cycle cap.
**Goal:** **Improve the product** (the domain models the tool generates). Paper/EMSE concerns (reproducibility, ablation, single-model lock, matrix re-run) are **explicitly out of scope for this WP** — the user re-prioritized to product quality on 2026-05-25.
**Process:** `brainstorming` (done) → (this spec) → `writing-plans` → `subagent-driven-development`.

---

## 1. TL;DR

Today: extraction (LLM) + deterministic validation. No agent reasons about *DDD design quality*.

We add an **active critique loop** (Topology A):

```
Scout (once) → [ Architect → Specialist → Synthesizer → DomainModel
                 → C critiques → route findings → revise → re-critique ]×≤N
              → keep-best model + its critique report
```

C is a **pure evaluator** (it never edits the model). Revision happens by **feeding C's findings back into the existing producer stages** (`architect_with_feedback`, `specialist_with_feedback`) and **re-synthesizing** — so the model is always re-built through the Synthesizer, keeping the grounding thread and D6/D7/D8 invariants valid **by construction**. The loop is **bounded** (N=3 default), uses **keep-best** (return the best cycle, not the last), **flap-detection** (stop on oscillation), and **Reflexion memory** (each cycle sees prior findings + what changed). Agent techniques map on naturally: **CoT** in C's prompt, **ReAct/Ralph** as the loop itself.

This is exactly the "active mode" the prior adversarial review (Codex) endorsed as the *right* way to mutate — "critic_report → feedback → Architect/Specialist rerun → synthesize → existing verifier, **not** direct DomainModel patch." We are building that as the primary feature.

---

## 2. Motivation

- **Missing design-reasoning layer.** Extraction emits a model; the deterministic Verifier (D1–D5 in the wired pipeline) + synthesizer invariants (D6/D7 hard-fail, D8 auto-heal) enforce *shape/grounding*, not *DDD quality*.
- **Rule-book philosophy → active improvement.** The user wants C's judgment to actually *improve* the model, not just annotate it — via a bounded, controllable loop with good agent techniques (ReAct, CoT, Reflexion, Ralph-style bounded iteration).
- **Reuse over reinvention.** The producer-feedback hooks and a bounded-loop primitive already exist; Topology A is mostly orchestration + the critic.

---

## 3. Locked design decisions (with rationale)

**D1 — Topology A: route findings to producer stages, re-synthesize.**
C's findings are routed to the stage that *owns* the criticized thing; that stage re-runs with feedback; the model is re-synthesized. No post-synthesis patching.
*Rationale:* (a) holistic critique (the high-value kind: "contexts over-split", "model too anemic") is naturally answered by *re-derivation*, not surgical edits; (b) re-flowing through the Synthesizer re-applies grounding + D6/D7/D8 + Verifier **for free** — no new guard subsystem; (c) reuses `architect_with_feedback` / `specialist_with_feedback` (already shipped + tested); (d) matches the user's mental model ("the relevant agent reads the critique and revises"). A surgical typed-tool agent (the other candidate) is deferred to a possible v2 (§13).

**D2 — C is a pure evaluator (no model mutation).**
`run_critic(model, scout, history) → CriticReport`. It reads, reasons, reports. It never mutates the `DomainModel`.
*Rationale:* keeps the critic testable and side-effect-free; all mutation flows through the producer/synthesizer path where invariants hold.

**D3 — Bounded loop, default N=3 (`DDD_CRITIC_MAX_CYCLES`), with keep-best + flap-detect.**
The loop runs at most N revision cycles. It returns the **best** cycle's model (lowest severity score), not necessarily the last. It stops early on convergence, flap, or failure.
*Rationale:* unbounded loops are unsafe; keep-best protects against a bad final cycle (hill-climb, not random walk); flap-detect kills oscillation (cycle A merges, cycle B re-splits).

**D4 — Continue trigger = HIGH or MEDIUM findings; LOW = advisory only.**
A cycle triggers another revision iff the current report has ≥1 `high` or `medium` finding. `low` findings are recorded in the final report as advice but never drive a regeneration.
*Rationale:* spend the expensive loop on impactful issues; avoid nit-chasing + oscillation on cosmetic findings.

**D5 — Reflexion memory.**
Each cycle, C receives the current model **plus** a compact history: prior cycles' findings + a deterministic **diff summary** of what the producers changed (contexts added/removed/renamed, entities added/removed/moved). The producer feedback also carries "what was tried."
*Rationale:* prevents re-flagging already-fixed issues, speeds convergence, and is the agent-technique depth the user asked for.

**D6 — Agent techniques: CoT (prompt) + ReAct/Ralph (loop structure).**
C's prompt walks the model section-by-section (CoT) before emitting structured findings. The loop *is* ReAct (C reasons = thought, producer re-run = act, re-critique = observation) and Ralph (bounded brute iteration). No external agent framework is introduced — the project uses plain `structured_output` calls and that stays.
*Rationale:* deliver the techniques where they add value without new infrastructure.

**D7 — Failure: single mode, explicit-but-non-fatal.**
If C fails (unrecoverable LLM error / `json_failed` after retries) or a regeneration fails on any cycle, the loop **breaks** and returns the **best model so far** (cycle-0 model if it fails before any successful critique). The failure is recorded in `CriticReport` + `manifest.errors`.
*Rationale:* paper-mode hard-fail is gone (no paper constraint); the product must never lose a generated model over an advisory loop, but the failure must be loud (AGENTS.md: explicit failure, no silent degradation).

**D8 — Opt-in flag, default OFF during development.**
The loop runs only when `DDD_CRITIC_LOOP=1`. `critic=None` (the default) → today's exact behavior.
*Rationale:* ship safely behind a flag; flip the default to ON once validated on real SRS inputs. Backward-compat for all existing call-sites/tests is guaranteed by the `None` default.

**D9 — Generation model.**
C uses the generation group via `STAGE_TO_GROUP["Critic"]="domain_extraction"` (G1 `gemini-3.1-pro-preview`, temp/seed per group). Provenance recorded in `CriticReport`.
*Rationale:* a holistic design critique needs the strongest model; consistent with the other generation stages and accuracy-over-cost.

---

## 4. Architecture — new `core/critic/` package

| File | Responsibility |
|---|---|
| `core/critic/types.py` | LLM I/O schema (`CriticResponse{analysis, findings}`, `ProposedFinding`) + persisted schema (`CritiqueFinding`, `CriticReport`, `CriticLoopTrace`). |
| `core/critic/prompt.py` | Build the critique prompt: serialize model + Scout sentences + **Reflexion** history; instruct **CoT** reasoning before findings. |
| `core/critic/critic.py` | `run_critic(model, scout, history, *, model_id) → CriticReport` — one LLM call, map + validate findings, compute score. Pure (no mutation). |
| `core/critic/routing.py` | Partition findings into structural vs content; adapt `CritiqueFinding → producer-feedback issue` objects; pick regeneration path; compute the model diff summary for Reflexion. |
| `core/critic/loop.py` | `run_critique_loop(scout, deps, srs_path) → DomainModel` — bounded loop, keep-best, flap-detect, Reflexion threading, failure handling. |

Target ≪500 effective LOC per file; split by responsibility, not size.

---

## 5. Data flow

```
deps.scout(srs)                      # ONCE — deterministic chunking, reused every cycle
   │ scout: ScoutOutput
   ▼
cycle 0:  _generate_once(scout, deps)            → (model₀, arch₀, specialist₀)
          run_critic(model₀, scout, history=[])  → report₀  (CoT)
          best ← (model₀, report₀)
          history ← [(report₀, diff=∅)]
   │
   ▼  for cycle k = 1..N-1:
        if report_{k-1} has no HIGH/MEDIUM     → STOP (converged)
        if flap(report_{k-1}, report_{k-2})    → STOP (oscillation)
        structural, content ← partition(report_{k-1}.findings)   # HIGH/MED only
        if structural:
            model_k, arch_k, specialist_k ← _generate_once(scout, deps,
                                                architect_feedback=adapt(structural),
                                                specialist_feedback=adapt(content))
        else:   # content-only → cheaper targeted path, reuse prior architecture
            specialist_k ← deps.specialist_with_feedback(arch_{k-1}, scout,
                                                specialist_{k-1}, adapt(content))
            specialist_k ← refine_until_clean(specialist_k, verifier, ≤2)   # grounding safety
            model_k ← deps.synthesizer(specialist_k);  arch_k ← arch_{k-1}
        report_k ← run_critic(model_k, scout, history)            # Reflexion: sees history
        diff_k ← diff_summary(model_{k-1}, model_k)
        history.append((report_k, diff_k))
        if score(report_k) < score(best.report): best ← (model_k, report_k)
   │
   ▼
attach best.report → best.model.critic_report
return best.model      → (run_pipeline returns) → AST enrich → save model.json
```

`run_critic` LLM **input**: serialized bounded-contexts + ubiquitous-language summary + Scout sentences (for grounding `evidence_sentence_indices`) + Reflexion history (prior findings + diffs).

---

## 6. Schema changes (`core/schemas.py`) — additive, backward-compatible

```python
class CritiqueFinding(BaseModel):
    finding_type: Literal[
        "CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "BOUNDARY_SMELL",      # structural
        "ANEMIC_ENTITY", "ANEMIC_MODEL", "MISSING_AGGREGATE",                  # content
        "MISPLACED_ENTITY", "NAMING_SMELL", "LOW_CONFIDENCE", "OTHER",         # content / advisory
    ]
    priority: Literal["high", "medium", "low"]
    target_ref: str                       # "context:Ordering" | "entity:Ordering.Order"
    rationale: str
    proposed_revision: str                # human-readable; consumed as producer feedback
    evidence_sentence_indices: List[int] = Field(default_factory=list)   # grounding into Scout

class CriticReport(BaseModel):
    model_id: str
    findings: List[CritiqueFinding] = Field(default_factory=list)   # final (best) cycle's findings
    score: float = 0.0                    # severity score of the best cycle (lower = better)
    malformed_findings: int = 0           # dropped during validation (observability)
    loop: "CriticLoopTrace"
    error: Optional[str] = None           # set if the loop ended on failure

class CriticLoopTrace(BaseModel):
    cycles_used: int
    best_cycle: int
    outcome: Literal["converged", "exhausted", "flapped", "failed"]
    score_per_cycle: List[float] = Field(default_factory=list)
    findings_count_per_cycle: List[int] = Field(default_factory=list)

# DomainModel gains exactly one optional field:
#   critic_report: Optional[CriticReport] = None
```

`CriticResponse` / `ProposedFinding` (in `core/critic/types.py`) are the **LLM-facing** schema (`analysis: str` for CoT, then `findings: List[ProposedFinding]`; no provenance/score). `run_critic` maps them to the persisted `CritiqueFinding`/`CriticReport`, mirroring the existing `pipeline_contracts` vs `schemas` split.

> Use `Field(default_factory=list)`, never `=[]`.

---

## 7. Control flow + wiring (the refactor)

**The risky part.** Today `core/orchestration/pipeline.py:run_pipeline` interleaves the **architect-rerun loop** and the **specialist-refine loop** inside one `while True` (a late `RefinementExhaustedError` with architect-stage issues can `continue` back to an architect rerun; they share the `architect_attempts` budget). The critique loop wraps this whole "generate one model" unit.

**Plan:**
1. **Extract** the current `run_pipeline` body (architect-rerun loop → specialist-refine → synthesizer → empty-checks) into:
   ```python
   def _generate_once(
       scout: ScoutOutput, deps: PipelineDeps, srs_path: str | None, *,
       architect_feedback: list | None = None,    # seeds the FIRST architect call (structural)
       specialist_feedback: list | None = None,    # seeds specialist extraction (content)
   ) -> tuple[DomainModel, ArchitectOutput, list[SpecialistAnalysis]]
   ```
   When both feedbacks are `None`, `_generate_once` must be **behavior-identical** to today's `run_pipeline` body (locked by a regression test).
2. **`run_pipeline`** becomes a thin dispatcher:
   ```python
   scout = deps.scout(srs_text)                      # once
   if deps.critic is None:
       return _generate_once(scout, deps, srs_path)[0]   # today's exact behavior
   return run_critique_loop(scout, deps, srs_path)       # new path
   ```
3. **`PipelineDeps`** gains `critic: Optional[CriticFn] = None` where
   `CriticFn = Callable[[DomainModel, ScoutOutput, list], CriticReport]`.
4. **`core/architect.py:analyze_document`** wires `critic=critic_fn` (gated on `DDD_CRITIC_LOOP`); `critic_fn` calls `run_critic` with `STAGE_TO_GROUP["Critic"]`.
5. **`configs/models.py`** — add `"Critic": "domain_extraction"` to `STAGE_TO_GROUP`.
6. **`core/observability/run_manifest.py`** — record a `critic` `StageRecord` per cycle (reuse `StageStatusLiteral`); store the `CriticLoopTrace` in the manifest. No new run-level `OutcomeLiteral` needed (non-fatal); loop failures land in `manifest.errors`.
7. **Intermediate dumps** — per cycle `core/intermediate/{ts}_critic_cycle{k}.json` (report + diff) + a final `{ts}_critic_loop.json` summary, matching the per-stage dump convention.

**Boundary note for the plan:** the architect-internal rerun + specialist-refine coupling stays *inside* `_generate_once`. The structural regeneration path goes through `_generate_once` (so that coupling is preserved); the content-only path bypasses architect entirely (architecture unchanged) and only re-runs `specialist_with_feedback → refine_until_clean → synthesizer`.

---

## 8. Finding routing & feedback adaptation (`routing.py`)

| finding_type | priority gate | route | regeneration path |
|---|---|---|---|
| `CONTEXT_SHOULD_MERGE` / `CONTEXT_SHOULD_SPLIT` / `BOUNDARY_SMELL` | high/med | **Architect** | structural → `_generate_once(architect_feedback=…, specialist_feedback=…)` |
| `ANEMIC_ENTITY` / `ANEMIC_MODEL` / `MISSING_AGGREGATE` / `MISPLACED_ENTITY` / `NAMING_SMELL` / `LOW_CONFIDENCE` | high/med | **Specialist** (targeted contexts) | content-only → `specialist_with_feedback → refine → synth` |
| any | low | — | advisory only (recorded, no regeneration) |
| `OTHER` | any | — | advisory only |

**Both present in a cycle:** structural wins — run the structural path; pass the cycle's content findings as `specialist_feedback` too, so the re-extraction also addresses them. (Architecture changed ⇒ contexts differ ⇒ can't reuse prior per-context specialist output; content findings re-surface against the new model next cycle if unaddressed.)

**Adapter:** `CritiqueFinding → issue-like object` exposing `.severity`, `.target`/`.location` (= `target_ref`), `.message` (= `rationale` + `proposed_revision`), so it slots into the existing `architect_with_feedback(scout, issues)` / `specialist_with_feedback(arch, scout, prev, issues)` prompts and `_format_issue` logging unchanged.

**MISPLACED_ENTITY** is content (Specialist re-extracts the two named contexts with "entity X belongs in context Y") — not Architect — because entity-to-context assignment is a Specialist responsibility, not a context-identification one.

---

## 9. Loop control details

- **Score (keep-best):** `score = 3·#high + 2·#med + 1·#low`; lower is better. Tie-break: fewer total findings, then earlier cycle (stability). The returned model is the argmin over all *successfully critiqued* cycles (a failed cycle is not a candidate).
- **Convergence:** zero high/med findings ⇒ `outcome="converged"`.
- **Flap:** signature = sorted set of `(finding_type, target_ref)` over high/med findings; if cycle k's signature equals cycle k-1's ⇒ `outcome="flapped"`, stop. (Analogous to the existing `_detect_flapping`, but for critique findings; implemented in `loop.py`, not entangled with the verifier flap.)
- **Exhaustion:** N reached with high/med still present ⇒ `outcome="exhausted"`.
- **Cycle 0 already clean:** 0 regenerations, return cycle-0 model + report (`converged`). C always runs ≥1 time when enabled.
- **Diff summary (Reflexion):** deterministic comparison of consecutive models — context set delta, per-context entity set delta, renames (matched by evidence overlap / fuzzy name). Compact text fed to C + producers next cycle.

---

## 10. Failure handling (D7)

- `run_critic` detects failure via `LLMResponse.json_failed` (the `structured_output` contract sets this and **does not raise**) after the standard retry/rotation; on persistent failure it returns a sentinel or raises `CriticError` caught at the loop boundary.
- **Loop boundary:** any `CriticError` or regeneration exception ⇒ stop, set `CriticReport.error`, `CriticLoopTrace.outcome="failed"`, record in `manifest.errors`, **return best-so-far** (or the cycle-0 raw model if nothing was critiqued).
- No bare `except`; specific `CriticError` in `core/critic/errors.py`. Never silently swallow.

---

## 11. Observability

- Per-cycle `critic` `StageRecord` (LLM call records, latency, `json_parse_failures`) via the existing `StageEmitter` (`_optional_stage("critic")` per cycle).
- `CriticLoopTrace` persisted in `critic_report` **and** the run manifest: `cycles_used`, `best_cycle`, `outcome`, `score_per_cycle`, `findings_count_per_cycle`.
- Intermediate dumps per §7.7 for offline inspection of how the model evolved cycle-to-cycle.

---

## 12. Testing (TDD, mocked LLM — no live API)

**Critic (pure):**
- Well-formed `CriticResponse` → `CriticReport.findings` populated; CoT `analysis` captured.
- Unresolvable `target_ref` / out-of-range `evidence_sentence_indices` → dropped + `malformed_findings++`.
- `json_failed` after retries → `CriticError` (caught upstream).

**Routing:**
- Each `finding_type` → correct partition (structural/content/advisory); priority gate (low → advisory).
- `MISPLACED_ENTITY` → content/Specialist with both contexts targeted.
- Adapter output exposes `.severity/.target/.message` consumable by the existing feedback signatures.

**Loop (mocked `_generate_once` + mocked critic):**
- Converged: cycle-0 clean → 0 regenerations, returns cycle-0.
- Keep-best: cycle-1 worse than cycle-0 → returns cycle-0; `best_cycle=0`.
- Flap: cycle-1 and cycle-2 identical signatures → stop, `outcome="flapped"`.
- Exhaustion: persistent high findings → N cycles, `outcome="exhausted"`.
- Failure: critic raises on cycle-2 → returns best of cycles 0–1, `outcome="failed"`, `error` set.
- Reflexion: cycle-k critic input includes prior findings + diff (assert prompt contains history).

**Pipeline integration & backward-compat (critical):**
- `critic=None` → `run_pipeline` byte-identical to pre-refactor output (golden/regression test on a fixed SRS fixture).
- `_generate_once(feedback=None)` ≡ old `run_pipeline` body (extract-refactor safety).
- `DDD_CRITIC_LOOP` unset → loop never runs.
- Old `model.json` (no `critic_report`) deserializes (`critic_report=None`).

**Invariant safety (the Topology-A payoff):**
- For a regeneration that the critic requested, the resulting model still passes Verifier D1–D5 + D6/D7 (hard) + D8 (auto-heal) + grounding subset — asserted via the existing synthesizer/verifier path (no separate guard).

---

## 13. Limitations & follow-ups

- **L1 — Cost.** Up to N full regenerations + N critic calls per document (worst case ≈ N× generation cost). Accepted under accuracy-over-cost; mitigated by the content-only cheaper path and early convergence.
- **L2 — Coarse revision.** Re-derivation can churn unrelated parts; keep-best + flap-detect + Reflexion bound the damage. Surgical typed-tool revision (the deferred Topology C) is the natural **v2** if churn proves costly.
- **L3 — Structural+content same cycle** defers unaddressed content findings to the next cycle (documented in §8).
- **L4 — Paper integration deferred.** When the EMSE side resumes, the loop becomes an ablation arm (C-off via `critic=None` vs C-on); reproducibility (seed/model lock) must be revisited then. Out of scope now.
- **L5 — `DDD_CRITIC_LOOP` default flip** to ON after validation on real SRS inputs is a follow-up decision.

---

## 14. Relationship to prior artifacts

- **Supersedes** the report-only draft of this same file (commit `bbb19d5`). C-as-evaluator survives unchanged as the loop's inner critic; the report-only *constraint* is lifted (mutation is back, done safely via Topology A).
- **Codex adversarial review (gpt-5.5, `bbkp9b9v9`)** still applies: (a) its endorsed "active mode" *is* Topology A; (b) its factual corrections hold and matter here — **D8 auto-heals** (not hard-fail) and the **wired Verifier is D1–D5 only** (D9/S3/D11 exist but are unwired), so the loop's "invariants re-applied for free" claim rests on D1–D5 + D6/D7 hard + D8 auto-heal; (c) `confidence` is already double-written by AST enrichment, so C only ever *flags* confidence (advisory `LOW_CONFIDENCE`), never writes it.

---

## 15. Cross-references

- Orchestrator: `core/orchestration/pipeline.py` (`run_pipeline`, `PipelineDeps`, `_optional_stage`, `_detect_flapping`).
- Producer feedback hooks: `core/architect.py` (`architect_with_feedback`, `_specialist_with_feedback`, `analyze_document`).
- Loop primitive: `core/refiner/loop.py` (`refine_until_clean`).
- Synthesizer invariants: `core/synthesizer/__init__.py` (D6/D7 hard-fail, D8 auto-heal).
- Verifier: wired checks in `core/architect.py:verifier_fn` (D1–D5).
- Schemas / contracts: `core/schemas.py`, `core/pipeline_contracts.py`.
- LLM layer: `core/llm/` (`get_client_for_model`, `structured_output`), `configs/models.py` (`STAGE_TO_GROUP`).
- Observability: `core/observability/run_manifest.py`.
- Handoff: `.planning/HANDOFF-2026-05-25-llm-augmentation.md`. Next agent **A — Context-Mapper** follows after this ships.
