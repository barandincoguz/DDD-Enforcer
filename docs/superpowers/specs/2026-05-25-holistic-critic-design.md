# Design Spec — C: Holistic Critic (report-only, v1)

**Date:** 2026-05-25
**Status:** DESIGN — awaiting user review before plan
**Feature:** A post-synthesis LLM **design-reasoning** stage for the domain-model pipeline.
**Scope decision:** Part of the **EMSE-studied system** (user chose option (i) at the decision gate — adding C forces an experimental-matrix re-run; reproducibility constraints hold).
**Process:** `brainstorming` → (this spec) → `writing-plans` → `subagent-driven-development`.
**Adversarial review:** Codex (`gpt-5.5`, xhigh, read-only repo access) reviewed the first draft and forced a **report-only pivot** + corrected three factual errors. Integrated below; rationale in §10.

---

## 1. TL;DR

The pipeline does **extraction** (LLM: Architect/Specialist/Synthesizer) + **validation** (deterministic: Verifier). It lacks the middle **design-reasoning** layer — no agent reads the assembled `DomainModel` and asks *"is this good DDD?"*.

C adds exactly that, as a **pure evaluator**: one LLM call over the synthesized `DomainModel`, producing a structured **`critic_report`** of design findings (anemic entities, naming smells, misplaced entities, contexts that should merge/split, missing aggregates, boundary smells). **C does not mutate the model.** Findings are advisory — grounded, prioritized, traceable — consistent with the user's framing that the domain model is a *DDD rule-book*, not a hard-error generator.

The intent that "C should eventually *improve* the model" is preserved but **deferred** to a future **active mode** (§9) that reuses the existing pre-synthesis feedback mechanism — not a post-synthesis patch subsystem.

---

## 2. Motivation

- **Missing layer.** Extraction emits a model; deterministic Verifier enforces shape/grounding invariants (D1–D5 in the wired pipeline; D6/D7 hard-fail + D8 auto-heal in the synthesizer). Neither reasons about *DDD design quality*.
- **Rule-book philosophy.** Per the user: the model is enforcement guidance, so a new LLM stage should *reason and propose*, not rigidly gate.
- **Paper value.** A structured, measurable design-critique artifact is a clean, attributable contribution for the EMSE study (ablation: C-off vs C-report-only — §8).

---

## 3. Locked design decisions (with rationale)

**D-C1 — Report-only; no model mutation.**
C reads `DomainModel` (+ Scout sentences) and emits `critic_report`. It does **not** edit `bounded_contexts`, entities, or any element.
*Rationale:* a post-synthesis editor needs a brand-new apply+guard subsystem; the existing Refiner only re-runs **pre-synthesis** stages on `SpecialistAnalysis` (`core/orchestration/pipeline.py`), so it cannot apply post-synthesis edits. An additive-only auto-apply (the first draft) produces metadata no validator consumes → dead weight; if a validator *did* consume it, it would silently contaminate downstream metrics. Separating **evaluation** from **mutation** removes the riskiest, highest-LOC, lowest-value part and yields clean experimental attribution.

**D-C2 — No per-element schema fields.**
All C output lives in `DomainModel.critic_report`. No `Entity.is_anemic`, no `Entity.design_notes`.
*Rationale:* per-element fields are scope-creep on the core schema and only earn their place once a validator consumes them as a stable contract. Until then, keep the blast radius at exactly one optional top-level field.

**D-C3 — Slot: 7th pipeline stage inside `run_pipeline`, pre-AST-enrichment.**
After Synthesizer + the empty-model check, before `run_pipeline` returns. Wrapped in `_optional_stage("critic")`. `PipelineDeps.critic: Optional[CriticFn] = None` (default `None` → every existing call-site/test is unchanged).
*Rationale:* single location, free `StageEmitter` observability, reproducible regardless of `WORKSPACE_PATH`. **Scope boundary:** C critiques the model's *internal DDD quality* (SRS-derivable — cohesion, naming, anemia-by-attributes, missing aggregates, boundary smells). It does **not** assess *model-vs-code fidelity* — that is code evidence, arrives later via AST enrichment in `main.py` (3 call sites), and belongs to the validation side. Because C's scope is the conceptual model, pre-AST is correct and the confidence-overwrite conflict (below) disappears.

**D-C4 — C never touches `confidence`.**
`confidence` is already double-owned (LLM-emitted on `Entity`, then mutated by AST enrichment, `core/AST/ast_signal_enrichment.py`). C adding a third writer is a smell. A "confidence looks wrong" observation becomes an advisory **finding**, never an edit.

**D-C5 — Failure mode is mode-split.**
- **Paper mode (strict):** C failure (LLM unrecoverable / `json_failed` after retries) → raise `CriticError` → run aborts and is marked `critic_error` in the manifest. Homogeneous N=10 is required for paper data; a mixed batch (some runs critique'd, some `status="failed"`) is worse than fail-fast.
- **Product mode (VS Code):** non-fatal — `critic_report.status="failed"` + `error` + `manifest.errors`, model persists so the user still gets a model.
*Rationale:* "explicit failure, no silent degradation" (AGENTS.md) is honored in both: paper raises; product records loudly. One-size-non-fatal was the first draft's rationalization of a soft fallback.

**D-C6 — Reproducibility.**
C uses the generation group: G1 `gemini-3.1-pro-preview`, temp 0.05, seed 42, via `STAGE_TO_GROUP["Critic"]="domain_extraction"`. Provenance (`model_id`, `temperature`, `seed`) is recorded in `critic_report` and the run manifest. Intermediate dump: `core/intermediate/{ts}_7_critic.json`.

---

## 4. Architecture — new `core/critic/` package

| File | Responsibility |
|---|---|
| `core/critic/types.py` | LLM I/O schema: `CriticResponse{findings: List[ProposedFinding]}`, `ProposedFinding`. |
| `core/critic/prompt.py` | Prompt builder: serialize `DomainModel` (+ Scout sentences) into the critique prompt. |
| `core/critic/critic.py` | `run_critic(model, scout, *, strict) -> DomainModel`: 1 LLM call, map+validate findings, attach `critic_report`, handle failure per mode. |

No `guard.py` (no apply). No `apply.py` (no mutation). Target ≪500 effective LOC total.

The validation of findings (target_ref resolves, evidence indices within Scout range) lives inline in `critic.py` as a small pure helper — invalid findings are **dropped + counted** (`malformed_findings` on the report), never crash. This is *filtering noise*, not *guarding mutations*.

---

## 5. Data flow

```
Synthesizer ─► DomainModel ─► (empty-check) ─► run_critic(model, scout, strict)
                                                  │ LLM (G1, structured_output(CriticResponse))
                                                  │   └─ response.json_failed? → retry / mode-split failure
                                                  │ map ProposedFinding → CritiqueFinding
                                                  │   └─ validate target_ref + evidence; drop+count invalid
                                                  │ attach critic_report (model OTHERWISE UNCHANGED)
                                                  ▼
                                           DomainModel + critic_report
                                                  ▼ (return from run_pipeline)
                                         main.py: AST enrich ─► save model.json
```

LLM input: serialized contexts + ubiquitous language summary + Scout sentences (so findings can carry `evidence_sentence_indices` grounded in Scout `index`, consistent with the existing grounding thread).

---

## 6. Schema changes (`core/schemas.py`) — additive, backward-compatible

```python
class CritiqueFinding(BaseModel):
    finding_type: Literal[
        "ANEMIC_ENTITY", "ANEMIC_MODEL", "NAMING_SMELL", "MISPLACED_ENTITY",
        "CONTEXT_SHOULD_MERGE", "CONTEXT_SHOULD_SPLIT", "MISSING_AGGREGATE",
        "BOUNDARY_SMELL", "LOW_CONFIDENCE", "OTHER",
    ]
    priority: Literal["high", "medium", "low"]
    target_ref: str                       # e.g. "context:Ordering" | "entity:Ordering.Order"
    rationale: str
    proposed_revision: str                # human-readable suggestion; NOT applied in v1
    evidence_sentence_indices: List[int] = Field(default_factory=list)  # grounding into Scout

class CriticReport(BaseModel):
    status: Literal["ok", "failed"]
    model_id: str
    temperature: float
    seed: int
    findings: List[CritiqueFinding] = Field(default_factory=list)
    malformed_findings: int = 0           # dropped-during-validation count (observability)
    error: Optional[str] = None           # populated when status == "failed"

# DomainModel gains exactly one optional field:
#   critic_report: Optional[CriticReport] = None
```

`CriticResponse` / `ProposedFinding` (in `core/critic/types.py`) are the **LLM-facing** schema (looser: no provenance, no status). `run_critic` maps them to the **persisted** `CritiqueFinding` / `CriticReport`. Keeping the two separate follows the existing `pipeline_contracts` vs `schemas` split.

> Note: use `Field(default_factory=list)`, never `=[]` (mutable default), per Codex + repo convention.

---

## 7. Control flow + wiring

1. **`core/pipeline_contracts.py`** — add `CriticFn = Callable[[DomainModel, ScoutOutput], DomainModel]` typedef location (or in `pipeline.py` beside the other `*Fn` aliases, matching current style).
2. **`core/orchestration/pipeline.py`** — add `critic: Optional[CriticFn] = None` to `PipelineDeps`. After the post-synthesis empty-check (`pipeline.py:448`), before `return model`:
   ```python
   if deps.critic is not None:
       with _optional_stage("critic"):
           model = deps.critic(model, scout)
   ```
   (`scout` is already in scope.) `critic=None` → exact current behavior.
3. **`core/architect.py:analyze_document`** — define `critic_fn` that calls `run_critic(model, scout, strict=self._critic_strict)` and wire it into `PipelineDeps(... , critic=critic_fn)`. `_critic_strict` is sourced from an env flag (e.g. `DDD_CRITIC_STRICT=1`, set by the paper orchestrator for run-spec YAMLs; default `False` for product/VS Code).
4. **`configs/models.py`** — add `"Critic": "domain_extraction"` to `STAGE_TO_GROUP` (SSOT; inherits G1/temp/seed).
5. **`core/observability/run_manifest.py`** — add `"critic_error"` to `OutcomeLiteral` (paper-mode abort). The critic `StageRecord.status` reuses existing `StageStatusLiteral` (`success`/`fail`).
6. **Intermediate dump** — `core/intermediate/{ts}_7_critic.json` (matches existing per-stage dump convention).

---

## 8. EMSE / experimental framing

- **Ablation is a config flip.** `critic=None` → **C-off** (baseline). `critic=critic_fn` → **C-report-only** (C-on). Clean two-arm comparison.
- **No metric contamination.** Because C does not mutate the model, the validation pipeline sees an *identical* model in both arms except for the additive `critic_report`. C's contribution is measured on the critique artifact itself (e.g., agreement with the 3-rater Fleiss's κ on whether findings are valid DDD critiques), not by perturbing downstream enforcement metrics.
- **Future third arm** — **C-active-feedback** (§9) — is explicitly out of scope for this WP; noted so the paper's design space is honest.
- **Re-run cost** is accepted (decision (i)): regenerate `model.json` for the studied subjects with C-on, keep C-off artifacts as baseline.

---

## 9. Future active mode (documented, NOT built here)

If/when C should *change* the model, the path is:
```
critic_report ─► select high-priority structural findings
              ─► feed as issue-style feedback into Architect / Specialist (architect_with_feedback / specialist_with_feedback already exist)
              ─► re-run pre-synthesis stages ─► re-synthesize ─► existing Verifier/Refiner
```
This reuses the **existing** pre-synthesis apply mechanism (no new post-synthesis patch subsystem), keeps the grounding thread and D6/D7/D8 invariants in force, and gives the paper a principled third arm. Out of scope for v1.

---

## 10. Adversarial-review integration (Codex, gpt-5.5 xhigh)

Codex reviewed the first draft against the live code and forced these changes:

1. **Pivot to report-only** (D-C1) — "post-synthesis critic as editor: bad; as evaluator: good." Adopted.
2. **Drop per-element schema fields** (D-C2). Adopted.
3. **Mode-split failure** (D-C5) — paper must fail-fast, not produce a heterogeneous batch. Adopted.
4. **Factual corrections (verified against code):**
   - **D8 is NOT hard-fail** — `core/synthesizer/__init__.py` auto-heals a dangling `aggregate.members` entry and returns the model; only **D6/D7** raise `SynthesizerInvariantError`. (First draft wrongly listed D8 as a hard invariant the guard relies on — moot now that there's no guard.)
   - **Wired Verifier runs D1–D5 only** — `core/architect.py:verifier_fn` calls D1/D2/D3/D4/D5; **D9/S3/D11 functions exist** (`core/verifier/checks_deterministic.py`) but are **not** wired into the pipeline. (First draft + handoff over-stated the wired check set. Irrelevant to report-only C, but corrected for accuracy.)
   - **`confidence` already double-written** by AST enrichment → C must not add a third writer (D-C4).
   - Use `default_factory=list`, not `=[]`.

---

## 11. Testing (TDD, mocked LLM — no live API)

- **Mapping/validation units:** well-formed `CriticResponse` → `critic_report.findings` populated; finding with unresolvable `target_ref` → dropped + `malformed_findings` incremented; out-of-range `evidence_sentence_indices` → dropped/trimmed; empty findings → `status="ok"`, empty list.
- **Failure handling:** `structured_output` returns `json_failed=True` after retries → **strict=True** raises `CriticError`; **strict=False** → `critic_report.status="failed"` + `error` set, model returned intact.
- **No-mutation invariant:** for any LLM response, the returned model's `bounded_contexts` are byte-identical to the input (only `critic_report` differs). This is the central guarantee — test it explicitly.
- **Pipeline integration:** `PipelineDeps(critic=None)` → model unchanged, no `critic_report` (backward-compat); `critic=critic_fn` (mocked) → `critic_report` present, status `ok`.
- **Schema back-compat:** a pre-C `model.json` (no `critic_report`) deserializes cleanly (`critic_report=None`).
- **Manifest:** strict-mode failure surfaces `outcome="critic_error"`; success records a `critic` `StageRecord`.

---

## 12. Limitations & follow-ups

- **L1 — Advisory findings have no consumer yet.** Their value is realized only when (a) the paper measures them, or (b) a future validator / active mode consumes them. v1 ships the artifact + the measurement hook, not an enforcement change.
- **L2 — No model-vs-code critique.** By scope (D-C3), C judges the conceptual model only. Code-fidelity critique is a separate, validation-side concern.
- **L3 — Active mode deferred** (§9).
- **L4 — `DDD_CRITIC_STRICT` wiring** into the paper orchestrator run-spec YAMLs is part of the implementation plan, not yet specified at the orchestrator level.

---

## 13. Cross-references

- Pipeline orchestrator: `core/orchestration/pipeline.py` (`run_pipeline`, `PipelineDeps`, `_optional_stage`).
- Stage wiring: `core/architect.py:analyze_document`.
- Schemas: `core/schemas.py` (`DomainModel`, `Entity`).
- LLM layer: `core/llm/` (`get_client_for_model`, `structured_output`), `configs/models.py` (`STAGE_TO_GROUP`).
- Synthesizer invariants: `core/synthesizer/__init__.py` (D6/D7 hard-fail, D8 auto-heal).
- Observability: `core/observability/run_manifest.py` (`OutcomeLiteral`, `StageRecord`).
- Handoff: `.planning/HANDOFF-2026-05-25-llm-augmentation.md`.
- Next agent A — Context-Mapper — builds after C ships (see handoff).
