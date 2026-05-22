# WP-CORE-6 — D1 verifier non-vacuous: Architect populates `supporting_sentence_ids` end-to-end

**Date:** 2026-05-21
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 5)
**Status:** REVISED v2 — addressed Codex xhigh adversarial review (4 CRITICAL + 4 WARN + 6 NIT + 1 OQ; all CRITICAL+WARN handled inline; 1 OQ deferred with rationale)
**Parent finding:** `.planning/pipeline_audit/findings/architect.md` finding **F-21** (MAJOR)
**Loop:** Domain Pipeline Hardening Loop (fifth WP; baseline 338 confirmed at HEAD `9608495`)
**Sibling iterations:**
- Iteration 1 — WP-CORE-2 shipped at `25e6880` (reference-heading truncation)
- Iteration 2 — WP-CORE-3 shipped at `daefeb0` (empty-input contract)
- Iteration 3 — WP-CORE-4 shipped at `02e0fe9` (`IntermediateSaveError` + `srs_path` propagation)
- Iteration 4 — WP-CORE-5 ABANDONED + WP-CORE-5b shipped at `27a5d98` (`SynthesizerEmptyModelError` taxonomy preservation)
**Codex review:** `decision_log.md` D-CODEX-REVIEW-WP-CORE-6 (to be appended at DOC commit).

---

## Revision history

- **v1 (draft, 2026-05-21 ~12:30 GMT+3)** — initial spec; sent to Codex xhigh for adversarial review.
- **v2 (this version, 2026-05-21 ~13:00 GMT+3)** — Codex xhigh review verdict: **4 CRITICAL + 4 WARN + 6 NIT + 1 OQ**. Dispositions:

  | # | finding | category | disposition |
  |---|---|---|---|
  | **C-1 (A7)** | `final-model-loss`: Even if Architect populates `supporting_sentence_ids`, Specialist receives `List[str]` of names only (`architect.py:788`), rebuilds `ContextHypothesis(context_name=ctx_name, description="")` fresh (`architect.py:621-623`) with default empty IDs. `SpecialistAnalysis.context` has empty IDs. Synthesizer merge (`merge.py:37-42`) copies empty list into `BoundedContext.supporting_sentence_ids`. Final DomainModel still vacuous. | scope gap | **ADOPTED.** Spec v2 widens scope: thread IDs through Specialist + Synthesizer. `extract_per_context_details` signature changes from `contexts: List[str]` to `contexts: List[ContextHypothesis]` (preserves IDs). `specialist_fn` passes `arch.contexts` directly. SpecialistAnalysis.context retains Architect's IDs. Synthesizer merge already copies `analysis.context.supporting_sentence_ids` — works correctly once Specialist preserves the input. New integration test T-INT-1 verifies E2E. |
  | **C-2 (A2-d3-mask)** | `d3-mask`: D-3 (D1 ERROR on empty IDs) doesn't enforce because Refiner exhausts → `pipeline.py:79-93` catches `RefinementExhaustedError` → discards issue detail → continues with best-effort. ERROR is logged-and-discarded, not fail-fast. | enforcement gap | **ADOPTED with reframe.** D-3 kept (defense-in-depth signal; closes Architect-prompt regression vector). Reframed as **"honest signal, not enforcement"** in §D-3 + Risks. Paired with **D-6** (NEW): improve `pipeline.py:75-78` degrade-log to include full issues list (closes A5-risk4 partial). Full enforcement (Refiner re-runs Architect on D1 ERROR) deferred to **F-22** (NEW backlog entry per A6-f22). |
  | **C-3 (A3-integration-gap)** | `integration-gap`: 5 tests insufficient — no E2E test verifies IDs survive Architect → Specialist → Synthesizer. | test gap | **ADOPTED.** New test T-INT-1: `analyze_document` end-to-end with mocked LLM responses for Scout/Architect/Specialist; assert final `DomainModel.bounded_contexts[].supporting_sentence_ids` is non-empty + matches Architect's IDs. |
  | **C-4 (A5-risk4)** | `risk4-understated`: `RefinementExhaustedError.issues` reduced to generic `print(type(exc).__name__)` at `pipeline.py:75-78`; D1 errors not reliably logged. | observability gap | **ADOPTED via D-6** (NEW): degrade-log emits the full `exc.issues` list (each issue's `stage`, `location`, `severity`, `message`) so post-mortem debugging works. ~5 LOC. |
  | **W-1 (A2-truncation)** | `truncation-mid-prefix`: `_truncate_with_head_tail` char-slices; can chop `[N] ` mid-prefix when text > 50k chars. Spec's "any `[N]` LLM sees is valid" not guaranteed. | correctness gap | **ADOPTED via D-4** (NEW): line-pair-aware truncation. New helper `_truncate_numbered_pairs(pairs: List[Tuple[int, str]], max_chars: int)` truncates by dropping whole `(idx, text)` pairs from the middle (head + tail kept), never chopping mid-pair. Existing `_truncate_with_head_tail` unchanged (still used by Specialist's text truncation). |
  | **W-2 (A3-top-level-old-shape)** | `top-level-list-bypass`: parser at `architect.py:464-476` accepts top-level list `["X", "Y"]` (old shape); strict-dict rejection only on dict branch. | parsing gap | **ADOPTED via D-5 tightening**: remove top-level list branch entirely. Architect must return `{"contexts": [{"name": ..., "supporting_sentence_ids": [...]}]}` shape strictly; anything else → retry → exhaustion → `ArchitectExtractionError`. |
  | **W-3 (A4-scope)** | `commit-plan-mis-scoped`: GREEN commit mis-scoped until integration + downstream propagation included. | scope | **ADOPTED** via C-1 + C-3 expansion. GREEN now covers `architect.py` (prompt + parser + signature + architect_fn + extract_per_context_details + Specialist ContextHypothesis preservation) + `verifier/checks_deterministic.py` (D-3 non-empty clause) + `orchestration/pipeline.py` (D-6 degrade-log). |
  | **W-4 (A6-f22)** | `refiner-cant-rerun-architect`: limitation should become F-22 or be fixed now. | backlog hygiene | **ADOPTED as F-22 backlog add.** New backlog entry in DOC commit: F-22 (MAJOR) — Refiner orchestration only re-runs Specialist (`pipeline.py:53-55`); Architect-stage verifier failures (e.g., D1 `ungrounded_context` ERROR) are non-recoverable and degrade silently. Fix scope: extend Refiner to dispatch re-runs by failing stage; differential error-handler per `RefinementExhaustedError.issues[].stage`. Deferred per WP-CORE-6 scope ("populate IDs end-to-end"; Refiner re-architecture is its own WP). |
  | **N-1 (A1-verified)** | Claims (a)-(d) in Discovery accurate. | confirmation | **ACCEPT-AS-IS.** |
  | **N-2 (A1-edge)** | LLM could return dicts today, but no path copies the field into `ContextHypothesis`. | precision | **ACCEPT-AS-IS** — confirms spec's audit. |
  | **N-3 (A2-strict-shape)** | Strict rejection of old shape correct (internal big-bang, no compat surface). | confirmation | **ACCEPT-AS-IS.** |
  | **N-4 (A3-tarch2-trace)** | RED expectation on T-ARCH-2 (old shape parses successfully today) accurate. | confirmation | **ACCEPT-AS-IS.** |
  | **N-5 (A3-prompt-test)** | T-ARCH-3 spy/substr assertion acceptable if it inspects `client.chat(messages=[...])` payload. | confirmation | **ACCEPT-AS-IS** — test plan tightens to inspect `mock_chat.call_args.kwargs["messages"][0]["content"]`. |
  | **N-6 (A5-methodology + A6-lineage)** | EMSE framing + F-21 lineage attribution accurate. | confirmation | **ACCEPT-AS-IS.** |
  | **OQ-1 (A6-srs-path)** | Adding `srs_path` to D1 issues not symmetric with WP-CORE-5b without verifier-issue schema change; defer unless issues become persisted artifacts. | scope | **DEFERRED with rationale.** Verifier issues are runtime-only (intermediate JSON dumps log `legacy_issues` already); they're not user-facing persisted artifacts like the orchestration errors. Adding `srs_path` would require `VerifierIssue` schema widening + threading through 5 call sites in `verifier_fn`. Out of WP-CORE-6 envelope. Revisit if F-22 (Refiner stage-aware re-runs) lands and verifier issues become Refiner-control-flow primary signals. |

  **Codex disposition summary**: 4 CRITICAL all ADOPTED inline; 4 WARN all ADOPTED inline; 6 NIT all confirmed (no action needed); 1 OQ deferred with explicit rationale. **First non-zero-deferred Codex disposition since WP-CORE-2** — but the OQ deferral is scope-bounded + has a concrete revisit trigger (post-F-22), not a vague "future work". 4-iteration zero-deferred streak (WP-CORE-3 → CORE-4 → CORE-5b → CORE-6) ends here, by design.

---

## Motivation

### The bug (D1 has passed vacuously for every project run)

`core/verifier/checks_deterministic.py:7-27`:

```python
def check_d1_supporting_sentence_ids_subset(
    contexts: List[Dict],
    scout_sentence_indices: Set[int],
) -> List[VerifierIssue]:
    """D1: every BC.supporting_sentence_ids ⊆ Scout-emitted indices."""
    issues: List[VerifierIssue] = []
    for ctx in contexts:
        bad = [i for i in ctx.get("supporting_sentence_ids", []) if i not in scout_sentence_indices]
        if bad:
            issues.append(...)
    return issues
```

The check iterates `ctx.get("supporting_sentence_ids", [])`. If the list is empty, the comprehension produces `[]`, `bad` is `[]`, no issue is emitted. **An empty list passes D1 trivially.**

### Why every context has empty `supporting_sentence_ids`

`core/architect.py:776-783`:

```python
def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
    sentence_texts = [s.text for s in scout.sentences]
    ctx_names = self.identify_contexts(sentence_texts)
    contexts = [
        ContextHypothesis(context_name=n, description=f"{n} context")
        for n in ctx_names
    ]
    return ArchitectOutput(contexts=contexts)
```

`ContextHypothesis.supporting_sentence_ids` has `Field(default_factory=list)` (`core/pipeline_contracts.py:91`). `architect_fn` never sets it — defaults to `[]`.

And `identify_contexts` itself never asks the LLM for IDs. The prompt at `architect.py:386-405` requests:

```
RESPOND WITH JSON:
{
  "contexts": ["ContextName1", "ContextName2", ...]
}
```

Bare string array. No sentence references. The Architect LLM has no way to surface which sentences justified a context name.

### Production reachability (per loop discipline lesson from iteration 4)

**LIVE — every `analyze_document` call hits this path.**

Unlike F-11 (parallel Scout dead from production) and F-14 (production-blocked by Architect upstream guard), F-21 fires on every single run:
- `analyze_document` → `run_pipeline` → `architect_fn` → `identify_contexts` returns names → `ContextHypothesis(...)` with default empty `supporting_sentence_ids` → verifier D1 check passes vacuously.

The EMSE methodology section claims "D1 catches contexts that cite sentences Scout did not emit" — empirically true ONLY IF supporting_sentence_ids is ever non-empty. Today it is never non-empty. **The claim is currently empirically vacuous for every project run.**

### Why fix now

Per Codex W-8 from the WP-CORE-5b review: "F-21 has clearer production methodology consequence: every project run passes a verifier check it should fail, which is an EMSE methodology gap." This is the highest-impact remaining MAJOR finding in the orchestrator layer. Effort estimate M (Architect prompt + parsing change + thread through Pydantic contract). LLM-behavior risk is bounded: new prompt is additive (asks for one more field per context), retry loop already catches malformed JSON via `ArchitectExtractionError`.

### Non-goals

- **No change to D1 check itself for vacuous-empty case.** D1 is defined as "subset of Scout indices" — semantically, an empty subset IS a subset. We fix the upstream (populate the IDs) rather than weakening the downstream semantics. **However**, given that the vacuous-pass IS the defect, the spec MAY include a tightening that flags empty `supporting_sentence_ids` as a separate ERROR. See §D-3 below for the decision.
- **No change to D2-D5 checks.** Out of scope.
- **No change to Scout chunking / numbering.** Scout already produces `SectionedSentence(index=i, text=...)` with `index` set; we just need to surface that index in the Architect prompt.
- **No new verifier check IDs.** If we add an empty-IDs check, it goes into D1 as a second clause (still `D1` issue_type).
- **No retry-on-empty.** If the LLM returns empty `supporting_sentence_ids` for a context, that's a Verifier-flagged failure, which feeds the Refiner per existing pipeline. We do NOT add a special retry inside `identify_contexts`.

---

## Design

### D-1 — Update Architect prompt to request numbered + ID-bearing JSON

`core/architect.py:386-405`. Two prompt changes:

1. **Present sentences with explicit indices** (matching Specialist pattern at line 584-586):
   ```python
   numbered_text = "\n".join(f"[{i}] {s}" for i, s in enumerate(domain_sentences))
   ```
2. **Ask for object-array JSON shape**:
   ```
   RESPOND WITH JSON:
   {
     "contexts": [
       {"name": "ContextName1", "supporting_sentence_ids": [0, 3, 12]},
       {"name": "ContextName2", "supporting_sentence_ids": [5, 9]}
     ]
   }
   ```
3. **Add prompt constraint**: "Each context MUST cite ≥1 supporting_sentence_id from the numbered list above. If you cannot justify a context with at least one sentence, do not include it."

The 50,000-char truncation still applies but now operates on the numbered text. Truncation can chop a `[N] ` prefix mid-line; the LLM gets fewer complete sentences but any `[N]` it sees has a valid index because we numbered BEFORE truncation.

### D-2 — Change `identify_contexts` return shape

`core/architect.py:367` signature changes:

```python
# BEFORE:
def identify_contexts(self, domain_sentences: List[str]) -> List[str]:

# AFTER:
def identify_contexts(
    self,
    domain_sentences: List[str],
) -> List[Dict[str, Any]]:
    """Returns: list of {"name": str, "supporting_sentence_ids": List[int]} dicts."""
```

`architect_fn` updates accordingly:

```python
def architect_fn(scout: ScoutOutput) -> ArchitectOutput:
    sentence_texts = [s.text for s in scout.sentences]
    ctx_proposals = self.identify_contexts(sentence_texts)
    contexts = [
        ContextHypothesis(
            context_name=c["name"],
            description=f"{c['name']} context",
            supporting_sentence_ids=c["supporting_sentence_ids"],
        )
        for c in ctx_proposals
    ]
    return ArchitectOutput(contexts=contexts)
```

### D-2b — Thread IDs through Specialist (Codex C-1)

`specialist_fn` (architect.py:785-790) currently extracts names only:

```python
# BEFORE:
def specialist_fn(arch: ArchitectOutput, scout: ScoutOutput) -> List[SpecialistAnalysis]:
    ctx_names = [c.context_name for c in arch.contexts]
    sentence_texts = [s.text for s in scout.sentences]
    return self.extract_per_context_details(ctx_names, sentence_texts)
```

```python
# AFTER:
def specialist_fn(arch: ArchitectOutput, scout: ScoutOutput) -> List[SpecialistAnalysis]:
    sentence_texts = [s.text for s in scout.sentences]
    return self.extract_per_context_details(list(arch.contexts), sentence_texts)
```

`extract_per_context_details` (architect.py:574) signature widens:

```python
# BEFORE:
def extract_per_context_details(
    self, contexts: List[str], domain_sentences: List[str]
) -> List[SpecialistAnalysis]:
    ...
    for ctx_name in contexts:
        ...
        # line 621-623:
        ctx = ContextHypothesis(context_name=ctx_name, description="")
        ...
```

```python
# AFTER:
def extract_per_context_details(
    self,
    contexts: List[ContextHypothesis],
    domain_sentences: List[str],
) -> List[SpecialistAnalysis]:
    ...
    for ctx in contexts:
        ctx_name = ctx.context_name
        ...
        # line 621-623 deleted — re-use the input ctx; preserves
        # supporting_sentence_ids set by Architect.
        ...
        analysis = DomainArchitect._validate_specialist_payload(result, ctx)
        ...
```

**Result**: `SpecialistAnalysis.context` carries Architect's `supporting_sentence_ids` into the Synthesizer stage. `core/synthesizer/merge.py:41` already copies `list(analysis.context.supporting_sentence_ids)` into `BoundedContext.supporting_sentence_ids` — no synthesizer change needed; it just receives non-empty input now.

### D-3 — Strengthen D1 to flag empty `supporting_sentence_ids` (honest signal, not enforcement — Codex C-2 reframe)

**Decision: YES, fold this in — but with explicit limitation acknowledgement.**

**Limitation per Codex A2-d3-mask + A5-risk4**: Refiner exhausts on D1 ERROR → `pipeline.py:79-93` catches `RefinementExhaustedError` → continues with best-effort Specialist output → pipeline ships a model anyway. D-3 does NOT fail-fast on empty IDs; it produces an honest log signal.

**Why keep it anyway:**
- Defense-in-depth against a future Architect prompt regression that silently drops the field. Without D-3, regression would go back to vacuous and require fresh discovery.
- 2-LOC addition; minimal cost.
- Paired with D-6 (degrade-log improvement) so the ERROR is actually visible in the run manifest, not just `type(exc).__name__`.
- True enforcement (fail-fast on architect-stage ERROR) requires Refiner to re-run Architect on architect-stage issues — that's **F-22 (NEW backlog entry)**, deferred to its own WP.

`core/verifier/checks_deterministic.py` change:

```python
# BEFORE:
def check_d1_supporting_sentence_ids_subset(
    contexts: List[Dict],
    scout_sentence_indices: Set[int],
) -> List[VerifierIssue]:
    """D1: every BC.supporting_sentence_ids ⊆ Scout-emitted indices."""
    issues: List[VerifierIssue] = []
    for ctx in contexts:
        bad = [i for i in ctx.get("supporting_sentence_ids", []) if i not in scout_sentence_indices]
        if bad:
            issues.append(...)
    return issues
```

```python
# AFTER:
def check_d1_supporting_sentence_ids_subset(
    contexts: List[Dict],
    scout_sentence_indices: Set[int],
) -> List[VerifierIssue]:
    """D1: every BC.supporting_sentence_ids is (a) non-empty AND (b) ⊆ Scout-emitted indices.

    Non-emptiness clause added per WP-CORE-6 to close the vacuous-pass
    defect (F-21): prior to WP-CORE-6, Architect never populated this
    field, so the subset check passed trivially for every project run.
    """
    issues: List[VerifierIssue] = []
    for ctx in contexts:
        ids = ctx.get("supporting_sentence_ids", [])
        if not ids:
            issues.append(VerifierIssue(
                stage="architect",
                location=f"architect:contexts[{ctx.get('name')}].supporting_sentence_ids",
                issue_type="ungrounded_context",
                severity=IssueSeverity.ERROR,
                message=f"Context {ctx.get('name')!r} has no supporting_sentence_ids — cannot verify SRS grounding",
                suggestion="Architect must cite ≥1 Scout-emitted sentence index per context",
            ))
            continue
        bad = [i for i in ids if i not in scout_sentence_indices]
        if bad:
            issues.append(...)  # unchanged
    return issues
```

### D-4 — Line-pair-aware truncation (Codex W-1)

`_truncate_with_head_tail(text, max_chars)` slices raw characters — can leave `[42] sentence te...9] continuation` fragments at the head/tail boundary. The LLM may misread `9` as a new sentence index.

**Fix**: introduce `_truncate_numbered_pairs(pairs: List[Tuple[int, str]], max_chars: int, head_ratio: float = 0.6)` in `core/architect.py` (alongside existing `_truncate_with_head_tail`). Behavior:

1. Compute formatted length per pair: `len(f"[{idx}] {text}\n")`.
2. If cumulative total ≤ max_chars, return all pairs.
3. Otherwise, greedily take pairs from head until cumulative ≥ `max_chars * head_ratio`; greedily take pairs from tail (reversed) until cumulative ≥ remaining budget. Concatenate.
4. Never partial-chop a pair — drops whole pairs from the middle.

Then in `identify_contexts`:

```python
# BEFORE:
text = "\n".join(domain_sentences)
max_chars = 50000
if len(text) > max_chars:
    text = _truncate_with_head_tail(text, max_chars=max_chars)
# (then prompt embeds `text`)
```

```python
# AFTER:
numbered_pairs = list(enumerate(domain_sentences))
max_chars = 50000
numbered_pairs = _truncate_numbered_pairs(numbered_pairs, max_chars=max_chars)
numbered_text = "\n".join(f"[{i}] {s}" for i, s in numbered_pairs)
# (then prompt embeds `numbered_text`)
```

`_truncate_with_head_tail` is **untouched** — it's still used by other call sites that operate on opaque text (e.g., Specialist's domain-sentences). The new helper is purpose-built for index-tagged pairs.

### D-5 — Backwards-compatibility: no shims (unchanged)

Per AGENTS.md "no backwards-compat hacks", the prompt + return-type change is breaking for any caller of `identify_contexts` that expected `List[str]`. Verified callers via grep (single production caller — `architect_fn` at line 778; plus 3 test files that exercise the error-propagation path, none of which depend on return value). Also: WP-CORE-5b set the precedent that "internal big-bang" changes don't need compat shims (Codex N-3 confirmed for this WP).

### D-5b — Strict JSON parse: reject BOTH old-dict-bare AND top-level-list branches (Codex W-2)

The existing 5-retry loop in `identify_contexts` (`architect.py:408-504`) handles malformed JSON via `_parse_json_response` + `ArchitectExtractionError` on exhaustion. The new prompt expects a different shape; the parser becomes strict.

Current parser has TWO success branches:

```python
# core/architect.py:449-476 (today)
if isinstance(result, dict) and "contexts" in result:
    contexts = result["contexts"]
    if contexts and len(contexts) > 0:
        self._save_intermediate(...)
        return contexts                       # ← accepts ["X", "Y"] under dict wrapper
elif isinstance(result, list) and len(result) > 0:
    self._save_intermediate(...)
    return result                              # ← accepts ["X", "Y"] at top level
```

Both branches happily accept bare-string arrays. Per Codex W-2, **both must be tightened**:

```python
# core/architect.py:449-476 (post-WP-CORE-6)
if isinstance(result, dict) and "contexts" in result:
    contexts = result["contexts"]
    if not (isinstance(contexts, list) and contexts):
        # missing/empty → retry path (existing behavior preserved)
        print(f"  ⚠️  Empty contexts - Retry {retry + 1}/5")
        if retry < 4:
            continue
        raise ArchitectExtractionError(...)
    if not all(
        isinstance(c, dict)
        and isinstance(c.get("name"), str)
        and isinstance(c.get("supporting_sentence_ids"), list)
        and all(isinstance(i, int) for i in c["supporting_sentence_ids"])
        for c in contexts
    ):
        print(f"  ⚠️  Shape error - Retry {retry + 1}/5")
        if retry < 4:
            continue
        raise ArchitectExtractionError(
            srs_path=getattr(self, "_current_srs_path", "<unknown>"),
            message="Architect produced malformed contexts after 5 retries (expected object array with name + supporting_sentence_ids)",
        )
    self._save_intermediate(...)
    return contexts  # List[Dict[str, Any]]

# Top-level list branch DELETED — strict dict-wrapper-only.
```

The `elif isinstance(result, list)` branch is **removed**. LLM responses without the `{"contexts": [...]}` wrapper are now retry-worthy parse failures.

### D-6 — Degrade-log improvement: emit full issues list (Codex C-4 / A5-risk4)

`core/orchestration/pipeline.py:65-79` currently logs `type(exc).__name__` only on `RefinementExhaustedError`:

```python
# BEFORE:
except Exception as exc:
    print(
        f"  ⚠️  refiner exhausted retries ({type(exc).__name__}); "
        f"continuing with last Specialist output"
    )
    refined_specialist = specialist_output
```

`RefinementExhaustedError` carries `self.issues: List[VerifierIssue]` (`errors.py:41-44`). The degrade path discards this, making post-mortem impossible.

```python
# AFTER:
except RefinementExhaustedError as exc:
    issues_summary = "; ".join(
        f"{i.severity.value if hasattr(i.severity, 'value') else i.severity}@"
        f"{i.stage}:{i.location}: {i.message}"
        for i in exc.issues
    )
    print(
        f"  ⚠️  refiner exhausted retries ({len(exc.issues)} unresolved issue(s)); "
        f"continuing with last Specialist output. Issues: {issues_summary}"
    )
    refined_specialist = specialist_output
except Exception as exc:
    # Non-RefinementExhaustedError: keep original generic log.
    print(
        f"  ⚠️  refiner exhausted retries ({type(exc).__name__}); "
        f"continuing with last Specialist output"
    )
    refined_specialist = specialist_output
```

**Net effect**: when D1 ERRORs fire and the pipeline degrades, the run manifest now contains every unresolved verifier issue's stage + location + message. Honest signal for debugging.

### D-7 — F-22 backlog entry (Codex W-4 / A6-f22)

**Add to `.planning/pipeline_audit/improvements_backlog.md` in DOC commit:**

```
| F-22 | core/orchestration/pipeline.py + core/refiner/loop.py | Refiner orchestration only re-runs the Specialist stage (`pipeline.py:53-55` `_re_run_specialist`). Architect-stage verifier failures (e.g., D1 `ungrounded_context` ERROR) cannot be auto-corrected; pipeline degrades to best-effort via `RefinementExhaustedError` handler, silently shipping a model. Fix: extend Refiner to dispatch re-runs by failing stage; differential handler per `exc.issues[].stage`. Architect-stage errors should be either (a) hard-fail (raise `ArchitectGroundingError`) or (b) re-run Architect with issue-aware prompt feedback. Discovered during WP-CORE-6 spec drafting (Codex W-4 / A6-f22). | MAJOR | M-L | PIPELINE | OPEN |
```

Recorded in DOC commit; deferred fix to its own WP.

### Production code changes (4 files, post-Codex)

| file | change | LOC est. |
|---|---|---|
| `core/architect.py` | (a) Prompt rewrite in `identify_contexts` (numbered sentences + object array); (b) line-pair-aware truncation helper `_truncate_numbered_pairs`; (c) parser strict-shape (delete top-level list branch + dict-shape validation); (d) signature change `-> List[Dict[str, Any]]`; (e) `architect_fn` threads `supporting_sentence_ids` into `ContextHypothesis`; (f) `specialist_fn` passes `List[ContextHypothesis]` instead of `List[str]`; (g) `extract_per_context_details` signature widens; (h) line 621-623 deleted (re-use input `ctx`) | ~80 |
| `core/verifier/checks_deterministic.py` | Add non-empty clause to D1; docstring update | ~15 |
| `core/orchestration/pipeline.py` | Degrade-log enriched with full `exc.issues` list (split `RefinementExhaustedError` branch from generic `Exception`) | ~12 |
| `core/pipeline_contracts.py` | No change (field already exists with default `[]`); chose verifier-layer over schema-layer enforcement (OQ-4 disposition) | 0 |

**Total estimated: ~107 LOC production change** (was ~55 in v1; expanded for C-1 + C-4 + W-1 + W-2 inline fixes).

---

## Red-phase tests (v2 — 10 tests, post-Codex)

10 new tests across 3 files (was 5 in v1; expanded for C-1 + C-3 + C-4 + W-1 + W-2 coverage):

### `tests/test_verifier_deterministic.py` — append 2 tests (T-D1-NV-1..T-D1-NV-2)

| # | id | name | invariant |
|---|---|---|---|
| 1 | T-D1-NV-1 | `test_d1_check_flags_context_with_empty_supporting_sentence_ids` | D1 check with `contexts=[{"name": "X", "supporting_sentence_ids": []}]` returns one ERROR issue with `issue_type="ungrounded_context"`. **Closes the vacuous-pass at the verifier layer.** |
| 2 | T-D1-NV-2 | `test_d1_check_passes_when_supporting_sentence_ids_non_empty_subset` | D1 with `contexts=[{"name": "X", "supporting_sentence_ids": [0, 2]}]` and `scout_indices={0, 1, 2, 3}` returns `[]` (existing happy path — regression-lock that the non-empty clause didn't break the subset clause). |

### `tests/test_architect_identify_contexts.py` — NEW FILE, 4 tests (T-ARCH-1..T-ARCH-4)

| # | id | name | invariant |
|---|---|---|---|
| 3 | T-ARCH-1 | `test_identify_contexts_returns_dict_shape_with_supporting_sentence_ids` | Mock `client.chat` returns JSON `{"contexts": [{"name": "OrderMgmt", "supporting_sentence_ids": [0, 2]}]}`. Call `arch.identify_contexts(["s0", "s1", "s2"])`. Assert return is `[{"name": "OrderMgmt", "supporting_sentence_ids": [0, 2]}]`. |
| 4 | T-ARCH-2 | `test_identify_contexts_retries_on_old_dict_shape` | Mock `client.chat` returns `{"contexts": ["OrderMgmt"]}` (old dict shape) on attempts 1-4, new shape on 5. Assert succeeds after retries. Validates strict-shape rejection of legacy dict branch. |
| 5 | T-ARCH-2b (**NEW per W-2**) | `test_identify_contexts_retries_on_top_level_list_shape` | Mock `client.chat` returns `["OrderMgmt", "Inventory"]` (top-level list) on attempts 1-4, new shape on 5. Assert succeeds after retries. Validates removal of the top-level-list branch. |
| 6 | T-ARCH-3 | `test_identify_contexts_prompt_includes_numbered_sentences` | Patch `client.chat` with spy; call `arch.identify_contexts(["s0", "s1"])`; assert `mock_chat.call_args.kwargs["messages"][0]["content"]` contains `"[0] s0"` and `"[1] s1"` substrings. Locks numbered-prefix invariant. |

### `tests/test_architect_id_propagation.py` — NEW FILE, 3 tests (T-PROP-1..T-PROP-3)

| # | id | name | invariant (Codex C-1) |
|---|---|---|---|
| 7 | T-PROP-1 | `test_extract_per_context_details_preserves_context_hypothesis_ids` | Build `ctx = ContextHypothesis(context_name="OrderMgmt", supporting_sentence_ids=[0, 3])`. Mock `client.chat` to return valid Specialist JSON. Call `arch.extract_per_context_details([ctx], ["s0", "s1", "s2", "s3"])`. Assert `result[0].context.supporting_sentence_ids == [0, 3]`. Closes the C-1 Specialist-rebuild loss. |
| 8 | T-PROP-2 | `test_synthesizer_merge_carries_supporting_sentence_ids_into_bounded_context` | Build `SpecialistAnalysis(context=ContextHypothesis(context_name="X", supporting_sentence_ids=[5, 9]), entities=[...])`. Call `build_deterministic_skeleton([analysis], project_name="T")`. Assert `result.bounded_contexts[0].supporting_sentence_ids == [5, 9]`. Regression-lock the synthesizer's existing copy-behavior (no change needed in merge.py, but the test guards against accidental future regressions). |
| 9 | T-INT-1 (**NEW per C-3**) | `test_analyze_document_e2e_preserves_supporting_sentence_ids_to_final_domain_model` | Mock `client.chat` to return: (a) Scout chunk → `{"sentences": [...]}`; (b) Architect → `{"contexts": [{"name": "OrderMgmt", "supporting_sentence_ids": [0, 1]}]}`; (c) Specialist per-context → valid entity JSON; (d) Synthesizer enrich → empty (skip_enrich path). Patch `_wait_for_rate_limit`. Call `arch.analyze_document(text="An order is placed by a customer.\nOrder contains items.", srs_path="test.srs")`. Assert `final_model.bounded_contexts[0].supporting_sentence_ids == [0, 1]`. **End-to-end ID-survival test.** |

### `tests/test_pipeline_orchestration.py` — append 1 test (T-DEGRADE-LOG-1)

| # | id | name | invariant (Codex C-4) |
|---|---|---|---|
| 10 | T-DEGRADE-LOG-1 | `test_refiner_exhaustion_log_includes_issues_list` | Build `deps` with a verifier that always returns `ok=False` (single D1 issue). Call `run_pipeline(...)` and capture stdout. Assert the degrade-log line contains the issue's `stage`, `location`, and `message` substrings. Locks D-6 contract. |

### Existing tests potentially affected

- `tests/test_intermediate_save.py:141` — `arch.identify_contexts(domain_sentences=["one.", "two."])`. Calls identify_contexts with strings; forces JSON parse failure → ArchitectExtractionError. Return type change doesn't affect this test (mocked path forces exception before return).
- `tests/test_architect_extraction_error.py:38,54` — same pattern. Unaffected.
- `tests/test_architect_srs_path.py:118` — same pattern. Unaffected.
- `tests/test_pipeline_orchestration.py:36-72` `_make_typed_deps` fixture — `architect_fn` returns `ArchitectOutput(contexts=[ContextHypothesis(context_name="OrderMgmt", description="x")])` with default empty `supporting_sentence_ids`. The pre-WP-CORE-6 verifier_fn in `test_pipeline_orchestration.py:63` returns `VerifierResult(ok=True, issues=[])` regardless; D1 doesn't run in this fixture. **No update needed**, but T-DEGRADE-LOG-1 builds its own deps fixture that DOES exercise D1 ERROR.

**Total new tests: 10** (2 in verifier file + 4 in new identify_contexts file + 3 in new id_propagation file + 1 in existing pipeline file).

**RED signal expectation:**
- T-D1-NV-1: **FAIL** today (D1 doesn't flag empty IDs).
- T-D1-NV-2: **PASS** today (regression-lock).
- T-ARCH-1: **FAIL** today (returns `List[str]`).
- T-ARCH-2: **FAIL** today (old dict shape currently accepted).
- T-ARCH-2b: **FAIL** today (top-level list currently accepted).
- T-ARCH-3: **FAIL** today (prompt lacks `[N]` numbering).
- T-PROP-1: **FAIL** today (extract_per_context_details takes `List[str]`, rebuilds ContextHypothesis fresh).
- T-PROP-2: **PASS** today (synthesizer copy-behavior unchanged; regression-lock).
- T-INT-1: **FAIL** today (end-to-end propagation broken at Specialist rebuild).
- T-DEGRADE-LOG-1: **FAIL** today (log uses `type(exc).__name__` only).

**Net RED: 8 failing + 2 passing = 10 collected; baseline 338 → 348 collected; 340 passed + 8 failed.**

---

## Atomic commit sequence (4 commits — matches WP-CORE-3/4/5b cadence; scope expanded post-Codex)

| # | type | scope | summary | gate |
|---|---|---|---|---|
| 1 | `test` | `architect, verifier, orchestration` | WP-CORE-6 red-phase tests for D1 non-vacuous + ID propagation end-to-end + degrade-log enrichment | 10 new tests across 4 files. T-D1-NV-1, T-ARCH-1, T-ARCH-2, T-ARCH-2b, T-ARCH-3, T-PROP-1, T-INT-1, T-DEGRADE-LOG-1 fail (8 RED-by-design); T-D1-NV-2 + T-PROP-2 pass (regression-locks). Pytest 348 collected, 340 passed, 8 failed. |
| 2 | `fix` | `architect, verifier, orchestration` | WP-CORE-6 Architect populates supporting_sentence_ids end-to-end + D1 non-empty clause + degrade-log enrichment | 4 files modified; ~107 production LOC. Pytest 348 passed, 31 deselected. |
| 3 | `chore` | `artifacts` | WP-CORE-6 dev_doc + audit state update + F-22 backlog entry | `development_docs/WP-CORE-6-d1-verifier-non-vacuous.md` + `INDEX.md` ACTIVE row #7 + `CURRENT.md` + `improvements_backlog.md` F-21 → SHIPPED + F-22 (NEW) added + `findings/architect.md` §F-21 status + handoff doc + `decision_log.md` D-CODEX-REVIEW-WP-CORE-6. |
| 4 | `chore` | `planning` | WP-CORE-6 spec v2 + plan into git history | Spec v2 + plan doc landed. |

---

## Risks

1. **LLM behavior under new prompt.** New shape may produce more `ArchitectExtractionError` retries until the LLM consistently outputs the object format. Mitigation: prompt explicitly demonstrates the shape; 5-retry loop already absorbs transient shape failures. If Pro tier shows >20% retry rate post-fix, follow-up tightens the prompt. New strict-rejection of top-level list + bare-string dict (D-5b) raises the bar — Codex W-2 dispositioned this as correct big-bang behavior.
2. **Truncation no longer chops `[N] ` mid-prefix.** Per D-4 fix (Codex W-1), line-pair-aware truncation drops whole pairs from the middle. Any `[N]` the LLM sees is now guaranteed valid.
3. **Indices the LLM cites may be hallucinations.** D1 subset check catches IDs not in `scout_sentence_indices` (existing). D-3 non-empty clause catches "didn't cite any" (new). Together: comprehensive grounding gate **at the verifier layer**.
4. **D1 ERROR is logged-and-discarded, not enforced** (Codex C-2 / A2-d3-mask). Refiner exhausts → `pipeline.py` catches `RefinementExhaustedError` → continues with best-effort Specialist → pipeline ships a model. D-6 fix improves the log (full issues list now visible) but does NOT change pipeline outcome. **Net effect post-WP**: D1 is now an *honest signal*; pipeline still ships best-effort when D1 fails. True enforcement requires F-22 (NEW backlog) — Refiner extension to re-run Architect on Architect-stage errors.
5. **Specialist contract change** (`extract_per_context_details` accepts `List[ContextHypothesis]` instead of `List[str]`). Breaking for any external caller. Verified via grep: only `specialist_fn` in `analyze_document` calls this method in production. Test files (`test_specialist_per_context_loop.py`, etc.) may need fixture updates — verify in GREEN commit.
6. **EMSE methodology paper claims need updating.** Pre-WP: paper could only claim "D1 check exists." Post-WP: paper can claim "D1 check is non-vacuously evaluated on every run; failures are logged via the degrade-log but currently degrade to best-effort (F-22 tracks full enforcement)." Methods section needs honest update — flag for advisor in handoff.
7. **Test files new pattern.** Two new test files (`tests/test_architect_identify_contexts.py` + `tests/test_architect_id_propagation.py`). Mitigation: reuse `_make_architect()` pattern from `tests/test_architect_helpers.py:95-99` (patch `core.llm.gemini.genai.Client` + `GEMINI_API_KEY`).
8. **Synthesizer copy-behavior assumption.** D-2b relies on `merge.py:41` already calling `list(analysis.context.supporting_sentence_ids)`. T-PROP-2 regression-locks this; if a future synthesizer refactor breaks the copy, T-PROP-2 catches it.
9. **T-INT-1 integration test mocks 4 LLM calls.** Brittle if Architect prompt format changes — but the prompt format is precisely what WP-CORE-6 specifies, so T-INT-1 also serves as a contract test for the prompt structure.
10. **Test count grows substantially** (5 → 10). Mitigation: tests are surgical (each tests one invariant); collective runtime <2s expected.

---

## Open questions — resolved post-Codex review

All v1 OQs (OQ-1 through OQ-7) dispositioned via Codex review + spec-v2 design:

| v1 OQ | resolution |
|---|---|
| OQ-1 (D-3 fold-in justified?) | **YES, with reframe.** D-3 kept as defense-in-depth + honest signal (NOT enforcement, given Refiner-mask limitation). Paired with D-6 degrade-log to make the signal visible. Codex C-2 confirmed the mask reality; addressed via reframe + D-6 + F-22 backlog. |
| OQ-2 (strict shape vs graceful fallback?) | **STRICT** (Codex N-3 confirmed). Both old-dict-bare AND top-level-list branches removed per Codex W-2 + D-5b. |
| OQ-3 (0-contexts risk?) | **ACCEPTED** as existing contract. WP-CORE-5b precedent: empty pipeline output is `ArchitectExtractionError` (PipelineError), not degradation. Prompt instruction reinforces honest-rejection of ungrounded contexts. |
| OQ-4 (min_length=1 on Pydantic?) | **NO** — chose verifier-layer over schema-layer. Verifier ERROR has explicit `ungrounded_context` issue_type + suggestion; Pydantic ValidationError is generic. D1 stays the canonical layer. |
| OQ-5 (Refiner re-runs Architect?) | **DEFERRED to F-22** (NEW backlog entry per Codex W-4). Out of WP-CORE-6 envelope. Concrete revisit trigger: when Architect-stage errors become Refiner-actionable. |
| OQ-6 (truncation budget?) | **NO budget change** (50000 stays). Per Codex W-1 fix, truncation is now line-pair-aware, so the ~5% overhead per pair is absorbed by dropping fewer pairs (rather than chopping mid-pair). Marginal. |
| OQ-7 (index alignment?) | **VERIFIED** in audit (Codex N-1). `enumerate(domain_sentences)` in `identify_contexts` matches `SectionedSentence.index` because both use `enumerate(0)` over the same list. No off-by-one. |

New OQs raised by Codex and dispositioned in revision-history table above:
- **A6-srs-path (OQ-1 new)**: DEFERRED. Verifier issues runtime-only; adding `srs_path` requires `VerifierIssue` schema widening + 5-site threading. Revisit if F-22 promotes verifier issues to Refiner control-flow primary signals.

**No deferred WARNs.** 1 OQ deferred (A6-srs-path) with explicit scope-bounded rationale + concrete revisit trigger — qualitatively different from "future work" deferral.

---

## Pre-mortem (what could go wrong post-merge)

1. **Refiner exhaustion rate spikes** because D1 now actually fires. Mitigation: pipeline already has the `RefinementExhaustedError` → best-effort handler; runs degrade gracefully. Log the issue count via existing intermediate JSON dumps.
2. **JSON parse failures spike** because new shape is novel to the LLM. Mitigation: 5-retry loop. Watch retry rate post-merge.
3. **A reviewer asks "why didn't you also re-run Architect on D1 errors?"** Answer: out of scope. The current Refiner only knows how to re-run Specialist (`_re_run_specialist` at `pipeline.py:53-55`); extending it is a separate WP. F-21 fix is "populate the field + check it"; Refiner extension is "act on the check failure."
4. **EMSE Methods section needs updating** to credit non-vacuous D1. Note in handoff doc; advisor-facing.
5. **A future contributor sees `ContextHypothesis.supporting_sentence_ids: List[int] = Field(default_factory=list)` and adds a context with empty IDs in a test.** D1 will flag it. If they're tests for a non-D1 path, they may need to provide a non-empty stub. Mitigation: spec calls this out; new test fixture pattern in `test_architect_identify_contexts.py` demonstrates correct usage.

---

## Cross-references

- Finding: `.planning/pipeline_audit/findings/architect.md` §F-21 (NEW from WP-CORE-4 spec drafting; status updates from WP-CORE-5b handoff)
- Backlog: `.planning/pipeline_audit/improvements_backlog.md` row F-21 (will move to SHIPPED)
- Sibling specs (style/cadence):
  - `docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md` (contract-tightening pattern)
  - `docs/superpowers/specs/2026-05-21-wp-core-4-intermediate-save-observability-design.md` (PipelineError taxonomy + atomic commit cadence)
  - `docs/superpowers/specs/2026-05-21-wp-core-5b-synthesizer-empty-model-policy-design.md` (layer-cake defense pattern)
- Codex W-8 priority bump: `.planning/pipeline_audit/decision_log.md` D-CODEX-REVIEW-WP-CORE-5b
- Sibling reusable pattern: Specialist numbered-sentence prompt at `architect.py:584-586`
- Existing D1 test: `tests/test_verifier_deterministic.py` (currently covers happy-path subset check; non-vacuous tests added in this WP)
- AGENTS.md "no silent degradation"; "no backwards-compat hacks"; "Stable entrypoints"
- CLAUDE.md §"Verifier D1-D5 deterministic checks"
