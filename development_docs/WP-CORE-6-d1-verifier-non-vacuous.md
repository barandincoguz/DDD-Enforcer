# WP-CORE-6 — D1 verifier non-vacuous: Architect populates `supporting_sentence_ids` end-to-end

**Status:** SHIPPED 2026-05-21
**Branch:** `main`
**Commit range:** `fd7f203` (RED) → `a86bbbb` (GREEN) → `{doc-sha}` (DOC, this commit) → `{planning-sha}` (PLANNING)
**Pre-WP HEAD:** `9608495` (WP-CORE-5b final state)
**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-6-d1-verifier-non-vacuous-design.md` (v2, post-Codex)
**Plan:** `docs/superpowers/plans/2026-05-21-wp-core-6-d1-verifier-non-vacuous.md`
**Parent finding:** `.planning/pipeline_audit/findings/architect.md` finding **F-21** (MAJOR)
**Test delta:** 338 → 348 (+10 tests, zero regression)

## TL;DR

The D1 verifier check (`check_d1_supporting_sentence_ids_subset`) has passed vacuously for every project run in history because Architect's `identify_contexts` produced bare context-name strings — never sentence indices — leaving `ContextHypothesis.supporting_sentence_ids` at its Pydantic default `[]`. Empty list ⊆ any set, so D1's subset check always returned `[]` issues. Specialist then rebuilt `ContextHypothesis` fresh (discarding any IDs Architect might have had), so the final `DomainModel.bounded_contexts[].supporting_sentence_ids` was also always empty.

This WP closes the loop end-to-end. Architect now numbers sentences and asks the LLM for `[{"name": ..., "supporting_sentence_ids": [...]}]` object array. The IDs propagate Architect → Specialist (signature widened from `List[str]` to `List[ContextHypothesis]`) → Synthesizer merge (already had the copy code; just feeds non-empty input now) → final DomainModel. D1 also flags empty IDs as ungrounded_context ERROR (honest signal — full enforcement requires F-22 Refiner extension).

## Motivation

### F-21 is LIVE (unlike F-11 and F-14)

Unlike WP-CORE-5 (F-11 parallel Scout race — dormant) and WP-CORE-5b (F-14 SynthesizerEmptyModelError — dormant per Architect upstream guard), F-21 fires on every single `analyze_document` invocation. EMSE methodology paper claims "D1 catches contexts that cite sentences Scout did not emit" — empirically vacuous since the cited sentences set has always been empty.

### Codex review reframed scope substantially

Spec v1 proposed a narrow fix: change `identify_contexts` prompt and parsing. Codex xhigh review (`decision_log.md` D-CODEX-REVIEW-WP-CORE-6) flagged 4 CRITICALs that revealed the v1 scope was insufficient:

- **A7 (final-model-loss)** — Specialist receives `List[str]` names only (architect.py:788), rebuilds `ContextHypothesis(context_name=ctx_name, description="")` fresh at line 707 with default empty IDs. Even if Architect populates IDs, they're discarded at the Specialist boundary. **Final DomainModel still vacuous post-v1.**
- **A2-d3-mask** — Refiner exhausts on D1 ERROR → pipeline.py catches and degrades to best-effort → pipeline ships a model anyway. D-3's new D1 ERROR is logged-and-discarded, not enforced.
- **A3-integration-gap** — 5 tests insufficient; no E2E test validating IDs survive Architect → Specialist → Synthesizer.
- **A5-risk4** — `RefinementExhaustedError.issues` discarded by `print(type(exc).__name__)`; D1 errors not visible in run manifest.

Plus 4 WARN findings (truncation chops mid-prefix, top-level-list bypass, scope mis-scoping, F-22 backlog hygiene). All CRITICAL+WARN handled inline in spec v2.

## Architectural decisions

### D-1 — Architect prompt numbers sentences + requests object-array shape

`identify_contexts` (architect.py:419+) now:

1. Builds `numbered_pairs = list(enumerate(domain_sentences))`.
2. Truncates via new line-pair-aware helper (D-4).
3. Formats as `"\n".join(f"[{i}] {s}" for i, s in truncated_pairs)`.
4. Prompt explicitly demands `{"contexts": [{"name": ..., "supporting_sentence_ids": [...]}]}` shape with constraint "every context MUST cite ≥1 supporting_sentence_id".

Specialist already used this `[N] sentence` numbering pattern (architect.py:668-670) — Architect now matches.

### D-2 — Signature change: `identify_contexts -> List[Dict[str, Any]]`

Return type widens from `List[str]` to `List[Dict[str, Any]]` with keys `name` (str) + `supporting_sentence_ids` (List[int]). Single production caller (`architect_fn`) updates accordingly. No backwards-compat shim per AGENTS.md "no backcompat hacks" and Codex N-3 disposition.

### D-2b — Specialist contract widened to preserve IDs (Codex C-1)

`extract_per_context_details` signature changes from `contexts: List[str]` to `contexts: List[ContextHypothesis]`. The fresh-rebuild at the old line 707 (`ctx = ContextHypothesis(context_name=ctx_name, description="")`) is DELETED — the input `ctx` is used directly. This preserves Architect's `supporting_sentence_ids` into `SpecialistAnalysis.context`.

Synthesizer (`core/synthesizer/merge.py:41`) already calls `list(analysis.context.supporting_sentence_ids)` into `BoundedContext.supporting_sentence_ids` — no synthesizer change; it just receives non-empty input now.

T-INT-1 regression-locks the full Architect → Specialist → Synthesizer → DomainModel propagation.

### D-3 — D1 non-empty clause: honest signal, not enforcement (Codex C-2 reframe)

`check_d1_supporting_sentence_ids_subset` (verifier/checks_deterministic.py:7-46) adds a clause:

```python
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
```

**Limitation explicit per Codex A2-d3-mask**: Refiner exhausts → `pipeline.py` degrade → best-effort → pipeline ships. D-3 ERROR is logged via D-6's enriched log, but pipeline still ships a (now-honestly-flagged) model. True fail-fast enforcement requires F-22 (NEW backlog) — Refiner to re-run Architect on architect-stage errors.

Defense-in-depth value: protects against future Architect prompt regression silently dropping the field again.

### D-4 — Line-pair-aware truncation (Codex W-1)

New helper `_truncate_numbered_pairs(pairs, max_chars, head_ratio=0.6)` in architect.py:86. Drops whole `(idx, text)` pairs from the middle rather than slicing characters across a `[N] ` prefix boundary. Any `[N]` the LLM sees is guaranteed valid. Old `_truncate_with_head_tail` unchanged (still used by callers that operate on opaque text).

### D-5b — Strict-shape parser (Codex W-2)

Parser at architect.py:501+ tightened to validate:
- `result` is `dict` with `"contexts"` key
- `result["contexts"]` is non-empty `list`
- every element is `dict` with `name: str` + `supporting_sentence_ids: List[int]`

The legacy `elif isinstance(result, list) and len(result) > 0:` branch (which accepted top-level lists `["X", "Y"]`) is **removed**. Pre-WP-CORE-6 the parser had two success branches; post-WP-CORE-6 only the strict dict-wrapper branch remains. Anything else → retry → exhaustion → `ArchitectExtractionError`.

### D-6 — Degrade-log emits full issues list (Codex C-4)

`core/orchestration/pipeline.py:65+` splits `except Exception` into `except RefinementExhaustedError as exc:` (with enriched issue-list log) and fallback `except Exception as exc:` (unchanged generic log). The enriched log emits `severity@stage:location: message` for every `exc.issues` entry. D1 errors are now visible in the run manifest's stdout dump.

### D-7 — F-22 backlog entry (Codex W-4)

New backlog row added to `.planning/pipeline_audit/improvements_backlog.md`:

> F-22 | core/orchestration/pipeline.py + core/refiner/loop.py | Refiner only re-runs Specialist; Architect-stage verifier failures degrade to best-effort. | MAJOR | M-L | PIPELINE | OPEN

Tracks the Refiner extension needed for true D1 enforcement. Deferred to its own WP.

## File-level changes

| file | change | LOC |
|---|---|---|
| `core/architect.py` | Prompt + parser + signature; `_truncate_numbered_pairs` helper; architect_fn/specialist_fn thread IDs; extract_per_context_details preserves input ctx | +193 / -50 |
| `core/verifier/checks_deterministic.py` | D1 non-empty clause + docstring | +25 / -5 |
| `core/orchestration/pipeline.py` | RefinementExhaustedError branch with issues-list log | +21 / -8 |
| `tests/test_architect_identify_contexts.py` | NEW — 4 tests for return shape + strict parser + prompt numbering | +112 |
| `tests/test_architect_id_propagation.py` | NEW — 3 tests for Specialist preservation + synthesizer merge + E2E integration | +185 |
| `tests/test_verifier_deterministic.py` | T-D1-NV-1 (empty IDs ERROR) + T-D1-NV-2 (subset regression-lock) | +28 |
| `tests/test_pipeline_orchestration.py` | T-DEGRADE-LOG-1 (issue list in degrade-log) | +41 |
| `tests/test_specialist_per_context_loop.py` | Fixture update — ContextHypothesis instead of str | +8 / -3 |
| `tests/test_intermediate_save.py` | Fixture update — strict-shape mocked parse + ContextHypothesis arg | +14 / -3 |

**Net diff**: 9 files, +627 / -69 LOC. Of the +220 in production files, ~50 LOC are doc comments explaining the layer-cake.

## Methodology applied

- **TDD with genuine RED-fail**: 10 new tests; RED commit `fd7f203` landed 8 failing tests (T-D1-NV-1, T-ARCH-1, T-ARCH-2, T-ARCH-2b, T-ARCH-3, T-PROP-1, T-INT-1, T-DEGRADE-LOG-1); GREEN commit `a86bbbb` turned all 8 green plus updated 2 pre-existing fixture sites that broke under the signature change.
- **Codex xhigh adversarial review**: 4 CRITICAL + 4 WARN + 6 NIT + 1 OQ. All CRITICAL+WARN handled inline. 1 OQ deferred with explicit revisit trigger (post-F-22) — **the 4-iteration zero-deferred streak (CORE-3/4/5b/6) ends here by design**, not drift. The deferred OQ has scope-bounded rationale and concrete promotion criterion.
- **Honest-framing precedent**: per Codex W-7, spec §Motivation explicitly frames the WP as "contract cleanup for paper-methodology integrity" not "production hardening" (because main.py still catches generic `Exception`).
- **Atomic Conventional Commits** with Claude trailer. RED → GREEN → DOC → PLANNING cadence matches WP-CORE-3/4/5b.
- **Production reachability subsection** in spec §Motivation (loop discipline lesson from iteration 4) — confirmed F-21 LIVE before drafting; contrasts WP-CORE-5/5b's dormant-in-production findings.

## Empirical results

- **Test count**: 338 → 348 (+10, all green at GREEN HEAD `a86bbbb`).
- **Regression count**: 0 (verified by full pytest at GREEN commit).
- **Pre-WP behavior**: every project run had `ContextHypothesis.supporting_sentence_ids = []`; D1 check passed vacuously; final `DomainModel.bounded_contexts[].supporting_sentence_ids = []`.
- **Post-WP behavior**: Architect populates IDs via prompt; IDs survive Architect → Specialist → Synthesizer → final DomainModel; D1 evaluated non-vacuously; degrade-log enriched with full issue details.
- **EMSE methodology impact**: positive and substantial. Methods section claim "D1 catches contexts citing un-emitted sentences" now holds empirically. Honest disclosure: when D1 fires, pipeline still degrades to best-effort due to F-22; paper Methods section should note this nuance.

## Limitations + follow-ups

1. **F-22 (NEW, HIGHEST PRIORITY) — Refiner stage-aware re-runs.** Refiner currently only re-runs Specialist (`pipeline.py:53-55`). Architect-stage verifier failures (D1 `ungrounded_context` ERROR) cannot be auto-corrected; pipeline degrades to best-effort via `RefinementExhaustedError` handler. F-22 tracks extending Refiner to dispatch re-runs by failing stage. Without F-22, D-3 is an honest signal but not enforcement. **Iteration-6 candidate.**
2. **A6-srs-path OQ deferred.** Adding `srs_path` to `VerifierIssue` requires schema widening + 5-site threading in `verifier_fn`. Verifier issues are currently runtime-only (intermediate JSON dumps + degrade-log); not user-facing persisted artifacts. Revisit if F-22 promotes verifier issues to Refiner control-flow primary signals.
3. **EMSE paper Methods section needs honest update.** Pre-WP: claim "D1 check exists" was technically true but empirically vacuous. Post-WP: "D1 check is non-vacuously evaluated on every run; failures are logged via the degrade-log; F-22 tracks full enforcement." Flag for advisor (Hoca) in handoff.
4. **Test count grew substantially** (5 → 10 new). All surgical; collective runtime <2s. Synthesizer regression-lock (T-PROP-2) guards against future refactor accidentally dropping the merge.py copy-behavior.
5. **Strict-shape parser raises retry rate.** LLM may occasionally emit old shape; 5-retry loop absorbs. If Pro tier post-fix shows >20% retry rate on Architect stage, follow-up tightens the prompt.

## Cross-references

- Sibling pattern: [[WP-CORE-4-intermediate-save-observability]] — same `srs_path` propagation pattern; layer-cake defense
- Sibling pattern: [[WP-CORE-5b-synthesizer-empty-model-policy]] — same `PipelineError` taxonomy + Codex C-1 layer-cake pattern
- Predecessor in iteration 5: [[wp-core-5-abandoned]] (F-11 spec; banner-marked ABANDONED — dormant in production)
- Codex review: `.planning/pipeline_audit/decision_log.md` D-CODEX-REVIEW-WP-CORE-6
- Reusable Specialist numbered-sentence pattern: `core/architect.py:668-670`
- AGENTS.md "Error handling: explicit failure. No silent degradation."
- CLAUDE.md §"Verifier D1-D5 deterministic checks"
- F-22 backlog entry: `.planning/pipeline_audit/improvements_backlog.md` orchestrator table
