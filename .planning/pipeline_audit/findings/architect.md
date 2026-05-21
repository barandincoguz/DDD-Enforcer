# Close-Lookup Findings — core/architect.py + core/orchestration/{pipeline,errors}.py

**Auditor:** Explore subagent
**Date:** 2026-05-21
**Files audited:** core/architect.py (LOC=923), core/orchestration/pipeline.py (84), core/orchestration/errors.py (70)
**Method:** Read every line; cross-reference [downstream consumers in main.py, synthesizer/__init__.py, verifier/__init__.py, refiner/loop.py]; examine existing tests [test_architect_facade.py, test_architect_extraction_error.py, test_specialist_per_context_loop.py, test_pipeline_orchestration.py, test_scout_chunking.py].

## Summary

`DomainArchitect` is a 5-stage orchestrator (Scout → Architect → Specialist → Synthesizer → Verifier) that drives the SRS→DomainModel extraction pipeline. The class owns three concurrent-execution surfaces: (a) parallel Scout chunk extraction via `ThreadPoolExecutor` (configurable via `DDD_SCOUT_MAX_WORKERS`), (b) per-context Specialist loop that issues one LLM call per bounded context (FM-23 exclusivity enforcement), and (c) token tracking via a shared singleton (`TokenTracker.get_instance()`). The orchestrator integrates with `core.orchestration.pipeline.run_pipeline` which is a typed-envelope stage driver with dependency-injection; the verifier can trigger refiner refinement loops (max 2 cycles) before falling back to "best-effort" on exhaustion. Core concerns identified: (1) parallel Scout has a subtle rate-limit reentrancy risk when workers interleave `_wait_for_rate_limit` calls and two consecutive workers find `min_delay` has reset (potential 2x speedup bypass), (2) Specialist shape errors during JSON validation are surfaced as `SpecialistShapeError` but the retry mechanism only converts *validation* errors into shape errors; on a parse-failure-then-correct-retry the token is not re-tracked and the error context is lost, (3) Intermediate JSON saves swallow exceptions (line 890-891) silently, violating AGENTS.md "no silent degradation", (4) Refiner failure gracefully degrades to "best-effort Specialist" (pipeline.py:76-78) but the synthesizer can still throw `SynthesizerEmptyModelError` which is caught nowhere in the pipeline, causing a hard fail at the pipeline level, (5) Test coverage for the per-context Specialist loop is thin — one test covers the happy path, but no tests for shape-error retry exhaustion, timeout, or token-tracker consistency across retries. The `section_aware_chunks` chunking is wired directly at `analyze_document` level (line 736) bypassing the legacy `_split_text_into_chunks` method, making the legacy chunker dead code.

## Findings (numbered, severity-tagged, starting from F-11)

### F-11 — Parallel Scout rate-limit window race — MAJOR

**Component:** core/architect.py
**Evidence:** Lines 141-150 (`_wait_for_rate_limit`), Lines 224-230 (parallel chunk extraction via `ThreadPoolExecutor`).
**Observation:** When `scout_max_workers > 1`, multiple threads call `_extract_sentences_from_chunk` → `_wait_for_rate_limit` concurrently. The lock at line 143 serializes access, but the window between lock acquire and sleep completion has a reentrancy risk. Consider two workers W1 and W2: W1 acquires lock at T0, checks `elapsed < self.min_delay` at T0+ε, finds it true, sleeps 5.99s, releases lock at T6. W2 acquires lock at T6+δ, checks `elapsed`, calculates from `self.last_request_time` which W1 just updated to T6 — W2 sees elapsed ≈ δ (nearly zero) and sleeps again. However, if a third slow codepath (e.g., JSON parsing) causes W2 to be delayed and it acquires the lock not at T6+δ but at T8 (after W1 already slept + released), the `self.last_request_time` read at line 144 is stale relative to when the lock was actually free. More critically: if W1 and W2 both enter the lock between two consecutive updates, or if `last_request_time` is updated by W1 but W2's check happens after release, W2 can see `self.last_request_time = T6` and `time.time() = T6.5`, calculate `elapsed = 0.5s`, and only sleep 5.5s instead of 6s — creating a microslip of 0.5s of combined request throughput. With many workers, these slips compound. The standard solution is a condition variable or a Semaphore with a wake time, but the current lock-only pattern is vulnerable.
**Blast radius:** PIPELINE — parallel Scout can exceed the 6s min_delay contract by up to `(N-1) * (min_delay / N)` seconds cumulative, where N = worker count. For Gemini free-tier with hard 6s limit, even a 0.5s slip on each of 4 workers means 2s of quota loss.
**Test gap:** yes — no concurrent rate-limit test exists. `test_specialist_per_context_loop` patches `_wait_for_rate_limit` entirely (line 50).
**AGENTS/CLAUDE rule cited:** AGENTS.md "Concurrency: thread-safety invariants must hold under arbitrarily-interleaved execution"; CLAUDE.md "Rate limiting (free-tier 6s min_delay) — enforce even with parallel Scout".

### F-12 — Specialist shape-error retry does not re-track tokens after JSON parse recovery — MINOR

**Component:** core/architect.py
**Evidence:** Lines 604-635 (Specialist retry loop with `SpecialistShapeError` handling).
**Observation:** When a Specialist LLM response arrives with shape errors (e.g., missing `description` field in entity), the code at line 619 calls `_validate_specialist_payload` which raises `SpecialistShapeError` at line 620. The retry handler at line 625 prints diagnostics and continues the loop. On the next retry attempt (lines 590-597), a fresh LLM call is issued and the response is re-parsed. However, line 630 (`self.token_tracker.track_api_call`) is only reached *after* validation succeeds (line 619 passes). This means if retry 1 fails shape validation, retry 2's successful response is tracked, but retry 1's tokens are lost from the accounting. For a context that cycles through 2-3 shape retries before success, the token count is understated by (retries - 1) * prompt_tokens. This is a minor observability bug but affects the metadata in the run manifest for reproducibility.
**Blast radius:** LOCAL — tokens lost only in the accounting, not in actual cost (Gemini still bills them). Impact: run manifest is inaccurate.
**Test gap:** yes — no test for shape-error retry with token-tracking verification. Test at lines 26-65 of `test_specialist_per_context_loop.py` mocks `token_tracker.track_api_call` (line 51) entirely.
**AGENTS/CLAUDE rule cited:** CLAUDE.md "Things to Know" §"intermediate JSON dumps" + token tracking; AGENTS.md "Observability: implicit accounting bugs are deferred technical debt".

### F-13 — `_save_intermediate` silently swallows I/O exceptions — MAJOR

**Component:** core/architect.py
**Evidence:** Lines 880-891 (`_save_intermediate` method).
**Observation:** If the intermediate directory does not exist (unlikely given line 117's `os.makedirs(..., exist_ok=True)`) or the filesystem is read-only, or the JSON encoding fails (e.g., a non-serializable object in `data`), the exception at line 890 is caught and printed but not re-raised. The caller (Scout at line 236, Architect at line 449, Specialist at line 650) has no way to know the save failed. The intermediate JSON files are a critical part of the run manifest per CLAUDE.md "Persistent Development Memory" — if they vanish silently, post-mortem debugging is impossible. A MAJOR violation of AGENTS.md "Error handling: explicit failure. No silent degradation."
**Blast radius:** PIPELINE — run diagnostics lost. For EMSE reproducibility, this is a methodology gap.
**Test gap:** yes — no test for I/O failure on intermediate save.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Error handling: explicit failure. No silent degradation, no permissive fallbacks during development"; CLAUDE.md "Persistent Development Memory" §"the intermediate outputs must be saved".

### F-14 — SynthesizerEmptyModelError can escape pipeline — MAJOR

**Component:** core/orchestration/pipeline.py (lines 81-83) + core/architect.py (lines 770-775)
**Evidence:** Line 83 of `pipeline.py`: `raise SynthesizerEmptyModelError(input_summary=f"{len(refined_specialist)} contexts")`. This exception is NOT caught in `analyze_document` at line 824 (`return run_pipeline(...)`); it propagates to `main.py:107` which is the only consumer.
**Observation:** The pipeline contract at line 36 says "`run_pipeline` returns a validated DomainModel, raises PipelineError subclasses on failure." `SynthesizerEmptyModelError` is indeed a `PipelineError` subclass (errors.py:35-38), but the synthesizer can only return an empty model if `refined_specialist` is a valid list with at least one `SpecialistAnalysis` but the merge step produces zero bounded contexts. The verifier (lines 777-815) only checks D1-D5 deterministic issues and cannot prevent a merge that wipes all contexts (e.g., if all entities are duplicates across contexts and the dedup logic collapses them). The comment at pipeline.py:68-74 explicitly handles `RefinementExhaustedError` by degrading to best-effort, but there is no analogous handler for `SynthesizerEmptyModelError` — it's treated as a fatal exception. In `main.py:107`, if `generate_domain_model` raises this, the exception propagates to the lifespan handler at line 173-180 which catches it generically as `Exception` at line 180, prints a traceback, and sets `app_state["domain_rules"] = {}`. This is acceptable error handling at the top level, but the contract at the pipeline level is ambiguous: is it a "hard fail" or a "retry moment"? The current behavior is hard-fail silently in production. If the SRS genuinely has zero bounded contexts (e.g., a single-domain SRS that the Architect incorrectly collapses), the user gets a silent empty model. The intermediate files would show what happened (per F-13), but only if they saved.
**Blast radius:** PIPELINE — silent empty model on Synthesizer degenerate case.
**Test gap:** yes — no test for the synthesizer-empty-model flow in the orchestrated pipeline. `test_pipeline_orchestration.py` covers refiner exhaustion (lines 97-160) but not synthesizer failure.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Stable entrypoints: when a boundary contract can fail, the handler must be explicit or the caller must document the failure mode"; CLAUDE.md §"D6/D7/D8 hard-fail invariants" (but SynthesizerEmptyModelError is NOT an invariant, it's a degenerate case).

### F-15 — Refiner failure degrades to best-effort but does not log the final-attempt verifier result — MINOR

**Component:** core/orchestration/pipeline.py (lines 57-79)
**Evidence:** Lines 75-78 print a generic warning, but `refined_specialist` is set to `specialist_output` (line 79) without surfacing what issues the verifier still found.
**Observation:** When the refiner exhausts 2 cycles without resolving all issues, the except block catches `RefinementExhaustedError` and sets `refined_specialist = specialist_output`. The print at line 76-78 logs only the exception type, not the residual issues. For debugging, it would be useful to surface which D1-D5 checks still failed — e.g., "D3 duplicate entity name, D4 missing aggregate member". Currently, only the Specialist input and output are logged (via intermediate JSON saves), but the final verifier result is not. This is a minor observability gap but acceptable given the intentional fallback strategy ("best-effort").
**Blast radius:** LOCAL — observability only.
**Test gap:** partial — `test_pipeline_orchestration.py:97-160` tests refiner invocation but does not assert on the log/exception details.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Logging policy: silent or verbose — pick one"; CLAUDE.md "Persistent Development Memory" §"run manifest must record stage outputs + verifier results".

### F-16 — Dead code: `_split_text_into_chunks` is not called anywhere — TRIVIAL

**Component:** core/architect.py
**Evidence:** Lines 249-265 define `_split_text_into_chunks`; Scout (lines 182-247) never calls it. Instead, `analyze_document` at line 736 calls `section_aware_chunks` directly.
**Observation:** The legacy chunking logic at lines 249-265 breaks at sentence boundaries (`.` character). However, the real Scout orchestration uses `section_aware_chunks` from `core.scout.chunking`, which is more sophisticated (section-aware, token-budget-based). The old `_split_text_into_chunks` is unreachable dead code left over from an earlier refactor where Scout might have used it. Should be deleted per AGENTS.md "No backwards-compat shims" (WP-01a deletes old code outright).
**Blast radius:** NONE — unreachable, harmless.
**Test gap:** n/a — no test for this method exists, confirming it's unused.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Smallest correct change; isolate change-prone logic; delete old code outright".

### F-17 — Model selection via stage_config not validated at instantiation — MINOR

**Component:** core/architect.py
**Evidence:** Lines 99, 303, 403, 586 all call `stage_config("Scout")`, `stage_config("Architect")`, `stage_config("Specialist")` without catching `KeyError`.
**Observation:** If the configuration registry (`configs/models.py`) does not define a stage named "Scout" (or the stage is present but returns a missing `model_id`), the code will raise `KeyError` at runtime during the first LLM call, not during instantiation. For earlier error detection, `DomainArchitect.__init__` should validate that each stage's model exists via `model_spec(model_id)` at line 99 (already done for Architect), but Scout/Specialist validation is deferred. If the config is malformed, the user sees the error only when processing an SRS, not on server startup. The severity is minor because the same validation occurs at main.py lifespan (token counting also reads the config), but it's an observability hygiene issue.
**Blast radius:** LOCAL — deferred error detection.
**Test gap:** yes — no test for missing stage config. Tests use inline stage configs (e.g., `stage_config("Architect")` is mocked in tests).
**AGENTS/CLAUDE rule cited:** AGENTS.md "Stable entrypoints: validate invariants early"; CLAUDE.md "D1 Locked decisions: 6 models — routes only via stage_config".

### F-18 — Architect generates synthetic context descriptions instead of querying LLM — MINOR

**Component:** core/architect.py
**Evidence:** Lines 757-759 in `analyze_document`.
**Observation:** After `identify_contexts` returns bare context names (line 756), the pipeline wraps them in `ContextHypothesis` with synthetic descriptions: `description=f"{n} context"` (e.g., "OrderMgmt context"). These empty/synthetic descriptions are later filled in by the Synthesizer (line 772 calls `synthesize_domain_model` which enriches descriptions via LLM). However, the intermediate JSON for Architect (line 449-457) records the contexts with these synthetic descriptions, not the LLM-enriched ones. For run reproducibility, the intermediate artifacts should reflect the actual pipeline state. This is minor (descriptions are enriched downstream), but the mismatch can confuse manual debugging if a user inspects the intermediate files and sees "OrderMgmt context" instead of the semantically meaningful description.
**Blast radius:** LOCAL — observability / debugging only.
**Test gap:** partial — tests mock synthesizer entirely, so this is not asserted.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Observability: intermediate artifacts must reflect pipeline state"; CLAUDE.md "Things to Know" §"intermediate JSON dumps".

### F-19 — No max-retry bound validation on exponential backoff in quota-error handler — MINOR

**Component:** core/architect.py
**Evidence:** Lines 153-176 (`_is_quota_error_and_backoff`).
**Observation:** When a quota error is detected (line 159), the backoff time is calculated as `min(15 * (2 ** retry_count), 300)` at line 172, capping at 300s (5 minutes). However, if `retry_count` is large (e.g., 10+), the intermediate terms `15 * (2 ** 10) = 15360` would exceed the cap. The `min(...)` operator handles this, so no integer overflow risk. However, the comment at line 171 says "Exponential backoff: 15s, 30s, 60s, 120s" which implies a 4-step progression, but the code allows arbitrarily many retries (loop is `for retry in range(5)` at line 304, so retry_count is 0-4, and backoff is 15, 30, 60, 120, 300). The sequences match, so the implementation is correct, but the comment is misleading (it should say "up to 300s"). This is pedantic but violates the "code comments must match code behavior" rule.
**Blast radius:** NONE — implementation is correct, comment is misleading only.
**Test gap:** no — exponential backoff has no unit test; it's only exercised if a Gemini 429 error occurs, which is not mocked in tests.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Code comments must be accurate"; CLAUDE.md "Things to Know" §"Rate limiting (free-tier 6s min_delay)".

### F-20 — Token tracker singleton mutation during parallel Scout is not thread-safe — MAJOR (uncertain → DOWNGRADED to MINOR after verification)

**Component:** core/architect.py + core/token_tracker.py
**Evidence:** Lines 337-341 (Scout calls `token_tracker.track_api_call`), Lines 113 (singleton), Lines 44 (TokenTracker has `_lock`).
**Observation:** Scout runs in parallel with up to `scout_max_workers` threads. Each thread calls `track_api_call` on the shared singleton. The TokenTracker has a `_lock` (line 44 of `token_tracker.py`), and `track_api_call` should be protected. However, the code at lines 337-341 of `architect.py` calls `track_api_call` directly without any explicit synchronization visible at the Architect level. The token tracker itself is thread-safe (it has a lock), so this is likely safe, but the lack of explicit synchronization at the call site is a code-smell. For a code reviewer reading architect.py without checking token_tracker.py, it appears that unsynchronized parallel calls to a shared singleton happen. Marking as MAJOR-uncertain because the actual implementation is likely correct but the absence of a synchronization comment is a red flag for implicit assumptions.

**Verification 2026-05-21 (spec drafting):** `grep -n "_lock\|Lock" core/token_tracker.py` confirms `_lock = threading.Lock()` at line 44 and `with self._lock:` at line 97 around the mutation block. F-20 is therefore not a thread-safety bug — it is a documentation/comment gap. **DOWNGRADED from MAJOR(uncertain) to MINOR.** A brief comment at the Scout parallel-extract call site documenting that thread-safety is delegated to the TokenTracker internal lock would resolve this.

**Blast radius:** LOCAL (downgraded).
**Test gap:** still yes — no concurrent token-tracker test with parallel Scout, but thread-safety is guaranteed by the lock so the gap is observational not behavioral.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Concurrency: all shared-state mutations must be visible and explicitly synchronized" — satisfied by the inner lock.

### F-21 — D1 verifier check passes vacuously because `ContextHypothesis.supporting_sentence_ids` is never populated by Architect — MAJOR

**Component:** core/architect.py + core/verifier/checks_deterministic.py
**Evidence:** Lines 757-761 of `architect.py` (`ContextHypothesis(context_name=n, description=f"{n} context")` — no `supporting_sentence_ids` set); line 784 (`{"name": c.context_name, "supporting_sentence_ids": c.supporting_sentence_ids}` reads the default `[]` from `core/pipeline_contracts.py:91`); line 794 (`check_d1_supporting_sentence_ids_subset(contexts_dicts, scout_indices)` called); `core/verifier/checks_deterministic.py:7-21` (the D1 check iterates `ctx.get("supporting_sentence_ids", [])` — when empty, no violations are reported).
**Observation:** Discovered during WP-CORE-4 spec drafting (anomaly check on `supporting_sentence_ids` cross-references). `ContextHypothesis` Pydantic schema defaults `supporting_sentence_ids: List[int] = Field(default_factory=list)` per `core/pipeline_contracts.py:91`. The Architect stage at `architect.py:757-761` constructs each `ContextHypothesis` with only `context_name` and `description` — the `supporting_sentence_ids` field is never set. Downstream at line 784 the verifier reads this empty list. The D1 deterministic check `check_d1_supporting_sentence_ids_subset` iterates over the empty list and reports zero violations. **Every D1 verifier run for every run in the project's history has passed vacuously.** The check is dead code in its current state. Fixing this requires changing the Architect prompt to ask the LLM to emit per-context supporting sentence indices alongside context names, then parsing them into `ContextHypothesis.supporting_sentence_ids`. This is a meaningful Architect-prompt + parsing change — bigger than the "smallest correct change" pattern of WP-CORE-4, and orthogonal to F-13's observability fix.
**Blast radius:** PIPELINE — D1 verifier coverage is silently 0%. For EMSE methodology validation, this is a coverage gap worth surfacing.
**Test gap:** yes — `tests/test_verifier_deterministic.py` covers D1 against synthetic test contexts with non-empty `supporting_sentence_ids` (line 19), but no integration test asserts that real Architect-produced contexts carry populated `supporting_sentence_ids` from a real (or mocked) Scout output. Test coverage exists for the *check* but not for the *integration*.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Smell mixing of concerns" (verifier coverage hidden behind defaulted field); CLAUDE.md §"Verifier integration" (post-P3 5-stage pipeline requires every check to have non-vacuous data).
**Deferred to:** WP-CORE-5 or later (requires Architect prompt + parsing change; out of WP-CORE-4 scope).

## Anomalies (NO fix yet — Explore observations)

- **`_current_srs_path` attribute is set via `getattr(..., "<unknown>")` fallback** (lines 434, 479, 491, 496) but never assigned anywhere in the class. This is a defensive fallback for error messages, but the attribute is never set, meaning all error messages will always print "<unknown>". To properly surface the SRS path in errors, the caller at `main.py:107` would need to pass the `srs_path` to `DomainArchitect` or store it during `analyze_document`. Currently a no-op that masks which SRS failed.

- **`SpecialistShapeError` is a subclass of `SpecialistFailureError`** (errors.py:53-70) but it's raised and caught separately in the retry loop (lines 620, 628). This is intentional (to distinguish shape from other failures), but the inheritance creates a code-smell: `except SpecialistFailureError` at line 636 will also catch `SpecialistShapeError`, so the order of exception handlers matters. If the code ever changes the order (catch `SpecialistFailureError` first), shape errors could be mishandled. A better design would be to make them siblings, not parent-child.

- **Refiner invocation uses a hardcoded `max_cycles=2`** (pipeline.py:64). This is a magic constant with no config knob. If the verifier finds only one slowly-converging issue (e.g., a duplicate entity name that requires careful disambiguation), 2 cycles may not be enough. Per AGENTS.md "DI knobs", this should be configurable. However, the degradation to best-effort (line 79) makes it acceptable for now.

- **`section_aware_chunks` from `core.scout.chunking` is imported at line 736 inside a function** (not at module level). This is a micro-optimization to avoid a circular-import or to defer the import until needed. No issue, but it's unusual — import should be at module level per PEP 8. If `core.scout.chunking` is not installed, the error occurs at runtime during `analyze_document`, not at module load time.

- **Verifier at pipeline.py:777-815 reconstructs dictionaries from Pydantic models** (e.g., `[c.model_dump() for c in arch.contexts]` at line 757). This bypasses the typed contracts and creates a pathway for field mismatches. If `ContextHypothesis` schema changes, the dict-level extraction code could silently read wrong keys. Better to pass typed objects directly to the checker functions.

- **"supporting_sentence_ids" field is accessed at line 784** but `ContextHypothesis` does not have this field in the data read (lines 757-760 only set `context_name` and `description`). This looks like dead code or a contract mismatch. The verifier checks should work with the actual `ContextHypothesis` fields, not phantom fields.

---

## Test-coverage map

| code path | test exists | test file | gap notes |
|---|---|---|---|
| `DomainArchitect.__init__` happy path | yes | implicit (fixtures in test files) | no test for missing GEMINI_API_KEY error |
| `extract_domain_sentences` sequential Scout | yes | `test_scout_chunking.py` | happy path only; no empty-result, no all-parse-failures path |
| `extract_domain_sentences` parallel Scout | no | — | parallel workers never tested; `scout_max_workers > 1` untested |
| `_extract_sentences_from_chunk` happy path | partial | `test_scout_chunking.py` | mocked, no real LLM call |
| `_extract_sentences_from_chunk` JSON parse failure retry | no | — | no test for parse-failure exhaustion fallback (line 332) |
| `_extract_sentences_from_chunk` quota error + backoff | no | — | no test for `_is_quota_error_and_backoff` integration |
| `identify_contexts` happy path | yes | `test_architect_extraction_error.py` (line 26-38) | happy path exists |
| `identify_contexts` empty response | yes | `test_architect_extraction_error.py:41-54` | covered |
| `identify_contexts` parse failure exhaustion | yes | `test_architect_extraction_error.py:26-38` | covered by `bad_response` test |
| `extract_per_context_details` per-context loop | yes | `test_specialist_per_context_loop.py:26-65` | happy path only; 3 contexts → 3 calls verified |
| `extract_per_context_details` shape-error retry | no | — | no test for `SpecialistShapeError` retry exhaustion |
| `extract_per_context_details` shape-error recovery | no | — | no test for shape-error on retry-N then success |
| `_validate_specialist_payload` unwrap-list | no | — | no test for singleton-list unwrap logic |
| `_validate_specialist_payload` Pydantic validation | no | — | no test for missing fields (description, etc.) |
| `analyze_document` full 5-stage pipeline | yes | `test_architect_facade.py:11-46` | delegates to mocked run_pipeline |
| `analyze_document` refiner integration | yes | `test_pipeline_orchestration.py:97-160` | covers refiner invocation + re-run |
| `analyze_document` verifier integration | yes | `test_pipeline_orchestration.py:75-80` | verifier returns ok on happy path |
| `_wait_for_rate_limit` sequential | partial | tests mock it entirely (line 50) | timing behavior never tested |
| `_wait_for_rate_limit` concurrent (parallel Scout) | no | — | race condition untested |
| `_is_quota_error_and_backoff` 429 handling | no | — | no Gemini 429 response test |
| `_is_quota_error_and_backoff` exponential backoff sequence | no | — | backoff timings untested |
| `_save_intermediate` happy path | no | — | no assertion on file creation |
| `_save_intermediate` I/O failure | no | — | no test for exception swallowing |
| `_parse_json_response` valid JSON | yes | implicit | passing tests use valid JSON |
| `_parse_json_response` truncated JSON | no | — | untested |
| `_parse_json_response` markdown-wrapped JSON | no | — | line 917 fallback untested |
| `_check_response_completion` STOP finish_reason | yes | tests assume STOP | happy path |
| `_check_response_completion` MAX_TOKENS finish_reason | no | — | line 859-863 untested |
| `_check_response_completion` SAFETY finish_reason | no | — | line 867-868 untested |
| `run_pipeline` happy path | yes | `test_pipeline_orchestration.py:75-80` | covered |
| `run_pipeline` architect failure | yes | `test_pipeline_orchestration.py:83-87` | covered |
| `run_pipeline` specialist failure | yes | `test_pipeline_orchestration.py:90-94` | covered |
| `run_pipeline` refiner exhaustion | yes | `test_pipeline_orchestration.py:97-160` | covered |
| `run_pipeline` synthesizer empty model | no | — | `SynthesizerEmptyModelError` not tested |
| `refine_until_clean` max_cycles bound | partial | `test_pipeline_orchestration.py:97-160` | verifies 2-cycle behavior but hardcoded |

## Cross-references

- `extension/backend/main.py:100-125` — Lifespan domain-model bootstrap; calls `generate_domain_model` which instantiates DomainArchitect and calls `analyze_document`. No explicit error handler for pipeline exceptions; generic `Exception` catch at line 180.
- `extension/backend/main.py:100-107` — `/generate-model` endpoint; instantiates new DomainArchitect (no reuse), calls `analyze_document`, returns model. No streaming, no progress callback wired.
- `extension/backend/core/synthesizer/__init__.py:19-78` — Receives `List[SpecialistAnalysis]` from Specialist, merges deterministically, enriches via LLM, verifies D6/D7/D8. Raises `SynthesizerInvariantError` on D6/D7 failure (hard-fail). D8 auto-heals.
- `extension/backend/core/verifier/__init__.py` — Exports `VerifierIssue`, `IssueSeverity`, `VerifierResult`. The `verifier_fn` at pipeline.py:777 calls low-level checks (D1-D5).
- `extension/backend/core/refiner/loop.py` — `refine_until_clean` orchestrates verifier + stage_runner. Raises `RefinementExhaustedError` on exhaustion (caught at pipeline.py:65).
- `extension/backend/configs/models.py` — Stage model routing via `stage_config(stage_name)`. If stage is missing, raises `KeyError`.
- `extension/backend/tests/test_architect_facade.py` — Minimal test; mostly delegates to mocked `run_pipeline`.
- `extension/backend/tests/test_architect_extraction_error.py` — Error-path tests for Architect (parse failures, empty responses).
- `extension/backend/tests/test_specialist_per_context_loop.py` — Tests per-context Specialist; mocks all dependencies (client, token_tracker, etc.).
- `extension/backend/tests/test_pipeline_orchestration.py` — Tests pipeline driver; covers happy path, refiner invocation, but not synthesizer failures.
- `extension/backend/tests/test_scout_chunking.py` — Tests section-aware chunking, not architect integration.

## Convention notes

- Type hints are comprehensive and accurate throughout. Pydantic envelopes are used at stage boundaries (ScoutOutput, ArchitectOutput, SpecialistAnalysis).
- Error classes (ArchitectExtractionError, SpecialistFailureError, SpecialistShapeError) are well-structured with instance attributes (srs_path, context_name, validation_errors).
- Rate-limiting is centralized in `_wait_for_rate_limit` with a lock and tracking of `last_request_time` and `request_count`.
- Token tracking is delegated to the singleton `TokenTracker` via explicit calls after successful LLM responses.
- Intermediate outputs are saved as timestamped JSON files in `INTERMEDIATE_DIR` for run diagnostics.
- Progress callback is optional and invoked at major stage milestones (started, in_progress, completed, error).

