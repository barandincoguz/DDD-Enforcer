# WP-01c: Token Tracking + Cost Telemetry (multi-provider extension)

**Owner:** Ali
**Depends-on:** [WP-01a]
**Effort:** S (existing `token_tracker.py` covers Gemini fully; only need to extend pricing table + plumbing through other providers)
**Status:** TODO
**Addresses instructor feedback:** [Hoca-1] (enables RQ2 cost column in Table 7)

## Goal

Make sure that for every LLM call, regardless of provider, we log `prompt_tokens`, `completion_tokens`, and a verifiable `cost_usd`. RQ2's cost column requires this — otherwise the Pareto frontier figure (`fig:rq2_pareto`) cannot be drawn. **Existing `extension/backend/core/token_tracker.py` already handles this for Gemini with verified pricing** (commit `a97d3e7`); WP-01c extends it to OpenAI, Anthropic, and OSS-local.

## Acceptance criteria

- [ ] `configs/pricing.yaml` lists USD-per-1M tokens for each model in `configs/scope.yaml`. For OSS-local, marks `mode: compute-only` and stores GPU-hour cost as a sensitivity figure (not a USD number).
- [ ] `TokenTracker.track_api_call(response, ...)` works for all 4 provider response shapes (Gemini's `usage_metadata`, OpenAI's `response.usage`, Anthropic's `response.usage`, Ollama's whatever-it-returns). Each provider client returns a normalized `prompt_tokens, completion_tokens` pair before `TokenTracker` is touched.
- [ ] Per-run `RunManifest.cost_usd` is the sum of all calls in the run, computed from `pricing.yaml` × token counts.
- [ ] Pre-flight cost estimate: `scripts/cost_estimate.py --pipeline P3 --model claude-sonnet-4-7 --srs D1 --runs 5` produces a USD estimate **before** committing the run. Used in week 7 to budget RQ2.
- [ ] Pricing entries cite their source URL in YAML comments (matching the discipline already in `token_tracker.py:23` for Gemini).

## Implementation steps

1. Audit `token_tracker.py` to confirm the existing Gemini cost flow (already done — `00-context-report.md` Q15–Q17).
2. Add `configs/pricing.yaml` with all 4 model entries (one per `configs/scope.yaml` model).
3. Each WP-01a concrete client extracts token counts in its `_call_provider` method; the base class's `complete()` returns them as part of `StructuredResponse`.
4. Modify `TokenTracker.track_api_call()` to accept the normalized `StructuredResponse` (drop the Gemini-specific dependency).
5. Write `scripts/cost_estimate.py`: dry-run a single call (or use a representative-token cache from previous runs) and project to N runs.
6. Update `RunManifest` to carry `cost_usd` (already in WP-01b spec — close the loop here).
7. Verify: a Gemini run pre- and post-refactor produces identical `cost_usd` (regression test).

## Outputs (file paths)

- `configs/pricing.yaml`
- Refactored `extension/backend/core/token_tracker.py` (provider-agnostic input)
- `scripts/cost_estimate.py`
- Updated `RunManifest` schema (joint with WP-01b)
- `tests/test_token_tracker_multi_provider.py`

## Risks & mitigations

- **Risk:** Cloud provider price changes during the study (RQ2 spans weeks). **Mitigation:** Snapshot `pricing.yaml` at run time into the `RunManifest`; replication package preserves the prices as of run date. Footnote in §4.3 records the snapshot date.
- **Risk:** OSS-local has no USD cost; comparing to API costs is apples-to-oranges. **Mitigation:** Add a sentence to §6 RQ2 noting "compute-only" for the OSS row; the Pareto figure annotates it as a separate marker shape.
- **Risk:** RQ2 cloud bill exceeds budget (R9 in `01-risks.md`). **Mitigation:** `cost_estimate.py` runs *before* the real RQ2 batch; if estimate > $300, swap to mini-models or reduce N to 3 for one provider only (with sensitivity check).
