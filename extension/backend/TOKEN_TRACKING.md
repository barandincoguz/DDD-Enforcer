# Token Usage Tracking & Cost Estimation

## Overview

Tracks per-call token usage and computes USD cost using the registry-driven
pricing in [`configs/models.py`](configs/models.py). Supports flat and
context-tiered pricing.

## Features

- ✅ **Automatic Tracking**: Every Gemini API call is automatically tracked
- ✅ **Stage Breakdown**: Token usage separated by pipeline stage (Scout, Architect, Specialist, Synthesizer, Validator)
- ✅ **Cost Estimation**: Real-time cost calculation derived from the registry
- ✅ **Detailed Logs**: Per-call timestamp, operation name, token counts, model_id, provider
- ✅ **JSON Export**: Full reports exportable for analysis

## Model Selection

Model selection and pricing live in [`configs/models.py`](configs/models.py).

To upgrade a model: edit the relevant entry in `STAGE_GROUPS`. No other file
should require changes. Pricing is read from the same module via `MODELS`,
which supports tiered context-based pricing for models like
`gemini-3.1-pro-preview` (different rates above and below 200k input tokens).

### Stage → group mapping

- **`domain_extraction`**: Scout, Architect, Specialist, Synthesizer
- **`validation`**: Validator

### Defaults (verify by reading `configs/models.py`)

- Domain extraction: `gemini-3.1-pro-preview`
- Validation: `gemini-3-flash-preview`

All Gemini 3 models are currently in **preview**; pricing and availability are
subject to change provider-side. The registry's `MODELS` dict carries a snapshot
date in its docstring (currently 2026-04-27).

After the model-registry consolidation refactor, the tracking implementation
spans three files for separation of concerns:

- `core/token_tracker.py` — stateful singleton recording API call usage
- `core/token_tracker_report.py` — pure functions that build reports / format console output / serialize JSON
- `core/token_tracker_types.py` — shared dataclasses (`TokenUsageStats`, `ModelTokenAccumulator`, `StageTokenAccumulator`, `APICallRecord`)

### Token Types

1. **Prompt Tokens**: Input code, domain rules, and prompts sent to the model (cached count is subtracted before billing — see "Billing"). 
2. **Completion Tokens**: Generated output — violations, analysis, suggestions (includes any reasoning).
3. **Cached Tokens**: Previously sent context reused from cache. **Subtracted from billable input** by `TokenTracker.track_api_call` (`token_tracker.py:62-63`); not billed separately by this codebase.
4. **Total Tokens**: Billable prompt + completion.

### Billing

Cost is computed at call time by `Pricing.cost_for(prompt_tokens, completion_tokens)` for the model bound to the stage. Algorithm (`configs/models.py:42-53`):

1. Walk the model's `tiers` in declaration order.
2. First tier whose `max_prompt_tokens` is `None` or `>= prompt_tokens` wins.
3. Cost = `prompt_tokens × tier.input_per_1m_usd + completion_tokens × tier.output_per_1m_usd`, divided by 1M.

For tiered models (e.g., `gemini-3.1-pro-preview` with a 200k breakpoint), the prompt-token count selects the tier; the completion-token component is priced at the same tier.

### Important Notes

- ⚠️ **Reasoning/thinking tokens are included in completion tokens** (output price covers thinking)
- 💾 **Context caching**: cached prompt tokens are excluded from billable input — see `token_tracker.py:62-63`
- 🔄 **Implicit caching** is enabled by default for prompts > 1024 tokens (Gemini side)
- 📊 **All tracking auto-derives from the registry** — pricing changes propagate by editing `configs/models.py` only

## API Endpoints

### Get Detailed Statistics

```bash
GET http://localhost:8000/tokens/stats
```

Returns:

- Total tokens (prompt + completion)
- Per-stage breakdown with costs
- Per-model breakdown (`by_model`) with provider, costs, token counts
- Detailed call history with timestamps + model_id

### Get Summary

```bash
GET http://localhost:8000/tokens/summary
```

Returns concise summary without detailed call history.

### Export Report

```bash
GET http://localhost:8000/tokens/export
```

Exports full report to `token_usage_export.json`.

### Reset Tracker (Testing)

```bash
POST http://localhost:8000/tokens/reset
```

Resets all counters to zero.

## Console Output

After domain model generation, `print_summary` prints to stdout:

```
======================================================================
📊 TOKEN USAGE & COST REPORT
======================================================================
  Total API Calls:        5
  Total Tokens:           12,450
    ↳ Input:              8,230
    ↳ Output:             4,220

----------------------------------------------------------------------
🤖 MODEL BREAKDOWN
----------------------------------------------------------------------

  gemini-3.1-pro-preview (provider: gemini):
    Calls:  4
    Input:  6,500 tokens
    Output: 3,800 tokens
    Cost:   $0.058600

  gemini-3-flash-preview (provider: gemini):
    Calls:  1
    Input:  1,730 tokens
    Output: 420 tokens
    Cost:   $0.002125

----------------------------------------------------------------------
💰 TOTAL COST ESTIMATION
----------------------------------------------------------------------
  Input Cost:  $0.013865
  Output Cost: $0.046860
  Total Cost:  $0.060725 USD

----------------------------------------------------------------------
📈 STAGE BREAKDOWN
----------------------------------------------------------------------

  Scout (gemini-3.1-pro-preview):
    Calls:  2
    Tokens: 5,120
    Cost:   $0.030240
…
======================================================================
```

(Numbers are illustrative; actual values depend on the prompt sizes seen in your run.)

## JSON Export Format

```json
{
  "session_start": "2026-04-27T10:30:45.123456",
  "session_end": "2026-04-27T10:32:12.789012",
  "summary": {
    "total_api_calls": 5,
    "total_prompt_tokens": 8230,
    "total_completion_tokens": 4220,
    "total_tokens": 12450
  },
  "model_usage": {
    "gemini-3.1-pro-preview": {
      "prompt_tokens": 6500,
      "completion_tokens": 3800,
      "total_tokens": 10300,
      "stages": ["Architect", "Scout", "Specialist", "Synthesizer"],
      "provider": "gemini",
      "call_count": 4
    },
    "gemini-3-flash-preview": {
      "prompt_tokens": 1730,
      "completion_tokens": 420,
      "total_tokens": 2150,
      "stages": ["Validator"],
      "provider": "gemini",
      "call_count": 1
    }
  },
  "cost_estimation": {
    "by_model": {
      "gemini-3.1-pro-preview": {
        "input_cost": 0.013000,
        "output_cost": 0.045600,
        "total_cost": 0.058600,
        "input_tokens": 6500,
        "output_tokens": 3800
      },
      "gemini-3-flash-preview": {
        "input_cost": 0.000865,
        "output_cost": 0.001260,
        "total_cost": 0.002125,
        "input_tokens": 1730,
        "output_tokens": 420
      }
    },
    "total_input_cost": 0.013865,
    "total_output_cost": 0.046860,
    "total_cost": 0.060725,
    "currency": "USD"
  },
  "stage_breakdown": {
    "Scout": {
      "model_id": "gemini-3.1-pro-preview",
      "call_count": 2,
      "prompt_tokens": 3500,
      "completion_tokens": 1620,
      "total_tokens": 5120,
      "estimated_cost": 0.030240
    }
    /* … one entry per pipeline stage … */
  },
  "call_history": [
    {
      "timestamp": "2026-04-27T10:30:46.234567",
      "stage": "Scout",
      "operation": "extract_sentences_chunk_1",
      "model_id": "gemini-3.1-pro-preview",
      "provider": "gemini",
      "prompt_tokens": 1800,
      "completion_tokens": 850,
      "total_tokens": 2650,
      "estimated_cost": 0.013800
    }
    /* … one entry per call … */
  ]
}
```

The `by_model` keys are full model_id strings (no `gemini-2.5-` legacy keys, no `flash_model` / `flash_lite_model` legacy shape).

## Usage in Code

### Automatic Tracking (Already Integrated)

All API calls in `architect.py` and `llm_client.py` automatically track tokens:

```python
response = self.client.models.generate_content(...)

# Automatically tracked:
self.token_tracker.track_api_call(
    response,
    stage="Scout",
    operation="extract_sentences",
)
```

### Manual Tracking (If Needed)

```python
from core.token_tracker import TokenTracker

tracker = TokenTracker.get_instance()

# Track a call
tracker.track_api_call(response, stage="Custom", operation="custom_op")

# Query a stage's running totals
validator_accum = tracker.tokens_for_stage("Validator")
print(validator_accum.cost_usd, validator_accum.call_count)

# Query a model's running totals
model_accum = tracker.tokens_for_model("gemini-3-flash-preview")

# Get a full report (data structures: see "JSON Export Format" above)
report = tracker.get_report(detailed=True)

# Print summary
tracker.print_summary()

# Export to file
tracker.export_to_json("report.json")
```

## Files Generated

1. **`token_usage_report.json`** — Generated after domain model creation (startup).
2. **`token_usage_export.json`** — Generated via `/tokens/export` endpoint.

## Cost Estimation (Order-of-Magnitude)

These are rough envelopes; real costs depend on the model bound to each stage in the registry, your prompt sizes, and Gemini's preview-tier pricing changes. Compute exact numbers from the live `cost_estimation` block in any token report.

For the current defaults (`gemini-3.1-pro-preview` for domain extraction, `gemini-3-flash-preview` for validation):

| Workload | Domain extraction (4 stages × Pro) | Per validation (Flash) |
|---|---|---|
| Small SRS (~15k input, ~5k output) | ~$0.090 USD | ~$0.0008 USD |
| Medium SRS (~60k input, ~20k output) | ~$0.36 USD | n/a |
| 100 file validations |  | ~$0.080 USD |

(Domain-extraction cost uses the cheap tier; if a stage's prompt exceeds the 200k breakpoint, the Pro model's expensive tier kicks in automatically — `Pricing.cost_for` handles this.)

## Updating Pricing

When provider rates change, edit the relevant `MODELS` entry in `configs/models.py`. The registry's `_validate_registry` runs at import time so a typo in `model_id` fails loudly. The `tests/test_registry_snapshot.py` drift guard fails when the default model_ids change, forcing the change to be acknowledged in the same commit.

## Source

- Pricing: <https://ai.google.dev/gemini-api/docs/pricing>
- Snapshot date: see the docstring at the top of `configs/models.py`.
- All Gemini 3 entries are currently **preview**; expect provider-side pricing/availability changes.
