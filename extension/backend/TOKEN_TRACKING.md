# Token Usage Tracking & Cost Estimation

## Overview

This system automatically tracks all LLM API calls and calculates cost estimations for the UBMK presentation and research paper.

## Features

- ✅ **Automatic Tracking**: Every Gemini API call is automatically tracked
- ✅ **Stage Breakdown**: Token usage separated by pipeline stage (Scout, Architect, Specialist, Synthesizer, Validator)
- ✅ **Cost Estimation**: Real-time cost calculation based on multi-model pricing
- ✅ **Detailed Logs**: Per-call timestamp, operation name, and token counts
- ✅ **JSON Export**: Full reports exportable for analysis

## Model Configuration

We use two different models for different purposes:

| Model                 | Use Case                | Stages                                    |
| --------------------- | ----------------------- | ----------------------------------------- |
| gemini-2.5-flash      | Domain Model Generation | Scout, Architect, Specialist, Synthesizer |
| gemini-2.5-flash-lite | Code Validation         | Validator                                 |

## Pricing (PAID Tier - January 2026)

**Official Source**: https://ai.google.dev/gemini-api/docs/pricing

### Gemini 2.5 Flash (Domain Model Generation)

| Type                | Price per 1M tokens | Price per token | Notes                    |
| ------------------- | ------------------- | --------------- | ------------------------ |
| Input (Prompt)      | $0.30               | $0.000000300    | Text/Image/Video input   |
| Output (Completion) | $2.50               | $0.000002500    | Includes thinking tokens |
| Context Caching     | $0.03               | $0.000000030    | Cached input reuse       |

### Gemini 2.5 Flash-Lite (Validation)

| Type                | Price per 1M tokens | Price per token | Notes                    |
| ------------------- | ------------------- | --------------- | ------------------------ |
| Input (Prompt)      | $0.10               | $0.000000100    | Text/Image/Video input   |
| Output (Completion) | $0.40               | $0.000000400    | Includes thinking tokens |
| Context Caching     | $0.01               | $0.000000010    | Cached input reuse       |

### Token Types

1. **Prompt Tokens**: Input code, domain rules, and prompts sent to the model (INCLUDES cached count)
2. **Completion Tokens**: Generated output - violations, analysis, suggestions (includes any reasoning)
3. **Cached Tokens**: Previously sent context reused from cache (billed at cache rate, NOT free!)
4. **Total Tokens**: Prompt + Completion

### Billing Formula

```python
# For Flash model (Domain Model Generation)
flash_input_cost = prompt_tokens × $0.30 / 1M
flash_output_cost = completion_tokens × $2.50 / 1M

# For Flash-Lite model (Validation)
lite_input_cost = prompt_tokens × $0.10 / 1M
lite_output_cost = completion_tokens × $0.40 / 1M

total_cost = flash_total + lite_total
```

### Important Notes

- ⚠️ **All reasoning/thinking is included in completion tokens** (output price covers thinking)
- 💾 **Context caching is NOT free** - It's cheaper but still billed
- 🔄 **Implicit caching** is enabled by default for prompts > 1024 tokens
- 📊 **Cached tokens are INCLUDED in prompt_token_count** - Must subtract for billing accuracy

## API Endpoints

### Get Detailed Statistics

```bash
GET http://localhost:8000/tokens/stats
```

Returns:

- Total tokens (prompt + completion)
- Per-stage breakdown with costs
- Detailed call history with timestamps

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

After domain model generation, the system automatically prints:

```
======================================================================
📊 TOKEN USAGE & COST REPORT
======================================================================
  Total API Calls: 5
  Total Tokens: 12,450
    ↳ Input:  8,230 tokens
    ↳ Output: 4,220 tokens

----------------------------------------------------------------------
💰 COST ESTIMATION (Gemini 2.5 Flash)
----------------------------------------------------------------------
  Input Cost:  $0.000617
  Output Cost: $0.001266
  Total Cost:  $0.001883 USD

----------------------------------------------------------------------
📈 STAGE BREAKDOWN
----------------------------------------------------------------------

  Scout:
    Calls: 2
    Tokens: 5,120
    Cost: $0.000768

  Architect:
    Calls: 1
    Tokens: 2,340
    Cost: $0.000351

  Specialist:
    Calls: 1
    Tokens: 3,150
    Cost: $0.000473

  Synthesizer:
    Calls: 1
    Tokens: 1,840
    Cost: $0.000276
======================================================================
```

## JSON Export Format

```json
{
  "session_start": "2025-12-15T10:30:45.123456",
  "session_end": "2025-12-15T10:32:12.789012",
  "summary": {
    "total_api_calls": 5,
    "total_prompt_tokens": 8230,
    "total_completion_tokens": 4220,
    "total_tokens": 12450
  },
  "cost_estimation": {
    "input_cost": 0.000617,
    "output_cost": 0.001266,
    "total_cost": 0.001883,
    "currency": "USD"
  },
  "stage_breakdown": {
    "Scout": {
      "call_count": 2,
      "prompt_tokens": 3500,
      "completion_tokens": 1620,
      "total_tokens": 5120,
      "estimated_cost": 0.000768
    },
    "Architect": { ... },
    "Specialist": { ... },
    "Synthesizer": { ... }
  },
  "call_history": [
    {
      "timestamp": "2025-12-15T10:30:46.234567",
      "stage": "Scout",
      "operation": "extract_sentences_chunk_1",
      "prompt_tokens": 1800,
      "completion_tokens": 850,
      "total_tokens": 2650
    },
    ...
  ]
}
```

## Usage in Code

### Automatic Tracking (Already Integrated)

All API calls in `architect.py` and `llm_client.py` automatically track tokens:

```python
response = self.client.models.generate_content(...)

# Automatically tracked:
self.token_tracker.track_api_call(
    response,
    stage="Scout",
    operation="extract_sentences"
)
```

### Manual Tracking (If Needed)

```python
from core.token_tracker import TokenTracker

tracker = TokenTracker.get_instance()

# Track a call
tracker.track_api_call(response, stage="Custom", operation="custom_op")

# Get report
report = tracker.get_report(detailed=True)

# Print summary
tracker.print_summary()

# Export to file
tracker.export_to_json("report.json")
```

## Files Generated

1. **`token_usage_report.json`** - Generated after domain model creation (startup)
2. **`token_usage_export.json`** - Generated via `/tokens/export` endpoint

## For UBMK Presentation

### Key Metrics to Include:

1. **Average tokens per validation request**
2. **Total cost for domain model generation**
3. **Cost per violation detection** (per file validation)
4. **Stage-wise cost breakdown** (which stage is most expensive)

### Test Scenarios:

```bash
# Scenario 1: Generate domain model
python main.py
# Check: token_usage_report.json

# Scenario 2: Multiple validations
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{"code": "...", "filename": "test.py"}'

# Check cumulative stats:
curl http://localhost:8000/tokens/stats

# Scenario 3: Export for analysis
curl http://localhost:8000/tokens/export
```

## Cost Estimation Examples

### Small Project (1 SRS file, ~50 KB)

- Input: ~15,000 tokens
- Output: ~5,000 tokens
- **NEW Estimated Cost: ~$0.017 USD** (Input: $0.0045 + Output: $0.0125)
- _Old price was: $0.002-0.003 USD (5.7x-8.5x cheaper)_

### Medium Project (Multiple SRS files, ~200 KB)

- Input: ~60,000 tokens
- Output: ~20,000 tokens
- **NEW Estimated Cost: ~$0.068 USD** (Input: $0.018 + Output: $0.050)
- _Old price was: $0.010-0.015 USD (4.5x-6.8x cheaper)_

### Per Validation Request

- Input: ~500-1000 tokens (code + domain rules)
- Output: ~200-500 tokens (violations)
- **NEW Estimated Cost: ~$0.0008-0.0015 USD** per validation
- _Old price was: $0.0001-0.0002 USD (4x-7.5x cheaper)_

### Academic Paper Example (UBMK)

**Scenario**: 10 SRS files, 100 code files validated

- Domain model generation: ~150,000 input + 50,000 output tokens = $0.170
- Code validations (100 files): ~50,000 input + 25,000 output tokens = $0.078
- **Total Project Cost: ~$0.25 USD** (approximately ₺8.75 TRY at 35 TRY/USD)
- _Old pricing would have been: $0.037 USD (6.8x difference)_

## Important Price Update Notice (Dec 12, 2025)

⚠️ **CRITICAL**: Gemini 2.5 Flash pricing has been updated to official PAID tier rates:

- Input: $0.30/1M (was incorrectly $0.075/1M - **4x increase**)
- Output: $2.50/1M (was incorrectly $0.30/1M - **8.3x increase**)
- Context Caching: $0.03/1M (was incorrectly listed as FREE)

The FREE tier exists but has strict rate limits. Production applications should use PAID tier pricing.

## Notes

- Token counts include **both** input (prompt) and output (completion)
- Prices are based on **Gemini 2.5 Flash PAID tier** (December 12, 2025)
- Source: https://ai.google.dev/gemini-api/docs/pricing
- For large projects, consider **Gemini 2.5 Flash Lite** ($0.10/$0.40 per 1M - 3x cheaper)
- All tracking is automatic and doesn't affect performance
- **Cached tokens cost $0.03/1M (10x cheaper than fresh input)**
