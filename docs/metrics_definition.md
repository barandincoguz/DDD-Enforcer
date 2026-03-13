# Metrics Definitions

## Validation Metrics

- `validation_time_ms`
  - End-to-end wall-clock time for a single validation run.
- `stage_latencies_ms.ast_parse`
  - Time spent parsing Python AST.
- `stage_latencies_ms.deterministic_rules`
  - Time spent in rule-based synonym/banned-term/naming checks.
- `stage_latencies_ms.advanced_llm`
  - Time spent in advanced LLM checks for context/value-object/event violations.
- `stage_latencies_ms.naive_llm`
  - Time spent in the naive baseline LLM call.
- `stage_latencies_ms.rag`
  - Time spent attaching traceability sources.

## Generation Metrics

- `Scout`, `Architect`, `Specialist`, `Synthesizer`
  - Per-stage domain-model generation durations in milliseconds.
- `total`
  - Total domain-model generation time across all stages.

## Token And Cost Metrics

- `llm_input_tokens`
  - Billable input tokens after cached-token subtraction.
- `llm_output_tokens`
  - Output tokens reported by the provider.
- `llm_total_tokens`
  - Billable input plus output tokens.
- `cached_tokens`
  - Tokens served from provider-side cache when reported.
- `cost_usd`
  - Estimated cost derived from configured per-model pricing in `extension/backend/config.py`.

## Quality Metrics

- `parseable_outputs`
  - Count of provider calls whose structured output parsed successfully.
- `unparseable_outputs`
  - Count of provider calls whose structured output failed parsing.
- `parseable_output_rate_percent`
  - `parseable_outputs / (parseable_outputs + unparseable_outputs) * 100`.

## Detection Metrics

- `precision`
  - `TP / (TP + FP)`.
- `recall`
  - `TP / (TP + FN)`.
- `f1`
  - Harmonic mean of precision and recall.
- `micro`
  - Metrics computed on aggregated counts across all files.
- `macro`
  - Mean of per-violation-type metrics.

## Retrieval Metrics

- `top1_percent`
  - Percentage of comparable queries whose first retrieved section matches expected sections.
- `top3_percent`
  - Percentage of comparable queries with at least one correct hit in the top three retrieved sections.

## Scaling Metrics

- `file_size_chars`
  - Raw source size in characters.
- `file_loc`
  - Line count of the validated file.
- `scaling_points`
  - Per-file latency records used to analyze size/latency relationships.
