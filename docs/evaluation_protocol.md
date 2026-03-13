# Evaluation Protocol

## Goal

Provide a repeatable process for evaluating DDD-Enforcer across benchmark manifests in either `pipeline` or `naive` mode.

## Inputs

- Benchmark manifest JSON following `experiments/benchmark_manifest.schema.json`
- One or more SRS paths
- Source files or source globs
- Ground truth JSON
- Provider/model selection
- Repeat count and optional seed

## Recommended Procedure

1. Prepare benchmark manifests per project/domain.
2. Freeze provider/model settings for the entire run.
3. Run the benchmark:

```bash
python3 experiments/run_benchmarks.py experiments/sample_benchmark.json
```

4. Inspect generated outputs:
  - `raw_predictions.json`
  - `raw_predictions.csv`
  - `summary.json`
  - research metric CSV exports
5. If needed, rescore existing raw outputs:

```bash
python3 experiments/score_results.py \
  experiments/results/sample-smoke/raw_predictions.json \
  experiments/fixtures/ecommerce_ground_truth.json
```

## Experimental Controls

- Use the same manifest except for the factor under study.
- Compare `pipeline` and `naive` modes with the same provider/model whenever testing the value of domain-model extraction and AST guidance.
- Use repeated runs whenever the provider is non-deterministic or rate-limited.
- Record the provider, model, repeat index, and seed in manifest metadata.

## Outputs To Report

- Micro precision, recall, F1
- Macro precision, recall, F1
- Per-violation-type metrics
- Average/min/max latency
- Cost per validation and total cost
- Parseable output rate
- Retrieval Top-1 / Top-3, if ground truth includes expected source sections
- File-size-to-latency scaling points

## Current Limits

- The sample benchmark is a smoke test, not a journal dataset.
- The repository currently ships a live Gemini adapter and a static offline provider only.
- Cross-project and multi-model claims require new manifests, adapters, and labeled data.
