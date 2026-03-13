# Paper Alignment Report

## Repository Snapshot

- Actual runtime shape: a VS Code extension under `extension/` with an embedded FastAPI backend under `extension/backend/`.
- Validation is Python-only and file-save triggered. The extension skips non-semantic saves, but validation is still full-file, not incremental.
- Domain-model generation uses a four-stage pipeline (`Scout`, `Architect`, `Specialist`, `Synthesizer`) plus optional AST enrichment from workspace Python files.
- Validation now supports two executable modes:
  - `pipeline`: AST + domain-model grounded validation + optional traceability retrieval
  - `naive`: raw-source prompt without AST, domain model, or RAG grounding
- Metrics are now persisted in analysis-friendly JSONL event streams and CSV exports through the research metrics store and experiment runner.

Status labels used below:

- `implemented`
- `partially implemented`
- `not implemented`
- `implemented but not measurable`
- `measurable but not automated`

## Confirmed Claims

| Manuscript reference | Claim | Code evidence | Status |
| --- | --- | --- | --- |
| Section 3, “four specialized LLM agents” | Multi-agent domain-model generation pipeline with Scout/Architect/Specialist/Synthesizer | `extension/backend/core/architect.py`, streaming progress in `extension/src/extension.ts` | `implemented` |
| Section 3.2, “AST Parsing” | AST-guided validation for classes, functions, imports, assignments, and calls | `extension/backend/core/parser.py`, orchestration in `extension/backend/core/validation_service.py` | `implemented` |
| Section 3.3, “traceability pipeline serves a different purpose” | Retrieval is separated from violation reasoning and attaches source references after detection | `extension/backend/core/rag_pipeline.py`, `extension/backend/core/validation_service.py` | `implemented` |
| Abstract / Section 3, “real-time semantic violation detection” | IDE workflow triggers backend validation on save and displays diagnostics | `extension/src/extension.ts`, `/validate` in `extension/backend/main.py` | `implemented` |
| RQ5 metrics text, “validation latency” and “cost per validation” | Per-validation latency, token usage, cached tokens, parseable output counts, and cost are recorded | `extension/backend/core/token_tracker.py`, `extension/backend/core/validation_metrics.py`, `extension/backend/core/research_metrics.py` | `implemented` |
| Section 3, “domain model generation from SRS documents” | SRS/TXT/DOCX/PDF parsing plus domain-model generation and workspace AST enrichment | `extension/backend/core/document_parser.py`, `extension/backend/core/model_generation_service.py`, `extension/backend/core/ast_model_signals.py` | `implemented` |
| RQ3 text, “naive LLM baseline” | Executable naive baseline mode now exists in shared validation flow and experiment runner | `extension/backend/core/llm_client.py`, `extension/backend/core/validation_service.py`, `experiments/run_benchmarks.py` | `implemented` |

## Partially Supported Claims

| Manuscript reference | Claim | Code evidence | Status | Notes |
| --- | --- | --- | --- | --- |
| Abstract, “[PLACEHOLDER: M] LLM models (including Google Gemini, ChatGPT and open-source alternatives)” | Multi-model/provider comparison | Provider abstraction in `extension/backend/core/llm_provider.py` | `partially implemented` | Only Gemini is a live provider today. `static-json` exists for offline smoke tests, not as a research model. |
| Section 3.2 / RQ4, “six violation categories” | Violation taxonomy | `extension/backend/core/llm_client.py` | `partially implemented` | Six research-facing categories exist, plus internal `SystemError`. Ground-truth tooling now supports per-type scoring, but no real annotated benchmark is in the repo. |
| RQ5, “latency scaling across codebase sizes” | File-size-to-latency scaling | Validation runs now record file size, LOC, and stage latencies | `measurable but not automated` | Automation exists in exports, but no real cross-project dataset has been executed yet. |
| Section 3.3 / RQ5, “Top-1 accuracy, Top-3 accuracy” | Retrieval accuracy | Retrieval events now support expected-section comparison hooks | `measurable but not automated` | Requires ground truth with expected requirement sections. |
| Threats section, “Language limitation” | Python-only AST parser limitation | `extension/backend/core/parser.py` | `implemented` | The paper must explicitly keep this limitation; multi-language support is not present. |
| Section 3 workflow placeholder, “[PLACEHOLDER: Explain the overall workflow of latest version of the extension]” | Updated extension workflow | Save-filtering, streaming generation progress, quick-fix source opening | `implemented` | The manuscript still needs prose/figure updates. |

## Unsupported Or Missing Claims

| Manuscript reference | Claim | Code evidence | Status | Gap |
| --- | --- | --- | --- | --- |
| Abstract / Contributions, “[PLACEHOLDER: N] open-source microservice systems” | Cross-project evaluation | `experiments/` now provides runner infrastructure only | `not implemented` | No real benchmark corpus is packaged in the repo. |
| RQ2, “Google Gemini, ChatGPT and open-source alternatives” | Live provider comparison | Only Gemini adapter exists in production code | `not implemented` | OpenAI/open-source adapters still need to be implemented and configured. |
| Section 4, “expert-validated ground truth annotations” | Expert-labeled benchmark data | `docs/ground_truth_schema.md`, sample fixture only | `not implemented` | No real human-annotated dataset is present. |
| RQ1 placeholder, “complete and correct relative to expert-created domain models” | Domain-model extraction evaluation against gold models | Experiment framework supports manifests and outputs only | `not implemented` | Gold domain models are absent. |
| Conclusion / Section 10 placeholders, “report results averaged over [PLACEHOLDER: N] independent runs” | Statistical summary over repeated real experiments | Repeat-count support added to runner | `partially implemented` | Infrastructure exists; actual repeated experiment data does not. |
| Section 3 diagram placeholder, “[PLACEHOLDER: Insert system architecture diagram]” | Paper figure | Repo contains no journal figure source | `not implemented` | Needs manuscript/figure work outside the codebase. |

## Placeholder Inventory

### Paper placeholders directly blocked by missing data

- Abstract / Contributions:
  - `[PLACEHOLDER: N] open-source microservice projects`
  - `[PLACEHOLDER: M] LLM models`
  - `[PLACEHOLDER: summarize key findings ...]`
- Section 4 evaluation tables:
  - subject systems, domains, LOC/service counts
  - model comparison table rows
  - pipeline vs naive results
  - latency/cost/scaling tables
- Threats / Conclusion:
  - actual run counts, variance, statistical power
  - concrete key findings
  - replication package links

### Paper placeholders now supported by code changes but still require execution

- Validation latency per model and per file-size bucket
- Domain-model generation time
- Per-stage latency breakdown
- Cost per validation and per generation run
- Parseable output rate
- Pipeline vs naive baseline outputs
- Per-violation-type precision/recall/F1
- Retrieval Top-1 / Top-3 summaries

## Required Experiments To Fill Remaining Placeholders

1. Assemble a real benchmark manifest set for multiple subject systems.
2. Create expert-labeled ground truth for violation annotations and, where possible, expected source sections.
3. Run repeated `pipeline` and `naive` experiments across the benchmark set.
4. Add at least one live non-Gemini adapter before claiming provider comparison.
5. Score per-project and aggregate results with the supplied scoring pipeline.
6. Capture repeated-run latency/cost distributions for the final tables and threats section.

## Required Code Changes Already Landed

- Shared validation service used by both backend runtime and benchmark runner.
- Provider abstraction isolating Gemini-specific request handling.
- Structured JSONL/CSV research metrics export.
- Executable naive baseline.
- Config-driven experiment manifest and runner.
- Per-stage and per-run latency/cost/token instrumentation.
- Contract-aligned local API tests and offline benchmark smoke run.

## Risks To Validity Caused By Remaining Gaps

- Construct validity:
  - The six-category taxonomy exists in code, but only real annotated data can validate whether it matches the paper’s intended phenomenon boundaries.
- External validity:
  - The repository still lacks a representative multi-project benchmark; current sample fixtures are only smoke-test artifacts.
- Internal validity:
  - Only Gemini is wired as a live provider, so any model-comparison claims would currently overstate the implementation.
- Measurement validity:
  - Retrieval accuracy hooks and false-positive/false-negative summaries are now measurable, but they depend entirely on benchmark labels not yet collected.
- Operational validity:
  - Extension integration remains Python-only and save-triggered; “real-time” should not be overstated beyond that workflow.

## File Relevance Map

### Core research logic

- `extension/backend/core/architect.py`
- `extension/backend/core/llm_client.py`
- `extension/backend/core/parser.py`
- `extension/backend/core/validation_service.py`
- `extension/backend/core/model_generation_service.py`
- `extension/backend/core/rag_pipeline.py`
- `extension/backend/core/ast_model_signals.py`

### Experiment-critical

- `experiments/run_benchmarks.py`
- `experiments/scoring.py`
- `experiments/benchmark_manifest.schema.json`
- `experiments/sample_benchmark.json`
- `experiments/fixtures/ecommerce_ground_truth.json`

### Metrics-critical

- `extension/backend/core/token_tracker.py`
- `extension/backend/core/validation_metrics.py`
- `extension/backend/core/research_metrics.py`
- `extension/backend/main.py`

### UI-only

- `extension/src/extension.ts`
- `extension/src/test/extension.test.ts`
- `extension/package.json`

### Legacy / likely removable or quarantine candidates

- `extension/backend/core/prompts.py.backup`
- `extension/backend/core/intermediate/` generated stage dumps
- `extension/backend/VALIDATION_METRICS_README.md` and `extension/backend/TOKEN_TRACKING.md` if not refreshed to the new architecture
