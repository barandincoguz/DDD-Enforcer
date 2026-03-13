# Research Roadmap

## Must-Do Before Journal Submission

- Build a real benchmark suite with multiple subject systems and checked-in manifests.
- Produce expert-labeled ground truth for violation detection and, where feasible, retrieval targets.
- Add at least one live non-Gemini provider adapter before claiming model-comparison results.
- Run repeated `pipeline` and `naive` experiments and populate:
  - overall precision/recall/F1
  - per-violation-type metrics
  - cross-project aggregates
  - latency and cost tables
  - retrieval Top-1 / Top-3 summaries
- Update manuscript figures/tables with actual outputs from the new experiment pipeline.
- Align the paper text to the implemented Python-only AST scope and save-triggered IDE workflow.

## High-Value Next

- Add OpenAI-compatible and local/open-source provider adapters behind the new provider interface.
- Extend the experiment runner with manifest validation against the JSON schema.
- Add richer ground-truth tooling for annotator agreement and confusion-matrix reports.
- Refresh `TOKEN_TRACKING.md` and `VALIDATION_METRICS_README.md` so they reflect the new JSONL/CSV event exports instead of the old presentation-only framing.
- Add CLI helpers for batch export of aggregated tables directly in manuscript-friendly CSV shape.

## Nice-to-Have

- Add multi-language AST front ends (Java, TypeScript, C#, Go) behind a parser interface.
- Version benchmark manifests and gold labels with explicit dataset metadata and changelogs.
- Add notebook or plotting scripts for Pareto-frontier and scaling figures.
- Package the backend as an installable Python module to decouple it from the extension layout.
- Add automated regression benchmarks in CI once provider credentials or stable mock baselines are available.
