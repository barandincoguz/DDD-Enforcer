# Replication Package Checklist

## Repository Artifacts

- [x] Benchmark manifest schema
- [x] Benchmark runner
- [x] Scoring script
- [x] Ground-truth schema documentation
- [x] Metrics definition documentation
- [x] Threats-to-validity notes
- [x] Paper alignment report
- [x] Research roadmap

## Still Needed For Submission-Grade Replication

- [ ] Real benchmark manifests for all subject systems
- [ ] Expert-labeled ground truth
- [ ] Live non-Gemini provider adapters, if model-comparison claims are retained
- [ ] Final manuscript figures/tables generated from benchmark outputs
- [ ] Exact provider/version metadata for all reported runs
- [ ] Public links to packaged raw results and annotations

## Recommended Package Layout

- `experiments/`
  - manifests, fixtures, runner, scoring
- `docs/`
  - protocol, schemas, metrics, validity notes
- `paper_alignment_report.md`
  - code-to-paper status map
- `research_roadmap.md`
  - submission priorities

## Suggested Replication Steps

1. Install backend Python dependencies from `extension/backend/requirements.txt`.
2. Install extension dependencies from `extension/package.json` if IDE validation is needed.
3. Prepare manifests and ground-truth JSON files.
4. Run `python3 experiments/run_benchmarks.py <manifest>`.
5. Archive:
  - raw predictions
  - summary outputs
  - CSV metric exports
  - manifest and ground truth used for each run
