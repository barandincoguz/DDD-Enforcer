# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 (GMT+3)
**Last action:** Iteration 7 SHIPPED — WP-CORE-8 (F-23 typed PipelineError handler in main.py) closed end-to-end. RED `72898af`, GREEN `a2bca34`, DOC `{this commit}`, PLANNING pending. F-23 SHIPPED.
**Next:** Iteration 8 — coordinator should pivot to ingestion-layer (5 consecutive orchestrator-layer iterations is enough). Candidates:
  - **F-1 (MAJOR-OPEN)** `read_pdf` defensive handling (corrupted/empty/password-protected PDFs)
  - **F-2 (MAJOR-OPEN)** `read_txt` cp1254 silent binary garbage (encoding detection or hard-fail)
  - **F-4 (MAJOR-OPEN-uncertain)** TOC heuristic 120-line + cluster<2
  - **F-24 (MINOR-OPEN)** srs_path in VerifierIssue schema (orchestrator layer)
  - Alternative: synthesizer / verifier deeper close-lookup (priority-3 audit walk)

**Baseline (sacred):** pytest -m "not integration" → 365 passed, 31 deselected.
**Pre-loop HEAD:** 3d13f26 (pre-iteration-1)
**Iteration 2 HEAD:** d7dc188
**Iteration 3 HEAD:** 2b8602f
**Iteration 4 HEAD:** 9608495
**Iteration 5 HEAD:** 4c8580c
**Iteration 6 HEAD:** cecfee1
**Iteration 7 GREEN HEAD:** a2bca34 (DOC + PLANNING commits in flight)
