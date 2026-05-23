# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 (GMT+3)
**Last action:** Iteration 9 SHIPPED — WP-CORE-10 (F-1 PDF defensive handling) closed end-to-end. RED `12a984a`, GREEN `5df3df6`, DOC `{this commit}`, PLANNING pending. F-1 SHIPPED.
**Next:** Iteration 10 — coordinator should continue ingestion-layer or pivot:
  - **F-7 (MINOR)** DOCX zero try/except — symmetric pattern to WP-CORE-9/-10 for `read_docx`.
  - **F-4 (MAJOR-OPEN-uncertain)** TOC heuristic refactor.
  - **F-24 (MINOR-OPEN)** srs_path in VerifierIssue schema (orchestrator).

**Baseline (sacred):** pytest -m "not integration" → 388 passed, 31 deselected.
**Iteration 9 GREEN HEAD:** 5df3df6 (DOC + PLANNING in flight)
