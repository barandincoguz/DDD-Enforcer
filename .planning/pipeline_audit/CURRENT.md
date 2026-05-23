# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-23 (GMT+3)
**Last action:** Iteration 8 SHIPPED — WP-CORE-9 (F-2 mislabeled-file detection in read_txt) closed end-to-end. RED `45d9cdf`, GREEN `ff28324`, DOC `{this commit}`, PLANNING pending. F-2 SHIPPED.
**Next:** Iteration 9 — coordinator should continue ingestion-layer or pivot. Candidates:
  - **F-1 (MAJOR-OPEN)** `read_pdf` defensive handling (corrupted/empty/password-protected PDFs). Symmetric to WP-CORE-9 for PDF reader.
  - **F-4 (MAJOR-OPEN-uncertain)** TOC heuristic 120-line + cluster<2.
  - **F-7 (MINOR)** DOCX zero try/except — pair with F-1 for symmetric reader defenses.
  - **F-24 (MINOR-OPEN)** srs_path in VerifierIssue schema (orchestrator pivot).

**Baseline (sacred):** pytest -m "not integration" → 373 passed, 31 deselected.
**Iteration 6 HEAD:** cecfee1
**Iteration 7 HEAD:** 0e43812
**Iteration 8 GREEN HEAD:** ff28324 (DOC + PLANNING in flight)
