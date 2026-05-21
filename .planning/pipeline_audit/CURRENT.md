# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-21 09:37 GMT+3
**Last action:** Iteration 2 in flight — WP-CORE-3 (F-3 empty-input contract) RED `91dbeb4` + GREEN `daefeb0` committed. DOC phase landing now. Atomic GREEN per spec v2 W-3 (parser raise + helper + 6 callsite migrations in one commit).
**Next:** Iteration 3 — coordinator should pick **`core/architect.py` close-lookup** (priority 2; 752 LOC; recommended per handoff). Alternative: continue ingestion layer with F-1 (`read_pdf` defensive handling) or F-2 (`read_txt` cp1254 binary garbage). See iteration 2 handoff doc when written.

**Baseline (sacred):** pytest -m "not integration" → 321 passed, 31 deselected.
**Pre-loop HEAD:** 3d13f26 (handoff iteration-1 close)
**Iteration 2 HEAD (post-GREEN):** daefeb0
