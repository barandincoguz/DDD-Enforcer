# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-22 21:30 GMT+3
**Last action:** Iteration 5 SHIPPED — WP-CORE-6 (F-21 D1 verifier non-vacuous) closed end-to-end. RED `fd7f203`, GREEN `a86bbbb`, DOC `{this commit}`, PLANNING pending. F-21 SHIPPED. NEW: F-22 backlog entry created (Refiner cannot re-run Architect).
**Next:** Iteration 6 — coordinator should pick **F-22 (Refiner stage-aware re-runs)** to complete D1 enforcement story. Refiner currently only re-runs Specialist; Architect-stage failures degrade to best-effort. M-L effort. Alternative: pivot to priority-3 audit walk (synthesizer / verifier deep close-lookup).

**Baseline (sacred):** pytest -m "not integration" → 348 passed, 31 deselected.
**Pre-loop HEAD:** 3d13f26 (pre-iteration-1)
**Iteration 2 HEAD:** d7dc188
**Iteration 3 HEAD:** 2b8602f
**Iteration 4 HEAD:** 9608495
**Iteration 5 GREEN HEAD:** a86bbbb (DOC + planning commits pending)
