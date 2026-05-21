# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-21 11:50 GMT+3
**Last action:** Iteration 4 closed — WP-CORE-5 ABANDONED (F-11 dormant; Codex review found 3 CRITICALs); pivoted to WP-CORE-5b (F-14) and shipped clean. RED `cc82e64`, GREEN `27a5d98`, DOC `{this commit}`, PLANNING pending. F-14 SHIPPED.
**Next:** Iteration 5 — coordinator should pick **F-21 (vacuous D1 verifier pass)** per Codex W-8 priority bump (affects every project run methodologically; D1 has been passing vacuously for every run in project history because `ContextHypothesis.supporting_sentence_ids` is never populated by Architect). Architect prompt + parsing change + cross-stage data flow; M effort.

**Baseline (sacred):** pytest -m "not integration" → 338 passed, 31 deselected.
**Pre-loop HEAD:** 3d13f26 (pre-iteration-1)
**Iteration 2 HEAD:** d7dc188
**Iteration 3 HEAD:** 2b8602f
**Iteration 4 final HEAD:** 27a5d98 (after GREEN; DOC + planning commits pending)
