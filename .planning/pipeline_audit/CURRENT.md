# Pipeline Audit — CURRENT pointer

**Last update:** 2026-05-21 02:17 GMT+3
**Last action:** WP-CORE-2 shipped at SHA 25e6880. F-5 backlog row moved to SHIPPED. Dev doc + INDEX updated. Pre-WP baseline 272 → post-WP 305 (+33 tests, all green).
**Next:** Loop iteration 2 — pick next OPEN finding from `improvements_backlog.md`. Highest unblocked priorities are F-3 (empty-input contract) and F-1 (PDF defensive handling). Consider close-lookup of next component by priority (`core/architect.py`, priority 2).

**Baseline (sacred):** pytest -m "not integration" → 305 passed, 31 deselected.
**Pre-loop HEAD:** 25e6880
