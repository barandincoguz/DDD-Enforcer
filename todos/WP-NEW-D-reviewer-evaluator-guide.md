# WP-NEW-D: Reviewer/Evaluator Guide (`EVALUATION.md`)

**Owner:** Ali
**Depends-on:** [WP-12 replication package]
**Effort:** S (~2-3 days)
**Status:** TODO (NEW WP, audit-driven)
**Addresses:** [EMSE Open Science Initiative artifact-evaluation requirement, LLM Guidelines G4]

---

## Goal

Reviewer (özellikle EMSE Open Science board'u) artifact'i **30 dakika içinde** değerlendirebilsin. "Reproduce Table 5" gibi konkret reproduction senaryosu.

---

## Deliverable: `EVALUATION.md`

```markdown
# DDD-Enforcer — Reviewer Evaluation Guide

> Estimated time: 30 minutes for full reproduction of Table 5 (RQ1).

## 1. Setup (5 minutes)
...

## 2. Run Smoke Test — Mock Provider (10 minutes)
For evaluators without API keys, we provide a mock provider:
```bash
DDD_USE_MOCK_PROVIDER=1 pytest -m "not integration"
```
Expected: 105 tests pass, simulated pipeline outputs match committed snapshots.

## 3. Reproduce Table 5 (RQ1) — Real API (15 minutes, requires API keys)
...

## 4. Verify Results
...

## 5. Troubleshooting
...
```

---

## Acceptance Criteria

- [ ] `EVALUATION.md` written with 5 sections: Setup, Smoke (mock), Reproduce Table 5, Verify, Troubleshooting
- [ ] Mock provider implemented (`MOCK_GEMINI_API_KEY` and offline fixture responses)
- [ ] `DDD_USE_MOCK_PROVIDER=1` env var enables it
- [ ] One golden-output snapshot for Table 5 (D1 + P3 + G1 + 1 run, deterministic seed)
- [ ] Reviewer can copy-paste 5-10 commands and complete evaluation
- [ ] Tested by external (non-author) reviewer at least once before submission

---

## Implementation Steps

1. Author `EVALUATION.md` skeleton
2. Build mock provider in `core/llm/mock.py` — returns fixture responses for known prompts
3. Capture golden output for Table 5 reproduction (D1+P3+G1+seed=42, single run)
4. Write reproduction script `scripts/reproduce_table_5.py`
5. Run end-to-end test as if you're a reviewer (clean clone, follow EVALUATION.md)
6. Iterate based on rough edges

---

## Outputs

- `EVALUATION.md` (root)
- `core/llm/mock.py`
- `scripts/reproduce_table_5.py`
- `runs/golden/table_5_d1_p3_g1.json` (snapshot for verification)

---

## Sync Points

- Depends on WP-12 (replication package finalized)
- Output critical for EMSE Open Science badge (artifact passes evaluation)
