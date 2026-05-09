# WP-NEW-C: Prompt Sensitivity Ablation

**Owner:** Ali
**Depends-on:** [WP-01a, WP-01d pipelines, WP-02 corpora]
**Effort:** S (~3-4 days)
**Status:** TODO (NEW WP, audit-driven)
**Addresses:** [LLM Guidelines G4 (prompt documentation), reviewer pre-emption (prompt sensitivity)]

---

## Goal

Her pipeline (P1, P2, P3) için **3 prompt variant** koş, F1 score variance'ı raporla. Reviewer "prompt wording could change results by 10pp" itirazına kanıt-temelli cevap.

---

## What's a "prompt variant"?

3 elder-prompt + 3 prompt variant per pipeline = 12 combinations (3 P × (1 base + 3 variants))... actually simpler: keep base prompt, generate 2 paraphrases.

**Base prompt** (existing in pipeline):
```
You are a DDD violation detector...
```

**Variant 1 — More terse**:
```
Detect DDD violations in this code. Output JSON.
```

**Variant 2 — More verbose**:
```
You are an expert in Domain-Driven Design (DDD). Carefully analyze the code
below for any violations of DDD principles, including but not limited to:
ubiquitous language consistency, bounded context boundaries, ...
```

**Variant 3 — Cross-style**:
```
Task: DDD audit
Inputs: <code>, <SRS>
Output: list of violations with citations
```

For each variant: run on D1 + P3 + G1 + N=5, measure F1 std.

Total runs: 4 prompts × 3 pipelines × N=5 = 60 runs (D1 only, single model G1).

Cost: ~$30 Gemini Pro. 1 day compute.

---

## Acceptance Criteria

- [ ] 3 prompt variants designed (per pipeline P1/P2/P3)
- [ ] Each pipeline accepts a `--prompt-variant` flag (default = base)
- [ ] Run all 4×3×5=60 runs on D1
- [ ] Output: `runs/prompt_sensitivity.json` with F1 mean + std per (pipeline, variant)
- [ ] Paper §9.3 (threats prose) cites variance: "F1 varies by ±X.X pp across prompt variants for P3"
- [ ] If variance > 5pp on any variant: flag for prompt-engineering investigation

---

## Implementation Steps

1. Inventory existing prompts in `core/architect.py` (or wherever P3 prompts live)
2. Author 3 paraphrases per pipeline (12 prompts total)
3. Add `--prompt-variant` parameter to pipeline runner (default = base)
4. Run 60 runs (use multi-run orchestrator from WP-01b)
5. Compute F1 mean + std per cell
6. Generate summary table for paper §9.3

---

## Outputs

- `prompts/p1_base.txt`, `prompts/p1_variant1.txt`, etc. (12 prompt files)
- `runs/prompt_sensitivity.json`
- Paper §9.3 paragraph (drafted in WP-13)

---

## Risks

- If F1 std > 5pp, methodology critique looms — must mitigate with rubric refinement
- Cost: 60 runs × ~$0.50 = ~$30 (acceptable)

---

## Sync Points

- Depends on WP-01d (pipelines as classes)
- Output feeds WP-13 threats prose
