# WP-07: Judge LLM + Rubric Pipeline

**Owner:** Ali
**Depends-on:** [WP-01a, WP-00]
**Effort:** M
**Status:** TODO
**Addresses instructor feedback:** [Hoca-6] (Cohen's κ requires Judge → audit chain functioning)

## Goal

Build the Judge harness that classifies every framework-reported violation as TP / FP / FN against the SRS or rulebook, with a citation requirement. The Judge LLM is **strictly stronger** than every model in `configs/scope.yaml` and **belongs to a different family** than the tested model where possible (cross-family logic). The Judge is parametrized so that adding a "second Judge" for sensitivity check costs ≤1 day (R10 mitigation).

## Acceptance criteria

- [ ] `prompts/judge_rubric.md` — full rubric prompt with: V1–V6 formal definitions (copied from §3.2 of paper), required citation format, TP/FP/FN classification rules, low-confidence flagging.
- [ ] `evaluation/judge.py` — accepts `(framework_violation_list, srs_path, source_file, model_id_under_test)` and returns `[(violation, classification, confidence, citation, reason)]`.
- [ ] Cross-family selection logic: given `model_id_under_test`, pick a Judge from a different family. Provider families: `gemini` / `openai` / `anthropic` / `oss`. Sensitivity check (intra-family) flagged in output.
- [ ] Judge model recommendation: GPT-5 or Claude Opus 4.7 (whichever is strictly stronger than all 4 tested). For tested = GPT-5, switch Judge to Claude Opus; for tested = Claude, switch Judge to GPT-5; for tested = Gemini or OSS, default Judge = Claude Opus or GPT-5 (pick one consistently).
- [ ] Output written to `judge_verdicts/<run_id>.json` per run.
- [ ] Citation requirement enforced: each verdict includes a quoted SRS or rulebook excerpt; verdicts without a valid citation are auto-flagged for audit (WP-08).
- [ ] Smoke test: 5 known TPs + 5 known FPs (hand-built) classified correctly with citations.

## Implementation steps

1. Author the rubric prompt in markdown — full formal definitions, examples (drawn from §3.2 paper text), explicit "if you cannot find a citation, mark FP".
2. Implement `evaluation/judge.py` using the WP-01a abstraction (Judge is just another `LLMClient` instance).
3. Implement cross-family selection: read `configs/scope.yaml.models` → for each tested model, pick the strongest model from a different family.
4. Add `judge_run.py` CLI: `python -m evaluation.judge --run-manifest runs/<run_id>/manifest.json --srs <path> --output judge_verdicts/<run_id>.json`.
5. Smoke test on 10 hand-built cases.
6. Hook into Makefile so `make rq1 && make judge-rq1` is the standard sequence.

## Outputs (file paths)

- `prompts/judge_rubric.md`
- `evaluation/judge.py`
- `evaluation/judge_run.py` (CLI)
- `judge_verdicts/` (gitignored except for smoke-test fixtures)
- `tests/test_judge_smoke.py`
- Updated `Makefile`: `judge-rq1`, `judge-rq2`, `judge-rq3`, `judge-rq4`

## Risks & mitigations

- **Risk:** Judge LLM reports differ between runs (non-determinism). **Mitigation:** `temperature=0`, `seed=42`, plus N=2 Judge runs per verdict for the highest-stakes cases (RQ4 seeds). Disagreement triggers audit (WP-08).
- **Risk:** Judge has stale knowledge of the SRS (it sees only what's in the prompt). **Mitigation:** Pass the full SRS (or relevant sections) every call. Token budget: tracked by WP-01c's pricing — Judge is the heaviest single cost, plan for it.
- **Risk:** Citation requirement misfires; LLM fabricates citations that don't exist in SRS. **Mitigation:** Post-process check: each citation must be a substring of the SRS file. Mismatches are auto-flagged for audit.
