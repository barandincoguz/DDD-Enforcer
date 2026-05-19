# WP-NEW-B Stage 1 — schema_probe Real Run

**Status:** SHIPPED 2026-05-19
**Branch:** `feat/schema-probe-real-run` → FF-merged into `main` (deleted)
**Commits on main (5):**
- `1d5fce1` docs(specs) — design doc with Codex juri integration
- `d3d34fa` docs(plans) — 15-task TDD implementation plan
- `27c11be` feat(llm/schema_probe) — code revisions (dotenv + manifest + transport_error split + fallback guard)
- `1e1fb71` fix(llm/gemini) — remove dead `gemini-3.1-flash-lite → gemini-2.5-flash` silent fallback (D1 integrity)
- `53fe15d` chore(artifacts) — schema_probe full run N=5 6×3 results

**Spec:** [`docs/superpowers/specs/2026-05-19-schema-probe-real-run-design.md`](../docs/superpowers/specs/2026-05-19-schema-probe-real-run-design.md)
**Plan:** [`docs/superpowers/plans/2026-05-19-schema-probe-real-run.md`](../docs/superpowers/plans/2026-05-19-schema-probe-real-run.md)
**Run artifact:** [`extension/backend/runs/probe-20260519-175042.json`](../extension/backend/runs/probe-20260519-175042.json) + `.manifest.json` (16-key sidecar)

---

## TL;DR

Shipped first real-LLM conformance probe of the **D1 6-model lock × 3 schema** matrix. 18 cells × 5 trials = 90 calls completed cleanly in 13min 32s wall time with zero transport errors. Closed Gemini scored **30/30** structured-output success; the 4 OSS-via-Ollama-Cloud models scored **1/60** combined — the strict `response_format={"type": "json_schema", "strict": true}` contract is not honored by gpt-oss, qwen3-coder, minimax, or gemma4. Three failure-mode classes captured in `raw_failures[]` (markdown fences, schema field renames, refusal-style prose). This is the source data for the paper's RQ2 Table 7 `json_failed` column. Before merging the run, the probe's own pre-flight check caught a silent runtime fallback in `GeminiClient` that would have mislabeled every G2 artifact in the EMSE submission — fixed inline as commit 4 of the chain.

---

## Motivation

D1 (locked in `todos/MASTER_PLAN.md`) commits the paper to six models in two providers:

- **G1** `gemini-3.1-pro-preview` (Gemini, paid tier)
- **G2** `gemini-3.1-flash-lite` (Gemini)
- **O1** `gpt-oss:120b-cloud` (Ollama Cloud)
- **O2** `qwen3-coder-next:cloud` (Ollama Cloud)
- **O3** `minimax-m2:cloud` (Ollama Cloud)
- **O4** `gemma4:31b-cloud` (Ollama Cloud)

Before launching RQ2 batch runs (a far more expensive operation), the project needs evidence that every model in this set can reliably emit Pydantic-validatable JSON under our prompt + schema contract. A 1.7%-OSS-conformance finding (which we now have) is critical: it tells us either that the OSS models are unfit for our pipeline as-currently-prompted, or that prompts need engineering (WP-NEW-C) before they can be benchmarked. Either way, this had to be measured directly, not assumed.

The probe itself existed in `core/llm/schema_probe.py` from WP-01a as a smoke-tested CLI, but had never run live. Production use would surface bugs hidden by mocked tests — and did.

---

## Architectural decisions (and their rationale)

### A1. Codex 5.5-xhigh as juri, Claude as filter

**Decision:** New review pattern introduced this session — user invokes Codex via the `codex:codex-rescue` subagent as an independent reviewer, then Claude filters all of Codex's findings through its own judgment. Codex makes no implementation changes; Claude decides what to accept and integrates into the design.

**Why:** Two-model review catches blind spots a single model misses. Codex's 11 review points exposed 2 actual blockers (transport_error mixing, silent fallback labeling) that Claude's initial design missed. Equally important: 1 of Codex's points was rejected after filtering (`with_suffix` "footgun" was speculative), keeping the change small. Without the filter step, Codex's review would have either been ignored entirely or accepted wholesale.

**Applied to:** initial design spec (11 points filtered to 10 acceptances), and again post-implementation as the code-quality reviewer round (11 sub-issues — 6 accepted, 1 partial override against the user's earlier "no raw capture" vote, 4 rejected as speculative or cosmetic).

### A2. Honest transport_error / json_failed separation

**Decision:** `CellResult` carries two independent counters. `transport_error` counts provider exceptions (auth, 429, 5xx, timeout, constructor failure). `json_failed` only increments when the model returned a parsed response that fails Pydantic schema validation. Markdown-wrapped JSON that breaks `json.loads` also lands in `json_failed` (it's a schema-output failure of the model, not infrastructure).

**Why:** RQ2 Table 7 wants to measure schema conformance under a strict API contract. Mixing infrastructure noise into the conformance metric inflates failures and misattributes them. Codex BLOCKER #2.

**Code shape:** `probe_cell()` has separate exception-handler branches; `main()` warns on stdout when any cell's `transport_error >= trials/2` because that cell's `json_failed` rate is no longer trustworthy.

### A3. Hard-fail on runtime model substitution

**Decision:** `probe_cell()` asserts `resp.model_id == requested_model_id` after every trial and raises `RuntimeError("Runtime fallback fired ...")` on mismatch. This is **paranoia even after the pre-flight check** because the pre-flight only catches *declared* fallbacks; the in-loop guard catches any server-side or undeclared substitution.

**Why:** D1 locks paper claims to specific model IDs. If `gemini-3.1-flash-lite` resolves at runtime to `gemini-2.5-flash`, every artifact labeled "G2" is actually measuring `gemini-2.5-flash` and the paper's D1 numbers are wrong. Codex BLOCKER #1.

**Coupling:** This guard is **load-bearing** — it caught the dead `_RUNTIME_FALLBACKS` entry in `gemini.py` on the very first real run, before any RQ data shipped. Without this guard the paper would have shipped with G2 silently being a different model.

### A4. Pre-flight check + partial-output-on-abort

**Decision:** Two-layer defense. `main()` reads `core.llm.gemini._RUNTIME_FALLBACKS` (private import — documented in code) and refuses (exit 1, stderr message) to start the probe if any requested model has a declared fallback. If the in-loop guard *does* fire (because something undeclared substituted server-side), `main()` writes a partial report + manifest with an `aborted: <message>` key before re-raising, so 17 cells of clean data don't get thrown away on cell 18.

**Why:** The first iteration of the code raised `RuntimeError` mid-loop and the report was only written AFTER `run_probe()` returned, so an abort produced ZERO files on disk. The code-quality reviewer caught this; without the fix, every mid-run failure would have been silent data loss. The pre-flight check is the cheap front line; the partial-output is the expensive backstop.

**Note:** This decision overruled the spec's original "single in-loop guard" design. The spec's Section 7 risk register predicted this exact failure mode; turning it from "stop the run" into "warn + finish + audit" came from the round-2 code-quality review.

### A5. Per-trial varying seed `[42, 43, 44, 45, 46]`

**Decision:** N=5 trials per cell now use *distinct* seeds, not a single fixed `seed=42`. Manifest records the strategy as `"per_trial_42_plus_i"`.

**Why:** Codex Q11. With fixed seed + temperature 0.05 + identical prompt, 5 trials = 5 (near-)deterministic repetitions, not 5 independent samples. Any reported success rate becomes 100% or 0%, with no within-cell variance signal. Varying seeds gives the probe genuine sample independence.

**Trade-off:** Different seeds make each trial unique, so two cells with the same model + schema are no longer bit-comparable. We lose seed-level reproducibility (mitigated by recording the seed list in manifest).

### A6. 500-char truncated raw capture on json_failed only

**Decision:** Whenever a trial returns `resp.json_failed=True`, append `(resp.content or "")[:500]` to `result.raw_failures`. Not captured on success; not captured on transport error.

**Why:** Paper appendix reviewers need to verify *what failure mode* a model exhibits. The original spec captured only `json_fail_reason` (the Pydantic error message), which says *what's wrong* but not *what the model returned*. The 500-char prefix is enough to see markdown fences, field renames, or refusal prose without bloating the artifact (final probe JSON is 22KB). Codex MEDIUM #10 — and this was an autonomous-mode **override** of the user's earlier vote on the "medium" fix scope, because Codex's audit argument was more compelling than my original sizing.

### A7. 16-key manifest with hard git provenance

**Decision:** Every probe run emits a sidecar manifest at `<out>.with_name(<stem>.manifest.json)` with: start/end ISO timestamps, `git_commit` (40-char SHA, *no* `"unknown"` fallback — `_git_head_or_raise` raises `CalledProcessError` if `git rev-parse` fails), `git_dirty` bool, `git_dirty_files` list, `python_version`, `platform`, `package_versions` (google-genai + openai + pydantic via `importlib.metadata`), `models`, `schemas`, `trials_per_cell`, `seed_strategy`, `temperature_default`, the **verbatim** `PROMPTS` dict, a `schema_fidelity_note` honest disclaimer, and `aborted` (None on clean runs).

**Why:** Paper-appendix artifact must be reproducible without trust in a human-written README. If a reviewer asks "what exact commit + what exact Python + what exact prompt", the manifest answers all three without needing to ask. Codex HIGH #5 forbade an "unknown" fallback — silent missing provenance is worse than a hard error.

### A8. Empty `_RUNTIME_FALLBACKS` in `gemini.py` (separate fix commit)

**Decision:** Verified via raw `google-genai` SDK call that `gemini-3.1-flash-lite` *is* now live on the API. The fallback entry → `gemini-2.5-flash` was defensive code written during WP-01a (when preview was gated) and has since become dead. Removed the entry, left the dict structure as an empty hook for future provider deprecations.

**Why:** Without this, the probe's own pre-flight check would have permanently blocked G2. We could have either: (a) accepted the silent substitution and mislabeled paper data, (b) dropped G2 from D1 (changes the locked spec), or (c) verified the model is live and remove the dead fallback. Option (c) is the smallest correct change that preserves D1.

**Coupling:** This was *not* in the original spec. It was committed inline mid-flight, between the code-change commit and the artifact commit, because the pre-flight check (correctly) refused to run until the situation was resolved. The fix is documented separately so future maintainers see *why* the dict is empty.

---

## File-level changes

### Files created

| Path | Purpose | Size |
|---|---|---|
| `docs/superpowers/specs/2026-05-19-schema-probe-real-run-design.md` | Design spec with Codex juri findings integrated | 316 lines |
| `docs/superpowers/plans/2026-05-19-schema-probe-real-run.md` | 15-task TDD implementation plan | 1315 lines |
| `extension/backend/runs/probe-20260519-175042.json` | 18 CellResult records, 90 trials of data | ~22KB |
| `extension/backend/runs/probe-20260519-175042.manifest.json` | 16-key reproducibility manifest | ~3.5KB |

### Files modified

| Path | Change | LOC diff |
|---|---|---|
| `extension/backend/core/llm/schema_probe.py` | Major revision: `CellResult` extended (3 fields), `probe_cell()` refactored, 5 new helpers (`_git_head_or_raise`, `_git_dirty_status`, `_pkg_version`, `_now_iso`, plus pre-flight import), `main()` expanded with dotenv + timestamped out + manifest sidecar + try/except for partial output | +198 -0 |
| `extension/backend/tests/test_llm_schema_probe.py` | 12 new smoke tests; one existing test rewritten because default `--out` semantics changed; three legacy tests adjusted to use matching model IDs (the new in-loop guard fires on mismatched mock responses) | +385 -7 |
| `extension/backend/core/llm/gemini.py` | `_RUNTIME_FALLBACKS` dict emptied; module docstring rewritten to explain why; comment block above the dict clarifies the policy | +5 -11 |
| `extension/backend/tests/test_llm_gemini.py` | Renamed + inverted: `test_gemini_client_falls_back_g2_to_2_5_flash` → `test_gemini_client_does_not_silently_rewrite_g2_model_id` (now asserts verbatim passthrough) | +3 -3 |

### Files NOT touched (intentionally)

`base.py`, `ollama.py`, `registry.py`, `retry.py`, `errors.py`, `_response_adapter.py`, `__init__.py`, `validator.py`, `main.py`, `architect.py`. The work was scoped to the probe + the gemini fix; touching the rest would have been speculative.

### Pre-existing dirty files (still uncommitted on main)

`AGENTS.md`, `extension/backend/core/AST/intermediate/ast_signals_diagnostics.json`, `extension/backend/validation_metrics_report.json`. These predate the branch and were intentionally not staged in any commit. Cleanup is open follow-up #6.

---

## Methodology applied

| Skill / pattern | How it was used |
|---|---|
| `superpowers:brainstorming` | 3 clarifying questions (scope, trials/cell, fix breadth) before writing the spec. Caught the `.env` autoload omission as a real bug before any code touched. |
| **Codex juri review (round 1)** | Pre-implementation review of the spec. 11 points raised → 10 accepted into spec → spec rewritten with `## 3. Juri review` table mapping each finding to a resolution. |
| `superpowers:writing-plans` | 15-task TDD plan with exact code blocks, test scaffolds, and expected pytest output per step. Saved as `2026-05-19-schema-probe-real-run.md`. |
| `superpowers:subagent-driven-development` | All 10 TDD micro-tasks bundled into a single implementer subagent (sonnet) because they all touched 2 files in tightly-coupled sequence. Then **spec-compliance reviewer** subagent (sonnet, read-only) followed by **code-quality reviewer** subagent (sonnet, read-only). |
| **Code-quality review iteration** | First quality review returned `NEEDS_REVISION` with 2 CRITICAL + 4 IMPORTANT + 3 minor issues. Fix subagent applied all but 3 (1 borderline kept, 2 cosmetic). Second quality review returned `APPROVED`. |
| **Manual phase-2 execution** | Tasks 13-15 (pre-flight smoke, full run, artifact commit) executed directly without subagent — live LLM calls + diagnostic judgment + commit message tied to live data. |
| **Final reviewer** | Whole-branch review (sonnet, read-only) before FF merge. Final verdict: MERGE. Surfaced 6 maintainer follow-ups. |
| **Autonomous mode** | User said "otomatik mode" partway through. Approval gates (design approval, code-quality re-review request) were short-circuited; explicit decisions still surfaced (D1 OSS gap, raw-output override). |

---

## Empirical results (paper-grade)

### Run summary

| Metric | Value |
|---|---|
| Cells executed | 18 (6 models × 3 schemas) |
| Trials per cell | 5 |
| Total calls | 90 |
| Successful (schema-valid JSON) | 31 (34.4%) |
| Schema-invalid (`json_failed`) | 59 |
| Transport errors | 0 |
| Wall time (start→end) | 13min 32s (2026-05-19 17:50:42 → 18:04:14) |
| Seeds used | `[42, 43, 44, 45, 46]` per cell |
| Default temperature | 0.05 |
| Reproducibility commit (manifest `git_commit`) | `1e1fb71d02412d6845b02bd94d38fed949e26bc0` |
| Python | 3.13.3 (note: project pins 3.12 — see follow-up #4) |
| `google-genai` | 1.41.0 |
| `openai` | 1.55.3 |
| `pydantic` | 2.11.10 |

### Per-model success rate

| Model | Provider | success / 15 | mean latency (basic / medium / complex) |
|---|---|---|---|
| G1 `gemini-3.1-pro-preview` | gemini | **15 / 15** (100%) | 4173 / 8213 / 9593 ms |
| G2 `gemini-3.1-flash-lite` | gemini | **15 / 15** (100%) | 1170 / 2794 / 1732 ms |
| O1 `gpt-oss:120b-cloud` | ollama | 0 / 15 (0%) | 6076 / 5428 / 9870 ms |
| O2 `qwen3-coder-next:cloud` | ollama | 0 / 15 (0%) | 1066 / 1562 / 5872 ms |
| O3 `minimax-m2:cloud` | ollama | 1 / 15 (6.7%) | 15753 / 8126 / 12179 ms |
| O4 `gemma4:31b-cloud` | ollama | 0 / 15 (0%) | 25233 / 13126 / 30302 ms |

**Bimodal**: closed Gemini family = perfect; OSS via Ollama Cloud = near-zero. Even the one OSS success (O3 × basic, 1/5) was an outlier within an otherwise-failing pattern.

### OSS failure modes (from `raw_failures[]` capture)

Three distinct classes, identified by reading first 500 chars of model output on every failing trial:

1. **Markdown code-fence wrap** (gpt-oss × all, gemma4 × medium+complex, minimax × basic+medium):
   Models emit ```` ```json\n{...}\n``` ```` — valid JSON nested inside markdown. Our `json.loads` fails at column 0. The OpenAI-compatible `response_format={"type": "json_schema", "strict": true}` contract does *not* suppress this wrapping for these models.

2. **Schema field renames** (qwen3-coder × all, minimax × medium, minimax × complex):
   Models emit valid raw JSON but with semantically-renamed keys: `entity` instead of `name`, `bounded_context` instead of `context_name`, `entity_name` instead of `name`, `attribute_name` instead of `name`. Pydantic validation fails on missing required fields. Strict schema enforcement is being ignored by the model.

3. **Refusal-style prose** (gemma4 × basic):
   Model emits English prose: *"Since you didn't provide a specific schema, I have generated a JSON object based on a standard Entity Definition Schema..."*. The `response_format` schema is invisible to gemma4 on the basic prompt.

### What this finding means for the paper

For RQ2 (which compares structured-output reliability across models), the **raw** finding is that closed Gemini handles strict JSON schema cleanly while OSS-via-Ollama-Cloud does not — a defensible empirical claim under the contract we tested. But the failure modes above are all candidates for prompt engineering (WP-NEW-C). The paper needs to decide:

- **Option A:** Report raw conformance as the headline; treat prompt engineering as a separate question.
- **Option B:** Run WP-NEW-C first, then report best-prompt-per-model conformance.
- **Option C:** Both — raw conformance as the main result, prompt-engineered conformance as a robustness check.

This decision is **open** (follow-up #2).

---

## Limitations + follow-ups

### Limitations of the probe as designed

- **Single prompt per schema.** No prompt engineering applied. The result measures *adherence under our specific contract*, not *achievable adherence after engineering*. The `schema_fidelity_note` in the manifest hedges this explicitly.
- **`ComplexViolation` approximates BoundedContext, not the exact production contract.** The manifest's `schema_fidelity_note` calls this out. A faithful BoundedContext schema is deeper and pulls in more constraints; using it for the probe would have confounded "model can emit deep nesting" with "model handles our specific aggregate/event vocabulary".
- **No retry on `json_failed` cells.** A failed trial counts as failed; we do not let the model try again with a corrective prompt. (Intentional — the probe measures one-shot conformance.)
- **No latency normalization for cold start.** O4 gemma4 × complex took 30 seconds mean — partly cold-start cost on Ollama Cloud's serverless infra. Latency comparisons across models are not apples-to-apples.

### Follow-ups (from final-reviewer subagent)

1. **WP-NEW-B Stage 2** — paper-side Markdown table render of `probe-20260519-175042.json` for the RQ2 Table 7 appendix. Deferred; not blocking.
2. **D1 OSS gap decision** — 4 of 6 models scored 0/15. Before RQ2 batch runs, the project must decide whether to report raw, run WP-NEW-C first, or both. **High priority.**
3. **WP-NEW-C prompt sensitivity ablation** — likely the right answer to (2). 3 prompt variants per pipeline, mean±std reporting; would directly address the markdown-wrap and field-rename failure modes.
4. **Broken `.venv` path** — the project's `.venv` is Python 3.14 (not 3.12 as `CLAUDE.md` claims) and lacks `pip`/`pytest`. The probe ran against system Python 3.13 from `/Library/Frameworks/Python.framework/Versions/3.13/bin/`. This means the `requirements.lock` hash-pinning is not actually being enforced for local development. **Medium priority.**
5. **Python 3.13 vs 3.12 pin reconciliation** — manifest records 3.13.3 but `pyrightconfig.json` and CI pin 3.12. For the EMSE appendix the paper should either disclose this or regenerate the probe on 3.12 once the venv is repaired.
6. **`AGENTS.md` dirt cleanup** — pre-existing uncommitted change on `main`. Belongs in a dedicated `chore(docs):` commit. **Low priority** but bloc clean future audits.

### `.env` rebuild (procedural note)

User's `.env` had 8 individual `OLLAMA_API_KEY`, `OLLAMA_API_KEY_V2`, ... `OLLAMA_API_KEY_V8` entries; `OllamaClient` expects a single comma-separated `OLLAMA_API_KEYS` plural variable. Bridged by appending a consolidated `OLLAMA_API_KEYS=k1,k2,...,k8` line to `.env` via python-dotenv (never reading the values into Claude's context — only the count was printed). Lesson: `.env.example` should make the plural form unambiguous (it does, since commit `e380983`); user's existing setup predated the example.

---

## Cross-references

- [[WP-01a-provider-abstraction]] — provides the `core/llm/` package this work patches; current INDEX entry is a pointer-only placeholder until the full doc backfills
- [[P3-verifier-refiner-refactor]] — the pipeline the probe ultimately serves (RQ1 territory); INDEX placeholder pending
- `todos/MASTER_PLAN.md` — canonical roadmap; this WP is row "WP-NEW-B" in the master table
- `todos/HOCA_GUNDEM.md` — TEDU external rater recruitment is the next blocker on the RQ critical path, independent of this work but flagged here for context

## Decision log (for paper revision lookups)

When the paper Methods or Results section references *anything* from the probe data, the relevant decision is:

| Paper claim site | Decision number | Doc section |
|---|---|---|
| "We measured one-shot schema conformance under a fixed strict-JSON contract" | A2 + Limitations | §3.A2, §Limitations |
| "Per-trial seeds 42-46 ensure independent samples" | A5 | §3.A5 |
| "All artifacts are version-pinned via 16-key manifest including git SHA and SDK versions" | A7 | §3.A7 |
| "We pre-screen models against declared runtime fallbacks to prevent label-vs-call drift" | A4 | §3.A4 |
| "We disabled the WP-01a gemini-3.1-flash-lite → gemini-2.5-flash fallback after verifying flash-lite is live on the API" | A8 | §3.A8 |
| "OSS-via-Ollama-Cloud models do not honor strict `response_format=json_schema`" | Empirical results + Limitations | §Empirical results, §Limitations |
