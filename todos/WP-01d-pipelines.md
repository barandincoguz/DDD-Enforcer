# WP-01d: Pipeline Implementations (P1, P2, P3) + Multi-Run Orchestrator

**Owner:** Ali
**Depends-on:** [WP-01a, WP-01b, WP-01c]
**Effort:** M (P3 already exists; P2 RAG mostly exists; P1 needs to be cleanly extracted; multi-run orchestrator is new)
**Status:** TODO
**Addresses instructor feedback:** [Hoca-1] (necessary substrate for tables)

## Goal

Land three explicit, swappable pipelines (P1 Naive, P2 RAG, P3 Multi-Agent) plus a multi-run orchestrator that runs the same configuration N=5 times. The paper claims a 3-pipeline RQ1 comparison; the codebase only ships P3 cleanly + P2 partially + P1 missing as a standalone class (`00-context-report.md` Q4–Q6). After WP-01d, all three pipelines are first-class citizens behind one CLI.

**Critical:** Each pipeline must produce `Violation` objects matching the `ValidationResponse` schema (`llm_client.py:33`). The single contract is what makes RQ1 comparable.

## Acceptance criteria

- [ ] `extension/backend/core/pipelines/p1_naive.py`: receives raw SRS + raw source code; single LLM call; returns `ValidationResponse`. No AST, no rulebook, no chunking. Free-form output is parsed best-effort; unparseable runs flagged in manifest (parseable% reported in Table 6).
- [ ] `extension/backend/core/pipelines/p2_rag.py`: chunks SRS, builds ChromaDB index, retrieves top-k=5 chunks per source file (query = file's class+method names), passes chunks + AST features to LLM, returns structured `ValidationResponse`. Mostly extracted from existing `rag_pipeline.py`.
- [ ] `extension/backend/core/pipelines/p3_multi_agent.py`: refactored from existing `architect.py`; preserves Scout→Architect→Specialist→Synthesizer flow; produces rulebook then validates per-file.
- [ ] All three implement the same interface: `Pipeline.run(srs_path, code_path, model_config) -> RunManifest`.
- [ ] Multi-run orchestrator: `python -m ddd_enforcer.run_pipeline --pipeline {p1|p2|p3} --srs <path> --code <path> --model <config> --runs 5 --out runs/`.
- [ ] `make rq1` Makefile target: D1 × 1 sabit LLM × 3 pipelines × 5 runs = 15 runs, all written to `runs/`.
- [ ] Reproducibility: `temperature=0.05` and `seed=42` propagate via `model_config`; OS env `PYTHONHASHSEED=0` set in Makefile target.
- [ ] Smoke test passes: `make rq1` produces 15 manifests in `runs/`, each table-buildable by WP-01b's `build_tables.py`.

## Implementation steps

1. Define `Pipeline` ABC in `extension/backend/core/pipelines/base.py` with `run()` signature.
2. Refactor `architect.py` into `p3_multi_agent.py` — minimal-change refactor preserving all 4 agent stages.
3. Refactor `rag_pipeline.py` into `p2_rag.py` — keep ChromaDB + MiniLM; expose as Pipeline.
4. Write `p1_naive.py` from scratch — straight LLM call with raw SRS + raw code; no AST, no retrieval. Best-effort regex parse if response is not JSON.
5. Write `run_pipeline.py` orchestrator: invokes the chosen pipeline N times with N different `seed` values (or repeated `seed=42` if temperature=0 is sufficient — investigate provider behavior; `00-context-report.md` Q19 confirms seed=42 hardcoded).
6. Wire `RunManifest` writes (from WP-01b) into orchestrator post-run.
7. Add `Makefile` targets: `make rq1`, `make rq2`, `make rq3`, `make rq4`, `make rq5` (last only if RQ5 chosen as ablation).
8. Regression-test against the 154 legacy intermediate files: ensure P3 produces equivalent output (allowing for refactor cosmetic differences).

## Outputs (file paths)

- `extension/backend/core/pipelines/__init__.py`
- `extension/backend/core/pipelines/base.py`
- `extension/backend/core/pipelines/p1_naive.py`
- `extension/backend/core/pipelines/p2_rag.py`
- `extension/backend/core/pipelines/p3_multi_agent.py`
- `extension/backend/run_pipeline.py` (CLI orchestrator)
- `Makefile` with `rq{1,2,3,4,5}` targets
- `tests/test_pipelines_smoke.py`
- Documentation: `docs/PIPELINES.md` (one paragraph each)

## Risks & mitigations

- **Risk:** P1 free-form output regex parsing leads to misleadingly low parseable%. **Mitigation:** Try 3 prompt formulations, pick the most parseable, and document the choice in §4.4 as a deliberate methodology decision (this is honest reporting; it does not hide P1's weakness).
- **Risk:** Provider-level seed=42 doesn't actually fix non-determinism (some providers ignore the seed). **Mitigation:** Run a quick "5× same seed → same output?" check per provider in the smoke test; if a provider ignores seed, document in §9.3.4 and treat its variance as larger.
- **Risk:** P3 refactor breaks the Scout→Architect→Specialist→Synthesizer chain. **Mitigation:** Tag commit before refactor; smoke test compares old vs new on `inputs/SRS.docx`.
