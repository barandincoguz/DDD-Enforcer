# SESSION_CONTEXT.md — Living Briefing for Future Claude

> **Read this first** if you are a fresh Claude instance arriving cold to this repo. Goal: get oriented in <30 seconds without re-exploring. **Update this file** at the end of any non-trivial work session.

**Last updated:** 2026-04-27 (modularity guideline relaxed: 400-line hard cap → ~500/800/1200 soft guideline)
**Last session ended at commit:** `b9ea67b` (origin/main)
**Branch:** `main` (= origin/main, clean)

---

## 1. Project Identity

- **Repo:** `DDD-Enforcer` — multi-agent LLM pipeline that detects Domain-Driven Design violations between an SRS and a codebase.
- **Target deliverable:** Springer EMSE journal extension of a UBMK conference paper. Paper sources live in `LaTeX_DL_468198_240419/`.
- **Primary language:** Python 3.13 (backend), TypeScript (`extension/src/extension.ts`, VS Code extension shell — not actively edited recently).
- **User:** Baran (paper writing, corpus, lit). Co-author Ali (infra, runs, stats). Hoca = supervisor.
- **Authoritative project rules:** `AGENTS.md`. File-size is a **guideline, not a wall**: ~500 effective lines sweet spot, ~800 review trigger, ~1200 pressure point. Split by responsibility, not by line count.

## 2. Architecture Cheat-Sheet

```
extension/backend/
├── configs/
│   └── models.py           ← SINGLE SOURCE OF TRUTH for models, pricing, stage→model mapping
├── core/
│   ├── architect.py        ← 4-stage agent pipeline (Scout, Architect, Specialist, Synthesizer)
│   ├── token_tracker.py    ← thread-safe singleton (track_api_call, tokens_for_stage)
│   ├── token_tracker_report.py  ← pure reporting fns (compute_cost, build_report, …)
│   ├── token_tracker_types.py   ← shared dataclasses
│   ├── AST/                ← modular AST analysis (signal_classification/discovery/enrichment/…)
│   ├── code_parser/        ← AST visitor pipeline (advanced_signals, helpers, models, service)
│   ├── document_parser.py + document_parser_readers.py  ← SRS ingestion
│   ├── parser.py           ← orchestrator
│   └── llm_client.py
├── config.py               ← AnalyzerConfig/ArchitectConfig now derive from configs/models.py
├── main.py                 ← FastAPI entrypoint; cost reporting uses registry helpers
└── tests/                  ← pytest suite, 83/83 passing as of b9ea67b
```

**Models in active use** (in `configs/models.py`):

- `gemini-3.1-pro-preview` (tiered: $2/$12 ≤200k prompt tokens, $4/$18 above) — used for Scout/Architect/Specialist/Synthesizer (domain extraction stages)
- `gemini-3-flash-preview` (flat $0.50/$3) — used for Validator (validation stage)
- **Never reintroduce `flash-lite`.** It was deliberately removed.

## 3. Recent Major Work (this session)

| Phase                              | Outcome                                                                                                                                                                                                                             | Key commits                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Model registry consolidation       | Created `configs/models.py` as single source of truth; tiered pricing; `stage_config()` / `model_for_stage()` helpers; 30-test registry suite                                                                                       | `3c32ead`, `bee7206`, `a085bc1`, `549ff8f`, `51bd58a`                       |
| TokenTracker refactor              | Split into 3 modules (tracker/report/types) for clarity; added `threading.Lock` for parallel Scout. (Note: under the new soft guideline a single ~500-line file would also have been fine — split was driven by then-stricter cap.) | `bee7206`, `10226a5`                                                        |
| Architect.py B-scope (bugs + perf) | Fixed B1–B5, added head+tail truncation, configurable `min_delay`, opt-in parallel Scout via `DDD_SCOUT_MAX_WORKERS` env or kwarg                                                                                                   | `2dfd1e6`, `4383680`, `0aca9ac`, `76402a9`, `0eff6d6`, `4e7e634`, `c1b444b` |
| Critical leak fixes                | main.py:737-739 was using literal flash-lite pricing (5–7× undercount); main.py:920-925 had hardcoded gemini-2.5-flash prices. Both now registry-derived                                                                            | `be9224a`, `7c2c06e`                                                        |
| AST/parser modularization          | Recovered + applied user's pre-session in-progress refactor (`AST/`, `code_parser/`, split parser/document_parser)                                                                                                                  | `56c42f3`                                                                   |
| Paper bundle commit                | LaTeX_DL_468198_240419/ + resources/ pushed to main                                                                                                                                                                                 | `942ced1`                                                                   |

## 4. Conventions & Feedback Learned

- **Modularity guideline (soft):** ~500 effective lines sweet spot, ~800 triggers a review-time "is this one responsibility?" question, ~1200 is a pressure point. **Do not split a cohesive file just to hit a number** — that was the prior failure mode. Split by responsibility.
- **No flash-lite, no gemini-2.5-\* anywhere.** Search-and-destroy if you see literals.
- **Prices/models must come from `configs/models.py`.** Never hardcode `0.10/0.40` etc. anywhere — that was the bug class C1/C2.
- **TDD discipline expected** for new behavior: failing test → run → impl → pass → commit. User noticed when skipped.
- **Brainstorming → writing-plans → execution** is the canonical skill chain for non-trivial work. User invoked `/superpowers:brainstorming` and `/effort high` explicitly.
- **No "novel" / "first" / "prove" in paper.** Tone is "framework / proof-of-concept / today's capability level / swappable" (per WP-14).
- **Commit messages:** Conventional prefix (`feat:`, `fix:`, `refactor:`, `perf:`, `chore:`, `docs:`). Co-Authored-By trailer present on session commits.
- **Stash safety:** I dropped the wrong stash once. Now I always `git stash show stash@{N}` before dropping. Don't repeat the mistake.

## 5. Pyright / IDE

- `pyrightconfig.json` at repo root sets `extraPaths: ["extension/backend"]` for module resolution. If you see false-positive `configs.models` import errors in IDE diagnostics, it's stale LSP cache — runtime tests are authoritative.

## 6. Test Suite

- Run: `cd extension/backend && pytest -q` → expect 83 passing.
- New tests added this session: `test_models_registry.py` (30), `test_token_tracker_v2.py` (9), `test_registry_snapshot.py` (3 drift guards), `test_architect_helpers.py` (~6), `test_token_tracker_concurrency.py` (1 smoke), plus parser/AST/document_parser tests added by user's recovered refactor.
- **Known issue:** `test_api.py` uses `httpx.ConnectError` workaround instead of `TestClient` — flagged as C3 in cumulative review, not yet migrated. Optional cleanup task.

## 7. EMSE Paper Plan (Phase 0–2 artifacts)

All in `todos/`:

- `00-paper-baseline.md`, `00-context-report.md`, `00-instructor-feedback-mapping.md`
- `01-brainstorming.md`, `01-risks.md`, `02-literature.md`
- `WP-00 … WP-18.md` (18 work packages, dependencies declared)
- `INDEX.md` is the entry point.
- Spec/plan pattern: `0X-<topic>-spec.md` followed by `0X+1-<topic>-plan.md` (e.g., `03/04` for registry, `05/06` for architect).

**Lit doc verification tags:** 🟢 VERIFIED-FROM-MEMORY, 🟡 PROBABLE, 🔴 RISKY. Do not strip them — they document confidence level since WebSearch was blocked.

## 8. Outstanding / Possible Next Tasks

User explicitly **deferred** these — do NOT start without explicit ask:

- WP-01a full provider abstraction (current registry is Gemini-only impl, interface-generic)
- Test infrastructure expansion ("aşırı test ve test infra çok yersiz şuanlık")
- WP-09 practitioner survey, WP-18 RQ5 (ablation vs developer study — Baran chooses)

**Cleanup candidates** (user-mentioned but not requested):

- `.gitignore` for `intermediate/*.json`, `validation_metrics_report.json` (runtime artifacts currently tracked)
- Drop stash@{0} (RECOVERED — content now in main) and stash@{1} (intermediate diagnostic)
- Delete remote `origin/feat/EnhancedDocumentParserModule` and local `feat/EnhancedDocumentParserModule`
- Migrate `test_api.py` to `TestClient` (C3 finding)

## 9. How to Resume

1. `git status` and `git log --oneline -5` to confirm where main is.
2. Check `todos/INDEX.md` if user mentions a WP number.
3. For any registry/pricing/model question → `extension/backend/configs/models.py`.
4. For agent pipeline question → `extension/backend/core/architect.py` (979 lines, 4 stages, parallel Scout opt-in).
5. Ask user before destructive ops (stash drop, branch delete, force push). They've been burned before.

## 10. Update Protocol

When you finish a session, edit this file:

- Bump `Last updated` and `Last session ended at commit`.
- Append a row to §3 if the work was material.
- Move items between §8 buckets as they get done or deferred.
- Add new conventions to §4 if user gives new feedback.
- Keep total file under 200 lines — trim history older than 2 sessions if needed.
