# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

DDD-Enforcer is a VS Code extension + Python FastAPI backend that detects **Domain-Driven Design (DDD)** violations in Python code. Workflow: a multi-agent LLM pipeline reads an SRS document, extracts a domain model (`domain/model.json`), then validates Python files against it on save with AST + LLM analysis. Violations surface as VS Code diagnostics with traceable references back to the SRS via RAG.

**Active research context — read this before any non-trivial change:** This repo is being prepared for submission to **Springer Empirical Software Engineering** (EMSE). Authors: Baran Dincoguz, Ali Kendir, Prof. Dr. Murat Karakaya. The locked roadmap, locked decisions (D1–D7), and per-WP specs are in `todos/`. Start with **`todos/AGENT_QUICKSTART.md`** for orientation, then **`todos/MASTER_PLAN.md`** for canonical roadmap, then **`todos/WP_DAGILIM_BARAN_ALI.md`** for ownership boundaries. **Do not touch WPs you don't own** (see allocation file). Communication language is Turkish; code/comments stay English.

## Common Commands

All Python commands run from `extension/backend/` unless stated.

### Setup (one-time)

```bash
# Use Python 3.12 (pinned everywhere — pyrightconfig, CI, README)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r extension/backend/requirements.lock   # hash-pinned, reproducible
```

`.env` (in `extension/backend/`, never committed):
```
GEMINI_API_KEY=<your-key>
OLLAMA_API_KEYS=key1,key2,key3,key4,key5,key6        # 6-key rotation pool, post-WP-01a
WORKSPACE_PATH=/abs/path/to/workspace                 # set by VS Code extension automatically
```

### Run backend

```bash
cd extension/backend
uvicorn main:app --reload
# FastAPI on http://127.0.0.1:8000
# /health, /status, /generate-model, /generate-model-stream, /validate
```

### Tests

```bash
cd extension/backend

# Unit tests only (CI default — fast, no live API needed)
pytest -m "not integration"

# Integration tests (require live backend at DDD_BACKEND_URL)
DDD_BACKEND_URL=http://localhost:8000 pytest -m integration

# Single test
pytest tests/test_unit.py::TestCodeParser::test_parse_valid_code -v

# With coverage gate (CI behavior)
pytest -m "not integration" --cov=. --cov-report=term --cov-fail-under=60

# Type-check (pyright config in repo root)
pyright   # strict mode for extension/backend
```

`tests/conftest.py` sets a placeholder `GEMINI_API_KEY` so unit tests don't need real credentials. `tests/test_api.py` is the only integration suite (marked `pytestmark = pytest.mark.integration`).

### Extension (TypeScript)

```bash
cd extension
npm ci
npm run compile   # type-check + emit
npm run lint
# Press F5 in VS Code to launch Extension Development Host
```

### Lint / format

No ruff/black config in repo. `pyright` is the active type-check gate. Tests enforce structure via `--strict-markers` in `pytest.ini`. **Do not introduce a new formatter without discussion** (see AGENTS.md "Do not make unrelated changes").

## High-Level Architecture

### The two halves

```
┌──────────────────────────────────────┐         ┌──────────────────────────────────────┐
│  extension/src/extension.ts          │         │  extension/backend/main.py            │
│  (VS Code extension, TypeScript)     │ HTTP    │  (FastAPI server, Python)             │
│                                      │ ──────► │                                       │
│  - Spawns backend as child process   │         │  - Lifespan: parses SRS → architects  │
│  - Validates on save (semantic       │ ◄────── │    domain model → enriches with AST   │
│    fingerprint to skip whitespace)   │         │    → indexes RAG                      │
│  - Renders violations as diagnostics │         │  - /validate: AST + LLM + RAG sources │
│  - Code actions to jump to SRS       │         │  - /generate-model{,-stream}: SSE     │
└──────────────────────────────────────┘         └──────────────────────────────────────┘
```

### Backend pipeline (where the multi-agent core lives)

`extension/backend/main.py` is thin glue. The actual work is in `core/`:

```
SRS document
   │
   ▼  core/document_parser.py + document_parser_readers.py
   │  (PDF/DOCX/TXT → plain text)
   │
   ▼  core/architect.py — DomainArchitect, the 4-stage pipeline:
   │
   │   1. Scout       → extract domain-relevant sentences (chunked, optionally parallel)
   │   2. Architect   → identify bounded contexts from sentences
   │   3. Specialist  → analyze each context for entities/value objects/services/aggregates
   │   4. Synthesizer → merge into core/schemas.py:DomainModel (Pydantic)
   │
   │  Each stage is rate-limited via _wait_for_rate_limit; intermediate JSON
   │  outputs are written to core/intermediate/ for debugging + replay.
   │  Stage→model mapping comes from configs/models.py:STAGE_TO_GROUP.
   │
   ▼  core/AST/ast_model_signals.py — ASTModelSignalExtractor.enrich_domain_model()
   │  Walks workspace Python files, attaches confidence + sources to each
   │  Entity/ValueObject so violations can be traced back to evidence.
   │
   ▼  domain/model.json (the persisted domain model)

Validation request (POST /validate)
   │
   ▼  core/parser.py — CodeParser facade, delegates to core/code_parser/service.parse_source_code
   │  AST extraction: classes, functions, imports, advanced signals (V4 boundary,
   │  V5 aggregate, V6 domain event candidates) via core/code_parser/visitor.py.
   │
   ▼  core/llm_client.py — LLMClient.analyze_violation()
   │  - Deterministic short-circuit: rule_based_name_violations (V1 synonym, V2 banned)
   │  - LLM call only if has_advanced_validation_signals(ast_data) is True
   │  - Pydantic-strict ValidationResponse via Gemini structured output
   │  - Hallucination filter; deterministic fallback after retries
   │
   ▼  core/rag_pipeline.py — RAGPipeline.search_violation_source()
   │  ChromaDB vector search over chunked SRS; top-k passages attached as sources.
   │
   ▼  Response: {is_violation, violations: [{type, message, suggestion, sources}], metrics}
```

### Module map (what to read where)

| Concern | Path | Notes |
|---|---|---|
| **FastAPI app + endpoints** | `extension/backend/main.py` | 972 LOC; lifespan, /generate-model{,-stream}, /validate |
| **Multi-agent pipeline** | `core/architect.py` | `DomainArchitect` — Scout/Architect/Specialist/Synthesizer; rate limit + parallel Scout via `DDD_SCOUT_MAX_WORKERS` |
| **LLM call layer (legacy)** | `core/llm_client.py` | **Will be deleted in WP-01a big-bang refactor**; new `core/llm/` package replaces it |
| **Domain Pydantic schemas** | `core/schemas.py` | `DomainModel`, `BoundedContext`, `Entity`, `ValueObject`, `Aggregate`, `DomainEvent`, `GlobalRules` |
| **Model registry (SSOT)** | `configs/models.py` | `MODELS`, `STAGE_GROUPS`, tiered pricing, import-time validation |
| **Application config** | `config.py` (root of `extension/backend`) | Paths, RAG, server, parser. `BASE_DIR`/`WORKSPACE_PATH` resolution |
| **AST analysis** | `core/AST/*.py` (8 files) | Signal classification/discovery/enrichment/grounding/types/utils + `ast_model_signals.py` |
| **Code parser (post-refactor)** | `core/code_parser/*.py` | New modular split (`service`, `visitor`, `models`, `helpers`, `advanced_signals`); `core/parser.py` is a thin facade |
| **RAG over SRS** | `core/rag_pipeline.py` | ChromaDB + sentence-transformers; chunk size 250, top-k=3 |
| **Token tracking** | `core/token_tracker*.py` (3 files) | Thread-safe singleton, per-stage breakdown, supports parallel Scout |
| **Validation metrics** | `core/validation_metrics.py` | Per-validation run records → `validation_metrics_report.json` |
| **VS Code extension** | `extension/src/extension.ts` | Lazy backend spawn, semantic fingerprint to skip no-op edits, SSE progress UI |

### Why `core/parser.py` and `core/ast_model_signals.py` look tiny

These are intentional facades. `core/parser.py` (12 lines) re-exports `CodeParser` which delegates to `core/code_parser/service.parse_source_code`. `core/ast_model_signals.py` (5 lines) re-exports `ASTModelSignalExtractor` from `core/AST/`. They keep public import paths stable (`from core.parser import CodeParser`, used by `main.py` and tests) while internals were modularized. **Do not bypass these facades from new code** — import the facade, not the inner module.

### Data + intermediate state

- `extension/backend/inputs/` — SRS files dropped in (D1 lives here as `SRS.docx`)
- `extension/backend/domain/model.json` — generated domain model (or in workspace if `WORKSPACE_PATH` set)
- `extension/backend/core/intermediate/` — Scout/Architect/Specialist/Synthesizer per-stage JSON dumps (timestamped); useful for debugging the pipeline without re-running
- `extension/backend/data/chroma_db/` — RAG vector store (gitignored)
- `extension/backend/validation_metrics_report.json` — committed runtime artifact (legacy pattern; new runs go under `runs/` post-WP-01b)

## Conventions (from AGENTS.md)

`AGENTS.md` is the canonical engineering charter. Key rules in priority order:

- **Smallest correct change.** Don't refactor unrelated parts; don't rename without need; don't mix feature work with cleanup.
- **Modularity sweet spot ~500 effective lines** (excluding comments/blanks). Review trigger ~800. Pressure point ~1200. Split by responsibility, not line count. `architect.py` (979 LOC) sits in review territory — leave it intact unless a real responsibility split exists.
- **Stable entrypoints; isolate change-prone logic.** The facade pattern in `core/parser.py` is an example.
- **Error handling: explicit failure.** No empty `try/except`, no silent degradation, no permissive fallbacks during development. Convert exceptions, add context, or rethrow.
- **No speculative generalization.** Don't add abstractions for hypothetical future requirements unless the change is already clearly expected.
- **No backward-compatibility shims when scope is big-bang.** Per `MASTER_PLAN.md`, WP-01a deletes `core/llm_client.py` outright; do not preserve old call sites with adapters.

## Active Submission Context (Critical)

The repo's `main` branch carries 7 commits since 2026-05-07 establishing the EMSE submission roadmap. The TODO state lives entirely in `todos/`:

- **`todos/AGENT_QUICKSTART.md`** — read this first; project orientation
- **`todos/MASTER_PLAN.md`** — canonical 6-phase roadmap; D1–D7 locked decisions; 23 active WPs; verification checklist
- **`todos/WP_DAGILIM_BARAN_ALI.md`** — ownership; cohesion clusters; 7 sync-point handoffs (read-only)
- **`todos/INDEX.md`** — status board with TODO/IN-PROGRESS/DONE flags
- **`todos/HOCA_GUNDEM.md`** — items to discuss with advisor (Murat Karakaya)
- **`todos/WP-XX-*.md`** — per-WP specs with acceptance criteria, implementation steps, risks
- **`todos/WP-NEW-{A,B,C,D}-*.md`** — four NEW WPs added by the 2026-05-08 audit
- **`todos/WP-09-*.md` and `todos/WP-18-*.md`** — DROPPED from active scope (banner at top); retained for history

### Locked decisions you must respect

- **D1 — 6 models**: G1 `gemini-3.1-pro-preview`, G2 `gemini-3.1-flash-lite`, plus 4 OSS via Ollama Cloud (`gpt-oss:120b-cloud`, `qwen3-coder-next:cloud`, `minimax-m2:cloud`, `gemma4:31b-cloud`). Provider abstraction = 2 clients (`GeminiClient` + `OllamaClient` over OpenAI-compatible API at `https://ollama.com/v1`). New track-able metric: `json_failed` rate.
- **D2 — Yaklaşım F**: 3 industries (D1 e-commerce, D2 banking, D3 healthcare); each domain ships 3 codebase variants (`code-clean/`, `code-drift-light/`, `code-drift-heavy/`) under `subjects/`. Drift injection is **automated** via `scripts/inject_drift.py` (WP-NEW-A); no manual drift edits.
- **D3 — 3-rater Fleiss's κ** (not Cohen's): Baran + Ali + 1 external TEDU professor. Murat is supervisor/author but **not a rater** — do not add him to audit pipeline code.
- **D4 — N=10 baseline** with a Hafta 4 pilot variance gate; orchestrator must support resume (idempotent worker, run-spec YAMLs in `runs/specs/`, outputs in `runs/outputs/`).
- **D6 — RQ5 is silently dropped.** Do not reference RQ5 in `paper.tex` or in any new docs/code. Reviewer-facing material is RQ1–RQ4 only.

### Files NOT to touch unless you are the owner

- WPs are owner-tagged; cross-touching them creates merge friction. See `WP_DAGILIM_BARAN_ALI.md`.
- `LaTeX_DL_468198_240419/paper.tex` is shared writing space — coordinate before editing.
- `extension/backend/core/intermediate/*.json` and `validation_metrics_report.json` are runtime artifacts; refresh in dedicated `chore(artifacts):` commits, not mixed with code changes.
- `extension/backend/.env` is never committed. Confirmed via `.gitignore`; never re-add.

## Things to Know Before Changing Code

- **`core/llm_client.py` is on the chopping block.** WP-01a's 9-commit big-bang refactor moves all LLM call sites to a new `core/llm/` package and deletes `llm_client.py`. New code should target the new package, not patch the old client.
- **Rate limiting is real.** `DomainArchitect.min_delay` defaults to 6 s (free-tier safe). Override via `DDD_MIN_DELAY_SECONDS` env or kwarg if you have Pro RPM. Quota-error backoff is exponential (15 → 300 s).
- **Parallel Scout is opt-in.** Set `DDD_SCOUT_MAX_WORKERS=N` to fan out chunk extraction. Default is 1 (sequential). Token tracker is thread-safe via `threading.Lock`.
- **Truncation is head + tail.** `_truncate_with_head_tail` keeps the first 60 % and last 40 % of an oversized SRS chunk to preserve both intro/glossary and acceptance-criteria sections.
- **Domain-model location depends on `WORKSPACE_PATH`.** When the VS Code extension launches the backend, it sets `WORKSPACE_PATH` so `domain/model.json` lands in the user's project. When developing the backend standalone, it falls back to `extension/backend/domain/`.
- **CI is live and strict.** `.github/workflows/backend-ci.yml` runs `pytest -m "not integration" --cov-fail-under=60`. The `pyright` step is `continue-on-error: true` for now (will tighten as type errors are addressed). Don't add `|| echo "skip"` patterns to silence failures — that anti-pattern was just removed in commit `56919da`.
- **Commit style is Conventional Commits**, atomic, with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- **TDD is the default for `core/llm/` work.** Each commit in the WP-01a 9-commit sequence (see `todos/WP-01a-provider-abstraction.md`) ships test before implementation, all green, all rollback-able.

## Quick Reference

| Want to... | Look at |
|---|---|
| Understand the multi-agent pipeline | `core/architect.py` |
| Add or change a model | `configs/models.py` (and post-WP-01a, `core/llm/registry.py`) |
| Trace a violation back to SRS | `core/rag_pipeline.py` + `extension.ts` `openSourceCommand` |
| Debug a pipeline run | `core/intermediate/{timestamp}_{stage}.json` files |
| Run on a new SRS | Drop in `extension/backend/inputs/` (or workspace `inputs/`) and trigger "DDD Enforcer: Initialize Domain Model" |
| Find what's next to build | `todos/INDEX.md` (status) → owner's `todos/WP-XX-*.md` (spec) |
| Check ownership before editing | `todos/WP_DAGILIM_BARAN_ALI.md` |
| Talk to the advisor | `todos/HOCA_GUNDEM.md` (current open topics) |
