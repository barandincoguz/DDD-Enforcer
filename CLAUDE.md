# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

DDD-Enforcer is a VS Code extension + Python FastAPI backend that detects **Domain-Driven Design (DDD)** violations in Python code. Workflow: a multi-agent LLM pipeline reads an SRS document, extracts a domain model (`domain/model.json`), then validates Python files against it on save with AST + LLM analysis. Violations surface as VS Code diagnostics with traceable references back to the SRS via RAG.

**Active research context — read this before any non-trivial change:** This repo is being prepared for submission to **Springer Empirical Software Engineering** (EMSE). Authors: Baran Dincoguz, Ali Kendir, Prof. Dr. Murat Karakaya. The locked roadmap, locked decisions (D1–D7), and per-WP specs are in `todos/`. Start with **`todos/AGENT_QUICKSTART.md`** for orientation, then **`todos/MASTER_PLAN.md`** for canonical roadmap, then **`todos/WP_DAGILIM_BARAN_ALI.md`** for ownership boundaries. **Do not touch WPs you don't own** (see allocation file). Communication language is Turkish; code/comments stay English.

**Persistent development memory — also read before non-trivial work:** `development_docs/` is the manual cross-session memory layer for paper-revision context and Claude-session context recovery. Start with **`development_docs/INDEX.md`** to see which WPs have full docs and which have pointer placeholders. When a WP ships, write a doc there following the convention in the INDEX (one doc per WP; sections: TL;DR, motivation, architectural decisions with rationale, file-level changes, methodology, empirical results, limitations + follow-ups, cross-references). Git history says what changed; these docs say *why*.

## Common Commands

All Python commands run from `extension/backend/` unless stated.

### Setup (one-time)

```bash
# Use Python 3.12 (pinned everywhere — pyrightconfig, CI, README)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r extension/backend/requirements.lock   # hash-pinned, reproducible
```

### Repair a broken `.venv` (WP-CORE-29)

If you inherit a `.venv/` that points at Python 3.14 (e.g. via Homebrew
auto-upgrade) and `pip` / `pytest` are missing, rebuild from scratch:

```bash
# 1. Verify which Python is on PATH and which the venv is pinned to.
ls -la extension/backend/.venv/bin/python   # symlink target reveals version
python3.12 --version                        # confirm 3.12 is installed

# 2. Nuke + recreate. The `.venv/` is gitignored so nothing to commit.
rm -rf extension/backend/.venv
python3.12 -m venv extension/backend/.venv
source extension/backend/.venv/bin/activate
pip install --upgrade pip
pip install -r extension/backend/requirements.lock

# 3. Smoke test.
pytest -m "not integration" --maxfail=1 -q
```

CI still uses Python 3.12 per `requirements.lock` regardless of local
drift. Local 3.13 fallback works for most tests but is not the
supported version — keep 3.12 if you want CI-parity reproducibility.

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
| **Multi-agent pipeline** | `core/architect.py` | `DomainArchitect` — Scout/Architect/Specialist/Verifier/Refiner/Synthesizer (5-stage post-P3); rate limit + parallel Scout via `DDD_SCOUT_MAX_WORKERS` |
| **LLM provider abstraction** | `core/llm/` package | Post-WP-01a (shipped 2026-05-19): 8 modules — `base.py` (ABC + `LLMResponse`), `errors.py`, `registry.py` (D1 6-model SSOT), `retry.py` (`@with_retry_and_rotation`), `gemini.py` (`GeminiClient`), `ollama.py` (`OllamaClient` via Ollama Cloud), `validator.py` (Validator-stage `LLMClient`), `_response_adapter.py` (legacy genai-shape shim), `schema_probe.py` (6×3 conformance CLI). |
| **Domain Pydantic schemas** | `core/schemas.py` | `DomainModel`, `BoundedContext`, `Entity`, `ValueObject`, `Aggregate`, `DomainEvent`, `GlobalRules` |
| **Model registry (SSOT)** | `core/llm/registry.py` | D1 6-model `MODELS` dict with `ModelSpec` (pricing tiers, context window, compute mode, capabilities flags). Import-time `_validate_registry()` enforces invariants. `configs/models.py` retained for stage→group mapping. |
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
- `extension/backend/runs/probe-{ts}.json` + `runs/probe-{ts}.manifest.json` — schema_probe artifacts (first shipped 2026-05-19 as `probe-20260519-175042.*`). 16-key manifest carries hard git provenance, package versions, verbatim prompts. New artifacts go here; do not overwrite — names are timestamped.

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

- **WP-01a is shipped.** All LLM calls now go through `core/llm/`. The old `core/llm_client.py` was renamed to `core/llm/validator.py` (the Validator-stage client). New code uses `from core.llm import get_client`, `get_client_for_model`, or imports a specific provider directly. Do NOT re-introduce `google.genai` or `openai` imports outside `core/llm/gemini.py` and `core/llm/ollama.py`.
- **`core/llm/gemini.py:_RUNTIME_FALLBACKS` is intentionally empty.** Silent runtime model substitution would mislabel paper artifacts under the D1 lock. The dict is retained as a hook for future provider deprecations but must stay empty unless a fallback is *also* registered as a separate model_id in `registry.py`. `schema_probe.py` has a pre-flight check that aborts the run if any requested model has a declared fallback. See `development_docs/WP-NEW-B-Stage-1-schema-probe.md` §A4 + §A8 for the full rationale.
- **Rate limiting is real.** `DomainArchitect.min_delay` defaults to 6 s (free-tier safe). Override via `DDD_MIN_DELAY_SECONDS` env or kwarg if you have Pro RPM. Quota-error backoff is exponential (15 → 300 s).
- **Parallel Scout is opt-in.** Set `DDD_SCOUT_MAX_WORKERS=N` to fan out chunk extraction. Default is 1 (sequential). Token tracker is thread-safe via `threading.Lock`.
- **Truncation is head + tail.** `_truncate_with_head_tail` keeps the first 60 % and last 40 % of an oversized SRS chunk to preserve both intro/glossary and acceptance-criteria sections.
- **Domain-model location depends on `WORKSPACE_PATH`.** When the VS Code extension launches the backend, it sets `WORKSPACE_PATH` so `domain/model.json` lands in the user's project. When developing the backend standalone, it falls back to `extension/backend/domain/`.
- **CI is live and strict.** `.github/workflows/backend-ci.yml` runs `pytest -m "not integration" --cov-fail-under=60`. The `pyright` step is `continue-on-error: true` for now (will tighten as type errors are addressed). Don't add `|| echo "skip"` patterns to silence failures — that anti-pattern was just removed in commit `56919da`.
- **Local `.venv` is currently broken.** The checked-in `extension/backend/.venv/` is a Python 3.14 shell missing `pip`/`pytest`. Tests in practice run against system Python 3.13 from `/Library/Frameworks/Python.framework/Versions/3.13/bin/`. CI still uses 3.12 per `requirements.lock`. Treat the `.venv` as needing a repair commit (open follow-up in `development_docs/WP-NEW-B-Stage-1-schema-probe.md` §Limitations).
- **Commit style is Conventional Commits**, atomic, with the trailer:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- **TDD is the default for `core/llm/` work.** WP-01a shipped via test-before-implementation, and the `schema_probe` revision (WP-NEW-B Stage 1) followed the same pattern with 12 smoke tests written before any production-code change.

## Persistent Development Memory (`development_docs/`)

A separate folder from `todos/` (which is roadmap + spec for *upcoming* work). `development_docs/` is the *retrospective* layer — write a doc here when a WP ships so the paper can reference architectural decisions without re-deriving them from git log.

- **Read** `development_docs/INDEX.md` first. It tracks `ACTIVE` (full docs), `RESERVED` (commit-range pointers for prior-session WPs awaiting backfill), and `PENDING` (WPs not yet started).
- **Write** a new doc when a WP merges into `main`. Filename: `WP-<code>-<short-kebab>.md`. Every doc starts with status / branch / commit SHAs / spec + plan paths / one-paragraph TL;DR, then sections in the order: motivation, architectural decisions (numbered, with rationale), file-level changes (table), methodology applied, empirical results (if any), limitations + follow-ups, cross-references. Use `[[doc-name]]` syntax to link related docs (search-friendly; placeholder names mark backfill targets).
- **Update** the `ACTIVE` table in `INDEX.md` in the same commit as the new doc.
- **Do not** duplicate the commit log — explain *why* the change was made and how the pieces fit; let `git show <sha>` answer *what changed line-by-line*.
- **Backfill** for prior-session WPs (`WP-01a`, `P3 Verifier+Refiner`) is a low-friction task when paper revision needs the rationale; the commit ranges in `INDEX.md` make it tractable.

## Quick Reference

| Want to... | Look at |
|---|---|
| Understand the multi-agent pipeline | `core/architect.py` (5-stage post-P3 — Scout/Architect/Specialist/Verifier/Refiner/Synthesizer) |
| Add or change a model | `core/llm/registry.py` (SSOT for D1 lock) — and check `_RUNTIME_FALLBACKS` policy in `core/llm/gemini.py` |
| Add or change an LLM call site | Use `get_client()` / `get_client_for_model()` from `core.llm`. Never import `google.genai` or `openai` outside `core/llm/gemini.py`/`ollama.py`. |
| Trace a violation back to SRS | `core/rag_pipeline.py` + `extension.ts` `openSourceCommand` |
| Debug a pipeline run | `core/intermediate/{timestamp}_{stage}.json` files |
| Run on a new SRS | Drop in `extension/backend/inputs/` (or workspace `inputs/`) and trigger "DDD Enforcer: Initialize Domain Model" |
| Run schema conformance probe | `cd extension/backend && python -m core.llm.schema_probe --trials 5` (writes `runs/probe-{ts}.json` + manifest) |
| Find what's next to build | `todos/INDEX.md` (status) → owner's `todos/WP-XX-*.md` (spec) |
| Look up *why* a past WP made a decision | `development_docs/INDEX.md` → relevant WP doc |
| Check ownership before editing | `todos/WP_DAGILIM_BARAN_ALI.md` |
| Talk to the advisor | `todos/HOCA_GUNDEM.md` (current open topics) |
