# Development Docs Index

> **Purpose**: Persistent design + implementation memory for the DDD-Enforcer EMSE submission. Two use cases drive what goes here:
>
> 1. **Paper revision substrate** — when revising the paper, every empirical claim or architectural decision should trace to a doc here that explains *why* it was made.
> 2. **Context recovery** — if a Claude session ends and a new one starts, reading the relevant docs here brings the new session up to speed without re-deriving decisions.

## Conventions

- **One file per Work Package** (WP) or per major architectural change. WPs follow the naming scheme in `todos/MASTER_PLAN.md`.
- **Filename**: `WP-<code>-<short-kebab>.md` (e.g. `WP-NEW-B-Stage-1-schema-probe.md`) or `arch-<topic>.md` for cross-cutting architecture notes.
- **Every doc leads with**: status, branch / commit SHAs, spec + plan paths, one-paragraph TL;DR.
- **Sections**: Motivation • Architectural decisions • File-level changes • Methodology applied • Empirical results (if any) • Limitations + follow-ups • Cross-references.
- **Cross-link** using `[[doc-name]]` syntax (no markdown link required — search-friendly).
- **Don't put commit messages here verbatim** — the git log is authoritative. Docs explain *why* and *how the pieces fit together*; commits explain *what changed line-by-line*.

## Active (shipped or in-flight, full docs)

| # | Doc | WP | Status | Headline |
|---|---|---|---|---|
| 1 | [WP-NEW-B Stage 1 — schema_probe real run](WP-NEW-B-Stage-1-schema-probe.md) | WP-NEW-B | SHIPPED 2026-05-19 | 6×3 conformance probe shipped real paper data; closed Gemini perfect (30/30), OSS-via-Ollama strict-schema near-zero (1/60) |

## Reserved (prior-session work; backfill on demand)

These WPs shipped in earlier sessions (before this folder existed). Full docs would require backfilling from git history. Pointers below cover what is needed for paper revision until full docs land.

| WP | Commit range | One-line summary | Backfill priority |
|---|---|---|---|
| **WP-00 Phase 0 hygiene** | `4a893c8`–`56919da` (4 commits, ≈2026-05-07) | Python 3.12 pin, `requirements.lock` hash-pinned, CI strictening to coverage 60% + pyright continue-on-error | Low — `git log` and `.github/workflows/backend-ci.yml` are self-documenting |
| **Planning docs** | `f5be4dc` and adjacents (≈2026-05-08) | `CLAUDE.md`, `todos/AGENT_QUICKSTART.md`, `todos/MASTER_PLAN.md`, `todos/WP_DAGILIM_BARAN_ALI.md` introduced | Low — the files in `todos/` ARE the documentation |
| **P3 Verifier+Refiner refactor** | `0bc10ec`–`f0da5d5` (25 commits, 2026-05-18→05-19) | 4-stage → 5-stage pipeline (Scout / Architect / Specialist / **Verifier** / **Refiner** / Synthesizer); FM-01/02/04/05/06/07/16/20/21 fixed; silent fallback removal + section-aware chunking + evidence-citation tightening (OQ1/OQ2/OQ3) | **Medium** — paper RQ1 leans on this; backfill before RQ1 write-up |
| **WP-01a Provider abstraction** | `b627505`–`e380983` (9 commits + 1 doc fix, 2026-05-19) | `core/llm/` package: 8 modules (base+errors+registry+retry+gemini+ollama+validator+_response_adapter+schema_probe). 6-model D1 registry. Retry+rotation decorator. `core/llm_client.py` → `core/llm/validator.py`. | **High** — paper Methods section references this; backfill before Methods write-up |

## Pending (not yet started)

| WP | Trigger / dependency | Notes |
|---|---|---|
| **WP-NEW-A AST drift injector** | RQ4 + Yaklaşım F unblock; ~3-5 days estimate | `scripts/inject_drift.py` V1-V6 quota CLI |
| **WP-NEW-B Stage 2** paper-side render | Stage 1 artifact (shipped) | Markdown table generator for `runs/probe-*.json` → RQ2 Table 7 appendix |
| **WP-NEW-C** prompt sensitivity ablation | Stage 1 findings — OSS conformance is bimodal | 3 prompt variants per pipeline, mean±std reporting |
| **WP-02** D2/D3 codebase sourcing | Public SRS + CLEAN codebases for banking + healthcare domains | Required before RQ3 cross-domain runs |

## How to use this folder going forward

- **Before starting a new WP**: spec it under `docs/superpowers/specs/`, plan it under `docs/superpowers/plans/`, *implement*, then write a doc here when it ships.
- **Inside a session**: if a doc here is touched (status change, new follow-up, empirical update), update the row in this INDEX too.
- **When paper-revising**: read the doc(s) for the WPs that produced the data or claim under revision; never recompute from git log without the doc.
- **Context-recovery primer**: read INDEX.md → most-recent doc → spec + plan → `git log --oneline main..main~10`. That sequence reconstructs ≈80% of a recent session's context in under 5 minutes.
