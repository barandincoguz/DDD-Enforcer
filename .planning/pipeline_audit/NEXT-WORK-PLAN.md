# NEXT-WORK-PLAN — Iteration 45+ candidates

**Written:** 2026-05-24 11:55 GMT+3
**Revised:** 2026-05-24 12:00 GMT+3 — WP-01d CANCELLED per user
direction (different pipeline architectures won't be explored on
this paper). Re-ranked accordingly.
**Author:** Claude Opus 4.7 + 5 parallel subagent investigation
**Baseline at write:** HEAD `39dc89e` (Pyright tightening complete),
716 tests passing, full `pyright` = 0 errors, 28 commits ahead origin.

This plan was built from 5 parallel subagent investigations
(cavecrew-investigator + Explore + general-purpose) over the full
codebase. It supersedes the menu in `handoff-2026-05-24-0010.md §10`
and the deferred items in `CURRENT.md`.

---

## 1. Quick rank — Top of the queue (post-WP-01d cancellation)

| Rank | WP | ROI | Risk | Autonomy | Effort | Approach |
|------|----|----|------|----------|--------|----------|
| 1 | **F-8 XXE close-out** | High (clears security register) | Low | Full | 2h | Direct |
| 2 | **Minor deferred sweep** (10 items) | Medium (latent-bug elimination) | Low | Full | 2.7h | Direct (6 atomic commits) |
| 3 | **VerifierResult dual-type unification** | Medium (architecture clean + cast removal) | Mid-High | Full | 4-6h | SDD (3-4 tasks) |
| 4 | **Tests/ pyright re-enable** | Low | Low | Full | 3-5h | Mixed (mostly bulk `# type: ignore`) |
| 5 | **WP-CORE-28 Extension UX wave 1** | Mid | High (no spec, manual smoke) | Partial | TBD | Spec first (brainstorm), then SDD + human F5 |
| 6 | **WP-CORE-32 Extension webviews** | Mid (visualizes WP-01b data) | High (no spec, webview manual smoke) | Partial | TBD | Spec first (brainstorm), then SDD + human F5 |
| 7 | **paper.tex `\input{rqN.tex}` integration** | High (paper-data wired E2E) | N/A | Human-only | n/a | Human coordinator task |

**~~WP-01d P1/P2/P3 pipeline classes~~ — CANCELLED 2026-05-24 per user.**
Different pipeline architectures (P1 naive vs P2 RAG vs P3 multi-agent
comparison) won't be explored for this EMSE submission. The current
`DomainArchitect` pipeline remains the sole production path. The
`PaperRunManifest.pipeline: Optional[Literal["P1","P2","P3"]]` field
stays on the schema (no migration cost; future contributors may
revisit). RQ1 will report on the single shipped pipeline only — the
paper narrative will need adjustment but that's a human-coordinator
task, not autonomous.

---

## 2. Recommended sequencing

| Iter | WP | Rationale |
|------|----|-----------|
| 45 | F-8 XXE close-out | 2h security register cleanup. No prereqs. Defense-in-depth. |
| 46 | Minor deferred sweep | 2.7h. Eliminates 10 carryover items in 6 atomic commits. Resets quality register before bigger refactors. |
| 47 | VerifierResult unification | 4-6h SDD. Removes the iter-44 cast bridge in pipeline.py. **No longer blocked by WP-01d.** |
| 48 | Tests/ pyright re-enable | 3-5h. Mostly cosmetic. Wait until VerifierResult unified (cluster 6 of test errors resolves as byproduct). |
| 49+ | WP-CORE-28 → 32 | Specs must be brainstormed + written first (post-iter-46 deliverable). Paired with human F5 smoke sessions. **Not autonomous-safe.** |

**Parallelizable**: Rank 1 (F-8) and Rank 2 (Minor sweep) touch
non-overlapping files; could ship in same session if user wants.

**Hard block**: WP-01d needs user un-defer signal. WP-CORE-28/32 need
user availability for VS Code smoke sessions.

---

## 3. WP details (one per item)

### Rank 1 — F-8 XXE close-out (2h)

**Subagent verdict (general-purpose):** *No Critical/High findings.*
Both python-docx (lxml with `resolve_entities=False` at
`docx/oxml/parser.py:19`) and pypdf (custom XMP builder denying
entities at `pypdf/xmp.py:170-206`) already harden against
CWE-611. The project never imports `xml.*`, `lxml`, or `defusedxml`
directly.

**Recommended scope (smallest defensible WP):**

1. Bump pypdf floor in `requirements.txt:20` from `>=4.0.0` to
   `>=6.0,<7` (keeps `_XmpBuilder` guarantees).
2. Add startup assertion in `main.py` lifespan that
   `docx.opc.oxml.oxml_parser.resolve_entities is False` and
   `docx.oxml.parser.oxml_parser.resolve_entities is False` — fails
   loud on upstream regression.
3. Add input size cap (`os.path.getsize(file_path) > MAX_SRS_BYTES`)
   in `SRSDocumentParser.parse_file` at `core/document_parser.py:84`
   for billion-laughs / decompression-bomb defense. Suggested cap:
   50 MB (configurable via env var).
4. Close F-8 in the security register; write up findings in
   `development_docs/F-8-xxe-assessment.md` linking to upstream
   library lines that mitigate.

**Acceptance tests (write these):**
- DOCX-XXE fixture (DOCTYPE + ENTITY): `read_docx(fixture)` returns
  literal `&xxe;` text, not `/etc/passwd` contents.
- DOCX billion-laughs: raises within bounded wall-clock budget.
- Size cap: 51 MB dummy `.docx` raises before `docx.Document` call.
- Startup assertion: monkey-patch flag to `True`, app refuses boot.

**Risk:** Low. Pure additive changes (new tests, new assertion,
dep floor bump). No existing test should break.

**Approach:** Direct (3-file scope). SDD threshold not reached.

**Why now:** Open security item lingering since codebase audit
(observation 3395, 2026-05-19). Trivial close-out.

---

### Rank 2 — Minor deferred concerns sweep (2.7h)

**10 items** from WP-CORE-30b + WP-01b/c reviews. Subagent
extracted from `CURRENT.md` + commit bodies; no new items found in
recent git log.

**Batched into 3 atomic commits:**

**Commit A: Severity + edge-case fixes (45 min)**
- `core/llm/validator.py:_to_legacy_issue` — silent severity-fallback
  warning log added (latent bug: silent str→Literal coercion masks
  invalid input).
- `core/refiner/prompts.py:render_refinement_prompt` — uppercase
  severity label (cosmetic; aligns with contract Literal).
- `core/architect.py:_specialist_with_feedback` — short-result-list
  guard (raise + log when N < threshold; latent bug).
- `core/token_tracker.py:by_stage` — case-fold stage keys at insert
  time so `"Scout"` and `"scout"` aggregate together (latent bug).

**Commit B: DRY utilities (60 min)**
- Extract `core/io_atomic.py:write_atomic(path, content)` helper.
  Refactor 3 call sites: `run_manifest.py`, `aggregate.py`,
  `latex_tables.py`. Add 5-test fixture (crash mid-fsync).
- Extract `core/run_id.py:sanitize_run_id_segment(s)` shared regex
  helper. Refactor `compose_run_id` + `compose_aggregate_key` to use it.
- `core/llm/validator.py:_parse_target_ctx` — refactor to call
  `_issue_stage` (DRY).
- `core/aggregate.py:AggregatedConfiguration.schema_version` writer
  guard (assert `version in SUPPORTED_VERSIONS` at construct, not
  just consume).

**Commit C: Test coverage (30 min)**
- `core/aggregate.py:pipeline=None` grouping — add 2 tests to
  `tests/test_aggregate.py` covering the `pipeline=None` branch
  (currently uncovered).

**Risk:** Low. Each fix surgical; commits independent. Tests catch
regression.

**Approach:** Direct (file-scope per commit ≤ 4 files). SDD threshold
not reached.

**Why now:** 10 carryover items will keep growing if not swept.
Latent-bug subset (severity-fallback + by_stage capitalization +
short-result-list + schema_version guard) is non-trivial.

---

### Rank 3 (CANCELLED) — WP-01d P1/P2/P3 pipeline classes

**STATUS: CANCELLED 2026-05-24 per user direction.** Different
pipeline architectures will NOT be explored for this paper. The
section below is preserved for historical context only. Skip to
Rank 4 (VerifierResult) for the next active WP.

Original scope (do NOT execute):

**Subagent verdict (Explore):** PaperRunManifest already has
`pipeline: Optional[Literal["P1", "P2", "P3"]]` (run_manifest.py:205,
shipped iter 38). No P1/P2/P3 classes exist; no `core/pipelines/`
directory.

**Scope:** 3 pipeline implementations + 1 ABC + 1 CLI orchestrator +
1 smoke regression test + Makefile `rq1` target. P3 is a surgical
refactor of `core/architect.py` (1270 LOC); P2 extracts
`core/rag_pipeline.py` (561 LOC); P1 is net-new (raw SRS+code →
single LLM call → regex-parsed violations).

**5 SDD tasks (parallel after Task 1):**

1. **Base ABC + P1** (1.5h) — `core/pipelines/base.py` + `p1_naive.py`
   + tests.
2. **P2 RAG** (1.5h) — `core/pipelines/p2_rag.py` (extract from
   `rag_pipeline.py`, no logic change).
3. **P3 Multi-agent** (2.5h) — `core/pipelines/p3_multi_agent.py`
   (refactor `architect.py` into the ABC interface; preserve all
   existing stages).
4. **CLI orchestrator** (1.5h) — `run_pipeline.py` with
   `--pipeline {P1,P2,P3} --runs N --domain D --model M` flags;
   writes per-run `PaperRunManifest`.
5. **Smoke + Makefile** (1h) — `tests/test_pipelines_smoke.py`
   (3 pipelines × 1 model × 1 domain × 2 runs = 6 manifests), then
   `make rq1` wires aggregate → build_tables → tables/RQ1.tex.

**Files to create:** `core/pipelines/{__init__,base,p1_naive,p2_rag,
p3_multi_agent}.py`, `run_pipeline.py`, `tests/test_pipelines_smoke.py`,
`docs/PIPELINES.md`. **Files to modify:** `Makefile`.

**Risk:** Mid. P3 refactor touches the load-bearing
`DomainArchitect` chain; mitigation = tag-before-refactor + 2-pipeline
smoke fixture before refactor + commit per task.

**Approach:** SDD strongly recommended. Precedent: WP-01b shipped
6 tasks via SDD with zero regression.

**Blocker:** User deferred this WP. **Must ask user before starting.**
If un-deferred, this becomes the highest-ROI iter 45 candidate
(direct paper-data unblock).

---

### Rank 3 — VerifierResult dual-type unification (4-6h)

**Subagent verdict (cavecrew-investigator):** *55 construction sites,
13+ import statements, 2 adapter functions to delete.* Both
`VerifierResult` AND `VerifierIssue` have the dual-type problem
(legacy dataclass at `core/verifier/types.py` vs Pydantic at
`core/pipeline_contracts.py`).

**Field mismatch (must reconcile before unification):**

| Legacy VerifierIssue | Contract VerifierIssue | Action |
|----------------------|------------------------|--------|
| `stage` (Literal["scout","architect","specialist","synthesizer"]) | — | KEEP — derive from `target` prefix if not on type; or add to Pydantic |
| `location` (str) | `target` (str) | RENAME (pick one) |
| `issue_type` (str) | `check_id` (str) | RENAME (pick one) |
| `severity` (IssueSeverity enum: "error"/"warn") | `severity` (Literal["ERROR","WARN"]) | NORMALIZE case + drop enum |
| `message` (str) | `message` (str) | same |
| `suggestion` (Optional[str]) | — | KEEP (developer-facing UX) |
| `srs_path` (Optional[str]) | `srs_path` (Optional[str]) | same |

**Recommendation:** Keep Pydantic, drop legacy dataclass. Add
`stage` + `suggestion` to Pydantic (use Optional for `suggestion`).
Rename `target` → `location` and `check_id` → `issue_type` so
existing 9 production check sites need fewer edits.

**3-4 SDD tasks:**

1. **Schema convergence** — update Pydantic VerifierIssue to be a
   superset (add `stage`, `suggestion`, rename `target`→`location`,
   rename `check_id`→`issue_type`); update VerifierResult same way.
2. **Check sites migration** (9 prod + 17 tests) — point
   `checks_deterministic.py`, `checks_semantic.py`,
   `checks_semantic_d6_d7_d8.py` and architect.py at Pydantic.
   Delete legacy dataclasses.
3. **Delete adapter + cast** — `core/architect.py:1000-1018`
   (`_to_contract_issue`), `core/orchestration/pipeline.py:355-357`
   cast bridge.
4. **Test fixture migration + regression smoke** — update 17 test
   construction sites.

**Risk:** Mid-High. 55 construction sites is the largest cross-file
refactor since WP-01a. Tests catch most regression; manual review
needed for stage-discard paths.

**Approach:** SDD (4 tasks, sequential — schema first, then
migrations).

**Why now (post WP-01d cancellation):** No longer blocked. After
iter 45 (F-8) + iter 46 (Minor sweep), this is the next big refactor
on the queue.

---

### Rank 4 — Tests/ pyright re-enable (3-5h)

**Subagent verdict (cavecrew-investigator):** 119 errors break
down to 41 noise (A: 5 MagicMock + B: 12 Optional + C: 24 Literal)
and 78 "real test bugs" (mostly cluster 1 in
`test_paper_run_manifest.py`).

**Important nuance:** "Real test bugs" doesn't mean code bugs — all
716 tests pass at runtime. These are static type violations the
tests intentionally exercise (e.g., parametrized loops passing
invalid strings to test that Pydantic rejects them). The category
boundary between bucket C and bucket E is fuzzy.

**Approach:**

1. **Bulk-suppress noise (1h)** — add `# type: ignore[arg-type]`
   to bucket C (intentional Literal violations, 24 errors). Add
   `# pyright: reportAttributeAccessIssue=false` at module top of
   files using `patch.object` heavily (5 errors). Add
   `# type: ignore[optional-subscript]` to bucket B sites that
   actually intend to test None paths (12 errors).
2. **Real test cleanups (2-4h)** — for bucket E cluster 1
   (`test_paper_run_manifest.py:120,357-420`), re-examine the
   parametrized fixtures: if intentional negative tests, suppress;
   if accidental misuse, fix the fixture. For clusters 2-11 (26
   errors), individual fixes (mock signatures, FastAPI BaseRoute
   `isinstance(route, Route)` guards, len() None guards).
3. **Re-enable in `pyrightconfig.json`** — remove
   `extension/backend/tests` from `exclude`.

**Risk:** Low. Suppression comments are reversible; real fixes
backed by 716 passing tests.

**Approach:** Mixed — bulk suppress = mechanical Direct edits;
real cleanups = SDD-lite if cluster 1 turns out to be a fixture
design issue.

**Why low priority:** Test pyright is cosmetic — real test bugs
already caught by pytest. Wait until VerifierResult is unified
(resolves cluster 6 of 5 errors as byproduct).

---

### Rank 5 — WP-CORE-28 Extension UX wave 1 (TBD)

**Subagent verdict (Explore):** *No spec file exists.* Only
referenced as "TypeScript work, manual smoke required, NOT
autonomous-safe" in CURRENT.md.

**Current state of extension/src/:**
- `extension.ts` — 1,581 LOC monolith. Handles: activation, backend
  child-process lifecycle, API key flow (settings → env → secret
  storage → user prompt), domain model init (SSE streaming +
  fallback POST), validate-on-save (semantic fingerprint), status
  bar, code action provider, port discovery.
- `test/extension.test.ts` — 204 LOC, 14 tests (API integration
  surface + URI handling + semantic classification). **No tests
  for backend HTTP, SSE parsing, webview, or diagnostic creation.**

**Pre-work needed:**

1. **Write spec** — `todos/WP-CORE-28-extension-ux-wave1.md`.
   Decide: which UX pain points are P0 (status bar verbosity?
   validation feedback delay? API key UX? command palette ergonomics?
   error toasts?). Acceptance criteria per feature.
2. **Spike** — 30-min sketch of refactor surface (monolith → modules)
   without touching code.

**Sequence: spec → SDD impl → human F5 smoke → merge.** Cannot be
completed autonomously.

---

### Rank 6 — WP-CORE-32 Extension webviews (TBD)

Same as Rank 6 but for webview dashboards (PaperRunManifest viewer,
validation results table, metrics breakdown). Builds on WP-01b
data (`core/metrics.py` + `core/run_manifest.py`). Manual smoke
+ visual review unavoidable.

**Pre-work:** Spec, then SDD impl, then human F5 smoke.

---

### Rank 7 — paper.tex `\input{rqN.tex}` integration

**Human task.** `LaTeX_DL_468198_240419/tables/README.md` lists
candidate line numbers (227, 394, 456, 479) for inserting the
4 RQ table blocks generated by `make tables`. Requires
paper-writing judgment Claude shouldn't make autonomously.

---

## 4. Cross-cutting infrastructure follow-ups

These don't merit standalone WPs but should ride along with the
nearest scoped change:

- **`core.AST.mutability_index` IDE pyright false positive** — only
  appears in IDE-side diagnostics, not CLI pyright. Likely cached
  state. Investigate if it surfaces in a fresh contributor env.
- **`.venv` repair commit** — local `.venv` was rebuilt in iter 43
  with python3.13 + `requirements.txt` + dev tools. Stays
  gitignored; CI uses python 3.12 + lockfile. No commit needed.
  CLAUDE.md docs still warn about broken `.venv`; update to
  reflect the rebuild recipe is now Python 3.13 too (since 3.12
  not always available on dev machines).
- **Pyright `extension/backend/legacy_pre_emse` exclude** —
  227 archived intermediate JSONs. Already excluded; mention in
  CONTRIBUTING when written.

---

## 5. Decision matrix

**RESOLVED 2026-05-24:**

1. ~~Un-defer WP-01d?~~ **CANCELLED** — different pipeline
   architectures off scope for this paper.
2. **Run F-8 close-out + Minor sweep as one session?** Plan:
   sequential within current session (iter 45 → iter 46), each
   with verification + commits before next.
3. **Schedule WP-CORE-28/32 spec-writing sessions?** Brainstorming
   immediately AFTER iter 46 ships, via AskUserQuestion in the
   same session.
4. ~~VerifierResult before or after WP-01d?~~ Moot — WP-01d cancelled.
   VerifierResult unification is now rank 3, scheduled for iter 47.

---

## 6. Process notes

- All recommendations assume the constraints in CLAUDE.md +
  `feedback-accuracy-over-cost` + `feedback-sdd-default`.
- `feedback-sdd-default` threshold: 3+ files OR cross-stage OR
  Codex-REQUIRE → SDD. Single-file mechanical fixes → Direct.
- No `git push` without explicit "push it".
- No model-tier downgrades (D1 lock).
- Atomic commits with conventional-commits trailer +
  `Co-Authored-By: Claude Opus 4.7 (1M context)`.

---

## 7. Source for this plan

5 parallel subagent dispatches on 2026-05-24 11:50 GMT+3:

1. cavecrew-investigator — `VerifierResult` dual-type call-site map
2. cavecrew-investigator — `tests/` pyright 119-error categorization
3. general-purpose — F-8 XXE threat-surface assessment
4. Explore — Extension UX state + WP-CORE-28/32 spec audit
5. Explore — WP-01d scope + minor deferred sweep extraction

Full raw outputs not retained in this document; available in
the iteration-44 session transcript for cross-reference.

End of plan.
