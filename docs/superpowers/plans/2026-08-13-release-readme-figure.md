# Release Engineering, README, and Pipeline Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, auditable `1.1.0` release and a publication-accurate English repository presentation with a validated pipeline SVG.

**Architecture:** One root quality command becomes the local/CI source of truth for locked Python installation, audits, backend/extension verification, documentation, Mermaid, citation, and VSIX packaging. Documentation is written only after runtime names and commands are final; the README distinguishes the published conference system from the expanded current implementation.

**Tech Stack:** Python 3.12, uv/pip hashes, pip-audit, Node 20, npm, VS Code 1.106.1, vsce, Mermaid CLI 11.16.0, GitHub Actions, CFF 1.2

## Global Constraints

- Extension version is `1.1.0`; Python runtime is exactly 3.12; Node CI is 20; VS Code engine/test runtime is 1.106.1.
- Runtime dependencies install from a universal hash-pinned `requirements.lock`; test/audit tools install from a universal hash-pinned `requirements-dev.lock`.
- `npm audit --omit=dev` and the pinned Python runtime audit are blocking and must report no known vulnerabilities at final verification.
- Full npm audit remains visible; compatible dev fixes are applied, but no unsafe forced major rewrite is allowed.
- Backend non-integration tests all pass with at least 90% coverage over maintained production Python modules.
- Actual VS Code tests run under `xvfb-run` on Linux CI; compile/lint alone are insufficient.
- VSIX contains runtime only, stays below 10 MiB, and is built to a temporary path; no generated VSIX is committed.
- Do not claim Marketplace availability, production readiness, full locality, automatic fixing, instant validation, every-language support, or self-healing.
- Gemini is the default external provider; the implemented Ollama-compatible path is an advanced backend option, not an extension selector.
- The software license statement applies only to `extension/`; do not add a root MIT license/badge or assign the paper DOI to the software.
- Published metrics are explicitly scoped to the IISEC 2026 conference system.
- Use `creating-mermaid-diagrams` for figure source/export and `humanizer` file mode for final README prose.
- Preserve `.DS_Store`; do not delete historical branches/stashes without explicit confirmation.

---

## File Structure

### Create

- `extension/backend/requirements-dev.in`
- `extension/backend/requirements-dev.lock`
- `extension/backend/.coveragerc`
- `extension/src/backend/setupCheck.ts`
- `extension/src/test/backend/setupCheck.test.ts`
- `extension/scripts/assert-vsix-contents.mjs`
- `scripts/quality.sh`
- `scripts/check-documentation.mjs`
- `.github/workflows/quality.yml`
- `docs/assets/ddd-enforcer-pipeline.mmd`
- `docs/assets/ddd-enforcer-pipeline.svg`
- `docs/assets/mermaid-config.json`
- `CITATION.cff`
- `SECURITY.md`
- `CONTRIBUTING.md`
- `README.md` through a case-safe rename of `readme.md`

### Modify

- `extension/backend/requirements.lock`
- `extension/package.json`
- `extension/package-lock.json`
- `extension/.vscodeignore`
- `extension/.vscode-test.mjs`
- `extension/CHANGELOG.md`
- `extension/README.md`
- `extension/src/extension.ts`
- `Makefile`

### Replace

- `.github/workflows/backend-ci.yml` and `.github/workflows/extension.ci.yml` with the consolidated quality workflow after the new workflow passes locally.

## Quality Interfaces

```text
make quality
  → scripts/quality.sh
  → clean locked Python 3.12 environment
  → pip check + pip-audit
  → backend pytest ≥90% + Pyright + compileall
  → npm ci + production/full audit
  → TypeScript + ESLint + VS Code tests
  → documentation/CFF/Mermaid validation
  → temporary VSIX + content/size assertions
```

---

### Task 1: Archive the Unique Governance Stash and Prepare Worktree Isolation

**Files:**

- Modify: `.gitignore` only if `.worktrees/` is not already ignored.

**Interfaces:**

- Produces: verified tag `archive/governance-stash-2026-03-13` resolving to `d6a1a22cec321a261ca32bb5d4b59c9f54e28204` and an isolated implementation worktree.

- [ ] **Step 1: Verify the immutable source SHA**

Run:

```bash
git cat-file -t d6a1a22cec321a261ca32bb5d4b59c9f54e28204
git stash list --format='%gd %H %gs'
```

Expected: `commit` and a stash entry with that exact SHA.

- [ ] **Step 2: Create and verify the archival tag**

```bash
git tag -a archive/governance-stash-2026-03-13 \
  d6a1a22cec321a261ca32bb5d4b59c9f54e28204 \
  -m "Archive incomplete ImprovingGeneral governance-contract experiment"
git rev-parse 'archive/governance-stash-2026-03-13^{}'
```

Expected: exact stash SHA. Do not drop the stash.

- [ ] **Step 3: Ensure project-local worktrees are ignored**

Run `git check-ignore -q .worktrees`. If it fails, add exactly `.worktrees/` to `.gitignore`, stage only `.gitignore`, run `git diff --cached --check`, and commit `chore: ignore local worktrees`.

- [ ] **Step 4: Create the implementation worktree**

Use the `superpowers:using-git-worktrees` skill and create `.worktrees/canonical-governance-release` on branch `feat/canonical-governance-release` from the planning commit. Confirm the implementation worktree starts clean. The primary checkout's `.DS_Store` remains untouched.

### Task 2: Repair and Pin Python Dependency Locks

**Files:**

- Create: `extension/backend/requirements-dev.in`
- Create: `extension/backend/requirements-dev.lock`
- Modify: `extension/backend/requirements.lock`

**Interfaces:**

- Produces: universal, hash-pinned runtime/dev environments.

- [ ] **Step 1: Add exact development inputs**

Create:

```text
-r requirements.txt
pytest==9.1.1
pytest-cov==7.1.0
pyright==1.1.411
pip-audit==2.10.1
cffconvert==2.0.0
```

- [ ] **Step 2: Compile both universal locks**

```bash
uv pip compile --universal --python-version 3.12 --generate-hashes \
  --output-file extension/backend/requirements.lock \
  extension/backend/requirements.txt
uv pip compile --universal --python-version 3.12 --generate-hashes \
  --output-file extension/backend/requirements-dev.lock \
  extension/backend/requirements-dev.in
```

Expected: runtime lock includes `openai`; headers show `--universal` and Python 3.12.

- [ ] **Step 3: Verify clean installs**

Create two `mktemp -d` locations, then:

```bash
python3.12 -m venv "$runtime_venv"
"$runtime_venv/bin/python" -m pip install --require-hashes -r extension/backend/requirements.lock
"$runtime_venv/bin/python" -m pip check
python3.12 -m venv "$dev_venv"
"$dev_venv/bin/python" -m pip install --require-hashes -r extension/backend/requirements-dev.lock
"$dev_venv/bin/python" -m pip check
```

Expected: both checks report no broken requirements.

- [ ] **Step 4: Audit runtime lock**

Run `"$dev_venv/bin/pip-audit" --require-hashes -r extension/backend/requirements.lock`.

Expected: no known vulnerabilities. If it fails, update the declared compatible dependency floor and recompile; never ignore an advisory silently.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/requirements.lock extension/backend/requirements-dev.in extension/backend/requirements-dev.lock
git commit -m "build: lock Python runtime and quality tools"
```

### Task 3: Make Coverage Scope Honest and Enforce 90 Percent

**Files:**

- Create: `extension/backend/.coveragerc`
- Modify: `.github/workflows/quality.yml` later through Task 8.

**Interfaces:**

- Produces: stable production-only coverage denominator.

- [ ] **Step 1: Write the coverage configuration**

Set `source = .`, branch coverage on, and omit tests, caches, intermediate/generated trees, `legacy_pre_emse`, runs, research scripts, local environments, and `__init__.py` files that contain exports only. The omit list must leave `core/`, `api/`, `configs/`, `main.py`, and `config.py` in scope. Configure `show_missing = True`, `skip_covered = True`, and `fail_under = 90`.

- [ ] **Step 2: Run the exact release coverage command**

```bash
cd extension/backend
python -m pytest -m "not integration" --cov --cov-config=.coveragerc --cov-report=term-missing --cov-report=xml
```

Expected: all tests pass and total is at least 90%. If new production modules lower coverage, add behavior-focused tests; do not omit maintained code merely to raise the number.

- [ ] **Step 3: Commit**

```bash
git add extension/backend/.coveragerc
git commit -m "test: enforce production coverage floor"
```

### Task 4: Update Node Dependencies and Pin the Extension Test Runtime

**Files:**

- Modify: `extension/package.json`
- Modify: `extension/package-lock.json`
- Modify: `extension/.vscode-test.mjs`

**Interfaces:**

- Produces: Axios `^1.19.0`, Mermaid CLI `11.16.0`, version `1.1.0`, Python default `python3.12`, and VS Code test version `1.106.1`.

- [ ] **Step 1: Update metadata and dependencies without forcing majors**

Use `npm install axios@^1.19.0` and `npm install --save-dev @mermaid-js/mermaid-cli@11.16.0`; ensure both package files record extension version `1.1.0`. Set the contributed `pythonPath` default to `python3.12` and the internal fallback in `extension.ts` to the same value.

- [ ] **Step 2: Pin the extension host**

Add `version: "1.106.1"` to `.vscode-test.mjs`, retain the existing isolated data/extension directories, and set a test timeout of 60 seconds.

- [ ] **Step 3: Run audits and static gates**

```bash
cd extension
npm ci
npm audit --omit=dev
npm audit
npm run compile
npm run lint
```

Expected: production audit reports zero. Full audit output is recorded; apply compatible non-breaking updates where available.

- [ ] **Step 4: Run actual tests**

On Linux run `xvfb-run -a npm test`; on macOS run `npm test`. Diagnose runner failures with `superpowers:systematic-debugging`; do not use executable-name symlinks or mark the suite optional.

- [ ] **Step 5: Commit**

```bash
git add extension/package.json extension/package-lock.json extension/.vscode-test.mjs extension/src/extension.ts
git commit -m "build: secure and pin extension runtime"
```

### Task 5: Add a Non-Mutating Backend Setup Check

**Files:**

- Create: `extension/src/backend/setupCheck.ts`
- Create: `extension/src/test/backend/setupCheck.test.ts`
- Modify: `extension/src/extension.ts`
- Modify: `extension/package.json`

**Interfaces:**

- Produces: `checkBackendSetup(...) -> Promise<BackendSetupReport>` and command `ddd-enforcer.checkBackendSetup`.

- [ ] **Step 1: Write process and result tests**

Assert Python 3.12 passes while 3.11/3.13 fail; runtime import probe covers `fastapi`, `uvicorn`, `google.genai`, `openai`, `chromadb`, `sentence_transformers`, `pydantic`, `dotenv`, `pypdf`, `docx`, `httpx`, and `numpy`; backend probe imports `main:app` without lifespan; credential probe returns presence only. Assert subprocess uses argument arrays, `shell: false`, never calls `pip`, never prompts, and output excludes credential contents.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension && npm test -- --grep "Backend Setup"`.

Expected: missing module/command.

- [ ] **Step 3: Implement and register the command**

Use `child_process.execFile` via an injectable wrapper. Check credential presence through SecretStorage, settings, or environment without reading it into report text. Display a concise pass/fail summary and send detailed safe items to the output channel. Do not call `getApiKey`.

- [ ] **Step 4: Run focused and static tests**

Run `cd extension && npm test -- --grep "Backend Setup" && npm run compile && npm run lint`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/backend/setupCheck.ts extension/src/test/backend/setupCheck.test.ts extension/src/extension.ts extension/package.json extension/package-lock.json
git commit -m "feat: add backend setup diagnostics"
```

### Task 6: Harden VSIX Contents and Package Smoke

**Files:**

- Modify: `extension/.vscodeignore`
- Create: `extension/scripts/assert-vsix-contents.mjs`

**Interfaces:**

- Produces: a VSIX under 10 MiB containing only runtime artifacts.

- [ ] **Step 1: Write package assertion script tests through fixture entry lists**

The script accepts a VSIX path, lists its zip entries, and exits nonzero for missing required or present forbidden paths. Required: `out/extension.js`, `backend/main.py`, `backend/config.py`, `backend/requirements.lock`, runtime `core/`, `api/`, and governance files. Forbidden: `.venv`, tests, runs, data, domain, inputs, `legacy_pre_emse`, backend scripts/research, intermediate trees, `out/test`, `out/t`, `out/e`, source TypeScript, maps, env files, coverage/caches, and dev locks.

- [ ] **Step 2: Tighten `.vscodeignore`**

Use repository-relative patterns only; remove hard-coded maintainer paths. Explicitly include the runtime lock and exclude local/untracked environments because `vsce` sees untracked files.

- [ ] **Step 3: Package and inspect**

```bash
cd extension
npx --no-install vsce package --out "$tmp/ddd-enforcer-1.1.0.vsix"
node scripts/assert-vsix-contents.mjs "$tmp/ddd-enforcer-1.1.0.vsix"
```

Expected: success, no forbidden entry, and archive size below 10 MiB. Do not commit the VSIX; leave the ignored historical local VSIX untouched unless the user separately authorizes deletion.

- [ ] **Step 4: Commit**

```bash
git add extension/.vscodeignore extension/scripts/assert-vsix-contents.mjs
git commit -m "build: harden extension package contents"
```

### Task 7: Update Release Metadata and Maintainer Documents

**Files:**

- Modify: `extension/CHANGELOG.md`
- Create: `SECURITY.md`
- Create: `CONTRIBUTING.md`
- Modify: `extension/README.md`

**Interfaces:**

- Produces: accurate `1.1.0` release entry, vulnerability policy, contribution workflow, and source-install guide.

- [ ] **Step 1: Replace stale extension documentation**

Document source-only installation, Python 3.12, locked dependency command, secure key storage, actual commands/artifacts, current limitations, and setup check. Remove Marketplace and Python 3.10 claims.

- [ ] **Step 2: Write the changelog**

Add `1.1.0` dated 2026-08-13 covering governance contracts/plans, shared observable generation, provenance/staleness, Context-Mapper/Critic/current UI, security/package/CI changes, and breaking local request contract (`workspace_root`, no output path). Retain `1.0.0` without emoji embellishment.

- [ ] **Step 3: Add security and contribution guidance**

`SECURITY.md` describes supported `1.1.x`, private reporting channel through GitHub Security Advisories, local-loopback trust boundary, external LLM disclosure, SecretStorage, no source execution, and no credentials in issues. `CONTRIBUTING.md` describes worktree/TDD expectations, exact locked setup, `make quality`, integration-test opt-in, docs/citation accuracy, and license scope.

- [ ] **Step 4: Run link/path audit and commit**

Use `rg` to ensure no `cd backend`, Python 3.10, Marketplace, or `domain_model.json` claim remains in these files.

```bash
git add extension/CHANGELOG.md extension/README.md SECURITY.md CONTRIBUTING.md
git commit -m "docs: prepare extension 1.1 release"
```

### Task 8: Consolidate CI Behind One Quality Command

**Files:**

- Create: `scripts/quality.sh`
- Create: `scripts/check-documentation.mjs`
- Create: `.github/workflows/quality.yml`
- Modify: `Makefile`
- Delete: `.github/workflows/backend-ci.yml`
- Delete: `.github/workflows/extension.ci.yml`

**Interfaces:**

- Produces: `make quality` and one matching GitHub Actions job.

- [ ] **Step 1: Implement fail-fast quality orchestration**

`scripts/quality.sh` uses `set -euo pipefail`, validates Python 3.12 and Node 20, creates a `mktemp -d` dev environment, installs `requirements-dev.lock` with hashes, runs `pip check`, the runtime audit, non-integration pytest/coverage, Pyright through the venv, compileall, `npm ci`, production and full audits, compile/lint, extension tests (`xvfb-run -a` when Linux), documentation/CFF checks, Mermaid temp render, and temporary VSIX/assertion. A trap removes only the exact temp directory. It prints that authenticated integration tests were deselected.

- [ ] **Step 2: Add deterministic documentation checks**

The Node script verifies local Markdown links/anchors and exact DOI/Xplore URLs without fetching the network, parses `CITATION.cff` via `cffconvert --validate`, checks Mermaid SVG XML has no script or remote href/src, and rejects known stale phrases/paths.

- [ ] **Step 3: Wire Make and CI**

Add `.PHONY: quality` and `quality: ; ./scripts/quality.sh`. The workflow checks out, sets Python 3.12 and Node 20, installs `uv` at a pinned release, installs Linux libraries plus `xvfb`, and invokes `make quality`. Use dependency caches keyed by both lockfiles. Remove the two weaker workflows only after the new commands are locally green.

- [ ] **Step 4: Run locally and commit**

Run `make quality` from repository root.

Expected: every gate passes; no credential-dependent test runs.

```bash
git add scripts Makefile .github/workflows
git commit -m "ci: enforce the complete release gate"
```

### Task 9: Create and Validate the Pipeline Figure

**Files:**

- Create: `docs/assets/ddd-enforcer-pipeline.mmd`
- Create: `docs/assets/ddd-enforcer-pipeline.svg`
- Create: `docs/assets/mermaid-config.json`

**Interfaces:**

- Produces: maintainable source and a self-contained README SVG.

- [ ] **Step 1: Use the diagram skill and write the four-region source**

The figure shows: domain generation; bounded Context-Mapper/Critic feedback; optional RAG traceability; on-save/manual Python validation. Include AST enrichment, provider boundary, `domain/model.json`, governance contracts, and read-only code/test plans. No plan has a write edge to source. Use “multi-stage pipeline,” “quality feedback and revision,” and “selected DomainModel”; never “eight-agent,” “self-healing,” or “correct model.”

- [ ] **Step 2: Validate before export**

```bash
cd extension
npx --no-install mmdc \
  -i ../docs/assets/ddd-enforcer-pipeline.mmd \
  -o "$tmp/pipeline-validation.svg" \
  -c ../docs/assets/mermaid-config.json \
  -b '#F8FAFC'
```

Expected: zero exit and parseable SVG.

- [ ] **Step 3: Export and visually inspect**

Run the same command to the committed SVG, then export a 2× temporary PNG. Inspect it with the image viewer for clipping, contrast, hierarchy, typo, and light/dark GitHub readability. Parse SVG XML and assert no scripts or remote URLs.

- [ ] **Step 4: Commit**

```bash
git add docs/assets/ddd-enforcer-pipeline.mmd docs/assets/ddd-enforcer-pipeline.svg docs/assets/mermaid-config.json
git commit -m "docs: add reproducible pipeline figure"
```

### Task 10: Add Verified Citation Metadata

**Files:**

- Create: `CITATION.cff`

**Interfaces:**

- Produces: software `1.1.0` citation with conference paper as preferred citation.

- [ ] **Step 1: Write the CFF**

Use CFF 1.2.0, title `DDD-Enforcer`, type `software`, release date `2026-08-13`, repository URL, and the three authors in official order. Do not put the paper DOI on the software entry and omit a repository-wide `license` key. Preferred citation is an `article`/conference-paper record with title, exact author order, conference name, 2026, pages 746–751, and DOI `10.1109/IISEC69317.2026.11418529`.

- [ ] **Step 2: Validate**

Run `cffconvert --validate -i CITATION.cff` and `cffconvert -f bibtex -i CITATION.cff`.

Expected: validation success and BibTeX containing the paper DOI/title/authors.

- [ ] **Step 3: Commit**

```bash
git add CITATION.cff
git commit -m "docs: add verified software citation"
```

### Task 11: Rewrite the Root README for Research and Engineering Impact

**Files:**

- Rename: `readme.md` → `README.md` with a two-step temporary rename on case-insensitive macOS.
- Modify: `README.md`

**Interfaces:**

- Consumes: final commands, artifacts, quality evidence, citation, figure, and verified publication data.

- [ ] **Step 1: Perform the case-safe rename**

```bash
git mv readme.md README.tmp
git mv README.tmp README.md
```

- [ ] **Step 2: Draft all thirteen approved sections**

Write professional English: hero; publication; research-to-engineering distinction; what it does; architecture SVG; capabilities; exact source quick start; usage; conference-version evaluation table; scope/limitations; verification/reproducibility; project structure; authors/citation/contribution/license. Link the DOI, IEEE Xplore record, `CITATION.cff`, `SECURITY.md`, `CONTRIBUTING.md`, quality workflow, and `extension/LICENSE`.

Use the exact publication record and metrics in the design spec. State explicitly that the conference evaluation covered the published system and that current `main` subsequently added Verifier/Refiner, Context-Mapper, Holistic Critic, provenance, governance, and expanded observability.

- [ ] **Step 3: Run the Humanizer file workflow**

Use `humanizer` in file mode: draft → audit → final. Preserve exact technical identifiers, commands, author spelling `Ahmet Baran Dinçoğuz`, DOI, titles, figures, and scoped claims. Only the final humanized copy remains in `README.md`.

- [ ] **Step 4: Validate commands, local links, and forbidden claims**

Run `node scripts/check-documentation.mjs`. Also search for Marketplace, production-ready, fully local, automatic fixing, instant, every language, self-healing, `cd backend`, `domain_model.json`, Python 3.10, and whole-repository MIT claims.

- [ ] **Step 5: Commit**

```bash
git add README.md readme.md
git commit -m "docs: present DDD Enforcer research and platform"
```

### Task 12: Final Release Verification and Independent Review

**Files:** No planned implementation changes.

- [ ] **Step 1: Run the complete quality gate from a clean worktree**

Run `make quality` and save the exit/status summary. Expected: backend all pass ≥90%; Pyright zero; compileall zero; TypeScript/lint/tests pass; both production audits clean; CFF/docs/Mermaid valid; VSIX passes content/size.

- [ ] **Step 2: Verify branch/stash/user-state invariants**

Assert archival tag resolves to the exact governance stash SHA; stash still exists; no historical branch was merged wholesale; `.DS_Store` is absent from the implementation diff; no VSIX/temp/venv artifact is tracked.

- [ ] **Step 3: Request independent code review**

Use `superpowers:requesting-code-review` with the design spec, all four plans, final diff, and test evidence. Resolve every actionable correctness/security/maintainability/documentation finding using `superpowers:receiving-code-review` and re-run affected gates.

- [ ] **Step 4: Re-run the complete quality gate after review fixes**

Run `make quality` again from fresh state. Expected: identical pass conditions.

- [ ] **Step 5: Use the finishing-development-branch workflow**

Invoke `superpowers:finishing-a-development-branch`. The user's requested destination is `main`, but present the skill's integration choices with current verification evidence before performing the final local merge/push decision. Do not delete historical branches or stashes as part of this release.
