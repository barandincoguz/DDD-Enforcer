# DDD-Enforcer Canonical Main, Governance, Release, and README Design

**Date:** 2026-08-13

**Status:** Approved; implementation plans prepared

**Canonical baseline:** `main` at `87f677b3d2a6e5529acbb7dd1e41d5cf84e48b6c`

**Process:** specification review → implementation plan → isolated worktree → TDD implementation → independent review → merge decision

## 1. Outcome

DDD-Enforcer will keep the current `main` history as its canonical product line. Historical branches will not be merged wholesale. The useful governance concept found only in an incomplete `ImprovingGeneral` stash will instead be rebuilt against the current modular architecture.

The release will add three read-only developer-governance capabilities:

1. typed validation contracts derived from the current domain model and its SRS evidence;
2. a bounded-context-aware code-generation plan;
3. a bounded-context-aware test-generation plan.

These features may write `<workspace>/domain/governance.json`, but they must never create, edit, delete, or execute workspace source code. The release also consolidates domain-model generation behind one observable implementation, closes dependency and packaging gaps, and presents the project with a publication-accurate, CV-quality English README and a reproducible pipeline figure.

## 2. Evidence and branch decision

### 2.1 Canonical branch

`main` is the only suitable base:

- it contains the latest Scout → Architect → Specialist → Verifier/Refiner → Synthesizer pipeline;
- it includes the Context-Mapper, bounded Holistic Critic loop, AST enrichment, RAG traceability, provider registry, run manifests, secure VS Code integration, and the broadest test suite;
- `feat/EnhancedDocumentParserModule`, `feature/ImprovingAccuracy`, and `feature/rag` are strict ancestors;
- `feature/ExtensionizingBundle` is a stale, divergent implementation whose useful behavior is already superseded on `main`;
- `feature/ImprovingGeneral` contains two unique commits but is not release-quality: one is `.DS_Store`-only, while the other is a large experimental rewrite with tracked `node_modules` and a failing backend suite.

Therefore no historical feature branch will be merged or cherry-picked wholesale. Any idea retained from history must be reimplemented in small, tested modules on top of `main`.

### 2.2 Stash preservation

The unique governance experiment is the stash commit:

```text
d6a1a22cec321a261ca32bb5d4b59c9f54e28204
```

It contains useful contract and VS Code command concepts but also imports a missing `GovernanceService` and is not directly applicable. Before implementation begins, preserve it with a verified annotated archival ref such as `archive/governance-stash-2026-03-13`. Do not apply it wholesale. No branch or stash may be deleted without explicit user confirmation, and no cleanup may rely on a mutable `stash@{n}` ordinal without first resolving its SHA.

The existing user modification to `.DS_Store` is out of scope and must remain untouched.

## 3. Product boundaries

### In scope

- Validation contracts grounded in the active domain model and SRS evidence.
- Code and test plans returned as structured data and rendered as readable Markdown/cards.
- Domain-model fingerprinting and explicit stale-governance detection.
- Workspace-confined path handling.
- Three FastAPI endpoints and three VS Code commands.
- A secure webview presentation with evidence and copy actions.
- One shared, observable generation workflow for synchronous and streaming routes.
- Dependency lock, audit, CI, VSIX packaging, metadata, documentation, citation, and release polish.
- A validated Mermaid source plus exported SVG pipeline figure.

### Out of scope

- Automatic source-code generation or mutation.
- Automatic test-file generation or mutation.
- Executing generated commands, code, or tests on the user's behalf.
- Replacing the current domain-model schema with the stale `3.0.0` experiment.
- Merging historical branches or applying historical stashes wholesale.
- Reworking the paper draft or claiming the unfinished EMSE draft is published.
- Multi-language validation; the current validator remains Python-specific.
- Publishing to VS Code Marketplace or pushing Git refs without a separate release decision.

## 4. User workflow

The domain model remains the root artifact:

```text
SRS documents
    → Generate Domain Model
    → <workspace>/domain/model.json
    → Generate Governance Contracts
    → <workspace>/domain/governance.json
    → Generate Code Plan / Generate Test Plan
    → inspect and copy the result; no source file is changed
```

Rules:

1. Governance generation requires a valid current `domain/model.json`.
2. Governance is generated explicitly, never as a hidden side effect of model generation.
3. Plans require a current governance artifact whose domain fingerprint matches the active model.
4. A materially changed domain model makes governance stale. Stale governance is rejected, not silently reused.
5. A code or test plan is returned to the UI only. The user decides whether and how to apply it.

## 5. Governance data model

### 5.1 Storage

The sole new persisted artifact is:

```text
<workspace>/domain/governance.json
```

It is written atomically using a temporary sibling file, flush, `fsync`, and `os.replace`. A failed generation or write must leave the previous valid file intact. Code and test plans are not persisted automatically.

### 5.2 Domain fingerprint

The fingerprint is a SHA-256 digest of a canonical JSON projection of the active `DomainModel`:

- include `project_name`, semantic project version/description, ordered source-document paths and hashes, bounded contexts, global rules, and the strategic context map;
- exclude volatile generation timestamps and `critic_report` telemetry;
- serialize dictionaries with sorted keys and deterministic separators;
- preserve list order because it is part of the emitted typed model;
- compute and compare through one pure function used by both creation and loading.

This definition prevents a timestamp-only regeneration from invalidating governance while still rejecting any change to domain structure or rules.

### 5.3 Persisted schema

`GovernanceDocument` contains:

- `schema_version`;
- `project_name`;
- `domain_model_fingerprint`;
- `generated_at` in UTC;
- `pipeline_version` and model/provider provenance;
- `source_references` for the domain model, SRS documents, and available AST snapshot;
- `validation_contracts`;
- `grounding_warnings`.

Each `ValidationContract` contains:

- stable `rule_id`;
- human-readable `title`;
- bounded-context or project `scope`;
- `severity` (`error`, `warning`, or `information`);
- `rationale`;
- explicit `checks`;
- `evidence` entries linking to relative SRS paths and sentence indices;
- affected domain concepts;
- optional implementation notes.

Unknown severity, empty checks, invalid context names, duplicate rule IDs, or malformed evidence fail Pydantic validation. Evidence index `-1` is allowed only when labeled as inference-only and must generate a visible grounding warning.

### 5.4 Plan schemas

`CodeGenerationPlan` contains:

- project and bounded-context identity;
- requested objective;
- ordered implementation steps;
- domain constraints and forbidden dependencies;
- affected concepts and suggested relative paths;
- acceptance criteria;
- applicable governance rule IDs;
- SRS evidence;
- grounding warnings.

`TestGenerationPlan` contains:

- project and bounded-context identity;
- requested objective and optional workspace-relative Python target;
- unit and integration scenarios;
- preconditions, fixtures, actions, and assertions;
- negative and boundary cases;
- applicable governance rule IDs;
- traceability to domain concepts and SRS evidence;
- grounding warnings.

Suggested paths are advisory strings and must be workspace-relative. A plan cannot claim that a file exists unless the validated workspace scan confirms it.

## 6. Backend architecture

### 6.1 Governance package

Add a focused package:

```text
extension/backend/core/governance/
├── __init__.py       # public facade only
├── schemas.py        # persisted and response Pydantic contracts
├── fingerprint.py    # pure canonical projection and SHA-256 logic
├── prompts.py        # grounded prompts for contracts and plans
├── service.py        # orchestration and typed structured-output calls
├── storage.py        # safe load and atomic governance write
├── workspace.py      # confinement, relative-path, and extension validation
└── errors.py         # typed domain errors
```

The service depends on the existing LLM provider abstraction and model registry. It does not instantiate a provider-specific SDK directly. Contract, code-plan, and test-plan operations each request structured output and validate the result before returning it. JSON parse exhaustion, missing credentials, and provider failures are explicit failures; there is no untyped text fallback.

### 6.2 API contracts

Add these routes:

```text
POST /governance/contracts
POST /generation/code-plan
POST /generation/test-plan
```

Common request fields:

- `workspace_root`: absolute path supplied by the trusted local extension and required to match the backend process's configured workspace after resolution;
- `bounded_context`: required for plan endpoints;
- `objective`: required, non-blank, bounded-length text for plan endpoints;
- `target_relative_path`: optional for a test plan, restricted to an existing `.py` file below the workspace root.

The server derives `domain/model.json` and `domain/governance.json` from the validated workspace root. The API accepts no arbitrary output path. Source references emitted in responses are workspace-relative.

Responses use typed bodies and meaningful HTTP status codes:

| Condition | Status |
|---|---:|
| Invalid request shape | `422` |
| Workspace path outside the allowed root or unsupported target | `403` |
| Missing domain model or governance artifact | `404` |
| Stale governance fingerprint or unknown bounded context | `409` |
| Missing or changed SRS provenance in the current domain model | `409` |
| Missing provider credential | `401` |
| Provider transport or structured-output exhaustion | `502` |
| Atomic write failure or unexpected internal error | `500` |
| Success | `200` |

Returned messages provide actionable context without exposing API keys, full prompts, absolute filesystem paths, or raw provider responses.

### 6.3 Shared model-generation implementation

The current `/generate-model` and `/generate-model-stream` routes duplicate ingestion, generation, AST enrichment, persistence, RAG initialization, and error conversion. The existing observable `_run_generate_pipeline` helper is exercised by tests but not by those production routes.

Add an optional, backward-compatible `ProjectMetadata.source_documents` list. Each entry contains a workspace-relative path and SHA-256 captured from the exact ordered SRS batch used for generation. The shared generation service populates it before persisting `domain/model.json`. Governance refuses an older model with no source provenance, or a model whose recorded source hash no longer matches the current file, and instructs the user to regenerate the domain model; it must not guess provenance by scanning the workspace.

Extract one focused generation service that owns the complete lifecycle:

```text
validate workspace paths
→ parse SRS batch
→ run DomainArchitect under RunManifest/StageEmitter
→ enrich from workspace AST
→ atomically persist domain/model.json
→ initialize RAG from the already parsed documents
→ update application state
→ return a typed GenerationResult
```

The synchronous endpoint calls this service directly. The streaming endpoint runs the same service in its worker thread and forwards the optional progress callback as SSE. Route wrappers are responsible only for transport serialization. Both paths therefore share outcomes, typed errors, persistence behavior, and run-manifest coverage.

The refactor must preserve existing successful response fields and SSE event shapes used by the extension. It must not silently swallow RAG failure: if RAG remains an optional post-generation capability, the result and manifest record an explicit degraded status and warning.

### 6.4 Workspace confinement

For every path-sensitive operation:

1. require an absolute workspace root and reject filesystem-root workspaces for these endpoints;
2. resolve symlinks and normalize the path;
3. require the resolved root to equal the backend process's configured `WORKSPACE_PATH` rather than trusting an arbitrary caller-selected root;
4. reject targets whose resolved path is not that root or a descendant;
5. derive writable artifact paths server-side;
6. allow SRS input only for configured PDF, DOCX, and TXT extensions;
7. allow an optional test target only when it is an existing `.py` file;
8. reject paths with a typed error before file access.

This boundary applies to model generation as well as governance routes, replacing the unrestricted `file_paths`/`output_path` behavior without changing the normal extension workflow.

## 7. VS Code architecture and experience

Add a focused extension package:

```text
extension/src/governance/
├── types.ts       # API request/response contracts
├── client.ts      # transport and typed error mapping
├── commands.ts    # command registration and workflow control
├── markdown.ts    # readable plan/contract projection
└── webview.ts     # secure presentation
```

Contribute three commands:

- `DDD Enforcer: Generate Governance Contracts`
- `DDD Enforcer: Generate Code Plan`
- `DDD Enforcer: Generate Test Plan`

Add `DDD Enforcer: Check Backend Setup` as a separate release-readiness command. It checks the configured Python version, locked runtime imports, backend launchability, and credential presence without installing packages or exposing the credential.

The commands first validate that a workspace and domain model exist. Plan commands prompt for a bounded context and a concise objective. The test-plan command may use the active Python file as an optional target after showing the relative path to the user.

The webview is designed for developer tasks, not for raw JSON. It displays:

- concise project and fingerprint status;
- validation-contract cards grouped by scope and severity;
- ordered plan steps and acceptance criteria;
- evidence chips that can open an existing workspace-relative source;
- grounding warnings in a distinct, non-dismissive state;
- copy-as-Markdown and copy-as-JSON actions.

Use a cryptographic nonce, a restrictive content-security policy, escaped text, command allow-listing, and validated message payloads. Provider keys remain in VS Code `SecretStorage`. Short user-facing errors appear in notifications; detailed, sanitized context goes to the DDD Enforcer output channel.

## 8. Error and failure model

There are no silent defaults, permissive retries, or invented context.

- Missing model: stop and tell the user to initialize it.
- Stale governance: stop and tell the user to regenerate contracts.
- Unknown context: return the current valid context names.
- Invalid workspace path: stop before reading or writing.
- Invalid LLM schema: exhaust the existing bounded structured-output retry policy, then fail with a typed provider/schema error.
- Missing evidence: retain only explicitly inference-labeled material and show a grounding warning.
- Atomic write failure: keep the previous artifact and report failure.
- Transport failure: preserve the typed backend message in the output channel.
- Generation failure: finalize its run manifest with the correct outcome and HTTP/SSE error representation.

No governance or planning error may fall back to source modification, generic prose, an old governance file, or a different bounded context.

## 9. README, publication, and visual design

### 9.1 Editorial direction

The root README will be rewritten in professional English for international reviewers, recruiters, researchers, and engineers. Its tone is precise and confident without emoji-heavy promotion, fake social proof, or unsupported superlatives. A final humanization pass will remove formulaic AI-writing patterns without weakening technical or academic precision.

The visual identity uses restrained navy, cyan, and neutral tones. Only real, maintained badges are shown. Do not claim Marketplace availability, production readiness, full locality, instant validation, automatic fixing, or support for every language/project.

### 9.2 README structure

1. Hero: title, one-sentence value proposition, verified build/runtime/license/publication links.
2. Publication: exact paper title, official author order, venue, date/location, pages, DOI, and IEEE Xplore link.
3. From research to engineering: distinguish the evaluated conference system from the expanded current `main` architecture.
4. What it does: requirements → typed model → governance → on-save Python diagnostics.
5. Architecture: exported pipeline SVG with a compact text explanation.
6. Key capabilities: generation, strategic mapping, critique/revision, governance, validation, traceability, observability.
7. Quick start: exact Python 3.12 and Node installation commands using the lockfile.
8. Usage: actual commands and actual artifact paths.
9. Published evaluation: conference-version metrics, explicitly scoped to that version.
10. Supported scope and limitations.
11. Verification and reproducibility.
12. Project structure.
13. Authors, citation, contribution, and license scope.

### 9.3 Publication record

Use the verified record:

- **Title:** *DDD-Enforcer: An AI-Powered Multi-Agent System for Real-Time Domain-Driven Design Enforcement*
- **Authors:** Ahmet Baran Dinçoğuz, Ali Kendir, Murat Karakaya
- **Venue:** 5th International Conference on Informatics and Software Engineering (IISEC 2026)
- **Location and dates:** Ankara, Türkiye, 5–6 February 2026
- **Pages:** 746–751
- **DOI:** `10.1109/IISEC69317.2026.11418529`
- **IEEE Xplore document:** `https://ieeexplore.ieee.org/document/11418529`

The publication card may identify Ahmet Baran Dinçoğuz as first author, matching the official author order, while preserving the complete author list with equal typographic care.

Add `CITATION.cff` with a software citation and this paper as `preferred-citation`, plus a copyable BibTeX block. The unfinished EMSE manuscript must be labeled as a draft if mentioned at all and must never be presented as the published work.

Published metrics may be displayed only under a label such as “Conference evaluation (published system)”:

- 100% detection across 15 evaluated cases;
- 4.49 s average validation latency;
- RAG Top-1 accuracy of 76.8%;
- RAG Top-3 accuracy of 88.8%;
- 140.52 ms average RAG retrieval latency.

The README must state that the current repository has since added the Verifier/Refiner, Context-Mapper, Holistic Critic, and expanded observability, so those published figures are not claimed as a benchmark of every current component.

### 9.4 Pipeline figure

Maintain both:

```text
docs/assets/ddd-enforcer-pipeline.mmd
docs/assets/ddd-enforcer-pipeline.svg
```

The Mermaid source uses a top-to-bottom flowchart with four clearly separated regions:

1. domain-model generation;
2. bounded quality feedback and revision;
3. SRS traceability index;
4. on-save/manual Python validation.

It must show the configured LLM-provider boundary (external Gemini by default, with the implemented Ollama-compatible option), optional RAG traceability, default-on but bounded Context-Mapper/Critic behavior, workspace AST enrichment, and the final `domain/model.json`. Governance contracts and read-only plans branch from that selected model and never point back to source files with a write edge.

Use “multi-stage pipeline,” “quality feedback and revision,” and “selected DomainModel.” Avoid “eight-agent,” “self-healing,” and “verified/correct model.” Verifier, Refiner, AST enrichment, and RAG must not be mislabeled as autonomous LLM agents.

The `.mmd` file must validate before export. The SVG is generated from the validated source, inspected for clipping/readability, contains no remote runtime dependency, and remains legible on GitHub light and dark backgrounds.

### 9.5 License scope

Do not broaden licensing by implication. The README will state that the software under `extension/` is licensed under its existing MIT license. Manuscripts, published papers, bundled SRS material, and third-party resources are not covered by that software license unless their own files say otherwise. Do not add a whole-repository MIT badge or root license that could accidentally relicense research artifacts.

## 10. Release engineering

### 10.1 Reproducible Python environment

- Regenerate `extension/backend/requirements.lock` from the declared Python 3.12 requirements with `uv pip compile --universal --python-version 3.12 --generate-hashes` so it includes every runtime dependency, including `openai`, without silently reflecting only the maintainer's host platform.
- Add `requirements-dev.in` (including `-r requirements.txt` plus `pytest`, `pytest-cov`, `pyright`, and the selected audit tool) and compile it to a separate universal, hash-pinned `requirements-dev.lock`; CI must not install floating test tools ad hoc.
- Use the runtime hash-pinned lock consistently in documented user installation and the development lock in backend CI.
- Verify a clean virtual environment can install only from the lock and passes `pip check`.
- Add a Python dependency advisory scan to CI using one explicitly pinned audit tool.

### 10.2 Node dependency and package hygiene

- Update `axios` and its lockfile-resolved transitive dependencies until `npm audit --omit=dev` reports zero vulnerabilities.
- Add the production dependency audit to CI.
- Keep dev-only audit findings visible and address compatible updates without forcing unsafe major rewrites.
- Tighten `.vscodeignore` so the VSIX excludes legacy experiments, runs, research scripts/data, caches, local environments, tests, and unrelated manuscripts while retaining the runtime backend and its intentional lockfile.
- Package to a temporary path and inspect the VSIX file list and size. Do not commit a generated VSIX.

### 10.3 CI and one-command quality gate

Provide one repository-level quality command that composes the same checks used by CI without embedding credentials. CI must include:

- clean hash-locked Python install and `pip check`;
- backend non-integration tests with at least 90% coverage measured over maintained production Python modules; tests, generated artifacts, caches, historical experiments, and local environments are excluded from the denominator through a committed coverage configuration;
- Pyright and Python bytecode compilation;
- TypeScript compilation and ESLint;
- actual extension tests with VS Code `1.106.1` pinned in `.vscode-test.mjs` and launched under `xvfb-run` on Linux CI;
- production npm audit;
- pinned Python advisory scan;
- `vsce package` smoke and package-content assertions.

Authenticated/live backend integration tests remain explicit opt-in release checks because they require an external credential. Their deselection must be visible rather than implied as passing.

### 10.4 Runtime setup and metadata

- Provide a clear setup/check command in the extension rather than silently assuming backend dependencies exist.
- Keep API-key migration and `SecretStorage` behavior.
- Reconcile Python version and executable defaults across package metadata and docs.
- Bump the extension from `1.0.0` to `1.1.0` for the backward-compatible governance feature release and replace the “initial release” changelog with an accurate entry.
- Add `CONTRIBUTING.md` and `SECURITY.md` with realistic local-service and vulnerability-reporting guidance.

## 11. Testing strategy

All behavioral work follows red–green–refactor TDD. Tests are grouped by boundary rather than by implementation detail.

### Backend

- Governance schema accepts valid contracts/plans and rejects invalid severities, duplicate IDs, empty checks, unknown contexts, absolute suggested paths, and malformed evidence.
- Fingerprint is deterministic, ignores only documented volatile fields, and changes for semantic model/rule/context-map changes.
- Storage detects missing/stale/malformed governance and preserves the old file on simulated atomic-write failure.
- Workspace confinement covers sibling-prefix attacks, `..`, symlinks, unsupported SRS files, and invalid Python targets.
- Service prompts contain only validated model/SRS/AST context and never invent unavailable files or evidence.
- Contract, code-plan, and test-plan services fail explicitly on provider and structured-output errors.
- Endpoint tests assert success bodies and every mapped HTTP error class.
- Model-generation endpoint tests prove sync and SSE routes call the same service and both finalize manifests for pre-pipeline and pipeline failures.
- Tests prove no governance/plan endpoint mutates `.py` files or writes outside `domain/governance.json`.

### Extension

- Command registration, workspace/model preconditions, context/objective prompts, cancellation, and typed error display.
- API request paths are workspace-confined and use the active Python file only as an explicit relative test target.
- Webview escaping, CSP/nonce, message allow-list, evidence-link validation, and copy rendering.
- Regression tests for existing initialization, validation, status, lifecycle, secrets, diagnostics, and run-manifest behavior.

### Documentation and release

- Mermaid source validation and SVG export.
- README command/path/link audit.
- `CITATION.cff` validation and DOI/IEEE links.
- clean lockfile install, audit, build, test, and VSIX content smoke.

## 12. Final acceptance gates

The work is complete only when fresh output demonstrates all of the following:

1. `main` remains the canonical base; no historical branch/stash was merged wholesale.
2. The unique governance stash SHA resolves through its verified archival ref.
3. Backend non-integration tests all pass with at least 90% aggregate coverage.
4. Pyright reports zero errors and Python `compileall` succeeds.
5. TypeScript compilation, ESLint, and the actual VS Code extension test suite pass.
6. A clean Python 3.12 environment installs from `requirements.lock` and `pip check` succeeds.
7. `npm audit --omit=dev` reports zero vulnerabilities and the pinned Python advisory scan passes.
8. VSIX packaging succeeds and content assertions show no excluded research/legacy/runtime-output trees.
9. Sync and streaming generation routes share one tested, manifest-producing implementation.
10. Governance contracts and plans are typed, stale-aware, workspace-confined, and demonstrably do not modify source.
11. The pipeline `.mmd` validates, its SVG renders cleanly, and the README accurately distinguishes published evidence from current extensions.
12. An independent sub-agent reviews the final diff; actionable findings are resolved or explicitly documented.
13. The user's pre-existing `.DS_Store` modification remains untouched.

Only after these gates pass may the implementation branch be offered for integration into `main`. Historical branch and stash deletion remains a separate, explicitly confirmed destructive action.

## 13. Rejected alternatives

### Merge `feature/ImprovingGeneral`

Rejected because its unique implementation is incomplete, has a missing imported service, tracks thousands of dependency files, and fails its clean backend test run. Resolving its conflicts would regress the current architecture rather than advance it.

### Cherry-pick the governance stash

Rejected because the stash was authored against an obsolete schema and service layout. Its product idea is retained, but the implementation must follow current boundaries and tests.

### Generate source files automatically

Rejected because it materially expands risk and contradicts the approved read-only planning workflow. The tool provides governed plans; humans or separate tools apply them.

### Embed only a Mermaid code block in the README

Rejected as the sole visual because exported SVG gives deterministic presentation and can be visually inspected. The `.mmd` source remains available for maintainability.

### Hide missing context with fallbacks

Rejected because a plan grounded in the wrong model or invented evidence is worse than an explicit failure. Missing, stale, or malformed inputs stop the operation with an actionable error.
