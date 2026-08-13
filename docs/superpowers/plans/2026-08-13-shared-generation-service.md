# Shared Generation Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the duplicated synchronous/streaming model-generation paths with one workspace-confined, atomic, fully observable service that records exact SRS provenance.

**Architecture:** A shared `core.generation` service owns ingestion through RAG indexing. FastAPI routes remain thin transport adapters, while a reusable `core.workspace` boundary validates every caller-supplied path against the backend's configured workspace. The current response and SSE shapes stay compatible, but arbitrary output paths disappear and every run finalizes a manifest.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, pathlib, pytest, VS Code TypeScript client

## Global Constraints

- Baseline is `main`; never merge or cherry-pick a historical feature branch or stash wholesale.
- Do not touch the user's pre-existing `.DS_Store` modification.
- SRS inputs must resolve below the configured `WORKSPACE_PATH` and use `.pdf`, `.docx`, or `.txt`.
- The server derives `<workspace>/domain/model.json`; requests contain no arbitrary output path.
- Additive model metadata must remain compatible with existing `model.json` files.
- Both `/generate-model` and `/generate-model-stream` must call the same production service.
- RAG failure may degrade a successful model generation only when the warning is explicit in both result and manifest.
- No silent exception swallowing, unbounded retry, or raw absolute path/provider response in API errors.
- All behavioral changes use red–green–refactor TDD.

---

## File Structure

### Create

- `extension/backend/core/workspace.py` — shared workspace-root and file confinement.
- `extension/backend/core/io_atomic.py` — harden the existing shared atomic UTF-8 persistence helper.
- `extension/backend/core/generation/__init__.py` — public generation facade.
- `extension/backend/core/generation/errors.py` — typed generation/path/persistence errors.
- `extension/backend/core/generation/types.py` — command, warning, and result contracts.
- `extension/backend/core/generation/service.py` — the complete model-generation lifecycle.
- `extension/backend/tests/test_workspace_scope.py` — traversal/symlink/extension/root tests.
- `extension/backend/tests/test_io_atomic.py` — unique-temp, replacement, cleanup, and failure-preservation tests.
- `extension/backend/tests/test_generation_service.py` — service lifecycle, provenance, manifest, and RAG tests.
- `extension/backend/tests/test_generation_routes.py` — sync/SSE shared-service and error-shape tests.

### Modify

- `extension/backend/core/schemas.py` — add source-document provenance to `ProjectMetadata`.
- `extension/backend/core/observability/emitter.py` — public emitter binding context manager.
- `extension/backend/core/observability/run_manifest.py` — warnings and post-generation stage support.
- `extension/backend/core/observability/__init__.py` — export the binding helper.
- `extension/backend/main.py` — thin request/response/SSE adapters and application-state update.
- `extension/backend/tests/test_main_wiring.py` — move duplication-specific expectations to service expectations.
- `extension/backend/tests/test_main_pipeline_error_endpoint.py` — preserve typed error siblings under the new service.
- `extension/backend/tests/test_pipeline_observability_e2e.py` — assert the production routes create manifests.
- `extension/src/types.ts` — add generation warning and typed SSE error fields.
- `extension/src/extension.ts` — send `workspace_root`, remove `output_path`, and reject external SRS selections before transport.
- `extension/src/test/extension.test.ts` — generation request and selection-boundary regression tests.

## Interfaces

```python
# core/schemas.py
class SourceDocumentReference(BaseModel):
    relative_path: str
    sha256: str

class ProjectMetadata(BaseModel):
    version: str
    generated_at: str
    description: Optional[str] = None
    source_documents: Optional[List[SourceDocumentReference]] = None

# core/workspace.py
@dataclass(frozen=True)
class WorkspaceScope:
    root: Path
    model_path: Path
    governance_path: Path

# Implement these exact public call signatures in core/workspace.py:
# resolve_workspace_scope(requested_root: str, configured_root: str | None = None) -> WorkspaceScope
# resolve_workspace_file(scope: WorkspaceScope, candidate: str, *, allowed_suffixes: Collection[str], must_exist: bool = True) -> Path
# workspace_relative(scope: WorkspaceScope, path: Path) -> str
# sha256_file(path: Path) -> str

# core/generation/types.py
ProgressCallback = Callable[[Dict[str, Any]], None]

@dataclass(frozen=True)
class GenerationCommand:
    workspace: WorkspaceScope
    srs_paths: tuple[Path, ...]

@dataclass(frozen=True)
class GenerationWarning:
    code: str
    message: str

@dataclass
class GenerationResult:
    model: DomainModel
    model_path: Path
    rag: RAGPipeline | None
    metrics: Dict[str, Any]
    warnings: list[GenerationWarning]

def run_generation(command: GenerationCommand, *, progress_callback: ProgressCallback | None = None) -> GenerationResult
```

The declarations above document signatures; implementation steps below define their bodies.

---

### Task 1: Add Exact SRS Provenance to the Domain Model

**Files:**

- Modify: `extension/backend/core/schemas.py:360-376`
- Test: `extension/backend/tests/test_schemas_strict.py`

**Interfaces:**

- Produces: `SourceDocumentReference(relative_path, sha256)` and backward-compatible `ProjectMetadata.source_documents`.
- Consumes: existing `ProjectMetadata` serialization and `DomainModel` Pydantic validation.

- [ ] **Step 1: Write failing provenance schema tests**

Add tests that assert an old metadata object defaults to `None`, a valid reference round-trips, an absolute path is rejected, `..` segments are rejected, and SHA-256 must match 64 lowercase hexadecimal characters:

```python
def test_project_metadata_source_documents_are_backward_compatible():
    meta = ProjectMetadata(version="1.0", generated_at="2026-08-13T00:00:00Z")
    assert meta.source_documents is None


@pytest.mark.parametrize("path", ["/tmp/srs.pdf", "../srs.pdf", "docs/../srs.pdf"])
def test_source_document_reference_rejects_unsafe_relative_path(path):
    with pytest.raises(ValidationError):
        SourceDocumentReference(relative_path=path, sha256="a" * 64)


def test_source_document_reference_round_trips():
    ref = SourceDocumentReference(relative_path="inputs/spec.pdf", sha256="a" * 64)
    assert SourceDocumentReference.model_validate_json(ref.model_dump_json()) == ref
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
cd extension/backend
python -m pytest tests/test_schemas_strict.py -q
```

Expected: failures because `SourceDocumentReference` and `source_documents` do not exist.

- [ ] **Step 3: Implement the additive schema**

Add a validator that uses `PurePosixPath`, rejects absolute/empty/dot-dot paths, normalizes backslashes to `/`, and validates the digest with `^[0-9a-f]{64}$`. Keep `source_documents` optional with a `None` default so historical model files still load while governance can distinguish legacy models.

- [ ] **Step 4: Run schema and model regression tests**

Run:

```bash
cd extension/backend
python -m pytest tests/test_schemas_strict.py tests/test_ast_model_signals_enrichment.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/schemas.py extension/backend/tests/test_schemas_strict.py
git commit -m "feat: record domain model source provenance"
```

### Task 2: Build the Shared Workspace Boundary

**Files:**

- Create: `extension/backend/core/workspace.py`
- Create: `extension/backend/tests/test_workspace_scope.py`

**Interfaces:**

- Consumes: process `WORKSPACE_PATH` when `configured_root` is omitted.
- Produces: `WorkspaceScope`, `resolve_workspace_scope`, `resolve_workspace_file`, `workspace_relative`, and `sha256_file`.

- [ ] **Step 1: Write path-confinement tests**

Cover matching roots, missing configuration, relative requested roots, filesystem root, sibling-prefix attacks (`/work/app-evil`), `..`, symlink escape, a symlinked existing `domain/` ancestor that escapes the workspace, missing file, unsupported extension, mixed-case extension, relative conversion, and stable SHA-256:

```python
def test_resolve_workspace_file_rejects_symlink_escape(tmp_path):
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret")
    (root / "link.txt").symlink_to(secret)
    scope = resolve_workspace_scope(str(root), str(root))
    with pytest.raises(WorkspacePathError):
        resolve_workspace_file(scope, str(root / "link.txt"), allowed_suffixes={".txt"})


def test_requested_workspace_must_equal_configured_workspace(tmp_path):
    configured = tmp_path / "workspace"
    sibling = tmp_path / "workspace-other"
    configured.mkdir()
    sibling.mkdir()
    with pytest.raises(WorkspaceConfigurationError):
        resolve_workspace_scope(str(sibling), str(configured))
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_workspace_scope.py -q`.

Expected: import failure because `core.workspace` does not exist.

- [ ] **Step 3: Implement explicit resolution**

Implement `resolve_workspace_scope` with `Path.resolve(strict=True)`, compare requested/configured roots for equality, reject `root == Path(root.anchor)`, and derive `domain/model.json` plus `domain/governance.json`. Resolve every existing ancestor of each derived writable path and reject a symlinked ancestor outside the root. Implement child checks with `candidate.resolve(strict=must_exist).relative_to(scope.root)`; translate `ValueError`, missing files, and suffix mismatch into typed errors from `core.generation.errors` without exposing the outside absolute path in the user message.

- [ ] **Step 4: Run focused tests**

Run `cd extension/backend && python -m pytest tests/test_workspace_scope.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/workspace.py extension/backend/core/generation/errors.py extension/backend/tests/test_workspace_scope.py
git commit -m "feat: confine backend paths to workspace"
```

### Task 3: Harden the Existing Atomic UTF-8 Writer

**Files:**

- Modify: `extension/backend/core/io_atomic.py`
- Modify: `extension/backend/core/observability/run_manifest.py`
- Create: `extension/backend/tests/test_io_atomic.py`
- Modify: `extension/backend/tests/test_observability_atomic_write.py`
- Modify: `extension/backend/tests/test_paper_run_manifest.py`
- Modify: `extension/backend/tests/test_aggregate.py`
- Modify: `extension/backend/tests/test_latex_tables.py`

**Interfaces:**

- Produces: the existing `write_text_atomic(target: Path, content: str, *, encoding: str = "utf-8") -> Path`, now with an unpredictable sibling temporary name and cleanup on failure.
- Consumed by: generation model persistence and governance storage.

- [ ] **Step 1: Write atomic-write failure tests**

Use a pre-existing target and monkeypatch `os.replace` to raise. Assert the old content is unchanged and no `*.tmp` file remains. Also assert UTF-8 content round-trips and nested parent creation succeeds.

- [ ] **Step 2: Run the tests and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_io_atomic.py -q`.

Expected: failure because the current helper uses the predictable `model.json.tmp` path and retains it after replacement failure.

- [ ] **Step 3: Implement atomic persistence**

Use `tempfile.NamedTemporaryFile(mode="w", encoding=encoding, dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False)`, then `flush`, `os.fsync`, and `os.replace`. In `finally`, unlink only the exact temporary path if it still exists. Never delete or truncate the target before replacement. Make `core.observability.run_manifest.write_manifest_atomic` delegate to this helper so there is one recipe; migrate tests that patch the old module-local `os.replace` or expect a retained predictable temp file.

- [ ] **Step 4: Run atomic writer tests**

Run `cd extension/backend && python -m pytest tests/test_io_atomic.py tests/test_observability_atomic_write.py tests/test_paper_run_manifest.py tests/test_aggregate.py tests/test_latex_tables.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/io_atomic.py extension/backend/core/observability/run_manifest.py extension/backend/tests/test_io_atomic.py extension/backend/tests/test_observability_atomic_write.py extension/backend/tests/test_paper_run_manifest.py extension/backend/tests/test_aggregate.py extension/backend/tests/test_latex_tables.py
git commit -m "fix: harden atomic artifact writes"
```

### Task 4: Expose Safe Run-Manifest Binding

**Files:**

- Modify: `extension/backend/core/observability/emitter.py`
- Modify: `extension/backend/core/observability/run_manifest.py`
- Modify: `extension/backend/core/observability/__init__.py`
- Test: `extension/backend/tests/test_observability_emitter.py`
- Test: `extension/backend/tests/test_observability_run_manifest.py`

**Interfaces:**

- Produces: `bind_emitter(emitter: StageEmitter) -> Iterator[None]` and `RunManifest.warnings`.
- Consumed by: `run_generation` around the whole lifecycle.

- [ ] **Step 1: Write failing binding and warning tests**

Assert `get_current_emitter()` returns the bound instance inside the context and the previous value afterward, including when the body raises. Assert warning dictionaries round-trip through `RunManifest.model_dump_json`.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
cd extension/backend
python -m pytest tests/test_observability_emitter.py tests/test_observability_run_manifest.py -q
```

Expected: missing export/field failures.

- [ ] **Step 3: Implement the public context manager**

Add a `@contextmanager` that sets `_emitter_var`, yields, and resets the exact token in `finally`. Add `warnings: List[Dict[str, Any]] = Field(default_factory=list)` to `RunManifest`. Keep the current private stage context behavior unchanged.

- [ ] **Step 4: Run observability tests**

Run `cd extension/backend && python -m pytest tests/test_observability_*.py tests/test_pipeline_stage_wrapping.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/observability extension/backend/tests/test_observability_emitter.py extension/backend/tests/test_observability_run_manifest.py
git commit -m "refactor: expose generation manifest binding"
```

### Task 5: Implement the Complete Generation Service

**Files:**

- Create: `extension/backend/core/generation/__init__.py`
- Create: `extension/backend/core/generation/types.py`
- Create: `extension/backend/core/generation/service.py`
- Modify: `extension/backend/core/generation/errors.py`
- Create: `extension/backend/tests/test_generation_service.py`

**Interfaces:**

- Consumes: `WorkspaceScope`, `write_text_atomic`, `DomainArchitect`, `SRSDocumentParser`, `ASTModelSignalExtractor`, `RAGPipeline`, `StageEmitter`.
- Produces: `build_generation_command(requested_workspace_root, configured_workspace_root, file_paths)` and `run_generation(command, progress_callback=None)` exported from `core.generation`.

- [ ] **Step 1: Write service tests with injected/monkeypatched boundaries**

Test the following independently:

- empty SRS list fails before parser construction;
- every SRS path is confined and extension-checked;
- parsed documents preserve request order;
- source metadata contains relative path and real SHA-256;
- the architect receives the same combined document markers and joined SRS label;
- AST enrichment runs against the validated workspace root;
- model persistence is atomic and targets only `domain/model.json`;
- RAG indexes the already parsed text without reparsing files;
- RAG failure returns a `RAG_INDEX_FAILED` warning and manifest `partial_degrade` stage;
- every success/failure finalizes exactly one manifest;
- a concurrent generation attempt fails immediately with typed `GenerationBusyError` and performs no parse, write, or state mutation;
- typed pipeline failures keep their existing fields.

The central success assertion should resemble:

```python
result = run_generation(command, progress_callback=progress.append)
assert result.model_path == workspace / "domain" / "model.json"
assert result.model.project_metadata.source_documents == [
    SourceDocumentReference(relative_path="inputs/a.txt", sha256=sha256_file(srs))
]
assert result.warnings == []
assert json.loads(result.model_path.read_text())["project_name"] == "TestProject"
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_generation_service.py -q`.

Expected: missing service/facade failures.

- [ ] **Step 3: Implement the service in one explicit lifecycle**

Protect the lifecycle with one module-level nonblocking `threading.Lock`; a busy attempt raises `GenerationBusyError` before side effects and an acquired lock is released in `finally`. Create the manifest before the empty-input check; use `bind_emitter` around ingestion, architect, AST, persistence, and RAG; set exact outcomes before raising/returning; attach source provenance before computing the domain summary; serialize with `model_dump_json(indent=2)` through `write_text_atomic`; index each `srs_doc["content"]`; and finalize in one `finally`. Convert expected boundary failures to typed `GenerationError` subclasses while allowing existing `PipelineError` payloads to retain their attributes.

- [ ] **Step 4: Run service and pipeline regressions**

Run:

```bash
cd extension/backend
python -m pytest tests/test_generation_service.py tests/test_pipeline_observability_e2e.py tests/test_pipeline_observability_failures.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/generation extension/backend/tests/test_generation_service.py
git commit -m "feat: unify observable domain model generation"
```

### Task 6: Wire Both FastAPI Routes to the Service

**Files:**

- Modify: `extension/backend/main.py:111-249,443-789`
- Create: `extension/backend/tests/test_generation_routes.py`
- Modify: `extension/backend/tests/test_main_wiring.py`
- Modify: `extension/backend/tests/test_main_pipeline_error_endpoint.py`
- Modify: `extension/backend/tests/test_pipeline_observability_e2e.py`

**Interfaces:**

- Consumes: `GenerationCommand` and `run_generation`.
- Produces: `GenerateModelRequest(file_paths: List[str], workspace_root: str)` with no `output_path`.

- [ ] **Step 1: Write route-level shared-service tests**

Monkeypatch `main.run_generation` with a spy. Assert sync invokes it once; draining SSE invokes it once with a progress callback; both update `app_state["domain_rules"]` and `app_state["rag"]` from the returned result; the success response retains `success`, `model_path`, `project_name`, `bounded_contexts_count`, `metrics`, and additive `warnings`.

Use `TestClient` for HTTP status assertions: invalid workspace `403`, concurrent generation `409`, no files `422`, typed provider/pipeline failure `502`, and sync success `200`. For SSE, assert the final error event keeps `error` as a string and typed fields as siblings.

- [ ] **Step 2: Run route tests and confirm RED**

Run:

```bash
cd extension/backend
python -m pytest tests/test_generation_routes.py tests/test_main_pipeline_error_endpoint.py -q
```

Expected: failures because routes still duplicate work and accept `output_path`.

- [ ] **Step 3: Replace duplicate route bodies**

Add a single `_execute_generation(request, progress_callback=None)` adapter that resolves the command, calls `run_generation`, updates app state, and returns the transport-neutral result. Sync serializes it. SSE invokes the same adapter inside the existing worker and forwards progress events. Remove the dead `_run_generate_pipeline` implementation and update its tests to assert the live routes instead.

Map expected errors to a sanitized `JSONResponse` with status codes before returning; do not return `{success: false}` with HTTP 200. SSE errors remain events after streaming begins.

- [ ] **Step 4: Run all endpoint and observability tests**

Run:

```bash
cd extension/backend
python -m pytest tests/test_generation_routes.py tests/test_main_wiring.py tests/test_main_pipeline_error_endpoint.py tests/test_main_pipeline_error_response.py tests/test_pipeline_observability_e2e.py tests/test_pipeline_observability_failures.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/main.py extension/backend/tests/test_generation_routes.py extension/backend/tests/test_main_wiring.py extension/backend/tests/test_main_pipeline_error_endpoint.py extension/backend/tests/test_pipeline_observability_e2e.py
git commit -m "refactor: route generation through shared service"
```

### Task 7: Update the VS Code Generation Client Boundary

**Files:**

- Modify: `extension/src/types.ts`
- Modify: `extension/src/extension.ts:658-982`
- Modify: `extension/src/test/extension.test.ts`

**Interfaces:**

- Consumes: new backend request `{file_paths, workspace_root}`.
- Produces: additive `warnings` and typed SSE error fields in client interfaces.

- [ ] **Step 1: Add failing pure request-boundary tests**

Extract and test `isPathInsideWorkspace(workspaceRoot, candidate)` and `buildGenerateModelRequest(workspaceRoot, filePaths)`. Assert sibling prefixes and `..`/resolved-outside paths fail, workspace files pass, and the result has no `output_path` key.

- [ ] **Step 2: Compile/tests to confirm RED**

Run `cd extension && npm run compile && npm test -- --grep "generation request"`.

Expected: missing exports/tests fail.

- [ ] **Step 3: Implement request construction and UI rejection**

Add pure helpers to a small `extension/src/backend/generationRequest.ts` module, re-export them from `extension.ts` for tests, set the open dialog `defaultUri` to the active workspace, reject any selected external file with a clear copy-into-workspace message, and send `workspace_root` to both fetch and axios paths. Render explicit degradation warnings in the output channel after success.

- [ ] **Step 4: Run extension regression checks**

Run:

```bash
cd extension
npm run compile
npm run lint
npm test -- --grep "generation request"
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/backend/generationRequest.ts extension/src/types.ts extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "feat: confine extension generation requests"
```

### Task 8: Verify the Shared Generation Deliverable

**Files:** No new implementation files unless a failing gate exposes a scoped defect.

- [ ] **Step 1: Run backend focused and full non-integration suites**

```bash
cd extension/backend
python -m pytest tests/test_workspace_scope.py tests/test_io_atomic.py tests/test_generation_service.py tests/test_generation_routes.py -q
python -m pytest -m "not integration" --cov=. --cov-report=term-missing
```

Expected: all selected and non-integration tests pass.

- [ ] **Step 2: Run static checks**

```bash
cd ../../
python -m compileall -q extension/backend
pyright
cd extension
npm run compile
npm run lint
```

Expected: zero compile, type, and lint errors.

- [ ] **Step 3: Inspect the final scoped diff**

Run `git diff --check` and `git status --short`. Confirm no source path outside the workspace is accepted, no output path is request-controlled, no duplicate generation lifecycle remains in `main.py`, and `.DS_Store` is neither staged nor changed by this work.

- [ ] **Step 4: Commit any verification-only correction, then stop this plan**

Use a narrowly scoped commit message describing the actual correction. Do not combine release/docs work into this deliverable.
