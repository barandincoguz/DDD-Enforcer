# Typed Governance Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add grounded, stale-aware validation contracts and read-only code/test generation plans without ever modifying workspace source code.

**Architecture:** A focused `core.governance` package loads the provenance-aware domain model, verifies the exact SRS sources, computes a semantic fingerprint, performs provider-neutral structured-output calls, and atomically persists only `domain/governance.json`. Three thin FastAPI routes expose typed contracts and errors; code/test plans remain in memory.

**Tech Stack:** Python 3.12, FastAPI, Pydantic v2, provider-neutral `core.llm`, pytest

## Global Constraints

- The shared-generation plan must land first and provide `ProjectMetadata.source_documents`, `WorkspaceScope`, `sha256_file`, and hardened `write_text_atomic`.
- Never scan `inputs/` or another directory to guess a legacy model's SRS provenance.
- `domain/governance.json` is the only new file governance may write.
- Code and test plans are response-only and must not persist or alter source files.
- All workspace, SRS, AST, and optional target paths must remain below the resolved configured `WORKSPACE_PATH`.
- Requests accept no output path, arbitrary source path list, prompt, model ID, or provider credential.
- Missing, stale, or malformed inputs fail explicitly; no generic-text or old-artifact fallback is allowed.
- Every Pydantic governance type uses `ConfigDict(extra="forbid", str_strip_whitespace=True)`.
- Objective length is 1–1000 trimmed characters in backend and extension.
- Structured schema failures receive at most three governance-level attempts; transport retry stays provider-owned.
- Preserve `.DS_Store` and the archived governance-stash SHA.

---

## File Structure and Locked Interfaces

Create `extension/backend/core/governance/` with `__init__.py`, `errors.py`, `schemas.py`, `fingerprint.py`, `workspace.py`, `storage.py`, `prompts.py`, and `service.py`. Create `extension/backend/api/governance.py` plus its package initializer. Tests live in the eight dedicated `test_governance_*.py` files named by the tasks below; shared explicit fixtures live in `tests/governance_fixtures.py`. Modify only `configs/models.py`, `main.py`, and `test_models_registry.py` outside those new packages.

The locked public interfaces are:

- `domain_model_fingerprint(model: DomainModel) -> str`
- `verify_source_documents(scope: WorkspaceScope, recorded: Sequence[SourceDocumentReference]) -> tuple[VerifiedSourceDocument, ...]`
- `resolve_python_target(scope: WorkspaceScope, target_relative_path: Optional[str]) -> Optional[Path]`
- `scan_python_sources(scope: WorkspaceScope) -> tuple[AstSourceSnapshot, ...]`
- `load_domain_model(scope: WorkspaceScope) -> DomainModel`
- `load_current_governance(scope: WorkspaceScope, model: DomainModel, *, validation_context: Mapping[str, object]) -> GovernanceDocument`
- `write_governance(scope: WorkspaceScope, document: GovernanceDocument) -> Path`
- `GovernanceService.generate_contracts(scope) -> GovernanceDocument`
- `GovernanceService.generate_code_plan(scope, *, bounded_context, objective) -> CodeGenerationPlan`
- `GovernanceService.generate_test_plan(scope, *, bounded_context, objective, target_relative_path) -> TestGenerationPlan`

Schema fields are exhaustively defined in Task 1 tests and mirror the design spec: contract identity/title/scope/severity/rationale/checks/evidence/concepts/notes; document schema/project/fingerprint/time/pipeline/model/source/contracts/warnings; code-plan context/objective/ordered steps/constraints/dependencies/concepts/paths/acceptance/rules/evidence/warnings; and test-plan target plus unit/integration/negative/boundary scenarios with preconditions/fixtures/actions/assertions.

### Task 1: Define Strict Governance Schemas and Errors

**Files:**

- Create: `extension/backend/core/governance/errors.py`
- Create: `extension/backend/core/governance/schemas.py`
- Create: `extension/backend/core/governance/__init__.py`
- Create: `extension/backend/tests/governance_fixtures.py`
- Create: `extension/backend/tests/test_governance_schemas.py`
- Modify: `extension/backend/configs/models.py`
- Modify: `extension/backend/tests/test_models_registry.py`

**Interfaces:**

- Produces: all persisted/request/response types, error classes, and `stage_config("Governance")`.
- Consumed by: all later governance tasks and the extension mirror types.

- [ ] **Step 1: Write strict-schema and registry tests**

Test a complete valid fixture plus these failures: extra keys; blank title/rationale/check/objective; severity outside the three literals; duplicate `rule_id`; project scope with a context; bounded-context scope without a context; evidence where `sentence_index == -1` disagrees with `inference_only`; grounded evidence with no source path; absolute/Windows/UNC/`..` suggested paths; step orders other than exactly `1..n`; non-UTC `generated_at`; inference-only evidence without a visible warning; unknown context/rule/evidence index under validation context; objective lengths 0 and 1001; and missing plan scenario assertions.

Representative tests:

```python
def test_inference_only_contract_requires_grounding_warning(valid_contract):
    payload = valid_contract.model_dump()
    payload["evidence"] = [{
        "source_path": None,
        "sentence_index": -1,
        "inference_only": True,
        "excerpt": None,
    }]
    with pytest.raises(ValidationError):
        GovernanceContractsDraft.model_validate({
            "validation_contracts": [payload],
            "grounding_warnings": [],
        })


def test_code_plan_rejects_unknown_governance_rule(valid_code_plan_payload):
    with pytest.raises(ValidationError):
        CodeGenerationPlan.model_validate(
            valid_code_plan_payload,
            context={
                "bounded_contexts": {"Ordering"},
                "governance_rule_ids": {"BC-ORDER-001"},
                "source_sentence_limits": {"inputs/spec.txt": 3},
            },
        )
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
cd extension/backend
python -m pytest tests/test_governance_schemas.py tests/test_models_registry.py -q
```

Expected: import and missing stage failures.

- [ ] **Step 3: Implement schemas and error taxonomy**

Define a base `GovernanceError` with stable `code`, `status_code`, and sanitized `public_message`. Subclasses use configuration/confinement/unsupported `403`; missing artifact `404`; malformed/provenance/stale/unknown context `409`; credential `401`; provider/schema exhaustion `502`; storage `500`. `UnknownBoundedContextError` additionally stores sorted `valid_bounded_contexts`. Implement cross-field/context-aware model validators and relative-path validation once in a private helper. Add `"Governance": "domain_extraction"` to `STAGE_TO_GROUP`.

- [ ] **Step 4: Run schema tests**

Run `cd extension/backend && python -m pytest tests/test_governance_schemas.py tests/test_models_registry.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/governance extension/backend/configs/models.py extension/backend/tests/governance_fixtures.py extension/backend/tests/test_governance_schemas.py extension/backend/tests/test_models_registry.py
git commit -m "feat: define typed governance contracts"
```

### Task 2: Implement the Semantic Domain Fingerprint

**Files:**

- Create: `extension/backend/core/governance/fingerprint.py`
- Create: `extension/backend/tests/test_governance_fingerprint.py`

**Interfaces:**

- Consumes: `DomainModel`.
- Produces: lowercase SHA-256 from one documented canonical projection.

- [ ] **Step 1: Write change-sensitivity tests**

Build two deep copies for each assertion. Changing `project_metadata.generated_at` or the whole `critic_report` must not change the digest. Changing project name/version/description, source path/hash/order, any bounded-context data/order, global rules, or context map must change it. Reordering dictionary insertion while preserving data must not change it. Assert exact 64-character lowercase hex.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_governance_fingerprint.py -q`.

Expected: missing module.

- [ ] **Step 3: Implement one pure projection**

Start from `model.model_dump(mode="json")`, construct an explicit dictionary containing `project_name`, metadata `version`, `description`, and `source_documents`, plus `bounded_contexts`, `global_rules`, and `context_map`. Do not delete fields from a full dump by broad name matching. Serialize using `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` and hash UTF-8 bytes.

- [ ] **Step 4: Run fingerprint tests**

Run `cd extension/backend && python -m pytest tests/test_governance_fingerprint.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/governance/fingerprint.py extension/backend/tests/test_governance_fingerprint.py
git commit -m "feat: fingerprint semantic domain models"
```

### Task 3: Verify Provenance and Build a Read-Only AST Snapshot

**Files:**

- Create: `extension/backend/core/governance/workspace.py`
- Create: `extension/backend/tests/test_governance_workspace.py`

**Interfaces:**

- Consumes: shared `WorkspaceScope`, `resolve_workspace_file`, `workspace_relative`, `sha256_file`, and source metadata.
- Produces: ordered `VerifiedSourceDocument`, optional confined Python target, and relative-only `AstSourceSnapshot` records.

- [ ] **Step 1: Write provenance and symlink tests**

Cover: missing/empty provenance; unsupported/missing/non-regular source; hash mismatch; exact order; SRS symlink escape; changed source between first and second hash; target absolute/traversal/missing/directory/non-Python/symlink escape; Python discovery ignoring `.venv`, `venv`, `node_modules`, `.git`, extension backend, and `domain`; discovered symlink escape; AST snapshots contain only relative POSIX paths, digest, and class names.

Explicitly monkeypatch any workspace scan and assert it is not called when provenance is absent.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_governance_workspace.py -q`.

Expected: missing governance workspace module.

- [ ] **Step 3: Implement deterministic verification and scanning**

For each recorded source, resolve it below `scope.root`, require a configured SRS suffix, compare current SHA-256 with the stored lowercase digest, parse later through the service, then compare SHA-256 a second time after parsing. Discover Python files with the existing AST discovery helper, resolve every file again, skip configured trees, and extract class names without writing intermediate diagnostics or source contents. Sort AST snapshots by relative path.

- [ ] **Step 4: Run workspace tests**

Run `cd extension/backend && python -m pytest tests/test_governance_workspace.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/governance/workspace.py extension/backend/tests/test_governance_workspace.py
git commit -m "feat: verify governance workspace evidence"
```

### Task 4: Load and Atomically Store Governance Artifacts

**Files:**

- Create: `extension/backend/core/governance/storage.py`
- Create: `extension/backend/tests/test_governance_storage.py`

**Interfaces:**

- Consumes: hardened `write_text_atomic`, `domain_model_fingerprint`, and strict schemas.
- Produces: `load_domain_model`, `load_current_governance`, and `write_governance`.

- [ ] **Step 1: Write storage lifecycle tests**

Test missing and malformed `domain/model.json`; missing/malformed governance; fingerprint mismatch; valid round trip; exact target `domain/governance.json`; no request-selected output path; replace failure preserves prior governance bytes and removes only its own unique temp; error messages contain no absolute workspace path.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_governance_storage.py -q`.

Expected: missing storage module.

- [ ] **Step 3: Implement explicit artifact operations**

Read using UTF-8, call `DomainModel.model_validate_json`/`GovernanceDocument.model_validate_json(context=validation_context)`, and translate `OSError`/`ValidationError` to typed sanitized errors. Recompute and constant-time compare the stored fingerprint with `hmac.compare_digest`. Write only through `write_text_atomic(scope.governance_path, document.model_dump_json(indent=2))`.

- [ ] **Step 4: Run storage tests**

Run `cd extension/backend && python -m pytest tests/test_governance_storage.py tests/test_io_atomic.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/governance/storage.py extension/backend/tests/test_governance_storage.py
git commit -m "feat: store governance artifacts atomically"
```

### Task 5: Build Grounded, Non-Mutating Prompts

**Files:**

- Create: `extension/backend/core/governance/prompts.py`
- Create: `extension/backend/tests/test_governance_prompts.py`

**Interfaces:**

- Consumes: semantic model projection, verified SRS passages, relative AST snapshot, selected context, and current rules.
- Produces: three deterministic user prompts.

- [ ] **Step 1: Write prompt privacy and grounding tests**

Assert contract prompt contains only recorded relative SRS paths and locally numbered passages; code/test prompts contain only the selected context, applicable rules, objective, optional relative target, and relative AST symbols. Assert all prompts exclude temporary absolute roots, API-key-like fixture strings, raw provider responses, unavailable paths, unselected context detail for context-specific plans, full Python source, and claims that files were created, modified, executed, or verified.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_governance_prompts.py -q`.

Expected: missing prompt module.

- [ ] **Step 3: Implement explicit prompt serialization**

Define `PromptSourceDocument(path: str, passages: tuple[str, ...])`. Parse verified SRS text into passages with `section_aware_chunks`, preserve source/path order, and serialize through `json.dumps` with Unicode retained and keys sorted. Include the exact constraints: output must satisfy the supplied JSON schema; use only numbered evidence; label unsupported inferences with `-1`; use only workspace-relative suggested paths; plans are advisory and must not claim filesystem mutation or execution.

- [ ] **Step 4: Run prompt tests**

Run `cd extension/backend && python -m pytest tests/test_governance_prompts.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/governance/prompts.py extension/backend/tests/test_governance_prompts.py
git commit -m "feat: build grounded governance prompts"
```

### Task 6: Generate and Validate Governance Contracts

**Files:**

- Create: `extension/backend/core/governance/service.py`
- Create: `extension/backend/tests/test_governance_service.py`

**Interfaces:**

- Consumes: provider-neutral `LLMClient`, `stage_config("Governance")`, verified workspace/model/SRS/AST, prompt builders, and storage.
- Produces: `GovernanceService.generate_contracts`.

- [ ] **Step 1: Write fake-client contract-generation tests**

Create `LLMResponse` fixtures whose `parsed` value is a `GovernanceContractsDraft`. Assert the call uses `schema=GovernanceContractsDraft`, configured model/temperature/seed, and no provider-specific client type. Assert deterministic document metadata: UTC time, `1.1.0`, fingerprint, provider/model ID, relative domain/SRS/AST references with hashes, unique contracts, and inference warnings. Assert validation happens before write and successful write happens once.

Failures: missing model/provenance/hash mismatch/parse failure/AST escape call no provider; `json_failed`/wrong parsed type/semantic validation retries exactly three then raises `StructuredOutputExhaustedError`; auth/transport exceptions map to typed errors; failed attempt preserves prior governance.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_governance_service.py -k contracts -q`.

Expected: missing service.

- [ ] **Step 3: Implement contracts orchestration**

Use this constructor:

```python
def __init__(
    self,
    *,
    client: LLMClient,
    model_id: str,
    provider: Literal["gemini", "ollama"],
    temperature: float,
    seed: Optional[int],
    schema_attempts: int = 3,
    pipeline_version: str = "1.1.0",
) -> None:
    if schema_attempts != 3:
        raise ValueError("schema_attempts must be exactly 3")
    self.client = client
    self.model_id = model_id
    self.provider = provider
    self.temperature = temperature
    self.seed = seed
    self.schema_attempts = schema_attempts
    self.pipeline_version = pipeline_version
```

Load model, reject no provenance, verify and parse exact sources, verify post-parse hashes, scan AST, call structured output under an optional `governance_contracts` manifest stage when an emitter is bound, context-validate the draft, add deterministic provenance/warnings, validate the final `GovernanceDocument`, then write atomically. Retry only `json_failed`, missing/wrong `parsed`, and context-validation failures.

- [ ] **Step 4: Run contract service tests**

Run `cd extension/backend && python -m pytest tests/test_governance_service.py -k contracts -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/governance/service.py extension/backend/tests/test_governance_service.py
git commit -m "feat: generate grounded governance contracts"
```

### Task 7: Generate Read-Only Code and Test Plans

**Files:**

- Modify: `extension/backend/core/governance/service.py`
- Modify: `extension/backend/tests/test_governance_service.py`

**Interfaces:**

- Produces: `generate_code_plan` and `generate_test_plan` with no write operation.
- Consumed by: API routes and extension webview.

- [ ] **Step 1: Write plan service tests**

For both plan methods assert: model/SRS verification and current-governance fingerprint check precede provider call; unknown context exposes sorted names and calls no provider; objective is trimmed; only selected-context/rule evidence is sent; valid parsed response is revalidated against context names, source index bounds, rule IDs, and relative paths; code/test results remain in memory; optional test target is validated and relative; absent target stays `None`; schema failures retry three times; stale governance never falls back to old rules.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_governance_service.py -k "code_plan or test_plan" -q`.

Expected: missing methods or failing behavior.

- [ ] **Step 3: Implement shared plan preparation and typed calls**

Add a private `_prepare_plan` returning the loaded model, verified sources/passages, current governance, selected context, AST snapshot, and optional target. Add a private `_structured_with_validation` implementing the same three-attempt policy. Neither plan method imports `write_text_atomic`, calls `write_governance`, opens files in write mode, nor mutates `app_state`.

- [ ] **Step 4: Run all service tests**

Run `cd extension/backend && python -m pytest tests/test_governance_service.py -q`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/core/governance/service.py extension/backend/tests/test_governance_service.py
git commit -m "feat: generate read-only code and test plans"
```

### Task 8: Expose Three Typed FastAPI Endpoints

**Files:**

- Create: `extension/backend/api/__init__.py`
- Create: `extension/backend/api/governance.py`
- Modify: `extension/backend/main.py`
- Create: `extension/backend/tests/test_governance_api.py`

**Interfaces:**

- Produces: `POST /governance/contracts`, `POST /generation/code-plan`, and `POST /generation/test-plan`.
- Consumes: request schemas, `resolve_workspace_scope`, provider registry/factory, and `GovernanceService`.

- [ ] **Step 1: Write router tests with an injected fake service**

Use a minimal `FastAPI()` app including only the router; override the service dependency. Assert all three 200 response bodies. Parametrize typed `401/403/404/409/500/502` errors and automatic `422`. Assert extra `output_path`, `file_paths`, model/provider/key/prompt fields are rejected. Assert error JSON has `code`, `message`, optional `valid_bounded_contexts`, and never includes absolute root, credential, raw prompt/response, traceback, or exception repr.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension/backend && python -m pytest tests/test_governance_api.py -q`.

Expected: missing router.

- [ ] **Step 3: Implement thin dependency and error mapping**

At request time require `WORKSPACE_PATH`, resolve it against `request.workspace_root`, get `stage_config("Governance")`, use `get_client_for_model(config.model_id)`, derive provider with `model_for_stage("Governance")`, and construct the service. Route bodies make exactly one service call. A `GovernanceError` handler returns `JSONResponse` with the exception status and `GovernanceErrorResponse`; unknown exceptions log a server-side correlation identifier but return only a generic 500 body.

- [ ] **Step 4: Include the router and run API regressions**

Run:

```bash
cd extension/backend
python -m pytest tests/test_governance_api.py tests/test_api.py tests/test_main_pipeline_error_endpoint.py -q
```

Expected: PASS; integration-marked live tests remain deselected unless explicitly enabled.

- [ ] **Step 5: Commit**

```bash
git add extension/backend/api extension/backend/main.py extension/backend/tests/test_governance_api.py
git commit -m "feat: expose typed governance endpoints"
```

### Task 9: Prove the Source No-Mutation Contract

**Files:**

- Create: `extension/backend/tests/test_governance_read_only.py`

**Interfaces:**

- Verifies: contracts write only governance; plans write nothing; failures preserve everything.

- [ ] **Step 1: Implement a byte/mode inventory helper in the test**

Inventory every regular file below the workspace as `{relative_path: (sha256, stat.S_IMODE)}` and separately hash an outside sentinel plus symlink target. Use a real service with fake provider output.

- [ ] **Step 2: Add contracts mutation test**

Assert the after inventory differs only by the allowed addition/replacement of `domain/governance.json`; every `.py`, SRS, `domain/model.json`, hidden file, outside sentinel, and symlink target is byte/mode identical.

- [ ] **Step 3: Add code/test plan zero-write tests**

Generate each plan and assert the complete before/after workspace inventory is identical, including governance. Assert no temporary, proposed path, test, or source file appeared.

- [ ] **Step 4: Add failure preservation tests**

For schema exhaustion and monkeypatched replace failure, assert prior governance bytes, complete workspace inventory, and outside targets are unchanged.

- [ ] **Step 5: Run and commit**

Run `cd extension/backend && python -m pytest tests/test_governance_read_only.py -q`.

Expected: PASS.

```bash
git add extension/backend/tests/test_governance_read_only.py
git commit -m "test: prove governance never edits source"
```

### Task 10: Verify the Governance Backend Deliverable

**Files:** No planned production changes.

- [ ] **Step 1: Run all governance tests**

```bash
cd extension/backend
python -m pytest \
  tests/test_governance_schemas.py \
  tests/test_governance_fingerprint.py \
  tests/test_governance_workspace.py \
  tests/test_governance_storage.py \
  tests/test_governance_prompts.py \
  tests/test_governance_service.py \
  tests/test_governance_api.py \
  tests/test_governance_read_only.py -q
```

Expected: all pass.

- [ ] **Step 2: Run backend regression and static gates**

```bash
python -m pytest -m "not integration" --cov=. --cov-report=term-missing
python -m compileall -q core/governance api main.py
cd ../..
pyright
```

Expected: all non-integration tests pass; Pyright has zero errors; compileall is silent and exits zero.

- [ ] **Step 3: Inspect API and filesystem invariants**

Run `git diff --check`, `git status --short`, and a focused search for write/execute primitives under `extension/backend/core/governance`. Confirm only `storage.py` imports/calls the atomic writer, prompts/service do not execute code, and plan methods contain no write primitive.

- [ ] **Step 4: Stop at the deliverable boundary**

Do not add extension UI, release tooling, README, or figure changes in this plan. Report any cross-plan interface mismatch to the primary agent before changing a public name.
