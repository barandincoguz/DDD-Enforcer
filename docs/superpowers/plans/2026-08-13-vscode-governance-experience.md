# VS Code Governance Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three safe, polished VS Code workflows for governance contracts and read-only code/test plans.

**Architecture:** A dedicated `src/governance` package mirrors backend types, owns sanitized HTTP transport, renders copyable Markdown, and presents a CSP-locked webview. Command orchestration receives the existing backend lifecycle through explicit dependencies, so `extension.ts` only registers the feature and no governance module can edit source files.

**Tech Stack:** TypeScript 5.9, VS Code API 1.106.1, Axios, Node crypto/path/fs, Mocha extension tests

## Global Constraints

- Backend governance request and response contracts are authoritative; mirror their exact snake_case JSON fields.
- Register exactly three product commands: `generateGovernanceContracts`, `generateCodePlan`, and `generateTestPlan`.
- Use the first workspace folder, matching the backend's current `WORKSPACE_PATH`; do not offer another root.
- Plans remain in memory and appear only in a webview; no source or test file is created or changed.
- Objectives are trimmed and limited to 1–1000 characters.
- Cancellation produces no request and no warning.
- Never log Axios configuration, response bodies, raw prompts, API keys, or absolute workspace paths.
- The webview receives data only after a `ready` message, uses DOM `textContent`, and opens evidence through host-owned opaque IDs.
- No `unsafe-inline`, command URI, remote asset, or backend-supplied HTML.
- Do not add governance tests to the already oversized `src/test/extension.test.ts`.
- Use `frontend-design` before implementing the webview and preserve native VS Code visual language.

---

## File Structure

### Create

- `extension/src/governance/types.ts` — exact backend mirrors and discriminated result.
- `extension/src/governance/client.ts` — typed transport/error mapping.
- `extension/src/governance/markdown.ts` — deterministic Markdown projection.
- `extension/src/governance/webview.ts` — CSP HTML, typed messages, panel host.
- `extension/src/governance/commands.ts` — prompts, preconditions, and registration.
- `extension/src/test/governance/client.test.ts`
- `extension/src/test/governance/markdown.test.ts`
- `extension/src/test/governance/webview.test.ts`
- `extension/src/test/governance/commands.test.ts`
- `extension/src/test/governance/readOnly.test.ts`
- `extension/src/test/activation.test.ts`

### Modify

- `extension/src/extension.ts` — export/register feature with injected lifecycle/log functions.
- `extension/package.json` — contribute commands.

## Public Interfaces

```ts
export interface GovernanceContractsRequest { workspace_root: string }
export interface CodePlanRequest {
  workspace_root: string;
  bounded_context: string;
  objective: string;
}
export interface TestPlanRequest extends CodePlanRequest {
  target_relative_path?: string;
}

export type GovernanceResult =
  | { kind: "contracts"; value: GovernanceDocument }
  | { kind: "code-plan"; value: CodeGenerationPlan }
  | { kind: "test-plan"; value: TestGenerationPlan };

export interface GovernanceCommandDeps {
  ensureBackendRunning(): Promise<boolean>;
  getBackendBaseUrl(): string;
  logSanitized(message: string): void;
}

export function registerGovernanceCommands(
  context: vscode.ExtensionContext,
  deps: GovernanceCommandDeps,
): vscode.Disposable[];
```

---

### Task 1: Mirror the Strict Governance API Types

**Files:**

- Create: `extension/src/governance/types.ts`
- Create: `extension/src/test/governance/client.test.ts`

**Interfaces:**

- Produces: request, evidence, contract, plan, error, and result TypeScript types.
- Consumed by: client, commands, Markdown, and webview.

- [ ] **Step 1: Add compile-time fixture assertions**

Create realistic fixtures for one `GovernanceDocument`, `CodeGenerationPlan`, and `TestGenerationPlan`, including inference-only and grounded evidence. Use `satisfies` so missing or renamed backend fields break compilation. Assert request fixtures contain `workspace_root` and never `output_path`.

- [ ] **Step 2: Compile and confirm RED**

Run `cd extension && npm run compile`.

Expected: missing module/type errors.

- [ ] **Step 3: Add exact interface mirrors**

Mirror backend literals and snake_case keys exactly. Keep runtime validation out of this file; transport validation is Task 2. Export `GovernanceResult` as the only webview input union.

- [ ] **Step 4: Compile**

Run `cd extension && npm run compile`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/governance/types.ts extension/src/test/governance/client.test.ts
git commit -m "feat: mirror governance API contracts"
```

### Task 2: Implement Sanitized Typed Transport

**Files:**

- Create: `extension/src/governance/client.ts`
- Modify: `extension/src/test/governance/client.test.ts`

**Interfaces:**

- Produces: `GovernanceClient` and `GovernanceApiError`.
- Consumes: injectable Axios-compatible `post` function and a fixed backend base URL.

- [ ] **Step 1: Write client success and error tests**

Test exact URLs/bodies for all three methods. Map statuses to kinds: `422 invalid_request`, `403 forbidden`, `404 missing_artifact`, `409 conflict`, `401 unauthorized`, `502 provider`, `500 server`, and missing-response/network to `transport`. Test malformed success payload as `server`. Assert error messages do not contain fixture key, absolute root, request config, prompt, or raw response.

Representative assertion:

```ts
await client.generateCodePlan({
  workspace_root: "/workspace",
  bounded_context: "Ordering",
  objective: "Add cancellation policy",
});
assert.deepStrictEqual(post.calls[0], {
  url: "http://127.0.0.1:8123/generation/code-plan",
  body: {
    workspace_root: "/workspace",
    bounded_context: "Ordering",
    objective: "Add cancellation policy",
  },
});
```

- [ ] **Step 2: Run focused test and confirm RED**

Run `cd extension && npm test -- --grep "GovernanceClient"`.

Expected: missing client failures.

- [ ] **Step 3: Implement the client**

Use an injected `post<T>(url, body, config)` defaulting to `axios.post`, a 300-second timeout, and response-type guards for required top-level fields. `GovernanceApiError` exposes only `status`, stable `kind`, and a concise sanitized message obtained from backend `message` when it is a plain bounded-length string. Never store the original Axios error as an enumerable field.

- [ ] **Step 4: Run tests and lint**

Run `cd extension && npm test -- --grep "GovernanceClient" && npm run lint`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/governance/client.ts extension/src/test/governance/client.test.ts
git commit -m "feat: add typed governance API client"
```

### Task 3: Render Deterministic Human-Readable Markdown

**Files:**

- Create: `extension/src/governance/markdown.ts`
- Create: `extension/src/test/governance/markdown.test.ts`

**Interfaces:**

- Produces: `escapeMarkdownText` and `renderGovernanceMarkdown`.
- Consumed by: copy action and webview host.

- [ ] **Step 1: Write snapshot-style semantic tests**

For contracts assert project/fingerprint, scope/severity, checks, evidence, affected concepts, and grounding warnings. For code plan assert ordered steps, constraints, forbidden dependencies, suggested paths, acceptance criteria, rules, and evidence. For test plan assert scenario categories with preconditions/fixtures/actions/assertions plus target. Test pipes, backticks, `<script>`, command links, nested Markdown links, and newlines cannot create executable links/HTML.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension && npm test -- --grep "Governance Markdown"`.

Expected: missing renderer.

- [ ] **Step 3: Implement explicit per-kind renderers**

Escape `\`, backticks, brackets, parentheses, angle brackets, pipes, and control characters. Render relative evidence paths as plain code text, not clickable Markdown links. Use stable headings and preserve backend order; do not sort steps or scenarios.

- [ ] **Step 4: Run focused tests**

Run `cd extension && npm test -- --grep "Governance Markdown"`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/governance/markdown.ts extension/src/test/governance/markdown.test.ts
git commit -m "feat: render governance plans as Markdown"
```

### Task 4: Build the Secure Governance Webview

**Files:**

- Create: `extension/src/governance/webview.ts`
- Create: `extension/src/test/governance/webview.test.ts`

**Interfaces:**

- Produces: `parseGovernanceWebviewMessage`, `buildGovernanceHtml`, and `showGovernanceWebview`.
- Consumes: `GovernanceResult`, Markdown renderer, workspace root, and extension context.

- [ ] **Step 1: Write CSP, message, and evidence tests**

Assert HTML has nonce-only script/style policies, no `unsafe-inline`, command URI, remote URL, or embedded backend values. Parse only `ready`, `copyMarkdown`, `copyJson`, and `openEvidence` with a nonblank opaque ID; ignore unknown/extra/malformed messages. Assert payloads containing scripts, quotes, command links, and traversal render as text. Assert evidence IDs cannot open an unlisted, missing, absolute, outside, or symlink-escaped path.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension && npm test -- --grep "Governance Webview"`.

Expected: missing webview module.

- [ ] **Step 3: Implement the native VS Code presentation**

Call the `frontend-design` skill before editing. Use restrained VS Code theme variables, a compact metadata header, severity badges, scope-grouped contract cards, numbered plan steps, scenario sections, warning callouts, and pinned copy controls. Build static HTML without data. On `ready`, post a typed view model; client script creates nodes and assigns `textContent`. Host owns `Map<evidenceId, relativePath>`, resolves/rechecks realpath under the workspace, then invokes `vscode.open`. Copy actions use the host-owned original result and `vscode.env.clipboard`.

- [ ] **Step 4: Run webview tests, compile, and lint**

Run:

```bash
cd extension
npm test -- --grep "Governance Webview"
npm run compile
npm run lint
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/governance/webview.ts extension/src/test/governance/webview.test.ts
git commit -m "feat: present governance results securely"
```

### Task 5: Orchestrate Commands and Preconditions

**Files:**

- Create: `extension/src/governance/commands.ts`
- Create: `extension/src/test/governance/commands.test.ts`

**Interfaces:**

- Consumes: injected lifecycle/base URL/logger, `GovernanceClient`, workspace/model/governance files, and webview.
- Produces: four command handlers internally and three registered product commands.

- [ ] **Step 1: Write command workflow tests**

Cover missing workspace/model/governance, malformed model, empty/invalid context list, backend start failure, context/objective/target confirmation cancellation, objective trimming/limit, exact client request, and concise error notifications. Contracts require model only. Code/test plans require model plus governance locally, while backend remains authoritative for staleness. Use active target only when it is a file-scheme `.py` below the workspace and user confirms the displayed relative path. Outside/non-Python targets are rejected without transport.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension && npm test -- --grep "Governance Commands"`.

Expected: missing command module.

- [ ] **Step 3: Implement dependency-injected handlers**

Read only `domain/model.json` to list `bounded_contexts[*].context_name`; reject malformed JSON explicitly. Use QuickPick and InputBox. Construct the client only after `ensureBackendRunning` succeeds. Log only operation name/status/error kind. On success call `showGovernanceWebview`; on cancellation return immediately. Return registered disposables so activation owns their lifecycle.

- [ ] **Step 4: Run command tests**

Run `cd extension && npm test -- --grep "Governance Commands"`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/governance/commands.ts extension/src/test/governance/commands.test.ts
git commit -m "feat: add governance command workflows"
```

### Task 6: Wire Activation and Package Contributions

**Files:**

- Modify: `extension/src/extension.ts:20-140,207-330`
- Modify: `extension/package.json:29-52`
- Create: `extension/src/test/activation.test.ts`

**Interfaces:**

- Consumes: `registerGovernanceCommands`.
- Produces: exact contributed and activated command IDs.

- [ ] **Step 1: Write activation tests**

Assert these exact commands are contributed and appear in `vscode.commands.getCommands(true)` after activation:

```text
ddd-enforcer.generateGovernanceContracts
ddd-enforcer.generateCodePlan
ddd-enforcer.generateTestPlan
```

Assert each registration is disposed with the extension context and activation does not start the backend eagerly.

- [ ] **Step 2: Run and confirm RED**

Run `cd extension && npm test -- --grep "Governance Activation"`.

Expected: missing command contributions.

- [ ] **Step 3: Register through narrow dependencies**

Import the command registrar into `extension.ts`; pass wrappers around the existing `ensureBackendRunning`, `backendPort`, and `log`. Push each returned disposable into `context.subscriptions`. Add the exact contributed titles to `package.json` without changing activation events.

- [ ] **Step 4: Run activation/compile/lint**

Run `cd extension && npm test -- --grep "Governance Activation" && npm run compile && npm run lint`.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add extension/src/extension.ts extension/package.json extension/src/test/activation.test.ts
git commit -m "feat: register governance extension commands"
```

### Task 7: Prove the Extension Is Read-Only

**Files:**

- Create: `extension/src/test/governance/readOnly.test.ts`

**Interfaces:**

- Verifies: UI flows never perform source writes.

- [ ] **Step 1: Create a temporary workspace fixture**

Include a provenance-aware `domain/model.json`, current `domain/governance.json`, Python target, non-Python file, SRS, hidden file, outside sentinel, and symlink escape when supported. Inventory byte hashes and file modes.

- [ ] **Step 2: Exercise all three commands with fake transport**

Contracts transport may simulate backend governance replacement; the extension itself must make no write call. Code/test plan commands display results and copy data while the complete workspace remains byte-identical.

- [ ] **Step 3: Add malicious evidence and cancellation cases**

Assert absolute/traversal/symlink evidence never opens or changes a file. Assert every cancellation leaves transport calls and inventory unchanged.

- [ ] **Step 4: Run and commit**

Run `cd extension && npm test -- --grep "Governance Read Only"`.

Expected: PASS.

```bash
git add extension/src/test/governance/readOnly.test.ts
git commit -m "test: prove governance UI is read-only"
```

### Task 8: Verify the VS Code Governance Deliverable

**Files:** No planned implementation changes.

- [ ] **Step 1: Run all governance extension tests**

```bash
cd extension
npm test -- --grep "Governance"
```

Expected: all governance client, Markdown, webview, command, activation, and read-only tests pass in the pinned extension host established by the release plan.

- [ ] **Step 2: Run static gates**

```bash
npm run compile
npm run lint
```

Expected: zero errors.

- [ ] **Step 3: Review structural boundaries**

Confirm no governance module imports `child_process` except none, no source mutation APIs occur, the webview has no `innerHTML` assignment for dynamic values, `extension.ts` did not grow governance workflow logic, and only command registration was added to the entrypoint.

- [ ] **Step 4: Stop at the deliverable boundary**

Do not change dependency locks, CI, package ignore, release version, or README in this plan.
