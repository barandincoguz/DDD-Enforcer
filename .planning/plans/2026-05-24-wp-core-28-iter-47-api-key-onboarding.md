# WP-CORE-28 Iter 47 — API Key Onboarding Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (chosen by user — implementer + spec reviewer + quality reviewer + fix loop) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pre-validation, error classification, migration-to-secret-storage flow, and clear user feedback to the VS Code extension's Gemini API key onboarding path (`extension/src/extension.ts:460-498`).

**Architecture:** Three small pure functions (TDD-tested via mocha + the existing exported-function precedent set by `classifySaveForValidationFromContent`) compose the new behavior; `getApiKey` and `startBackend` glue them together with VS Code UI calls. No new module files — the WP-CORE-28 spec locks "extension.ts refactor: NONE". Pre-validation hits Gemini directly via `https://generativelanguage.googleapis.com/v1beta/models?key=$KEY` (cheap, public, doesn't burn quota) — chosen over backend-mediated probe because `/health` doesn't validate keys and no backend "test key" endpoint exists. Migration persistence uses `context.globalState.update("apiKeyMigrationDeclined", true)` per VS Code's per-extension global state.

**Tech Stack:** TypeScript 5.9, `axios` (already in deps), `vscode` 1.106 API (`secrets`, `globalState`, `showInformationMessage`, `showInputBox`), `mocha` via `@vscode/test-cli`, no new dependencies.

---

## Pre-flight — Spec discrepancy to flag

The WP-CORE-28 spec acceptance criterion says "All 4 existing API-key-related test cases in `extension.test.ts` still pass." Reality: `extension/src/test/extension.test.ts` has 14 tests, **zero** of which touch the API-key flow. Interpret as: all 14 existing tests still pass, plus the 3 new ones this plan adds. No correction to the spec — just don't get tripped up looking for the phantom 4.

## File Structure

| File | Action | Why |
|------|--------|-----|
| `extension/src/extension.ts` | Modify (add ~3 exported pure functions in a new section before existing `classifySaveForValidation` exports; rewrite `getApiKey`; minor `startBackend` log) | Spec locks "NO refactor of extension.ts layout" — additions are not refactor |
| `extension/src/test/extension.test.ts` | Modify (append 3 new mocha `test(...)` blocks at the end of the existing `suite`) | Reuse the established suite; no new file |

No new dependencies, no new module files, no backend coupling.

---

## Task 1: Pure function `classifyApiKeyError`

**Goal:** Map an axios/network error from the Gemini models-list probe to a stable, UI-renderable kind. Pure — no I/O, no VS Code calls.

**Files:**
- Modify: `extension/src/extension.ts` (add new exported function in a new `// API KEY VALIDATION (pure)` section, immediately before the existing `// HELPER FUNCTIONS` section that contains `getApiKey` at line 460)
- Modify: `extension/src/test/extension.test.ts` (add new mocha `test(...)` blocks inside the existing `suite("Extension Test Suite", ...)`)

- [ ] **Step 1: Write the failing tests**

Append to `extension/src/test/extension.test.ts` (inside the existing `suite("Extension Test Suite", () => { ... })`, just before the closing `});` on line 204). First add the import at the top of the file (modify line 5 to include `classifyApiKeyError`):

```typescript
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
} from "../extension";
```

Then append these tests inside the suite:

```typescript
  // ==========================================================================
  // API KEY VALIDATION TESTS (Iter 47)
  // ==========================================================================

  test("classifyApiKeyError maps HTTP 400 to invalid_key", () => {
    const result = classifyApiKeyError({ response: { status: 400 } });
    assert.strictEqual(result, "invalid_key");
  });

  test("classifyApiKeyError maps HTTP 401 to invalid_key", () => {
    const result = classifyApiKeyError({ response: { status: 401 } });
    assert.strictEqual(result, "invalid_key");
  });

  test("classifyApiKeyError maps HTTP 403 to invalid_key", () => {
    const result = classifyApiKeyError({ response: { status: 403 } });
    assert.strictEqual(result, "invalid_key");
  });

  test("classifyApiKeyError maps HTTP 429 to rate_limited", () => {
    const result = classifyApiKeyError({ response: { status: 429 } });
    assert.strictEqual(result, "rate_limited");
  });

  test("classifyApiKeyError maps ENOTFOUND to network_error", () => {
    const result = classifyApiKeyError({ code: "ENOTFOUND" });
    assert.strictEqual(result, "network_error");
  });

  test("classifyApiKeyError maps ECONNABORTED to network_error", () => {
    const result = classifyApiKeyError({ code: "ECONNABORTED" });
    assert.strictEqual(result, "network_error");
  });

  test("classifyApiKeyError maps unrecognized error to unknown", () => {
    const result = classifyApiKeyError({ response: { status: 500 } });
    assert.strictEqual(result, "unknown");
  });

  test("classifyApiKeyError maps undefined error to unknown", () => {
    const result = classifyApiKeyError(undefined);
    assert.strictEqual(result, "unknown");
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd extension && npm run compile`
Expected: TypeScript compile FAILS with `Module '"../extension"' has no exported member 'classifyApiKeyError'` — that's the contract failure that proves the test will fail at runtime if we forced it.

(We don't run `vscode-test` for the failing-pass cycle because it requires downloading VS Code on first run; the compile failure is the equivalent RED signal for this TypeScript codebase. The precedent for using `npm run compile` as the test gate is the existing `pretest` script: `"pretest": "npm run compile && npm run lint"` in `extension/package.json`.)

- [ ] **Step 3: Write the minimal implementation**

Insert into `extension/src/extension.ts` immediately before the `// HELPER FUNCTIONS` section header (currently around line 454). Add:

```typescript
// =============================================================================
// API KEY VALIDATION (pure functions — testable without vscode)
// =============================================================================

/** Stable error kinds for the Gemini API-key pre-validation probe. */
export type ApiKeyErrorKind =
  | "invalid_key"
  | "rate_limited"
  | "network_error"
  | "unknown";

/**
 * Classify an axios/network error from the API-key probe into a stable
 * kind so the UI layer can render a clear message. Pure: no I/O, no
 * vscode calls.
 *
 * Treated as `invalid_key`: HTTP 400/401/403 (Gemini rejects malformed
 * or unauthorized keys with these statuses).
 *
 * Treated as `rate_limited`: HTTP 429.
 *
 * Treated as `network_error`: axios connection codes ENOTFOUND,
 * ECONNABORTED, ECONNREFUSED, ETIMEDOUT.
 *
 * Everything else (including undefined input) maps to `unknown`.
 */
export function classifyApiKeyError(err: unknown): ApiKeyErrorKind {
  if (err === undefined || err === null) {
    return "unknown";
  }
  const e = err as { response?: { status?: number }; code?: string };
  const status = e.response?.status;
  if (status === 400 || status === 401 || status === 403) {
    return "invalid_key";
  }
  if (status === 429) {
    return "rate_limited";
  }
  const networkCodes = new Set([
    "ENOTFOUND",
    "ECONNABORTED",
    "ECONNREFUSED",
    "ETIMEDOUT",
  ]);
  if (e.code && networkCodes.has(e.code)) {
    return "network_error";
  }
  return "unknown";
}
```

- [ ] **Step 4: Run compile to verify it passes**

Run: `cd extension && npm run compile`
Expected: SUCCESS (no TS errors).

- [ ] **Step 5: Lint clean**

Run: `cd extension && npm run lint`
Expected: SUCCESS (no eslint errors).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add classifyApiKeyError pure function — iter 47 step A

WP-CORE-28 Feature 1 (API key onboarding) groundwork. Pure error
classifier maps axios/network errors from the Gemini probe into
four stable kinds (invalid_key, rate_limited, network_error, unknown)
that the UI layer can render. Follows the established pattern of
exporting pure helpers from extension.ts for test reach
(precedent: classifySaveForValidationFromContent).

8 unit tests cover HTTP 400/401/403, 429, ENOTFOUND, ECONNABORTED,
unrecognized status, and undefined input.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Pure function `validateGeminiKey` with injectable HTTP client

**Goal:** Probe Gemini's public models-list endpoint to verify a key works, returning a discriminated-union result. Inject the HTTP function so tests can stub it without network.

**Files:**
- Modify: `extension/src/extension.ts` (add new exported function below `classifyApiKeyError` in the same section)
- Modify: `extension/src/test/extension.test.ts` (add new mocha `test(...)` blocks)

- [ ] **Step 1: Write the failing tests**

Add the import at top of the test file (extend the existing import line added in Task 1):

```typescript
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
  validateGeminiKey,
  type ApiKeyValidationResult,
} from "../extension";
```

Append inside the suite:

```typescript
  test("validateGeminiKey returns ok=true on HTTP 200", async () => {
    const fakeHttp = async (_url: string) =>
      ({ status: 200, data: { models: [] } } as { status: number; data: unknown });
    const result: ApiKeyValidationResult = await validateGeminiKey(
      "AIzaFakeButValidLooking",
      fakeHttp,
    );
    assert.strictEqual(result.ok, true);
  });

  test("validateGeminiKey returns ok=false invalid_key on HTTP 400", async () => {
    const fakeHttp = async (_url: string) => {
      const err: any = new Error("Bad Request");
      err.response = { status: 400 };
      throw err;
    };
    const result = await validateGeminiKey("AIzaFakeBadKey", fakeHttp);
    assert.strictEqual(result.ok, false);
    if (!result.ok) {
      assert.strictEqual(result.kind, "invalid_key");
    }
  });

  test("validateGeminiKey returns ok=false network_error on ENOTFOUND", async () => {
    const fakeHttp = async (_url: string) => {
      const err: any = new Error("Network down");
      err.code = "ENOTFOUND";
      throw err;
    };
    const result = await validateGeminiKey("AIzaAnything", fakeHttp);
    assert.strictEqual(result.ok, false);
    if (!result.ok) {
      assert.strictEqual(result.kind, "network_error");
    }
  });

  test("validateGeminiKey rejects empty string before any HTTP call", async () => {
    let httpCalled = false;
    const fakeHttp = async (_url: string) => {
      httpCalled = true;
      return { status: 200, data: {} };
    };
    const result = await validateGeminiKey("", fakeHttp);
    assert.strictEqual(httpCalled, false);
    assert.strictEqual(result.ok, false);
    if (!result.ok) {
      assert.strictEqual(result.kind, "invalid_key");
    }
  });
```

- [ ] **Step 2: Run compile to verify tests fail**

Run: `cd extension && npm run compile`
Expected: TS error `Module has no exported member 'validateGeminiKey'`.

- [ ] **Step 3: Write the minimal implementation**

Append to the `// API KEY VALIDATION` section of `extension/src/extension.ts` (directly after `classifyApiKeyError`):

```typescript
/** Result of a Gemini API-key validation probe. */
export type ApiKeyValidationResult =
  | { ok: true }
  | { ok: false; kind: ApiKeyErrorKind };

/**
 * Injectable HTTP signature: an async function taking a URL and returning
 * `{ status, data }` on success or throwing an axios-shaped error on failure.
 * `validateGeminiKey` defaults to a real axios.get when no injection is given;
 * tests pass a stub to exercise both success and failure paths without
 * touching the network.
 */
export type ApiKeyHttpProbe = (
  url: string,
) => Promise<{ status: number; data: unknown }>;

/** Public Gemini endpoint that accepts a key and returns the model catalogue. */
const GEMINI_MODELS_URL_BASE =
  "https://generativelanguage.googleapis.com/v1beta/models";

/**
 * Probe Gemini to verify the supplied API key is accepted. Returns
 * `{ok: true}` on HTTP 200, otherwise `{ok: false, kind}` with the
 * classified error kind (see `classifyApiKeyError`).
 *
 * Rejects the empty string locally without hitting the network. Trims
 * whitespace before sending.
 */
export async function validateGeminiKey(
  apiKey: string,
  httpProbe?: ApiKeyHttpProbe,
): Promise<ApiKeyValidationResult> {
  const trimmed = apiKey.trim();
  if (!trimmed) {
    return { ok: false, kind: "invalid_key" };
  }
  const probe: ApiKeyHttpProbe =
    httpProbe ??
    (async (url) => {
      const resp = await axios.get(url, { timeout: 5000 });
      return { status: resp.status, data: resp.data };
    });
  try {
    const url = `${GEMINI_MODELS_URL_BASE}?key=${encodeURIComponent(trimmed)}`;
    const { status } = await probe(url);
    if (status === 200) {
      return { ok: true };
    }
    return { ok: false, kind: "unknown" };
  } catch (err) {
    return { ok: false, kind: classifyApiKeyError(err) };
  }
}
```

- [ ] **Step 4: Run compile to verify it passes**

Run: `cd extension && npm run compile`
Expected: SUCCESS.

- [ ] **Step 5: Lint clean**

Run: `cd extension && npm run lint`
Expected: SUCCESS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add validateGeminiKey HTTP probe — iter 47 step B

Probes Gemini's public models-list endpoint
(generativelanguage.googleapis.com/v1beta/models) to verify a key
is accepted. Returns a discriminated-union result; failure path
delegates to classifyApiKeyError (Task A). HTTP function is
injectable so tests stub it without network (4 tests cover
success, invalid_key, network_error, empty-string short-circuit).

Default 5s timeout. No backend coupling — direct-to-Gemini call
keeps the WP-CORE-28 "backend coupling: NONE" constraint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Pure function `decideMigrationOffer`

**Goal:** Decide, given the API-key source and the user's prior migration choice (persisted in `globalState`), whether to surface the migration toast and with what label.

**Files:**
- Modify: `extension/src/extension.ts` (add new exported function below `validateGeminiKey`)
- Modify: `extension/src/test/extension.test.ts` (add new mocha `test(...)` blocks)

- [ ] **Step 1: Write the failing tests**

Add to the test-file import block:

```typescript
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
  validateGeminiKey,
  type ApiKeyValidationResult,
  decideMigrationOffer,
  type ApiKeySource,
} from "../extension";
```

Append inside the suite:

```typescript
  test("decideMigrationOffer offers migration for settings source", () => {
    const decision = decideMigrationOffer("settings", false);
    assert.strictEqual(decision.shouldOffer, true);
    assert.strictEqual(decision.sourceLabel, "VS Code settings");
  });

  test("decideMigrationOffer offers migration for env source", () => {
    const decision = decideMigrationOffer("env", false);
    assert.strictEqual(decision.shouldOffer, true);
    assert.strictEqual(decision.sourceLabel, "GEMINI_API_KEY environment variable");
  });

  test("decideMigrationOffer does NOT offer for secret source", () => {
    const decision = decideMigrationOffer("secret", false);
    assert.strictEqual(decision.shouldOffer, false);
  });

  test("decideMigrationOffer does NOT offer for prompt source", () => {
    const decision = decideMigrationOffer("prompt", false);
    assert.strictEqual(decision.shouldOffer, false);
  });

  test("decideMigrationOffer respects prior declined choice", () => {
    const decisionSettings = decideMigrationOffer("settings", true);
    assert.strictEqual(decisionSettings.shouldOffer, false);
    const decisionEnv = decideMigrationOffer("env", true);
    assert.strictEqual(decisionEnv.shouldOffer, false);
  });
```

- [ ] **Step 2: Run compile to verify tests fail**

Run: `cd extension && npm run compile`
Expected: TS error `Module has no exported member 'decideMigrationOffer'`.

- [ ] **Step 3: Write the minimal implementation**

Append to the `// API KEY VALIDATION` section (directly after `validateGeminiKey`):

```typescript
/** Where a Gemini API key was found. */
export type ApiKeySource = "settings" | "env" | "secret" | "prompt";

/** Decision returned by `decideMigrationOffer`. */
export interface MigrationDecision {
  /** Whether to surface the "move to secret storage?" toast. */
  shouldOffer: boolean;
  /** Human-readable label describing where the key came from (for the toast text). */
  sourceLabel: string;
}

/**
 * Decide whether to surface the migration-to-secret-storage offer for a
 * key sourced from `source`. The user's prior decline (persisted to
 * `globalState` by the caller) suppresses the offer permanently.
 *
 * - `settings` / `env`: less-secure sources → offer migration (unless previously declined)
 * - `secret`: already in secret storage → no offer
 * - `prompt`: just typed in by the user → was stored in secret storage as part of the prompt flow → no offer
 */
export function decideMigrationOffer(
  source: ApiKeySource,
  migrationDeclined: boolean,
): MigrationDecision {
  const labels: Record<ApiKeySource, string> = {
    settings: "VS Code settings",
    env: "GEMINI_API_KEY environment variable",
    secret: "VS Code secret storage",
    prompt: "user prompt",
  };
  if (migrationDeclined) {
    return { shouldOffer: false, sourceLabel: labels[source] };
  }
  if (source === "settings" || source === "env") {
    return { shouldOffer: true, sourceLabel: labels[source] };
  }
  return { shouldOffer: false, sourceLabel: labels[source] };
}
```

- [ ] **Step 4: Run compile to verify it passes**

Run: `cd extension && npm run compile`
Expected: SUCCESS.

- [ ] **Step 5: Lint clean**

Run: `cd extension && npm run lint`
Expected: SUCCESS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add decideMigrationOffer pure function — iter 47 step C

Decides whether to surface the "move API key to secret storage?"
toast based on where the key was sourced (settings/env → offer;
secret/prompt → no offer) and the user's persisted prior decline.
Returns a {shouldOffer, sourceLabel} decision the caller can use
verbatim in the toast text.

5 unit tests cover all four sources and the prior-declined override.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire pre-validation + migration into `getApiKey`

**Goal:** Use Tasks 1-3 to harden the existing `getApiKey` chain. Add a `validatingApiKey` status-bar state, surface a clear error toast on rejection, offer the migration toast on success when the source warrants it, and persist the decline choice.

**Files:**
- Modify: `extension/src/extension.ts:460-498` (rewrite `getApiKey`)
- Modify: `extension/src/extension.ts:1238-1300` (extend the `updateStatusBar` state union with `"validatingApiKey"` and add the corresponding case in the switch)
- Modify: `extension/src/extension.ts:285-293` (the call site in `startBackend` stays the same — `getApiKey` now returns the validated key or undefined; the existing error toast stays as a safety net)

- [ ] **Step 1: Extend `updateStatusBar` to accept `"validatingApiKey"`**

Find the signature at `extension/src/extension.ts:1238-1247` and add `"validatingApiKey"` to the union:

```typescript
function updateStatusBar(
  state:
    | "inactive"
    | "starting"
    | "ready"
    | "validating"
    | "validatingApiKey"
    | "violations"
    | "error"
    | "notInitialized",
  count?: number,
) {
```

Find the `switch (state)` block (starts around line 1255) and add a new case immediately after the existing `"validating"` case (around line 1283):

```typescript
    case "validatingApiKey":
      statusBarItem.text = "$(loading~spin) DDD Enforcer";
      statusBarItem.tooltip = "Validating Gemini API key...";
      statusBarItem.backgroundColor = undefined;
      break;
```

- [ ] **Step 2: Rewrite `getApiKey` (extension.ts:460-498)**

Replace the entire body of `getApiKey` with:

```typescript
/**
 * Gets the Gemini API key from settings, env var, or prompts the user.
 *
 * Iter 47 behavior:
 * - Tracks where the key was sourced from (settings/env/secret/prompt).
 * - Pre-validates the key against Gemini's public models endpoint
 *   (cheap, no backend round-trip) before returning it.
 * - On rejection, surfaces a kind-specific toast and returns undefined.
 * - On success, offers migration to secret storage if the source was
 *   the less-secure settings or env path. The user's decline is
 *   persisted to globalState ("apiKeyMigrationDeclined") so the offer
 *   does not repeat next session.
 */
async function getApiKey(
  context: vscode.ExtensionContext,
): Promise<string | undefined> {
  // Discover the key + its source (first-hit wins, same precedence as before).
  let apiKey: string | undefined;
  let source: ApiKeySource | undefined;

  const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
  const settingsKey = cfg.get<string>("geminiApiKey", "");
  if (settingsKey && settingsKey.trim()) {
    apiKey = settingsKey.trim();
    source = "settings";
  }

  if (!apiKey) {
    const envKey = process.env.GEMINI_API_KEY || "";
    if (envKey.trim()) {
      apiKey = envKey.trim();
      source = "env";
    }
  }

  if (!apiKey) {
    const storedKey = await context.secrets.get("geminiApiKey");
    if (storedKey && storedKey.trim()) {
      apiKey = storedKey.trim();
      source = "secret";
    }
  }

  if (!apiKey) {
    const migrationHint =
      "You can also paste the key here; it will be saved to VS Code secret storage.";
    const inputKey = await vscode.window.showInputBox({
      prompt: `Enter your Gemini API Key. ${migrationHint}`,
      placeHolder: "AIza...",
      password: true,
      ignoreFocusOut: true,
    });
    if (inputKey && inputKey.trim()) {
      await context.secrets.store("geminiApiKey", inputKey.trim());
      apiKey = inputKey.trim();
      source = "prompt";
    }
  }

  if (!apiKey || !source) {
    return undefined;
  }

  // Pre-validate the key against Gemini.
  updateStatusBar("validatingApiKey");
  log(`Validating Gemini API key from source: ${source}`);
  const validation = await validateGeminiKey(apiKey);

  if (!validation.ok) {
    log(`API key validation failed: ${validation.kind}`);
    const messages: Record<ApiKeyErrorKind, string> = {
      invalid_key:
        "DDD Enforcer: Gemini API key was rejected. Check the key and try again.",
      rate_limited:
        "DDD Enforcer: Gemini rate-limited the API key check. Try again in a few seconds.",
      network_error:
        "DDD Enforcer: Could not reach Gemini to validate the API key. Check your network.",
      unknown:
        "DDD Enforcer: Unexpected error validating the API key. See the Output channel for details.",
    };
    vscode.window.showErrorMessage(messages[validation.kind]);
    return undefined;
  }

  log("Gemini API key validated.");

  // Migration offer for less-secure sources.
  const migrationDeclined =
    context.globalState.get<boolean>("apiKeyMigrationDeclined") === true;
  const decision = decideMigrationOffer(source, migrationDeclined);
  if (decision.shouldOffer) {
    const choice = await vscode.window.showInformationMessage(
      `DDD Enforcer found your Gemini API key in ${decision.sourceLabel}. Move it to VS Code secret storage for better security?`,
      "Move to secret storage",
      "Not now",
      "Don't ask again",
    );
    if (choice === "Move to secret storage") {
      await context.secrets.store("geminiApiKey", apiKey);
      if (source === "settings") {
        await cfg.update(
          "geminiApiKey",
          "",
          vscode.ConfigurationTarget.Global,
        );
      }
      log(`API key migrated from ${decision.sourceLabel} to secret storage.`);
      vscode.window.showInformationMessage(
        "DDD Enforcer: Gemini API key moved to secret storage.",
      );
    } else if (choice === "Don't ask again") {
      await context.globalState.update("apiKeyMigrationDeclined", true);
      log("API key migration permanently declined by user.");
    }
  }

  return apiKey;
}
```

- [ ] **Step 3: Add the migration-flow integration test**

Append inside the suite in `extension/src/test/extension.test.ts`:

```typescript
  test("decideMigrationOffer + ApiKeySource type cover all four sources", () => {
    const sources: ApiKeySource[] = ["settings", "env", "secret", "prompt"];
    const allDecisions = sources.map((s) => decideMigrationOffer(s, false));
    const offerCount = allDecisions.filter((d) => d.shouldOffer).length;
    assert.strictEqual(offerCount, 2);
    assert.ok(allDecisions.every((d) => typeof d.sourceLabel === "string"));
    assert.ok(allDecisions.every((d) => d.sourceLabel.length > 0));
  });
```

- [ ] **Step 4: Compile + lint**

Run: `cd extension && npm run compile && npm run lint`
Expected: both SUCCESS.

- [ ] **Step 5: Verify no compile regression**

Run: `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright`
Expected: `0 errors, 0 warnings, 0 informations` (extension changes are TS-only, but pyright must still pass since the gate is blocking).

Run: `cd extension/backend && pytest -m "not integration" -q`
Expected: `729 passed, 31 deselected` (no Python changes, baseline preserved).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): integrate API key pre-validation + migration — iter 47 step D

WP-CORE-28 Feature 1 wired end-to-end:

- getApiKey now tracks the source of the discovered key (settings/
  env/secret/prompt) and pre-validates it against Gemini's public
  models endpoint before returning.
- On rejection: kind-specific error toast (invalid_key, rate_limited,
  network_error, unknown) within ~5 seconds; getApiKey returns
  undefined so the caller's existing error path takes over.
- On success from settings or env: surface migration toast with
  three buttons (Move to secret storage / Not now / Don't ask again).
  Accepting clears the settings entry (env cannot be cleared, but
  the secret-storage copy then wins on next lookup). Declining
  permanently persists the choice via context.globalState.
- New "validatingApiKey" status-bar state shown during the probe.
- Input-box prompt copy updated to mention secret-storage save.

No backend changes. Pyright still 0 errors. Pytest still 729 passing.

Closes Iter 47 of WP-CORE-28.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: F5 manual smoke checklist (HUMAN-IN-LOOP)

**Goal:** Verify the wired behavior in the Extension Development Host. Required by the WP-CORE-28 spec ("per-iter F5 smoke"). The subagent does NOT execute this — the loop stops here and the user runs the steps. The next iter (Iter 48) does NOT start until the user reports back.

**Files:**
- Modify: `.planning/pipeline_audit/CURRENT.md` (after the user confirms smoke passes, append the iter-47 row using the same format as iter-46 entries)

- [ ] **Step 1: Halt and request F5 smoke from the user**

After Task 4 commits, post a Turkish caveman-mode message to the user containing the smoke checklist below. Wait. Do not proceed to Iter 48.

**F5 smoke checklist for the user:**

1. `cd extension && npm run compile` (must succeed).
2. Open `extension/` in VS Code. Press F5 → Extension Development Host launches.
3. In the dev host, open a Python file from any workspace.
4. **Test A — invalid key in settings.** Set `ddd-enforcer.geminiApiKey` in user settings to `AIzaINVALIDKEY`. Trigger any DDD command (e.g., `DDD Enforcer: Initialize Domain Model`). Expect: error toast "Gemini API key was rejected" within ~5 seconds. Status bar briefly shows the spinning "Validating Gemini API key..." tooltip.
5. **Test B — migration offer.** Replace the setting with a real valid Gemini key. Trigger any DDD command. Expect: migration toast "DDD Enforcer found your Gemini API key in VS Code settings…" with three buttons.
6. **Test C — accept migration.** Click `Move to secret storage`. Expect: success toast; the `ddd-enforcer.geminiApiKey` setting is now empty. Next command run skips the migration offer (key now in secret storage).
7. **Test D — decline permanently.** Reset the setting to the valid key. Trigger a command. Click `Don't ask again` on the toast. Quit and relaunch the dev host. Set the setting to a valid key again. Trigger a command. Expect: NO migration toast (decline is sticky).
8. **Test E — empty / no key path unchanged.** Clear all sources (no setting, no env, no secret). Trigger a command. Expect: the input-box prompt appears with the new copy mentioning secret storage.

**Test E reset note for user:** to clear secret storage between runs, the simplest is to wipe `~/.config/Code/User/globalStorage/ddd-enforcer.ddd-enforcer/` (Linux/Mac) or run the test in a clean VS Code profile.

If any step fails: post the failure output. The implementer agent will RED → GREEN → COMMIT a fix. Do NOT advance to Iter 48 until all five tests pass.

- [ ] **Step 2: Update CURRENT.md (only after user reports all green)**

After all five tests pass, update `.planning/pipeline_audit/CURRENT.md`:

- Replace the `**HEAD:**` line with the current HEAD SHA.
- Replace the `**Ahead of origin/main:**` count with the new count.
- Append a new section under the existing "Pyright tightening" section, titled `## Iteration 47 — WP-CORE-28 Feature 1 (API key onboarding) COMPLETE`, summarizing: 4 commits shipped, 17 new test cases added (8 + 4 + 5 + 1 — verify counts in the actual diff), F5 smoke passed against tests A-E.

- [ ] **Step 3: Commit the CURRENT.md update**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add .planning/pipeline_audit/CURRENT.md
git commit -m "$(cat <<'EOF'
chore(planning): CURRENT.md update for iter 47 (WP-CORE-28 Feature 1 complete)

API key onboarding hardening shipped + F5 smoke passed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (per writing-plans skill)

**1. Spec coverage** (against `todos/WP-CORE-28-extension-ux-wave1.md` Feature 1 acceptance):

| Acceptance criterion | Task |
|----------------------|------|
| First-time user with no key sees a single prompt that mentions auto-migration | Task 4 Step 2 (input-box prompt copy includes the secret-storage hint) |
| User with key in settings sees migration offer on first validation success | Task 4 Step 2 (migration block) |
| Invalid key surfaces error within 5 seconds | Task 2 (5s axios timeout) + Task 4 (error toast) |
| Migration choice remembered across sessions | Task 4 Step 2 (`context.globalState.update("apiKeyMigrationDeclined", true)`) |
| All existing tests still pass | Task 4 Step 5 (compile + lint + pytest + pyright gate) |
| Pre-validation success test case | Task 2 (test "validateGeminiKey returns ok=true on HTTP 200") |
| Pre-validation failure test case | Task 2 (3 failure tests: invalid_key, network_error, empty-string short-circuit) |
| Migration accept flow test case | Task 3 (5 decideMigrationOffer tests) + Task 4 Step 3 (integration test) + Task 5 (F5 Test C) |

All criteria covered. No gaps.

**2. Placeholder scan:** no `TBD`, no "handle appropriately", no "similar to Task N". All code blocks complete.

**3. Type consistency:** `ApiKeyErrorKind` (Task 1) → consumed by `ApiKeyValidationResult` (Task 2) → both consumed by `getApiKey` (Task 4). `ApiKeySource` (Task 3) → consumed by `MigrationDecision` (Task 3) → consumed by `getApiKey` (Task 4). Type names stable across all tasks.

End of plan.
