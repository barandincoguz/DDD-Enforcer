# WP-CORE-28 Iter 48 — Backend Lifecycle Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (implementer + spec reviewer + quality reviewer + fix loop) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the VS Code extension's backend child-process lifecycle resilient against crashes and port collisions. When the backend exits unexpectedly, show a user dialog and offer auto-restart with exponential backoff (1s, 2s, 4s, 8s, 16s; 5-attempt cap). When the user manually restarts via the command palette, prompt for a reason and log it. Keep all changes inside `extension/src/extension.ts` per the WP-CORE-28 "no refactor" lock.

**Architecture:** Four small pure functions (`computeBackoffMs`, `shouldAttemptRestart`, `formatExitReason`, `classifyExitForRestart`) plus a new `attemptAutoRestart` controller. Pure helpers TDD-tested via mocha; the controller is exercised through F5 smoke (no VS Code mocking framework). A new module-level flag `backendIntentionalStop` tags `stopBackend`/`restartBackend` paths so the existing `child.on('exit')` handler can distinguish a planned shutdown from a crash. `findAvailablePort` already scans 100 ports (extension.ts:1662) — the "port collision recovery" acceptance criterion is already satisfied by the existing code; this iter only adds a clearer log line surfaced in the Output channel.

**Tech Stack:** TypeScript 5.9, `child_process` (already imported), `vscode` API (`showInformationMessage`, `showInputBox`, `outputChannel.show`), `mocha` via `@vscode/test-cli`. No new dependencies.

---

## Pre-flight — Spec / reality reconciliation

- WP-CORE-28 Feature 2 spec line 86: *"Port collision: backend tries one port, gives up if taken."*
- Reality: `findAvailablePort` at extension.ts:1662 already loops 100 ports. The "gives up if taken" claim was wrong as of an earlier code state; it has since been fixed. This iter does NOT add port-scan logic — it only adds an Output-channel log line documenting which port was chosen when the preferred port was unavailable. The acceptance criterion *"Port collision triggers port scan; extension reports new port via Output panel"* is satisfied by the log addition.
- Acceptance criterion *"3 new test cases: mock spawn + exit handler, restart backoff sequence, port collision scan"* — the first and third require mocking infra we don't have (no sinon). This plan substitutes pure-function unit tests for `computeBackoffMs`, `shouldAttemptRestart`, `formatExitReason`, `classifyExitForRestart` (≥10 tests). F5 smoke covers the integration behavior. The plan author accepts this substitution as consistent with Iter 47's precedent.

## File Structure

| File | Action | Why |
|------|--------|-----|
| `extension/src/extension.ts` | Modify (add new `// BACKEND LIFECYCLE (pure helpers)` section just before existing `// BACKEND MANAGEMENT` section; add `backendIntentionalStop` + `restartAttemptCount` globals; modify `child.on('exit')` handler; modify `stopBackend`; modify `restartBackend` to prompt for reason; add new `handleUnexpectedExit` + `attemptAutoRestart` functions; add Output-channel log to `findAvailablePort`) | Spec locks "NO refactor of extension.ts layout" — additive only |
| `extension/src/test/extension.test.ts` | Modify (append new `// BACKEND LIFECYCLE TESTS (Iter 48)` section after Iter 47's tests) | Reuse the established suite |

No new files. No new deps.

---

## Task 1: Pure function `computeBackoffMs`

**Goal:** Return exponential backoff delay in milliseconds for a given attempt number. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (add new exported function in a new `// BACKEND LIFECYCLE (pure helpers — testable without vscode)` section, placed immediately before the existing global state declarations around line 110 — i.e., add the section between the TYPES section and the GLOBAL STATE section so the helpers live near related backend code)
- Modify: `extension/src/test/extension.test.ts` (extend import block; append mocha `test(...)` blocks inside the existing suite, after the Iter 47 tests)

- [ ] **Step 1: Write the failing tests**

Extend the import block at the top of `extension/src/test/extension.test.ts`:

```typescript
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
  validateGeminiKey,
  type ApiKeyValidationResult,
  decideMigrationOffer,
  type ApiKeySource,
  computeBackoffMs,
} from "../extension";
```

Append inside the suite, just before the closing `});`:

```typescript
  // ==========================================================================
  // BACKEND LIFECYCLE TESTS (Iter 48)
  // ==========================================================================

  test("computeBackoffMs returns 1000 for attempt 0", () => {
    assert.strictEqual(computeBackoffMs(0), 1000);
  });

  test("computeBackoffMs returns 2000 for attempt 1", () => {
    assert.strictEqual(computeBackoffMs(1), 2000);
  });

  test("computeBackoffMs returns 4000 for attempt 2", () => {
    assert.strictEqual(computeBackoffMs(2), 4000);
  });

  test("computeBackoffMs returns 8000 for attempt 3", () => {
    assert.strictEqual(computeBackoffMs(3), 8000);
  });

  test("computeBackoffMs returns 16000 for attempt 4", () => {
    assert.strictEqual(computeBackoffMs(4), 16000);
  });

  test("computeBackoffMs clamps at default maxMs=30000 for large attempts", () => {
    assert.strictEqual(computeBackoffMs(10), 30000);
    assert.strictEqual(computeBackoffMs(100), 30000);
  });

  test("computeBackoffMs honors custom baseMs and maxMs", () => {
    assert.strictEqual(computeBackoffMs(0, 500, 8000), 500);
    assert.strictEqual(computeBackoffMs(3, 500, 8000), 4000);
    assert.strictEqual(computeBackoffMs(10, 500, 8000), 8000);
  });

  test("computeBackoffMs floors negative attempts at baseMs", () => {
    assert.strictEqual(computeBackoffMs(-1), 1000);
    assert.strictEqual(computeBackoffMs(-100), 1000);
  });
```

- [ ] **Step 2: Compile RED**

Run: `cd extension && npm run compile`
Expected: TS error `Module has no exported member 'computeBackoffMs'`.

- [ ] **Step 3: Implementation**

Add this section to `extension/src/extension.ts` between the TYPES section (which ends around line 108) and the GLOBAL STATE section header (currently at line 110). Place the new section header first, then the function. The order inside the new section matters because subsequent tasks append more helpers; keep them in dependency order (computeBackoffMs first, shouldAttemptRestart second, etc.).

```typescript
// =============================================================================
// BACKEND LIFECYCLE (pure helpers — testable without vscode)
// =============================================================================

/**
 * Return exponential backoff delay in milliseconds for a given attempt
 * number. Attempt 0 returns `baseMs`; each subsequent attempt doubles
 * the delay, capped at `maxMs`. Negative attempts are floored at `baseMs`.
 * Pure: no I/O, no time/Date access.
 */
export function computeBackoffMs(
  attempt: number,
  baseMs: number = 1000,
  maxMs: number = 30000,
): number {
  if (attempt <= 0) {
    return baseMs;
  }
  const raw = baseMs * Math.pow(2, attempt);
  return Math.min(raw, maxMs);
}
```

- [ ] **Step 4: Compile GREEN + lint + pyright**

Run: `cd extension && npm run compile && npm run lint`
Expected: both SUCCESS.
Run: `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright`
Expected: `0 errors, 0 warnings, 0 informations`.

- [ ] **Step 5: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add computeBackoffMs pure helper — iter 48 step A

WP-CORE-28 Feature 2 (backend lifecycle resilience) groundwork.
Pure exponential-backoff calculator: attempt 0 → 1s, 1 → 2s,
2 → 4s, 3 → 8s, 4 → 16s, clamped at 30s default. Caller passes
the attempt index from the auto-restart loop.

8 unit tests cover the 1s..16s sequence, large-attempt clamp,
custom base/max overrides, and negative-attempt floor.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Pure function `shouldAttemptRestart`

**Goal:** Decide whether to attempt another auto-restart given the current attempt count. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (append to the `// BACKEND LIFECYCLE (pure helpers)` section established in Task 1, directly after `computeBackoffMs`)
- Modify: `extension/src/test/extension.test.ts` (extend import block, append tests)

- [ ] **Step 1: Update test import**

Replace the existing import block (which after Task 1 imports `computeBackoffMs`) with:

```typescript
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
  validateGeminiKey,
  type ApiKeyValidationResult,
  decideMigrationOffer,
  type ApiKeySource,
  computeBackoffMs,
  shouldAttemptRestart,
} from "../extension";
```

- [ ] **Step 2: Append tests inside the suite (after Task 1 tests)**

```typescript
  test("shouldAttemptRestart allows attempts 0 through 4", () => {
    for (const attempt of [0, 1, 2, 3, 4]) {
      assert.strictEqual(
        shouldAttemptRestart(attempt),
        true,
        `attempt ${attempt} should be allowed`,
      );
    }
  });

  test("shouldAttemptRestart rejects attempt 5 by default", () => {
    assert.strictEqual(shouldAttemptRestart(5), false);
  });

  test("shouldAttemptRestart rejects attempts beyond cap", () => {
    assert.strictEqual(shouldAttemptRestart(6), false);
    assert.strictEqual(shouldAttemptRestart(100), false);
  });

  test("shouldAttemptRestart honors custom maxAttempts", () => {
    assert.strictEqual(shouldAttemptRestart(2, 3), true);
    assert.strictEqual(shouldAttemptRestart(3, 3), false);
  });
```

- [ ] **Step 3: Compile RED**

Expected: missing-export error.

- [ ] **Step 4: Implementation**

Append directly after `computeBackoffMs`:

```typescript
/**
 * Decide whether to attempt another auto-restart. Returns true while
 * `attempt < maxAttempts` (default 5). Pure.
 */
export function shouldAttemptRestart(
  attempt: number,
  maxAttempts: number = 5,
): boolean {
  return attempt < maxAttempts;
}
```

- [ ] **Step 5: Compile GREEN + lint + pyright** (all SUCCESS).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add shouldAttemptRestart pure helper — iter 48 step B

Decides whether the auto-restart loop should attempt another spawn
given the current attempt count. Returns true while attempt < cap
(default 5). Caller increments attempt after each failed spawn;
loop terminates when this returns false.

4 unit tests cover allowed range, default cap, beyond-cap, and
custom-cap override.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Pure functions `formatExitReason` + `classifyExitForRestart`

**Goal:** Two helpers that translate the Node child-process `('exit', code, signal)` payload into human-readable text + a restart decision. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (append to the `// BACKEND LIFECYCLE (pure helpers)` section after `shouldAttemptRestart`)
- Modify: `extension/src/test/extension.test.ts` (extend import block, append tests)

- [ ] **Step 1: Update test import**

```typescript
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
  validateGeminiKey,
  type ApiKeyValidationResult,
  decideMigrationOffer,
  type ApiKeySource,
  computeBackoffMs,
  shouldAttemptRestart,
  formatExitReason,
  classifyExitForRestart,
  type ExitDisposition,
} from "../extension";
```

- [ ] **Step 2: Append tests inside the suite (after Task 2 tests)**

```typescript
  test("formatExitReason describes a clean exit (code=0, signal=null)", () => {
    assert.strictEqual(formatExitReason(0, null), "exited cleanly (code 0)");
  });

  test("formatExitReason describes a non-zero exit", () => {
    assert.strictEqual(
      formatExitReason(1, null),
      "crashed (exit code 1)",
    );
    assert.strictEqual(
      formatExitReason(137, null),
      "crashed (exit code 137)",
    );
  });

  test("formatExitReason describes a signal kill", () => {
    assert.strictEqual(
      formatExitReason(null, "SIGKILL"),
      "killed by signal SIGKILL",
    );
    assert.strictEqual(
      formatExitReason(null, "SIGTERM"),
      "killed by signal SIGTERM",
    );
  });

  test("formatExitReason prefers signal when both are present", () => {
    assert.strictEqual(
      formatExitReason(1, "SIGKILL"),
      "killed by signal SIGKILL",
    );
  });

  test("formatExitReason handles both-null fallback", () => {
    assert.strictEqual(formatExitReason(null, null), "exited (unknown reason)");
  });

  test("classifyExitForRestart returns intentional when stopBackend was called", () => {
    const result: ExitDisposition = classifyExitForRestart(1, null, true);
    assert.strictEqual(result, "intentional");
  });

  test("classifyExitForRestart returns intentional regardless of code when intentional=true", () => {
    assert.strictEqual(classifyExitForRestart(0, null, true), "intentional");
    assert.strictEqual(
      classifyExitForRestart(null, "SIGKILL", true),
      "intentional",
    );
  });

  test("classifyExitForRestart returns crash on non-zero exit code", () => {
    assert.strictEqual(classifyExitForRestart(1, null, false), "crash");
    assert.strictEqual(classifyExitForRestart(137, null, false), "crash");
  });

  test("classifyExitForRestart returns crash on any signal", () => {
    assert.strictEqual(
      classifyExitForRestart(null, "SIGKILL", false),
      "crash",
    );
    assert.strictEqual(
      classifyExitForRestart(null, "SIGTERM", false),
      "crash",
    );
  });

  test("classifyExitForRestart returns cleanExit on code=0 signal=null intentional=false", () => {
    assert.strictEqual(classifyExitForRestart(0, null, false), "cleanExit");
  });
```

- [ ] **Step 3: Compile RED**

Expected: missing-export errors for `formatExitReason`, `classifyExitForRestart`, `ExitDisposition`.

- [ ] **Step 4: Implementation**

Append after `shouldAttemptRestart`:

```typescript
/** Outcome bucket for a backend exit event. */
export type ExitDisposition = "intentional" | "crash" | "cleanExit";

/**
 * Render a human-readable description of a Node child-process exit
 * event. Signal takes priority because a signal-kill carries more
 * diagnostic information than the resulting exit code.
 *
 * Examples:
 * - `(0, null)`        → "exited cleanly (code 0)"
 * - `(1, null)`        → "crashed (exit code 1)"
 * - `(null, "SIGKILL")` → "killed by signal SIGKILL"
 * - `(null, null)`     → "exited (unknown reason)"
 *
 * Pure: no I/O.
 */
export function formatExitReason(
  code: number | null,
  signal: NodeJS.Signals | null,
): string {
  if (signal) {
    return `killed by signal ${signal}`;
  }
  if (code === null) {
    return "exited (unknown reason)";
  }
  if (code === 0) {
    return "exited cleanly (code 0)";
  }
  return `crashed (exit code ${code})`;
}

/**
 * Classify a backend exit event so the lifecycle controller can decide
 * whether to surface the crash dialog. If the controller flagged the
 * exit as intentional (because `stopBackend` or `restartBackend` was
 * just invoked), always return "intentional". Otherwise a non-zero
 * code or any signal counts as a crash; code=0 + signal=null is a
 * clean exit. Pure.
 */
export function classifyExitForRestart(
  code: number | null,
  signal: NodeJS.Signals | null,
  intentional: boolean,
): ExitDisposition {
  if (intentional) {
    return "intentional";
  }
  if (signal !== null) {
    return "crash";
  }
  if (code !== null && code !== 0) {
    return "crash";
  }
  return "cleanExit";
}
```

- [ ] **Step 5: Compile GREEN + lint + pyright** (all SUCCESS).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add formatExitReason + classifyExitForRestart helpers — iter 48 step C

Two pure helpers for the backend-lifecycle controller:

- formatExitReason(code, signal) renders human-readable text for the
  crash dialog (signal takes priority over code; explicit fallback
  for both-null).
- classifyExitForRestart(code, signal, intentional) buckets the
  exit event into intentional | crash | cleanExit. The intentional
  flag is owned by stopBackend / restartBackend; a non-zero code or
  any signal otherwise counts as a crash.

10 unit tests cover the format matrix (clean, non-zero, signal,
both-set, both-null) and the classification matrix (intentional
short-circuit + crash + clean-exit branches).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire helpers into the backend lifecycle controller

**Goal:** Use Tasks 1-3 plus a new `backendIntentionalStop` flag to implement the crash dialog + auto-restart loop. Add reason prompt to `restartBackend`. Add Output-channel log to `findAvailablePort` showing which port was chosen when the preferred port was unavailable. No new tests this task — coverage is via the pure-function tests in Tasks 1-3 and F5 smoke in Task 5.

**Files:**
- Modify: `extension/src/extension.ts` (global state additions; `child.on('exit')` rewrite; `stopBackend` + `restartBackend` updates; new `handleUnexpectedExit` + `attemptAutoRestart` functions; `findAvailablePort` log line)

- [ ] **Step 1: Add new global state declarations**

In the `// GLOBAL STATE` section (currently lines 110-118), append two new declarations after the existing `let backendStarting: boolean = false;` line:

```typescript
/** Flag set by stopBackend / restartBackend so the child.on('exit') handler does not interpret the planned shutdown as a crash. Reset to false at the start of every startBackend invocation. */
let backendIntentionalStop: boolean = false;

/** Number of consecutive auto-restart attempts since the last successful boot. Reset to 0 when the backend reaches the ready state. Bounded by shouldAttemptRestart. */
let backendRestartAttempts: number = 0;
```

- [ ] **Step 2: Update `startBackend` to reset the new flags**

Inside `startBackend(context)` (currently lines 272-398), find the existing block that sets `backendStarting = true;` and `updateStatusBar("starting");` near line 279-281. Add ONE line directly after `backendStarting = true;`:

```typescript
backendIntentionalStop = false;
```

Then find the block where the backend successfully reports ready (currently around lines 380-386 where it logs "Backend server is ready!" and sets `isBackendReady = true;`). Add ONE line directly after `isBackendReady = true;`:

```typescript
backendRestartAttempts = 0;
```

- [ ] **Step 3: Rewrite the `child.on('exit')` handler**

Currently at extension.ts lines 357-363 the handler looks like:

```typescript
backendProcess.on("exit", (code) => {
  log(`Backend process exited with code ${code}`);
  isBackendReady = false;
  backendStarting = false;
  backendProcess = null;
  updateStatusBar("inactive");
});
```

Replace it with:

```typescript
backendProcess.on("exit", (code, signal) => {
  const reason = formatExitReason(code, signal);
  const disposition = classifyExitForRestart(
    code,
    signal,
    backendIntentionalStop,
  );
  log(`Backend process ${reason} (disposition: ${disposition}).`);
  isBackendReady = false;
  backendStarting = false;
  backendProcess = null;
  if (disposition === "crash") {
    updateStatusBar("error");
    void handleUnexpectedExit(context, reason);
  } else {
    updateStatusBar("inactive");
  }
});
```

The `context` parameter is already in scope because the handler lives inside `startBackend(context)`.

- [ ] **Step 4: Update `stopBackend` to flag intentional stops**

Currently `stopBackend()` (lines 428-437) is:

```typescript
function stopBackend() {
  if (backendProcess) {
    log("Stopping backend server...");
    backendProcess.kill();
    backendProcess = null;
    isBackendReady = false;
    backendStarting = false;
    updateStatusBar("inactive");
  }
}
```

Replace with:

```typescript
function stopBackend() {
  if (backendProcess) {
    log("Stopping backend server...");
    backendIntentionalStop = true;
    backendProcess.kill();
    backendProcess = null;
    isBackendReady = false;
    backendStarting = false;
    updateStatusBar("inactive");
  }
}
```

- [ ] **Step 5: Update `restartBackend` to prompt for a reason**

Currently `restartBackend(context)` (lines 442 onward) is:

```typescript
async function restartBackend(context: vscode.ExtensionContext) {
  stopBackend();
  await sleep(1000);
  const success = await startBackend(context);
  if (success) {
    vscode.window.showInformationMessage(
      "DDD Enforcer: Backend restarted successfully!",
    );
  } else {
    vscode.window.showErrorMessage(
      "DDD Enforcer: Failed to restart backend",
    );
  }
}
```

Replace with:

```typescript
async function restartBackend(context: vscode.ExtensionContext) {
  const reason = await vscode.window.showInputBox({
    prompt: "Reason for restarting the DDD Enforcer backend (optional)",
    placeHolder: "e.g. backend logs went silent, want a clean slate, ...",
    ignoreFocusOut: true,
  });
  if (reason && reason.trim()) {
    log(`Manual restart requested. Reason: ${reason.trim()}`);
  } else {
    log("Manual restart requested. (No reason supplied.)");
  }
  stopBackend();
  await sleep(1000);
  const success = await startBackend(context);
  if (success) {
    vscode.window.showInformationMessage(
      "DDD Enforcer: Backend restarted successfully!",
    );
  } else {
    vscode.window.showErrorMessage(
      "DDD Enforcer: Failed to restart backend",
    );
  }
}
```

If the user dismisses the input box (returns `undefined`), the restart still proceeds — the log line just records "No reason supplied". This matches the spec acceptance criterion *"Manual `ddd-enforcer.restartBackend` command prompts for a reason string (stored in extension log for debugging)"* — the prompt is offered; supplying a reason is optional.

- [ ] **Step 6: Add the new `handleUnexpectedExit` function**

Add this function definition immediately AFTER the existing `restartBackend` function (so the source order is `startBackend` → `waitForBackend` → `stopBackend` → `restartBackend` → `handleUnexpectedExit` → `attemptAutoRestart`):

```typescript
/**
 * Surface the crash dialog after the backend exited with a "crash"
 * disposition. Three buttons: "Restart automatically", "Show logs",
 * "Cancel". Yes triggers attemptAutoRestart; "Show logs" reveals the
 * Output channel; Cancel sets the status to "error" and exits.
 */
async function handleUnexpectedExit(
  context: vscode.ExtensionContext,
  reason: string,
): Promise<void> {
  const choice = await vscode.window.showWarningMessage(
    `DDD Enforcer backend ${reason}. Restart automatically?`,
    "Restart automatically",
    "Show logs",
    "Cancel",
  );
  if (choice === "Restart automatically") {
    backendRestartAttempts = 0;
    await attemptAutoRestart(context);
  } else if (choice === "Show logs") {
    outputChannel.show();
    log(
      "User chose 'Show logs' after backend crash. No restart attempted.",
    );
  } else {
    log(
      "User declined auto-restart after backend crash. Use 'DDD Enforcer: Restart Backend Server' to retry manually.",
    );
  }
}
```

- [ ] **Step 7: Add the new `attemptAutoRestart` function**

Add immediately AFTER `handleUnexpectedExit`:

```typescript
/**
 * Auto-restart loop with exponential backoff (computeBackoffMs).
 * Caps at shouldAttemptRestart's default 5 attempts. On final failure,
 * surfaces a persistent error toast and stops the loop — does NOT
 * spawn indefinitely.
 */
async function attemptAutoRestart(
  context: vscode.ExtensionContext,
): Promise<void> {
  while (shouldAttemptRestart(backendRestartAttempts)) {
    const delayMs = computeBackoffMs(backendRestartAttempts);
    log(
      `Auto-restart attempt ${backendRestartAttempts + 1}/5 in ${delayMs}ms...`,
    );
    await sleep(delayMs);
    backendRestartAttempts += 1;
    const success = await startBackend(context);
    if (success) {
      log("Auto-restart succeeded.");
      vscode.window.showInformationMessage(
        "DDD Enforcer: Backend restarted automatically.",
      );
      return;
    }
    log(`Auto-restart attempt ${backendRestartAttempts}/5 failed.`);
  }
  log(
    "Auto-restart gave up after 5 failed attempts. Use 'DDD Enforcer: Restart Backend Server' to retry manually.",
  );
  vscode.window.showErrorMessage(
    "DDD Enforcer: Backend could not be restarted automatically after 5 attempts. Open the Output channel for details and use 'DDD Enforcer: Restart Backend Server' once the underlying issue is fixed.",
  );
  updateStatusBar("error");
}
```

- [ ] **Step 8: Add Output-channel log to `findAvailablePort`**

Currently `findAvailablePort()` (lines 1652-1670) is:

```typescript
async function findAvailablePort(): Promise<number> {
  const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
  const preferredPort = cfg.get<number>("backendPort", 8000);

  if (await isPortAvailable(preferredPort)) {
    return preferredPort;
  }

  for (let port = preferredPort + 1; port < preferredPort + 100; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }

  return preferredPort;
}
```

Replace with:

```typescript
async function findAvailablePort(): Promise<number> {
  const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
  const preferredPort = cfg.get<number>("backendPort", 8000);

  if (await isPortAvailable(preferredPort)) {
    return preferredPort;
  }

  log(
    `Preferred port ${preferredPort} is in use. Scanning for an available port in the next 99 candidates...`,
  );
  for (let port = preferredPort + 1; port < preferredPort + 100; port++) {
    if (await isPortAvailable(port)) {
      log(`Selected port ${port} (preferred port ${preferredPort} was unavailable).`);
      return port;
    }
  }

  log(
    `WARNING: no available port found in ${preferredPort}..${preferredPort + 99}. Falling back to preferred port ${preferredPort} (backend startup is likely to fail).`,
  );
  return preferredPort;
}
```

- [ ] **Step 9: Compile + lint + pyright + pytest**

Run:
- `cd extension && npm run compile && npm run lint`
- `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright`
- `cd extension/backend && pytest -m "not integration" -q`

Expected: compile + lint SUCCESS; pyright `0 errors, 0 warnings, 0 informations`; pytest `729 passed, 31 deselected`.

- [ ] **Step 10: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts
git commit -m "$(cat <<'EOF'
feat(extension): integrate backend-crash dialog + auto-restart loop — iter 48 step D

WP-CORE-28 Feature 2 wired end-to-end:

- Two new module-level flags: backendIntentionalStop (set by
  stopBackend / restartBackend so the exit handler distinguishes
  planned shutdown from crash) and backendRestartAttempts (counter
  reset on every successful boot and bounded by shouldAttemptRestart).
- child.on('exit') now classifies the event via classifyExitForRestart
  + formatExitReason. On 'crash' disposition it sets status='error'
  and delegates to handleUnexpectedExit.
- handleUnexpectedExit shows a Warning toast with 3 buttons (Restart
  automatically / Show logs / Cancel). 'Restart automatically' starts
  attemptAutoRestart.
- attemptAutoRestart loops with exponential backoff (1s, 2s, 4s,
  8s, 16s) up to 5 attempts. Success surfaces an info toast; final
  failure surfaces a persistent error toast and stops looping.
- restartBackend now prompts for a reason via showInputBox (optional;
  empty/dismiss is fine) and logs it to the Output channel for
  debugging.
- findAvailablePort logs to the Output channel which port was selected
  when the preferred port was unavailable, plus a WARNING when no port
  in the 100-candidate scan is free.

No backend changes. Pyright still 0 errors. Pytest still 729 passing.

Closes Iter 48 of WP-CORE-28.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: F5 manual smoke checklist (HUMAN-IN-LOOP)

**Goal:** Verify the wired behavior in the Extension Development Host. WP-CORE-28 spec requires per-iter F5 smoke. Iter 47's smoke was waived by the user mid-session; Iter 48 will offer the checklist again and let the user choose. Either way, the loop stops here — Iter 49 does NOT start until the user signals continuation.

**Files:**
- Modify: `.planning/pipeline_audit/CURRENT.md` after the user reports back (or explicitly waives smoke).

- [ ] **Step 1: Halt and post the smoke checklist**

After Task 4 commits, post a Turkish caveman-mode message containing the checklist below. Wait. Do not proceed to Iter 49.

**F5 smoke checklist for Iter 48 (Backend lifecycle resilience):**

Compile first: `cd extension && npm run compile`. Then open `extension/` in VS Code, press F5 for the Extension Development Host.

1. **Test A — clean start.** Open a Python file in the dev host. Trigger any DDD command. Confirm the backend boots, status reaches "ready". Watch the Output channel — note the port chosen.
2. **Test B — port collision.** While the dev host's backend is running, in a separate terminal `python3 -m http.server <preferredPort>` (bind whatever port shows in step A). Then in the dev host run `DDD Enforcer: Restart Backend Server` (you'll be prompted for a reason — type `port collision smoke` and submit). Confirm: Output channel shows `Preferred port N is in use. Scanning...` and `Selected port N+1`. Backend reaches "ready" on the new port. Then `kill` the rogue http.server.
3. **Test C — crash dialog.** In the dev host's terminal (View → Terminal inside the dev host), find the backend PID via `ps aux | grep uvicorn`, then `kill -9 <PID>`. Within ~2 seconds the dev host should display a Warning toast: `DDD Enforcer backend killed by signal SIGKILL. Restart automatically?` with 3 buttons. Confirm the toast appears.
4. **Test D — auto-restart success.** On the toast from Test C, click `Restart automatically`. Within ~5 seconds the backend should be back up, status returns to "ready", and an info toast `Backend restarted automatically` shows. Output channel should record `Auto-restart attempt 1/5 in 1000ms...` then `Auto-restart succeeded.`
5. **Test E — backoff cap.** Make the backend fail-to-boot: in the workspace, set `ddd-enforcer.pythonPath` to a non-existent path (e.g., `/nonexistent/python`). Trigger a DDD command — startBackend fails. `kill -9` the backend if it somehow lingers, then in the Output channel verify either (a) the spawn error path or (b) the crash dialog. If the crash dialog appears, click `Restart automatically`. The loop should attempt 5 times with delays 1s / 2s / 4s / 8s / 16s (visible in the Output channel) and finally surface a persistent error toast. Confirm no further attempts happen after the 5th failure. Reset `ddd-enforcer.pythonPath` to `python3` afterward.
6. **Test F — Show logs button.** Reproduce Test C (kill -9 the running backend). When the toast appears, click `Show logs`. The Output channel should focus / become visible. Confirm.
7. **Test G — Cancel button.** Reproduce Test C. When the toast appears, click `Cancel`. Status should stay at "error"; no auto-restart should fire. Confirm.

If any step fails, post the failure output (toast text, Output-channel excerpt). The implementer agent will RED → GREEN → COMMIT a fix.

- [ ] **Step 2: Update CURRENT.md (after user reports green OR explicitly waives)**

Append a new `## Iteration 48 — WP-CORE-28 Feature 2 (Backend lifecycle resilience) COMPLETE` section in the same shape as the Iter-47 section already there. Include: 4 commits + 1 fix-if-any, ~22 new test cases, F5 smoke outcome (passed / waived).

- [ ] **Step 3: Commit the CURRENT.md update**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add .planning/pipeline_audit/CURRENT.md
git commit -m "$(cat <<'EOF'
chore(planning): CURRENT.md update for iter 48 (WP-CORE-28 Feature 2 complete)

Backend lifecycle resilience shipped.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (per writing-plans skill)

**1. Spec coverage** (against `todos/WP-CORE-28-extension-ux-wave1.md` Feature 2 acceptance):

| Acceptance criterion | Task |
|----------------------|------|
| `kill -9` backend → extension dialog within 2 seconds | Task 4 Step 3 (`child.on('exit')` → `handleUnexpectedExit`) + Test C |
| User-accepted auto-restart succeeds + health-check within 5 seconds | Task 4 Step 7 (`attemptAutoRestart` first attempt delay = 1s) + Test D |
| Port collision triggers port scan; extension reports new port via Output panel | Task 4 Step 8 (`findAvailablePort` log additions) + Test B |
| After 5 failed restart attempts: persistent error, no infinite loop | Task 4 Step 7 (`shouldAttemptRestart` cap + persistent error toast + status "error") + Test E |
| Manual restart prompts for reason, logged | Task 4 Step 5 (`restartBackend` showInputBox + log) + Test B |
| 3 new test cases | Substituted with 22 pure-function tests (Tasks 1-3) — documented in Pre-flight; F5 smoke covers integration |

**2. Placeholder scan:** no TBD, no "handle appropriately", no "similar to Task N". All steps have complete code.

**3. Type consistency:**
- `ExitDisposition` (Task 3) → consumed in `classifyExitForRestart` return → consumed by `child.on('exit')` handler in Task 4 Step 3.
- `computeBackoffMs` + `shouldAttemptRestart` (Tasks 1-2) → consumed by `attemptAutoRestart` in Task 4 Step 7.
- `formatExitReason` (Task 3) → consumed by `child.on('exit')` handler in Task 4 Step 3 (passes the rendered string to `handleUnexpectedExit`).
- Module-level globals `backendIntentionalStop` + `backendRestartAttempts` (Task 4 Step 1) → reset in `startBackend` (Step 2) + set in `stopBackend` (Step 4) + read in `child.on('exit')` (Step 3) + read+written in `attemptAutoRestart` (Step 7) + reset in `handleUnexpectedExit` (Step 6).

All cross-task names match. Plan is consistent.

End of plan.
