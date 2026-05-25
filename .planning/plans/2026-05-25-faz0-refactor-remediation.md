# Faz 0 Refactor — Remediation Plan

**Date:** 2026-05-25
**Scope:** Bug/security findings from reviewing the Faz 0 modular refactor (7 new modules extracted from `extension/src/extension.ts`).
**Status of refactor:** functionally sound — `npm run compile`, `npm run lint`, and `npm test` (`113 passing`) all verified green; backward-compat re-exports real; zero regression confirmed. This plan fixes **latent bugs the refactor preserved** plus **new coupling it introduced** (circular import).
**Reviewer:** Claude Opus 4.7 + 3 delegated `agy` (antigravity) subagents (cluster-based), findings filtered for false-positives/overrates.

> **COMPLETION STATUS (2026-05-25): ALL 15 TASKS DONE.**
> - T1 `067f033`, T2 `257b106`, T3 `9619154`, T4 `882d00a`, T5 `e5d35bc`, T6 `e566ee8`, T7 `e3f0f65`, T8 `2fa3c1f`, T9 `b934033`, T12 `a859706`, T13 `24d7448` (+ `e2532b3` keyword-escape follow-up), T14 (guard shipped in tree), T15 `3c0287f`.
> - **T10 + T11** (GROUP C, was NEEDS-DECISION): resolved via **option (a) fix**, executed through a full brainstorm→spec→plan→SDD cycle as a consolidated two-phase rewrite of `normalizePythonSemantics`. Commits `f4c39a0`, `10e1cce`, `ed37c6e`, `0496716`, `96f46b6` (lone-CR false-equal fix from final review). Spec: `docs/superpowers/specs/2026-05-25-fingerprint-indent-string-hardening-design.md`; Plan: `docs/superpowers/plans/2026-05-25-fingerprint-indent-string-hardening.md`.
> - Suite: 113 → 147 passing. All shipped to `origin/main` (HEAD `96f46b6`).

---

## How to use this document

Each task (`T1`–`T15`) is **self-contained**: file, exact location, current→target code, tests, acceptance. A subagent should execute one task without re-investigating.

**Global gates — run after EVERY task; all must stay green:**
```bash
cd extension && npm run compile && npm run lint && npm test
# expect: tsc exit 0, eslint exit 0, "113 passing" + any new tests added by the task
```

**Conventions (repo rules — non-negotiable):**
- **TDD:** write the test first in `extension/src/test/extension.test.ts` (import the symbol `from "../extension"`, which re-exports all module helpers), then implement.
- **Atomic commits**, Conventional Commits style, one commit per task, with trailer:
  ```
  Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
  ```
- **Smallest correct change.** No drive-by refactors, no renames, no formatter changes.
- If a task's gate fails and can't be fixed within the task's scope, **revert the task** and report — do not weaken gates.

**File-collision rule (critical for parallelism):** tasks touching the same file CANNOT run in parallel. See the Dispatch Plan at the bottom.

---

## GROUP A — `extension/src/apiKeyManager.ts` (6 tasks, run in order T1→T2→T3→T4→T5→T6)

### T1 — 🔴 HIGH — Store key only AFTER validation (fix lockout)
**File:** `src/apiKeyManager.ts`, `getApiKey()` (~lines 178–217)
**Problem:** The prompt branch stores the typed key to secret storage (~line 188) BEFORE the validation probe (~line 201). A bad key persists; next launch the secret-source hit short-circuits the prompt → re-validates the same bad key → fails → no UI path to re-enter. Soft-lockout.
**Change:**
1. In the prompt branch, **delete** `await context.secrets.store("geminiApiKey", inputKey.trim());`. Leave only:
```ts
    if (inputKey && inputKey.trim()) {
      apiKey = inputKey.trim();
      source = "prompt";
    }
```
2. After validation succeeds (right after `log("Gemini API key validated.");`, ~line 219), add:
```ts
  // Persist a prompted key only after it validates (prevents lockout on a bad key).
  if (source === "prompt") {
    await context.secrets.store("geminiApiKey", apiKey);
  }
```
**Test:** `getApiKey` depends on vscode (not pure). Minimum: confirm `npm test` stays green; add a code comment documenting the invariant. Optional: a scenario test using `validateGeminiKey`'s injectable probe.
**Acceptance:** a rejected key is NOT written to secret storage; a validated prompted key IS.

### T2 — 🟠 MED — Clear key from ALL config targets (workspace included)
**File:** `src/apiKeyManager.ts`, "Move to secret storage" branch (~lines 235–248)
**Problem:** only `ConfigurationTarget.Global` is cleared; a key in workspace `.vscode/settings.json` stays in plaintext and re-triggers the migration prompt.
**Change:** replace the single `cfg.update(... Global)` call inside the `source === "settings"` block with:
```ts
      if (source === "settings") {
        const inspected = cfg.inspect<string>("geminiApiKey");
        const targets: vscode.ConfigurationTarget[] = [];
        if (inspected?.globalValue) targets.push(vscode.ConfigurationTarget.Global);
        if (inspected?.workspaceValue) targets.push(vscode.ConfigurationTarget.Workspace);
        if (inspected?.workspaceFolderValue) targets.push(vscode.ConfigurationTarget.WorkspaceFolder);
        for (const t of targets) {
          try {
            await cfg.update("geminiApiKey", undefined, t);
          } catch (err) {
            settingsClearFailed = true;
            log(`API key copied to secret storage but failed to clear settings target ${t}: ${err instanceof Error ? err.message : String(err)}`);
          }
        }
      }
```
Note: `undefined` removes the key (`""` would leave an empty string).
**Acceptance:** both global and workspace entries cleared; failure sets `settingsClearFailed`.

### T3 — 🟠 MED — Break the env-source migration re-offer loop
**File:** `src/apiKeyManager.ts`, `decideMigrationOffer` (~lines 117–134) + `getApiKey` migration block
**Problem:** precedence is settings > env > secret. An env key migrated to secret can't clear the env var; env outranks secret → migration is re-offered every launch.
**Change (minimal — separate suppression flag):**
1. In `getApiKey`, where `migrationDeclined` is read, add:
```ts
  const envMigrationDone =
    context.globalState.get<boolean>("apiKeyEnvMigrationDone") === true;
```
2. Add a third param to `decideMigrationOffer`:
```ts
export function decideMigrationOffer(
  source: ApiKeySource,
  migrationDeclined: boolean,
  envMigrationDone: boolean = false,
): MigrationDecision {
  const labels: Record<ApiKeySource, string> = { /* unchanged */ };
  if (migrationDeclined) return { shouldOffer: false, sourceLabel: labels[source] };
  if (source === "env" && envMigrationDone) return { shouldOffer: false, sourceLabel: labels[source] };
  if (source === "settings" || source === "env") return { shouldOffer: true, sourceLabel: labels[source] };
  return { shouldOffer: false, sourceLabel: labels[source] };
}
```
Pass `envMigrationDone` at the call site.
3. On successful "Move to secret storage" for an env key, set the flag:
```ts
      if (source === "env") {
        await context.globalState.update("apiKeyEnvMigrationDone", true);
        log("Env-sourced API key migrated to secret storage; suppressing future env migration offers. Unset GEMINI_API_KEY to let secret storage take over.");
      }
```
4. Append to the env migration toast text: "Note: unset GEMINI_API_KEY afterward so secret storage takes precedence."
**Test (`decideMigrationOffer` is PURE):**
- `decideMigrationOffer("env", false, true)` → `shouldOffer === false`
- `decideMigrationOffer("env", false, false)` → `shouldOffer === true`
- `decideMigrationOffer("settings", false, true)` → `shouldOffer === true` (settings unaffected by env flag)
**Acceptance:** env migration offered once, then suppressed.

### T4 — 🟠 MED — `classifyApiKeyError` covers modern axios codes
**File:** `src/apiKeyManager.ts:34-39` (`networkCodes` set)
**Change:**
```ts
  const networkCodes = new Set([
    "ENOTFOUND",
    "ECONNABORTED",
    "ECONNREFUSED",
    "ETIMEDOUT",
    "ERR_NETWORK",
    "ERR_BAD_RESPONSE",
    "EAI_AGAIN",
  ]);
```
**Test (`classifyApiKeyError` is PURE):**
- `classifyApiKeyError({ code: "ERR_NETWORK" })` → `"network_error"`
- `classifyApiKeyError({ code: "EAI_AGAIN" })` → `"network_error"`
**Acceptance:** new codes classify as `network_error`.

### T5 — 🟡 LOW — Send API key in header, not URL query
**File:** `src/apiKeyManager.ts:79-87` (default `httpProbe`)
**Problem:** `?key=` in the URL can leak via `AxiosError.config.url` if logged elsewhere.
**Change:**
```ts
  const probe: ApiKeyHttpProbe =
    httpProbe ??
    (async (url) => {
      const resp = await axios.get(url, {
        timeout: 5000,
        headers: { "x-goog-api-key": trimmed },
      });
      return { status: resp.status, data: resp.data };
    });
  try {
    const url = GEMINI_MODELS_URL_BASE; // key now in header, not query
    const { status } = await probe(url);
```
Remove the `?key=...` from the URL construction.
**Test:** existing `validateGeminiKey` tests use the injectable probe (signature unchanged) → must stay green.
**Acceptance:** default probe sends key via `x-goog-api-key` header; no key in URL.

### T6 — 🟠 MED — Break circular import (dependency injection)
**File:** `src/apiKeyManager.ts:8` + `src/extension.ts:373`
**Problem:** `apiKeyManager.ts` imports `{ log, updateStatusBar } from "./extension"` while `extension.ts` imports from `apiKeyManager` — circular. Couples a leaf module back into the monolith.
**Change:**
1. **Delete** line 8 of `apiKeyManager.ts` (`import { log, updateStatusBar } from "./extension";`).
2. Add a deps param to `getApiKey`:
```ts
export interface ApiKeyDeps {
  log: (msg: string) => void;
  updateStatusBar: (state: "validatingApiKey") => void;
}

export async function getApiKey(
  context: vscode.ExtensionContext,
  deps: ApiKeyDeps,
): Promise<string | undefined> {
```
Replace all `log(...)` → `deps.log(...)` and `updateStatusBar("validatingApiKey")` → `deps.updateStatusBar("validatingApiKey")` in the function body.
3. Update the call site `extension.ts:373`:
```ts
    const apiKey = await getApiKey(context, { log, updateStatusBar });
```
**Acceptance:** `grep 'from "./extension"' src/apiKeyManager.ts` returns nothing; compile + test green. **Do this AFTER T1–T5** so the signature change doesn't churn the other edits.

---

## GROUP B — `extension/src/backend/processManager.ts` (3 tasks, order T7→T8→T9)

### T7 — 🟠 MED — `classifyExitForRestart(null, null)` → "crash"
**File:** `src/backend/processManager.ts:72-87`
**Problem:** code=null && signal=null (silent death / spawn anomaly) returns "cleanExit"; should be "crash" (no restart, no crash dialog otherwise).
**Change:** replace the tail of the function:
```ts
  if (signal !== null) {
    return "crash";
  }
  // signal === null past this point; only exit code 0 is a clean exit.
  return code === 0 ? "cleanExit" : "crash";
```
(Delete the intermediate `if (code !== null && code !== 0) return "crash";` — the new return subsumes it.)
**Test (PURE):**
- `classifyExitForRestart(null, null, false)` → `"crash"`
- `classifyExitForRestart(0, null, false)` → `"cleanExit"`
- `classifyExitForRestart(1, null, false)` → `"crash"`
- `classifyExitForRestart(null, null, true)` → `"intentional"`

### T8 — 🟠 MED — Port scan must not exceed 65535
**File:** `src/backend/processManager.ts:121`
**Problem:** loop runs to `preferredPort + 100`; if `preferredPort ≳ 65436`, `net.listen(>65535)` throws a synchronous RangeError → unhandled rejection crashes port discovery.
**Change:**
```ts
  const maxPort = Math.min(preferredPort + 100, 65535);
  for (let port = preferredPort + 1; port < maxPort; port++) {
    if (await isPortAvailable(port)) {
```
Update log messages referencing `preferredPort + 99` to use `maxPort - 1` for consistency.
**Test:** `findAvailablePort`/`isPortAvailable` are net-bound (not pure) → minimum compile+test green.

### T9 — 🟡 LOW — `computeBackoffMs` NaN guard
**File:** `src/backend/processManager.ts:18-28`
**Change:** before `if (attempt <= 0)`:
```ts
  if (!Number.isFinite(attempt) || attempt <= 0) {
    return baseMs;
  }
```
**Test (PURE):** `computeBackoffMs(NaN)` → `1000` (baseMs).

---

## GROUP C — `extension/src/semanticFingerprint.ts` (✅ RESOLVED — option (a) fix shipped; see COMPLETION STATUS above)

### T10 — 🟠 MED — `[NEEDS-DECISION]` Does the fingerprint preserve Python indentation?
**Problem:** `normalizePythonSemantics` strips ALL whitespace outside strings → indentation lost → moving a statement into/out of a block collapses to the same fingerprint → validation SKIPPED on a genuine semantic change.
**Example:**
```python
if x:          if x:
    a()            a()
b()                b()   # moved inside/outside the block
```
Both normalize to `ifx:a()b()`.
**Decision required:**
- **(a) Fix:** preserve `\n` + leading indentation; collapse only intra-line whitespace. This is a normalizer rewrite (newline as sentinel, leading-whitespace-aware). Medium effort; behavior change → existing fingerprint tests must be updated. **Request the detailed rewrite spec from the reviewer before implementing.**
- **(b) Accept + document:** add a module comment "indentation-only changes are intentionally treated as non-semantic" and a limitation note in `development_docs/`. Zero code.
**Do not dispatch until (a)/(b) is chosen.**

### T11 — 🟠 MED — `[NEEDS-DECISION]` Fingerprint hand-parser edge cases
**Problem:** the hand-rolled string-state parser has holes: no escape tracking inside triple-quoted strings; a quote adjacent to the closing delimiter causes a premature close + state inversion (corrupts the rest of the file's fingerprint); Python 3.12 f-string quote-reuse inside `{}` (e.g. `f"...{x.split(" ")}..."`) closes the string early. **Python 3.12 is the locked D1 version → topical.**
**Decision required:**
- **(a) Targeted patches:** track `escaped` inside triple-string branches; maintain a brace-depth counter for f-strings. Medium-high effort, needs many edge-case tests.
- **(b) Accept + document:** these are low-frequency; a mis-fingerprint only causes over/under-validation (not a correctness/security defect). Document as a known limitation.
**Recommendation:** evaluate T10+T11 together (same file). If (a) is chosen, do a single consolidated "fingerprint hardening" task with extensive tests. **Needs approval before dispatch.**

---

## GROUP D — `extension/src/ui/hoverProvider.ts` + `extension/src/extension.ts` (2 tasks, order T12→T13)

### T12 — 🔴 HIGH — Trusted-markdown command-link injection (PRIMARY: allowlist)
**File:** `src/extension.ts:1428`
**Problem:** `markdown.isTrusted = true` wraps `formatHoverMarkdown(violation, ...)` output, which contains LLM-derived `violation.message`/`source.summary`. Trusted markdown renders `command:` links as clickable → a prompt-injected SRS → LLM emits a malicious command link → user click executes it.
**Change (the actual security control, one line):**
```ts
    markdown.isTrusted = { enabledCommands: ["ddd-enforcer.openSource"] };
```
Only `ddd-enforcer.openSource` command links can fire; any injected `command:` URI is inert.
**Test:** not pure (vscode.MarkdownString). Compile + test green; manual F5 note: a message containing `[x](command:workbench.action.terminal.sendSequence?...)` is not clickable in the hover.
**Acceptance:** `enabledCommands` allowlist set.

### T13 — 🟡 LOW — Defense-in-depth: escape hover fields + `LruCache` NaN guard
**File:** `src/ui/hoverProvider.ts`
**Part A (escape):** add a helper and apply it in `formatHoverMarkdown` (T12's allowlist closes the RCE; this is for render correctness):
```ts
function escapeInlineMarkdown(s: string): string {
  return s.replace(/[\\`<>\[\]]/g, "\\$&");
}
```
Apply to `violation.type`, `violation.message`, `source.section`, `source.document`. For the excerpt: order `truncateExcerpt` → `escapeInlineMarkdown` → `boldMatchingSpan` (so the `**` that bold adds survives; keyword match on escaped text is acceptable/approximate).
**Part B (LruCache NaN):** `src/ui/hoverProvider.ts:18-22` constructor guard:
```ts
    if (!Number.isFinite(capacity) || capacity < 1) {
      throw new Error("LruCache capacity must be a finite number >= 1");
    }
```
**Test:**
- if `escapeInlineMarkdown` is exported: a message containing `[x](command:evil)` is escaped in the output.
- `new LruCache(NaN)` throws.
- existing `formatHoverMarkdown` tests (4) must still pass — escaping must not garble readable text; update expectations if needed.

---

## GROUP E — `extension/src/utils/progress.ts` (1 task)

### T14 — 🟡 LOW — `computeEtaMs` NaN guard
**File:** `src/utils/progress.ts:117-130`
**Problem:** a NaN `overallPercent` propagates to `formatEta` → garbage `"NaNhNaNm"` in the status bar.
**Change:** before `if (overallPercent <= 0)`:
```ts
  if (!Number.isFinite(overallPercent) || overallPercent <= 0) {
    return null;
  }
```
**Test (PURE):** `computeEtaMs(5000, NaN)` → `null`; `computeEtaMs(5000, Infinity)` → `null` (consistent — `Number.isFinite` catches both).

---

## GROUP F — `extension/src/ui/runManifestsWebview.ts` (1 task)

### T15 — 🟡 LOW — `generateNonce` uses crypto-random
**File:** `src/ui/runManifestsWebview.ts:125-133`
**Problem:** `Math.random()` nonce is predictable (low risk for a local webview with `default-src 'none'`, but a trivial hardening).
**Change:**
```ts
import * as crypto from "crypto";

export function generateNonce(): string {
  return crypto.randomBytes(32).toString("base64").replace(/[^A-Za-z0-9]/g, "").slice(0, 32);
}
```
**Test:** existing 2 nonce tests (32-char alphanumeric + different each call) must pass. `randomBytes(32)` base64 ≈ 43 chars; after stripping non-alphanumerics and slicing to 32, length is guaranteed 32.
**Acceptance:** output is 32 chars, alphanumeric, crypto-random.

---

## DISPATCH PLAN

**Parallel workstreams (file-isolated, no collisions):**

| Workstream | File | Tasks (sequential within WS) | Notes |
|---|---|---|---|
| WS-A | `apiKeyManager.ts` (+1 line in extension.ts) | T1→T2→T3→T4→T5→**T6** | SERIAL within WS |
| WS-B | `processManager.ts` | T7→T8→T9 | SERIAL within WS |
| WS-C | `semanticFingerprint.ts` | T10, T11 | ⛔ decide first |
| WS-D | `hoverProvider.ts` (+1 line in extension.ts) | T12→T13 | SERIAL within WS |
| WS-E | `progress.ts` | T14 | single |
| WS-F | `runManifestsWebview.ts` | T15 | single |

**Collision warning:** WS-A (T6) and WS-D (T12) both touch `extension.ts` (different regions, ~line 373 vs ~1428). Still the same file → **serialize the `extension.ts` edits of WS-A and WS-D** (one commits, the other rebases), OR assign both extension.ts edits to a single agent.

**Recommended order:**
1. **First:** T12 (HIGH security, one line) + T1 (HIGH lockout) — shipping blockers.
2. **Parallel:** WS-B, WS-E, WS-F (fully isolated).
3. **Serial:** remaining WS-A (T2–T6), then WS-D (T13).
4. **Last / optional:** WS-C (T10/T11) — needs a decision; can be deferred/documented.

**One task = one atomic commit.** After each commit, `cd extension && npm run compile && npm run lint && npm test` must be green; otherwise revert that task.

---

## Findings explicitly REJECTED (agy false-positives / overrates — do NOT action)
- `isPortAvailable` "listener leak" — `.once` self-removes, server is GC'd → non-issue.
- line-continuation `\` appended outside strings — actually correct for a fingerprint (joins lines).
- `renderDetail` missing `suggestion`/`srs_path` — feature gap, not a bug.
- `summarizeManifest` non-string `pipeline` — webview `esc()` String()-coerces → harmless.
- TOCTOU port race (agy rated HIGH) — inherent and accepted; the brief `close()`-before-resolve window in `isPortAvailable` is theoretical.
