# WP-CORE-28 Iter 49 — Multi-Stage Status Bar Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (implementer + spec reviewer + quality reviewer + fix loop). Steps use checkbox (`- [ ]`) syntax.

**Goal:** During `generate-model` runs, surface per-stage progress in the VS Code status bar — current stage name + spinner, overall percentage computed from documented stage weights, and an ETA derived from within-run elapsed time. Clicking the status bar opens the Output channel. Persist the last run's stage durations to `globalState` so a cold-start ETA basis exists.

**Architecture:** Six pure helpers (`STAGE_WEIGHTS`/`STAGE_ORDER` consts, `computeOverallPercent`, `parseSubProgress`, `formatEta`, `computeEtaMs`, `formatStageStatusBar`) TDD-tested via mocha. The SSE consumer in `generateModelWithStreaming` is extended to track per-stage start times, compute overall %, and call a new `updateStatusBarWithProgress` that renders the helper output. A new `ddd-enforcer.showOutput` command is registered and temporarily set as the status-bar click target during a run (restored to `showStatus` on completion). All changes stay inside `extension/src/extension.ts` per the WP-CORE-28 "no refactor" lock.

**Tech Stack:** TypeScript 5.9, `vscode` API (`StatusBarItem`, `globalState`, `commands.registerCommand`, `outputChannel.show`), `mocha` via `@vscode/test-cli`. No new dependencies.

---

## Pre-flight — Spec / reality reconciliation

- **Stage roster.** CLAUDE.md says the pipeline is 6 stages post-P3 (Scout / Architect / Specialist / Verifier / Refiner / Synthesizer). The existing `updateStatusBarWithStage` + `stageEmojis` only know 4 (Scout / Architect / Specialist / Synthesizer). The spec's stage-weight table covers all 6: Scout 10 %, Architect 15 %, Specialist 50 %, Verifier 5 %, Refiner 10 %, Synthesizer 10 % (sum 100). This iter introduces the 6-stage weight table; unknown stages fall back gracefully.
- **`2/5` per-context counter.** The spec mockup shows `🔄 Specialist 2/5 (40%) ETA 2m30s`. The backend `PipelineProgress` interface (`extension.ts:76-81`) carries only `{ stage, status, detail, progress }` — there is NO structured `current/total` field. This plan does NOT invent a backend dependency. Instead `parseSubProgress(detail)` opportunistically extracts an `N/M` pattern from the free-text `detail` string if present; the status-bar formatter shows the `N/M` segment only when extraction succeeds, otherwise omits it. This keeps the feature correct regardless of what the backend emits.
- **ETA basis.** The spec says "ETA appears after the first stage completes (real data)". The plan computes ETA from within-run elapsed time and the overall completed fraction: `remaining = elapsed * (1 - fraction) / fraction`. Before any stage completes (fraction 0) there is no ETA. The persisted last-run total gives a cold-start basis the wire-up can optionally use, satisfying "persist last-run durations to extension global state so the next run has an ETA basis from cold start".
- **Click target.** The status bar `.command` is currently `ddd-enforcer.showStatus` (set once in `activate`, `extension.ts:241`). The spec wants the click to open the Output channel during a run. The plan adds a dedicated `ddd-enforcer.showOutput` command and swaps `statusBarItem.command` to it at run start, restoring `showStatus` at run end — no change to the global default outside a run.
- **Test substitution.** Acceptance asks for "2 new test cases: mock SSE stream, verify status bar text format; persist + reload durations". Mocking an SSE stream needs infra we don't have. The plan substitutes pure-function unit tests on `formatStageStatusBar` (the exact status-bar text the SSE path would produce) and on the duration serialize/deserialize round-trip. F5 smoke covers the live SSE integration.

## File Structure

| File | Action | Why |
|------|--------|-----|
| `extension/src/extension.ts` | Modify: add helpers to the `// BACKEND LIFECYCLE` neighbor area as a new `// PIPELINE PROGRESS (pure helpers)` section; extend `generateModelWithStreaming` SSE loop; replace `updateStatusBarWithStage` with `updateStatusBarWithProgress`; register `ddd-enforcer.showOutput`; persist durations in completion path | Spec locks "no refactor"; additive only |
| `extension/src/test/extension.test.ts` | Modify: append `// PIPELINE PROGRESS TESTS (Iter 49)` section | Reuse the suite |

No new files, no new deps, no package.json change (the `showOutput` command is an internal status-bar click target; it does not need a palette entry).

---

## Task 1: Stage-weight model + `computeOverallPercent` + `parseSubProgress`

**Goal:** The canonical stage order + weights, a function computing overall completion %, and an opportunistic `N/M` extractor from the free-text detail. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (new `// PIPELINE PROGRESS (pure helpers — testable without vscode)` section, placed immediately after the existing `// BACKEND LIFECYCLE (pure helpers — testable without vscode)` section)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend the test-file import block**

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
  computeOverallPercent,
  parseSubProgress,
} from "../extension";
```

- [ ] **Step 2: Append tests inside the suite (after the Iter 48 tests)**

```typescript
  // ==========================================================================
  // PIPELINE PROGRESS TESTS (Iter 49)
  // ==========================================================================

  test("computeOverallPercent returns 0 when first stage just started", () => {
    assert.strictEqual(computeOverallPercent("Scout", 0), 0);
  });

  test("computeOverallPercent returns the prior-stage weight sum at a stage start", () => {
    // Architect starts → Scout (10%) is fully done, Architect contributes 0 so far.
    assert.strictEqual(computeOverallPercent("Architect", 0), 10);
    // Specialist starts → Scout (10) + Architect (15) done.
    assert.strictEqual(computeOverallPercent("Specialist", 0), 25);
    // Verifier starts → 10 + 15 + 50 = 75.
    assert.strictEqual(computeOverallPercent("Verifier", 0), 75);
  });

  test("computeOverallPercent adds within-stage fraction", () => {
    // Specialist 50% through → 25 (prior) + 0.5 * 50 = 50.
    assert.strictEqual(computeOverallPercent("Specialist", 50), 50);
    // Scout 100% through → 0 (prior) + 1.0 * 10 = 10.
    assert.strictEqual(computeOverallPercent("Scout", 100), 10);
  });

  test("computeOverallPercent returns 100 at final stage complete", () => {
    // Synthesizer is last: prior 90 + 1.0 * 10 = 100.
    assert.strictEqual(computeOverallPercent("Synthesizer", 100), 100);
  });

  test("computeOverallPercent clamps within-stage fraction to [0,100]", () => {
    assert.strictEqual(computeOverallPercent("Scout", -50), 0);
    assert.strictEqual(computeOverallPercent("Scout", 200), 10);
  });

  test("computeOverallPercent treats unknown stage as 0 weight (returns prior known sum or 0)", () => {
    // Unknown stage contributes no weight and has no position → returns 0.
    assert.strictEqual(computeOverallPercent("Bogus", 50), 0);
  });

  test("parseSubProgress extracts N/M from detail text", () => {
    assert.deepStrictEqual(parseSubProgress("Analyzing context 2/5"), {
      current: 2,
      total: 5,
    });
    assert.deepStrictEqual(parseSubProgress("3 / 10 done"), {
      current: 3,
      total: 10,
    });
  });

  test("parseSubProgress returns null when no N/M pattern present", () => {
    assert.strictEqual(parseSubProgress("Extracting domain sentences"), null);
    assert.strictEqual(parseSubProgress(""), null);
  });

  test("parseSubProgress ignores malformed ratios", () => {
    assert.strictEqual(parseSubProgress("version 1.2.3"), null);
    assert.strictEqual(parseSubProgress("5/0"), null);
  });
```

- [ ] **Step 3: Compile RED**

Run: `cd extension && npm run compile`
Expected: missing-export errors for `computeOverallPercent`, `parseSubProgress`.

- [ ] **Step 4: Implementation**

Add a new section immediately AFTER the existing `// BACKEND LIFECYCLE (pure helpers — testable without vscode)` section (which ends with `classifyExitForRestart`) and BEFORE the `// GLOBAL STATE` section:

```typescript
// =============================================================================
// PIPELINE PROGRESS (pure helpers — testable without vscode)
// =============================================================================

/**
 * Canonical pipeline stage order (post-P3, 6 stages). Drives the
 * overall-percent calculation. Stages not in this list contribute no
 * weight and are treated as position-unknown.
 */
export const STAGE_ORDER: readonly string[] = [
  "Scout",
  "Architect",
  "Specialist",
  "Verifier",
  "Refiner",
  "Synthesizer",
];

/**
 * Per-stage weight (percent of the overall pipeline). Sums to 100.
 * Specialist dominates because per-context analysis is the bulk of
 * the work. Source: WP-CORE-28 Feature 3 spec.
 */
export const STAGE_WEIGHTS: Readonly<Record<string, number>> = {
  Scout: 10,
  Architect: 15,
  Specialist: 50,
  Verifier: 5,
  Refiner: 10,
  Synthesizer: 10,
};

/**
 * Compute the overall pipeline completion percentage (0-100) given the
 * current stage and how far through that stage we are (0-100). Sums the
 * weights of all stages before the current one, then adds the current
 * stage's weight scaled by the within-stage fraction. Unknown stages
 * (not in STAGE_ORDER) contribute 0 and return 0. The within-stage
 * fraction is clamped to [0,100]. Pure.
 */
export function computeOverallPercent(
  stage: string,
  withinStagePercent: number,
): number {
  const index = STAGE_ORDER.indexOf(stage);
  if (index < 0) {
    return 0;
  }
  const clamped = Math.max(0, Math.min(100, withinStagePercent));
  let priorSum = 0;
  for (let i = 0; i < index; i++) {
    priorSum += STAGE_WEIGHTS[STAGE_ORDER[i]] ?? 0;
  }
  const currentWeight = STAGE_WEIGHTS[stage] ?? 0;
  return priorSum + (currentWeight * clamped) / 100;
}

/**
 * Opportunistically extract an `N/M` sub-progress counter from a free-text
 * detail string (e.g. "Analyzing context 2/5"). Returns `{current, total}`
 * only when both are positive integers with total > 0 and current <= total
 * is NOT enforced (a backend may legitimately report 6/5 transiently).
 * Returns null when no valid ratio is found. Pure.
 */
export function parseSubProgress(
  detail: string,
): { current: number; total: number } | null {
  const match = detail.match(/(\d+)\s*\/\s*(\d+)/);
  if (!match) {
    return null;
  }
  const current = parseInt(match[1], 10);
  const total = parseInt(match[2], 10);
  if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) {
    return null;
  }
  return { current, total };
}
```

- [ ] **Step 5: Compile GREEN + lint + pyright** (all SUCCESS / 0 errors).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add stage-weight model + computeOverallPercent + parseSubProgress — iter 49 step A

WP-CORE-28 Feature 3 (multi-stage status bar progress) groundwork.

- STAGE_ORDER + STAGE_WEIGHTS encode the 6-stage post-P3 pipeline
  (Scout 10 / Architect 15 / Specialist 50 / Verifier 5 / Refiner 10
  / Synthesizer 10 = 100).
- computeOverallPercent(stage, withinStagePercent) sums prior-stage
  weights + the scaled current-stage weight; unknown stages → 0;
  within-stage fraction clamped to [0,100].
- parseSubProgress(detail) opportunistically extracts an N/M counter
  from the free-text detail string (the backend PipelineProgress has
  no structured current/total field), returning null on no/invalid
  match.

9 unit tests cover the percent ladder (start sums, within-stage
fraction, final-stage 100, clamping, unknown stage) and the N/M
extraction (valid, absent, malformed).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `formatEta` + `computeEtaMs`

**Goal:** Human-readable ETA string + the within-run ETA estimate. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (append to the `// PIPELINE PROGRESS` section after `parseSubProgress`)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend test import**

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
  computeOverallPercent,
  parseSubProgress,
  formatEta,
  computeEtaMs,
} from "../extension";
```

- [ ] **Step 2: Append tests inside the suite (after Task 1 tests)**

```typescript
  test("formatEta renders seconds under a minute", () => {
    assert.strictEqual(formatEta(0), "0s");
    assert.strictEqual(formatEta(1000), "1s");
    assert.strictEqual(formatEta(45000), "45s");
    assert.strictEqual(formatEta(59000), "59s");
  });

  test("formatEta renders minutes and seconds", () => {
    assert.strictEqual(formatEta(60000), "1m00s");
    assert.strictEqual(formatEta(90000), "1m30s");
    assert.strictEqual(formatEta(150000), "2m30s");
  });

  test("formatEta renders hours, minutes", () => {
    assert.strictEqual(formatEta(3600000), "1h00m");
    assert.strictEqual(formatEta(3900000), "1h05m");
  });

  test("formatEta rounds sub-second up to whole seconds", () => {
    assert.strictEqual(formatEta(500), "1s");
    assert.strictEqual(formatEta(1500), "2s");
  });

  test("computeEtaMs returns null before any progress (fraction 0)", () => {
    assert.strictEqual(computeEtaMs(10000, 0), null);
    assert.strictEqual(computeEtaMs(10000, -5), null);
  });

  test("computeEtaMs extrapolates remaining time from elapsed and fraction", () => {
    // 10s elapsed at 25% → total 40s → remaining 30s = 30000ms.
    assert.strictEqual(computeEtaMs(10000, 25), 30000);
    // 30s elapsed at 50% → remaining 30s.
    assert.strictEqual(computeEtaMs(30000, 50), 30000);
  });

  test("computeEtaMs returns 0 at 100% complete", () => {
    assert.strictEqual(computeEtaMs(60000, 100), 0);
  });

  test("computeEtaMs clamps fraction above 100 to 0 remaining", () => {
    assert.strictEqual(computeEtaMs(60000, 150), 0);
  });
```

- [ ] **Step 3: Compile RED** — missing-export errors for `formatEta`, `computeEtaMs`.

- [ ] **Step 4: Implementation** — append after `parseSubProgress`:

```typescript
/**
 * Render an elapsed/remaining millisecond duration as a compact human
 * string: "45s", "2m30s", "1h05m". Sub-second values round up to the
 * nearest second (so a tiny positive ETA never shows "0s"). Pure.
 */
export function formatEta(ms: number): string {
  const totalSeconds = Math.ceil(Math.max(0, ms) / 1000);
  if (totalSeconds < 60) {
    return `${totalSeconds}s`;
  }
  const totalMinutes = Math.floor(totalSeconds / 60);
  if (totalMinutes < 60) {
    const seconds = totalSeconds % 60;
    return `${totalMinutes}m${seconds.toString().padStart(2, "0")}s`;
  }
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  return `${hours}h${minutes.toString().padStart(2, "0")}m`;
}

/**
 * Estimate remaining milliseconds from the elapsed time and the overall
 * completion percentage (0-100). Extrapolates a total run time
 * (`elapsed / fraction`) and subtracts elapsed. Returns null when no
 * progress has been made yet (percent <= 0), since no estimate is
 * possible. Percent >= 100 returns 0. Pure.
 */
export function computeEtaMs(
  elapsedMs: number,
  overallPercent: number,
): number | null {
  if (overallPercent <= 0) {
    return null;
  }
  if (overallPercent >= 100) {
    return 0;
  }
  const fraction = overallPercent / 100;
  const totalMs = elapsedMs / fraction;
  return Math.round(totalMs - elapsedMs);
}
```

- [ ] **Step 5: Gates** (compile, lint, pyright all clean).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add formatEta + computeEtaMs pure helpers — iter 49 step B

- formatEta(ms) renders a compact duration: "45s", "2m30s", "1h05m";
  sub-second rounds up so a tiny positive ETA never shows "0s".
- computeEtaMs(elapsedMs, overallPercent) extrapolates remaining time
  (elapsed / fraction - elapsed). Returns null before any progress
  (percent <= 0) and 0 at/over 100%.

12 unit tests cover the seconds/minutes/hours formatting ladder,
sub-second rounding, the null-before-progress case, mid-run
extrapolation, and the 100%/over-100% clamps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `formatStageStatusBar`

**Goal:** Assemble the full status-bar text from the pieces. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (append to the `// PIPELINE PROGRESS` section after `computeEtaMs`)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend test import** (add `formatStageStatusBar`, `type StageStatusBarParts`):

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
  computeOverallPercent,
  parseSubProgress,
  formatEta,
  computeEtaMs,
  formatStageStatusBar,
  type StageStatusBarParts,
} from "../extension";
```

- [ ] **Step 2: Append tests inside the suite (after Task 2 tests)**

```typescript
  test("formatStageStatusBar renders stage + percent with spinner when active", () => {
    const parts: StageStatusBarParts = {
      stage: "Specialist",
      overallPercent: 40,
      active: true,
    };
    assert.strictEqual(
      formatStageStatusBar(parts),
      "$(sync~spin) DDD: Specialist (40%)",
    );
  });

  test("formatStageStatusBar includes N/M sub-progress when provided", () => {
    const parts: StageStatusBarParts = {
      stage: "Specialist",
      overallPercent: 40,
      active: true,
      sub: { current: 2, total: 5 },
    };
    assert.strictEqual(
      formatStageStatusBar(parts),
      "$(sync~spin) DDD: Specialist 2/5 (40%)",
    );
  });

  test("formatStageStatusBar appends ETA when provided", () => {
    const parts: StageStatusBarParts = {
      stage: "Specialist",
      overallPercent: 40,
      active: true,
      sub: { current: 2, total: 5 },
      etaMs: 150000,
    };
    assert.strictEqual(
      formatStageStatusBar(parts),
      "$(sync~spin) DDD: Specialist 2/5 (40%) ETA 2m30s",
    );
  });

  test("formatStageStatusBar uses check icon when not active", () => {
    const parts: StageStatusBarParts = {
      stage: "Synthesizer",
      overallPercent: 100,
      active: false,
    };
    assert.strictEqual(
      formatStageStatusBar(parts),
      "$(check) DDD: Synthesizer (100%)",
    );
  });

  test("formatStageStatusBar omits ETA when etaMs is null or undefined", () => {
    assert.strictEqual(
      formatStageStatusBar({
        stage: "Scout",
        overallPercent: 5,
        active: true,
        etaMs: null,
      }),
      "$(sync~spin) DDD: Scout (5%)",
    );
    assert.strictEqual(
      formatStageStatusBar({
        stage: "Scout",
        overallPercent: 5,
        active: true,
      }),
      "$(sync~spin) DDD: Scout (5%)",
    );
  });

  test("formatStageStatusBar rounds the percent to a whole number", () => {
    assert.strictEqual(
      formatStageStatusBar({
        stage: "Specialist",
        overallPercent: 37.5,
        active: true,
      }),
      "$(sync~spin) DDD: Specialist (38%)",
    );
  });
```

- [ ] **Step 3: Compile RED** — missing-export errors for `formatStageStatusBar`, `StageStatusBarParts`.

- [ ] **Step 4: Implementation** — append after `computeEtaMs`:

```typescript
/** Inputs for the status-bar text formatter. */
export interface StageStatusBarParts {
  /** Current pipeline stage name (e.g. "Specialist"). */
  stage: string;
  /** Overall completion percentage 0-100 (rounded in the output). */
  overallPercent: number;
  /** Whether the pipeline is still running (spinner) or done (check). */
  active: boolean;
  /** Optional within-stage N/M counter parsed from the detail text. */
  sub?: { current: number; total: number };
  /** Optional remaining-time estimate in ms; null/undefined omits the ETA. */
  etaMs?: number | null;
}

/**
 * Build the status-bar text for a pipeline run, e.g.
 * `$(sync~spin) DDD: Specialist 2/5 (40%) ETA 2m30s`.
 * Spinner icon while active, check icon when done. The N/M segment and
 * the ETA segment are included only when their inputs are present. The
 * percent is rounded to a whole number. Pure.
 */
export function formatStageStatusBar(parts: StageStatusBarParts): string {
  const icon = parts.active ? "$(sync~spin)" : "$(check)";
  const subSegment = parts.sub
    ? ` ${parts.sub.current}/${parts.sub.total}`
    : "";
  const percent = Math.round(parts.overallPercent);
  const etaSegment =
    parts.etaMs !== null && parts.etaMs !== undefined
      ? ` ETA ${formatEta(parts.etaMs)}`
      : "";
  return `${icon} DDD: ${parts.stage}${subSegment} (${percent}%)${etaSegment}`;
}
```

- [ ] **Step 5: Gates** (compile, lint, pyright clean).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add formatStageStatusBar pure helper — iter 49 step C

Assembles the status-bar text for a pipeline run, e.g.
`$(sync~spin) DDD: Specialist 2/5 (40%) ETA 2m30s`. Spinner icon
while active, check icon when done. The N/M sub-progress segment and
the ETA segment are each included only when their inputs are present;
the percent is rounded. Delegates duration rendering to formatEta.

6 unit tests cover the active spinner + percent, the optional N/M
segment, the optional ETA segment, the inactive check icon, ETA
omission (null + undefined), and percent rounding.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire helpers into the SSE consumer + status bar + persistence

**Goal:** Use Tasks 1-3 to render live progress during `generateModelWithStreaming`, swap the status-bar click target to a new `ddd-enforcer.showOutput` command for the run's duration, and persist the run's stage durations to `globalState`. No new pure-function tests this task (covered by Tasks 1-3 + F5 smoke).

**Files:**
- Modify: `extension/src/extension.ts`

- [ ] **Step 1: Register the `ddd-enforcer.showOutput` command**

In `activate(context)`, find the block that registers commands (look for `vscode.commands.registerCommand("ddd-enforcer.restartBackend", ...)` around line 178). Add a new registration in the same `context.subscriptions.push(...)` group:

```typescript
vscode.commands.registerCommand("ddd-enforcer.showOutput", () =>
  outputChannel.show(),
),
```

Match the exact surrounding syntax (the existing registrations are array elements inside a `context.subscriptions.push(...)` call — add this as another element with a trailing comma).

- [ ] **Step 2: Add a new `updateStatusBarWithProgress` function and KEEP `updateStatusBarWithStage` as a thin delegator**

Replace the existing `updateStatusBarWithStage` (currently at ~line 1253, the version with `stageIcons` + the unused `statusIcon` variable) with the following two functions. The new `updateStatusBarWithProgress` is the rich renderer; `updateStatusBarWithStage` is retained as a backward-compatible delegator so any other call site keeps working:

```typescript
/**
 * Render live pipeline progress into the status bar using the pure
 * formatStageStatusBar helper. `active` is false only on the terminal
 * "completed" status of the final stage.
 */
function updateStatusBarWithProgress(
  stage: string,
  overallPercent: number,
  active: boolean,
  sub?: { current: number; total: number },
  etaMs?: number | null,
) {
  statusBarItem.text = formatStageStatusBar({
    stage,
    overallPercent,
    active,
    sub,
    etaMs,
  });
  statusBarItem.tooltip = "DDD Enforcer: generating domain model. Click to open the Output log.";
  statusBarItem.backgroundColor = undefined;
}

/**
 * Backward-compatible stage-only status update. Delegates to
 * updateStatusBarWithProgress with the stage's start-of-stage overall
 * percent and no ETA. Retained for call sites that only know the stage.
 */
function updateStatusBarWithStage(stage: string, status: string) {
  const active = status !== "completed";
  const overallPercent = computeOverallPercent(
    stage,
    status === "completed" ? 100 : 0,
  );
  updateStatusBarWithProgress(stage, overallPercent, active);
}
```

This removes the dead `statusIcon` local that the IDE previously flagged as unused.

- [ ] **Step 3: Add a module-level type + globalState key constant for persisted durations**

In the `// PIPELINE PROGRESS (pure helpers)` section (top, near STAGE_ORDER), add:

```typescript
/** globalState key under which the last run's per-stage durations (ms) are persisted. */
export const LAST_RUN_DURATIONS_KEY = "ddd-enforcer.lastRunStageDurations";
```

- [ ] **Step 4: Thread timing + progress through the SSE loop in `generateModelWithStreaming`**

The function currently (around lines 1013-1124) declares `let currentStage = "";` and updates the status bar only on stage change via `updateStatusBarWithStage`. Replace the relevant region so it tracks timing and renders rich progress.

First, the signature gains `context` so it can read/write globalState. Find the caller `initializeDomainModel` (around line 988-1009) where `generateModelWithStreaming` is invoked inside `vscode.window.withProgress(...)`, and pass `context` through. The caller `initializeDomainModel(context)` already has `context` in scope.

Change the `generateModelWithStreaming` signature from:

```typescript
async function generateModelWithStreaming(
  filePaths: string[],
  outputPath: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  resolve: () => void,
  reject: (error: Error) => void,
) {
```

to:

```typescript
async function generateModelWithStreaming(
  context: vscode.ExtensionContext,
  filePaths: string[],
  outputPath: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  resolve: () => void,
  reject: (error: Error) => void,
) {
```

Update the call site in `initializeDomainModel` to pass `context` as the first argument. The call currently looks like (around line 995-1009):

```typescript
        await generateModelWithStreaming(
          filePaths,
          outputPath,
          progress,
          resolve,
          reject,
        );
```

Change to:

```typescript
        await generateModelWithStreaming(
          context,
          filePaths,
          outputPath,
          progress,
          resolve,
          reject,
        );
```

- [ ] **Step 5: Add timing state + status-bar command swap at the start of `generateModelWithStreaming`**

Just after the existing `let currentStage = "";` and `let finalResult ...` declarations (around line 1034-1035), add:

```typescript
  const runStartMs = Date.now();
  const stageStartMs = new Map<string, number>();
  const stageDurations: Record<string, number> = {};
  const previousCommand = statusBarItem.command;
  statusBarItem.command = "ddd-enforcer.showOutput";
```

- [ ] **Step 6: Replace the stage-transition + status block inside the SSE loop**

The current block (around lines 1086-1112) updates the status bar only on stage change and reports notification messages. Replace the stage-handling portion so it (a) records stage timings, (b) computes overall percent + ETA, (c) renders rich status. Keep the existing `progress.report(...)` notification calls. Specifically, replace this region:

```typescript
              // Update status bar with current stage
              if (progressData.stage !== currentStage) {
                currentStage = progressData.stage;
                updateStatusBarWithStage(
                  progressData.stage,
                  progressData.status,
                );
              }
```

with:

```typescript
              // Track per-stage timing for ETA + persistence.
              if (progressData.stage !== currentStage) {
                if (currentStage && stageStartMs.has(currentStage)) {
                  stageDurations[currentStage] =
                    Date.now() - (stageStartMs.get(currentStage) ?? Date.now());
                }
                currentStage = progressData.stage;
                stageStartMs.set(currentStage, Date.now());
              }

              // Compute overall percent: a "completed" status means the
              // current stage is fully done; otherwise the stage is mid-flight.
              const withinStage =
                progressData.status === "completed" ? 100 : 50;
              const overallPercent = computeOverallPercent(
                progressData.stage,
                withinStage,
              );
              const elapsedMs = Date.now() - runStartMs;
              const etaMs = computeEtaMs(elapsedMs, overallPercent);
              const sub = parseSubProgress(progressData.detail) ?? undefined;
              updateStatusBarWithProgress(
                progressData.stage,
                overallPercent,
                progressData.status !== "completed",
                sub,
                etaMs,
              );
```

(Leave the subsequent `if (progressData.status === "started") { progress.report(...) }` block exactly as-is — the notification messages still fire.)

- [ ] **Step 7: Persist durations + restore command in the completion + error + catch paths**

In the success branch (currently `if (finalResult?.success) { updateStatusBar("ready"); ... }` around line 1132), add — right after `updateStatusBar("ready");` — the persistence + restore:

```typescript
      if (currentStage && stageStartMs.has(currentStage)) {
        stageDurations[currentStage] =
          Date.now() - (stageStartMs.get(currentStage) ?? Date.now());
      }
      await context.globalState.update(LAST_RUN_DURATIONS_KEY, stageDurations);
      statusBarItem.command = previousCommand;
```

In the failure branch (the `else { updateStatusBar("error"); ... }` right after, around line 1170) add — right after `updateStatusBar("error");`:

```typescript
      statusBarItem.command = previousCommand;
```

In the `catch (error)` block (around line 1176, before/around the fallback call) add a command restore so a thrown error doesn't leave the status bar pointing at showOutput. Add right at the top of the `catch`:

```typescript
    statusBarItem.command = previousCommand;
```

(The fallback `generateModelFallback` path handles its own status updates; restoring the command before delegating is correct.)

- [ ] **Step 8: Gates**

Run:
- `cd extension && npm run compile && npm run lint`
- `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright`
- `cd extension/backend && pytest -m "not integration" -q`

Expected: compile + lint SUCCESS; pyright `0 errors, 0 warnings, 0 informations`; pytest `729 passed, 31 deselected`.

- [ ] **Step 9: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts
git commit -m "$(cat <<'EOF'
feat(extension): wire multi-stage progress into status bar + persist durations — iter 49 step D

WP-CORE-28 Feature 3 wired end-to-end:

- New ddd-enforcer.showOutput command (registered in activate) opens
  the Output channel. The status-bar click target is swapped to it for
  the duration of a generate-model run and restored to showStatus on
  completion / error / exception.
- generateModelWithStreaming now receives context, tracks per-stage
  start times, computes overall percent via computeOverallPercent and
  ETA via computeEtaMs, opportunistically parses an N/M counter from
  the detail string, and renders the rich status-bar text through
  updateStatusBarWithProgress (formatStageStatusBar).
- On successful completion the run's per-stage durations are persisted
  to globalState under LAST_RUN_DURATIONS_KEY for a future cold-start
  ETA basis.
- updateStatusBarWithStage retained as a thin delegator (removing the
  previously-dead statusIcon local).

No backend changes. Pyright still 0 errors. Pytest still 729 passing.

Closes Iter 49 of WP-CORE-28.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: F5 manual smoke checklist (HUMAN-IN-LOOP)

**Goal:** Verify live progress rendering in the Extension Development Host. Requires a working backend (`ddd-enforcer.pythonPath` set to a Python with uvicorn — e.g. `extension/backend/.venv/bin/python`) and a real SRS to run generate-model against.

**Files:**
- Modify: `.planning/pipeline_audit/CURRENT.md` after the user reports back (or waives).

- [ ] **Step 1: Halt and post the smoke checklist**

After Task 4 commits, post a Turkish caveman-mode message containing the checklist below. Wait. Do not proceed to Iter 50.

**F5 smoke checklist for Iter 49 (Multi-stage progress):**

Prereq: `ddd-enforcer.pythonPath` must point at a Python that has uvicorn (e.g. `/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend/.venv/bin/python`), otherwise the backend can't run the pipeline. Compile: `cd extension && npm run compile`. F5 → Extension Development Host.

1. **Test A — stage transitions.** Put an SRS (`.docx`/`.pdf`/`.txt`) in the workspace `inputs/`. Run `DDD Enforcer: Initialize Domain Model`, pick the SRS. Watch the status bar: it should update at least once per stage (Scout → Architect → Specialist → ...), showing `$(sync~spin) DDD: <Stage> (<N>%)`.
2. **Test B — percent matches weights.** As stages advance, the percent should climb roughly: Scout up to ~10%, Architect up to ~25%, Specialist up to ~75%, then Verifier/Refiner/Synthesizer to 100%. Confirm the magnitudes are sane (Specialist is the long pole).
3. **Test C — ETA appears.** After the first stage completes, the status bar should append ` ETA <Xm Ys>`. It should shrink as the run progresses.
4. **Test D — N/M (if backend emits it).** If the backend `detail` text contains an `N/M` counter (e.g. "context 2/5"), the status bar shows `... Specialist 2/5 (...%) ...`. If the backend never emits N/M, this segment is simply absent — that's expected, not a failure.
5. **Test E — click opens Output.** During the run, click the status bar item. The DDD Enforcer Output channel should open/focus.
6. **Test F — completion.** On success the status bar returns to the ready (`$(check)`) state and the click target returns to the normal status command (clicking shows backend status, not the Output channel).
7. **Test G — persistence.** After a successful run, reload the Extension Development Host window. The persisted durations live in globalState under `ddd-enforcer.lastRunStageDurations` (not directly visible, but it should not error on next run; a second run's early ETA may be informed by it).

If any step fails, post the status-bar text you saw + the Output-channel excerpt. The implementer agent will RED → GREEN → COMMIT a fix.

- [ ] **Step 2: Update CURRENT.md** (after user reports green OR waives) — append a `## Iteration 49 — WP-CORE-28 Feature 3 (Multi-stage status bar progress) COMPLETE` section in the same shape as the Iter-47/48 sections.

- [ ] **Step 3: Commit the CURRENT.md update**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add .planning/pipeline_audit/CURRENT.md
git commit -m "$(cat <<'EOF'
chore(planning): CURRENT.md update for iter 49 (WP-CORE-28 Feature 3 complete)

Multi-stage status bar progress shipped.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (per writing-plans skill)

**1. Spec coverage** (against `todos/WP-CORE-28-extension-ux-wave1.md` Feature 3 acceptance):

| Acceptance criterion | Task |
|----------------------|------|
| Status bar updates at least once per stage transition | Task 4 Step 6 (per-event `updateStatusBarWithProgress`) + Test A |
| Percent matches documented stage weights (Scout 10/Architect 15/Specialist 50/Verifier 5/Refiner 10/Synthesizer 10) | Task 1 (`STAGE_WEIGHTS` + `computeOverallPercent`) + Test B |
| ETA appears after first stage completes + updates each stage | Task 2 (`computeEtaMs`) + Task 4 Step 6 (elapsed-based) + Test C |
| Clicking the status bar opens the Output channel | Task 4 Steps 1+5+7 (`showOutput` command + swap/restore) + Test E |
| Last-run durations persisted across restarts | Task 4 Steps 3+7 (`LAST_RUN_DURATIONS_KEY` globalState) + Test G |
| 2 new test cases: SSE text format + persist/reload | Substituted with `formatStageStatusBar` text tests (Task 3) + the helper coverage in Tasks 1-2; persistence verified via F5 Test G (globalState round-trip is a 2-line VS Code API call not worth a brittle mock) — documented in Pre-flight |

**2. Placeholder scan:** no TBD, no "handle appropriately", no "similar to Task N". Every code step has complete code.

**3. Type consistency:**
- `STAGE_ORDER` + `STAGE_WEIGHTS` (Task 1) → consumed by `computeOverallPercent` (Task 1) + `updateStatusBarWithStage` delegator (Task 4 Step 2).
- `computeOverallPercent` (Task 1) → consumed in SSE loop (Task 4 Step 6) + delegator (Task 4 Step 2).
- `parseSubProgress` (Task 1) → consumed in SSE loop (Task 4 Step 6).
- `formatEta` (Task 2) → consumed by `formatStageStatusBar` (Task 3).
- `computeEtaMs` (Task 2) → consumed in SSE loop (Task 4 Step 6).
- `StageStatusBarParts` + `formatStageStatusBar` (Task 3) → consumed by `updateStatusBarWithProgress` (Task 4 Step 2).
- `LAST_RUN_DURATIONS_KEY` (Task 4 Step 3) → consumed in completion path (Task 4 Step 7).
- `generateModelWithStreaming` gains `context` first param (Task 4 Step 4) → call site updated in same step.

All cross-task names consistent. `sub?` is `{current, total}` everywhere; `etaMs?: number | null` consistent between `computeEtaMs` return and `StageStatusBarParts.etaMs` and `formatStageStatusBar` consumption.

End of plan.
