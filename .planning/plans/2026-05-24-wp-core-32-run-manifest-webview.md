# WP-CORE-32 — PaperRunManifest Webview Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (implementer + spec reviewer + quality reviewer + fix loop). Steps use checkbox (`- [ ]`) syntax.

**Goal:** A read-only VS Code webview (`ddd-enforcer.showRunManifests` command) that lists every PaperRunManifest under the workspace `runs/` directory, lets the user sort the list and click a run to see its full detail (violations, metrics, provenance), and auto-refreshes when new manifests are written. Vanilla HTML + JS, file-load only, no backend change.

**Architecture:** Pure helpers (`summarizeManifest`, `sortRunSummaries`, `buildRunManifestsHtml`, `generateNonce`) are TDD-tested via mocha. The extension-side `openRunManifestsWebview` discovers `runs/**/manifest.json`, parses + summarizes each, and posts the summaries to a `WebviewPanel` whose inline HTML (CSP-locked, nonce'd inline script) renders the table + detail pane and posts back `openDetail`/`refresh` messages. A debounced `FileSystemWatcher` re-discovers on manifest writes; it is disposed when the panel closes. All extension code lives in `extension/src/extension.ts` (consistent with WP-CORE-28's no-new-module discipline); one command entry is added to `package.json`.

**Tech Stack:** TypeScript 5.9, `vscode` API (`WebviewPanel`, `postMessage`, `createFileSystemWatcher`, `workspace.findFiles`/`fs`), `mocha` via `@vscode/test-cli`. No new dependencies, no chart lib, no webview-ui-toolkit.

---

## Pre-flight — schema + reality reconciliation (read before planning tasks)

The PaperRunManifest schema is defined in `extension/backend/core/run_manifest.py:188-274`. The TypeScript interface below MUST mirror it. Verified fields:

```
run_id: string
timestamp_utc: string                       // ISO-8601 Z
pipeline: "P1" | "P2" | "P3" | null         // usually null (WP-01d cancelled)
model_id: string
provider: "gemini" | "ollama"
srs_path: string
srs_sha256: string                          // 64-hex
code_root: string | null
code_sha256: string | null                  // 64-hex
violations: Violation[]                      // {violation_type, location, severity:"ERROR"|"WARN", message, srs_path?, suggestion?}
latency_seconds: number
prompt_tokens: number
completion_tokens: number
cost_usd: number
judge_verdict_path: string | null
audit_overrides_path: string | null
seed_manifest_path: string | null
schema_version: string                       // "1.0"
```

**Spec vs schema gaps (handled in this plan, not by inventing fields):**
- The spec's list column "srs_label" does not exist as a field → derive it as `basename(srs_path)`.
- The spec's "token counts per stage" does not exist → the manifest carries only flat `prompt_tokens` / `completion_tokens`; the detail pane shows those two plus their sum, NOT a per-stage breakdown.
- The spec's "provenance hashes (sha256_of_file, sha256_of_code_tree)" map to the fields `srs_sha256` + `code_sha256`.
- **Write path:** `write_paper_run_manifest` writes to `runs/<run_id>/manifest.json` (per-run subdirectory) — the spec's `runs/**/manifest.json` glob is therefore correct. The pre-existing flat `runs/*.manifest.json` files are a DIFFERENT internal observability format (`core.observability.run_manifest.RunManifest`) and MUST be ignored — only match `manifest.json` (exact basename) under `runs/`, and skip files that fail PaperRunManifest validation (no `run_id` field). `summarizeManifest` returns `null` for non-conforming JSON so legacy/foreign files are filtered out.
- **No PaperRunManifest files exist on disk yet** (the paper pipeline hasn't been run to produce them). The webview must therefore handle the empty-runs case gracefully (friendly empty state) and the tests use constructed-literal manifests, not on-disk fixtures.

## File Structure

| File | Action | Why |
|------|--------|-----|
| `extension/src/extension.ts` | Modify: add `// RUN MANIFEST WEBVIEW (pure helpers)` section (interfaces + `summarizeManifest` + `sortRunSummaries` + `generateNonce` + `buildRunManifestsHtml`); add `openRunManifestsWebview` + register the command | No new module (WP-CORE-28 discipline) |
| `extension/package.json` | Modify: add the `ddd-enforcer.showRunManifests` command to `contributes.commands` | Palette discoverability |
| `extension/src/test/extension.test.ts` | Modify: append `// RUN MANIFEST WEBVIEW TESTS (WP-CORE-32)` section | Reuse suite |

No new files, no new deps. Inline HTML (no separate `.html`), so no `media/` dir or `localResourceRoots` for scripts.

---

## Task 1: Manifest types + `summarizeManifest` + `sortRunSummaries`

**Goal:** TypeScript mirrors of the PaperRunManifest schema, a summarizer that extracts the 7 list columns (returning null for non-conforming JSON), and a pure sort. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (new `// RUN MANIFEST WEBVIEW (pure helpers — testable without vscode)` section, placed immediately after the `// VALIDATION HOVER (pure helpers …)` section, before `// GLOBAL STATE`)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend the test-file import block** — add the new names (append after the Iter-50 `formatHoverMarkdown`):

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
  STAGE_ORDER,
  STAGE_WEIGHTS,
  formatEta,
  computeEtaMs,
  formatStageStatusBar,
  type StageStatusBarParts,
  LruCache,
  truncateExcerpt,
  boldMatchingSpan,
  formatHoverMarkdown,
  summarizeManifest,
  sortRunSummaries,
  type RunSummary,
  type PaperRunManifest,
} from "../extension";
```

- [ ] **Step 2: Append tests inside the suite (after the Iter-50 VALIDATION HOVER TESTS)**

```typescript
  // ==========================================================================
  // RUN MANIFEST WEBVIEW TESTS (WP-CORE-32)
  // ==========================================================================

  const sampleManifest: PaperRunManifest = {
    run_id: "gemini-3.1-pro__ecommerce__seed7__20260520-1642",
    timestamp_utc: "2026-05-20T16:42:42Z",
    pipeline: null,
    model_id: "gemini-3.1-pro-preview",
    provider: "gemini",
    srs_path: "/abs/workspace/inputs/SRS.docx",
    srs_sha256: "a".repeat(64),
    code_root: "/abs/workspace",
    code_sha256: "b".repeat(64),
    violations: [
      {
        violation_type: "ubiquitous_language_drift",
        location: "orders/service.py:42",
        severity: "ERROR",
        message: "Use 'Customer' not 'Client'.",
      },
      {
        violation_type: "aggregate_boundary",
        location: "orders/model.py:10",
        severity: "WARN",
        message: "Modify Order via its aggregate root.",
      },
    ],
    latency_seconds: 12.5,
    prompt_tokens: 1000,
    completion_tokens: 500,
    cost_usd: 0.0123,
    judge_verdict_path: null,
    audit_overrides_path: null,
    seed_manifest_path: null,
    schema_version: "1.0",
  };

  test("summarizeManifest extracts the seven list columns", () => {
    const summary = summarizeManifest(sampleManifest);
    assert.ok(summary, "returns a summary for a valid manifest");
    if (summary) {
      assert.strictEqual(summary.runId, sampleManifest.run_id);
      assert.strictEqual(summary.pipeline, "—"); // null pipeline rendered as em-dash
      assert.strictEqual(summary.modelId, "gemini-3.1-pro-preview");
      assert.strictEqual(summary.srsLabel, "SRS.docx"); // basename of srs_path
      assert.strictEqual(summary.timestamp, "2026-05-20T16:42:42Z");
      assert.strictEqual(summary.costUsd, 0.0123);
      assert.strictEqual(summary.violationCount, 2);
    }
  });

  test("summarizeManifest renders a non-null pipeline verbatim", () => {
    const summary = summarizeManifest({ ...sampleManifest, pipeline: "P3" });
    assert.ok(summary);
    if (summary) {
      assert.strictEqual(summary.pipeline, "P3");
    }
  });

  test("summarizeManifest derives srsLabel from a posix or windows path", () => {
    const posix = summarizeManifest({
      ...sampleManifest,
      srs_path: "/a/b/c/Banking.pdf",
    });
    assert.strictEqual(posix?.srsLabel, "Banking.pdf");
    const win = summarizeManifest({
      ...sampleManifest,
      srs_path: "C:\\runs\\Health.txt",
    });
    assert.strictEqual(win?.srsLabel, "Health.txt");
  });

  test("summarizeManifest returns null for non-conforming JSON (no run_id)", () => {
    assert.strictEqual(summarizeManifest({ foo: "bar" } as unknown), null);
    assert.strictEqual(summarizeManifest(null as unknown), null);
    assert.strictEqual(summarizeManifest(42 as unknown), null);
  });

  test("summarizeManifest tolerates a missing violations array", () => {
    const noViol = summarizeManifest({
      ...sampleManifest,
      violations: undefined as unknown as [],
    });
    assert.ok(noViol);
    if (noViol) {
      assert.strictEqual(noViol.violationCount, 0);
    }
  });

  test("sortRunSummaries sorts by cost ascending and descending", () => {
    const rows: RunSummary[] = [
      { runId: "a", pipeline: "—", modelId: "m", srsLabel: "s", timestamp: "t1", costUsd: 0.5, violationCount: 1 },
      { runId: "b", pipeline: "—", modelId: "m", srsLabel: "s", timestamp: "t2", costUsd: 0.1, violationCount: 3 },
      { runId: "c", pipeline: "—", modelId: "m", srsLabel: "s", timestamp: "t3", costUsd: 0.9, violationCount: 2 },
    ];
    const asc = sortRunSummaries(rows, "costUsd", "asc");
    assert.deepStrictEqual(asc.map((r) => r.runId), ["b", "a", "c"]);
    const desc = sortRunSummaries(rows, "costUsd", "desc");
    assert.deepStrictEqual(desc.map((r) => r.runId), ["c", "a", "b"]);
  });

  test("sortRunSummaries sorts strings case-insensitively", () => {
    const rows: RunSummary[] = [
      { runId: "Zeta", pipeline: "—", modelId: "m", srsLabel: "s", timestamp: "t", costUsd: 0, violationCount: 0 },
      { runId: "alpha", pipeline: "—", modelId: "m", srsLabel: "s", timestamp: "t", costUsd: 0, violationCount: 0 },
    ];
    const asc = sortRunSummaries(rows, "runId", "asc");
    assert.deepStrictEqual(asc.map((r) => r.runId), ["alpha", "Zeta"]);
  });

  test("sortRunSummaries does not mutate the input array", () => {
    const rows: RunSummary[] = [
      { runId: "a", pipeline: "—", modelId: "m", srsLabel: "s", timestamp: "t", costUsd: 0.5, violationCount: 1 },
      { runId: "b", pipeline: "—", modelId: "m", srsLabel: "s", timestamp: "t", costUsd: 0.1, violationCount: 3 },
    ];
    const before = rows.map((r) => r.runId);
    sortRunSummaries(rows, "costUsd", "asc");
    assert.deepStrictEqual(rows.map((r) => r.runId), before);
  });
```

- [ ] **Step 3: Compile RED** — missing-export errors for `summarizeManifest`, `sortRunSummaries`, `RunSummary`, `PaperRunManifest`.

- [ ] **Step 4: Implementation** — add the new section after `// VALIDATION HOVER (pure helpers …)` and before `// GLOBAL STATE`:

```typescript
// =============================================================================
// RUN MANIFEST WEBVIEW (pure helpers — testable without vscode)
// =============================================================================

/** One DDD violation as recorded in a PaperRunManifest (mirror of the backend schema). */
export interface ManifestViolation {
  violation_type: string;
  location: string;
  severity: "ERROR" | "WARN";
  message: string;
  srs_path?: string | null;
  suggestion?: string | null;
}

/** Mirror of extension/backend/core/run_manifest.py:PaperRunManifest (schema_version "1.0"). */
export interface PaperRunManifest {
  run_id: string;
  timestamp_utc: string;
  pipeline: "P1" | "P2" | "P3" | null;
  model_id: string;
  provider: "gemini" | "ollama";
  srs_path: string;
  srs_sha256: string;
  code_root: string | null;
  code_sha256: string | null;
  violations: ManifestViolation[];
  latency_seconds: number;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  judge_verdict_path: string | null;
  audit_overrides_path: string | null;
  seed_manifest_path: string | null;
  schema_version: string;
}

/** The seven list-view columns for one run. */
export interface RunSummary {
  runId: string;
  pipeline: string; // "—" when null
  modelId: string;
  srsLabel: string; // basename of srs_path
  timestamp: string;
  costUsd: number;
  violationCount: number;
}

/** A column key for RunSummary sorting. */
export type RunSummaryColumn = keyof RunSummary;

/**
 * Extract the basename of a path that may use POSIX (/) or Windows (\)
 * separators. Pure (does not touch the filesystem).
 */
function basenameOf(p: string): string {
  const norm = p.replace(/\\/g, "/");
  const idx = norm.lastIndexOf("/");
  return idx >= 0 ? norm.slice(idx + 1) : norm;
}

/**
 * Build a RunSummary from a parsed manifest object. Returns null when the
 * input is not a conforming PaperRunManifest (no string `run_id`), so
 * legacy/foreign JSON files under runs/ are silently filtered out. A
 * missing `violations` array counts as zero. Pure.
 */
export function summarizeManifest(raw: unknown): RunSummary | null {
  if (typeof raw !== "object" || raw === null) {
    return null;
  }
  const m = raw as Partial<PaperRunManifest>;
  if (typeof m.run_id !== "string") {
    return null;
  }
  return {
    runId: m.run_id,
    pipeline: m.pipeline ?? "—",
    modelId: typeof m.model_id === "string" ? m.model_id : "",
    srsLabel: typeof m.srs_path === "string" ? basenameOf(m.srs_path) : "",
    timestamp: typeof m.timestamp_utc === "string" ? m.timestamp_utc : "",
    costUsd: typeof m.cost_usd === "number" ? m.cost_usd : 0,
    violationCount: Array.isArray(m.violations) ? m.violations.length : 0,
  };
}

/**
 * Return a NEW array of summaries sorted by `column`. Numeric columns
 * compare numerically; everything else compares case-insensitive strings.
 * Does not mutate the input. Pure.
 */
export function sortRunSummaries(
  rows: RunSummary[],
  column: RunSummaryColumn,
  direction: "asc" | "desc",
): RunSummary[] {
  const numericColumns: ReadonlySet<RunSummaryColumn> = new Set([
    "costUsd",
    "violationCount",
  ]);
  const sign = direction === "asc" ? 1 : -1;
  const copy = [...rows];
  copy.sort((a, b) => {
    if (numericColumns.has(column)) {
      return (Number(a[column]) - Number(b[column])) * sign;
    }
    const av = String(a[column]).toLowerCase();
    const bv = String(b[column]).toLowerCase();
    if (av < bv) {
      return -1 * sign;
    }
    if (av > bv) {
      return 1 * sign;
    }
    return 0;
  });
  return copy;
}
```

- [ ] **Step 5: Gates** — `cd extension && npm run compile && npm run lint` (SUCCESS); `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright` (0 errors).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add PaperRunManifest types + summarizeManifest + sortRunSummaries — wp-core-32 step A

WP-CORE-32 (run-manifest webview) groundwork. TypeScript mirrors of the
backend PaperRunManifest schema (run_manifest.py:188), plus two pure
helpers:

- summarizeManifest(raw) extracts the seven list-view columns (run id,
  pipeline rendered "—" when null, model id, srs label = basename of
  srs_path, timestamp, cost, violation count). Returns null for
  non-conforming JSON so legacy/foreign runs/*.json files are filtered
  out; a missing violations array counts as zero.
- sortRunSummaries(rows, column, direction) returns a new sorted array
  (numeric compare for cost/violationCount, case-insensitive string
  compare otherwise) without mutating the input.

8 unit tests cover column extraction, null + "P3" pipeline rendering,
posix/windows basename derivation, non-conforming-JSON filtering,
missing-violations tolerance, numeric asc/desc sort, case-insensitive
string sort, and input immutability.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `generateNonce` + `buildRunManifestsHtml`

**Goal:** A CSP nonce generator and the inline webview HTML (table + detail pane + sort + message passing) as a pure string builder. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (append to the `// RUN MANIFEST WEBVIEW (pure helpers)` section after `sortRunSummaries`)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend test import** (add `generateNonce`, `buildRunManifestsHtml`).

- [ ] **Step 2: Append tests (after Task 1 tests)**

```typescript
  test("generateNonce returns a 32-char alphanumeric string", () => {
    const nonce = generateNonce();
    assert.strictEqual(nonce.length, 32);
    assert.ok(/^[A-Za-z0-9]+$/.test(nonce), "alphanumeric only");
  });

  test("generateNonce returns a different value each call", () => {
    const a = generateNonce();
    const b = generateNonce();
    assert.notStrictEqual(a, b);
  });

  test("buildRunManifestsHtml embeds the nonce on the script tag and in the CSP", () => {
    const html = buildRunManifestsHtml("NONCE123", "vscode-resource:");
    assert.ok(
      html.includes(`<script nonce="NONCE123">`),
      "script tag carries the nonce",
    );
    assert.ok(
      html.includes(`script-src 'nonce-NONCE123'`),
      "CSP allows only the nonce'd script",
    );
    assert.ok(
      html.includes("Content-Security-Policy"),
      "has a CSP meta tag",
    );
  });

  test("buildRunManifestsHtml CSP uses the provided cspSource and forbids external loads", () => {
    const html = buildRunManifestsHtml("N", "vscode-webview://abc");
    assert.ok(html.includes("default-src 'none'"), "locks default-src to none");
    assert.ok(html.includes("vscode-webview://abc"), "uses the cspSource for styles");
    // No http(s) origins hard-coded in the HTML → no external network.
    assert.ok(!/https?:\/\//.test(html), "no external http(s) URLs in the HTML");
  });

  test("buildRunManifestsHtml contains the table scaffold and message wiring", () => {
    const html = buildRunManifestsHtml("N", "vscode-webview://abc");
    assert.ok(html.includes("acquireVsCodeApi()"), "acquires the vscode api");
    assert.ok(html.includes("addEventListener(\"message\""), "listens for messages");
    assert.ok(html.includes("id=\"runs-table\"") || html.includes("id='runs-table'"), "has the runs table element");
    assert.ok(html.includes("id=\"detail\"") || html.includes("id='detail'"), "has the detail pane element");
  });
```

- [ ] **Step 3: Compile RED** — missing-export errors for `generateNonce`, `buildRunManifestsHtml`.

- [ ] **Step 4: Implementation** — append after `sortRunSummaries`:

```typescript
/**
 * Generate a 32-character alphanumeric nonce for the webview CSP.
 * Pure-enough for testing (uses Math.random; not used for cryptographic
 * secrecy — only to satisfy VS Code's inline-script CSP requirement).
 */
export function generateNonce(): string {
  const chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let nonce = "";
  for (let i = 0; i < 32; i++) {
    nonce += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return nonce;
}

/**
 * Build the complete inline HTML document for the run-manifest webview.
 * Locks the CSP to `default-src 'none'`, allows only the nonce'd inline
 * script and `cspSource`-scoped inline styles, and contains no external
 * URLs (no network). The script acquires the VS Code API, renders the
 * summaries table (sortable headers post a re-sort handled client-side),
 * and posts `openDetail` / `refresh` messages back to the extension. The
 * extension posts `runList` (RunSummary[]) and `runDetail` (full manifest)
 * messages in. Pure (returns a string).
 */
export function buildRunManifestsHtml(
  nonce: string,
  cspSource: string,
): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DDD Enforcer — Run Manifests</title>
<style>
  body { font-family: var(--vscode-font-family); color: var(--vscode-foreground); padding: 0 12px; }
  h2 { margin: 12px 0 4px; }
  table { border-collapse: collapse; width: 100%; margin-top: 8px; }
  th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--vscode-panel-border); font-size: 12px; }
  th { cursor: pointer; user-select: none; }
  th:hover { color: var(--vscode-textLink-foreground); }
  tr.run-row { cursor: pointer; }
  tr.run-row:hover { background: var(--vscode-list-hoverBackground); }
  #empty { color: var(--vscode-descriptionForeground); margin-top: 16px; }
  #detail { margin-top: 16px; border-top: 2px solid var(--vscode-panel-border); padding-top: 8px; display: none; }
  .sev-ERROR { color: var(--vscode-errorForeground); font-weight: bold; }
  .sev-WARN { color: var(--vscode-editorWarning-foreground); }
  pre { white-space: pre-wrap; word-break: break-word; background: var(--vscode-textCodeBlock-background); padding: 8px; }
  button { margin-top: 8px; }
</style>
</head>
<body>
<h2>Run Manifests</h2>
<button id="refresh">Refresh</button>
<div id="empty" style="display:none;">No run manifests found under <code>runs/</code>. Run the pipeline first.</div>
<table id="runs-table" style="display:none;">
  <thead><tr>
    <th data-col="runId">Run ID</th>
    <th data-col="pipeline">Pipeline</th>
    <th data-col="modelId">Model</th>
    <th data-col="srsLabel">SRS</th>
    <th data-col="timestamp">Timestamp</th>
    <th data-col="costUsd">Cost (USD)</th>
    <th data-col="violationCount">Violations</th>
  </tr></thead>
  <tbody id="runs-body"></tbody>
</table>
<div id="detail"></div>
<script nonce="${nonce}">
  const vscode = acquireVsCodeApi();
  let runs = [];
  let sortCol = "timestamp";
  let sortDir = "desc";

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function sortRuns(rows, col, dir) {
    const numeric = col === "costUsd" || col === "violationCount";
    const sign = dir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      if (numeric) { return (Number(a[col]) - Number(b[col])) * sign; }
      const av = String(a[col]).toLowerCase();
      const bv = String(b[col]).toLowerCase();
      return av < bv ? -1 * sign : av > bv ? 1 * sign : 0;
    });
  }

  function renderList() {
    const table = document.getElementById("runs-table");
    const empty = document.getElementById("empty");
    const body = document.getElementById("runs-body");
    if (!runs.length) {
      table.style.display = "none";
      empty.style.display = "block";
      body.innerHTML = "";
      return;
    }
    empty.style.display = "none";
    table.style.display = "table";
    const sorted = sortRuns(runs, sortCol, sortDir);
    body.innerHTML = sorted.map((r) =>
      '<tr class="run-row" data-run="' + esc(r.runId) + '">' +
      "<td>" + esc(r.runId) + "</td>" +
      "<td>" + esc(r.pipeline) + "</td>" +
      "<td>" + esc(r.modelId) + "</td>" +
      "<td>" + esc(r.srsLabel) + "</td>" +
      "<td>" + esc(r.timestamp) + "</td>" +
      "<td>" + esc(Number(r.costUsd).toFixed(4)) + "</td>" +
      "<td>" + esc(r.violationCount) + "</td>" +
      "</tr>"
    ).join("");
    body.querySelectorAll("tr.run-row").forEach((row) => {
      row.addEventListener("click", () => {
        vscode.postMessage({ type: "openDetail", runId: row.getAttribute("data-run") });
      });
    });
  }

  function renderDetail(manifest) {
    const detail = document.getElementById("detail");
    detail.style.display = "block";
    const violations = (manifest.violations || []).map((v) =>
      '<li><span class="sev-' + esc(v.severity) + '">[' + esc(v.severity) + "]</span> " +
      esc(v.violation_type) + " — " + esc(v.location) + "<br/>" + esc(v.message) + "</li>"
    ).join("");
    detail.innerHTML =
      "<h2>" + esc(manifest.run_id) + "</h2>" +
      "<p><b>Model:</b> " + esc(manifest.model_id) + " (" + esc(manifest.provider) + ")" +
      " &nbsp; <b>Pipeline:</b> " + esc(manifest.pipeline || "—") +
      " &nbsp; <b>Timestamp:</b> " + esc(manifest.timestamp_utc) + "</p>" +
      "<p><b>Metrics:</b> latency " + esc(manifest.latency_seconds) + "s, " +
      "prompt " + esc(manifest.prompt_tokens) + " + completion " + esc(manifest.completion_tokens) +
      " = " + esc(Number(manifest.prompt_tokens) + Number(manifest.completion_tokens)) + " tokens, " +
      "cost $" + esc(Number(manifest.cost_usd).toFixed(6)) + "</p>" +
      "<p><b>Provenance:</b><br/>srs: " + esc(manifest.srs_path) + "<br/>" +
      "srs_sha256: " + esc(manifest.srs_sha256) + "<br/>" +
      "code_root: " + esc(manifest.code_root || "—") + "<br/>" +
      "code_sha256: " + esc(manifest.code_sha256 || "—") + "</p>" +
      "<h3>Violations (" + ((manifest.violations || []).length) + ")</h3>" +
      "<ul>" + violations + "</ul>";
  }

  document.querySelectorAll("th[data-col]").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.getAttribute("data-col");
      if (sortCol === col) { sortDir = sortDir === "asc" ? "desc" : "asc"; }
      else { sortCol = col; sortDir = "asc"; }
      renderList();
    });
  });

  document.getElementById("refresh").addEventListener("click", () => {
    vscode.postMessage({ type: "refresh" });
  });

  window.addEventListener("message", (event) => {
    const msg = event.data;
    if (msg.type === "runList") { runs = msg.runs || []; renderList(); }
    else if (msg.type === "runDetail") { renderDetail(msg.manifest); }
  });

  vscode.postMessage({ type: "ready" });
</script>
</body>
</html>`;
}
```

- [ ] **Step 5: Gates** (compile, lint, pyright clean).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add generateNonce + buildRunManifestsHtml pure helpers — wp-core-32 step B

- generateNonce() returns a 32-char alphanumeric nonce for the webview
  CSP inline-script allowance.
- buildRunManifestsHtml(nonce, cspSource) returns the complete inline
  webview document: CSP locked to default-src 'none' with only the
  nonce'd inline script + cspSource-scoped styles and NO external URLs
  (no network). Vanilla JS renders a sortable summaries table, posts
  openDetail/refresh/ready messages out, and renders runList/runDetail
  messages in. Uses VS Code theme CSS variables; HTML-escapes all
  interpolated values.

9 unit tests cover the 32-char alphanumeric nonce + uniqueness, the
nonce on the script tag and in the CSP, default-src 'none' + cspSource
usage + the no-external-URL guarantee, and the presence of the table /
detail / message-wiring scaffold.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Command registration + `openRunManifestsWebview` wire-up

**Goal:** Register `ddd-enforcer.showRunManifests` (package.json + activate), discover manifests under `runs/`, render the webview, handle `ready`/`openDetail`/`refresh` messages, and auto-refresh via a debounced FileSystemWatcher disposed on panel close. No new pure-function tests (covered by Tasks 1-2 + F5 smoke).

**Files:**
- Modify: `extension/package.json` (add the command)
- Modify: `extension/src/extension.ts` (command registration + `openRunManifestsWebview`)

- [ ] **Step 1: Add the command to package.json**

In `extension/package.json`, `contributes.commands`, append after the `ddd-enforcer.restartBackend` entry (mind the comma):

```json
      {
        "command": "ddd-enforcer.showRunManifests",
        "title": "DDD Enforcer: Show Run Manifests"
      }
```

- [ ] **Step 2: Register the command in `activate`**

Next to the other `registerCommand` calls in `activate` (e.g. near `ddd-enforcer.showOutput` / `ddd-enforcer.openSource`), add:

```typescript
  context.subscriptions.push(
    vscode.commands.registerCommand("ddd-enforcer.showRunManifests", () =>
      openRunManifestsWebview(context),
    ),
  );
```

- [ ] **Step 3: Add `openRunManifestsWebview`**

Add this function near the other webview/UI functions (e.g. after `openSourceCommand`). It uses the pure helpers from Tasks 1-2.

```typescript
/**
 * Resolve the workspace runs/ directory. Prefers the WORKSPACE_PATH env
 * var (set by the extension when it spawns the backend), else the first
 * open workspace folder. Returns undefined when neither is available.
 */
function resolveRunsDir(): string | undefined {
  const fromEnv = process.env.WORKSPACE_PATH;
  if (fromEnv && fromEnv.trim()) {
    return path.join(fromEnv.trim(), "runs");
  }
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (folder) {
    return path.join(folder.uri.fsPath, "runs");
  }
  return undefined;
}

/**
 * Discover PaperRunManifest files (runs/<run_id>/manifest.json), parse +
 * summarize each, and return the summaries plus a runId→absolutePath map
 * for detail lookups. Non-conforming JSON (legacy runs/*.manifest.json
 * observability files, or malformed files) is filtered out by
 * summarizeManifest returning null.
 */
async function discoverRunManifests(
  runsDir: string,
): Promise<{ summaries: RunSummary[]; pathByRunId: Map<string, string> }> {
  const summaries: RunSummary[] = [];
  const pathByRunId = new Map<string, string>();
  let entries: [string, vscode.FileType][] = [];
  try {
    entries = await vscode.workspace.fs.readDirectory(vscode.Uri.file(runsDir));
  } catch {
    return { summaries, pathByRunId };
  }
  for (const [name, fileType] of entries) {
    if (fileType !== vscode.FileType.Directory) {
      continue;
    }
    const manifestPath = path.join(runsDir, name, "manifest.json");
    try {
      const bytes = await vscode.workspace.fs.readFile(
        vscode.Uri.file(manifestPath),
      );
      const parsed = JSON.parse(Buffer.from(bytes).toString("utf8"));
      const summary = summarizeManifest(parsed);
      if (summary) {
        summaries.push(summary);
        pathByRunId.set(summary.runId, manifestPath);
      }
    } catch {
      // Skip missing/unreadable/non-JSON manifest files silently.
    }
  }
  return { summaries, pathByRunId };
}

/**
 * Open a read-only run-manifest webview panel. Lists every conforming
 * PaperRunManifest under the workspace runs/ directory, lets the user
 * sort + click into a detail view, refreshes on demand, and
 * auto-refreshes (debounced) when a manifest file changes. File-load
 * only — no backend communication. Each invocation opens a fresh panel
 * (MVP: no singleton reveal); the FileSystemWatcher is scoped to the
 * panel and disposed on its close.
 */
async function openRunManifestsWebview(context: vscode.ExtensionContext) {
  const runsDir = resolveRunsDir();
  if (!runsDir) {
    vscode.window.showWarningMessage(
      "DDD Enforcer: open a workspace folder to view run manifests.",
    );
    return;
  }

  const panel = vscode.window.createWebviewPanel(
    "dddRunManifests",
    "DDD Enforcer — Run Manifests",
    vscode.ViewColumn.Active,
    { enableScripts: true, retainContextWhenHidden: true },
  );

  const nonce = generateNonce();
  panel.webview.html = buildRunManifestsHtml(nonce, panel.webview.cspSource);

  let pathByRunId = new Map<string, string>();

  const pushList = async () => {
    const result = await discoverRunManifests(runsDir);
    pathByRunId = result.pathByRunId;
    void panel.webview.postMessage({
      type: "runList",
      runs: result.summaries,
    });
  };

  panel.webview.onDidReceiveMessage(
    async (msg: { type: string; runId?: string }) => {
      if (msg.type === "ready" || msg.type === "refresh") {
        await pushList();
      } else if (msg.type === "openDetail" && msg.runId) {
        const manifestPath = pathByRunId.get(msg.runId);
        if (!manifestPath) {
          return;
        }
        try {
          const bytes = await vscode.workspace.fs.readFile(
            vscode.Uri.file(manifestPath),
          );
          const manifest = JSON.parse(Buffer.from(bytes).toString("utf8"));
          void panel.webview.postMessage({ type: "runDetail", manifest });
        } catch {
          vscode.window.showErrorMessage(
            `DDD Enforcer: could not read manifest for ${msg.runId}.`,
          );
        }
      }
    },
    undefined,
    context.subscriptions,
  );

  // Auto-refresh on manifest writes (debounced 500ms).
  const watcher = vscode.workspace.createFileSystemWatcher(
    new vscode.RelativePattern(vscode.Uri.file(runsDir), "**/manifest.json"),
  );
  let debounce: ReturnType<typeof setTimeout> | undefined;
  const scheduleRefresh = () => {
    if (debounce) {
      clearTimeout(debounce);
    }
    debounce = setTimeout(() => {
      void pushList();
    }, 500);
  };
  watcher.onDidCreate(scheduleRefresh);
  watcher.onDidChange(scheduleRefresh);
  watcher.onDidDelete(scheduleRefresh);

  panel.onDidDispose(() => {
    if (debounce) {
      clearTimeout(debounce);
    }
    watcher.dispose();
  });
}
```

`path` and `process` are already imported / available (`import * as path from "path";` at the top; `process` via `@types/node`). `Buffer` is a Node global.

- [ ] **Step 4: Gates**

Run:
- `cd extension && npm run compile && npm run lint`
- `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright`
- `cd extension/backend && pytest -m "not integration" -q`

Expected: compile + lint SUCCESS; pyright `0 errors, 0 warnings, 0 informations`; pytest `729 passed, 31 deselected`.

- [ ] **Step 5: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/package.json extension/src/extension.ts
git commit -m "$(cat <<'EOF'
feat(extension): wire run-manifest webview command + discovery + watcher — wp-core-32 step C

WP-CORE-28 sibling WP-CORE-32 wired end-to-end:

- New ddd-enforcer.showRunManifests command (package.json +
  registerCommand) opens a read-only WebviewPanel.
- resolveRunsDir prefers WORKSPACE_PATH, falls back to the first
  workspace folder; discoverRunManifests reads runs/<run_id>/
  manifest.json files, summarizes each via summarizeManifest (Task A)
  and filters out non-conforming JSON (legacy runs/*.manifest.json
  observability files are ignored).
- The panel renders buildRunManifestsHtml (Task B) and exchanges
  ready/refresh/openDetail messages; detail lookups re-read the
  specific manifest by run id.
- A debounced (500ms) FileSystemWatcher on runs/**/manifest.json
  auto-refreshes the list and is disposed on panel close.

File-load only — no backend endpoint, no backend code change. Pyright
still 0 errors. Pytest still 729 passing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: F5 manual smoke + CURRENT.md (HUMAN-IN-LOOP, deferred)

**Goal:** Verify the live webview. Per the user's standing direction, F5 smoke is DEFERRED — post the checklist for later, update CURRENT.md, and stop.

**Files:**
- Modify: `.planning/pipeline_audit/CURRENT.md`

- [ ] **Step 1: Update CURRENT.md** — append `## WP-CORE-32 — PaperRunManifest webview viewer COMPLETE` in the same shape as the iter-47-50 sections: commits, ~17 new tests, SDD telemetry, locked invariants (glob convention, legacy-file filtering, CSP/nonce, watcher dispose), and an F5-smoke-deferred note. Bump the HEAD + ahead-count lines.

- [ ] **Step 2: Commit CURRENT.md**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add .planning/pipeline_audit/CURRENT.md
git commit -m "$(cat <<'EOF'
chore(planning): CURRENT.md update for WP-CORE-32 (run-manifest webview complete)

PaperRunManifest webview viewer shipped via SDD. File-load only,
vanilla HTML+JS, no backend change. Live F5 smoke deferred per user.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Post the deferred F5 checklist to the user** (for when they pick smoke back up):

**F5 smoke checklist for WP-CORE-32 (run-manifest webview):**
1. `cd extension && npm run compile`. F5 → Extension Dev Host. Open a workspace that has (or will have) a `runs/<run_id>/manifest.json`.
2. Command palette → `DDD Enforcer: Show Run Manifests` → a webview panel opens.
3. With no manifests: friendly empty state ("No run manifests found…").
4. With manifests: a table lists every run (run id, pipeline, model, SRS, timestamp, cost, violations). Click each column header → sorts asc/desc.
5. Click a row → detail pane shows metadata, metrics, provenance hashes, and the violations list (ERROR red / WARN yellow).
6. Click Refresh → re-reads the directory.
7. Write a new `runs/<id>/manifest.json` (or re-run the pipeline) → the list auto-refreshes within ~1s (debounced watcher).
8. Close + reopen the panel → no errors; the watcher from the first panel was disposed.

---

## Self-Review (per writing-plans skill)

**1. Spec coverage** (against `todos/WP-CORE-32-extension-webviews.md` acceptance):

| Acceptance criterion | Task |
|----------------------|------|
| Command appears in palette + opens a WebviewPanel | Task 3 Steps 1-3 |
| List view shows every manifest under workspace runs/ | Task 3 (`discoverRunManifests`) + Task 1 (`summarizeManifest`) |
| Sorting works on all 7 columns | Task 2 (client-side `sortRuns` mirrors Task 1's `sortRunSummaries`); headers wired in the HTML |
| Clicking a row opens the detail pane on the same panel | Task 2 (`openDetail` message + `renderDetail`) + Task 3 (detail message handler) |
| Detail pane shows violations + metrics + provenance hashes | Task 2 (`renderDetail`) |
| Refresh button re-reads the directory | Task 2 (refresh button) + Task 3 (`refresh` handler → `pushList`) |
| File watcher auto-refreshes on new manifest | Task 3 (debounced `createFileSystemWatcher`) |
| Webview JS makes no external network calls | Task 2 (CSP `default-src 'none'`, no http(s) URLs — asserted by a test) |
| No new backend endpoint or backend code change | Whole plan touches only extension TS + package.json |
| 5 new test cases | Substituted with 8 (Task 1) + 9 (Task 2) pure-function tests; the vscode-bound glob/watcher/message-dispatch paths are F5-verified — documented in Pre-flight |
| Manual F5 smoke passes | Task 4 (deferred per user) |

**2. Placeholder scan:** no TBD, no "handle appropriately", no "similar to Task N". Every code step is complete, including the full inline HTML.

**3. Type consistency:**
- `PaperRunManifest` / `ManifestViolation` / `RunSummary` / `RunSummaryColumn` (Task 1) → consumed by `summarizeManifest` + `sortRunSummaries` (Task 1) and `discoverRunManifests` + the message payloads (Task 3).
- `generateNonce` + `buildRunManifestsHtml` (Task 2) → consumed by `openRunManifestsWebview` (Task 3).
- The client-side `sortRuns` in the HTML (Task 2) intentionally mirrors `sortRunSummaries`'s numeric-vs-string logic (Task 1) — the numeric columns set (`costUsd`, `violationCount`) is identical in both.
- Message protocol is consistent: extension → webview `{type:"runList", runs}` / `{type:"runDetail", manifest}`; webview → extension `{type:"ready"}` / `{type:"refresh"}` / `{type:"openDetail", runId}`.

End of plan.
