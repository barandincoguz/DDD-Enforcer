/**
 * DDD Enforcer VS Code Extension
 * HTML Page Builder for Run Manifests Webview
 */

import * as crypto from "crypto";

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

/**
 * Generate a 32-character alphanumeric nonce for the webview CSP.
 * Pure-enough for testing (uses Math.random; not used for cryptographic
 * secrecy — only to satisfy VS Code's inline-script CSP requirement).
 */
export function generateNonce(): string {
  return crypto.randomBytes(32).toString("base64").replace(/[^A-Za-z0-9]/g, "").slice(0, 32);
}

/**
 * Build the complete inline HTML document for the run-manifest webview.
 * Locks the CSP to `default-src 'none'`, allows only the nonce'd inline
 * script and `cspSource`-scoped inline styles, and contains no external
 * URLs (no network).
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
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
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
