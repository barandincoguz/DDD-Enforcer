/**
 * DDD Enforcer VS Code Extension
 *
 * Validates Python code against Domain-Driven Design rules on save.
 * Extracts domain model from SRS/design documents using AI.
 * Shows violations as diagnostics with clickable source references.
 *
 * @version 1.0.0
 */

import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import * as net from "net";
import { ChildProcess, spawn } from "child_process";
import axios from "axios";

// =============================================================================
// TYPES
// =============================================================================

/** Source reference for a violation from the RAG pipeline */
interface ViolationSource {
  document: string;
  section: string;
  page: number;
  summary: string;
  file_path: string;
  relevance_score: number;
}

/** Single DDD violation detected in code */
interface Violation {
  type: string;
  message: string;
  suggestion: string;
  sources?: ViolationSource[];
}

/** Metrics from validation */
interface ValidationMetrics {
  validation_time_ms: number;
  code_file_tokens: number;
  llm_input_tokens: number;
  llm_output_tokens: number;
  llm_total_tokens: number;
  cost_usd: number;
  api_calls: number;
}

/** Response from the backend validation endpoint */
interface ValidationResponse {
  is_violation: boolean;
  violations: Violation[];
  metrics?: ValidationMetrics;
}

/** Response from the backend health endpoint */
interface HealthResponse {
  status: string;
  domain_model_loaded: boolean;
  rag_initialized: boolean;
}

/** Response from the generate-model endpoint */
interface GenerateModelResponse {
  success: boolean;
  error?: string;
  model_path?: string;
  project_name?: string;
  bounded_contexts_count?: number;
  metrics?: CombinedMetrics;
}

/** Progress update from streaming endpoint */
interface PipelineProgress {
  stage: string;
  status: "started" | "in_progress" | "completed" | "error";
  detail: string;
  progress: number;
}

/** SSE event from streaming endpoint */
interface SSEEvent {
  type: "progress" | "complete" | "error" | "heartbeat";
  data?: PipelineProgress | GenerateModelResponse;
  error?: string;
}

/** Token usage metrics */
interface CombinedMetrics {
  total_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
  total_cost_usd: number;
  api_calls: number;
  by_stage: Record<
    string,
    {
      tokens: number;
      input_tokens: number;
      output_tokens: number;
      cost_usd: number;
      api_calls: number;
    }
  >;
}

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

// =============================================================================
// PIPELINE PROGRESS (pure helpers — testable without vscode)
// =============================================================================

/** globalState key under which the last run's per-stage durations (ms) are persisted. */
export const LAST_RUN_DURATIONS_KEY = "ddd-enforcer.lastRunStageDurations";

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

/** A within-stage progress counter (e.g. context 2 of 5). */
export type SubProgress = { current: number; total: number };

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
 * only when both are positive integers with total > 0. Returns null when
 * no valid ratio is found. Pure.
 *
 * `detail` is trusted to be pipeline progress text (e.g. "Analyzing
 * context 2/5"), never a date — a date like "2024/05/24" would
 * mis-match as {2024, 5}, but the backend never emits dates here.
 */
export function parseSubProgress(
  detail: string,
): SubProgress | null {
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

/** Inputs for the status-bar text formatter. */
export interface StageStatusBarParts {
  /** Current pipeline stage name (e.g. "Specialist"). */
  stage: string;
  /** Overall completion percentage 0-100 (rounded in the output). */
  overallPercent: number;
  /** Whether the pipeline is still running (spinner) or done (check). */
  active: boolean;
  /** Optional within-stage N/M counter parsed from the detail text. */
  sub?: SubProgress;
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

// =============================================================================
// VALIDATION HOVER (pure helpers — testable without vscode)
// =============================================================================

/**
 * A minimal capacity-bounded cache with least-recently-used eviction.
 * Backed by a Map, which preserves insertion order; `get` and `set`
 * re-insert the touched key so the iteration order tracks recency
 * (oldest first). When `set` would exceed `capacity`, the oldest key is
 * evicted. Pure: no I/O, no vscode. `capacity` must be >= 1.
 */
export class LruCache<K, V> {
  private readonly store = new Map<K, V>();

  constructor(private readonly capacity: number) {
    if (capacity < 1) {
      throw new Error("LruCache capacity must be >= 1");
    }
  }

  get size(): number {
    return this.store.size;
  }

  has(key: K): boolean {
    return this.store.has(key);
  }

  get(key: K): V | undefined {
    if (!this.store.has(key)) {
      return undefined;
    }
    const value = this.store.get(key) as V;
    // Promote recency: re-insert so this key becomes most-recent.
    this.store.delete(key);
    this.store.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    // Re-insert to promote recency (and update value).
    if (this.store.has(key)) {
      this.store.delete(key);
    }
    this.store.set(key, value);
    // Evict oldest while over capacity.
    while (this.store.size > this.capacity) {
      const oldest = this.store.keys().next().value as K;
      this.store.delete(oldest);
    }
  }

  delete(key: K): boolean {
    return this.store.delete(key);
  }
}

/**
 * Trim `text` to at most `maxChars`, preferring to cut at the last
 * word boundary at or before the limit, and append a single-character
 * ellipsis (…). Returns the text unchanged when already within the
 * limit. Pure.
 */
export function truncateExcerpt(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }
  const hardCut = text.slice(0, maxChars);
  const lastSpace = hardCut.lastIndexOf(" ");
  const trimmed =
    lastSpace > 0 ? hardCut.slice(0, lastSpace) : hardCut;
  return `${trimmed.trimEnd()}…`;
}

/**
 * Bold the first case-insensitive occurrence of `keyword` in `excerpt`
 * using Markdown `**…**`, preserving the original casing of the matched
 * span. Returns the excerpt unchanged when the keyword is empty or not
 * found. Pure.
 */
export function boldMatchingSpan(excerpt: string, keyword: string): string {
  if (!keyword) {
    return excerpt;
  }
  const lowerExcerpt = excerpt.toLowerCase();
  const idx = lowerExcerpt.indexOf(keyword.toLowerCase());
  if (idx < 0) {
    return excerpt;
  }
  const before = excerpt.slice(0, idx);
  const matched = excerpt.slice(idx, idx + keyword.length);
  const after = excerpt.slice(idx + keyword.length);
  return `${before}**${matched}**${after}`;
}

/**
 * Build the Markdown body for a validation hover. Includes the violation
 * type as a header, the message, and — when the violation has at least
 * one source — a source block with the section/document/page, a bolded,
 * truncated excerpt from the source summary, and an "Open SRS source"
 * link that invokes the `ddd-enforcer.openSource` command with the same
 * (file_path, section) arguments the Code Action uses.
 *
 * The returned string is intended to be wrapped in a trusted
 * `vscode.MarkdownString` by the caller (the command link only fires
 * when `isTrusted` is set). Pure: no vscode, no I/O.
 */
export function formatHoverMarkdown(
  violation: Violation,
  keyword: string,
): string {
  const lines: string[] = [];
  lines.push(`**DDD Violation: ${violation.type}**`);
  lines.push("");
  lines.push(violation.message);

  const source = violation.sources && violation.sources[0];
  if (source) {
    lines.push("");
    lines.push(`**Source:** ${source.section} — ${source.document} (p. ${source.page})`);
    const excerpt = boldMatchingSpan(
      truncateExcerpt(source.summary, 200),
      keyword,
    );
    lines.push("");
    lines.push(`> ${excerpt}`);
    const args = encodeURIComponent(
      JSON.stringify([source.file_path, source.section]),
    );
    lines.push("");
    lines.push(`[Open SRS source](command:ddd-enforcer.openSource?${args})`);
  }

  return lines.join("\n");
}

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

// =============================================================================
// GLOBAL STATE
// =============================================================================

let backendProcess: ChildProcess | null = null;
let backendPort: number = 8000;
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;
let isBackendReady: boolean = false;
let backendStarting: boolean = false;

/** Reference to the specific ChildProcess that stopBackend has intentionally killed. Captured at registration time by each child.on('exit') handler so per-process exits are judged against the correct target — protects against the race where a slow-exiting old process is misclassified as a crash after a new process has already started. Reset to null after the matched exit fires. */
let intentionalStopTarget: ChildProcess | null = null;

/** Number of consecutive auto-restart attempts since the last successful boot. Reset to 0 when the backend reaches the ready state. Bounded by shouldAttemptRestart. */
let backendRestartAttempts: number = 0;

// Track last validated semantic content per document.
// Semantic fingerprint ignores whitespace/comment-only edits.
const lastValidatedContentFingerprint = new Map<string, string>();

// Store sources for code actions (keyed by document URI + line number)
const violationSources = new Map<string, ViolationSource[]>();

/**
 * Per-file cache of the most recent validation's violations, keyed by
 * document URI string, value = Map of diagnostic line number → Violation.
 * Drives the HoverProvider without a re-fetch. LRU-bounded to 20 files;
 * invalidated by clearSourcesForDocument (which runs on every validate).
 */
const validationViolationCache = new LruCache<
  string,
  Map<number, Violation>
>(20);

// =============================================================================
// EXTENSION ACTIVATION
// =============================================================================

/**
 * Called when the extension is activated.
 * Sets up diagnostics, commands, and event handlers.
 */
export function activate(context: vscode.ExtensionContext) {
  // Create output channel for logging
  outputChannel = vscode.window.createOutputChannel("DDD Enforcer");
  log("DDD Enforcer is activating...");

  // Create status bar item
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100,
  );
  statusBarItem.command = "ddd-enforcer.showStatus";
  context.subscriptions.push(statusBarItem);

  // Show status bar if enabled
  const config = vscode.workspace.getConfiguration("ddd-enforcer");
  if (config.get<boolean>("showStatusBar", true)) {
    updateStatusBar("inactive");
    statusBarItem.show();
  }

  // Create diagnostic collection
  const diagnosticCollection =
    vscode.languages.createDiagnosticCollection("ddd-enforcer");
  context.subscriptions.push(diagnosticCollection);

  // Register commands
  context.subscriptions.push(
    vscode.commands.registerCommand("ddd-enforcer.initializeDomainModel", () =>
      initializeDomainModel(context),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ddd-enforcer.validateCurrentFile", () =>
      validateCurrentFile(context, diagnosticCollection),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ddd-enforcer.showStatus", showStatus),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ddd-enforcer.restartBackend", () =>
      restartBackend(context),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ddd-enforcer.showOutput", () =>
      outputChannel.show(),
    ),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand(
      "ddd-enforcer.openSource",
      openSourceCommand,
    ),
  );

  // Register code action provider
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      "python",
      new DDDSourceCodeActionProvider(),
      { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] },
    ),
  );

  // Register hover provider for violation peeks
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(
      "python",
      new DDDViolationHoverProvider(),
    ),
  );

  // Validate on save (lazy start - starts backend on first save)
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
      if (
        document.languageId === "python" &&
        cfg.get<boolean>("validateOnSave", true)
      ) {
        // Skip validation for files inside extension's backend folder
        if (shouldSkipValidation(document.uri.fsPath, context)) {
          return;
        }

        const decision = shouldValidateOnSave(document);
        if (!decision.shouldValidate) {
          log(
            `Validate skip: ${decision.reason} (${path.basename(document.fileName)})`,
          );
          return;
        }

        log(
          `Validate trigger: semantic code change (${path.basename(document.fileName)})`,
        );

        await ensureBackendRunning(context);
        if (isBackendReady) {
          await validateCode(document, diagnosticCollection);
        }
      }
    }),
  );

  // Cleanup on deactivation
  context.subscriptions.push({
    dispose: () => stopBackend(),
  });

  log("DDD Enforcer activated successfully!");
}

export function deactivate() {
  log("DDD Enforcer deactivating...");
  stopBackend();
}

// =============================================================================
// BACKEND MANAGEMENT
// =============================================================================

/**
 * Ensures the backend server is running, starts it if not.
 */
async function ensureBackendRunning(
  context: vscode.ExtensionContext,
): Promise<boolean> {
  // Already running?
  if (isBackendReady) {
    return true;
  }

  // Already starting?
  if (backendStarting) {
    // Wait for it to finish starting
    return waitForBackend(30000);
  }

  return startBackend(context);
}

/**
 * Starts the backend Python server.
 */
async function startBackend(
  context: vscode.ExtensionContext,
): Promise<boolean> {
  if (backendStarting) {
    return false;
  }

  backendStarting = true;
  updateStatusBar("starting");
  log("Starting backend server...");

  try {
    // Get API key
    const apiKey = await getApiKey(context);
    if (!apiKey) {
      vscode.window.showErrorMessage(
        "DDD Enforcer: Gemini API Key is required. Please configure it in settings or provide when prompted.",
      );
      backendStarting = false;
      updateStatusBar("error");
      return false;
    }

    // Find available port
    backendPort = await findAvailablePort();
    log(`Using port: ${backendPort}`);

    // Get paths
    const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
    const pythonPath = cfg.get<string>("pythonPath", "python3");
    const backendPath = getBackendPath(context);

    if (!backendPath || !fs.existsSync(backendPath)) {
      vscode.window.showErrorMessage(
        `DDD Enforcer: Backend not found at ${backendPath}`,
      );
      backendStarting = false;
      updateStatusBar("error");
      return false;
    }

    // Get workspace path for domain model
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    const workspacePath = workspaceFolder?.uri.fsPath || "";

    log(`Backend path: ${backendPath}`);
    log(`Python path: ${pythonPath}`);
    log(`Workspace path: ${workspacePath}`);

    // Start the backend process
    backendProcess = spawn(
      pythonPath,
      [
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        "127.0.0.1",
        "--port",
        backendPort.toString(),
      ],
      {
        cwd: backendPath,
        env: {
          ...process.env,
          GEMINI_API_KEY: apiKey,
          WORKSPACE_PATH: workspacePath,
          PYTHONUNBUFFERED: "1",
        },
      },
    );

    // Handle stdout
    backendProcess.stdout?.on("data", (data) => {
      const message = data.toString().trim();
      log(`[Backend] ${message}`);
    });

    // Handle stderr
    backendProcess.stderr?.on("data", (data) => {
      const message = data.toString().trim();
      log(`[Backend] ${message}`);
    });

    // Capture this specific process so the exit handler can detect
    // whether it was intentionally stopped (per-process flag protects
    // against the race where a slow-exiting old process fires its exit
    // event after a new process has already started).
    const thisProcess = backendProcess;

    // Handle process exit
    backendProcess.on("exit", (code, signal) => {
      const reason = formatExitReason(code, signal);
      const wasIntentional = intentionalStopTarget === thisProcess;
      if (wasIntentional) {
        intentionalStopTarget = null;
      }
      const disposition = classifyExitForRestart(code, signal, wasIntentional);
      log(`Backend process ${reason} (disposition: ${disposition}).`);
      isBackendReady = false;
      backendStarting = false;
      backendProcess = null;
      if (disposition === "crash") {
        updateStatusBar("error");
        handleUnexpectedExit(context, reason).catch((err) =>
          log(
            `handleUnexpectedExit failed: ${err instanceof Error ? err.message : String(err)}`,
          ),
        );
      } else {
        updateStatusBar("inactive");
      }
    });

    // Handle errors
    backendProcess.on("error", (err) => {
      log(`Backend process error: ${err.message}`);
      vscode.window.showErrorMessage(
        `DDD Enforcer: Failed to start backend - ${err.message}`,
      );
      isBackendReady = false;
      backendStarting = false;
      updateStatusBar("error");
    });

    // Wait for backend to be ready
    const ready = await waitForBackend(60000);
    backendStarting = false;

    if (ready) {
      log("Backend server is ready!");
      isBackendReady = true;
      backendRestartAttempts = 0;

      // Check if domain model is loaded and update status accordingly
      await updateStatusFromBackend();
      return true;
    } else {
      log("Backend failed to start within timeout");
      updateStatusBar("error");
      return false;
    }
  } catch (error) {
    log(`Error starting backend: ${error}`);
    backendStarting = false;
    updateStatusBar("error");
    return false;
  }
}

/**
 * Waits for the backend to respond to health checks.
 */
async function waitForBackend(timeoutMs: number): Promise<boolean> {
  const startTime = Date.now();
  const checkInterval = 1000;

  while (Date.now() - startTime < timeoutMs) {
    try {
      const response = await axios.get<HealthResponse>(
        `http://127.0.0.1:${backendPort}/health`,
        { timeout: 2000 },
      );
      if (response.data.status === "healthy") {
        return true;
      }
    } catch {
      // Backend not ready yet
    }
    await sleep(checkInterval);
  }

  return false;
}

/**
 * Stops the backend server.
 */
function stopBackend() {
  if (backendProcess) {
    log("Stopping backend server...");
    intentionalStopTarget = backendProcess;
    backendProcess.kill();
    backendProcess = null;
    isBackendReady = false;
    backendStarting = false;
    updateStatusBar("inactive");
  }
}

/**
 * Restarts the backend server.
 */
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

// =============================================================================
// API KEY MANAGEMENT
// =============================================================================

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
      let settingsClearFailed = false;
      if (source === "settings") {
        try {
          await cfg.update(
            "geminiApiKey",
            "",
            vscode.ConfigurationTarget.Global,
          );
        } catch (err) {
          settingsClearFailed = true;
          log(
            `API key copied to secret storage but failed to clear settings entry: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      }
      if (settingsClearFailed) {
        vscode.window.showWarningMessage(
          "DDD Enforcer: API key saved to secret storage, but could not clear the existing setting. Please remove ddd-enforcer.geminiApiKey from your settings manually.",
        );
      } else {
        log(`API key migrated from ${decision.sourceLabel} to secret storage.`);
        vscode.window.showInformationMessage(
          "DDD Enforcer: Gemini API key moved to secret storage.",
        );
      }
    } else if (choice === "Don't ask again") {
      await context.globalState.update("apiKeyMigrationDeclined", true);
      log("API key migration permanently declined by user.");
    }
    // "Not now" or dismiss: no state change; the offer will re-appear on the next getApiKey call.
  }

  return apiKey;
}

// =============================================================================
// DOMAIN MODEL INITIALIZATION
// =============================================================================

/**
 * Command: Initialize Domain Model
 * Opens file picker for SRS documents and generates model.json with streaming progress.
 */
async function initializeDomainModel(context: vscode.ExtensionContext) {
  log("Initializing domain model...");

  // Show file picker
  const files = await vscode.window.showOpenDialog({
    canSelectMany: true,
    openLabel: "Select SRS/Design Documents",
    filters: {
      Documents: ["pdf", "docx", "txt"],
    },
    title: "Select SRS or Design Documents for Domain Model Generation",
  });

  if (!files || files.length === 0) {
    vscode.window.showWarningMessage("DDD Enforcer: No files selected.");
    return;
  }

  // Get workspace folder
  const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
  if (!workspaceFolder) {
    vscode.window.showErrorMessage(
      "DDD Enforcer: Please open a workspace folder first.",
    );
    return;
  }

  // Ensure backend is running
  updateStatusBar("starting");
  const backendReady = await ensureBackendRunning(context);
  if (!backendReady) {
    vscode.window.showErrorMessage(
      "DDD Enforcer: Could not start backend server.",
    );
    return;
  }

  const filePaths = files.map((f) => f.fsPath);
  const outputPath = path.join(
    workspaceFolder.uri.fsPath,
    "domain",
    "model.json",
  );

  log(`Generating model from: ${filePaths.join(", ")}`);
  log(`Output path: ${outputPath}`);

  // Show progress with streaming updates
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "DDD Enforcer: Generating Domain Model",
      cancellable: false,
    },
    async (progress) => {
      return new Promise<void>((resolve, reject) => {
        // Try streaming endpoint first
        generateModelWithStreaming(
          context,
          filePaths,
          outputPath,
          progress,
          resolve,
          reject,
        );
      });
    },
  );
}

/**
 * Generate domain model using streaming endpoint for real-time progress.
 */
async function generateModelWithStreaming(
  context: vscode.ExtensionContext,
  filePaths: string[],
  outputPath: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  resolve: () => void,
  reject: (error: Error) => void,
) {
  const stageEmojis: Record<string, string> = {
    Scout: "🔍",
    Architect: "🏛️",
    Specialist: "🔬",
    Synthesizer: "🔧",
  };

  const stageDescriptions: Record<string, string> = {
    Scout: "Extracting domain sentences",
    Architect: "Identifying bounded contexts",
    Specialist: "Analyzing context details",
    Synthesizer: "Creating final model",
  };

  let currentStage = "";
  let finalResult: GenerateModelResponse | null = null;

  const runStartMs = Date.now();
  const stageStartMs = new Map<string, number>();
  const stageDurations: Record<string, number> = {};
  const previousCommand = statusBarItem.command;
  statusBarItem.command = "ddd-enforcer.showOutput";

  try {
    // Use fetch for SSE support
    const response = await fetch(
      `http://127.0.0.1:${backendPort}/generate-model-stream`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          file_paths: filePaths,
          output_path: outputPath,
        }),
      },
    );

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("No response body");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const event: SSEEvent = JSON.parse(line.slice(6));

            if (event.type === "progress" && event.data) {
              const progressData = event.data as PipelineProgress;
              const emoji = stageEmojis[progressData.stage] || "⚙️";
              const desc =
                stageDescriptions[progressData.stage] || progressData.detail;

              // Track per-stage timing for ETA + persistence.
              if (progressData.stage !== currentStage) {
                if (currentStage && stageStartMs.has(currentStage)) {
                  stageDurations[currentStage] =
                    Date.now() - (stageStartMs.get(currentStage) ?? Date.now());
                }
                currentStage = progressData.stage;
                stageStartMs.set(currentStage, Date.now());
              }

              // Compute overall percent. A "completed" status means the
              // current stage is fully done (100%); any other status uses a
              // coarse 50% mid-stage placeholder — the backend does not emit
              // a fine-grained within-stage fraction, so this keeps the
              // overall-percent monotonic without over-promising precision.
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

              // Update progress notification
              if (progressData.status === "started") {
                progress.report({
                  message: `${emoji} ${progressData.stage}: ${desc}...`,
                });
                log(`Stage started: ${progressData.stage}`);
              } else if (progressData.status === "in_progress") {
                progress.report({
                  message: `${emoji} ${progressData.stage}: ${progressData.detail}`,
                });
              } else if (progressData.status === "completed") {
                progress.report({
                  message: `${emoji} ${progressData.stage}: ✓ Complete`,
                });
                log(
                  `Stage completed: ${progressData.stage} - ${progressData.detail}`,
                );
              }
            } else if (event.type === "complete" && event.data) {
              finalResult = event.data as GenerateModelResponse;
            } else if (event.type === "error") {
              throw new Error(event.error || "Unknown error");
            }
          } catch (parseError) {
            // Ignore parse errors for non-JSON lines, but keep lightweight trace.
            log("Streaming parse warning: skipped malformed SSE data line.");
          }
        }
      }
    }

    // If stream ended without a completion payload, fallback to non-stream API.
    if (!finalResult) {
      throw new Error("Streaming ended without completion payload");
    }

    // Handle completion
    if (finalResult?.success) {
      updateStatusBar("ready");
      if (currentStage && stageStartMs.has(currentStage)) {
        stageDurations[currentStage] =
          Date.now() - (stageStartMs.get(currentStage) ?? Date.now());
      }
      await context.globalState.update(LAST_RUN_DURATIONS_KEY, stageDurations);
      statusBarItem.command = previousCommand;

      // Show success message with metrics
      const metricsInfo = finalResult.metrics
        ? `\n\n📊 Metrics:\n` +
          `• Total Tokens: ${finalResult.metrics.total_tokens.toLocaleString()}\n` +
          `• API Calls: ${finalResult.metrics.api_calls}\n` +
          `• Cost: $${finalResult.metrics.total_cost_usd.toFixed(4)}`
        : "";

      setTimeout(async () => {
        const openAction = "Open Model";
        const viewMetrics = "View Details";
        const actions = finalResult?.metrics
          ? [openAction, viewMetrics]
          : [openAction];

        const result = await vscode.window.showInformationMessage(
          `DDD Enforcer: Domain Model created successfully!\n` +
            `Project: ${finalResult?.project_name}\n` +
            `Bounded Contexts: ${finalResult?.bounded_contexts_count}` +
            metricsInfo,
          ...actions,
        );

        if (result === openAction && finalResult?.model_path) {
          const doc = await vscode.workspace.openTextDocument(
            finalResult.model_path,
          );
          await vscode.window.showTextDocument(doc);
        } else if (result === viewMetrics && finalResult?.metrics) {
          showMetricsDetails(finalResult.metrics);
        }
      }, 100);

      resolve();
    } else {
      updateStatusBar("error");
      statusBarItem.command = previousCommand;
      vscode.window.showErrorMessage(
        `DDD Enforcer: Failed to generate model - ${finalResult?.error || "Unknown error"}`,
      );
      resolve();
    }
  } catch (error) {
    // Fallback to non-streaming endpoint
    statusBarItem.command = previousCommand;
    log(`Streaming failed, using fallback: ${error}`);
    await generateModelFallback(
      filePaths,
      outputPath,
      progress,
      resolve,
      reject,
    );
  }
}

/**
 * Fallback to non-streaming endpoint if streaming fails.
 */
async function generateModelFallback(
  filePaths: string[],
  outputPath: string,
  progress: vscode.Progress<{ message?: string; increment?: number }>,
  resolve: () => void,
  reject: (error: Error) => void,
) {
  progress.report({ message: "Analyzing documents..." });

  try {
    const response = await axios.post<GenerateModelResponse>(
      `http://127.0.0.1:${backendPort}/generate-model`,
      {
        file_paths: filePaths,
        output_path: outputPath,
      },
      { timeout: 300000 },
    );

    if (response.data.success) {
      progress.report({ message: "Domain model created!" });
      updateStatusBar("ready");

      setTimeout(async () => {
        const openAction = "Open Model";
        const result = await vscode.window.showInformationMessage(
          `DDD Enforcer: Domain Model created successfully!\n` +
            `Project: ${response.data.project_name}\n` +
            `Bounded Contexts: ${response.data.bounded_contexts_count}`,
          openAction,
        );

        if (result === openAction && response.data.model_path) {
          const doc = await vscode.workspace.openTextDocument(
            response.data.model_path,
          );
          await vscode.window.showTextDocument(doc);
        }
      }, 100);
      resolve();
    } else {
      vscode.window.showErrorMessage(
        `DDD Enforcer: Failed to generate model - ${response.data.error}`,
      );
      updateStatusBar("error");
      resolve();
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    log(`Error generating model: ${errorMessage}`);
    vscode.window.showErrorMessage(
      `DDD Enforcer: Error generating model - ${errorMessage}`,
    );
    updateStatusBar("error");
    resolve();
  }
}

/**
 * Render live pipeline progress into the status bar using the pure
 * formatStageStatusBar helper. `active` is false only on the terminal
 * "completed" status of the final stage.
 */
function updateStatusBarWithProgress(
  stage: string,
  overallPercent: number,
  active: boolean,
  sub?: SubProgress,
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
 * Show detailed metrics in output channel.
 */
function showMetricsDetails(metrics: CombinedMetrics) {
  outputChannel.show();
  outputChannel.appendLine("\n" + "=".repeat(60));
  outputChannel.appendLine("📊 DOMAIN MODEL GENERATION METRICS");
  outputChannel.appendLine("=".repeat(60));
  outputChannel.appendLine(`\n📈 Summary:`);
  outputChannel.appendLine(
    `   Total Tokens: ${metrics.total_tokens.toLocaleString()}`,
  );
  outputChannel.appendLine(
    `   Input Tokens: ${metrics.total_input_tokens.toLocaleString()}`,
  );
  outputChannel.appendLine(
    `   Output Tokens: ${metrics.total_output_tokens.toLocaleString()}`,
  );
  outputChannel.appendLine(`   API Calls: ${metrics.api_calls}`);
  outputChannel.appendLine(
    `   Total Cost: $${metrics.total_cost_usd.toFixed(4)} USD`,
  );

  outputChannel.appendLine(`\n📋 By Stage:`);
  for (const [stage, data] of Object.entries(metrics.by_stage)) {
    outputChannel.appendLine(`   ${stage}:`);
    outputChannel.appendLine(`      Tokens: ${data.tokens.toLocaleString()}`);
    outputChannel.appendLine(`      API Calls: ${data.api_calls}`);
    outputChannel.appendLine(`      Cost: $${data.cost_usd.toFixed(4)}`);
  }
  outputChannel.appendLine("=".repeat(60) + "\n");
}

// =============================================================================
// VALIDATION
// =============================================================================

/**
 * Validates the currently active Python file.
 */
async function validateCurrentFile(
  context: vscode.ExtensionContext,
  diagnosticCollection: vscode.DiagnosticCollection,
) {
  const editor = vscode.window.activeTextEditor;
  if (!editor || editor.document.languageId !== "python") {
    vscode.window.showWarningMessage(
      "DDD Enforcer: Please open a Python file to validate.",
    );
    return;
  }

  await ensureBackendRunning(context);
  if (isBackendReady) {
    await validateCode(editor.document, diagnosticCollection);
  }
}

/**
 * Validates Python code against DDD rules via the backend API.
 */
async function validateCode(
  document: vscode.TextDocument,
  collection: vscode.DiagnosticCollection,
) {
  // Clear previous diagnostics and sources
  collection.delete(document.uri);
  clearSourcesForDocument(document.uri.toString());

  const codeContent = document.getText();
  const fileName = document.fileName;

  updateStatusBar("validating");
  log(`Validating: ${fileName}`);

  try {
    const response = await axios.post<ValidationResponse>(
      `http://127.0.0.1:${backendPort}/validate`,
      { filename: fileName, content: codeContent },
      { timeout: 30000 },
    );

    // Mark this normalized content as validated after successful API call
    lastValidatedContentFingerprint.set(
      document.uri.toString(),
      getValidationFingerprint(document.getText()),
    );

    const data = response.data;

    // Log metrics if available
    if (data.metrics) {
      const m = data.metrics;
      log(`📊 Validation Metrics:`);
      log(`   ⏱️  Time: ${m.validation_time_ms.toFixed(2)}ms`);
      log(`   📝 Code tokens: ${m.code_file_tokens.toLocaleString()}`);
      log(
        `   🤖 LLM tokens: ${m.llm_total_tokens.toLocaleString()} (in: ${m.llm_input_tokens.toLocaleString()}, out: ${m.llm_output_tokens.toLocaleString()})`,
      );
      log(`   💰 Cost: $${m.cost_usd.toFixed(6)}`);
    }

    if (data.is_violation && data.violations) {
      const diagnostics: vscode.Diagnostic[] = [];
      const lineToViolation = new Map<number, Violation>();

      data.violations.forEach((violation) => {
        const diagnostic = createDiagnostic(document, violation);
        diagnostics.push(diagnostic);
        // Cache by diagnostic line for the hover provider. If two
        // violations land on the same line, the last one wins (the hover
        // shows one violation per line); the DiagnosticCollection still
        // holds all of them.
        lineToViolation.set(diagnostic.range.start.line, violation);
      });

      collection.set(document.uri, diagnostics);
      validationViolationCache.set(document.uri.toString(), lineToViolation);
      log(`Found ${diagnostics.length} violation(s)`);
      updateStatusBar("violations", diagnostics.length);
    } else {
      log("No violations found");
      updateStatusBar("ready");
    }
  } catch (error: unknown) {
    const axiosError = error as { code?: string; message?: string };
    log(`Validation error: ${axiosError.message}`);

    if (axiosError.code === "ECONNREFUSED") {
      vscode.window.showErrorMessage(
        "DDD Enforcer: Backend server not running. Use 'DDD Enforcer: Restart Backend Server' to start it.",
      );
      isBackendReady = false;
      updateStatusBar("error");
    } else {
      vscode.window.showErrorMessage(
        `DDD Enforcer: Validation error - ${axiosError.message}`,
      );
    }
  }
}

/**
 * Returns true when this save should trigger backend validation.
 * Uses normalized content fingerprint (blank lines removed) so empty-line-only
 * edits do not trigger backend validation.
 */
function shouldValidateOnSave(document: vscode.TextDocument): {
  shouldValidate: boolean;
  reason: string;
} {
  const key = document.uri.toString();
  const lastFingerprint = lastValidatedContentFingerprint.get(key);
  return classifySaveForValidation(lastFingerprint, document.getText());
}

/**
 * Creates a stable semantic fingerprint for validation decisions.
 * Ignores comments and whitespace outside string literals.
 */
function getValidationFingerprint(content: string): string {
  return normalizePythonSemantics(content);
}

/**
 * Test helper: compare two file snapshots and return validate/skip decision.
 */
export function classifySaveForValidation(
  previousFingerprint: string | undefined,
  currentContent: string,
): { shouldValidate: boolean; reason: string } {
  const curr = getValidationFingerprint(currentContent);

  if (previousFingerprint === undefined) {
    return { shouldValidate: true, reason: "first semantic snapshot" };
  }
  if (previousFingerprint === curr) {
    return { shouldValidate: false, reason: "non-semantic change" };
  }
  return { shouldValidate: true, reason: "semantic code change" };
}

/**
 * Test helper: compare raw text snapshots directly.
 */
export function classifySaveForValidationFromContent(
  previousContent: string | undefined,
  currentContent: string,
): { shouldValidate: boolean; reason: string } {
  const prev = previousContent
    ? getValidationFingerprint(previousContent)
    : undefined;
  return classifySaveForValidation(prev, currentContent);
}

/**
 * Lightweight Python semantic normalization:
 * - removes comments outside strings
 * - removes whitespace outside strings
 * - preserves string literal content
 */
function normalizePythonSemantics(content: string): string {
  let result = "";
  let i = 0;
  let inSingle = false;
  let inDouble = false;
  let inTripleSingle = false;
  let inTripleDouble = false;
  let escaped = false;

  const isWhitespace = (ch: string) => /\s/.test(ch);

  while (i < content.length) {
    const ch = content[i];
    const next3 = content.slice(i, i + 3);

    if (inTripleSingle) {
      result += ch;
      if (next3 === "'''") {
        result += "''";
        i += 3;
        inTripleSingle = false;
        continue;
      }
      i += 1;
      continue;
    }

    if (inTripleDouble) {
      result += ch;
      if (next3 === '"""') {
        result += '""';
        i += 3;
        inTripleDouble = false;
        continue;
      }
      i += 1;
      continue;
    }

    if (inSingle) {
      result += ch;
      if (!escaped && ch === "'") {
        inSingle = false;
      }
      escaped = !escaped && ch === "\\";
      i += 1;
      continue;
    }

    if (inDouble) {
      result += ch;
      if (!escaped && ch === '"') {
        inDouble = false;
      }
      escaped = !escaped && ch === "\\";
      i += 1;
      continue;
    }

    // Outside string literals
    if (next3 === "'''") {
      inTripleSingle = true;
      result += "'''";
      i += 3;
      continue;
    }

    if (next3 === '"""') {
      inTripleDouble = true;
      result += '"""';
      i += 3;
      continue;
    }

    if (ch === "'") {
      inSingle = true;
      escaped = false;
      result += ch;
      i += 1;
      continue;
    }

    if (ch === '"') {
      inDouble = true;
      escaped = false;
      result += ch;
      i += 1;
      continue;
    }

    // Strip comments outside strings
    if (ch === "#") {
      while (i < content.length && content[i] !== "\n") {
        i += 1;
      }
      continue;
    }

    // Strip whitespace outside strings
    if (isWhitespace(ch)) {
      i += 1;
      continue;
    }

    result += ch;
    i += 1;
  }

  return result;
}

/**
 * Creates a diagnostic for a single violation.
 */
function createDiagnostic(
  document: vscode.TextDocument,
  violation: Violation,
): vscode.Diagnostic {
  const keyword = extractKeyword(violation.message);
  const range = findKeywordRange(document, keyword);

  let message = `${violation.message}\n\n💡 Suggestion: ${violation.suggestion}`;

  if (violation.sources && violation.sources.length > 0) {
    const sourceRefs = violation.sources
      .map((src, i) => `[${i + 1}] ${src.section}`)
      .join("  ");
    message += `\n\n📚 Sources: ${sourceRefs}`;

    const key = `${document.uri.toString()}-${range.start.line}`;
    violationSources.set(key, violation.sources);
  }

  const diagnostic = new vscode.Diagnostic(
    range,
    message,
    vscode.DiagnosticSeverity.Warning, // Warning (yellow) - DDD violations are best practice suggestions, not errors
  );

  diagnostic.source = "DDD Enforcer";
  diagnostic.code = violation.type;

  return diagnostic;
}

// =============================================================================
// STATUS & UI
// =============================================================================

/**
 * Updates status bar based on backend status.
 */
async function updateStatusFromBackend() {
  try {
    const response = await axios.get<HealthResponse>(
      `http://127.0.0.1:${backendPort}/health`,
      { timeout: 2000 },
    );

    if (response.data.domain_model_loaded) {
      updateStatusBar("ready");
      log("Domain model is loaded");
    } else {
      updateStatusBar("notInitialized");
      log("Domain model is NOT loaded - need to initialize");
    }
  } catch {
    updateStatusBar("error");
  }
}

/**
 * Shows detailed status information.
 */
async function showStatus() {
  if (!isBackendReady) {
    vscode.window.showInformationMessage(
      "DDD Enforcer: Backend is not running. Save a Python file to start it.",
    );
    return;
  }

  try {
    const response = await axios.get(`http://127.0.0.1:${backendPort}/status`);
    const status = response.data;

    const modelStatus = status.domain_model.loaded
      ? "✅ Loaded"
      : "❌ Not loaded - Run 'Initialize Domain Model' command";

    const ragStatus = status.rag.initialized
      ? "✅ Initialized"
      : "❌ Not initialized";

    const message =
      `DDD Enforcer Status\n\n` +
      `Backend: Running on port ${backendPort}\n` +
      `Domain Model: ${modelStatus}\n` +
      `RAG: ${ragStatus}`;

    vscode.window.showInformationMessage(message);
  } catch {
    vscode.window.showErrorMessage("DDD Enforcer: Could not get status.");
  }
}

/**
 * Updates the status bar with current state.
 */
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
  const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
  if (!cfg.get<boolean>("showStatusBar", true)) {
    statusBarItem.hide();
    return;
  }

  switch (state) {
    case "inactive":
      statusBarItem.text = "$(circle-outline) DDD Enforcer";
      statusBarItem.tooltip = "Click to see status. Backend not running.";
      statusBarItem.backgroundColor = undefined;
      break;
    case "starting":
      statusBarItem.text = "$(loading~spin) DDD Enforcer";
      statusBarItem.tooltip = "Starting backend server...";
      statusBarItem.backgroundColor = undefined;
      break;
    case "ready":
      statusBarItem.text = "$(check) DDD Enforcer";
      statusBarItem.tooltip = "DDD Enforcer is ready. Click for status.";
      statusBarItem.backgroundColor = undefined;
      break;
    case "notInitialized":
      statusBarItem.text = "$(warning) DDD Enforcer";
      statusBarItem.tooltip =
        "Domain model not loaded. Run 'Initialize Domain Model' command.";
      statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground",
      );
      break;
    case "validating":
      statusBarItem.text = "$(loading~spin) DDD Enforcer";
      statusBarItem.tooltip = "Validating code...";
      statusBarItem.backgroundColor = undefined;
      break;
    case "validatingApiKey":
      statusBarItem.text = "$(loading~spin) DDD Enforcer";
      statusBarItem.tooltip = "Validating Gemini API key...";
      statusBarItem.backgroundColor = undefined;
      break;
    case "violations":
      statusBarItem.text = `$(error) DDD Enforcer (${count})`;
      statusBarItem.tooltip = `${count} DDD violation(s) found. Click for status.`;
      statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.errorBackground",
      );
      break;
    case "error":
      statusBarItem.text = "$(warning) DDD Enforcer";
      statusBarItem.tooltip = "DDD Enforcer has errors. Click for status.";
      statusBarItem.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground",
      );
      break;
  }

  statusBarItem.show();
}

// =============================================================================
// CODE ACTION PROVIDER
// =============================================================================

/**
 * Provides "View Source" quick fix actions for DDD violations.
 */
class DDDSourceCodeActionProvider implements vscode.CodeActionProvider {
  provideCodeActions(
    document: vscode.TextDocument,
    _range: vscode.Range,
    context: vscode.CodeActionContext,
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];

    for (const diagnostic of context.diagnostics) {
      if (diagnostic.source !== "DDD Enforcer") {
        continue;
      }

      const key = `${document.uri.toString()}-${diagnostic.range.start.line}`;
      const sources = violationSources.get(key);

      if (sources && sources.length > 0) {
        sources.forEach((source, index) => {
          const action = new vscode.CodeAction(
            `📚 View Source [${index + 1}]: ${source.section}`,
            vscode.CodeActionKind.QuickFix,
          );
          action.command = {
            command: "ddd-enforcer.openSource",
            title: "Open Source Document",
            arguments: [source.file_path, source.section],
          };
          action.diagnostics = [diagnostic];
          action.isPreferred = index === 0;
          actions.push(action);
        });
      }
    }

    return actions;
  }
}

/**
 * Shows a Markdown peek for a DDD violation when the user hovers over a
 * line that produced a diagnostic. Reads the cached violation for the
 * hovered line from validationViolationCache (no re-fetch) and renders
 * formatHoverMarkdown in a trusted MarkdownString so the embedded
 * "Open SRS source" command link works.
 */
class DDDViolationHoverProvider implements vscode.HoverProvider {
  provideHover(
    document: vscode.TextDocument,
    position: vscode.Position,
  ): vscode.Hover | undefined {
    const lineToViolation = validationViolationCache.get(
      document.uri.toString(),
    );
    if (!lineToViolation) {
      return undefined;
    }
    const violation = lineToViolation.get(position.line);
    if (!violation) {
      return undefined;
    }
    const keyword = extractKeyword(violation.message);
    const markdown = new vscode.MarkdownString(
      formatHoverMarkdown(violation, keyword),
    );
    markdown.isTrusted = true;
    return new vscode.Hover(markdown);
  }
}

/**
 * Returns true when `filePath` resolves to a location inside one of the
 * currently-open workspace folders. Used to gate opening backend-supplied
 * source paths: in-workspace paths open silently, out-of-workspace paths
 * require explicit user confirmation (defense-in-depth against a
 * compromised backend or a hallucinated path).
 */
function isPathInsideWorkspace(filePath: string): boolean {
  const folders = vscode.workspace.workspaceFolders;
  if (!folders || folders.length === 0) {
    return false;
  }
  const resolved = path.resolve(filePath);
  return folders.some((folder) => {
    const folderPath = path.resolve(folder.uri.fsPath);
    const rel = path.relative(folderPath, resolved);
    // Inside the folder when the relative path doesn't climb out (no "..")
    // and isn't an absolute path on another root.
    return (
      rel === "" ||
      (!rel.startsWith("..") && !path.isAbsolute(rel))
    );
  });
}

/**
 * Opens a source document and navigates to the relevant section.
 * In-workspace paths (the common case — SRS files under the workspace
 * inputs/) open silently. Out-of-workspace paths require an explicit
 * modal confirmation before opening (defense-in-depth: the file_path
 * originates from a backend-supplied RAG response).
 */
async function openSourceCommand(filePath: string, section: string) {
  try {
    if (!isPathInsideWorkspace(filePath)) {
      const choice = await vscode.window.showWarningMessage(
        `DDD Enforcer wants to open a file outside your workspace:\n${filePath}\n\nOpen it?`,
        { modal: true },
        "Open",
      );
      if (choice !== "Open") {
        log(`Declined to open out-of-workspace source path: ${filePath}`);
        return;
      }
    }

    const uri = vscode.Uri.file(filePath);
    const doc = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(doc, {
      preview: true,
      viewColumn: vscode.ViewColumn.Beside,
    });

    const text = doc.getText();
    const sectionIndex = text.indexOf(section);
    if (sectionIndex !== -1) {
      const position = doc.positionAt(sectionIndex);
      editor.selection = new vscode.Selection(position, position);
      editor.revealRange(
        new vscode.Range(position, position),
        vscode.TextEditorRevealType.InCenter,
      );
    }
  } catch {
    vscode.window.showErrorMessage(`Could not open source file: ${filePath}`);
  }
}

// =============================================================================
// HELPERS
// =============================================================================

/**
 * Gets the path to the bundled backend.
 */
function getBackendPath(context: vscode.ExtensionContext): string {
  // In production: bundled with extension
  const bundledPath = path.join(context.extensionPath, "backend");
  if (fs.existsSync(bundledPath)) {
    return bundledPath;
  }

  // In development: use extension/backend from workspace
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (workspaceFolders) {
    for (const folder of workspaceFolders) {
      // First try extension/backend (new structure)
      const extensionBackendPath = path.join(
        folder.uri.fsPath,
        "extension",
        "backend",
      );
      if (fs.existsSync(extensionBackendPath)) {
        return extensionBackendPath;
      }
    }
  }

  // Fallback: relative to extension
  return path.join(context.extensionPath, "..", "backend");
}

/**
 * Finds an available port starting from the configured port.
 */
async function findAvailablePort(): Promise<number> {
  const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
  const preferredPort = cfg.get<number>("backendPort", 8000);

  // Try the preferred port first
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

/**
 * Checks if a port is available.
 */
function isPortAvailable(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const server = net.createServer();

    server.once("error", () => resolve(false));
    server.once("listening", () => {
      server.close();
      resolve(true);
    });

    server.listen(port, "127.0.0.1");
  });
}

/**
 * Clears stored sources for a document.
 */
function clearSourcesForDocument(uriString: string) {
  for (const key of violationSources.keys()) {
    if (key.startsWith(uriString)) {
      violationSources.delete(key);
    }
  }
  validationViolationCache.delete(uriString);
}

/**
 * Extracts the problematic keyword from a violation message.
 */
function extractKeyword(message: string): string {
  const match = message.match(/'([^']+)'/);
  return match ? match[1] : "";
}

/**
 * Finds the range of a keyword in the document.
 */
function findKeywordRange(
  document: vscode.TextDocument,
  keyword: string,
): vscode.Range {
  if (!keyword) {
    return new vscode.Range(0, 0, 0, 0);
  }

  const text = document.getText();

  // Look in class/function definitions first
  const definitionPatterns = [
    new RegExp(`class\\s+${escapeRegex(keyword)}\\b`),
    new RegExp(`def\\s+${escapeRegex(keyword)}\\b`),
    new RegExp(`^\\s*${escapeRegex(keyword)}\\s*=`, "m"),
  ];

  for (const pattern of definitionPatterns) {
    const match = pattern.exec(text);
    if (match) {
      const keywordIndexInMatch = match[0].indexOf(keyword);
      const absoluteIndex = match.index + keywordIndexInMatch;
      const positionStart = document.positionAt(absoluteIndex);
      const positionEnd = document.positionAt(absoluteIndex + keyword.length);
      return new vscode.Range(positionStart, positionEnd);
    }
  }

  // Fall back to searching in non-comment lines
  const lines = text.split("\n");
  for (let lineNum = 0; lineNum < lines.length; lineNum++) {
    const line = lines[lineNum];
    const trimmedLine = line.trim();

    if (
      trimmedLine.startsWith("#") ||
      trimmedLine.startsWith('"""') ||
      trimmedLine.startsWith("'''")
    ) {
      continue;
    }

    const keywordIndex = line.indexOf(keyword);
    if (keywordIndex !== -1) {
      const commentIndex = line.indexOf("#");
      if (commentIndex === -1 || keywordIndex < commentIndex) {
        return new vscode.Range(
          new vscode.Position(lineNum, keywordIndex),
          new vscode.Position(lineNum, keywordIndex + keyword.length),
        );
      }
    }
  }

  return new vscode.Range(0, 0, 0, 0);
}

/**
 * Escapes special regex characters.
 */
function escapeRegex(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Checks if a file should be skipped from validation.
 * Excludes extension's bundled backend folder and common non-project paths.
 */
function shouldSkipValidation(
  filePath: string,
  context: vscode.ExtensionContext,
): boolean {
  // Skip files inside the extension's bundled backend
  const extensionBackendPath = path.join(context.extensionPath, "backend");
  if (filePath.startsWith(extensionBackendPath)) {
    return true;
  }

  // Skip common paths that shouldn't be validated
  const skipPatterns = [
    /[/\\]node_modules[/\\]/,
    /[/\\]\.venv[/\\]/,
    /[/\\]venv[/\\]/,
    /[/\\]__pycache__[/\\]/,
    /[/\\]\.git[/\\]/,
    /[/\\]site-packages[/\\]/,
  ];

  for (const pattern of skipPatterns) {
    if (pattern.test(filePath)) {
      return true;
    }
  }

  return false;
}

/**
 * Logs a message to the output channel.
 */
function log(message: string) {
  const timestamp = new Date().toLocaleTimeString();
  outputChannel.appendLine(`[${timestamp}] ${message}`);
}

/**
 * Sleep helper.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
