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
// MODULAR EXPORTS AND IMPORTS (Refactored from Monolithic structure)
// =============================================================================

export {
  ViolationSource,
  Violation,
  ValidationMetrics,
  ValidationResponse,
  HealthResponse,
  PipelineProgress,
  SSEEvent,
  GenerateModelResponse,
  CombinedMetrics,
} from "./types";

export {
  ExitDisposition,
  computeBackoffMs,
  shouldAttemptRestart,
  formatExitReason,
  classifyExitForRestart,
  isPortAvailable,
  findAvailablePort,
} from "./backend/processManager";

export {
  LAST_RUN_DURATIONS_KEY,
  STAGE_ORDER,
  STAGE_WEIGHTS,
  SubProgress,
  computeOverallPercent,
  parseSubProgress,
  formatEta,
  computeEtaMs,
  StageStatusBarParts,
  formatStageStatusBar,
} from "./utils/progress";

export {
  LruCache,
  truncateExcerpt,
  boldMatchingSpan,
  escapeInlineMarkdown,
  formatHoverMarkdown,
} from "./ui/hoverProvider";

export {
  ManifestViolation,
  PaperRunManifest,
  RunSummary,
  RunSummaryColumn,
  summarizeManifest,
  sortRunSummaries,
  generateNonce,
  buildRunManifestsHtml,
} from "./ui/runManifestsWebview";

export {
  ApiKeyErrorKind,
  ApiKeyValidationResult,
  ApiKeyHttpProbe,
  ApiKeySource,
  MigrationDecision,
  validateGeminiKey,
  decideMigrationOffer,
  getApiKey,
  classifyApiKeyError,
} from "./apiKeyManager";

export {
  normalizePythonSemantics,
  getValidationFingerprint,
  classifySaveForValidation,
  classifySaveForValidationFromContent,
  tokenizeIndentation,
  extractLogicalLines,
  type LogicalLine,
} from "./semanticFingerprint";

import {
  ViolationSource,
  Violation,
  ValidationMetrics,
  ValidationResponse,
  HealthResponse,
  PipelineProgress,
  SSEEvent,
  GenerateModelResponse,
  CombinedMetrics,
} from "./types";

import {
  ExitDisposition,
  computeBackoffMs,
  shouldAttemptRestart,
  formatExitReason,
  classifyExitForRestart,
  isPortAvailable,
  findAvailablePort,
} from "./backend/processManager";

import {
  LAST_RUN_DURATIONS_KEY,
  STAGE_ORDER,
  STAGE_WEIGHTS,
  SubProgress,
  computeOverallPercent,
  parseSubProgress,
  formatEta,
  computeEtaMs,
  StageStatusBarParts,
  formatStageStatusBar,
} from "./utils/progress";

import {
  LruCache,
  truncateExcerpt,
  boldMatchingSpan,
  formatHoverMarkdown,
} from "./ui/hoverProvider";

import {
  ManifestViolation,
  PaperRunManifest,
  RunSummary,
  RunSummaryColumn,
  summarizeManifest,
  sortRunSummaries,
  generateNonce,
  buildRunManifestsHtml,
} from "./ui/runManifestsWebview";

import {
  getApiKey,
  validateGeminiKey,
  decideMigrationOffer,
} from "./apiKeyManager";

import {
  normalizePythonSemantics,
  getValidationFingerprint,
  classifySaveForValidation,
  classifySaveForValidationFromContent,
} from "./semanticFingerprint";

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
export const validationViolationCache = new LruCache<
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

  context.subscriptions.push(
    vscode.commands.registerCommand("ddd-enforcer.showRunManifests", () =>
      openRunManifestsWebview(context),
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
    const apiKey = await getApiKey(context, { log, updateStatusBar });
    if (!apiKey) {
      vscode.window.showErrorMessage(
        "DDD Enforcer: Gemini API Key is required. Please configure it in settings or provide when prompted.",
      );
      backendStarting = false;
      updateStatusBar("error");
      return false;
    }

    // Find available port
    backendPort = await findAvailablePort(log);
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


// API key discovery, validation, and migration are handled by src/apiKeyManager.ts



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


// Semantic fingerprinting helpers are imported from src/semanticFingerprint.ts


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
export function updateStatusBar(
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
  if (!statusBarItem) {
    return;
  }
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
export class DDDViolationHoverProvider implements vscode.HoverProvider {
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
    markdown.isTrusted = { enabledCommands: ["ddd-enforcer.openSource"] };
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
// RUN MANIFEST WEBVIEW (integration — command + discovery + watcher)
// =============================================================================

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


// Port scanning helpers are imported from src/backend/processManager.ts


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
export function log(message: string) {
  if (!outputChannel) {
    console.log(message);
    return;
  }
  const timestamp = new Date().toLocaleTimeString();
  outputChannel.appendLine(`[${timestamp}] ${message}`);
}

/**
 * Sleep helper.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
