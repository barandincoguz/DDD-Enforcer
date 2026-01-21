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

/** Response from the backend validation endpoint */
interface ValidationResponse {
  is_violation: boolean;
  violations: Violation[];
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

// Store sources for code actions (keyed by document URI + line number)
const violationSources = new Map<string, ViolationSource[]>();

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

    // Handle process exit
    backendProcess.on("exit", (code) => {
      log(`Backend process exited with code ${code}`);
      isBackendReady = false;
      backendStarting = false;
      backendProcess = null;
      updateStatusBar("inactive");
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
  stopBackend();
  await sleep(1000);
  const success = await startBackend(context);
  if (success) {
    vscode.window.showInformationMessage(
      "DDD Enforcer: Backend restarted successfully!",
    );
  }
}

// =============================================================================
// API KEY MANAGEMENT
// =============================================================================

/**
 * Gets the Gemini API key from settings, env var, or prompts the user.
 */
async function getApiKey(
  context: vscode.ExtensionContext,
): Promise<string | undefined> {
  // 1. Check settings
  const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
  let apiKey: string | undefined = cfg.get<string>("geminiApiKey", "");

  if (apiKey && apiKey.trim()) {
    return apiKey.trim();
  }

  // 2. Check environment variable
  apiKey = process.env.GEMINI_API_KEY || "";
  if (apiKey && apiKey.trim()) {
    return apiKey.trim();
  }

  // 3. Check secret storage
  const storedKey = await context.secrets.get("geminiApiKey");
  if (storedKey && storedKey.trim()) {
    return storedKey.trim();
  }

  // 4. Prompt user
  const inputKey = await vscode.window.showInputBox({
    prompt: "Enter your Gemini API Key",
    placeHolder: "AIza...",
    password: true,
    ignoreFocusOut: true,
  });

  if (inputKey && inputKey.trim()) {
    // Save to secret storage
    await context.secrets.store("geminiApiKey", inputKey.trim());
    return inputKey.trim();
  }

  return undefined;
}

// =============================================================================
// DOMAIN MODEL INITIALIZATION
// =============================================================================

/**
 * Command: Initialize Domain Model
 * Opens file picker for SRS documents and generates model.json
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

  // Show progress
  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "DDD Enforcer: Generating Domain Model",
      cancellable: false,
    },
    async (progress) => {
      progress.report({ message: "Analyzing documents..." });

      try {
        const filePaths = files.map((f) => f.fsPath);
        const outputPath = path.join(
          workspaceFolder.uri.fsPath,
          "domain",
          "model.json",
        );

        log(`Generating model from: ${filePaths.join(", ")}`);
        log(`Output path: ${outputPath}`);

        const response = await axios.post<GenerateModelResponse>(
          `http://127.0.0.1:${backendPort}/generate-model`,
          {
            file_paths: filePaths,
            output_path: outputPath,
          },
          { timeout: 300000 }, // 5 minutes timeout for large documents
        );

        if (response.data.success) {
          progress.report({ message: "Domain model created!" });

          // Update status bar immediately
          updateStatusBar("ready");

          // Store for later use (outside withProgress)
          const modelPath = response.data.model_path;
          const projectName = response.data.project_name;
          const boundedContextsCount = response.data.bounded_contexts_count;

          // Show success message after progress completes (non-blocking)
          setTimeout(async () => {
            const openAction = "Open Model";
            const result = await vscode.window.showInformationMessage(
              `DDD Enforcer: Domain Model created successfully!\n` +
                `Project: ${projectName}\n` +
                `Bounded Contexts: ${boundedContextsCount}`,
              openAction,
            );

            if (result === openAction && modelPath) {
              const doc = await vscode.workspace.openTextDocument(modelPath);
              await vscode.window.showTextDocument(doc);
            }
          }, 100);
        } else {
          vscode.window.showErrorMessage(
            `DDD Enforcer: Failed to generate model - ${response.data.error}`,
          );
          updateStatusBar("error");
        }
      } catch (error: unknown) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        log(`Error generating model: ${errorMessage}`);
        vscode.window.showErrorMessage(
          `DDD Enforcer: Error generating model - ${errorMessage}`,
        );
        updateStatusBar("error");
      }
    },
  );
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

    const data = response.data;

    if (data.is_violation && data.violations) {
      const diagnostics: vscode.Diagnostic[] = [];

      data.violations.forEach((violation) => {
        const diagnostic = createDiagnostic(document, violation);
        diagnostics.push(diagnostic);
      });

      collection.set(document.uri, diagnostics);
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
    vscode.DiagnosticSeverity.Error,
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
 * Opens a source document and navigates to the relevant section.
 */
async function openSourceCommand(filePath: string, section: string) {
  try {
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

  // Find another available port
  for (let port = preferredPort + 1; port < preferredPort + 100; port++) {
    if (await isPortAvailable(port)) {
      return port;
    }
  }

  // Fallback
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
