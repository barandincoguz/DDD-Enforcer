/**
 * DDD Enforcer VS Code Extension
 *
 * Validates Python code against Domain-Driven Design rules on save.
 * Shows violations as diagnostics with clickable source references.
 */

import * as vscode from "vscode";
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
  console.log("DDD Enforcer is now active!");

  // Create diagnostic collection for showing violations
  const diagnosticCollection =
    vscode.languages.createDiagnosticCollection("ddd-enforcer");
  context.subscriptions.push(diagnosticCollection);

  // Register command to open source documents
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "ddd-enforcer.openSource",
      openSourceCommand
    )
  );

  // Register code action provider for "View Source" quick fixes
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      "python",
      new DDDSourceCodeActionProvider(),
      { providedCodeActionKinds: [vscode.CodeActionKind.QuickFix] }
    )
  );

  // Validate on file save
  context.subscriptions.push(
    vscode.workspace.onDidSaveTextDocument(async (document) => {
      if (document.languageId === "python") {
        await validateCode(document, diagnosticCollection);
      }
    })
  );
}

export function deactivate() {}

// =============================================================================
// COMMANDS
// =============================================================================

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

    // Try to find and highlight the section
    const text = doc.getText();
    const sectionIndex = text.indexOf(section);
    if (sectionIndex !== -1) {
      const position = doc.positionAt(sectionIndex);
      editor.selection = new vscode.Selection(position, position);
      editor.revealRange(
        new vscode.Range(position, position),
        vscode.TextEditorRevealType.InCenter
      );
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Could not open source file: ${filePath}`);
  }
}

// =============================================================================
// CODE ACTION PROVIDER
// =============================================================================

/**
 * Provides "View Source" quick fix actions for DDD violations.
 * Shows all available source references as numbered options.
 */
class DDDSourceCodeActionProvider implements vscode.CodeActionProvider {
  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range,
    context: vscode.CodeActionContext
  ): vscode.CodeAction[] {
    const actions: vscode.CodeAction[] = [];

    for (const diagnostic of context.diagnostics) {
      if (diagnostic.source !== "DDD Enforcer") {
        continue;
      }

      // Get stored sources for this diagnostic
      const key = `${document.uri.toString()}-${diagnostic.range.start.line}`;
      const sources = violationSources.get(key);

      if (sources && sources.length > 0) {
        sources.forEach((source, index) => {
          const action = new vscode.CodeAction(
            `[${index + 1}] View Source: ${source.section}`,
            vscode.CodeActionKind.QuickFix
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

// =============================================================================
// VALIDATION
// =============================================================================

/**
 * Validates Python code against DDD rules via the backend API.
 * Creates diagnostics for any violations found.
 */
async function validateCode(
  document: vscode.TextDocument,
  collection: vscode.DiagnosticCollection
) {
  // Clear previous diagnostics and sources
  collection.clear();
  clearSourcesForDocument(document.uri.toString());

  const codeContent = document.getText();
  const fileName = document.fileName;

  try {
    const response = await axios.post<ValidationResponse>(
      "http://127.0.0.1:8000/validate",
      { filename: fileName, content: codeContent }
    );

    const data = response.data;

    if (data.is_violation && data.violations) {
      const diagnostics = data.violations.map((violation) =>
        createDiagnostic(document, violation)
      );
      collection.set(document.uri, diagnostics);
    }
  } catch (error) {
    console.error("Error validating code:", error);
    vscode.window.showErrorMessage(
      "DDD Enforcer: Could not connect to backend server."
    );
  }
}

/**
 * Creates a diagnostic for a single violation.
 */
function createDiagnostic(
  document: vscode.TextDocument,
  violation: Violation
): vscode.Diagnostic {
  // Find the location of the violation in the code
  const keyword = extractKeyword(violation.message);
  const range = findKeywordRange(document, keyword);

  // Build diagnostic message
  let message = `${violation.message}\nSuggestion: ${violation.suggestion}`;

  if (violation.sources && violation.sources.length > 0) {
    const sourceRefs = violation.sources
      .map((src, i) => `[${i + 1}] ${src.section}`)
      .join("  ");
    message += `\n\n${sourceRefs}`;

    // Store sources for code action provider
    const key = `${document.uri.toString()}-${range.start.line}`;
    violationSources.set(key, violation.sources);
  }

  const diagnostic = new vscode.Diagnostic(
    range,
    message,
    vscode.DiagnosticSeverity.Error
  );

  diagnostic.source = "DDD Enforcer";

  return diagnostic;
}

// =============================================================================
// HELPERS
// =============================================================================

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
 * Looks for quoted terms like 'ClientManager'.
 */
function extractKeyword(message: string): string {
  const match = message.match(/'([^']+)'/);
  return match ? match[1] : "";
}

/**
 * Finds the range of a keyword in the document.
 * Prioritizes class/function definitions over other occurrences.
 */
function findKeywordRange(
  document: vscode.TextDocument,
  keyword: string
): vscode.Range {
  if (!keyword) {
    return new vscode.Range(0, 0, 0, 0);
  }

  const text = document.getText();

  // First, look in class/function definitions
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

    // Skip comments and docstrings
    if (
      trimmedLine.startsWith("#") ||
      trimmedLine.startsWith('"""') ||
      trimmedLine.startsWith("'''")
    ) {
      continue;
    }

    const keywordIndex = line.indexOf(keyword);
    if (keywordIndex !== -1) {
      // Check if keyword is before any inline comment
      const commentIndex = line.indexOf("#");
      if (commentIndex === -1 || keywordIndex < commentIndex) {
        const positionStart = new vscode.Position(lineNum, keywordIndex);
        const positionEnd = new vscode.Position(
          lineNum,
          keywordIndex + keyword.length
        );
        return new vscode.Range(positionStart, positionEnd);
      }
    }
  }

  // Default to first line if not found
  return new vscode.Range(0, 0, 0, 0);
}

/**
 * Escapes special regex characters in a string.
 */
function escapeRegex(string: string): string {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
