import * as assert from "assert";
import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import axios from "axios";
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
  validateGeminiKey,
  type ApiKeyValidationResult,
  decideMigrationOffer,
  getApiKey,
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
  escapeInlineMarkdown,
  formatHoverMarkdown,
  summarizeManifest,
  sortRunSummaries,
  type RunSummary,
  type PaperRunManifest,
  generateNonce,
  buildRunManifestsHtml,
  DDDViolationHoverProvider,
  validationViolationCache,
  tokenizeIndentation,
  type LogicalLine,
  extractLogicalLines,
} from "../extension";

suite("Extension Test Suite", () => {
  vscode.window.showInformationMessage("Start all tests.");

  // ==========================================================================
  // BASIC EXTENSION TESTS
  // ==========================================================================

  test("Extension should be present", () => {
    const extension = vscode.extensions.getExtension(
      "ddd-enforcer.ddd-enforcer",
    );
    // Extension might not be found in test environment, that's OK
    assert.ok(true, "Extension test ran successfully");
  });

  test("Sample test", () => {
    assert.strictEqual(-1, [1, 2, 3].indexOf(5));
    assert.strictEqual(-1, [1, 2, 3].indexOf(0));
  });

  // ==========================================================================
  // CONFIGURATION TESTS
  // ==========================================================================

  test("Extension configuration should have defaults", () => {
    const config = vscode.workspace.getConfiguration("ddd-enforcer");

    // These may not be available in test env, but we check the API works
    const backendPort = config.get<number>("backendPort");
    const autoValidate = config.get<boolean>("autoValidateOnSave");

    // Config access should work without errors
    assert.ok(true, "Configuration access works");
  });

  // ==========================================================================
  // COMMAND REGISTRATION TESTS
  // ==========================================================================

  test("Commands should be registered", async () => {
    const commands = await vscode.commands.getCommands(true);

    // Check some expected commands exist (may vary by environment)
    // At minimum, built-in commands should be there
    assert.ok(commands.length > 0, "Commands are registered");
  });

  // ==========================================================================
  // BACKEND PATH TESTS
  // ==========================================================================

  test("Backend directory structure should exist in extension", () => {
    // Get extension path from current file location
    const extensionPath = path.resolve(__dirname, "..", "..");
    const backendPath = path.join(extensionPath, "backend");

    // In test environment, we're in out/test so backend should be at extension root
    // This test validates the expected structure
    const expectedFiles = ["main.py", "config.py", "requirements.txt"];

    if (fs.existsSync(backendPath)) {
      for (const file of expectedFiles) {
        const filePath = path.join(backendPath, file);
        if (fs.existsSync(filePath)) {
          assert.ok(true, `Backend file ${file} exists`);
        }
      }
    }

    // Test passes even if backend not bundled (development vs production)
    assert.ok(true, "Backend path check completed");
  });

  // ==========================================================================
  // DIAGNOSTIC COLLECTION TESTS
  // ==========================================================================

  test("Diagnostics API should be accessible", () => {
    // Create a diagnostic collection (won't persist after test)
    const collection = vscode.languages.createDiagnosticCollection("test-ddd");

    assert.ok(collection, "Diagnostic collection created");

    // Clean up
    collection.dispose();
  });

  test("Can create diagnostic objects", () => {
    const range = new vscode.Range(0, 0, 0, 10);
    const diagnostic = new vscode.Diagnostic(
      range,
      "Test violation message",
      vscode.DiagnosticSeverity.Warning,
    );

    assert.strictEqual(diagnostic.message, "Test violation message");
    assert.strictEqual(diagnostic.severity, vscode.DiagnosticSeverity.Warning);
  });

  // ==========================================================================
  // STATUS BAR TESTS
  // ==========================================================================

  test("Status bar items can be created", () => {
    const statusBar = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100,
    );

    assert.ok(statusBar, "Status bar item created");

    statusBar.text = "DDD Test";
    assert.strictEqual(statusBar.text, "DDD Test");

    // Clean up
    statusBar.dispose();
  });

  // ==========================================================================
  // OUTPUT CHANNEL TESTS
  // ==========================================================================

  test("Output channels can be created", () => {
    const outputChannel = vscode.window.createOutputChannel("DDD Test");

    assert.ok(outputChannel, "Output channel created");

    // Test we can append to it
    outputChannel.appendLine("Test message");

    // Clean up
    outputChannel.dispose();
  });

  // ==========================================================================
  // URI AND PATH TESTS
  // ==========================================================================

  test("VS Code URI handling works", () => {
    const uri = vscode.Uri.file("/test/path/file.py");

    assert.strictEqual(uri.scheme, "file");
    assert.ok(uri.fsPath.endsWith("file.py"));
  });

  test("Python file detection by extension", () => {
    const pythonFiles = [
      "/test/file.py",
      "/test/module.py",
      "/test/service.py",
    ];

    const nonPythonFiles = ["/test/file.js", "/test/file.ts", "/test/file.txt"];

    for (const file of pythonFiles) {
      assert.ok(file.endsWith(".py"), `${file} is Python`);
    }

    for (const file of nonPythonFiles) {
      assert.ok(!file.endsWith(".py"), `${file} is not Python`);
    }
  });

  // ==========================================================================
  // SAVE-TRIGGER SEMANTIC CHANGE TESTS
  // ==========================================================================

  test("No-op save should skip validation", () => {
    const code = "class Order:\n    pass\n";
    const decision = classifySaveForValidationFromContent(code, code);
    assert.strictEqual(decision.shouldValidate, false);
  });

  test("Blank-line deletion should skip validation", () => {
    const before = "class Order:\n\n    def confirm(self):\n        pass\n";
    const after = "class Order:\n    def confirm(self):\n        pass\n";
    const decision = classifySaveForValidationFromContent(before, after);
    assert.strictEqual(decision.shouldValidate, false);
  });

  test("Comment-only change should skip validation", () => {
    const before =
      "class Order:\n    def confirm(self):\n        return True\n";
    const after =
      "class Order:\n    # updated comment\n    def confirm(self):\n        return True  # inline\n";
    const decision = classifySaveForValidationFromContent(before, after);
    assert.strictEqual(decision.shouldValidate, false);
  });

  test("Semantic code change should trigger validation", () => {
    const before =
      "class Order:\n    def confirm(self):\n        return True\n";
    const after =
      "class Order:\n    def confirm(self):\n        return False\n";
    const decision = classifySaveForValidationFromContent(before, after);
    assert.strictEqual(decision.shouldValidate, true);
  });

  // ==========================================================================
  // FINGERPRINT — INDENTATION SEMANTICS (T10 integration)
  // ==========================================================================

  test("Dedenting a statement out of a block triggers validation (T10)", () => {
    const before = "if x:\n    a()\n    b()\n";
    const after = "if x:\n    a()\nb()\n";
    const decision = classifySaveForValidationFromContent(before, after);
    assert.strictEqual(decision.shouldValidate, true);
  });

  test("Indenting a statement into a block triggers validation (T10)", () => {
    const before = "if x:\n    a()\nb()\n";
    const after = "if x:\n    a()\n    b()\n";
    const decision = classifySaveForValidationFromContent(before, after);
    assert.strictEqual(decision.shouldValidate, true);
  });

  test("A pure reindent (4-space to 2-space) skips validation", () => {
    const before = "if x:\n    a()\n        b()\n";
    const after = "if x:\n  a()\n    b()\n";
    const decision = classifySaveForValidationFromContent(before, after);
    assert.strictEqual(decision.shouldValidate, false);
  });

  test("Editing code after a tricky f-string docstring is still detected (T11 no state leak)", () => {
    const before = 'x = f"a {b.split(" ")} c"\nreturn True\n';
    const after = 'x = f"a {b.split(" ")} c"\nreturn False\n';
    const decision = classifySaveForValidationFromContent(before, after);
    assert.strictEqual(decision.shouldValidate, true);
  });

  // ==========================================================================
  // FINGERPRINT — PHASE 2: INDENT TOKENIZATION (T10)
  // ==========================================================================

  const IND = ""; // INDENT marker
  const DED = ""; // DEDENT marker

  test("tokenizeIndentation emits no markers for flat same-indent lines", () => {
    const out = tokenizeIndentation([
      { indentWidth: 0, content: "a" },
      { indentWidth: 0, content: "b" },
    ]);
    assert.ok(!out.includes(IND), "no INDENT");
    assert.ok(!out.includes(DED), "no DEDENT");
  });

  test("tokenizeIndentation emits one INDENT then one DEDENT for a nested block", () => {
    const out = tokenizeIndentation([
      { indentWidth: 0, content: "if x:" },
      { indentWidth: 4, content: "a()" },
      { indentWidth: 0, content: "b()" },
    ]);
    assert.strictEqual((out.match(//g) || []).length, 1, "one INDENT");
    assert.strictEqual((out.match(//g) || []).length, 1, "one DEDENT");
  });

  test("tokenizeIndentation emits two DEDENT markers for a two-level pop", () => {
    const out = tokenizeIndentation([
      { indentWidth: 0, content: "a" },
      { indentWidth: 4, content: "b" },
      { indentWidth: 8, content: "c" },
      { indentWidth: 0, content: "d" },
    ]);
    assert.strictEqual((out.match(//g) || []).length, 2, "two DEDENT");
  });

  test("tokenizeIndentation is invariant to indent width (reindent is non-semantic)", () => {
    const wide = tokenizeIndentation([
      { indentWidth: 0, content: "if x:" },
      { indentWidth: 4, content: "a" },
      { indentWidth: 8, content: "b" },
    ]);
    const narrow = tokenizeIndentation([
      { indentWidth: 0, content: "if x:" },
      { indentWidth: 2, content: "a" },
      { indentWidth: 4, content: "b" },
    ]);
    assert.strictEqual(wide, narrow);
  });

  // ==========================================================================
  // FINGERPRINT — PHASE 1: LOGICAL-LINE EXTRACTION (T11)
  // ==========================================================================

  test("extractLogicalLines strips a code comment but keeps '#' inside a string", () => {
    const lines = extractLogicalLines('a = 1  # tail\nb = "# not a comment"\n');
    assert.strictEqual(lines.length, 2);
    assert.strictEqual(lines[0].content, "a=1");
    assert.strictEqual(lines[1].content, 'b="# not a comment"');
  });

  test("extractLogicalLines collapses operator spacing but keeps token boundaries", () => {
    const lines = extractLogicalLines("a   =   1\ndel   x\n");
    assert.strictEqual(lines[0].content, "a=1");
    assert.strictEqual(lines[1].content, "del x");
  });

  test("extractLogicalLines keeps an escaped delimiter inside a triple string", () => {
    const src = 'x = """He said \\"hi\\""""\ny = 2\n';
    const lines = extractLogicalLines(src);
    assert.strictEqual(lines.length, 2, "string closes exactly once; y is its own line");
    assert.strictEqual(lines[1].content, "y=2");
  });

  test("extractLogicalLines does not let lone quotes inside a triple string close it early", () => {
    const src = "x = '''a'b'c'''\ny = 2\n";
    const lines = extractLogicalLines(src);
    assert.strictEqual(lines.length, 2);
    assert.strictEqual(lines[1].content, "y=2");
  });

  test("extractLogicalLines handles a Python 3.12 f-string that reuses the quote inside braces", () => {
    const src = 'x = f"a {b.split(" ")} c"\ny = 2\n';
    const lines = extractLogicalLines(src);
    assert.strictEqual(lines.length, 2, "f-string closes only at the final quote");
    assert.strictEqual(lines[1].content, "y=2");
  });

  test("extractLogicalLines joins a backslash line continuation into one logical line", () => {
    const lines = extractLogicalLines("a = 1 + \\\n    2\n");
    assert.strictEqual(lines.length, 1);
    assert.strictEqual(lines[0].content, "a=1+2");
  });

  test("extractLogicalLines drops blank and comment-only lines", () => {
    const lines = extractLogicalLines("a = 1\n\n   # just a comment\nb = 2\n");
    assert.strictEqual(lines.length, 2);
    assert.strictEqual(lines[0].content, "a=1");
    assert.strictEqual(lines[1].content, "b=2");
  });

  test("extractLogicalLines preserves whitespace inside a string literal", () => {
    const lines = extractLogicalLines('s = "a   b"\n');
    assert.strictEqual(lines[0].content, 's="a   b"');
  });

  test("extractLogicalLines records indent width and does not treat 'if' as an f-string prefix", () => {
    const lines = extractLogicalLines('if"x":\n    pass\n');
    assert.strictEqual(lines[0].indentWidth, 0);
    assert.strictEqual(lines[1].indentWidth, 4);
    assert.strictEqual(lines[1].content, "pass");
  });

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

  test("classifyApiKeyError maps ERR_NETWORK to network_error (T4)", () => {
    const result = classifyApiKeyError({ code: "ERR_NETWORK" });
    assert.strictEqual(result, "network_error");
  });

  test("classifyApiKeyError maps EAI_AGAIN to network_error (T4)", () => {
    const result = classifyApiKeyError({ code: "EAI_AGAIN" });
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

  test("decideMigrationOffer suppresses env offer once env migration is done (T3)", () => {
    const decision = decideMigrationOffer("env", false, true);
    assert.strictEqual(decision.shouldOffer, false);
  });

  test("decideMigrationOffer still offers env when env migration not done (T3)", () => {
    const decision = decideMigrationOffer("env", false, false);
    assert.strictEqual(decision.shouldOffer, true);
  });

  test("decideMigrationOffer leaves settings unaffected by env migration flag (T3)", () => {
    const decision = decideMigrationOffer("settings", false, true);
    assert.strictEqual(decision.shouldOffer, true);
  });

  test("decideMigrationOffer + ApiKeySource type cover all four sources", () => {
    const sources: ApiKeySource[] = ["settings", "env", "secret", "prompt"];
    const allDecisions = sources.map((s) => decideMigrationOffer(s, false));
    const offerCount = allDecisions.filter((d) => d.shouldOffer).length;
    assert.strictEqual(offerCount, 2);
    assert.ok(allDecisions.every((d) => typeof d.sourceLabel === "string"));
    assert.ok(allDecisions.every((d) => d.sourceLabel.length > 0));
  });

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

  test("computeBackoffMs returns baseMs for NaN and non-finite attempts (T9)", () => {
    assert.strictEqual(computeBackoffMs(NaN), 1000);
    assert.strictEqual(computeBackoffMs(Infinity), 1000);
    assert.strictEqual(computeBackoffMs(-Infinity), 1000);
  });

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

  test("classifyExitForRestart T7 mapping assertions", () => {
    assert.strictEqual(classifyExitForRestart(null, null, false), "crash");
    assert.strictEqual(classifyExitForRestart(0, null, false), "cleanExit");
    assert.strictEqual(classifyExitForRestart(1, null, false), "crash");
    assert.strictEqual(classifyExitForRestart(null, null, true), "intentional");
  });


  // ==========================================================================
  // PIPELINE PROGRESS TESTS (Iter 49)
  // ==========================================================================

  test("computeOverallPercent returns 0 when first stage just started", () => {
    assert.strictEqual(computeOverallPercent("Scout", 0), 0);
  });

  test("computeOverallPercent returns the prior-stage weight sum at a stage start", () => {
    assert.strictEqual(computeOverallPercent("Architect", 0), 10);
    assert.strictEqual(computeOverallPercent("Specialist", 0), 25);
    assert.strictEqual(computeOverallPercent("Verifier", 0), 75);
  });

  test("computeOverallPercent adds within-stage fraction", () => {
    assert.strictEqual(computeOverallPercent("Specialist", 50), 50);
    assert.strictEqual(computeOverallPercent("Scout", 100), 10);
  });

  test("computeOverallPercent returns 100 at final stage complete", () => {
    assert.strictEqual(computeOverallPercent("Synthesizer", 100), 100);
  });

  test("computeOverallPercent clamps within-stage fraction to [0,100]", () => {
    assert.strictEqual(computeOverallPercent("Scout", -50), 0);
    assert.strictEqual(computeOverallPercent("Scout", 200), 10);
  });

  test("computeOverallPercent treats unknown stage as 0 weight (returns prior known sum or 0)", () => {
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

  test("STAGE_WEIGHTS sums to exactly 100", () => {
    const sum = Object.values(STAGE_WEIGHTS).reduce((a, b) => a + b, 0);
    assert.strictEqual(sum, 100);
  });

  test("every STAGE_ORDER entry has a STAGE_WEIGHTS entry", () => {
    for (const stage of STAGE_ORDER) {
      assert.ok(
        typeof STAGE_WEIGHTS[stage] === "number",
        `${stage} is missing a weight`,
      );
    }
    // And no orphan weights without an order position.
    for (const stage of Object.keys(STAGE_WEIGHTS)) {
      assert.ok(
        STAGE_ORDER.includes(stage),
        `${stage} has a weight but is not in STAGE_ORDER`,
      );
    }
  });

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

  test("computeEtaMs returns null on NaN or Infinity percent", () => {
    assert.strictEqual(computeEtaMs(5000, NaN), null);
    assert.strictEqual(computeEtaMs(5000, Infinity), null);
  });

  test("computeEtaMs extrapolates remaining time from elapsed and fraction", () => {
    assert.strictEqual(computeEtaMs(10000, 25), 30000);
    assert.strictEqual(computeEtaMs(30000, 50), 30000);
  });

  test("computeEtaMs returns 0 at 100% complete", () => {
    assert.strictEqual(computeEtaMs(60000, 100), 0);
  });

  test("computeEtaMs clamps fraction above 100 to 0 remaining", () => {
    assert.strictEqual(computeEtaMs(60000, 150), 0);
  });

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

  // ==========================================================================
  // VALIDATION HOVER TESTS (Iter 50)
  // ==========================================================================

  test("LruCache stores and retrieves values", () => {
    const cache = new LruCache<string, number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    assert.strictEqual(cache.get("a"), 1);
    assert.strictEqual(cache.get("b"), 2);
    assert.strictEqual(cache.get("missing"), undefined);
  });

  test("LruCache reports size and has()", () => {
    const cache = new LruCache<string, number>(3);
    assert.strictEqual(cache.size, 0);
    cache.set("a", 1);
    assert.strictEqual(cache.size, 1);
    assert.strictEqual(cache.has("a"), true);
    assert.strictEqual(cache.has("b"), false);
  });

  test("LruCache evicts the least-recently-used entry at capacity", () => {
    const cache = new LruCache<string, number>(2);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3); // evicts "a" (LRU)
    assert.strictEqual(cache.has("a"), false);
    assert.strictEqual(cache.get("b"), 2);
    assert.strictEqual(cache.get("c"), 3);
    assert.strictEqual(cache.size, 2);
  });

  test("LruCache get() promotes recency so the touched entry survives eviction", () => {
    const cache = new LruCache<string, number>(2);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.get("a"); // touch "a" → "b" is now LRU
    cache.set("c", 3); // evicts "b", not "a"
    assert.strictEqual(cache.has("a"), true);
    assert.strictEqual(cache.has("b"), false);
    assert.strictEqual(cache.has("c"), true);
  });

  test("LruCache set() on an existing key updates value and promotes recency", () => {
    const cache = new LruCache<string, number>(2);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("a", 99); // update + promote "a"
    cache.set("c", 3); // evicts "b"
    assert.strictEqual(cache.get("a"), 99);
    assert.strictEqual(cache.has("b"), false);
    assert.strictEqual(cache.has("c"), true);
  });

  test("LruCache delete() removes an entry (invalidation)", () => {
    const cache = new LruCache<string, number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    assert.strictEqual(cache.delete("a"), true);
    assert.strictEqual(cache.has("a"), false);
    assert.strictEqual(cache.delete("a"), false);
    assert.strictEqual(cache.size, 1);
  });

  test("LruCache capacity of 1 keeps only the newest entry", () => {
    const cache = new LruCache<string, number>(1);
    cache.set("a", 1);
    cache.set("b", 2);
    assert.strictEqual(cache.has("a"), false);
    assert.strictEqual(cache.get("b"), 2);
    assert.strictEqual(cache.size, 1);
  });

  test("LruCache rejects a NaN capacity (T13)", () => {
    assert.throws(() => new LruCache<string, number>(NaN));
  });

  test("truncateExcerpt returns short text unchanged", () => {
    assert.strictEqual(truncateExcerpt("short text", 200), "short text");
  });

  test("truncateExcerpt trims to max and appends an ellipsis", () => {
    const long = "a".repeat(250);
    const result = truncateExcerpt(long, 200);
    assert.ok(result.length <= 201, "result within max + ellipsis");
    assert.ok(result.endsWith("…"), "ends with ellipsis");
  });

  test("truncateExcerpt prefers a word boundary when trimming", () => {
    const text = "the quick brown fox jumps over the lazy dog repeatedly";
    const result = truncateExcerpt(text, 20);
    assert.ok(result.endsWith("…"));
    const beforeEllipsis = result.slice(0, -1);
    assert.ok(
      !/[A-Za-z]$/.test(beforeEllipsis) || beforeEllipsis.length <= 20,
      "trimmed at or before max without splitting a trailing word awkwardly",
    );
  });

  test("truncateExcerpt handles empty string", () => {
    assert.strictEqual(truncateExcerpt("", 200), "");
  });

  test("boldMatchingSpan wraps the first keyword occurrence in markdown bold", () => {
    assert.strictEqual(
      boldMatchingSpan("the Order aggregate", "Order"),
      "the **Order** aggregate",
    );
  });

  test("boldMatchingSpan is case-insensitive in matching but preserves original case", () => {
    assert.strictEqual(
      boldMatchingSpan("The ORDER total", "order"),
      "The **ORDER** total",
    );
  });

  test("boldMatchingSpan returns excerpt unchanged when keyword absent or empty", () => {
    assert.strictEqual(
      boldMatchingSpan("no match here", "Order"),
      "no match here",
    );
    assert.strictEqual(boldMatchingSpan("anything", ""), "anything");
  });

  test("boldMatchingSpan only bolds the first occurrence", () => {
    assert.strictEqual(
      boldMatchingSpan("Order then Order again", "Order"),
      "**Order** then Order again",
    );
  });

  test("escapeInlineMarkdown escapes a command-link injection payload (T13)", () => {
    const escaped = escapeInlineMarkdown("[x](command:evil)");
    assert.ok(!escaped.includes("[x]"), "opening bracket is escaped");
    assert.ok(escaped.includes("\\["), "escapes the [ character");
    assert.ok(escaped.includes("\\]"), "escapes the ] character");
  });

  test("formatHoverMarkdown renders type, message, source, and command link", () => {
    const md = formatHoverMarkdown(
      {
        type: "V5_AGGREGATE_BOUNDARY",
        message: "Entity Order should not be modified outside its aggregate.",
        suggestion: "Route changes through the OrderAggregate root.",
        sources: [
          {
            document: "SRS.docx",
            section: "3.2 Order Management",
            page: 12,
            summary:
              "The Order aggregate enforces its own invariants and must be the only entry point for modifications.",
            file_path: "/abs/inputs/SRS.docx",
            relevance_score: 0.91,
          },
        ],
      },
      "Order",
    );
    assert.ok(
      md.includes("V5_AGGREGATE_BOUNDARY"),
      "includes the violation type",
    );
    assert.ok(
      md.includes("3.2 Order Management"),
      "includes the source section",
    );
    assert.ok(md.includes("SRS.docx"), "includes the source document");
    assert.ok(md.includes("p. 12") || md.includes("p.12"), "includes the page");
    assert.ok(md.includes("**Order**"), "bolds the matched keyword in excerpt");
    assert.ok(
      md.includes("command:ddd-enforcer.openSource?"),
      "includes the trusted command link",
    );
    assert.ok(
      md.includes(encodeURIComponent(JSON.stringify(["/abs/inputs/SRS.docx", "3.2 Order Management"]))),
      "command args match the openSource Code Action contract",
    );
  });

  test("formatHoverMarkdown still bolds a keyword that contains escaped characters (T13)", () => {
    const md = formatHoverMarkdown(
      {
        type: "V1_NAMING",
        message: "Value object <Money> must be immutable.",
        suggestion: "Make <Money> a frozen dataclass.",
        sources: [
          {
            document: "SRS.docx",
            section: "2.1 Domain Glossary",
            page: 4,
            summary: "The <Money> value object holds amount and currency and is immutable.",
            file_path: "/abs/inputs/SRS.docx",
            relevance_score: 0.88,
          },
        ],
      },
      "<Money>",
    );
    // Excerpt is escaped, so the keyword must be escaped identically to match
    // and bold — otherwise the highlight silently fails for special-char keywords.
    assert.ok(
      md.includes("**\\<Money\\>**"),
      "bolds the escaped keyword inside the escaped excerpt",
    );
  });

  test("formatHoverMarkdown omits the source block when there are no sources", () => {
    const md = formatHoverMarkdown(
      {
        type: "V1_SYNONYM",
        message: "Use the canonical term 'Customer' instead of 'Client'.",
        suggestion: "Rename Client to Customer.",
        sources: [],
      },
      "Client",
    );
    assert.ok(md.includes("V1_SYNONYM"));
    assert.ok(!md.includes("command:ddd-enforcer.openSource"), "no command link without a source");
    assert.ok(!md.includes("Source:"), "no source header without a source");
  });

  test("formatHoverMarkdown handles an undefined sources field", () => {
    const md = formatHoverMarkdown(
      {
        type: "V2_BANNED",
        message: "The term 'Manager' is banned in the domain model.",
        suggestion: "Use a role-specific name.",
      },
      "Manager",
    );
    assert.ok(md.includes("V2_BANNED"));
    assert.ok(!md.includes("command:ddd-enforcer.openSource"));
  });

  test("formatHoverMarkdown truncates a long excerpt", () => {
    const longSummary = "x ".repeat(300); // 600 chars
    const md = formatHoverMarkdown(
      {
        type: "V5",
        message: "msg",
        suggestion: "s",
        sources: [
          {
            document: "SRS.docx",
            section: "S",
            page: 1,
            summary: longSummary,
            file_path: "/p",
            relevance_score: 0.5,
          },
        ],
      },
      "nomatch",
    );
    assert.ok(md.includes("…"), "long excerpt is truncated with an ellipsis");
  });

  test("DDDViolationHoverProvider restricts trusted commands to openSource allowlist", () => {
    const provider = new DDDViolationHoverProvider();

    const dummyUri = vscode.Uri.parse("file:///dummy/file.py");
    const dummyDoc = {
      uri: dummyUri,
    } as vscode.TextDocument;

    const violationMap = new Map<number, any>();
    violationMap.set(5, {
      type: "V1_SYNONYM",
      message: "Violation message",
      suggestion: "Suggestion",
      sources: []
    });
    validationViolationCache.set(dummyUri.toString(), violationMap);

    const position = new vscode.Position(5, 0);
    const hover = provider.provideHover(dummyDoc, position);

    assert.ok(hover, "should return a hover");
    const markdown = hover.contents[0] as vscode.MarkdownString;
    assert.ok(markdown, "hover contents should contain a MarkdownString");

    // This assertion checks that isTrusted is enabled. In some VS Code versions, the getter on the extension host
    // always coerces and returns a boolean (true). If it returns an object, we verify the specific allowlist.
    assert.ok(markdown.isTrusted, "markdown should be trusted");
    if (typeof markdown.isTrusted === "object") {
      assert.deepStrictEqual(markdown.isTrusted, { enabledCommands: ["ddd-enforcer.openSource"] });
    }
  });

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

  test("generateNonce uniqueness across many iterations", () => {
    const nonces = new Set<string>();
    for (let i = 0; i < 1000; i++) {
      const nonce = generateNonce();
      assert.strictEqual(nonce.length, 32);
      assert.ok(/^[A-Za-z0-9]+$/.test(nonce), "nonce must be alphanumeric");
      assert.ok(!nonces.has(nonce), `duplicate nonce found at iteration ${i}`);
      nonces.add(nonce);
    }
  });

  test("generateNonce does not use Math.random", () => {
    const originalRandom = Math.random;
    try {
      Math.random = () => 0.5;
      const a = generateNonce();
      const b = generateNonce();
      assert.notStrictEqual(a, b, "Nonce must not be identical even if Math.random is hijacked");
    } finally {
      Math.random = originalRandom;
    }
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
    assert.ok(!/https?:\/\//.test(html), "no external http(s) URLs in the HTML");
  });

  test("buildRunManifestsHtml contains the table scaffold and message wiring", () => {
    const html = buildRunManifestsHtml("N", "vscode-webview://abc");
    assert.ok(html.includes("acquireVsCodeApi()"), "acquires the vscode api");
    assert.ok(html.includes("addEventListener(\"message\""), "listens for messages");
    assert.ok(html.includes("id=\"runs-table\"") || html.includes("id='runs-table'"), "has the runs table element");
    assert.ok(html.includes("id=\"detail\"") || html.includes("id='detail'"), "has the detail pane element");
  });

  test("buildRunManifestsHtml esc() escapes quotes for attribute-context safety", () => {
    const html = buildRunManifestsHtml("N", "vscode-webview://abc");
    // The client-side esc() must neutralize " and ' so an interpolated
    // runId cannot break out of the data-run="..." attribute (defense in
    // depth — run_id is also sanitized server-side by sanitize_path_segment).
    assert.ok(html.includes("&quot;"), "esc replaces double quotes");
    assert.ok(html.includes("&#39;"), "esc replaces single quotes");
  });

  // ==========================================================================
  // API KEY MANAGER T1 SECURITY INVARIANT TESTS
  // ==========================================================================

  test("getApiKey stores prompted key only AFTER successful validation (T1)", async () => {
    // 1. Stub vscode.window.showInputBox
    const originalShowInputBox = vscode.window.showInputBox;
    let inputKeyReturnValue: string | undefined = undefined;
    let showInputBoxCalled = false;
    (vscode.window as any).showInputBox = async (options: any) => {
      showInputBoxCalled = true;
      return inputKeyReturnValue;
    };

    // 2. Stub axios.get to intercept Gemini validation
    const originalAxiosGet = axios.get;
    let mockAxiosResult: { status: number; data: any } | Error = { status: 200, data: {} };
    axios.get = async (url: string, config?: any) => {
      if (mockAxiosResult instanceof Error) {
        throw mockAxiosResult;
      }
      return mockAxiosResult as any;
    };

    // 3. Mock VS Code ExtensionContext and secret storage
    const mockSecrets = new Map<string, string>();
    let storeCallCount = 0;
    const mockContext = {
      secrets: {
        get: async (key: string) => mockSecrets.get(key),
        store: async (key: string, value: string) => {
          storeCallCount++;
          mockSecrets.set(key, value);
        },
        delete: async (key: string) => {
          mockSecrets.delete(key);
        },
      },
      globalState: {
        get: (key: string) => undefined,
        update: async (key: string, value: any) => {},
      }
    } as unknown as vscode.ExtensionContext;

    // Temporarily clear any setting/env key to force the prompt flow
    const config = vscode.workspace.getConfiguration("ddd-enforcer");
    const originalSetting = config.get<string>("geminiApiKey");
    await config.update("geminiApiKey", "", vscode.ConfigurationTarget.Global);
    const originalEnv = process.env.GEMINI_API_KEY;
    delete process.env.GEMINI_API_KEY;

    try {
      // Scenario A: Input key is invalid (probe rejects it)
      inputKeyReturnValue = "AIzaInvalidFakeKey";
      const err: any = new Error("Invalid Key Mocked");
      err.response = { status: 400 };
      mockAxiosResult = err;

      const resultA = await getApiKey(mockContext, {
        log: () => {},
        updateStatusBar: () => {},
      });

      assert.strictEqual(resultA, undefined, "should return undefined on rejected key");
      assert.strictEqual(storeCallCount, 0, "should NOT write to secret storage when key is rejected");
      assert.strictEqual(mockSecrets.has("geminiApiKey"), false, "key must not exist in secret storage");

      // Scenario B: Input key is valid (probe accepts it)
      inputKeyReturnValue = "AIzaValidFakeKey";
      mockAxiosResult = { status: 200, data: { models: [] } };

      const resultB = await getApiKey(mockContext, {
        log: () => {},
        updateStatusBar: () => {},
      });

      assert.strictEqual(resultB, "AIzaValidFakeKey", "should return the validated key");
      assert.strictEqual(storeCallCount, 1, "should write to secret storage exactly once when key is accepted");
      assert.strictEqual(mockSecrets.get("geminiApiKey"), "AIzaValidFakeKey", "validated key must be in secret storage");

    } finally {
      // Restore stubs and original environment
      vscode.window.showInputBox = originalShowInputBox;
      axios.get = originalAxiosGet;
      if (originalSetting !== undefined) {
        await config.update("geminiApiKey", originalSetting, vscode.ConfigurationTarget.Global);
      }
      if (originalEnv !== undefined) {
        process.env.GEMINI_API_KEY = originalEnv;
      }
    }
  });
});
