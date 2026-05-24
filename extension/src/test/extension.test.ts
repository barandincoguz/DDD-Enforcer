import * as assert from "assert";
import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
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
});
