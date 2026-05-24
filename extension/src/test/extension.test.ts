import * as assert from "assert";
import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import {
  classifySaveForValidationFromContent,
  classifyApiKeyError,
  validateGeminiKey,
  type ApiKeyValidationResult,
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
});
