/**
 * DDD Enforcer VS Code Extension
 * Gemini API Key Management, Validation, and Secure Storage Migration
 */

import * as vscode from "vscode";
import axios from "axios";
import { log, updateStatusBar } from "./extension";

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
    "ERR_NETWORK",
    "ERR_BAD_RESPONSE",
    "EAI_AGAIN",
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
      const resp = await axios.get(url, {
        timeout: 5000,
        headers: { "x-goog-api-key": trimmed },
      });
      return { status: resp.status, data: resp.data };
    });
  try {
    const url = GEMINI_MODELS_URL_BASE; // key now in header, not query
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

/**
 * Gets the Gemini API key from settings, env var, or prompts the user.
 *
 * - Tracks where the key was sourced from (settings/env/secret/prompt).
 * - Pre-validates the key against Gemini's public models endpoint
 *   (cheap, no backend round-trip) before returning it.
 * - On rejection, surfaces a kind-specific toast and returns undefined.
 * - On success, offers migration to secret storage if the source was
 *   the less-secure settings or env path. The user's decline is
 *   persisted to globalState ("apiKeyMigrationDeclined") so the offer
 *   does not repeat next session.
 */
export async function getApiKey(
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

  // Persist a prompted key only after it validates (prevents lockout on a bad key).
  if (source === "prompt") {
    await context.secrets.store("geminiApiKey", apiKey);
  }

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
  }

  return apiKey;
}
