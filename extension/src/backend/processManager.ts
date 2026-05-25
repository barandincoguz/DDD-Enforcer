/**
 * DDD Enforcer VS Code Extension
 * Backend Process Lifecycle and Network Port Management
 */

import * as vscode from "vscode";
import * as net from "net";

/** Outcome bucket for a backend exit event. */
export type ExitDisposition = "intentional" | "crash" | "cleanExit";

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

/**
 * Render a human-readable description of a Node child-process exit
 * event. Signal takes priority because a signal-kill carries more
 * diagnostic information than the resulting exit code.
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
  // signal === null past this point; only exit code 0 is a clean exit.
  return code === 0 ? "cleanExit" : "crash";
}

/**
 * Checks if a port is available on localhost.
 */
export function isPortAvailable(port: number): Promise<boolean> {
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
 * Finds an available port starting from the configured port.
 */
export async function findAvailablePort(log: (msg: string) => void): Promise<number> {
  const cfg = vscode.workspace.getConfiguration("ddd-enforcer");
  const preferredPort = cfg.get<number>("backendPort", 8000);

  // Try the preferred port first
  if (await isPortAvailable(preferredPort)) {
    return preferredPort;
  }

  const maxPort = Math.min(preferredPort + 100, 65535);
  log(
    `Preferred port ${preferredPort} is in use. Scanning for an available port in the next ${Math.max(0, maxPort - 1 - preferredPort)} candidates...`,
  );
  for (let port = preferredPort + 1; port < maxPort; port++) {
    if (await isPortAvailable(port)) {
      log(`Selected port ${port} (preferred port ${preferredPort} was unavailable).`);
      return port;
    }
  }

  log(
    `WARNING: no available port found in ${preferredPort}..${maxPort - 1}. Falling back to preferred port ${preferredPort} (backend startup is likely to fail).`,
  );
  return preferredPort;
}
