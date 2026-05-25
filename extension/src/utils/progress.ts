/**
 * DDD Enforcer VS Code Extension
 * Pipeline Progress Estimation and Duration Math Helpers
 */

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
