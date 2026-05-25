/**
 * DDD Enforcer VS Code Extension
 * Hover Diagnostics Tooltips and LRU Violation Caching
 */

import { Violation } from "../types";

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
    if (!Number.isFinite(capacity) || capacity < 1) {
      throw new Error("LruCache capacity must be a finite number >= 1");
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
 * Escape the Markdown control characters that could let an LLM-derived
 * string break out of inline text — backslash, backtick, angle brackets,
 * and square brackets (the latter two enable HTML / link injection).
 * Defense-in-depth for hover fields rendered in a trusted MarkdownString.
 * Pure.
 */
export function escapeInlineMarkdown(s: string): string {
  return s.replace(/[\\`<>\[\]]/g, "\\$&");
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
  lines.push(`**DDD Violation: ${escapeInlineMarkdown(violation.type)}**`);
  lines.push("");
  lines.push(escapeInlineMarkdown(violation.message));

  const source = violation.sources && violation.sources[0];
  if (source) {
    lines.push("");
    lines.push(
      `**Source:** ${escapeInlineMarkdown(source.section)} — ${escapeInlineMarkdown(source.document)} (p. ${source.page})`,
    );
    const excerpt = boldMatchingSpan(
      escapeInlineMarkdown(truncateExcerpt(source.summary, 200)),
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
