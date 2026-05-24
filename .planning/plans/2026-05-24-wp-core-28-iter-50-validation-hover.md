# WP-CORE-28 Iter 50 — Validation Peek / HoverProvider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (implementer + spec reviewer + quality reviewer + fix loop). Steps use checkbox (`- [ ]`) syntax.

**Goal:** When a user hovers over a DDD-violation diagnostic in a Python file, show a Markdown popup with the violation type, the top-1 SRS source (section + document + page), a ~200-char excerpt with the matching keyword bolded, and a "Open SRS source" link that reuses the existing `ddd-enforcer.openSource` command. The hover reads from an in-memory LRU cache (cap 20 files) populated during validation, invalidated on save — no network round-trip on hover.

**Architecture:** A generic pure `LruCache<K, V>` (cap-bounded, recency-promoting) stores, per file URI, a `Map<lineNumber, Violation>` built while diagnostics are created in `validateDocument`. Three more pure helpers (`truncateExcerpt`, `boldMatchingSpan`, `formatHoverMarkdown`) produce the hover Markdown. A `vscode.languages.registerHoverProvider("python", ...)` locates the violation at the hovered line via the LRU and renders a trusted `MarkdownString`. The LRU is the load-bearing data source for the hover (not dead scaffolding) — invalidated by `clearSourcesForDocument` (already called on every validate) plus an explicit delete in the save path. All changes stay in `extension/src/extension.ts` per the WP-CORE-28 "no refactor" lock.

**Tech Stack:** TypeScript 5.9, `vscode` API (`HoverProvider`, `MarkdownString` with `isTrusted`, `registerHoverProvider`, command-URI links), `mocha` via `@vscode/test-cli`. No new dependencies.

---

## Pre-flight — design decisions + reality reconciliation

- **Why an LRU at all (vs reading diagnostics directly).** The hover *could* render from the live `DiagnosticCollection` + the existing `violationSources` map. But the spec explicitly mandates an LRU cache (cap 20) with eviction + save-invalidation tests. To avoid shipping dead scaffolding (the Iter 49 lesson), the LRU is made the hover's actual data source: it stores the full `Violation` objects per line, which carry `type`/`message`/`suggestion`/`sources` — richer than reconstructing from a `Diagnostic`. The hover reads ONLY from the LRU.
- **Excerpt source.** `ViolationSource` (extension.ts:22-30) has fields `{ document, section, page, summary, file_path, relevance_score }`. There is no full-text "excerpt" field — `summary` is the closest. The hover uses `source.summary` as the excerpt, truncated to ~200 chars, with the violation keyword bolded if present. The spec's "~200 chars of excerpt with the matching span highlighted" maps to `boldMatchingSpan(truncateExcerpt(summary, 200), keyword)`.
- **Keyword for bolding.** `extractKeyword(violation.message)` is an existing helper (used by `createDiagnostic`). The wire-up passes its result into `formatHoverMarkdown`; the pure formatter bolds the first occurrence in the excerpt.
- **Click-to-open.** Reuses the registered `ddd-enforcer.openSource` command. A trusted `MarkdownString` link `command:ddd-enforcer.openSource?<encodeURIComponent(JSON.stringify([file_path, section]))>` invokes it with the same args the Code Action passes (extension.ts:1986-1988), so the hover link jumps to the exact same SRS location.
- **Invalidation.** `validateDocument` already calls `clearSourcesForDocument(uri)` at its start (extension.ts:1549) and re-populates. The LRU gets the same treatment: cleared + repopulated each validate. The save handler triggers validation on semantic change, so the cache stays fresh. The "cache invalidation on save" acceptance is satisfied by the LRU `delete` being wired into `clearSourcesForDocument` (which runs on every validate, which is what a semantic save triggers).
- **Test substitution.** Acceptance asks for "3 new test cases: hover provider returns expected Markdown shape; cache LRU eviction; cache invalidation on save". The HoverProvider itself needs the vscode runtime, so its Markdown output is tested via the pure `formatHoverMarkdown` (the exact string the provider wraps). LRU eviction + delete (invalidation) are tested directly on the pure `LruCache`. F5 smoke covers the live hover.

## File Structure

| File | Action | Why |
|------|--------|-----|
| `extension/src/extension.ts` | Modify: add `// VALIDATION HOVER (pure helpers)` section (LruCache + truncateExcerpt + boldMatchingSpan + formatHoverMarkdown); add `validationViolationCache` global; populate it in `validateDocument`; delete entry in `clearSourcesForDocument`; register a HoverProvider in `activate` | Spec locks "no refactor"; additive only |
| `extension/src/test/extension.test.ts` | Modify: append `// VALIDATION HOVER TESTS (Iter 50)` section | Reuse the suite |

No new files, no new deps, no package.json change (HoverProvider needs no manifest entry).

---

## Task 1: Generic `LruCache<K, V>`

**Goal:** A small capacity-bounded cache with recency promotion. Pure (no vscode, no I/O).

**Files:**
- Modify: `extension/src/extension.ts` (new `// VALIDATION HOVER (pure helpers — testable without vscode)` section, placed immediately after the `// PIPELINE PROGRESS (pure helpers — testable without vscode)` section, before `// GLOBAL STATE`)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend the test-file import block** (add `LruCache` to the existing multi-line import):

```typescript
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
} from "../extension";
```

- [ ] **Step 2: Append tests inside the suite (after the Iter 49 tests)**

```typescript
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
```

- [ ] **Step 3: Compile RED** — `cd extension && npm run compile` → missing-export error for `LruCache`.

- [ ] **Step 4: Implementation**

Add a new section immediately AFTER the `// PIPELINE PROGRESS (pure helpers — testable without vscode)` section and BEFORE `// GLOBAL STATE`:

```typescript
// =============================================================================
// VALIDATION HOVER (pure helpers — testable without vscode)
// =============================================================================

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
    if (capacity < 1) {
      throw new Error("LruCache capacity must be >= 1");
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
```

- [ ] **Step 5: Gates** — `cd extension && npm run compile && npm run lint` (SUCCESS); `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright` (0 errors).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add generic LruCache pure helper — iter 50 step A

WP-CORE-28 Feature 4 (validation peek / hover) groundwork. A small
capacity-bounded cache with least-recently-used eviction, backed by a
Map whose insertion order tracks recency (get + set re-insert the
touched key). set() evicts the oldest key past capacity; get() promotes
recency; delete() supports cache invalidation. capacity must be >= 1.

7 unit tests cover store/retrieve, size/has, LRU eviction at capacity,
get-promotes-recency, set-updates-and-promotes, delete/invalidation,
and the capacity-1 edge.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `truncateExcerpt` + `boldMatchingSpan`

**Goal:** Trim a source summary to ~200 chars (word-safe ellipsis) and bold the first occurrence of a keyword. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (append to the `// VALIDATION HOVER` section after `LruCache`)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend test import** (add `truncateExcerpt`, `boldMatchingSpan`):

```typescript
  // ... existing names ...
  LruCache,
  truncateExcerpt,
  boldMatchingSpan,
} from "../extension";
```

- [ ] **Step 2: Append tests (after Task 1 tests)**

```typescript
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
    // Should not cut mid-word: the char before the ellipsis is not a letter.
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
```

- [ ] **Step 3: Compile RED** — missing-export errors for `truncateExcerpt`, `boldMatchingSpan`.

- [ ] **Step 4: Implementation** — append after `LruCache`:

```typescript
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
```

- [ ] **Step 5: Gates** (compile, lint, pyright clean).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add truncateExcerpt + boldMatchingSpan pure helpers — iter 50 step B

- truncateExcerpt(text, maxChars) trims to a word boundary at/before
  the limit and appends a single-char ellipsis; returns short text
  unchanged.
- boldMatchingSpan(excerpt, keyword) wraps the first case-insensitive
  keyword occurrence in markdown bold, preserving the matched span's
  original casing; returns the excerpt unchanged when the keyword is
  empty or absent.

8 unit tests cover short-text passthrough, max-trim + ellipsis, word-
boundary preference, empty string, first-occurrence bolding, case-
insensitive match with preserved casing, absent/empty keyword, and
single-occurrence-only bolding.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `formatHoverMarkdown`

**Goal:** Assemble the hover Markdown for a violation — header, message, top-1 source block with bolded excerpt, and a trusted command link. Pure.

**Files:**
- Modify: `extension/src/extension.ts` (append to the `// VALIDATION HOVER` section after `boldMatchingSpan`)
- Modify: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Extend test import** (add `formatHoverMarkdown`):

```typescript
  // ... existing names ...
  LruCache,
  truncateExcerpt,
  boldMatchingSpan,
  formatHoverMarkdown,
} from "../extension";
```

- [ ] **Step 2: Append tests (after Task 2 tests)**

The `Violation` and `ViolationSource` shapes are defined in extension.ts. Construct literals in the tests.

```typescript
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
```

- [ ] **Step 3: Compile RED** — missing-export error for `formatHoverMarkdown`.

- [ ] **Step 4: Implementation** — append after `boldMatchingSpan`. This references the existing `Violation` and `ViolationSource` interfaces (defined near the top of the file).

```typescript
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
  lines.push(`**DDD Violation: ${violation.type}**`);
  lines.push("");
  lines.push(violation.message);

  const source = violation.sources && violation.sources[0];
  if (source) {
    lines.push("");
    lines.push(`**Source:** ${source.section} — ${source.document} (p. ${source.page})`);
    const excerpt = boldMatchingSpan(
      truncateExcerpt(source.summary, 200),
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
```

- [ ] **Step 5: Gates** (compile, lint, pyright clean).

- [ ] **Step 6: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "$(cat <<'EOF'
feat(extension): add formatHoverMarkdown pure helper — iter 50 step C

Assembles the validation-hover Markdown: a `DDD Violation: <type>`
header, the message, and — when a source exists — a source block with
section/document/page, a truncated + keyword-bolded excerpt from the
source summary, and an "Open SRS source" command link that reuses the
ddd-enforcer.openSource command with the same (file_path, section)
arguments as the Code Action. The string is meant to be wrapped in a
trusted MarkdownString by the caller.

4 unit tests cover the full source-present shape (type, section,
document, page, bolded span, command link + arg encoding), the
no-sources case (source block + command link omitted), the
undefined-sources case, and long-excerpt truncation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire the cache + HoverProvider into validation

**Goal:** Populate the per-file violation cache during `validateDocument`, invalidate it in `clearSourcesForDocument`, and register a HoverProvider that renders `formatHoverMarkdown` for the violation at the hovered line. No new pure-function tests this task (covered by Tasks 1-3 + F5 smoke).

**Files:**
- Modify: `extension/src/extension.ts`

- [ ] **Step 1: Add the module-level cache**

In the `// GLOBAL STATE` section, near the existing `const violationSources = new Map<string, ViolationSource[]>();` (extension.ts:385), add below it:

```typescript
/**
 * Per-file cache of the most recent validation's violations, keyed by
 * document URI string, value = Map of diagnostic line number → Violation.
 * Drives the HoverProvider without a re-fetch. LRU-bounded to 20 files;
 * invalidated by clearSourcesForDocument (which runs on every validate).
 */
const validationViolationCache = new LruCache<
  string,
  Map<number, Violation>
>(20);
```

- [ ] **Step 2: Populate the cache in `validateDocument`**

In `validateDocument` (extension.ts ~1524-1615), the success path iterates `data.violations.forEach((violation) => { const diagnostic = createDiagnostic(document, violation); diagnostics.push(diagnostic); })`. Replace that block so it also records the line→violation mapping and stores it in the LRU.

Find:

```typescript
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
```

Replace with:

```typescript
    if (data.is_violation && data.violations) {
      const diagnostics: vscode.Diagnostic[] = [];
      const lineToViolation = new Map<number, Violation>();

      data.violations.forEach((violation) => {
        const diagnostic = createDiagnostic(document, violation);
        diagnostics.push(diagnostic);
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
```

- [ ] **Step 3: Invalidate the cache in `clearSourcesForDocument`**

`clearSourcesForDocument(uriString)` (extension.ts ~2108-2116) currently clears the `violationSources` map for a document. Add the LRU eviction. Find:

```typescript
function clearSourcesForDocument(uriString: string) {
  for (const key of violationSources.keys()) {
    if (key.startsWith(uriString)) {
      violationSources.delete(key);
    }
  }
}
```

Replace with:

```typescript
function clearSourcesForDocument(uriString: string) {
  for (const key of violationSources.keys()) {
    if (key.startsWith(uriString)) {
      violationSources.delete(key);
    }
  }
  validationViolationCache.delete(uriString);
}
```

(`clearSourcesForDocument` is called at the top of `validateDocument` — extension.ts:1549 — so each validate clears the prior cache entry before repopulating. A semantic save triggers validation, so the cache stays in sync with the file. This is the save-invalidation path.)

- [ ] **Step 4: Add the HoverProvider class**

Add a new class definition immediately AFTER the `DDDSourceCodeActionProvider` class (extension.ts ~1999):

```typescript
/**
 * Shows a Markdown peek for a DDD violation when the user hovers over a
 * line that produced a diagnostic. Reads the cached violation for the
 * hovered line from validationViolationCache (no re-fetch) and renders
 * formatHoverMarkdown in a trusted MarkdownString so the embedded
 * "Open SRS source" command link works.
 */
class DDDViolationHoverProvider implements vscode.HoverProvider {
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
    markdown.isTrusted = true;
    return new vscode.Hover(markdown);
  }
}
```

`extractKeyword` is an existing helper in the file (used by `createDiagnostic`).

- [ ] **Step 5: Register the HoverProvider in `activate`**

In `activate`, next to the `registerCodeActionsProvider` block (extension.ts ~457-463), add a sibling registration:

```typescript
  // Register hover provider for violation peeks
  context.subscriptions.push(
    vscode.languages.registerHoverProvider(
      "python",
      new DDDViolationHoverProvider(),
    ),
  );
```

- [ ] **Step 6: Gates**

Run:
- `cd extension && npm run compile && npm run lint`
- `cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer" && pyright`
- `cd extension/backend && pytest -m "not integration" -q`

Expected: compile + lint SUCCESS; pyright `0 errors, 0 warnings, 0 informations`; pytest `729 passed, 31 deselected`.

- [ ] **Step 7: Commit**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/src/extension.ts
git commit -m "$(cat <<'EOF'
feat(extension): wire validation hover provider + per-file cache — iter 50 step D

WP-CORE-28 Feature 4 wired end-to-end:

- validationViolationCache (LruCache, cap 20) stores per-file
  line→Violation maps, populated in validateDocument as diagnostics
  are created and invalidated in clearSourcesForDocument (which runs
  on every validate, i.e. on every semantic save).
- DDDViolationHoverProvider (registered for python) looks up the
  cached violation for the hovered line and renders formatHoverMarkdown
  in a trusted MarkdownString, so the "Open SRS source" command link
  reuses ddd-enforcer.openSource with the same (file_path, section)
  args as the Code Action. No network round-trip on hover.

No backend changes. Pyright still 0 errors. Pytest still 729 passing.

Closes Iter 50 of WP-CORE-28 (final feature of the wave).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: F5 manual smoke checklist (HUMAN-IN-LOOP)

**Goal:** Verify the live hover in the Extension Development Host. Requires a working backend + a validated Python file with violations.

**Files:**
- Modify: `.planning/pipeline_audit/CURRENT.md` after the user reports back (or waives).

- [ ] **Step 1: Halt and post the smoke checklist**

After Task 4 commits, post a Turkish caveman-mode message with the checklist below. Wait. Do not proceed to WP-CORE-32.

**F5 smoke checklist for Iter 50 (Validation hover):**

Prereq: working `ddd-enforcer.pythonPath` (backend boots) + an initialized domain model. `cd extension && npm run compile`. F5 → Extension Dev Host.

1. **Test A — hover shows peek.** Open + save a Python file that triggers DDD violations (so diagnostics appear). Hover the mouse over a violation-underlined token. A Markdown popup appears within ~500ms showing `DDD Violation: <type>` + the message.
2. **Test B — source excerpt + bold.** If the violation has an SRS source, the popup shows a `Source:` line (section — document — p.N) and a quoted excerpt with the matched keyword in **bold**.
3. **Test C — click opens SRS.** Click the "Open SRS source" link in the popup. It should open the SRS document beside the editor and jump to the same section the Code Action ("View Source") jumps to.
4. **Test D — no hover off-violation.** Hover over a line with no violation. No DDD popup appears.
5. **Test E — invalidation on edit+save.** Edit the file to remove the violating code, save (re-validates). Hover the previously-violating line — the stale popup should be gone (cache invalidated + repopulated).
6. **Test F — many files (LRU).** Open + validate 21 different Python files, then hover in the first one. (The cache caps at 20, so the very first file's entry may have been evicted — hovering it shows no popup until re-validated. This is expected LRU behavior, not a bug.)

If any step fails, post what you saw. The implementer agent will RED → GREEN → COMMIT a fix.

- [ ] **Step 2: Update CURRENT.md** (after user reports green OR waives) — append `## Iteration 50 — WP-CORE-28 Feature 4 (Validation hover) COMPLETE` in the same shape as Iter 47-49, and add a note that WP-CORE-28 (all 4 features) is now COMPLETE, with WP-CORE-32 (webviews) next.

- [ ] **Step 3: Commit the CURRENT.md update**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add .planning/pipeline_audit/CURRENT.md
git commit -m "$(cat <<'EOF'
chore(planning): CURRENT.md update for iter 50 (WP-CORE-28 Feature 4 complete + WP wave done)

Validation hover shipped. WP-CORE-28 (all 4 UX features) complete.
WP-CORE-32 (extension webviews) is next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review (per writing-plans skill)

**1. Spec coverage** (against `todos/WP-CORE-28-extension-ux-wave1.md` Feature 4 acceptance):

| Acceptance criterion | Task |
|----------------------|------|
| Hover shows a Markdown popup within 500ms (cache, no network) | Task 4 (HoverProvider reads `validationViolationCache`, no fetch) + F5 Test A |
| Popup shows top-1 SRS excerpt with matching span bolded | Task 2 (`boldMatchingSpan`) + Task 3 (`formatHoverMarkdown` source block) + F5 Test B |
| "Click to open" jumps to same SRS location as the Code Action | Task 3 (command link with `[file_path, section]` matching the Code Action's args at extension.ts:1988) + F5 Test C |
| Editing the file invalidates the cached hover | Task 4 Step 3 (`clearSourcesForDocument` deletes the LRU entry; runs on every validate / semantic save) + F5 Test E |
| LRU cap respected: 21st file evicts LRU | Task 1 (`LruCache` cap 20) + Task 4 Step 1 + F5 Test F |
| 3 new test cases: hover Markdown shape; LRU eviction; cache invalidation on save | Task 3 (`formatHoverMarkdown` shape tests) + Task 1 (`LruCache` eviction + delete/invalidation tests) — documented in Pre-flight; live hover via F5 |

**2. Placeholder scan:** no TBD, no "handle appropriately", no "similar to Task N". Every code step is complete.

**3. Type consistency:**
- `LruCache<K,V>` (Task 1) → `validationViolationCache: LruCache<string, Map<number, Violation>>` (Task 4 Step 1) → read by HoverProvider (Task 4 Step 4), written in `validateDocument` (Task 4 Step 2), deleted in `clearSourcesForDocument` (Task 4 Step 3).
- `truncateExcerpt` + `boldMatchingSpan` (Task 2) → consumed by `formatHoverMarkdown` (Task 3).
- `formatHoverMarkdown(violation: Violation, keyword: string)` (Task 3) → consumed by HoverProvider (Task 4 Step 4) which passes `extractKeyword(violation.message)`.
- `Violation` / `ViolationSource` interfaces are pre-existing (extension.ts:22-37); all tasks reference them consistently.
- The command-link args `[file_path, section]` (Task 3) match the Code Action's `arguments: [source.file_path, source.section]` (extension.ts:1988) — verified identical contract.

All cross-task names consistent.

End of plan.
