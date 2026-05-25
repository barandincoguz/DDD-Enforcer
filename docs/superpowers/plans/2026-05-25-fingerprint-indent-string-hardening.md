# Fingerprint Indentation + String-Parser Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `normalizePythonSemantics` as two pure phases so the save-trigger fingerprint preserves Python indentation (T10) and survives tricky string literals (T11) without false skip/validate decisions.

**Architecture:** Phase 1 `extractLogicalLines` — a string-aware char state machine that emits `{indentWidth, content}` per non-empty logical line (comments stripped, code whitespace collapsed, string literals verbatim, f-string brace-depth tracked). Phase 2 `tokenizeIndentation` — walks an indent stack emitting `INDENT`/`DEDENT` control markers + content. `normalizePythonSemantics` composes them; its `(string)→string` signature is unchanged.

**Tech Stack:** TypeScript, mocha + `@vscode/test-electron` (`npm test`), eslint, tsc.

**Spec:** `docs/superpowers/specs/2026-05-25-fingerprint-indent-string-hardening-design.md`

**Conventions:** Tests live in `extension/src/test/extension.test.ts` and import symbols from `../extension` (which re-exports module helpers). Commit trailer: `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`. RED phase in this TS setup = `npm run compile` fails on a missing export. Run all commands from `extension/`.

---

## Task 1: Phase 2 — `tokenizeIndentation` + `LogicalLine` + markers

**Files:**
- Modify: `extension/src/semanticFingerprint.ts` (add markers, `LogicalLine`, `tokenizeIndentation`)
- Modify: `extension/src/extension.ts` (re-export the new symbols)
- Test: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Write the failing tests**

Add to `extension/src/test/extension.test.ts` inside the top-level `suite(...)` (after the existing SAVE-TRIGGER block, before the API KEY block). First extend the import from `"../extension"` with `tokenizeIndentation` and `type LogicalLine`.

```ts
  // ==========================================================================
  // FINGERPRINT — PHASE 2: INDENT TOKENIZATION (T10)
  // ==========================================================================

  const IND = "\u0001"; // INDENT marker
  const DED = "\u0002"; // DEDENT marker

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
    assert.strictEqual((out.match(/\u0001/g) || []).length, 1, "one INDENT");
    assert.strictEqual((out.match(/\u0002/g) || []).length, 1, "one DEDENT");
  });

  test("tokenizeIndentation emits two DEDENT markers for a two-level pop", () => {
    const out = tokenizeIndentation([
      { indentWidth: 0, content: "a" },
      { indentWidth: 4, content: "b" },
      { indentWidth: 8, content: "c" },
      { indentWidth: 0, content: "d" },
    ]);
    assert.strictEqual((out.match(/\u0002/g) || []).length, 2, "two DEDENT");
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm run compile`
Expected: FAIL — `error TS2305: Module '"../extension"' has no exported member 'tokenizeIndentation'` (and `LogicalLine`).

- [ ] **Step 3: Implement Phase 2 in `semanticFingerprint.ts`**

Add near the top of `extension/src/semanticFingerprint.ts` (after the file header comment, before `getValidationFingerprint`):

```ts
/**
 * Fingerprint control markers. Chosen as low-codepoint control characters
 * that effectively never appear in Python source, so they cannot be confused
 * with real code or string content.
 */
const INDENT_MARK = "\u0001";
const DEDENT_MARK = "\u0002";
const LINE_SEP = "\u0003";

/** One logical (newline-joined, continuation-aware) source line. */
export interface LogicalLine {
  /** Tab-expanded leading-whitespace width measured at the logical-line start. */
  indentWidth: number;
  /** Comment-stripped, whitespace-collapsed code; string literals kept verbatim. */
  content: string;
}

/**
 * Phase 2: turn logical lines into a fingerprint string by encoding block
 * structure as INDENT/DEDENT markers relative to an indent stack. Width is
 * compared relatively, so a global reindent (e.g. 4-space → 2-space) yields
 * the same output while moving a statement across a block boundary does not.
 * Inconsistent dedents (Python IndentationError) snap to the nearest level so
 * the function stays total. Pure.
 */
export function tokenizeIndentation(lines: LogicalLine[]): string {
  const parts: string[] = [];
  const stack: number[] = [0];
  for (const line of lines) {
    const top = stack[stack.length - 1];
    if (line.indentWidth > top) {
      stack.push(line.indentWidth);
      parts.push(INDENT_MARK);
    } else if (line.indentWidth < top) {
      while (stack.length > 1 && stack[stack.length - 1] > line.indentWidth) {
        stack.pop();
        parts.push(DEDENT_MARK);
      }
    }
    parts.push(line.content);
    parts.push(LINE_SEP);
  }
  return parts.join("");
}
```

Add the re-export in `extension/src/extension.ts`, in the existing `export { ... } from "./semanticFingerprint";` block (add only `tokenizeIndentation` and `type LogicalLine` now; Task 2 adds `extractLogicalLines`):

```ts
export {
  normalizePythonSemantics,
  getValidationFingerprint,
  classifySaveForValidation,
  classifySaveForValidationFromContent,
  tokenizeIndentation,
  type LogicalLine,
} from "./semanticFingerprint";
```

- [ ] **Step 4: Run to verify it passes**

Run: `npm run compile && npm run lint && npm test 2>&1 | grep -E "tokenizeIndentation|passing|failing"`
Expected: the 4 `tokenizeIndentation` tests show `✔`; suite reports `132 passing` (128 prior + 4); exit 0.

- [ ] **Step 5: Commit**

```bash
git add extension/src/semanticFingerprint.ts extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "feat(fingerprint): add indent-depth tokenizer phase (T10)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Phase 1 — `extractLogicalLines` (string-aware extractor)

**Files:**
- Modify: `extension/src/semanticFingerprint.ts` (add `extractLogicalLines`)
- Modify: `extension/src/extension.ts` (add `extractLogicalLines` to the re-export block)
- Test: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Write the failing tests**

Extend the `"../extension"` import with `extractLogicalLines`. Add this block after the Phase 2 tests:

```ts
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
    // Triple-double string whose content contains an escaped double-quote.
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
    // "if" must NOT be detected as an f-string prefix; the string still closes.
    assert.strictEqual(lines[1].content, "pass");
  });
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm run compile`
Expected: FAIL — `error TS2305: Module '"../extension"' has no exported member 'extractLogicalLines'`.

- [ ] **Step 3: Implement Phase 1 in `semanticFingerprint.ts`**

Add this function (and the small `isWordChar` helper) to `extension/src/semanticFingerprint.ts`, above `normalizePythonSemantics`:

```ts
/** True for Python identifier characters (used by the whitespace-collapse rule). */
function isWordChar(ch: string): boolean {
  return ch !== "" && /[A-Za-z0-9_]/.test(ch);
}

type StringMode = "outside" | "s" | "d" | "ts" | "td";

/**
 * Phase 1: split Python source into logical lines, robust to strings,
 * comments, line continuations, and Python 3.12 f-strings.
 *
 * - Comments outside strings are stripped; '#' inside strings is preserved.
 * - Code whitespace between two word-chars collapses to one space; otherwise
 *   it is removed (so `a = 1` → `a=1` but `del x` stays `del x`).
 * - String-literal content is preserved verbatim.
 * - Escapes are tracked in every string mode (including triple); f-strings
 *   track brace depth so a reused quote inside `{...}` does not close early.
 * - Indentation is tab-expanded (tabstop 8) and measured at the logical-line
 *   start. Blank and comment-only lines are dropped.
 *
 * Pure: no I/O.
 */
export function extractLogicalLines(src: string): LogicalLine[] {
  const lines: LogicalLine[] = [];
  const n = src.length;
  let i = 0;

  let content = "";
  let indentWidth = 0;
  let measuringIndent = true;
  let pendingWs = false;
  let hasContent = false;

  let mode: StringMode = "outside";
  let escaped = false;
  let isFString = false;
  let braceDepth = 0;

  const finishLine = () => {
    if (hasContent && content.length > 0) {
      lines.push({ indentWidth, content });
    }
    content = "";
    indentWidth = 0;
    measuringIndent = true;
    pendingWs = false;
    hasContent = false;
  };

  while (i < n) {
    const ch = src[i];

    if (mode === "outside") {
      if (measuringIndent) {
        if (ch === " ") {
          indentWidth += 1;
          i++;
          continue;
        }
        if (ch === "\t") {
          indentWidth += 8 - (indentWidth % 8);
          i++;
          continue;
        }
        if (ch === "\r") {
          i++;
          continue;
        }
        if (ch === "\n") {
          finishLine();
          i++;
          continue;
        }
        if (ch === "#") {
          while (i < n && src[i] !== "\n") {
            i++;
          }
          continue;
        }
        measuringIndent = false;
        // fall through to content handling for this same char (no i++)
      }

      // line continuation: backslash immediately before a newline
      if (
        ch === "\\" &&
        (src[i + 1] === "\n" || (src[i + 1] === "\r" && src[i + 2] === "\n"))
      ) {
        i += src[i + 1] === "\r" ? 3 : 2;
        continue;
      }
      if (ch === "\r") {
        i++;
        continue;
      }
      if (ch === "\n") {
        finishLine();
        i++;
        continue;
      }
      if (ch === "#") {
        while (i < n && src[i] !== "\n") {
          i++;
        }
        continue;
      }
      if (ch === " " || ch === "\t") {
        pendingWs = true;
        i++;
        continue;
      }

      if (ch === "'" || ch === '"') {
        // A quote is not a word-char, so a pending space is dropped here.
        pendingWs = false;
        const prefix = /[A-Za-z]*$/.exec(content)![0];
        isFString =
          prefix.length > 0 &&
          prefix.length <= 2 &&
          /^[rbufRBUF]+$/.test(prefix) &&
          /[fF]/.test(prefix);
        braceDepth = 0;
        escaped = false;
        const triple = src.substr(i, 3);
        if (ch === "'" && triple === "'''") {
          content += "'''";
          mode = "ts";
          hasContent = true;
          i += 3;
          continue;
        }
        if (ch === '"' && triple === '"""') {
          content += '"""';
          mode = "td";
          hasContent = true;
          i += 3;
          continue;
        }
        content += ch;
        mode = ch === "'" ? "s" : "d";
        hasContent = true;
        i++;
        continue;
      }

      // ordinary code char: apply the whitespace-collapse rule
      if (pendingWs) {
        const last = content.length > 0 ? content[content.length - 1] : "";
        if (isWordChar(last) && isWordChar(ch)) {
          content += " ";
        }
        pendingWs = false;
      }
      content += ch;
      hasContent = true;
      i++;
      continue;
    }

    // ===== inside a string =====
    const isTriple = mode === "ts" || mode === "td";
    const delim = mode === "s" ? "'" : mode === "d" ? '"' : mode === "ts" ? "'''" : '"""';

    if (isFString && !escaped) {
      if (ch === "{" && src[i + 1] === "{") {
        content += "{{";
        i += 2;
        continue;
      }
      if (ch === "}" && src[i + 1] === "}") {
        content += "}}";
        i += 2;
        continue;
      }
      if (ch === "{") {
        braceDepth += 1;
        content += ch;
        i++;
        continue;
      }
      if (ch === "}" && braceDepth > 0) {
        braceDepth -= 1;
        content += ch;
        i++;
        continue;
      }
    }

    if (escaped) {
      content += ch;
      escaped = false;
      i++;
      continue;
    }
    if (ch === "\\") {
      content += ch;
      escaped = true;
      i++;
      continue;
    }

    // close the string only when not inside an f-string expression
    if (braceDepth === 0) {
      if (isTriple) {
        if (src.substr(i, 3) === delim) {
          content += delim;
          mode = "outside";
          isFString = false;
          i += 3;
          continue;
        }
      } else if (ch === delim) {
        content += ch;
        mode = "outside";
        isFString = false;
        i++;
        continue;
      }
    }

    content += ch;
    i++;
  }

  finishLine();
  return lines;
}
```

Add `extractLogicalLines` to the `export { ... } from "./semanticFingerprint";` block in `extension/src/extension.ts` (alongside `tokenizeIndentation`, `type LogicalLine` added in Task 1).

- [ ] **Step 4: Run to verify it passes**

Run: `npm run compile && npm run lint && npm test 2>&1 | grep -E "extractLogicalLines|passing|failing"`
Expected: the 9 `extractLogicalLines` tests show `✔`; suite reports `141 passing` (132 + 9); exit 0.

- [ ] **Step 5: Commit**

```bash
git add extension/src/semanticFingerprint.ts extension/src/extension.ts extension/src/test/extension.test.ts
git commit -m "feat(fingerprint): add string-aware logical-line extractor (T11)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Compose the phases + indentation-aware integration tests

**Files:**
- Modify: `extension/src/semanticFingerprint.ts` (rewrite `normalizePythonSemantics` body)
- Test: `extension/src/test/extension.test.ts`

- [ ] **Step 1: Write the failing tests**

Add after the existing SAVE-TRIGGER block (these exercise the composed function via the public API):

```ts
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `npm run compile && npm test 2>&1 | grep -E "Dedenting|reindent|state leak|failing"`
Expected: FAIL — `Dedenting a statement out of a block triggers validation (T10)` fails (current `normalizePythonSemantics` strips indentation, so `before`/`after` collide and `shouldValidate` is `false`).

- [ ] **Step 3: Rewrite `normalizePythonSemantics` to compose the phases**

Replace the entire body of `normalizePythonSemantics` in `extension/src/semanticFingerprint.ts` (keep the signature and the doc comment intact, update the comment to describe the two-phase approach):

```ts
/**
 * Build a stable semantic fingerprint of Python source. Two phases:
 * (1) extract logical lines (string/comment/continuation aware), then
 * (2) encode block structure as INDENT/DEDENT markers. Two sources produce
 * the same fingerprint iff they have the same statements AND the same block
 * structure — so comment/blank/spacing/reindent edits are non-semantic while
 * moving a statement across a block boundary is semantic.
 */
export function normalizePythonSemantics(content: string): string {
  return tokenizeIndentation(extractLogicalLines(content));
}
```

Delete the old character-stripping loop body entirely (the previous implementation that walked `inSingle`/`inDouble`/`inTripleSingle`/`inTripleDouble` and stripped all whitespace). Ensure no now-unused locals remain (lint will flag them).

- [ ] **Step 4: Run to verify it passes**

Run: `npm run compile && npm run lint && npm test 2>&1 | grep -E "passing|failing"`
Expected: `145 passing` (141 + 4); `0 failing`; exit 0. In particular the four original SAVE-TRIGGER tests (no-op / blank-line / comment-only skip, semantic-change validate) remain green.

- [ ] **Step 5: Commit**

```bash
git add extension/src/semanticFingerprint.ts extension/src/test/extension.test.ts
git commit -m "feat(fingerprint): compose two-phase normalizer, preserve indentation (T10+T11)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Final verification + spec/doc cleanup

**Files:**
- Modify: `docs/superpowers/specs/2026-05-25-fingerprint-indent-string-hardening-design.md` (flip Status to Implemented)

- [ ] **Step 1: Full gate**

Run: `npm run compile && npm run lint && npm test 2>&1 | tail -4`
Expected: compile 0, lint 0, `145 passing`, exit 0.

- [ ] **Step 2: Confirm scope is clean**

Run: `git diff --name-only b934033..HEAD -- extension/src | sort -u`
Expected: only `extension/src/extension.ts`, `extension/src/semanticFingerprint.ts`, `extension/src/test/extension.test.ts` (plus the unrelated Task-1..3 files already listed). No file outside `semanticFingerprint.ts`/`extension.ts`/test touched by this plan.

- [ ] **Step 3: Mark the spec Implemented**

In `docs/superpowers/specs/2026-05-25-fingerprint-indent-string-hardening-design.md`, change `**Status:** Approved design — pending spec review → writing-plans.` to `**Status:** Implemented (see plan 2026-05-25-fingerprint-indent-string-hardening.md).`

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-05-25-fingerprint-indent-string-hardening-design.md
git commit -m "docs(spec): mark fingerprint hardening (T10+T11) implemented

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the executor

- **Test count math** assumes the suite is at `128 passing` before Task 1 (verified at `e2532b3`). If the baseline differs, keep the deltas (+4, +9, +4) and adjust the absolute numbers.
- **RED in TypeScript:** a test that imports a not-yet-exported symbol fails at `npm run compile`, not at test runtime — that is the expected RED signal here.
- **Do not** touch `extension/.vscode-test.mjs` or any `*.json` runtime artifact; they are intentionally dirty in the working tree.
- **Marker codepoints** must stay exactly `\u0001` (INDENT), `\u0002` (DEDENT), `\u0003` (LINE_SEP) — the Task-1 tests hard-code `\u0001`/`\u0002`.
