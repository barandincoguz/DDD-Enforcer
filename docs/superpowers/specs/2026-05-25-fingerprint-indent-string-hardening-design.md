# Design: Fingerprint Indentation + String-Parser Hardening (T10 + T11)

**Date:** 2026-05-25
**Topic:** Rewrite `extension/src/semanticFingerprint.ts` `normalizePythonSemantics` so the save-trigger validation fingerprint (T10) preserves Python indentation semantics and (T11) hardens the hand-rolled string-state parser.
**Status:** Approved design — pending spec review → writing-plans.
**Approach:** Two-phase (string-aware logical-line extractor → indent-depth tokenizer). Chosen over single-pass-interleaved and minimal-patch for testability and clean boundaries.

---

## 1. Motivation

The validation fingerprint decides, on each save, whether a Python file changed *semantically* (→ run validation) or only cosmetically (→ skip). Two latent defects in the current `normalizePythonSemantics`:

- **T10 — indentation is erased.** It strips ALL whitespace outside strings, including leading indentation. In Python, indentation IS block structure. Moving a statement into/out of a block produces an identical fingerprint → the change is classified non-semantic → **validation is silently skipped on a genuinely different program.** Example:
  ```python
  if x:          if x:
      a()            a()
  b()                b()   # dedented out of the if-block
  ```
  Both currently normalize to `ifx:a()b()`.

- **T11 — the string-state parser has holes.** No escape tracking inside triple-quoted strings; a quote adjacent to a closing triple-delimiter closes early; Python 3.12 f-strings that reuse the outer quote inside `{…}` (`f"...{x.split(" ")}..."`) close early. Any mis-detected boundary inverts in/out-of-string state for the **rest of the file**, corrupting its fingerprint (can cause false-equal → skip, the dangerous direction). Python 3.12 is the locked D1 version, so 3.12 f-strings will appear in the paper's test subjects.

T10 and T11 are inseparable: correct indentation requires correct string state, because newlines inside a multi-line string are NOT logical-line breaks.

**Failure-mode bias:** when uncertain, treat content as DIFFERENT (validate). False-skip (the T10 bug) is harmful; false-validate is only a cheap extra validation run.

---

## 2. Architecture — two pure phases

Public facade unchanged: `getValidationFingerprint(content)`, `classifySaveForValidation*`, and `normalizePythonSemantics(content: string): string` keep their signatures. `normalizePythonSemantics` now delegates:

```
normalizePythonSemantics(src)
  → lines = extractLogicalLines(src)        // Phase 1
  → return tokenizeIndentation(lines)       // Phase 2
```

Both internal helpers are **exported** so they can be unit-tested directly via the `../extension` re-export (repo convention). File stays `semanticFingerprint.ts` (no new file); est. ~250–300 lines, within the AGENTS.md sweet spot.

### 2.1 Phase 1 — `extractLogicalLines(src: string): LogicalLine[]`

```ts
interface LogicalLine {
  indentWidth: number; // expanded leading-whitespace width at the logical-line start
  content: string;     // comment-stripped, ws-collapsed code; string literals verbatim
}
```

A single-pass character state machine. **String state:** `outside | single (') | double (") | tripleSingle (''') | tripleDouble (""")`. Carries an `escaped` flag **in every string mode including triple** (fixes T11 hole 1). Carries `fstringBraceDepth` when the active string was opened with an `f`/`F` prefix.

Rules:
- **Triple-quote close** consumes the full 3-char delimiter; a quote adjacent to the delimiter is handled by the `escaped` flag and greedy 3-char match so it does not close early (fixes T11 hole 2).
- **f-string prefix detection:** when opening a string, inspect the immediately preceding run of identifier letters; if it contains `f`/`F` (e.g. `f`, `rf`, `fR`, `bf` is invalid in Python so ignore `b`+`f` combos but harmless), mark the string f-string. Inside an f-string, a `{` that is not `{{` increments `fstringBraceDepth`; `}` (not `}}`) decrements it. While `fstringBraceDepth > 0`, the f-string's quote char does NOT terminate the string (fixes T11 hole 3 / 3.12 quote-reuse). `{{` and `}}` are literal-brace escapes and do not change depth.
- **Comments:** a `#` encountered *outside* any string starts a comment; consume to end-of-physical-line, emit nothing. A `#` inside a string is preserved as content.
- **Line continuation:** a backslash at the physical end-of-line *outside* a string joins the next physical line into the same logical line (no break, no re-measure of indent).
- **Newline (outside string):** ends the current logical line. Inside any string, a newline is part of the string content (NOT a logical-line break).
- **Indentation:** measured only at the START of a logical line (always outside a string). Leading whitespace is tab-expanded to a fixed tabstop of 8, then counted → `indentWidth`.
- **Whitespace collapse (code, outside strings):** a run of whitespace between two word-chars (`[A-Za-z0-9_]`) collapses to a single space; otherwise it is removed. So `a = 1` → `a=1` (spacing around operators is non-semantic) while `del x` → `del x` (token boundary preserved; prevents the `delx` token-merge collision latent in the current implementation).
- **String-literal content** is preserved verbatim (its internal whitespace and newlines are part of `content`).
- **Empty logical lines** (blank, or comment-only after stripping → empty `content`) are DROPPED (not emitted). This preserves the existing "blank-line deletion" and "comment-only change" skip behavior.

### 2.2 Phase 2 — `tokenizeIndentation(lines: LogicalLine[]): string`

Walk the non-empty logical lines with an indent stack (starts `[0]`):
- `indentWidth > top` → push `indentWidth`, emit `INDENT` marker.
- `indentWidth == top` → emit nothing (same block).
- `indentWidth < top` → pop while `top > indentWidth`, emit one `DEDENT` marker per pop. After popping, if `top != indentWidth` (inconsistent dedent — a Python `IndentationError`), snap: continue deterministically without error (fingerprint must be total, not validating syntax).

After the indent markers, append the line's `content`. Separate logical lines with a `LINE_SEP` marker. Markers are control chars unlikely to appear in source: `INDENT = \u0001`, `DEDENT = \u0002`, `LINE_SEP = \u0003`.

**Consequence (desired):** a global reindent (e.g. 4-space → 2-space) changes every `indentWidth` but preserves the relative push/pop pattern → identical token stream → skip. Moving a statement across a block boundary changes the pattern → different stream → validate.

---

## 3. Backward compatibility

- `normalizePythonSemantics(string): string` signature and the `classifySaveForValidation*` API are unchanged. The fingerprint string format changes, but it is internal (never persisted across versions; computed fresh each session) so there is no migration concern.
- All four existing save-trigger tests must still pass:
  - "No-op save should skip" — identical input → identical stream. ✅
  - "Blank-line deletion should skip" — blank lines dropped in Phase 1. ✅
  - "Comment-only change should skip" — comment-only lines drop to empty → dropped. ✅
  - "Semantic code change should trigger validation" — `return True` vs `return False` → `return True` vs `return False` content differs. ✅

---

## 4. Test plan (TDD — write first)

**Phase 2 `tokenizeIndentation` (pure, direct):**
1. Flat lines, same indent → no INDENT/DEDENT markers.
2. One nested block → exactly one INDENT then one DEDENT.
3. Multi-level dedent (pop 2 levels at once) → two DEDENT markers.
4. Global reindent (widths 4,8 vs 2,4 with same structure) → identical output.

**Phase 1 `extractLogicalLines` (pure, direct):**
5. Comment outside string stripped; `#` inside a string preserved.
6. Triple-string containing an escaped delimiter (`'''a\'''…'''`) does not close early.
7. Triple-string with a quote adjacent to the close (`"""he said """"`) handled without state inversion.
8. Python 3.12 f-string quote-reuse `f"a {b.split(" ")} c"` does not close early.
9. Line continuation `\` joins two physical lines into one logical line.
10. Blank and comment-only lines dropped.
11. Whitespace collapse: `a = 1` → `a=1`; `del x` → `del x` (boundary kept).
12. String-literal internal whitespace preserved verbatim.

**Integration `classifySaveForValidationFromContent` (the T10 core):**
13. **T10 regression:** `if x:\n    a()\n    b()\n` vs `if x:\n    a()\nb()\n` → `shouldValidate === true` (statement dedented out of the block — the headline bug).
14. Reindent-only (4-space vs 2-space, same structure) → `shouldValidate === false`.
15. T11 robustness: a one-character edit to code *after* a tricky docstring (escaped/adjacent/f-string) is correctly detected as semantic, proving no state corruption leaked past the string.
16. The four existing tests remain green.

**Acceptance:** `cd extension && npm run compile && npm run lint && npm test` all green; new tests pass; `normalizePythonSemantics` signature unchanged; no files touched outside `semanticFingerprint.ts` + `test/extension.test.ts`.

---

## 5. Out of scope (YAGNI)

- A full Python tokenizer / real INDENT/DEDENT grammar with `NEWLINE`/`ENDMARKER`.
- f-string format-spec (`:…`) and conversion (`!r`) parsing — brace-depth tracking is sufficient to prevent quote-reuse state inversion.
- Perfect raw-string (`r"..."`) backslash semantics — treating `\` as a potential escape inside raw strings is harmless for fingerprint equality.
- Detecting/erroring on invalid Python — the fingerprint must be a total function over any input.

---

## 6. Risks

- **Marker collision:** control chars `\u0001`–`\u0003` could in theory appear inside a preserved string literal, blurring a boundary. Effect is bounded to false-equal only in pathological crafted input; acceptable under the failure-mode bias and negligible for real Python.
- **f-string prefix mis-detection** on exotic prefixes — bounded, localized, bias-safe (errs toward validate).
