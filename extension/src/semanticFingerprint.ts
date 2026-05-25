/**
 * DDD Enforcer VS Code Extension
 * Semantic Fingerprinting & Change Classification
 *
 * Lightweight Python semantic normalization:
 * - removes comments outside strings
 * - removes whitespace outside strings
 * - preserves string literal content
 */

/**
 * Fingerprint control markers. Chosen as low-codepoint control characters
 * that effectively never appear in Python source, so they cannot be confused
 * with real code or string content.
 */
const INDENT_MARK = "\x02";
const DEDENT_MARK = "\x03";
const LINE_SEP = "\x1f";

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
 * compared relatively, so a global reindent (e.g. 4-space to 2-space) yields
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

/**
 * Creates a stable semantic fingerprint for validation decisions.
 * Ignores comments and whitespace outside string literals.
 */
export function getValidationFingerprint(content: string): string {
  return normalizePythonSemantics(content);
}

/**
 * Compare two file fingerprints and return validate/skip decision.
 * Pure function: no VS Code dependencies.
 */
export function classifySaveForValidation(
  previousFingerprint: string | undefined,
  currentContent: string,
): { shouldValidate: boolean; reason: string } {
  const curr = getValidationFingerprint(currentContent);

  if (previousFingerprint === undefined) {
    return { shouldValidate: true, reason: "first semantic snapshot" };
  }
  if (previousFingerprint === curr) {
    return { shouldValidate: false, reason: "non-semantic change" };
  }
  return { shouldValidate: true, reason: "semantic code change" };
}

/**
 * Test helper: compare raw text snapshots directly.
 * Pure function: no VS Code dependencies.
 */
export function classifySaveForValidationFromContent(
  previousContent: string | undefined,
  currentContent: string,
): { shouldValidate: boolean; reason: string } {
  const prev = previousContent
    ? getValidationFingerprint(previousContent)
    : undefined;
  return classifySaveForValidation(prev, currentContent);
}

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
 *   it is removed (so `a = 1` becomes `a=1` but `del x` stays `del x`).
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
        pendingWs = false;
        const prefix = /[A-Za-z]*$/.exec(content)![0];
        isFString =
          prefix.length > 0 &&
          prefix.length <= 2 &&
          /^[rbufRBUF]+$/.test(prefix) &&
          /[fF]/.test(prefix);
        braceDepth = 0;
        escaped = false;
        const triple = src.slice(i, i + 3);
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

    if (braceDepth === 0) {
      if (isTriple) {
        if (src.slice(i, i + 3) === delim) {
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
