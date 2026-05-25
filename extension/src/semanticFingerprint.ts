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

/**
 * Lightweight Python semantic normalization:
 * - removes comments outside strings
 * - removes whitespace outside strings
 * - preserves string literal content
 */
export function normalizePythonSemantics(content: string): string {
  let result = "";
  let i = 0;
  let inSingle = false;
  let inDouble = false;
  let inTripleSingle = false;
  let inTripleDouble = false;
  let escaped = false;

  const isWhitespace = (ch: string) => /\s/.test(ch);

  while (i < content.length) {
    const ch = content[i];
    const next3 = content.slice(i, i + 3);

    if (inTripleSingle) {
      result += ch;
      if (next3 === "'''") {
        result += "''";
        i += 3;
        inTripleSingle = false;
        continue;
      }
      i += 1;
      continue;
    }

    if (inTripleDouble) {
      result += ch;
      if (next3 === '"""') {
        result += '""';
        i += 3;
        inTripleDouble = false;
        continue;
      }
      i += 1;
      continue;
    }

    if (inSingle) {
      result += ch;
      if (!escaped && ch === "'") {
        inSingle = false;
      }
      escaped = !escaped && ch === "\\";
      i += 1;
      continue;
    }

    if (inDouble) {
      result += ch;
      if (!escaped && ch === '"') {
        inDouble = false;
      }
      escaped = !escaped && ch === "\\";
      i += 1;
      continue;
    }

    // Outside string literals
    if (next3 === "'''") {
      inTripleSingle = true;
      result += "'''";
      i += 3;
      continue;
    }

    if (next3 === '"""') {
      inTripleDouble = true;
      result += '"""';
      i += 3;
      continue;
    }

    if (ch === "'") {
      inSingle = true;
      escaped = false;
      result += ch;
      i += 1;
      continue;
    }

    if (ch === '"') {
      inDouble = true;
      escaped = false;
      result += ch;
      i += 1;
      continue;
    }

    // Strip comments outside strings
    if (ch === "#") {
      while (i < content.length && content[i] !== "\n") {
        i += 1;
      }
      continue;
    }

    // Strip whitespace outside strings
    if (isWhitespace(ch)) {
      i += 1;
      continue;
    }

    result += ch;
    i += 1;
  }

  return result;
}
