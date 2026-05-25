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
