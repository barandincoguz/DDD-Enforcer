# WP-CORE-2 Reference-Truncation Correctness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `_truncate_at_references` to correctly handle Turkish `Kaynaklar` (locale parity), trailing-colon heading variants (`References:`), and to suppress mid-document false-positive truncation on subsection names like `3.4 References`, without regressing legitimate bibliography subsections like `5.1 References` at document end.

**Architecture:** Three coordinated edits in a single production file: (1) new class constant `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`, (2) regex pattern adds `kaynaklar` alternation and optional trailing colon `[:：]`, (3) `_truncate_at_references` skips lines before `int(len(lines) * 0.5)` so the regex can only fire in the latter half of an SRS. Tests use in-memory `.txt` files via `tmp_path` plus a `pytest.mark.parametrize` direct grammar audit on the regex object.

**Tech Stack:** Python 3.13 (local dev), Python 3.12 (CI), `pytest` `-m "not integration"`, `re` module with `re.IGNORECASE`, `pathlib.Path`, `tmp_path` fixture.

**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-2-reference-truncate-fix-design.md` (v2 — Codex-reviewed)
**Audit finding:** `.planning/pipeline_audit/findings/document_parser.md` F-5 (MAJOR)
**Pre-WP HEAD:** `029f187`
**Pre-WP pytest baseline:** 272 passed, 31 deselected

---

## File Structure

| file | role | change type |
|---|---|---|
| `extension/backend/core/document_parser.py` | SRS parser entry point + post-processing pipeline. Owns `_truncate_at_references` and the heading regex. | MODIFY (~11 LOC production) |
| `extension/backend/tests/test_document_parser.py` | Unit tests for `SRSDocumentParser`. | MODIFY (append T1–T7 + T-regex parametrize, ~170 LOC) |
| `development_docs/WP-CORE-2-reference-truncate-fix.md` | Persistent dev memory for this WP. | CREATE |
| `development_docs/INDEX.md` | Dev-doc status board. | MODIFY (new ACTIVE row) |
| `.planning/pipeline_audit/improvements_backlog.md` | Audit backlog. | MODIFY (F-5 OPEN → SHIPPED) |
| `.planning/pipeline_audit/CURRENT.md` | Audit pointer. | MODIFY (last action + next) |
| `.planning/pipeline_audit/decision_log.md` | Audit decisions. | MODIFY (append SHIPPED outcome) |

No new files in `extension/backend/`. No file renames. No new dependencies.

---

## Task 1: RED — Add failing + regression-positive tests

**Goal:** Capture the locale gap (T2), false-positive (T4), and trailing-colon gap (T7) as red tests. Add positive regression coverage (T1, T3, T5, T6) and a direct grammar audit (T-regex). The current production code is unchanged, so T2 / T4 / T7 / certain T-regex parametrize cases must FAIL; the rest PASS.

**Files:**
- Modify: `extension/backend/tests/test_document_parser.py` (append at end; preserve existing tests untouched)

**Pre-step — confirm baseline:**

- [ ] **Step 0: Confirm pre-task pytest baseline = 272**

Run from repo root:

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend"
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=no -q 2>&1 | tail -3
```

Expected output ends with: `272 passed, 31 deselected in <N>s`. If the number differs, STOP — escalate "baseline already broken" to the loop coordinator.

- [ ] **Step 1: Open `extension/backend/tests/test_document_parser.py` for append**

Confirm the current file ends at line 151 (`parser.parse_file("/nonexistent/file.txt")`). Append the new tests after that line.

- [ ] **Step 2: Append T1 (Turkish singular `Kaynakça`) — positive truncation**

Append this block at the end of `extension/backend/tests/test_document_parser.py`:

```python


def test_parse_txt_truncates_at_kaynakca_heading(tmp_path):
    txt_file = tmp_path / "tr-references.txt"
    # 5 lines: bibliography at index 3 (>= midpoint 2). Should truncate.
    txt_file.write_text(
        "Müşteri siparişleri kayıt altına alınmalıdır.\n"
        "Sipariş iptalleri kayıt altına alınmalıdır.\n"
        "Sistem ödeme tahsil etmelidir.\n"
        "Kaynakça\n"
        "1. Smith, J. (2020). DDD in practice.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "Müşteri siparişleri" in content
    assert "Sistem ödeme tahsil etmelidir." in content
    assert "Smith, J." not in content
```

- [ ] **Step 3: Append T2 (Turkish plural `Kaynaklar`) — positive (F-5 core fix)**

Append at end of file:

```python


def test_parse_txt_truncates_at_kaynaklar_heading(tmp_path):
    txt_file = tmp_path / "tr-references-plural.txt"
    # 5 lines: bibliography at index 3 (>= midpoint 2). Should truncate.
    txt_file.write_text(
        "Sistem ödeme kaydını tutmalıdır.\n"
        "Müşteri profili oluşturulmalıdır.\n"
        "Sepet özeti hesaplanmalıdır.\n"
        "Kaynaklar\n"
        "1. Evans, E. (2003). Domain-Driven Design.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "Sistem ödeme" in content
    assert "Sepet özeti" in content
    assert "Evans, E." not in content
```

- [ ] **Step 4: Append T3 (numbered top-level `5. References`) — positive regression**

Append at end of file:

```python


def test_parse_txt_truncates_at_numbered_top_level_references(tmp_path):
    txt_file = tmp_path / "numbered-references.txt"
    # 5 lines: bibliography at index 3 (>= midpoint 2). Should truncate.
    txt_file.write_text(
        "The system shall log all transactions.\n"
        "The system shall enforce idempotent submission.\n"
        "The system shall reject malformed payloads.\n"
        "5. References\n"
        "[1] IEEE 830-1998. Recommended Practice.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "log all transactions" in content
    assert "idempotent submission" in content
    assert "IEEE 830-1998" not in content
```

- [ ] **Step 5: Append T4 (`3.4 References` mid-doc) — negative (false-positive guard)**

Append at end of file:

```python


def test_parse_txt_does_not_truncate_at_nested_subsection_references_in_first_half(tmp_path):
    txt_file = tmp_path / "subsection-references.txt"
    # 8 lines: `3.4 References` at index 1 (< midpoint 4). Should NOT truncate.
    txt_file.write_text(
        "Section 3 — Functional Requirements.\n"
        "3.4 References\n"
        "The system shall accept ISO 8601 timestamps.\n"
        "The system shall reject unauthenticated requests.\n"
        "4. Non-functional Requirements\n"
        "The system shall respond within 200 ms.\n"
        "6. Glossary\n"
        "Term: aggregate root.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "3.4 References" in content
    assert "ISO 8601 timestamps" in content
    assert "respond within 200 ms" in content
    assert "Term: aggregate root." in content
```

- [ ] **Step 6: Append T5 (inline `Kaynaklarımız`) — negative (locale safety)**

Append at end of file:

```python


def test_parse_txt_does_not_truncate_inline_kaynaklar_mention(tmp_path):
    txt_file = tmp_path / "inline-kaynaklar.txt"
    # 4 lines, inline use of "Kaynaklarımız" — never matches the regex.
    txt_file.write_text(
        "Sistem güvenliği önemlidir.\n"
        "Kaynaklarımız geniştir ve sistem buna göre tasarlanmalıdır.\n"
        "Veritabanı yedeklemeleri saatlik alınır.\n"
        "Sipariş yönetimi modülü güncellenmelidir.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "Kaynaklarımız geniştir" in content
    assert "Sipariş yönetimi" in content
```

- [ ] **Step 7: Append T6 (nested `5.1 References` at end) — positive (Codex WARN-1 guard)**

Append at end of file:

```python


def test_parse_txt_truncates_at_nested_5_1_references_at_end(tmp_path):
    txt_file = tmp_path / "appendix-references.txt"
    # 6 lines: `5.1 References` at index 4 (>= midpoint 3). Should truncate.
    # Verifies that the position guard does NOT block legitimate bibliography
    # subsections like `5.1 References` (appendix-style layout).
    txt_file.write_text(
        "The system shall persist orders.\n"
        "The system shall enforce price floors.\n"
        "5. Appendix\n"
        "Appendix A — error codes.\n"
        "5.1 References\n"
        "[1] Fowler, M. (2002). Enterprise Patterns.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "persist orders" in content
    assert "Appendix A" in content
    assert "Fowler, M." not in content
```

- [ ] **Step 8: Append T7 (`References:` trailing colon) — positive (Codex WARN-2a)**

Append at end of file:

```python


def test_parse_txt_truncates_at_references_with_trailing_colon(tmp_path):
    txt_file = tmp_path / "colon-references.txt"
    # 5 lines: `References:` at index 3 (>= midpoint 2). Should truncate.
    txt_file.write_text(
        "The system shall validate input.\n"
        "The system shall log audit trails.\n"
        "The system shall enforce role-based access.\n"
        "References:\n"
        "[1] OWASP ASVS 4.0.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))

    assert "validate input" in content
    assert "role-based access" in content
    assert "OWASP" not in content
```

- [ ] **Step 9: Append T-regex parametrized direct grammar audit (Codex WARN-5)**

Append at end of file:

```python


@pytest.mark.parametrize(
    "line,should_match",
    [
        # positive matches
        ("References", True),
        ("REFERENCES", True),
        ("Bibliography", True),
        ("BIBLIOGRAPHY", True),
        ("Kaynakça", True),
        ("KAYNAKÇA", True),
        ("Kaynaklar", True),
        ("KAYNAKLAR", True),
        ("# References", True),
        ("## Bibliography", True),
        ("5. References", True),
        ("5.1 References", True),
        ("5.1 Kaynakça", True),
        ("5.1.2 Kaynaklar", True),
        ("References:", True),
        ("Kaynaklar:", True),
        ("Kaynaklar：", True),  # U+FF1A fullwidth colon
        ("references", True),
        # negative — must NOT match
        ("References to external systems shall be preserved.", False),
        ("Kaynaklarımız geniştir.", False),
        ("Some References", False),
        ("3.4 References to Other Documents", False),
        ("3.4 Subsection References", False),
        ("Bibliography of authors:", False),
        ("", False),
        ("  ", False),
    ],
)
def test_reference_heading_pattern_direct_grammar(line, should_match):
    parser = SRSDocumentParser()
    match = parser.reference_heading_pattern.match(line.strip())
    assert bool(match) is should_match, f"Pattern mismatch for: {line!r}"
```

Note on the fullwidth colon: the entry uses Python's `：` escape so the test source is byte-portable across editors. Do NOT replace it with a literal fullwidth-colon glyph — the escape form is intentionally explicit and survives any encoding flattening.

- [ ] **Step 10: Run tests, confirm RED phase failures match expectations**

Run from repo root:

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend"
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" tests/test_document_parser.py -v 2>&1 | tail -40
```

Expected RED-phase outcomes (only `test_document_parser.py`):

| test | expected | reason |
|---|---|---|
| `test_parse_pdf_merges_wrapped_lines_and_stops_at_references` | PASS | existing test, English standalone `References` still matches |
| `test_parse_docx_preserves_lists_and_tables` | PASS | unaffected |
| `test_parse_txt_supports_utf16_input` | PASS | unaffected |
| `test_parse_txt_does_not_truncate_regular_requirement_lines` | PASS | existing negative test |
| `test_parse_nonexistent_file_raises_file_not_found` | PASS | unaffected |
| `test_parse_txt_truncates_at_kaynakca_heading` (T1) | PASS | `kaynakça` already in regex |
| `test_parse_txt_truncates_at_kaynaklar_heading` (T2) | **FAIL** | `kaynaklar` NOT in regex → no truncation → `Evans, E.` still in content |
| `test_parse_txt_truncates_at_numbered_top_level_references` (T3) | PASS | `5. References` matches existing regex |
| `test_parse_txt_does_not_truncate_at_nested_subsection_references_in_first_half` (T4) | **FAIL** | existing regex truncates `3.4 References` → assertions on later content fail |
| `test_parse_txt_does_not_truncate_inline_kaynaklar_mention` (T5) | PASS | inline mention never matches |
| `test_parse_txt_truncates_at_nested_5_1_references_at_end` (T6) | PASS | existing regex matches `5.1 References` (we are pre-edit) |
| `test_parse_txt_truncates_at_references_with_trailing_colon` (T7) | **FAIL** | `References:` does not match (no optional `:` in regex) → no truncation |
| `test_reference_heading_pattern_direct_grammar` parametrize | mixed | the `Kaynaklar`, `KAYNAKLAR`, `Kaynaklar:`, `Kaynaklar:`(fullwidth) cases FAIL; the rest pass; total expected `≥4` FAILs |

Total expected RED failures: **3 named tests + several T-regex parametrize FAILs** (each `Kaynaklar*` case + `References:` case).

If unexpected tests fail beyond this list, STOP and investigate — do not proceed to GREEN.

- [ ] **Step 11: Confirm wider repo baseline did not drop due to the new test file**

Run the full unit suite (still pre-edit):

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=no -q 2>&1 | tail -3
```

Expected: count is **272 PASS + expected RED failures + parametrize expansion** ≈ `XXX passed, ≥4 failed, 31 deselected`. The exact RED count depends on parametrize expansion (26 parametrize cases minus failing subset). Document the exact RED-phase line in the commit message.

- [ ] **Step 12: Commit Task 1 (RED phase)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/backend/tests/test_document_parser.py
git commit -m "$(cat <<'EOF'
test(parser): WP-CORE-2 red-phase tests for reference-truncation correctness

Add T1-T7 integration tests + T-regex parametrized grammar audit covering
F-5 locale parity (Kaynaklar), nested-subsection false-positive guard, and
trailing-colon heading variants. Tests are written before the production
change per TDD: T2, T4, T7, plus several T-regex parametrize cases fail
red against the current regex; remaining tests pass and act as regression
guards for the subsequent GREEN commit.

Spec: docs/superpowers/specs/2026-05-21-wp-core-2-reference-truncate-fix-design.md
Finding: .planning/pipeline_audit/findings/document_parser.md F-5

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

After the commit, verify:

```bash
git log --oneline -1
```

Expected: the new commit at HEAD; SHA captured for Task 2's GREEN-commit message.

---

## Task 2: GREEN — Apply three production edits + verify all tests pass

**Goal:** Apply the regex tweak, the position guard, and the new class constant in a single atomic commit so the red tests turn green and the existing tests remain green.

**Files:**
- Modify: `extension/backend/core/document_parser.py:7-12` (add constant + change regex)
- Modify: `extension/backend/core/document_parser.py:60-65` (replace `_truncate_at_references` body)

- [ ] **Step 1: Edit the regex pattern at `extension/backend/core/document_parser.py:9-12`**

The current code is:

```python
        self.reference_heading_pattern = re.compile(
            r"^(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography|kaynakça)\s*$",
            re.IGNORECASE,
        )
```

Replace with:

```python
        self.reference_heading_pattern = re.compile(
            r"^(?:#{1,6}\s*)?"
            r"(?:\d+(?:\.\d+)*\.?\s+)?"
            r"(?:references|bibliography|kaynakça|kaynaklar)"
            r"(?:\s*[:：])?\s*$",
            re.IGNORECASE,
        )
```

Changes:
- Split into 4 string literals concatenated by Python's string-literal joining (no `+`, just adjacent strings), purely for readability.
- Added `|kaynaklar` to the keyword alternation.
- Added `(?:\s*[:：])?` to allow optional trailing ASCII colon `:` or fullwidth colon `：` (U+FF1A).
- Nested-number prefix `(?:\d+(?:\.\d+)*\.?\s+)?` retained (so `5.1 References` still matches at the regex level).

- [ ] **Step 2: Add the new class constant immediately before `def __init__` at `extension/backend/core/document_parser.py:7-8`**

The current class header is:

```python
class SRSDocumentParser:
    def __init__(self):
```

Replace with:

```python
class SRSDocumentParser:
    # A bibliography sits in the latter half of an SRS; treating a match in the
    # first half as a false-positive avoids mid-document subsection name clashes.
    REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5

    def __init__(self):
```

- [ ] **Step 3: Replace `_truncate_at_references` body at `extension/backend/core/document_parser.py:60-65`**

The current method is:

```python
    def _truncate_at_references(self, text: str) -> str:
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if self.reference_heading_pattern.match(line.strip()):
                return "\n".join(lines[:index])
        return text
```

Replace with:

```python
    def _truncate_at_references(self, text: str) -> str:
        lines = text.split("\n")
        if not lines:
            return text
        earliest_match_index = int(
            len(lines) * self.REFERENCE_HEADING_MIN_DOCUMENT_FRACTION
        )
        for index in range(earliest_match_index, len(lines)):
            if self.reference_heading_pattern.match(lines[index].strip()):
                return "\n".join(lines[:index])
        return text
```

Changes:
- Early return on empty `lines` (preserves existing empty-string-in/empty-string-out behavior).
- `earliest_match_index` computed once per call.
- Loop iterates from `earliest_match_index` to end instead of from `0`.

- [ ] **Step 4: Run the document_parser tests and confirm GREEN**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend"
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" tests/test_document_parser.py -v 2>&1 | tail -50
```

Expected: every test under `test_document_parser.py` passes, including:
- All 5 pre-existing tests (PASS).
- All 7 new integration tests T1–T7 (PASS).
- All `test_reference_heading_pattern_direct_grammar` parametrize cases (PASS — 26 cases).

If any test still fails, STOP. Inspect failure, do NOT add compensating fallbacks; verify the regex string was transcribed exactly (especially the U+FF1A fullwidth colon).

- [ ] **Step 5: Run the full unit suite and verify baseline does not regress**

```bash
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=short 2>&1 | tail -10
```

Expected: total PASS count = `272 + 7 + 26 = 305` (or however parametrize expands; final number stable across runs). Any other test should not regress. Document the post-edit total in the commit message.

If anything outside `tests/test_document_parser.py` regressed (impossible by construction but verify), STOP. Roll back:

```bash
git reset --hard $(git log --oneline --pretty=format:'%H' | sed -n '2p')   # back to RED commit (Task 1 HEAD)
```

Then escalate to the loop coordinator.

- [ ] **Step 6: Commit Task 2 (GREEN phase)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/backend/core/document_parser.py
git commit -m "$(cat <<'EOF'
fix(parser): WP-CORE-2 reference truncation — locale parity + position guard

F-5: `_truncate_at_references` had two real silent-content-loss defects:
locale-asymmetric Turkish coverage (kaynakça matched but the more common
plural Kaynaklar did not) and a mid-document false-positive on numbered
subsections like `3.4 References` whose entire line matched the regex.

Fix:
- Add `kaynaklar` to the heading-keyword alternation.
- Allow optional trailing colon (ASCII `:` or fullwidth `：` U+FF1A).
- Add REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5 class constant.
- Position-guard `_truncate_at_references` to scan only the latter half
  of the document, so a legitimate `5.1 References` at the end still
  truncates while a stray `3.4 References` subsection mid-doc no longer
  does.

The nested-number regex prefix is preserved so legitimate bibliography
subsections (`5.1 References`, `5.1.2 Kaynaklar`) still match when they
appear in the latter half.

Spec: docs/superpowers/specs/2026-05-21-wp-core-2-reference-truncate-fix-design.md (v2, Codex-reviewed)
Plan: docs/superpowers/plans/2026-05-21-wp-core-2-reference-truncate-fix.md
Decision log: .planning/pipeline_audit/decision_log.md (D-PICK-WP-CORE-2, D-CODEX-REVIEW-WP-CORE-2)
Tests: 7 integration + 26 parametrized grammar-audit cases added in prior commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify:

```bash
git log --oneline -2
```

Expected: GREEN commit at HEAD; RED commit one below.

---

## Task 3: DOC — Write WP dev_doc + INDEX entry + close out audit state

**Goal:** Write the persistent development doc for WP-CORE-2 and update related state files. Audit state files (CURRENT, backlog, decision log) mark the WP shipped with the GREEN-commit SHA.

**Files:**
- Create: `development_docs/WP-CORE-2-reference-truncate-fix.md`
- Modify: `development_docs/INDEX.md` (ACTIVE table — add a row)
- Modify: `.planning/pipeline_audit/improvements_backlog.md` (F-5 → SHIPPED)
- Modify: `.planning/pipeline_audit/CURRENT.md` (last action + next)
- Modify: `.planning/pipeline_audit/decision_log.md` (append SHIPPED outcome)

- [ ] **Step 1: Capture the GREEN-commit SHA**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
WP_CORE_2_SHA=$(git rev-parse HEAD)
echo "$WP_CORE_2_SHA"
```

Note this SHA — used in dev_doc, INDEX, backlog, and decision_log.

- [ ] **Step 2: Create `development_docs/WP-CORE-2-reference-truncate-fix.md`**

Use the canonical dev_doc structure (status / branch / SHAs / spec + plan / TL;DR / motivation / decisions / file-level / methodology / empirical / limitations / cross-refs). Concrete content:

```markdown
# WP-CORE-2 — Reference-heading truncation correctness

**Status:** SHIPPED
**Branch:** main
**Pre-WP SHA:** 029f187
**RED commit SHA:** <from git log; commit one below HEAD>
**GREEN commit SHA:** <from $WP_CORE_2_SHA>
**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-2-reference-truncate-fix-design.md` (v2 — Codex-reviewed)
**Plan:** `docs/superpowers/plans/2026-05-21-wp-core-2-reference-truncate-fix.md`
**Audit finding:** `.planning/pipeline_audit/findings/document_parser.md` F-5
**Codex review summary:** `.planning/pipeline_audit/decision_log.md` D-CODEX-REVIEW-WP-CORE-2

## TL;DR

`SRSDocumentParser._truncate_at_references` had two real silent-content-loss defects: it matched Turkish `kaynakça` but not the more common plural `Kaynaklar`, and the existing regex permitted mid-document false-positives on numbered subsections like `3.4 References`. The fix adds `kaynaklar` and an optional trailing-colon variant to the regex, and bounds the truncation scan to the latter half of the document via a single named constant (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`). All seven new integration tests and 26 parametrized grammar cases pass, and the prior 272 baseline is preserved.

## Motivation

(Reference finding F-5 evidence at `document_parser.py:9-12` and `:60-65`. Briefly cite the Codex WARN-1 escalation that moved the design from regex-narrowing to position-guarded match.)

## Architectural decisions

1. **Position guard over regex narrowing.** Codex pointed out that narrowing the section-number prefix from `\d+(?:\.\d+)*` to `\d+` would silently regress legitimate `5.1 References` appendix-style bibliography layouts. The position guard preserves all current truncation behavior in the latter half of the document while suppressing the mid-document false-positive — a cleaner separation of concerns.
2. **Named class constant for the magic number.** `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5` with a one-line rationale comment, per AGENTS.md "explicit structure over hidden complexity." Avoids the rejected Alt B (unnamed `0.3` magic literal).
3. **Direct parametrized regex test.** Codex WARN-5: integration tests through `parse_file` did not provide auditable grammar coverage. Added `test_reference_heading_pattern_direct_grammar` with 26 explicit accepted/rejected lines.
4. **Out-of-scope WARNs documented, not silently dropped.** Multiword Turkish headings, code-fence guards, Unicode lookalike resilience, and ASCII-folded `kaynakca` are all tracked in `decision_log.md` D-CODEX-REVIEW-WP-CORE-2 for follow-up if observed in D2/D3 live runs.

## File-level changes

| file | change |
|---|---|
| `extension/backend/core/document_parser.py:7-12` | Add `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION` class constant; expand regex to include `kaynaklar` and optional trailing `[:：]`; format regex on four adjacent string literals for readability. |
| `extension/backend/core/document_parser.py:60-69` | `_truncate_at_references` now skips the first half of the document via `earliest_match_index = int(len(lines) * REFERENCE_HEADING_MIN_DOCUMENT_FRACTION)`. Early return on empty `lines`. |
| `extension/backend/tests/test_document_parser.py` | Append T1–T7 integration tests + `test_reference_heading_pattern_direct_grammar` parametrize with 26 cases. |

## Methodology

- Spec written, Codex xhigh adversarial review on v1; spec revised to v2 incorporating WARN-1 (position guard) + WARN-2a (trailing colon) + WARN-5 (parametrize); other 4 WARNs accepted with rationale in `decision_log.md`.
- TDD red→green→commit per task. RED commit established failing tests against unchanged production; GREEN commit applied production edits; full pytest baseline verified between commits.
- No new dependencies. No imports changed. No facade or call-site changes elsewhere in the codebase.

## Empirical results

- Pre-WP pytest baseline: 272 passed, 31 deselected.
- Post-WP pytest baseline: 305 passed (272 + 7 integration + 26 parametrize), 31 deselected. (Confirm exact number from CI log.)
- No live D1 SRS E2E re-run mandated for this WP — regex change strictly reduces false-positive matches and adds an alternation that does not affect English `References` (D1 corpus). Live re-run optional follow-up.

## Limitations & follow-ups

- Multiword Turkish headings (e.g., `Yararlanılan Kaynaklar`) still not matched. Open in backlog as a follow-up WP if a D2/D3 corpus exhibits the pattern.
- Code-fence / preformatted-text guard not implemented; position guard mitigates the common case but a heading-style line embedded in a code fence in the latter half can still false-positive. Future hardening.
- Unicode lookalike defenses (BiDi marks, Cyrillic homoglyphs) not implemented. Future hardening.
- ASCII-folded `kaynakca` (no diacritic) intentionally excluded pending corpus evidence.
- `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5` is a single, named magic number. If a D2/D3 corpus uses bibliography placement past mid-doc but in the latter quarter, the value can be tuned; tests will catch the change.

## Cross-references

- Parent loop: `.planning/pipeline_audit/CURRENT.md`
- Audit catalog: `.planning/pipeline_audit/component_catalog.md`
- Finding F-5: `.planning/pipeline_audit/findings/document_parser.md`
- Decision log entries: `.planning/pipeline_audit/decision_log.md` D-CL1, D-PICK-WP-CORE-2, D-CODEX-REVIEW-WP-CORE-2
- Related future WPs: F-1 (PDF defensive handling), F-2 (.txt printability), F-3 (empty-input contract), F-4 (TOC heuristic), F-6 (`_should_merge` regex completeness), F-7 (DOCX try/except), F-8 (XXE), F-9 (logging), F-10 (double-parse)
- Engineering charter: `AGENTS.md`
- Project conventions: `CLAUDE.md`

## Backlinks

- See `[[WP-CORE-1-typed-pipeline-deterministic-synthesizer]]` for the parent typed-pipeline refactor (placeholder backfill if doc not yet written).
- See `[[WP-NEW-B-Stage-1-schema-probe]]` for the parallel schema-conformance probe work.
```

(When writing the file, substitute the actual RED and GREEN commit SHAs in the `**RED commit SHA:**` and `**GREEN commit SHA:**` fields.)

- [ ] **Step 3: Add an ACTIVE-table row to `development_docs/INDEX.md`**

Read the current ACTIVE table; append a new row consistent with the existing format. Typical row format (verify against actual table headers when reading the file):

```markdown
| WP-CORE-2 | Reference-heading truncation correctness | SHIPPED | <date> | <SHA> | development_docs/WP-CORE-2-reference-truncate-fix.md |
```

If the table format differs, match it exactly. Do not reorder existing rows.

- [ ] **Step 4: Update `.planning/pipeline_audit/improvements_backlog.md` — F-5 to SHIPPED**

Move the F-5 row from the `## Open` table to a new entry under `## Shipped`:

```markdown
| F-5 | document_parser.py | `_truncate_at_references` matched Turkish `kaynakça` but NOT `Kaynaklar`; also false-positive on numbered `3.4 References` mid-document. Fixed via regex alternation expansion + optional trailing colon + position guard (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5`). | MAJOR | S | PIPELINE | **SHIPPED (<GREEN SHA>)** |
```

Also strip the `(WP-CORE-2)` annotation from the previous "IN-PROGRESS" line.

- [ ] **Step 5: Update `.planning/pipeline_audit/CURRENT.md`**

Edit the existing `Last update` / `Last action` / `Next` block to read:

```markdown
**Last update:** <date/time>
**Last action:** WP-CORE-2 shipped at SHA <GREEN SHA>. F-5 backlog row moved to SHIPPED. Dev doc + INDEX updated.
**Next:** Loop iteration 2 — pick next OPEN finding from `improvements_backlog.md`. Highest unblocked priorities are F-3 (empty-input contract) and F-1 (PDF defensive handling). Consider close-lookup of next component by priority (`core/architect.py`, priority 2).

**Baseline (sacred):** pytest -m "not integration" → 305 passed, 31 deselected.
**Pre-loop HEAD:** <GREEN SHA>
```

- [ ] **Step 6: Append SHIPPED outcome to `.planning/pipeline_audit/decision_log.md`**

```markdown


## 2026-05-21 <HH:MM> D-SHIP-WP-CORE-2

WP-CORE-2 SHIPPED. SHAs:
- RED commit (test-first): `<RED SHA>` — `test(parser): WP-CORE-2 red-phase tests…`
- GREEN commit (production): `<GREEN SHA>` — `fix(parser): WP-CORE-2 reference truncation…`
- DOC commit (this entry's commit): `<DOC SHA>` — `chore(artifacts): WP-CORE-2 dev_doc + INDEX entry`

Post-WP pytest baseline: 305 passed, 31 deselected. Live D1 E2E re-run skipped — regex change strictly reduces false-positive matches; English `References` (D1 corpus) unaffected.

Dev doc: `development_docs/WP-CORE-2-reference-truncate-fix.md`.
INDEX row: appended to ACTIVE table.
```

- [ ] **Step 7: Commit Task 3 (DOC + state)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add development_docs/WP-CORE-2-reference-truncate-fix.md \
        development_docs/INDEX.md \
        .planning/pipeline_audit/improvements_backlog.md \
        .planning/pipeline_audit/CURRENT.md \
        .planning/pipeline_audit/decision_log.md
git commit -m "$(cat <<'EOF'
chore(artifacts): WP-CORE-2 dev_doc + INDEX + audit state

Document the WP-CORE-2 reference-truncation fix in development_docs/ and
update the pipeline_audit state (F-5 moved to SHIPPED; CURRENT pointer
advanced; decision_log entry D-SHIP-WP-CORE-2 with SHAs and post-WP
baseline).

This commit produces no source changes; all production code already
shipped in the prior GREEN commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Verify:

```bash
git log --oneline -3
git status
```

Expected: three commits at HEAD (DOC, GREEN, RED); working tree clean (aside from preexisting untracked intermediate artifacts which are out-of-scope for this WP).

- [ ] **Step 8: Final baseline confirmation**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend"
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=no -q 2>&1 | tail -3
```

Expected: `305 passed, 31 deselected`. If the count differs from `< pre-WP + new tests >`, escalate.

- [ ] **Step 9: Do NOT push.**

Per the loop coordinator's standing rule, no `git push` is performed by this plan. The three commits stay on `main` locally until the user explicitly authorizes a push.

---

## Self-Review

**1. Spec coverage:**
- Spec §"Motivation" (F-5 locale + false-positive) → Task 1 T2, T4; Task 2 regex + position guard. ✓
- Spec §"Chosen approach (v2)" three coordinated edits → Task 2 Steps 1, 2, 3. ✓
- Spec §"Testing strategy" T1–T7 + T-regex → Task 1 Steps 2–9. ✓
- Spec §"Out-of-scope" — multiword Turkish, code-fence, Unicode lookalikes, `kaynakca` ASCII fold — documented as future-WP follow-ups in Task 3 dev_doc. ✓
- Spec §"Risks + rollback" — rollback procedure captured in Task 2 Step 5 (escalation branch). ✓
- Spec §"Acceptance criteria" — all 9 items mapped to plan tasks. ✓
- Spec §"Codex review history" — referenced in Task 3 dev_doc + decision_log update. ✓

**2. Placeholder scan:** No TBD/TODO/fill-in placeholders. SHA placeholders (`<RED SHA>`, `<GREEN SHA>`, `<HH:MM>`) are explicit substitution targets bound to commands run in the plan itself, not vague TBDs.

**3. Type consistency:**
- Constant name: `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION` — used identically in Task 2 Step 2 (definition) and Task 2 Step 3 (usage) and Task 3 dev_doc. ✓
- `earliest_match_index` — defined in Task 2 Step 3 only; not referenced elsewhere. ✓
- Test function names: T1–T7 names match between spec and plan exactly. ✓
- Regex pattern: spec v2 multi-line string literal exactly matches Task 2 Step 1 transcription, including U+FF1A. ✓
- File paths: every cited file:line in the plan matches the actual source file (verified during spec phase). ✓
- Pytest invocation: `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration"` used consistently. ✓

No issues found. Plan is ready for execution.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-21-wp-core-2-reference-truncate-fix.md`.

Per the autonomous loop spec at orchestration level, execution proceeds via **superpowers:subagent-driven-development** with fresh implementer + spec-reviewer + code-quality-reviewer subagents per task. Atomic-commit gating verifies the pytest baseline after each commit.
