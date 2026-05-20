# WP-CORE-2 — Reference-heading truncation correctness (locale parity + mid-doc false-positive)

**Date:** 2026-05-21
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 1)
**Status:** REVISED v2 — passed Codex xhigh adversarial review (0 CRITICAL; 3 WARN handled, 4 WARN accepted-with-rationale per `.planning/pipeline_audit/decision_log.md`)
**Parent:** `.planning/pipeline_audit/findings/document_parser.md` finding F-5 (MAJOR)
**Loop:** Domain Pipeline Hardening Loop (first WP after baseline 272 confirmed at HEAD `029f187`)
**Codex consult:** review reply summarized at `decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-2` (2026-05-21 01:54)

---

## Motivation

`SRSDocumentParser._truncate_at_references` (`extension/backend/core/document_parser.py:60-65`) walks each line of the post-normalized SRS text top-to-bottom and, on the first line matching `self.reference_heading_pattern`, drops everything from that line onward. The pattern lives at `extension/backend/core/document_parser.py:9-12`:

```python
self.reference_heading_pattern = re.compile(
    r"^(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography|kaynakça)\s*$",
    re.IGNORECASE,
)
```

Two real defects in this single pattern produce **silent content loss** for downstream Scout/Architect:

1. **Locale-asymmetric Turkish coverage.** The alternation lists `kaynakça` (the diacritic-bearing singular), but the more common plural Turkish bibliography heading is **`Kaynaklar`** — not matched. A Turkish-authored SRS using `Kaynaklar` as the bibliography header is **ingested in full including the bibliography section** (extra noise into Scout). The same SRS authored with `Kaynakça` truncates correctly. This is a single-character class away from full parity, and the asymmetry is the kind of locale bug an EMSE reviewer can spot in 10 seconds. Confirmed against `extension/backend/core/document_parser.py:9-12`; no existing test in `extension/backend/tests/test_document_parser.py` covers `Kaynaklar`.
2. **Nested-section-number false-positive.** The optional section-number group `(?:\d+(?:\.\d+)*\.?\s+)?` accepts **any depth** of dotted nesting (e.g., `3.4 `, `3.4.1 `). A legitimate subsection titled exactly `3.4 References` (followed by no extra text) will match — and the loop returns the document **truncated from that mid-document line onward**, silently. The bibliography is universally a top-level chapter; nested-section nesting is the wrong domain for the regex prefix. Cross-reference: existing test at `tests/test_document_parser.py:132-143` only covers the case where extra text follows the keyword (`"References to external systems..."`), not the case where a bare numbered subsection `"3.4 References"` appears mid-doc.

Both defects fire under the same regex, so a single edit addresses them; both are within the file already audited; both have clean TDD coverage paths. This satisfies the "smallest correct change" rule from AGENTS.md, with two paired test cases per defect (one positive, one negative).

---

## Alternatives considered

### Alt A — Regex-narrowing only (rejected by Codex review v1)
Original plan: drop nested dotting in the section-number prefix to block `3.4 References` and add `kaynaklar`. Rejected after Codex consult because narrowing `\d+(?:\.\d+)*` → `\d+` would silently regress legitimate bibliography subsections like `5.1 References` / `5.1 Kaynakça` (common in IEEE-style appendix layouts where bibliography sits in a numbered subsection of the final chapter). The regex cannot, by itself, distinguish "mid-document false-positive" from "legitimate bibliography subsection."

### Alt B — Position-guarded match (chosen, v2)
Keep the nested-number prefix; instead, only consider the regex when the matched line sits in the latter half of the document. A bibliography is structurally at the end of an SRS — codify that explicitly via a single document-position constant. This:
- Preserves all legitimate matches the current regex catches (`References`, `5. References`, `5.1 References`, `Kaynakça`, etc.) when they appear in the latter half.
- Blocks the mid-document false-positive of `3.4 References` when it appears in the earlier half.
- Adds `kaynaklar` to the alternation (locale parity, the original F-5 motivation).
- Adds optional trailing colon (`References:` / `Kaynaklar:`) per Codex WARN-2a.

Net code diff: one regex string change + one new class constant + small loop guard in `_truncate_at_references` (~5 LOC production).

**Tradeoff:** Introduces one magic constant (`0.5`), but it is named (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION`) and its rationale is one comment line: "bibliography sits in the latter half of an SRS." This is "explicit structure over hidden complexity" (AGENTS.md), not a hidden number. The alternative — building a semantic heading hierarchy — is rejected as overengineering (was Alt C v1).

### Alt C — Parse semantic heading hierarchy (rejected)
Build a true heading tree (Heading 1 / Heading 2 levels) and only truncate at a top-level `References` chapter. Rejected because: (i) cross-format (PDF vs DOCX vs TXT) hierarchy detection is a sub-project, (ii) overengineering for a content-fidelity bug, (iii) explicitly violates "no speculative generalization" + "do not introduce abstractions unless they clearly improve the design" from AGENTS.md.

---

## Chosen approach (v2)

**Three coordinated edits** at `extension/backend/core/document_parser.py`:

### (1) Regex change at `:9-12`

- Add `kaynaklar` to the keyword alternation: `references|bibliography|kaynakça|kaynaklar` (Turkish plural parity).
- Add optional trailing colon `(?:\s*[:：])?` to allow `References:` / `Kaynaklar:` / `Kaynakça:` / fullwidth `Kaynaklar：` headings.
- Keep the nested-number prefix `(?:\d+(?:\.\d+)*\.?\s+)?` unchanged (so `5.1 References` still matches).

**Final regex:**

```python
self.reference_heading_pattern = re.compile(
    r"^(?:#{1,6}\s*)?"
    r"(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:references|bibliography|kaynakça|kaynaklar)"
    r"(?:\s*[:：])?\s*$",
    re.IGNORECASE,
)
```

### (2) New class constant just above `__init__` (or near top of class body)

```python
class SRSDocumentParser:
    # A bibliography sits in the latter half of an SRS; treating a match in the
    # first half as a false-positive avoids mid-document subsection name clashes.
    REFERENCE_HEADING_MIN_DOCUMENT_FRACTION = 0.5

    def __init__(self):
        ...
```

### (3) Position guard inside `_truncate_at_references` at `:60-65`

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

Lines below `earliest_match_index` are scanned; lines above are skipped entirely. For an empty input, the early return preserves current behavior (empty string in, empty string out). Documents shorter than 2 lines still scan all lines (since `int(0 * 0.5) = 0`, `int(1 * 0.5) = 0`).

Net production diff: ~9 LOC (1 regex string spanning 4 lines for readability + 1 constant w/ comment + ~5 loop lines).

---

## Architecture sketch (no abstraction creep)

```
Before                                 After (v2)
----------                             ----------
parse_file(path)  ──┐                  parse_file(path)  ──┐
                    ▼                                       ▼
              _post_process(text)                     _post_process(text)
                    │                                       │
                    ├─ _normalize_text                      ├─ _normalize_text
                    ├─ _truncate_at_references ◄──── EDIT ─►├─ _truncate_at_references
                    │      └─ regex matches headings        │      ├─ skip first half (position guard, NEW)
                    │         anywhere in document          │      └─ regex matches headings only past midpoint
                    │                                       │         (regex now also matches kaynaklar + trailing colon)
                    ├─ _clean_lines                         ├─ _clean_lines
                    ├─ _merge_wrapped_lines                 ├─ _merge_wrapped_lines
                    └─ _normalize_text                      └─ _normalize_text
```

No new modules, no new classes, no signature changes, no new dependencies. Public API (`parse_file`) stable. One new class-level constant (`REFERENCE_HEADING_MIN_DOCUMENT_FRACTION`) — not an abstraction, just a named magic number per AGENTS.md "explicit structure over hidden complexity."

---

## Error handling + failure modes

- **No new error paths introduced.** The `_truncate_at_references` function still returns the (possibly truncated) text; on no match it returns text unchanged, matching current behavior.
- **No silent fallbacks.** The regex either matches (truncate) or does not (preserve). No `try/except` added.
- **Failure mode that could regress:** if downstream code consumed the now-truncated content from a real `3.4 References to External Systems` heading where the heading-only false-match used to fire by accident, that would be hidden behavior. Risk assessment: zero — the regex was already requiring `\s*$` (end-of-line anchored), so `3.4 References to External Systems` never matched (extra text after keyword). Existing test `tests/test_document_parser.py:132-143` already proves this.
- **Failure mode for the locale fix:** the `kaynaklar` alternation introduces a new positive match. If any legitimate SRS content used the standalone Turkish word `Kaynaklar` outside a heading position, it could now truncate mid-content. Risk assessment: very low — the regex requires the entire line to match (`^…\s*$`), so an inline mention like `"Kaynaklar listesi aşağıdadır."` does not match. Tested explicitly via a negative test case (`Kaynaklarımız geniştir.` → not truncated).

---

## Testing strategy (TDD red→green→commit)

Two test additions to `extension/backend/tests/test_document_parser.py`:

1. **Integration tests T1–T8** — exercise `parse_file → _post_process → _truncate_at_references` through the public API (Codex WARN-1, WARN-2a regression coverage).
2. **Parametrized regex-level test T-regex** — direct match table against `SRSDocumentParser().reference_heading_pattern.match(line.strip())` (Codex WARN-5 — auditable grammar coverage).

All integration tests use in-memory `.txt` written to `tmp_path`. The regex operates on post-normalized text which is format-agnostic by the time it reaches `_truncate_at_references`, so PDF/DOCX format coverage is sufficient via the existing pre-edit tests at `tests/test_document_parser.py:81-97` and `:100-120`.

To satisfy the position-guard's "earliest_match_index = int(len(lines) * 0.5)" rule, each test crafts the input so that the heading sits in the latter half (positive truncate) or earlier half (negative no-truncate), with comments explaining the line layout.

### T1 — Turkish singular "Kaynakça" — positive truncation (line in latter half)

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

### T2 — Turkish plural "Kaynaklar" — positive truncation (locale parity, F-5 core fix)

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

### T3 — Numbered top-level "5. References" — positive truncation (regression check)

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

### T4 — Numbered nested subsection "3.4 References" mid-doc — negative no-truncation (false-positive guard)

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

### T5 — Inline Turkish "Kaynaklar" use not at heading position — negative no-truncation (locale-safety)

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

### T6 — Nested-number `5.1 References` in latter half — positive (Codex WARN-1 regression guard)

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

### T7 — Trailing-colon `References:` — positive (Codex WARN-2a)

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

### T-regex — Parametrized direct grammar audit (Codex WARN-5)

Exercises `SRSDocumentParser().reference_heading_pattern.match(line.strip())` directly — independent of position guard. Tests grammar coverage of the regex alone:

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
        ("Kaynaklar：", True),  # fullwidth colon
        ("references", True),
        # negative — must NOT match
        ("References to external systems shall be preserved.", False),
        ("Kaynaklarımız geniştir.", False),  # inline; suffix attached
        ("Some References", False),  # no recognised section-number/header prefix
        ("3.4 References to Other Documents", False),  # extra text after keyword
        ("3.4 Subsection References", False),  # extra text after section number
        ("Bibliography of authors:", False),  # extra text after keyword
        ("", False),
        ("  ", False),
    ],
)
def test_reference_heading_pattern_direct_grammar(line, should_match):
    parser = SRSDocumentParser()
    match = parser.reference_heading_pattern.match(line.strip())
    assert bool(match) is should_match, f"Pattern mismatch for: {line!r}"
```

**Existing tests that must still pass unchanged:**

- `tests/test_document_parser.py:81-97` — `test_parse_pdf_merges_wrapped_lines_and_stops_at_references`: 5 lines, `References` at index 3 (>= midpoint 2) → still truncates.
- `tests/test_document_parser.py:132-143` — `test_parse_txt_does_not_truncate_regular_requirement_lines`: 2 lines, `References to external systems...` at index 0 (< midpoint 1) → never matched anyway (extra text after keyword); position guard is benign here.

**TDD discipline:**

- **Red phase**: add all new tests (T1–T7 + T-regex) to `test_document_parser.py` with the **current** regex + truncation function unchanged. Expected failures:
  - T2 fails (Kaynaklar not matched).
  - T4 fails (3.4 References mid-doc currently false-positive truncates → "ISO 8601 timestamps" etc. drops out).
  - T7 fails (`References:` with trailing colon currently does not match).
  - T-regex parametrized cases for `Kaynaklar`, `Kaynaklar:`, `Kaynaklar：` fail (no match).
  - T-regex case `5.1 Kaynakça` — current regex DOES match (passes), but only because nested-number prefix is permissive. With v2 (position guard, nested prefix retained), still matches at regex level. OK.
  - T1, T3, T5, T6, all other T-regex cases pre-pass.
- **Green phase**: apply the three production edits (regex, constant, loop guard). Re-run; T1–T7 + T-regex all pass.
- **Refactor phase**: no refactor needed; the regex is multiline-formatted for readability but is a single logical change.

Each phase is a separate atomic commit per SDD.

---

## File-level change list

| action | file | scope | LOC delta |
|---|---|---|---|
| MODIFY | `extension/backend/core/document_parser.py:7-12` | Add `REFERENCE_HEADING_MIN_DOCUMENT_FRACTION` class constant + change one regex pattern (now multi-line for readability). | +≈6 LOC |
| MODIFY | `extension/backend/core/document_parser.py:60-65` | Replace `_truncate_at_references` body with position-guarded loop (~5 LOC inside the function). | +≈5 LOC |
| MODIFY | `extension/backend/tests/test_document_parser.py` | Append T1–T7 integration tests + T-regex parametrized grammar audit. | +≈170 LOC |

Total diff: ~180 LOC added (≥80 % tests), ~11 LOC modified production.

No file create, no file delete, no rename. No import changes (T-regex uses already-imported `pytest`). No new dependencies. No `requirements.lock` change.

---

## Out-of-scope (explicit list)

The following findings from `findings/document_parser.md` are **explicitly not addressed** in this WP. They remain OPEN in `improvements_backlog.md` for future iterations:

- **F-1** PDF reader defensive handling (encrypted/image-only/malformed) — separate WP.
- **F-2** `.txt` near-binary acceptance via `_looks_like_text` 0.95 ratio — separate WP.
- **F-3** Empty-string return contract from `parse_file` — separate WP.
- **F-4** TOC heuristic 120-line window + cluster<2 — separate WP, needs Codex consult.
- **F-6** `_should_merge` quote/bracket terminator gap — separate WP, lower severity.
- **F-7** DOCX no try/except around `docx.Document` — separate WP.
- **F-8** XXE / external-entity hardening on DOCX XML — separate WP, security-policy track.
- **F-9** Logging gap — observability WP (multi-module scope).
- **F-10** Double-parse in `main.py:366,480` — perf WP, cross-cutting `main.py`.

**Also out of scope:**

- Refactor of `_truncate_at_references` to support "truncate only past midpoint" heuristic — rejected as Alt B, considered overengineering.
- Adding logging to the truncation event — observability work belongs to F-9 WP.
- Migration of `reference_heading_pattern` to a config-driven keyword list — speculative generalization (AGENTS.md), no second consumer exists.
- Any change to `_find_toc_line_indexes` — separate F-4 WP.
- Any change to existing call sites in `main.py` — no API surface changes.

---

## Risks + rollback plan

### Risks

| risk | severity | mitigation |
|---|---|---|
| New regex causes a previously-passing test to fail. | LOW | Full unit-test run gates every commit; baseline = 272. T1–T5 written before edit (red→green). Pre-existing tests (`test_parse_pdf_merges_wrapped_lines_and_stops_at_references`, `test_parse_txt_does_not_truncate_regular_requirement_lines`) explicitly re-verified in the post-edit run. |
| `kaynaklar` alternation truncates a legitimate Turkish SRS that contains the word as a heading-style false-positive in a non-bibliography section. | LOW | T5 covers inline mention. Heading-position match still requires `^…\s*$` line anchoring. Risk only realizes if the SRS uses `Kaynaklar` as the standalone title of a non-bibliography section (extremely unusual; if encountered, follow-up WP). |
| Tightening the section-number prefix breaks an SRS that uses `3.4 References` as a *legitimate* bibliography location (e.g., embedded bibliography under section 3). | VERY LOW | Such a layout is non-standard. Existing test corpus + the D1 SRS already in `extension/backend/inputs/` do not exhibit this pattern (verified during finding catalog). If the EMSE evaluation corpus does encounter this, a future WP can broaden the prefix. Current change is *more* conservative than the existing regex (subset of previous match space), so any document the current code truncates at a numbered top-level header still truncates. |
| Live D1 SRS E2E pipeline regression (4 ctx × 7 entity baseline). | VERY LOW | Regex change strictly reduces false-positive matches and adds one synonym. The D1 SRS is in English and uses standard `References` (verified in close-lookup F-5 evidence). Live re-run optional but recommended. |
| Hidden ASCII fold for `Kaynakça` (e.g., `Kaynakca` no diacritic) silently breaks parity. | OUT-OF-SCOPE | Diacritic-stripped Turkish was not in F-5; tracked as follow-up if observed. Do not add `kaynakca` speculatively. |

### Rollback

```bash
# If pytest baseline drops or live E2E regresses:
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git reset --hard $(cat /tmp/wp-core-2-pre-sha.txt)
# Then update decision_log.md with REJECTED status + reason.
```

Per-commit rollback is also viable since this WP ships in two atomic commits (test-first + production change), but full-WP `reset --hard` to pre-SHA is the documented procedure if either commit lands and baseline drops.

---

## Acceptance criteria

- [ ] T1–T7 + T-regex parametrized added to `tests/test_document_parser.py`, all PASS.
- [ ] Existing `tests/test_document_parser.py:81-97`, `:100-120`, `:123-129`, `:132-143`, `:146-150` still PASS unchanged.
- [ ] `pytest -m "not integration"` count ≥ 272 + 7 + N (where N = parametrized expansions = 26) = **305** new test-case count target. (Approximate; CI count is canonical.)
- [ ] No new files created.
- [ ] No new dependencies introduced.
- [ ] No change to `core/parser.py`, `core/llm/`, `core/architect.py`, `main.py`, or `config.py`.
- [ ] Atomic commits with Conventional Commits prefix + Claude trailer.
- [ ] dev_doc `development_docs/WP-CORE-2-reference-truncate-fix.md` written + INDEX updated.
- [ ] `improvements_backlog.md` F-5 status: SHIPPED with SHA.
- [ ] `decision_log.md` Codex review entry (D-CODEX-REVIEW-WP-CORE-2) referenced from dev_doc.

---

## Codex review history

**Round 1 (2026-05-21 01:54)** — Codex xhigh adversarial review on spec v1. Verdict: REVISE. 0 BLOCKER/CRITICAL, 6 WARN.

Disposition (full table in `decision_log.md` D-CODEX-REVIEW-WP-CORE-2):

- W-1 (`5.1 References` regression risk from regex narrowing) → **HANDLED** in v2: replace regex narrowing with position guard, keep nested-number prefix. New tests T6 + T-regex parametrize cover this.
- W-2a (`References:` trailing colon) → **HANDLED** in v2: regex now accepts optional `[:：]`. New test T7 + T-regex parametrize cover this.
- W-2b (multiword Turkish `Yararlanılan Kaynaklar`) → ACCEPTED — out of scope; tracked for follow-up if observed in D2/D3 live runs.
- W-3 (no code-fence guard) → ACCEPTED — out of scope; mitigated for the common case by position guard. Tracked for future hardening.
- W-4 (Unicode lookalikes / BiDi marks) → ACCEPTED — out of scope; Turkish dotted-I analysis shows current alternation safe. Tracked for future hardening.
- W-5 (no direct parametrized regex test) → **HANDLED** in v2: T-regex parametrize block added.
- W-6 (ASCII-folded `kaynakca`) → ACCEPTED — out of scope; no speculative addition without corpus evidence. Tracked for follow-up if observed in D2/D3 live runs.

Spec v2 status: 3 WARN handled in spec, 4 WARN accepted-with-rationale in decision_log. Ready for plan + SDD.
