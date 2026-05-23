# WP-CORE-9 — Mislabeled-file detection in `read_txt`

**Status:** SHIPPED 2026-05-23
**Branch / commits:**
- RED `45d9cdf` — test(document_parser): WP-CORE-9 red-phase tests
- GREEN `ff28324` — fix(document_parser): WP-CORE-9 MisLabeledFileError + magic-byte detection
- DOC `{this commit}` — chore(artifacts): WP-CORE-9 dev_doc + audit state
- PLANNING `{pending}` — chore(planning): WP-CORE-9 spec v2 + plan

**Spec:** `docs/superpowers/specs/2026-05-23-wp-core-9-mislabeled-file-detection-design.md` (v2)
**Plan:** `docs/superpowers/plans/2026-05-23-wp-core-9-mislabeled-file-detection.md`
**Parent finding:** `.planning/pipeline_audit/findings/document_parser.md` **F-2** (MAJOR) — now SHIPPED.

## TL;DR

`read_txt` previously let cp1254 single-byte fallback silently decode renamed binary files (`.docx`/.pdf saved as `.txt`) into gibberish OR fail with opaque `UnicodeDecodeError`. WP-CORE-9 adds magic-byte detection upfront via `_detect_binary_signature` + 10 binary-format signatures (3 ZIP variants + PDF + OLE + PNG + JPEG + 2 GIF variants + gzip). On match raises typed `MisLabeledFileError(ValueError)` with `file_path` + `detected_format` attrs. Re-exported from `core.document_parser` for clean import path (mirrors `EmptySRSDocumentError`).

Dual benefit: (1) rare silent-accept case caught at byte-0 signature; (2) common-case diagnostics improved — typed error names the actual format.

Baseline: 365 → 373 passing (+8 tests, zero regression).

## Motivation

F-2 audit text: "cp1254/cp1252 are single-byte encodings — virtually any byte stream decodes successfully under them, so a malformed `.txt` will be accepted under cp1254 if it happens to contain ≥ 95% printable characters in that codepage."

Codex W-6 reframed: the *common* binary-file case actually fails — real ZIP/DOCX headers contain NUL bytes early; `_looks_like_text`'s NUL-rejection catches them; falls through to `UnicodeDecodeError` with no format detail. The *rare* silent-accept case happens only on no-NUL printable content. So WP-CORE-9 has dual benefit, not a single bug fix.

Production reachability (Codex W-5): VSCode file picker (`extension.ts:511-518`) accepts `.txt`/.pdf`/`.docx` by extension; user renames file by mistake; `extension.ts:613-615` sends path to backend; `parse_file` (`document_parser.py:53-60`) dispatches by extension only; `read_txt` invoked on binary content.

## Architectural decisions

### D-1 — Magic-byte signature table over heuristic tightening

Two design options considered:
- **(A)** Tighten `_looks_like_text` thresholds (raise printable ratio, add word-character ratio, etc.). Fragile, risks regression on legitimate non-Latin text.
- **(B)** Magic-byte detection upfront (chosen). Deterministic. Targets the realistic F-2 trigger (renamed binaries).

### D-2 — Re-export `MisLabeledFileError` from `core.document_parser` (Codex OQ)

The exception class lives in `core/document_parser_readers.py` (co-located with `read_txt`) but is re-exported via `core/document_parser.py`'s `__all__` so consumers can import via the parser entrypoint:

```python
from core.document_parser import MisLabeledFileError  # public path
```

Mirrors `EmptySRSDocumentError`'s import location. Avoids forcing callers to know reader-module internals.

### D-3 — Signature table covers 10 formats including 3 ZIP variants (Codex W-1)

`PK\x03\x04` (normal local-file-header) + `PK\x05\x06` (empty archive end-of-central-directory) + `PK\x07\x08` (split/spanned data descriptor). Most realistic `.docx` files start with `PK\x03\x04`, but completeness against ZIP edge cases costs ~2 LOC.

### D-4 — Check runs BEFORE encoding-decode loop

`_detect_binary_signature` is invoked at the top of `read_txt`, before `_candidate_text_encodings` or `_looks_like_text`. This ensures even pure-printable binaries (which would currently pass `_looks_like_text`) are caught.

### D-5 — `.txt`-only scope (Codex W-8, explicit non-goal)

`parse_file` dispatches by extension only:
- `.txt` → `read_txt` — WP-CORE-9 scope.
- `.pdf` → `read_pdf` — F-1 backlog (separate WP).
- `.docx` → `read_docx` — F-7 backlog (separate WP).

A `.docx` renamed `.pdf` routes to `read_pdf` and gets a `pypdf` error; symmetric defenses for `read_pdf` / `read_docx` are part of F-1 / F-7's future WPs.

## File-level changes

| File | Change | LOC delta |
|---|---|---|
| `core/document_parser_readers.py` | + `MisLabeledFileError(ValueError)` class; + `_BINARY_MAGIC_SIGNATURES` tuple constant (10 signatures); + `_detect_binary_signature` helper; + pre-decode check in `read_txt` | +85 |
| `core/document_parser.py` | + Re-export `MisLabeledFileError` via `__all__` | +14 |
| `tests/test_document_parser_mislabeled_file.py` (NEW) | T-MFE-1..8 (6 RED-by-design + 2 GREEN-from-start regression guards) | +221 |

## Methodology applied

- **TDD with reclassified RED.** Codex C-1 caught that T-MFE-5 was misclassified as RED (it tests current legitimate-text behavior which already passes). Reclassified as GREEN-from-start regression guard. T-MFE-7 similarly classified. RED commit accepted 6 failing tests (ImportError × 6), 2 passing tests.
- **Spec → Codex xhigh → spec v2 → atomic commits.** 1 CRITICAL + 8 WARN + 4 NIT + 1 OQ; all CRITICAL+WARN inline; 4 NIT inlined; OQ resolved with re-export pattern.
- **Smallest correct change.** Magic-byte detection scope-bounded to `.txt`; `read_pdf` / `read_docx` symmetric defenses deferred to F-1 / F-7.

## Empirical results

- **Test baseline**: 365 → 373 (+8 tests, zero regression).
- **LOC delta vs WP-CORE-8**: +85 (readers) + 14 (parser re-export) = +99 production / +221 test.
- **Failure surface**: pre-WP-CORE-9 mislabeled `.txt` paths produced opaque `UnicodeDecodeError` OR gibberish on no-NUL content. Post-WP-CORE-9: typed `MisLabeledFileError` with detected format label.

## Limitations + follow-ups

- **Scope `.txt`-only**. F-1 (read_pdf) and F-7 (read_docx) tracked separately. Pattern could be re-used: invoke `_detect_binary_signature` at top of each reader to reject extension-mismatch upfront. Open as future WP.
- **`_looks_like_text` threshold untouched** (95% printable). Codex N-1 deferred tightening; current heuristic + new signature detection is sufficient.
- **Signature table is closed-set**. New formats (e.g., WebP, Apache Arrow) added to industry won't be detected. Acceptable maintenance burden.
- **No BOM-prefixed binary edge case**. `\xEF\xBB\xBF` (UTF-8 BOM) routes through BOM detection at `_candidate_text_encodings:113-114`; the magic-byte check sees `data.startswith(b"\xEF\xBB\xBF...")` which doesn't match any binary signature. Correct behavior per T-MFE-7.

## Cross-references

- **Predecessor / sibling**: `[[WP-CORE-3-empty-input-contract]]` (EmptySRSDocumentError; ValueError taxonomy) — `MisLabeledFileError` mirrors its design.
- **Invariant chain**:
  - WP-CORE-3 invariant: `parse_file` raises `EmptySRSDocumentError` on empty content (not return `""`).
  - **WP-CORE-9 NEW invariant**: `read_txt` raises `MisLabeledFileError` on magic-byte-detected mislabeled binary content (not return gibberish or raise `UnicodeDecodeError`).
- **EMSE paper**: pre-WP-CORE-9 mislabeled binary inputs could pollute the domain model silently. Post-WP-CORE-9 rejected at ingestion boundary with format label. Flag for advisor at next paper revision.
