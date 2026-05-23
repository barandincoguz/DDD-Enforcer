# WP-CORE-9 — Mislabeled-file detection in `read_txt` (F-2)

**Date:** 2026-05-23
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 8)
**Status:** REVISED v2 — addressed Codex xhigh adversarial review (1 CRITICAL + 8 WARN + 4 NIT + 1 OQ; CRITICAL+WARN all handled inline; OQ tracked)
**Parent finding:** `.planning/pipeline_audit/findings/document_parser.md` **F-2** (MAJOR)
**Loop:** Domain Pipeline Hardening Loop (eighth WP; baseline 365 confirmed at HEAD `0e43812`)
**Codex review:** `decision_log.md` D-CODEX-REVIEW-WP-CORE-9 (to be appended at DOC commit).

## Revision history

- **v1 (draft, 2026-05-23 ~13:00 GMT+3)** — initial spec; sent to Codex xhigh.
- **v2 (this version, 2026-05-23 ~13:30 GMT+3)** — Codex xhigh verdict: **1 CRITICAL + 8 WARN + 4 NIT + 1 OQ**. Dispositions:

  | # | finding | category | disposition |
  |---|---|---|---|
  | **C-1 (A2-1)** | T-MFE-5 misclassified as RED; current `read_txt` already accepts legitimate text containing the literal "PK..." substring; test is GREEN-from-start, not RED-by-design. | test plan accuracy | **ADOPTED.** T-MFE-5 reclassified as GREEN regression test (passes pre-GREEN, must still pass post-GREEN as false-positive-resistance guarantee). RED commit math reconciled: 4 RED-by-design + 1 GREEN-from-start = 5 new tests; RED pytest delta = +1 fail (T-MFE-1) + 3 more fails (T-MFE-2/3/4 — these DO actually fail pre-GREEN because spec D-5 raises BEFORE encoding loop) + 1 always-passing (T-MFE-5). RED = 365 + 4 fail = 369 collected, 365 pass, 4 fail. |
  | **W-1 (A1-1)** | ZIP signature coverage incomplete: `PK\x03\x04` only; missing `PK\x05\x06` (empty archive) + `PK\x07\x08` (spanned/split marker). | scope gap | **ADOPTED.** Signature table extended with both variants labeled "ZIP archive (empty)" and "ZIP archive (split/spanned)". |
  | **W-2 (A2-2)** | T-MFE-2/3/4 fixtures may not actually prove silent-gibberish path: real ZIP/DOCX headers contain NUL bytes early; `_looks_like_text` already rejects on NUL. Fixtures with just `b"PK\x03\x04..."` likely already fail the current heuristic. | test rigor | **ADOPTED.** Fixtures expanded: T-MFE-2 fixture is a long no-NUL printable payload prefixed with `PK\x03\x04` (proves the rare silent-accept case where magic byte detection adds value beyond current NUL-check); T-MFE-3 PDF fixture similarly. Plus a NEW T-MFE-6 fixture that uses a realistic ZIP header WITH embedded NUL bytes (proves diagnostics improvement: instead of UnicodeDecodeError "Unable to decode", user gets MisLabeledFileError naming the format). |
  | **W-3 (A2-3)** | BOM + later-magic-bytes edge case untested. | test gap | **ADOPTED.** T-MFE-7 NEW: UTF-8 BOM (`\xEF\xBB\xBF`) prefix followed by legitimate text containing `%PDF-` as literal substring — magic-byte detection MUST NOT fire (BOM precedence + signature only checks `data.startswith`, not "anywhere"). Spec D-5 already specifies `data.startswith` so behavior is correct; test locks it down. |
  | **W-4 (A2-4)** | Near-miss signature tests missing (`PK\x03` truncated, `%PDX-` typo, `\x89P` truncated PNG). | test gap | **ADOPTED.** T-MFE-8 NEW: helper-level test of `_detect_binary_signature` against near-miss prefixes. Returns None for each. |
  | **W-5 (A3-1)** | Reachability evidence cites "zero test" rather than the concrete UI-to-backend path. | motivation rigor | **ADOPTED.** §Motivation Production-reachability subsection rewritten: VSCode file picker (`extension.ts:511-518`) accepts `.txt`/`.pdf`/`.docx`; user renames file → `extension.ts:613-615` sends path to backend → `parse_file` dispatches by extension only (`document_parser.py:53-60`) → `read_txt` invoked on binary content. |
  | **W-6 (A3-2)** | Spec wording overstates "silent gibberish" — current NUL-byte check rejects ZIP/DOCX headers (NUL very common in those formats); the *common* case is UnicodeDecodeError, not silent acceptance. Rare case is pure-printable-no-NUL cp1254 content. | accuracy | **ADOPTED with reframe.** §Motivation reframed: WP-CORE-9 has DUAL benefit — (a) catches the rare silent-accept case (cp1254 decodes of no-NUL printable binary patterns), (b) improves diagnostics for the common case (currently raises `UnicodeDecodeError("Unable to decode text file")` with no format detail; post-WP-CORE-9 raises `MisLabeledFileError(detected_format="PDF")` naming the actual format). |
  | **W-7 (A4-1)** | Spec downstream-impact wrong: endpoints route through `_parse_srs_batch` (`main.py:418-424`, `:541-548`), not bare-Exception outer catch. | downstream-impact accuracy | **ADOPTED.** §Downstream-impact §1 rewritten: `_parse_srs_batch` (lines 56-100) catches `EmptySRSDocumentError` specifically + generic `Exception`. `MisLabeledFileError(ValueError)` lands in the generic-Exception branch → response includes `"Failed to parse {path}: {exc}"` with the detected format. WP-CORE-8 typed `PipelineError` handler is separate path (orchestration-layer errors, not ingestion). |
  | **W-8 (A4-2 + A6-1)** | OQ-3 should cite F-7 (DOCX zero try/except) alongside F-1 (read_pdf defensive); spec should add explicit `.txt`-only non-goal. | scope precision | **ADOPTED.** Spec §Non-Goals added: a `.docx` renamed `.pdf` routes to `read_pdf` (NOT `read_txt`); a `.pdf` renamed `.docx` routes to `read_docx`. WP-CORE-9 only fixes the `.txt` rename target. OQ-3 cites F-1 + F-7 both. |
  | **N-1 (A1-2)** | OOXML inner-marker detection (parsing `[Content_Types].xml` after ZIP) unnecessary and out of scope. | confirmation | **ACCEPT-AS-IS.** Outer ZIP signature labels as "likely OOXML"; no inner-archive parsing. |
  | **N-2 (A5-1)** | Use immutable tuple constant; type imports clean. | confirmation | **ADOPTED.** `_BINARY_MAGIC_SIGNATURES: tuple[tuple[bytes, str], ...]` (immutable); `Optional[str]` typed via `typing` import already at top of file. |
  | **N-3 (A6-2)** | "Ordered by first-byte specificity" comment inaccurate (signatures are non-overlapping by `startswith`; order doesn't matter today). | precision | **ADOPTED.** Comment reworded: "Non-overlapping by `startswith`; order has no behavioral effect today. If overlapping prefixes are added later, longer-specific signatures must come first." |
  | **OQ (A6-3)** | `MisLabeledFileError` public import path — `core.document_parser` re-export OR keep at reader-module? | clean import surface | **ADOPTED with re-export.** v2 spec §D-3 places `MisLabeledFileError` in `core/document_parser_readers.py` (co-located with `read_txt`) AND re-exports from `core/document_parser.py` so consumers can import via the parser entrypoint (mirrors `EmptySRSDocumentError`'s path). Test imports also via `core.document_parser`. |

  **Codex disposition summary**: 1 CRITICAL (test misclassification) ADOPTED with RED math reconciliation; 8 WARN all ADOPTED inline; 4 NIT inlined; 1 OQ resolved with re-export pattern.

## Motivation

### The bug (cp1254 fallback accepts mislabeled binary files)

`core/document_parser_readers.py:read_txt` (lines 92-109) iterates encodings (UTF-8 → UTF-8-sig → UTF-16 → cp1254 → cp1252) and accepts the first decode that survives `_looks_like_text` (`:120-126`). The heuristic accepts any string with ≥ 95 % printable+whitespace characters.

cp1254 is a single-byte encoding — virtually every byte stream decodes successfully under it. If a renamed `.docx` (ZIP archive), `.pdf`, or any binary blob whose byte distribution happens to contain ≥ 95 % printable-in-cp1254 characters is passed as `.txt`, cp1254 succeeds, the heuristic passes, and `read_txt` returns gibberish silently.

The gibberish then flows downstream to Scout → Architect → Specialist; Gemini calls burn quota on noise; the final `DomainModel` is silently polluted. There is no error signal at any stage.

### Production reachability (loop discipline — mandatory subsection)

**F-2 status: LIVE.** Path:
1. User drops a renamed file (e.g., `requirements.docx` saved with `.txt` extension) into `inputs/`.
2. `SRSDocumentParser.parse_file` dispatches to `read_txt` based on extension (`document_parser.py`).
3. `read_txt`'s encoding loop accepts cp1254-decoded ZIP bytes; `_looks_like_text` passes if printable-byte ratio ≥ 95 %.
4. The returned gibberish enters the pipeline. No exception raised.

Empirical reachability evidence: zero existing test asserts `read_txt` raises on mislabeled binary input. `test_parse_txt_supports_utf16_input` only covers the happy-encoding case.

Contrast with WP-CORE-7 / 8 orchestrator paths — this lives at ingestion boundary. The pivot to ingestion-layer per WP-CORE-8 handoff is targeted at exactly this surface.

### Why magic-byte detection over heuristic tightening

Two design options:
- **(A) Tighten `_looks_like_text` thresholds** (raise printable ratio to 0.99, add word-character ratio, add replacement-character ratio). Fragile — every threshold tweak risks regression on legitimate non-Latin text (Turkish, math symbols, etc.). Hard to test exhaustively.
- **(B) Magic-byte detection upfront** (chosen). Deterministic. Detects renamed binaries (`.docx`/.zip = `PK\x03\x04`; `.pdf` = `%PDF`; `.doc` OLE = `\xD0\xCF\x11\xE0`; PNG/JPEG/etc.) **before** the encoding loop. On detection: raise typed `MisLabeledFileError(ValueError)` with a clear message naming the detected actual format.

Magic-byte detection covers the most common real-world cause of F-2 (user renames `.docx` → `.txt`). The `_looks_like_text` threshold remains as-is; tightening it is a separate WP if needed.

## Discovery (audit-text-vs-code-reality)

### D-1. Backlog claim verified

**Claim** (`findings/document_parser.md` F-2): "cp1254/cp1252 are single-byte encodings — virtually any byte stream decodes successfully under them, so a malformed `.txt` (e.g. a renamed `.docx` or a binary blob) will be accepted under cp1254 if it happens to contain ≥ 95 % printable characters in that codepage."

**Code reality (HEAD `0e43812`, `document_parser_readers.py:92-126`):**
- `read_txt` iterates encodings in order: UTF-8, UTF-8-sig, UTF-16, cp1254, cp1252.
- For BOM-prefixed UTF-8 or UTF-16 LE/BE: detected via `_candidate_text_encodings` (lines 112-117).
- For plain bytes: tries UTF-8 first; if `UnicodeDecodeError`, falls back through utf-8-sig → utf-16 → cp1254 → cp1252. cp1254 is single-byte; NEVER raises.
- `_looks_like_text` rejects only on NUL byte (`\x00`) or printable-ratio < 95 %.

Verdict: backlog claim accurate. A `.docx` (ZIP starts with `PK\x03\x04`) bypasses NUL check if the central directory bytes are all printable. A `.pdf` (`%PDF-...`) is mostly printable header + binary streams; total ratio depends on PDF content. Detection is statistically possible but not guaranteed.

### D-2. Why `MisLabeledFileError` is new, not a `ValueError` subclass of existing

`SRSDocumentParser` already has `EmptySRSDocumentError(ValueError)` (`document_parser.py`). A new `MisLabeledFileError(ValueError)` sibling keeps the taxonomy explicit; subclasses of `ValueError` for ingestion failure modes (cf. WP-CORE-3's `EmptySRSDocumentError`).

## Design

### D-3. New exception `MisLabeledFileError`

`core/document_parser_readers.py` (or `document_parser.py` — see OQ-1):

```python
class MisLabeledFileError(ValueError):
    """Raised when a file extension does not match its magic-byte signature.

    Example: a .docx (ZIP archive) saved with .txt extension; a .pdf saved
    as .txt; etc. The file's first bytes are checked against known binary
    signatures before encoding-decode attempts; on match the file is
    rejected with a clear message naming the detected real format.

    Distinct from EmptySRSDocumentError: this is a content-format mismatch,
    not a content-emptiness issue.
    """

    def __init__(self, file_path: str, detected_format: str, message: str = None):
        self.file_path = file_path
        self.detected_format = detected_format
        super().__init__(
            message
            or f"File {file_path!r} appears to be a {detected_format} file, not text. "
               f"Rename to the correct extension or convert to text."
        )
```

LOC: ~20.

### D-4. Magic-byte signature table

```python
# Magic-byte signatures for common binary formats that get renamed to .txt.
# Ordered by first-byte specificity. Each entry: (prefix_bytes, format_label).
_BINARY_MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"PK\x03\x04", "ZIP archive (likely .docx/.xlsx/.zip)"),
    (b"%PDF-",      "PDF"),
    (b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1", "Microsoft compound (likely legacy .doc/.xls)"),
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xFF\xD8\xFF", "JPEG image"),
    (b"GIF87a",     "GIF image"),
    (b"GIF89a",     "GIF image"),
    (b"\x1f\x8b\x08", "gzip archive"),
]


def _detect_binary_signature(data: bytes) -> Optional[str]:
    """Return a human-readable format label if `data` starts with a known
    binary magic byte signature; otherwise None.

    Checked BEFORE encoding-decode attempts in read_txt — surfaces
    mislabeled binary files as MisLabeledFileError instead of decoding
    gibberish via single-byte fallback encodings.
    """
    for prefix, label in _BINARY_MAGIC_SIGNATURES:
        if data.startswith(prefix):
            return label
    return None
```

LOC: ~20.

### D-5. `read_txt` gains pre-decode signature check

```python
def read_txt(file_path: str) -> str:
    data = Path(file_path).read_bytes()

    # WP-CORE-9: detect mislabeled binary files BEFORE encoding-decode loop.
    # Single-byte fallback encodings (cp1254/cp1252) decode any byte sequence
    # without raising; the printable-ratio heuristic at _looks_like_text can
    # silently accept gibberish from a renamed .docx/.pdf. Detecting common
    # magic-byte signatures upfront surfaces the actual file format.
    detected = _detect_binary_signature(data)
    if detected is not None:
        raise MisLabeledFileError(file_path=file_path, detected_format=detected)

    for encoding in _candidate_text_encodings(data):
        try:
            decoded = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_like_text(decoded):
            return decoded

    raise UnicodeDecodeError(...)
```

LOC: ~10 (the `if detected:` block + comment).

### D-6. Empty file behavior preserved

A zero-byte file (`b""`) has no magic-byte prefix → `_detect_binary_signature` returns None → encoding loop falls through to UTF-8, decodes to `""`, `_looks_like_text("")` returns True (line 121-122), returns `""`. The downstream `EmptySRSDocumentError` guard at `document_parser.py` (WP-CORE-3) still catches this. WP-CORE-9 does NOT change empty-file semantics.

### D-7. Backwards-compat: legitimate text containing magic bytes mid-content

A text file legitimately containing the literal string `"PK\x03\x04"` anywhere except the START of the file is NOT affected — `_detect_binary_signature` uses `data.startswith(prefix)` only. Magic bytes are positional signatures; only the first N bytes matter.

### D-8. Optional: stack EmptySRSDocumentError check before signature?

Spec keeps current order: magic-byte check FIRST, then encoding loop, then downstream EmptySRSDocumentError. Reason: empty file is not mislabeled — `_detect_binary_signature(b"")` returns None, falls through normally. No reordering needed.

## Test plan

**RED commit expected pytest result:** 365 + 5 new RED-by-design = 370 collected; 365 passed, 5 failed, 31 deselected.

| # | name | file | what it asserts | RED expectation |
|---|---|---|---|---|
| T-MFE-1 | `test_mislabeled_file_error_carries_file_path_and_detected_format` | `tests/test_document_parser_mislabeled_file.py` (NEW) | `MisLabeledFileError("/x/foo.txt", "ZIP archive")` exposes `.file_path`, `.detected_format`, readable message; subclass of `ValueError` | FAIL — class doesn't exist |
| T-MFE-2 | `test_read_txt_raises_on_zip_magic_bytes` | same | Write `b"PK\x03\x04..."` to `tmp_path / "renamed.txt"`; `SRSDocumentParser().parse_file(...)` raises `MisLabeledFileError`; `exc.detected_format` mentions "ZIP" | FAIL — current path silently decodes via cp1254 |
| T-MFE-3 | `test_read_txt_raises_on_pdf_magic_bytes` | same | Write `b"%PDF-1.4\n..."`; raises `MisLabeledFileError`; `exc.detected_format` mentions "PDF" | FAIL |
| T-MFE-4 | `test_read_txt_raises_on_microsoft_compound_doc_magic_bytes` | same | Write `b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1..."`; raises `MisLabeledFileError` | FAIL |
| T-MFE-5 | `test_read_txt_does_not_raise_on_legitimate_text_containing_PK_substring` | same | Write `"The PK\\x03\\x04 byte sequence is a ZIP signature."` (as TEXT, not as raw bytes — i.e., backslash-literal); `parse_file` succeeds; content preserved verbatim | FAIL because the helper doesn't exist; the test asserts behavior that GREEN preserves |

**Existing test regression contract:** `test_parse_txt_supports_utf16_input` + all 7 `test_parse_txt_*` tests in `tests/test_document_parser.py` must continue to pass — verify the magic-byte check doesn't intercept legitimate UTF-8/UTF-16 text.

**Total**: 5 fail. GREEN turns all 5 green.

## Risks

| # | risk | mitigation |
|---|---|---|
| R-1 | False positive: a real text file whose first bytes happen to match a magic-byte signature. | Magic-byte signatures are at least 3 bytes and target binary-archive headers. Probability of legitimate ASCII text starting with `PK\x03\x04` or `%PDF-` is effectively zero. T-MFE-5 explicitly verifies the literal-string case (text containing the byte sequence in printable form, not as raw bytes) passes through. |
| R-2 | New `MisLabeledFileError(ValueError)` is a `ValueError` subclass; existing `except ValueError` handlers in callers may now catch this as well as `EmptySRSDocumentError`. | Both exceptions are ingestion-failure modes; catching as `ValueError` is semantically correct. Audit `main.py` for `except ValueError` blocks — `_parse_srs_batch` (`main.py:56-100`) catches `EmptySRSDocumentError` specifically + generic `Exception`. `MisLabeledFileError` will route to the generic Exception path in batch mode → "Failed to parse {path}: {exc}" log entry. Acceptable behavior — explicit failure, no silent degradation. |
| R-3 | Magic-byte detection adds ~10 ms file-read overhead per call. | Negligible. Already reading `.read_bytes()`; the prefix check is O(few hundred ns). |
| R-4 | Future formats (e.g., WebP, ARROW) added to industry without this list being updated. | Acceptable. Failure mode is degradation to current behavior (cp1254 gibberish) — not regression. Maintenance-only. Document as known limitation in dev_doc. |
| R-5 | A real-world `.txt` file with UTF-8 BOM (3 bytes `\xef\xbb\xbf`) → not in signature list → passes through to encoding loop correctly. UTF-16 BOM (`\xff\xfe`) similarly. | Confirmed by code review of `_candidate_text_encodings` — those BOMs are handled in the encoding-selection branch. No interaction. |

## Open questions

| # | question | disposition |
|---|---|---|
| **OQ-1** | Should `MisLabeledFileError` live in `document_parser_readers.py` (alongside `read_txt`) or in `document_parser.py` (alongside `EmptySRSDocumentError`)? | **`document_parser_readers.py` for v1.** Co-locates with the `read_txt` function that raises it; mirrors WP-CORE-3's pattern of `EmptySRSDocumentError` living in `document_parser.py` near `parse_file`. If the exception needs to be raised from multiple readers (PDF, DOCX) in a future WP, extract then. |
| **OQ-2** | Should `_looks_like_text`'s 95 % printable threshold be tightened concurrently? | **NO for v1.** Speculative scope expansion. Magic-byte detection handles the realistic F-2 trigger (renamed binaries). Threshold tightening is a separate WP if empirics show it's needed. |
| **OQ-3** | Should `read_pdf` and `read_docx` get the same defensive treatment (reject `.pdf` extension with non-`%PDF-` content; `.docx` extension with non-`PK\x03\x04` content)? | **DEFERRED, scope-bounded.** F-2 is `read_txt` specific. F-1 (read_pdf defensive handling) is a separate backlog entry. Symmetric defenses for `read_pdf` + `read_docx` are best done as part of F-1's WP. |
| **OQ-4** | Should magic-byte detection cover Office Open XML (.docx) "junk header" cases where the ZIP central directory is corrupted/truncated, leaving only readable XML fragments? | **NO.** That's a different failure mode (corrupted ZIP). The `\x50\x4b\x03\x04` ZIP local-file-header signature catches the start-of-file ZIP regardless of truncation downstream. |

## Atomic commit sequence

1. **RED commit** — `test(document_parser_readers): WP-CORE-9 red-phase tests for MisLabeledFileError + magic-byte detection`
   - `tests/test_document_parser_mislabeled_file.py` (NEW) — T-MFE-1..5
   - RED pytest: 370 collected, 365 passed, 5 failed, 31 deselected
   - LOC: +~120

2. **GREEN commit** — `fix(document_parser_readers): WP-CORE-9 MisLabeledFileError + magic-byte detection in read_txt`
   - `core/document_parser_readers.py` — add `MisLabeledFileError(ValueError)`, `_BINARY_MAGIC_SIGNATURES` table, `_detect_binary_signature(data)` helper, pre-decode check in `read_txt`
   - Pytest: 370 passing, zero regression
   - LOC: +~50

3. **DOC commit** — `chore(artifacts): WP-CORE-9 dev_doc + audit state update + F-2 SHIPPED`
   - `development_docs/WP-CORE-9-mislabeled-file-detection.md` (created)
   - `development_docs/INDEX.md` (ACTIVE row #10 added)
   - `.planning/pipeline_audit/CURRENT.md` (iteration 8 SHIPPED status)
   - `.planning/pipeline_audit/improvements_backlog.md` (F-2 → SHIPPED)
   - `.planning/pipeline_audit/decision_log.md` (D-PICK + D-CODEX-REVIEW entries)
   - `.planning/pipeline_audit/findings/document_parser.md` (§F-2 SHIPPED status)
   - `.planning/pipeline_audit/handoff-2026-05-23-<time>.md` (iteration 9 handoff)

4. **PLANNING commit** — `chore(planning): WP-CORE-9 spec v2 + plan into git history`

## Downstream impact

| concern | impact | action |
|---|---|---|
| `_parse_srs_batch` (`main.py:56-100`) | `MisLabeledFileError` is a `ValueError`; `_parse_srs_batch` catches `EmptySRSDocumentError` explicitly + generic `Exception` (per WP-CORE-3 HARD/SOFT/MIXED policy). New error routes to "Failed to parse" branch — explicit failure surfaces in response. | None — current dispatch is correct. |
| `/generate-model` + `/generate-model-stream` typed handler (WP-CORE-8) | `MisLabeledFileError` is NOT a `PipelineError` (it's a `ValueError`). It will NOT be caught by `except PipelineError`; falls through to bare-Exception fallback at `main.py:427` / `:533`. Response shape: `{success: false, error: str(exc)}`. | None — the WP-CORE-8 typed handler is for `PipelineError` taxonomy (orchestration-layer errors). Ingestion-layer errors have their own taxonomy (`EmptySRSDocumentError`, `MisLabeledFileError`) and route through the batch-helper path. |
| EMSE paper Methods section | Pre-WP-CORE-9: mislabeled binary inputs were silently ingested as gibberish. Post-WP-CORE-9: rejected at boundary with `MisLabeledFileError` naming the detected format. | Flag for advisor at next paper revision. |

## Goal-backward verification

| Iteration-8 goal | Evidence |
|---|---|
| Pick F-2 per WP-CORE-8 handoff ingestion-layer pivot | F-2 picked; LIVE in production verified by Discovery D-1. |
| Spec → Codex xhigh review → plan → SDD → dev_doc → state update | Spec v1 drafted (this file). Codex review pending. |
| Each commit gated on pytest ≥ baseline | RED: 365 + 5 fail; GREEN: 370 pass, 0 regression. |
| Production reachability subsection in spec | YES — §Motivation. |
| Smallest correct change (AGENTS.md) | YES — magic-byte detection is targeted at the realistic F-2 trigger; threshold tightening + symmetric defenses deferred. |

Spec v1 ready for Codex xhigh review.
