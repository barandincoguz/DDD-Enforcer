# Close-Lookup Findings — core/document_parser.py + core/document_parser_readers.py

**Auditor:** general-purpose subagent
**Date:** 2026-05-21
**Files audited:** extension/backend/core/document_parser.py (LOC=182), extension/backend/core/document_parser_readers.py (LOC=127)
**Method:** Read every line; cross-reference downstream consumers (main.py, architect.py) and existing tests (test_document_parser.py, test_unit.py); check AGENTS.md/CLAUDE.md conventions.

## Summary

`SRSDocumentParser` is the single SRS-ingestion entry-point feeding every Scout/Architect run; it dispatches on file extension to one of three readers and post-processes with a five-step lexical pipeline (normalize → references-truncate → clean-lines → merge-wrapped-lines → re-normalize). Readers cover PDF (pypdf, layout mode with TypeError fallback), DOCX (python-docx, traverses block-level paragraphs/tables) and TXT (BOM/heuristic encoding sniff with `_looks_like_text` printability ratio). The most concerning categories are: (a) silent permissive fallbacks that emit empty strings instead of raising on unreadable inputs — directly violating AGENTS.md "no silent degradation", (b) PDF-only edge cases that crash without context (encrypted, image-only, malformed) because `read_pdf` performs zero defensive checks, and (c) a TOC heuristic anchored to the first 120 lines that is fragile under layout-mode PDFs and silently lets the TOC bleed into Scout input. Test coverage is thin: 6 happy-path tests plus 1 FileNotFoundError; every error branch, encrypted/image-PDF path, and DOCX edge case is untested. Parser is currently stateful (compiled regexes on `__init__`) and re-instantiated in three loop sites in `main.py:307/435/479`; thread-safety is fine but observability is nonexistent — no logger anywhere in either module.

## Findings (numbered, severity-tagged)

### F-1 — `read_pdf` has no defensive handling for encrypted/image-only/malformed PDFs — MAJOR

**Component:** document_parser_readers.py
**Evidence:** `extension/backend/core/document_parser_readers.py:16-19` (3-line body, no try/except, no `reader.is_encrypted` check); `extension/backend/core/document_parser_readers.py:22-27` (only `TypeError` is caught inside the per-page helper)
**Observation:** `PdfReader(file_path)` raises `pypdf.errors.PdfReadError` on malformed PDFs and `pypdf.errors.FileNotDecryptedError` on encrypted PDFs at access time; neither is caught or converted to a domain-appropriate exception. For an image-only PDF (scan with no OCR), every page returns an empty string and `read_pdf` returns `""` — which then propagates to `main.py:61 "Document is empty or could not be parsed."` losing the distinction between "user dropped a blank file" and "user dropped a scanned PDF". `_extract_pdf_page_text` catches only `TypeError` (older pypdf without `extraction_mode` kwarg) — any `PdfReadError`/`PdfReadWarning`/`DependencyError` from a page propagates up uncaught and aborts the whole document. There is also no `is_encrypted` short-circuit, so a password-protected PDF will surface as a low-level pypdf error from deep inside, not a clean "encrypted file" diagnostic.
**Blast radius:** PIPELINE — the lifespan handler at `main.py:128-141` swallows the exception, sets `app_state["domain_rules"] = {}`, and the backend boots in a silently-broken state.
**Test gap:** yes — no encrypted PDF, no image-only PDF, no truncated PDF, no zero-page PDF test exists.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Error handling: explicit failure. No silent degradation, no permissive fallbacks during development." (CLAUDE.md "Conventions" §error handling)

### F-2 — `read_txt` silently emits binary garbage when `_looks_like_text` heuristic passes a near-binary file — MAJOR

**Component:** document_parser_readers.py
**Evidence:** `extension/backend/core/document_parser_readers.py:92-109` and `_looks_like_text` at `:120-126`
**Observation:** The `_looks_like_text` predicate accepts any decoded string whose printable+whitespace ratio is ≥ 0.95, and an empty decoded result returns `True` (line 122). cp1254/cp1252 are single-byte encodings — virtually any byte stream decodes successfully under them, so a malformed `.txt` (e.g. a renamed `.docx` or a binary blob) will be accepted under cp1254 if it happens to contain ≥ 95 % printable characters in that codepage. The function silently returns the gibberish string instead of surfacing the encoding mismatch. The 0.95 threshold is also unconfigurable and undocumented. NUL-byte rejection (line 123-124) is the only structural check; everything else relies on a magic ratio.
**Blast radius:** PIPELINE — gibberish text reaches Scout, wastes Gemini calls, may pollute the domain model silently.
**Test gap:** yes — no test asserts that mis-encoded or binary-renamed .txt files surface a clear error; `test_parse_txt_supports_utf16_input` is the only encoding test.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Silent fallbacks / permissive defaults"; CLAUDE.md "Things to Know" §error handling policy.

### F-3 — `SRSDocumentParser.parse_file` returns empty string for empty inputs instead of raising — MAJOR

**Component:** document_parser.py
**Evidence:** `extension/backend/core/document_parser.py:32-44` (no empty-content check post-read); `_post_process` at `:46-51` strips and returns; downstream guard lives only in `main.py:61`, `:101`, `:326`, `:449`
**Observation:** Every reader on an empty input legitimately returns `""` (PDF with all-empty pages → empty join; DOCX with no blocks → `"\n\n".join([])` = `""`; TXT with 0 bytes → `b""` decodes successfully to `""` via `_looks_like_text`'s `if not text: return True` branch). `parse_file` happily returns that empty string. The contract is therefore "may return empty string, never raises on empty input." Four call sites in `main.py` repeat the same `if not raw_text.strip(): raise ...` guard — a smell that the invariant belongs inside `parse_file`. Worse, `main.py:307-323` only catches per-file failure but happily concatenates an empty `raw_text` into `combined_text` (line 316) if a non-supported but readable file slipped past validation.
**Blast radius:** MODULE+PIPELINE — every consumer must re-implement the empty-string check; failures look identical to "file not found upstream" once the guard fires.
**Test gap:** yes — no test for empty-file behavior across the three reader types.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Stable entrypoints" + "no permissive fallbacks during development"; the contract is leaky.

### F-4 — TOC heuristic is anchored to first 120 raw lines and `cluster < 2` threshold; layout-mode PDFs leak TOC into Scout — MAJOR (uncertain)

**Component:** document_parser.py
**Evidence:** `extension/backend/core/document_parser.py:81-101` (`_find_toc_line_indexes`); `:103-117` (`_flush_toc_cluster`); regex at `:17-20`
**Observation:** The dot-leader regex requires at least four `.` characters (`\.{4,}`) and a trailing line number, then only collects matches in the first 120 lines, and only flushes a cluster of `len ≥ 2`. PDFs rendered in pypdf layout mode (`extraction_mode="layout"` at line 24) often replace dot leaders with multiple spaces or tabs (which `_normalize_line` at `:121` *normalizes to single spaces or `|`* before the TOC regex sees it — meaning a TOC that survived as `"Section 3 ........ 14"` in plain mode becomes `"Section 3 14"` and never matches). The first-120-line window also assumes the document starts with a cover/TOC; for an SRS with an executive summary first, the TOC sits beyond line 120 and is silently kept in the Scout input. Additionally, a single-row TOC slips through (`< 2` is dropped). I mark this MAJOR-uncertain because Codex consult could clarify whether downstream Scout filtering compensates; from the architect.py:267-308 prompt-driven extraction I read, Scout has no TOC-stripping logic of its own.
**Blast radius:** PIPELINE — directly affects domain-model quality for the EMSE paper; TOC entries become "domain sentences" and pollute the bounded-context graph.
**Test gap:** yes — `test_document_parser.py` has no TOC-detection test; the only TOC-adjacent assertion is the negative "References" truncation test.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Smell mixing of concerns" — the heuristic is undocumented and untested; CLAUDE.md §"Truncation is head + tail" implies head/tail logic only at architect.py:66, not here.

### F-5 — `_truncate_at_references` matches the first line containing "references" — including a numbered Section 3 with "References" inside the heading — MAJOR

**Component:** document_parser.py
**Evidence:** `extension/backend/core/document_parser.py:9-12` (regex), `:60-65` (loop)
**Observation:** The regex permits an optional markdown-style `#`, an optional section number, and then `references|bibliography|kaynakça` as the whole word match. A legitimate SRS section like `3.4 References to External Systems` will NOT match (the regex anchors to `\s*$` after the keyword), but a section titled exactly `3.4 References` (numbered references chapter common in IEEE/ISO templates) WILL match and truncate. More importantly, the regex matches Turkish `kaynakça` but not the much more common Turkish form `Kaynaklar` (plain plural) — meaning a Turkish-language SRS using `Kaynaklar` as the bibliography header will be ingested in full including the bibliography, while the same SRS using `Kaynakça` will be truncated. This is a locale-dependent silent behavior split. The truncation is also greedy in time (first match wins) — an inline mention transformed by `_normalize_line` cannot match because the regex anchors to start-of-line, but a heading like `## References` (markdown) on page 2 (false positive — perhaps a forward reference in a chapter) will silently kill the rest of the document.
**Blast radius:** PIPELINE — silent content loss; impossible to detect without a content-diff baseline.
**Test gap:** yes — `test_parse_txt_does_not_truncate_regular_requirement_lines` covers the no-truncate case, but no test asserts `Kaynaklar` vs `Kaynakça` symmetry, and no test for false-positive mid-document truncation.
**AGENTS/CLAUDE rule cited:** AGENTS.md §"language-specific assumptions"; CLAUDE.md §"D2 — 3 industries" implies multi-language SRS support is implicit.

### F-6 — `_should_merge` collapses sentence-final-punctuation rule but joins quote-terminated and bracket-terminated lines silently — MINOR

**Component:** document_parser.py
**Evidence:** `extension/backend/core/document_parser.py:154-165` (`_should_merge`); `:163` regex `[.!?;:]$`
**Observation:** The merge predicate considers only Latin sentence terminators. A line ending in `"` (closing quote), `)`, `]`, `…` (Unicode ellipsis), `”` (curly quote), or Turkish suspension `…` will be merged into the next paragraph. SRS documents quoting acceptance criteria like `"... shall not exceed 5%"` followed by a new paragraph get glued together. Also the hyphen-wrap detection at `:155` does not distinguish a soft-hyphen (typesetter line-break dash) from a compound-word hyphen — `"customer-`\n`order"` collapses to `customerorder` rather than `customer-order` (the line-169 strip removes the trailing hyphen). This is a corpus-quality concern but not data-loss-on-failure.
**Blast radius:** LOCAL — corpus noise only.
**Test gap:** yes — no test for hyphenated compound words or quote-terminated lines.
**AGENTS/CLAUDE rule cited:** AGENTS.md "regex / content-filter brittleness".

### F-7 — DOCX reader has no try/except around `docx.Document(file_path)` — MINOR

**Component:** document_parser_readers.py
**Evidence:** `extension/backend/core/document_parser_readers.py:30-31`
**Observation:** `docx.Document(...)` raises `docx.opc.exceptions.PackageNotFoundError` on a non-DOCX file masquerading with a `.docx` extension, `KeyError` / generic exceptions on corrupted ZIP containers, and python-docx has known issues with documents containing OLE objects, embedded media, or unusual content types. Unlike PDF where `_extract_pdf_page_text` at least catches `TypeError`, the DOCX path has zero defensive handling — any malformed `.docx` aborts with a python-docx-internal traceback. The block iterator at `:48-53` also silently drops `CT_SectPr`, `CT_Bookmark`, and section-end nodes (probably fine, but undocumented). The table-cell deduplication via `id(cell._tc)` at `:80-83` relies on Python's `id()` being stable for the lifetime of the iteration — true today but a brittle invariant for documents with merged cells across rows.
**Blast radius:** LOCAL.
**Test gap:** yes — no test for corrupted DOCX, no test with merged-cell tables, no test for nested tables.
**AGENTS/CLAUDE rule cited:** AGENTS.md "Error handling: explicit failure".

### F-8 — No XXE / external-entity hardening on DOCX XML parsing — MINOR (uncertain)

**Component:** document_parser_readers.py
**Evidence:** `extension/backend/core/document_parser_readers.py:31` (default `docx.Document` constructor)
**Observation:** python-docx delegates XML parsing to lxml under the hood. By default, lxml's parsers do NOT resolve external entities, so XXE is not exploitable in stock python-docx. However, the project ships no XML-hardening config nor documents the assumption. For an EMSE submission where third-party SRS documents are accepted, an audit reviewer may flag the absence of an explicit `XMLParser(resolve_entities=False, no_network=True)` invocation as a missing defense-in-depth. I mark this MINOR-uncertain — a Codex consult could verify the current pypdf/python-docx default postures against the lockfile.
**Blast radius:** REPO — security review surface for the paper.
**Test gap:** n/a — security-config check, not a behavioral test.
**AGENTS/CLAUDE rule cited:** AGENTS.md §"Security boundary".

### F-9 — No logging anywhere in either module — observability gap — MINOR

**Component:** document_parser.py and document_parser_readers.py
**Evidence:** No `import logging`, no `logger = logging.getLogger(...)`, no `print(...)` either. All progress reporting lives at the caller (`main.py:59 print(f"   -> Parsed document: {len(raw_text)} characters")` etc.).
**Observation:** The parser is a critical step in the WP-NEW-B run manifest (per CLAUDE.md), but it emits no structured signal — no page count, no character count, no truncation event count, no TOC-cluster count, no encoding detected, no fallback triggered. The PDF fallback at `_extract_pdf_page_text:25-27` silently downgrades layout-mode to plain extraction and the caller has no way to know it happened. For EMSE reproducibility (`runs/probe-{ts}.manifest.json`), this is a methodology gap: the manifest cannot record what the parser did because the parser does not say.
**Blast radius:** PIPELINE — methodology / reproducibility cost for the paper.
**Test gap:** n/a — observability concern.
**AGENTS/CLAUDE rule cited:** CLAUDE.md §"Persistent Development Memory" + AGENTS.md §"Logging policy: silent (no logger) or verbose".

### F-10 — Parser re-instantiated per loop iteration; regex compilation pays cost on every call — TRIVIAL

**Component:** document_parser.py and call sites
**Evidence:** `extension/backend/core/document_parser.py:8-30` (`__init__` compiles 6 regexes); call sites `main.py:307` and `main.py:435` instantiate once but `main.py:366` and `main.py:480` re-parse the same file twice in the same request (once for Architect input, once for RAG indexing — duplicate I/O and duplicate post-processing).
**Observation:** `_extract_pdf_page_text`-level recompute is fine in single-document mode, but `main.py` parses each file twice (once for `combined_text`, once for RAG indexing) — at `main.py:366` and `:480`. For a 50 MB PDF this is a ~2x cost on the largest request type. The parser itself doesn't memoize. Regexes are at least instance-level (not function-level), which is correct.
**Blast radius:** LOCAL — perf only.
**Test gap:** n/a.
**AGENTS/CLAUDE rule cited:** AGENTS.md §"Performance smells".

## Convention notes

- Type hints are complete and accurate; no `Optional` masking a silently-returned `None`.
- File sizes (182 LOC / 127 LOC) are well within the ~500 LOC sweet spot from AGENTS.md.
- The reader/post-processor split is clean: `document_parser.py` owns content shaping; `document_parser_readers.py` owns I/O. This matches the "isolate change-prone logic" rule.
- `FileNotFoundError` raised early at `parse_file:33-34` is the one explicit-failure path that follows the AGENTS.md charter correctly.
- `ValueError(f"Unsupported file type: {ext}")` at `parse_file:43` is also correct explicit-failure.
- Regex patterns are pre-compiled on `__init__` — efficient, instance-bound.

## Anomalies

- `LIST_ITEM_PATTERN` is defined at module level in `document_parser_readers.py:13` and also inside `SRSDocumentParser.__init__` at `document_parser.py:28-30`. Two copies of the same regex in two different modules — drift risk if one is updated and the other is not.
- `read_txt`'s `UnicodeDecodeError` reraise at `:103-109` synthesizes a new exception with `"document_parser"` as the encoding label (positional arg). This is legal but misleading — debuggers and CI reports will display `encoding='document_parser'` which has no semantic meaning.
- `_extract_pdf_page_text:27` returns `text if text.strip() else (page.extract_text() or "")` — a third extraction attempt that *re-calls* `page.extract_text()` even after the first call returned text. This is dead-ish: it only fires when `extraction_mode="layout"` returned whitespace-only content, but it re-runs the same default-mode call that the `TypeError` branch would have used — meaning on a pypdf that supports `extraction_mode` and returned whitespace-only layout, we call default mode once; on the older pypdf path (TypeError) we already used default mode, so the post-check runs default mode a *second* time redundantly.
- `_find_toc_line_indexes:82` hardcodes `min(len(raw_lines), 120)` — magic number, no constant, no comment explaining why 120.
- Per CLAUDE.md "Local `.venv` is currently broken" — the docx/pypdf import resolution may differ between local and CI; no version pin appears in the modules themselves (relies on `requirements.lock`).

## Test-coverage map

| code path | test exists | test file:line | gap notes |
|---|---|---|---|
| `parse_file` → `.pdf` happy path | yes | `tests/test_document_parser.py:81-97` | layout-mode-only PDF; no encrypted/image-only/malformed coverage |
| `parse_file` → `.docx` happy path | yes | `tests/test_document_parser.py:100-120` | no corrupted DOCX, no merged-cell tables, no embedded media |
| `parse_file` → `.txt` happy path | yes | `tests/test_document_parser.py:123-129`, `tests/test_unit.py:409-421` | utf-16 covered; no cp1254 Turkish, no BOM-only, no binary-renamed |
| `parse_file` → unsupported ext (`ValueError`) | no | — | line 43 untested |
| `parse_file` → nonexistent file | yes | `tests/test_document_parser.py:146-150`, `tests/test_unit.py:423-430` | covered |
| `_extract_pdf_page_text` `TypeError` branch | no | — | older-pypdf path untested |
| `_extract_pdf_page_text` whitespace-only retry | no | — | dead-leg never exercised |
| `_is_list_paragraph` numbered-list branch | no | — | only bullet-list tested |
| `_extract_docx_table` merged-cell dedup | no | — | `id(cell._tc)` invariant untested |
| `read_txt` BOM utf-8-sig path | no | — | `:113-114` untested |
| `read_txt` cp1254 / cp1252 fallback | no | — | Turkish single-byte path untested |
| `read_txt` `_looks_like_text` rejection (NUL byte) | no | — | line 123-124 untested |
| `read_txt` `UnicodeDecodeError` reraise | no | — | line 103-109 untested |
| `_truncate_at_references` Turkish `Kaynakça` | no | — | line 10 untested |
| `_truncate_at_references` Turkish `Kaynaklar` (negative) | no | — | locale-asymmetry untested |
| `_find_toc_line_indexes` cluster < 2 single TOC entry | no | — | line 109 untested |
| `_find_toc_line_indexes` TOC beyond line 120 | no | — | window assumption untested |
| `_merge_wrapped_lines` hyphen-wrap collapse | partial | `tests/test_document_parser.py:81-97` | tests merge; no soft-hyphen-vs-compound test |
| `_should_merge` quote/bracket terminators | no | — | line 163 regex incomplete |

## Cross-references

- `extension/backend/main.py:57-58` — Lifespan domain-model bootstrap; consumes raw_text via the silent empty-string contract.
- `extension/backend/main.py:307-324` — `/generate-model` endpoint; catches per-file exceptions but propagates empty strings into `combined_text`.
- `extension/backend/main.py:366` and `:480` — Same file is re-parsed for RAG indexing (duplicate I/O — see F-10).
- `extension/backend/main.py:435-447` — `/generate-model-stream` SSE handler; same pattern as `:307`.
- `extension/backend/core/architect.py:182-247` — Scout consumes the raw_text via `analyze_document → extract_domain_sentences`. Chunk size is fixed at 10000 chars (`:196`). Scout has no TOC filter of its own — it trusts the parser entirely.
- `extension/backend/core/architect.py:249-265` — `_split_text_into_chunks` breaks at `.` boundaries; depends on the parser preserving sentence-terminator punctuation, which `_should_merge` at `document_parser.py:163` correctly preserves but `_merge_wrapped_lines` could elide in pathological cases.
- `extension/backend/tests/test_document_parser.py` — 6 tests total; all happy-path except the FileNotFoundError test.
- `extension/backend/tests/test_unit.py:406-430` — Duplicate `TestDocumentParser` class with 2 tests already covered by `test_document_parser.py`. Redundant.
- `extension/backend/tests/test_p3_integration.py:23-24` — Treats the parser as a fixture; integration test only, no error-branch coverage.
