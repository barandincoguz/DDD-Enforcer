# WP-CORE-3 — Empty-input contract for `SRSDocumentParser.parse_file`

**Date:** 2026-05-21
**Owner:** Baran (autonomous pipeline-hardening loop, iteration 2)
**Status:** REVISED v2 — addressed Codex xhigh adversarial review (2 CRITICAL handled, 5 WARN handled inline)
**Parent:** `.planning/pipeline_audit/findings/document_parser.md` finding **F-3** (MAJOR)
**Loop:** Domain Pipeline Hardening Loop (second WP; baseline 305 confirmed at HEAD `3d13f26`)
**Sibling iteration 1:** WP-CORE-2 shipped at `25e6880` (reference-heading truncation); same file (`core/document_parser.py`)
**Codex consult:** review at runtime 2026-05-21 ~07:46; raw output preserved in `decision_log.md` entry `D-CODEX-REVIEW-WP-CORE-3`

---

## Revision history

- **v1 (draft)** — initial spec; sent to Codex xhigh for adversarial review.
- **v2 (this version)** — 2 CRITICAL + 5 WARN disposed:
  - **C-1 (mixed-batch behavior mislabeled):** Batch loops now **skip-and-continue** on `EmptySRSDocumentError` (not return). Post-loop check rewritten to test `srs_docs` emptiness (not `combined_text.strip()`, which was broken by separator inclusion). Aggregate "all-empty" failure preserved; mixed-batch (one empty + one good) succeeds.
  - **C-2 (call-site migration untested):** Added §"Wiring tests" T-WIRE-1..4: small extracted helper `_parse_srs_batch` becomes the wiring seam; the helper + `generate_domain_model` + `initialize_rag` get behavior tests via monkeypatched `parse_file`.
  - **W-1 (SOFT vs uncaught policy ambiguity):** New §"Per-path empty-input policy" table at top of §"Chosen approach" — HARD = propagate / explicit error, SOFT = skip+log+continue. No ambiguity.
  - **W-2 (LLM-layer precondition unstated):** New §"Scope and preconditions" — `parse_file` is the sole ingress for all `DomainArchitect.analyze_document(text=...)` calls in production; verified by grep evidence below. Direct LLM calls bypassing `parse_file` are out of scope (none exist in production code).
  - **W-3 (intermediate GREEN commit half-migrates):** Implementation order collapses GREEN + REFACTOR into one atomic behavior commit. Test-first discipline preserved by RED having both parser tests AND wiring tests (all red until GREEN lands the parser raise + helper + all 6 call-site updates together).
  - **W-4 (false AGENTS citation):** Logging policy rationale rewritten — cites the actual `main.py` convention (every progress signal uses `print`) rather than a nonexistent AGENTS.md rule.
  - **W-5 (overbroad acceptance grep):** Behavior acceptance criteria added: control-flow checks for `except EmptySRSDocumentError`, post-loop `srs_docs` test, and pytest assertions for the wiring tests. Grep checks retained as **secondary** cleanup verification, not primary acceptance.

---

## Motivation

`SRSDocumentParser.parse_file` (`extension/backend/core/document_parser.py:39-51`) returns a plain `str` and **never raises on empty content**. Every reader legitimately returns `""` for an empty input:

- PDF with all-empty pages → `_extract_pdf_page_text` joins empty per-page strings (`document_parser_readers.py`).
- DOCX with zero blocks → `"\n\n".join([])` → `""`.
- TXT with 0 bytes → `_looks_like_text` branch `if not text: return True` returns `""`.

After `_post_process` strips, the result is still `""`. The contract is therefore "may return empty string, never raises on empty input." Six call sites in `main.py` each re-implement the empty-string check, in three different shapes:

| line | shape | behavior on empty |
|---|---|---|
| `main.py:61` | `if not raw_text.strip(): raise ValueError("Document is empty…")` | HARD — sync `generate_domain_model` |
| `main.py:101` | `if raw_text.strip():` (positive guard, silent skip) | SOFT — `initialize_rag` |
| `main.py:326-330` | post-loop `if not combined_text.strip(): return {"success": False, "error": "All documents are empty…"}` | HARD (BROKEN) — `/generate-model` |
| `main.py:367` | `if raw_text.strip():` | SOFT — RAG re-index in `/generate-model` |
| `main.py:449-451` | same as `:326-330` but inside thread | HARD (BROKEN) — `/generate-model-stream` |
| `main.py:481` | `if raw_text.strip():` | SOFT — RAG re-index in `/generate-model-stream` |

**(BROKEN) annotation, new in v2:** The post-loop combined-text guards at `:326-330` and `:449-451` are dead code today. The loop appends `f"\n\n--- Document: {Path(file_path).name} ---\n\n"` (non-whitespace separator) **before** `raw_text` on every iteration — so even when every file parses to `""`, `combined_text` ends up containing only the separator headers, and `combined_text.strip()` returns the separator text, not `""`. The "all documents empty" branch is unreachable in current code. This is a latent bug uncovered by drafting WP-CORE-3, and folding its fix into the WP saves a duplicate audit cycle.

Three problems with the status quo:

1. **Leaky contract.** Six duplicated guards (across three behavioral shapes, two of them broken) are evidence the invariant belongs inside the parser. AGENTS.md "Stable entrypoints; isolate change-prone logic" applies.
2. **Silent data swallowing.** The SOFT branches (`:101`, `:367`, `:481`) silently drop an empty file from RAG indexing with no log, no exception, no metric. For an EMSE methodology run this is unrecorded behavior.
3. **Indistinguishable from upstream failures.** The two HARD branches use the same `"Document is empty or could not be parsed"` string for two genuinely different failure modes: file read failed vs. file read OK but empty. F-3 evidence: a `ValueError` in the loop currently masks a `FileNotFoundError` propagated by `parse_file:40-41` (it gets re-wrapped as "Failed to parse {name}: ...").

This is the second WP in the document_parser audit context; same file as WP-CORE-2, so the audit cache is hot. Smaller blast radius than priority-2 (`core/architect.py` close-lookup, 752 LOC); cohesive continuation per handoff `.planning/pipeline_audit/handoff-2026-05-21-0220.md` §"Rationale to prefer WP-CORE-3 = F-3."

---

## Scope and preconditions

**Sole ingress assumption:** `SRSDocumentParser.parse_file` is the only path that produces text passed to `DomainArchitect.analyze_document(text=...)` in production code. Verified by grep:

- `git grep -n "analyze_document" extension/backend/` in production code yields only `main.py:65, 337, 455` (the three call sites), each fed by `parse_file` upstream in the same function.
- Test code (`extension/backend/tests/`) may call `analyze_document` with synthetic text directly, but that bypasses production ingress and is outside the empty-input contract's scope.

Therefore: enforcing the empty-input invariant inside `parse_file` is sufficient to guarantee that no production call to `DomainArchitect.analyze_document` is fed empty text. If a future direct caller skips `parse_file`, that caller takes on the responsibility — out of scope for this WP.

**Direct callers of `parse_file`:** the six sites listed above. No others in production code (grep `git grep -n "parse_file" extension/backend/core extension/backend/main.py`).

---

## Per-path empty-input policy (response to W-1)

| call site | path class | post-change behavior on empty file |
|---|---|---|
| `main.py:61` (`generate_domain_model`, sync) | HARD | Exception propagates to caller (lifespan `except Exception` at `:136-141`) — explicit log, then continues with `app_state["domain_rules"] = {}` (unchanged outcome shape). |
| `main.py:101` (`initialize_rag`) | SOFT | `try/except EmptySRSDocumentError` → `print("[RAG] skip empty SRS: …")` → return empty RAG. |
| `main.py:326-330` (`/generate-model` batch) | MIXED | Per-file empty: skip + log + continue. Per-file non-empty exception: return error (preserves current behavior for read failures). All-empty aggregate: return `"All documents were empty after parsing"`. |
| `main.py:367` (RAG re-index in `/generate-model`) | SOFT | `try/except EmptySRSDocumentError` → `print("[RAG] skip empty SRS: …")` → `continue` in loop. |
| `main.py:449-451` (`/generate-model-stream` batch) | MIXED | Same as `/generate-model` batch — empty-skip + non-empty-error + all-empty-aggregate. |
| `main.py:481` (RAG re-index in `/generate-model-stream`) | SOFT | Same as `/generate-model` SOFT — skip + log + continue. |

Three classes: HARD (propagate), SOFT (catch + log + continue), MIXED (in-loop catch with per-file class disambiguation).

---

## Alternatives considered

### Alt A — Caller-only check (status quo + log line)

Add `print(f"[WARN] empty SRS: {path}")` at each soft-skip site; leave parser silent. **Rejected** because it makes the leak permanent and doubles the number of inconsistent guard shapes.

### Alt B — Parser raises only for hard sites; soft sites keep silent skip (rejected)

Two `parse_file` entry points: one raising, one not. **Rejected** because it codifies the leaky contract into the API. AGENTS.md "Stable entrypoints" rules this out directly.

### Alt C — Parser raises uniformly; all six callers updated to explicit per-path policy (chosen)

`SRSDocumentParser.parse_file` raises `EmptySRSDocumentError(ValueError)` after `_post_process` if the result is empty. All six call sites are updated per the per-path policy table above. The two broken post-loop guards (`:326-330`, `:449-451`) are **replaced** (not just deleted) with `if not srs_docs: return {"success": False, "error": "All documents were empty after parsing"}` — fixing the latent bug noted in §"Motivation."

`EmptySRSDocumentError` subclasses `ValueError` so any coarse `except ValueError` callers keep the same outcome — no surprise regression.

**Tradeoff:** Three new try/except blocks at RAG sites + two MIXED-class batch loops with double-clause `except` (one for `EmptySRSDocumentError`, one for `Exception`). Net code delta is roughly neutral. The change isolates the empty-input invariant into `core/document_parser.py` per "stable entrypoints, isolate change-prone logic."

### Alt D — Parser returns Optional[str] / Result type (rejected)

Forces every caller into a None-check ladder; the existing codebase has zero `Optional[str]` return types in `core/`. **Rejected** as overengineering and as inconsistent with the rest of `core/`.

### Alt E — Helper-extraction-only without parser raise (rejected)

Extract `_parse_srs_batch` helper at `main.py`, leaving parser silent. **Rejected** because the leaky contract remains at the parser layer; the helper only papers over the batch case and leaves the single-file `generate_domain_model` and SOFT RAG sites unchanged.

---

## Chosen approach (v2)

### (1) New exception class at top of `core/document_parser.py` (after imports, before `class SRSDocumentParser`)

```python
class EmptySRSDocumentError(ValueError):
    """Raised by ``SRSDocumentParser.parse_file`` when the post-processed
    text is empty.  Subclassing ``ValueError`` preserves coarse
    ``except ValueError`` callers; subclassing here means the empty-input
    failure is type-distinct from generic value errors when callers want
    to be explicit."""
```

**Rationale for IS-A `ValueError`:** preserves backwards compatibility for any `except ValueError` callers (lifespan has `except Exception`, so the choice is safe either way) and remains type-distinct for new explicit handlers.

### (2) Raise at end of `parse_file` at `:39-51`

```python
def parse_file(self, file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        raw = read_pdf(file_path)
    elif ext == ".docx":
        raw = read_docx(file_path)
    elif ext == ".txt":
        raw = read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    processed = self._post_process(raw)
    if not processed:
        raise EmptySRSDocumentError(
            f"Document parsed to empty content (post-processing): {file_path}"
        )
    return processed
```

`_post_process` already ends with `.strip()` (`:58`), so `if not processed:` covers whitespace-only inputs too.

### (3) New helper at module scope in `main.py` (above the `/generate-model` endpoint)

The helper centralizes the batch parsing+skip+aggregate-check logic that `/generate-model` and `/generate-model-stream` currently duplicate. This is the testable wiring seam called out by Codex C-2.

```python
def _parse_srs_batch(
    parser: SRSDocumentParser,
    file_paths: List[str],
) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse all SRS files for a batch endpoint.

    Returns (combined_text, srs_docs, error).  When ``error`` is not None,
    the caller MUST return it immediately (per-file read failure or
    all-empty aggregate).  Per-file ``EmptySRSDocumentError`` is logged
    and the file is skipped; partial batches succeed.
    """
    combined_text = ""
    srs_docs: List[Dict[str, Any]] = []
    for file_path in file_paths:
        try:
            raw_text = parser.parse_file(file_path)
        except EmptySRSDocumentError as exc:
            print(f"  ⚠️  Skipping empty document: {exc}")
            continue
        except Exception as exc:
            print(f"  ❌ Failed to parse: {exc}")
            return (
                "",
                [],
                {
                    "success": False,
                    "error": f"Failed to parse {Path(file_path).name}: {exc}",
                },
            )
        combined_text += f"\n\n--- Document: {Path(file_path).name} ---\n\n"
        combined_text += raw_text
        srs_docs.append({"path": file_path, "content": raw_text})

    if not srs_docs:
        return (
            "",
            [],
            {"success": False, "error": "All documents were empty after parsing"},
        )
    return combined_text, srs_docs, None
```

Imports added at top of `main.py`: `from typing import Tuple, Optional` (likely already present; verify) and `from core.document_parser import EmptySRSDocumentError, SRSDocumentParser` (replaces the existing import).

### (4) Six call-site updates at `extension/backend/main.py`

#### HARD — `main.py:55-62` (`generate_domain_model`)

Delete `if not raw_text.strip(): raise ValueError(...)`. Let `parse_file` raise `EmptySRSDocumentError` directly — it propagates to lifespan's `except Exception` (`:136-141`).

#### SOFT — `main.py:96-109` (`initialize_rag`)

```python
def initialize_rag(srs_files: List[str]) -> RAGPipeline:
    rag = RAGPipeline()
    if srs_files:
        srs_path = srs_files[0]
        doc_parser = SRSDocumentParser()
        try:
            raw_text = doc_parser.parse_file(srs_path)
        except EmptySRSDocumentError as exc:
            print(f"[RAG] skip empty SRS: {exc}")
            return rag
        filename = Path(srs_path).name
        ext = Path(srs_path).suffix[1:]
        chunk_count = rag.index_document(
            raw_text=raw_text, doc_id="srs_main", doc_name=filename, doc_type=ext
        )
        print(f"[RAG] Indexed {chunk_count} chunks from {filename}")
    return rag
```

#### MIXED — `main.py:305-330` (`/generate-model` batch)

Replace inline loop (`:307-330`) with:

```python
doc_parser = SRSDocumentParser()
combined_text, srs_docs, error = _parse_srs_batch(doc_parser, request.file_paths)
if error is not None:
    return error
print(f"  📊 Total combined text: {len(combined_text)} characters")
```

#### SOFT — `main.py:363-378` (RAG re-index inside `/generate-model`)

Wrap the per-file `parse_file` call:

```python
for file_path in request.file_paths:
    try:
        raw_text = doc_parser.parse_file(file_path)
    except EmptySRSDocumentError as exc:
        print(f"     [RAG] skip empty SRS: {exc}")
        continue
    filename = Path(file_path).name
    ext = Path(file_path).suffix[1:]
    chunk_count = rag.index_document(...)
    print(f"     -> Indexed {chunk_count} chunks from {filename}")
```

#### MIXED — `main.py:434-451` (`/generate-model-stream` batch, threaded)

Same as `/generate-model` MIXED — use the helper. Mapping: on `error is not None`, set `result_holder["error"] = error["error"]; return`.

#### SOFT — `main.py:476-487` (RAG re-index inside `/generate-model-stream`)

Same wrap pattern as `/generate-model` SOFT.

### (5) Logging policy (response to W-4)

Use `print` to match the existing module-wide convention: every progress signal in `main.py` already uses `print` (e.g., `:59 print(f"   -> Parsed document: {len(raw_text)} characters")`, `:107 print(f"[RAG] Indexed {chunk_count} chunks…")`, etc.). One line per empty-document skip, prefixed `[RAG] skip empty SRS: …` for SOFT sites and `⚠️  Skipping empty document: …` inside MIXED batch loops to match existing batch-loop emoji style (`📄`, `❌`). **Introducing the `logging` module is out of scope** — that is F-9, a separate finding.

---

## Test plan

All tests live in `extension/backend/tests/test_document_parser.py` (parser-level) and a new file `extension/backend/tests/test_main_wiring.py` (wiring-level, response to Codex C-2). The wiring tests are unit-scope (no live FastAPI, no real LLM, no real RAG), achieved by monkeypatching `parse_file`, `RAGPipeline`, and `DomainArchitect`.

### Parser-level tests (T-EMPTY-1..11, 11 new)

#### T-EMPTY-1 — PDF parsing to empty raises `EmptySRSDocumentError`

Monkeypatch `core.document_parser.read_pdf` to return `""`. Assert raises.

#### T-EMPTY-2 — DOCX parsing to empty raises `EmptySRSDocumentError`

Monkeypatch `core.document_parser.read_docx` to return `""`. Assert raises.

#### T-EMPTY-3 — TXT parsing to empty raises `EmptySRSDocumentError`

Real `tmp_path` 0-byte `.txt` file; no monkeypatching.

#### T-EMPTY-4 — Whitespace-only TXT raises `EmptySRSDocumentError`

Real `tmp_path` `.txt` containing only `"\n\n   \t\n"`. `_post_process` strips → `""`. Raises.

#### T-EMPTY-5 — IS-A `ValueError`

```python
with pytest.raises(ValueError):
    parser.parse_file(empty_txt_path)
```

#### T-EMPTY-6 — `FileNotFoundError` still raised for missing path (no regression)

Direct port of existing test.

#### T-EMPTY-7 — Unsupported extension still raises `ValueError(...)` (NOT `EmptySRSDocumentError`)

```python
tmp = tmp_path / "x.xyz"; tmp.write_text("anything")
with pytest.raises(ValueError, match="Unsupported file type"):
    parser.parse_file(str(tmp))
```

#### T-EMPTY-8 — Non-empty TXT returns text normally (no regression)

```python
tmp = tmp_path / "ok.txt"; tmp.write_text("Hello world.\n")
assert parser.parse_file(str(tmp)) == "Hello world."
```

#### T-EMPTY-9 — TXT with only `"References\n"` raises `EmptySRSDocumentError`

Demonstrates that post-truncation-empty (cross-WP integration with WP-CORE-2) is caught by the new contract.

#### T-EMPTY-10 — Exception message contains the file path (debuggability)

```python
with pytest.raises(EmptySRSDocumentError, match=re.escape(str(empty_txt_path))):
    parser.parse_file(str(empty_txt_path))
```

#### T-EMPTY-11 (parametrized) — Reader-level monkeypatching contract

Parametrize over `(extension, patch_target)`; covers the import-site monkeypatching trap (`core.document_parser.read_pdf`, not `core.document_parser_readers.read_pdf`).

### Wiring-level tests (T-WIRE-1..4, response to C-2)

New file: `extension/backend/tests/test_main_wiring.py`. Each test monkeypatches `core.document_parser.SRSDocumentParser.parse_file` (or imports `_parse_srs_batch` directly with a stub parser) so no real PDFs / LLMs / Chroma is involved.

#### T-WIRE-1 — `_parse_srs_batch` skips empty file and continues with non-empty

```python
class StubParser:
    def parse_file(self, p):
        if p == "empty.txt":
            raise EmptySRSDocumentError("empty.txt")
        return "good content"

combined, docs, err = _parse_srs_batch(StubParser(), ["empty.txt", "good.txt"])
assert err is None
assert len(docs) == 1
assert docs[0]["path"] == "good.txt"
assert "good content" in combined
```

#### T-WIRE-2 — `_parse_srs_batch` returns aggregate error when all files empty

```python
class StubParser:
    def parse_file(self, p):
        raise EmptySRSDocumentError(p)

combined, docs, err = _parse_srs_batch(StubParser(), ["a.txt", "b.txt"])
assert err is not None
assert err["success"] is False
assert "All documents were empty" in err["error"]
assert docs == []
assert combined == ""
```

#### T-WIRE-3 — `_parse_srs_batch` returns per-file error for non-empty parse failure

```python
class StubParser:
    def parse_file(self, p):
        raise FileNotFoundError(p)

combined, docs, err = _parse_srs_batch(StubParser(), ["missing.txt"])
assert err is not None
assert err["success"] is False
assert "Failed to parse missing.txt" in err["error"]
```

This locks in the **disambiguation** between empty-skip and read-error (a CRITICAL-1 behavior).

#### T-WIRE-4 — `initialize_rag` silently returns empty RAG on empty SRS

```python
import main

def fake_parse_file(self, path):
    raise EmptySRSDocumentError(path)

monkeypatch.setattr(
    "core.document_parser.SRSDocumentParser.parse_file", fake_parse_file
)

class FakeRAG:
    def __init__(self): self.indexed = []
    def index_document(self, **kw): self.indexed.append(kw); return 0

monkeypatch.setattr("main.RAGPipeline", FakeRAG)

rag = main.initialize_rag(["empty.txt"])
# RAG returned, no indexing performed
assert isinstance(rag, FakeRAG)
assert rag.indexed == []
```

This locks in the SOFT-path skip behavior (W-1 / C-2).

### Total: 11 parser-level + 4 wiring-level = **15 new tests**, target baseline **305 + 15 = 320 passed**, 31 deselected.

---

## Implementation order (response to W-3 — atomic GREEN)

1. **RED commit** (`test(parser): WP-CORE-3 empty-input contract — red-phase parser + wiring tests`)
   - Add all 11 T-EMPTY-* tests in `tests/test_document_parser.py`.
   - Add new file `tests/test_main_wiring.py` with all 4 T-WIRE-* tests.
   - Add `from core.document_parser import EmptySRSDocumentError` — expected `ImportError` at collection time.
   - `pytest -m "not integration"` → red.

2. **GREEN commit** (`fix(parser, main): WP-CORE-3 EmptySRSDocumentError — explicit empty-input contract`)
   - Add the `EmptySRSDocumentError` exception class.
   - Add the `parse_file` raise.
   - Add the `_parse_srs_batch` helper.
   - Update all six `main.py` call sites.
   - `pytest -m "not integration"` → **320 passed, 31 deselected**.

   Combined commit ensures the codebase is never in a state where the parser raises but the call sites are stale. Test-first discipline is preserved by the RED commit landing first.

3. **DOC commit** (`chore(artifacts): WP-CORE-3 dev_doc + audit state update`)
   - `development_docs/WP-CORE-3-empty-input-contract.md` created.
   - `development_docs/INDEX.md` ACTIVE table updated.
   - `.planning/pipeline_audit/improvements_backlog.md` — F-3 → SHIPPED.
   - `.planning/pipeline_audit/CURRENT.md` pointer updated.
   - `.planning/pipeline_audit/decision_log.md` — entry `D-EMPTY-INPUT-CONTRACT-2026-05-21` + `D-CODEX-REVIEW-WP-CORE-3`.

4. **PLANNING commit** (`chore(planning): WP-CORE-3 spec v2 + plan into git history`)
   - This spec + the plan file.

5. **HANDOFF commit** (`chore(planning): WP-CORE-3 iteration 2 handoff + CURRENT pointer`)

---

## Risks and concerns

### R-1 — Lifespan startup behavior on empty SRS

If the only file in `inputs/` is empty, `generate_domain_model` raises `EmptySRSDocumentError` at `parse_file` (was: explicit `ValueError("Document is empty…")` at `:62`). Both old and new land in `except Exception` at `:136-141` → traceback printed, `app_state["domain_rules"] = {}`. **Net behavior unchanged.**

### R-2 — Test isolation: monkeypatch at import site

T-EMPTY-1, T-EMPTY-2, T-WIRE-* monkeypatch module-level functions. Must monkeypatch at the import-site (`core.document_parser.read_pdf`, `main.RAGPipeline`) not at the source — because the importing module binds the names locally. Spec is explicit about this in T-EMPTY-11 and T-WIRE-4.

### R-3 — Coarse `except Exception` in lifespan masks the new exception type

Lifespan `except Exception` will catch `EmptySRSDocumentError` and log as generic "Generation failed." Acceptable for WP-CORE-3 (the new message at least includes "empty content" + file path); out of scope for F-9 follow-up.

### R-4 — `EmptySRSDocumentError IS-A ValueError` may surprise reviewers

Documented in §"Chosen approach" (1). Chosen specifically to preserve `except ValueError` backwards compatibility. Future `DomainParserError` hierarchy is a separate WP if needed.

### R-5 — Batch behavior changes (response to C-1)

**v1 said "no change to batch atomicity" — wrong.** Reality:
- **Before:** Mixed batch (1 empty + 1 good) silently included the empty file's separator-only content into `combined_text`; the broken post-loop check never tripped. Effective behavior: garbage in, garbage out, no explicit "we processed an empty file" signal.
- **After:** Mixed batch skips empty files with a `print` log; aggregate check based on `srs_docs` (not the broken `combined_text.strip()`) returns "All documents were empty after parsing" only when truly all empty. Cleaner, more observable, fixes the latent post-loop-check bug.
- **Read-error atomicity preserved:** A `FileNotFoundError` or other non-empty exception still kills the batch immediately, as today.

**This is a behavior change, not a behavior preservation.** Documented above + tested by T-WIRE-1, T-WIRE-2, T-WIRE-3.

### R-6 — `_post_process` could in theory return non-empty whitespace

`_post_process:53-58` ends with `.strip()`, so `processed` is empty iff every non-whitespace character was removed. Sufficient.

### R-7 — No test for "PDF-with-pages-but-all-blank" case

T-EMPTY-1 monkeypatches `read_pdf` to return `""` directly; does not exercise per-page-blank join logic. The per-page concern is `document_parser_readers.py` territory (F-1; out of scope).

### R-8 — `_parse_srs_batch` helper introduces a new module-scope symbol in `main.py`

This is a minimal extraction (~25 LOC). It's tested directly via T-WIRE-1..3. It does **not** widen the public API surface — `main.py` is not imported as a library; it's the FastAPI entrypoint. Helper visibility prefixed with `_` to signal module-private intent.

### R-9 — Stream endpoint threading

`/generate-model-stream` runs the batch parsing inside a `threading.Thread` (`:427-451`). The helper extraction works inside the thread the same way (no shared mutable state in the helper itself; `print` is thread-safe per CPython GIL). The error path uses `result_holder["error"] = error["error"]` instead of `return error` — wiring detail in §"Chosen approach" (4) MIXED for `/generate-model-stream`.

### R-10 — Stub-based wiring tests vs. integration tests

T-WIRE-* tests use stubs/monkeypatching, not real FastAPI / real RAG. Trade-off: faster, no transitive deps in the unit-test pool; cost: cannot catch a request-shape regression. Acceptable because integration tests (`pytest -m integration`) are the canonical place for endpoint-shape coverage, and they remain green (no changes to request/response shapes here).

---

## Acceptance criteria (response to W-5 — behavior-aware)

### Behavior (primary)

- `pytest -m "not integration"` baseline goes from **305 → 320 passed, 31 deselected** (15 new tests).
- Zero regression in existing 305 tests.
- T-WIRE-1: mixed batch returns no error, contains the non-empty file.
- T-WIRE-2: all-empty batch returns aggregate error with substring `"All documents were empty"`.
- T-WIRE-3: per-file `FileNotFoundError` returns batch error with substring `"Failed to parse"` (disambiguation locked).
- T-WIRE-4: `initialize_rag` returns without `index_document` being called.
- `pyright` passes (no new typing errors); `EmptySRSDocumentError` typed as a subclass of `ValueError`; `_parse_srs_batch` typed with `Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]`.

### Cleanup verification (secondary, after behavior gates pass)

- `git grep "raw_text.strip()" extension/backend/main.py` returns 0 results (all six sites migrated).
- `git grep "All documents are empty or could not be parsed" extension/backend/main.py` returns 0 results (both broken post-loop guards replaced).
- `git grep "Document is empty or could not be parsed" extension/backend/main.py` returns 0 results (sync guard at `:62` deleted).
- `git grep -n "except EmptySRSDocumentError" extension/backend/main.py` returns ≥ 4 matches (helper + 3 SOFT sites).

### Process

- All commits atomic, Conventional, with `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` trailer.
- No `git push` until explicit user instruction.
- Each commit's pytest must be ≥ baseline (RED commit: explicit red OK; GREEN onward: ≥ 320).

---

## File touch list

| file | type | net delta |
|---|---|---|
| `extension/backend/core/document_parser.py` | source | +8 LOC (exception class + raise) |
| `extension/backend/main.py` | source | net ~+15 LOC (helper +25, deleted post-loop guards -6, deleted sync guard -2, simplified RAG sites -2) |
| `extension/backend/tests/test_document_parser.py` | test | +~120 LOC (11 new T-EMPTY-* tests) |
| `extension/backend/tests/test_main_wiring.py` | test | new file, +~80 LOC (4 T-WIRE-* tests) |
| `development_docs/WP-CORE-3-empty-input-contract.md` | doc | new |
| `development_docs/INDEX.md` | doc | +1 ACTIVE row |
| `.planning/pipeline_audit/improvements_backlog.md` | audit | F-3 status flip |
| `.planning/pipeline_audit/CURRENT.md` | audit | pointer update |
| `.planning/pipeline_audit/decision_log.md` | audit | +2 entries (Codex review + decision) |
| `docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md` | planning | this file (v2) |
| `docs/superpowers/plans/2026-05-21-wp-core-3-empty-input-contract.md` | planning | new |
| `.planning/pipeline_audit/handoff-*.md` | handoff | new |

---

## Resolved questions (from v1)

1. **`IS-A ValueError` vs new hierarchy?** **Decision:** IS-A `ValueError`; smallest-correct-change; preserves existing coarse handlers; new hierarchy is a speculative generalization (rejected, see Alt D).
2. **Empty check before vs after `_post_process`?** **Decision:** After. Catches post-truncation-empty (T-EMPTY-9) for free.
3. **SOFT-path propagate vs skip?** **Decision:** Skip + log. RAG init must remain best-effort (matches existing `app_state["rag"] = None` on init failure pattern at `:381`). Per-path policy table makes this explicit.
4. **Batch atomicity?** **Decision: CHANGED.** Skip empty files + continue (was: silent inclusion of empty separators). Aggregate "all empty" still returns error. Explicit per-file empty-skip is observable via `print` log. v1's claim of "no change" was incorrect (the post-loop guard was already broken; mixed-batch behavior today is effectively "include the empty as garbage"). New behavior is **strictly better** observably + cleaner failure modes.
5. **Edge case where empty-after-strip is desired?** **Decision: no such case in this codebase.** Any document stripping to empty post-`_post_process` is either corrupt, mismatched format, or a truncation artifact — all conditions where loud failure is the correct response.

---

## Open questions for v2 reviewer (if any)

(none — all v1 questions resolved; v2 is implementation-ready)

---

## References

- Parent finding: `.planning/pipeline_audit/findings/document_parser.md` §F-3
- Iteration 1 (cohesive context): WP-CORE-2 spec at `docs/superpowers/specs/2026-05-21-wp-core-2-reference-truncate-fix-design.md`
- Handoff (re-entry ritual): `.planning/pipeline_audit/handoff-2026-05-21-0220.md`
- AGENTS.md rules cited: "Stable entrypoints", "no permissive fallbacks during development", "Error handling: explicit failure"
- CLAUDE.md: TDD convention extended from `core/llm/` to this loop
- Codex review: D-CODEX-REVIEW-WP-CORE-3 in `decision_log.md` (post-DOC commit)
