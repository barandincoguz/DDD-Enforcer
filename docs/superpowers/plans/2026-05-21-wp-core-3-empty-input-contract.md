# WP-CORE-3 Empty-Input Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the leaky empty-input contract from `SRSDocumentParser.parse_file` (six duplicated guards across `main.py`) by raising a new `EmptySRSDocumentError(ValueError)` and migrating all six call sites to explicit per-path policy (HARD propagate, SOFT skip+log, MIXED batch via helper). Fixes a latent post-loop-guard bug in batch endpoints along the way.

**Architecture:** Two edits in `core/document_parser.py` (new exception class + raise at end of `parse_file`); one new module-scope helper `_parse_srs_batch` in `main.py`; six call-site updates in `main.py`. Tests: 11 parser-level T-EMPTY-* + 4 wiring-level T-WIRE-* (new test file `tests/test_main_wiring.py`).

**Tech Stack:** Python 3.13 (local dev), Python 3.12 (CI), `pytest` `-m "not integration"`, `monkeypatch` fixture, `tmp_path` fixture, `pyright` strict on `core/`.

**Spec:** `docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md` (v2 — Codex xhigh reviewed; 2 CRITICAL + 5 WARN handled)
**Audit finding:** `.planning/pipeline_audit/findings/document_parser.md` F-3 (MAJOR)
**Pre-WP HEAD:** `3d13f26`
**Pre-WP pytest baseline:** 305 passed, 31 deselected
**Target pytest baseline post-GREEN:** 320 passed, 31 deselected (15 new tests)

---

## File Structure

| file | role | change type |
|---|---|---|
| `extension/backend/core/document_parser.py` | SRS parser entry point + post-processing pipeline. Owns `parse_file` and the empty-input invariant. | MODIFY (~8 LOC: exception class + raise) |
| `extension/backend/main.py` | FastAPI entrypoint; hosts the six call sites + the new `_parse_srs_batch` helper. | MODIFY (~+15 LOC net: helper +25, deletions -10) |
| `extension/backend/tests/test_document_parser.py` | Parser-level tests. | MODIFY (append T-EMPTY-1..11, ~120 LOC) |
| `extension/backend/tests/test_main_wiring.py` | NEW — unit-scope wiring tests for `_parse_srs_batch` + `initialize_rag`. | CREATE (~80 LOC) |
| `development_docs/WP-CORE-3-empty-input-contract.md` | Persistent dev memory for this WP. | CREATE |
| `development_docs/INDEX.md` | Dev-doc status board. | MODIFY (new ACTIVE row) |
| `.planning/pipeline_audit/improvements_backlog.md` | Audit backlog. | MODIFY (F-3 OPEN → SHIPPED) |
| `.planning/pipeline_audit/CURRENT.md` | Audit pointer. | MODIFY (last action + next) |
| `.planning/pipeline_audit/decision_log.md` | Audit decisions. | MODIFY (+2 entries: Codex review + chosen approach) |

No file renames. No new dependencies.

---

## Task 1: RED — Add failing parser-level + wiring-level tests

**Goal:** Capture the new contract (parser raises on empty; helper skips empty in batch; SOFT site silently returns) as red tests. The current code does not export `EmptySRSDocumentError`, so collection fails at `ImportError`; this is the expected red signal.

**Files:**
- Modify: `extension/backend/tests/test_document_parser.py` (append T-EMPTY-1..11)
- Create: `extension/backend/tests/test_main_wiring.py` (T-WIRE-1..4)

**Pre-step — confirm baseline:**

- [ ] **Step 0: Confirm pre-task pytest baseline = 305**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend"
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=no -q 2>&1 | tail -3
```

Expected: `305 passed, 31 deselected in <N>s`. If different, STOP — escalate to coordinator.

- [ ] **Step 1: Open `extension/backend/tests/test_document_parser.py`**

Confirm the current file ends after `test_reference_heading_pattern_direct_grammar` (~line 380 region from WP-CORE-2). Append new T-EMPTY-* tests at the end.

- [ ] **Step 2: Append T-EMPTY-1..11 block**

Append the following block verbatim:

```python
# ---------- WP-CORE-3 — Empty-input contract (F-3) ----------

import re

from core.document_parser import EmptySRSDocumentError


def test_parse_empty_txt_raises_empty_srs_document_error(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    parser = SRSDocumentParser()
    with pytest.raises(EmptySRSDocumentError):
        parser.parse_file(str(empty))


def test_parse_whitespace_only_txt_raises_empty_srs_document_error(tmp_path):
    ws = tmp_path / "ws.txt"
    ws.write_text("\n\n   \t\n")
    parser = SRSDocumentParser()
    with pytest.raises(EmptySRSDocumentError):
        parser.parse_file(str(ws))


def test_parse_empty_pdf_raises_empty_srs_document_error(tmp_path, monkeypatch):
    placeholder = tmp_path / "x.pdf"
    placeholder.write_bytes(b"%PDF-1.4\n%dummy\n")
    monkeypatch.setattr("core.document_parser.read_pdf", lambda p: "")
    parser = SRSDocumentParser()
    with pytest.raises(EmptySRSDocumentError):
        parser.parse_file(str(placeholder))


def test_parse_empty_docx_raises_empty_srs_document_error(tmp_path, monkeypatch):
    placeholder = tmp_path / "x.docx"
    placeholder.write_bytes(b"PK\x03\x04dummy")
    monkeypatch.setattr("core.document_parser.read_docx", lambda p: "")
    parser = SRSDocumentParser()
    with pytest.raises(EmptySRSDocumentError):
        parser.parse_file(str(placeholder))


def test_empty_srs_document_error_is_value_error(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    parser = SRSDocumentParser()
    with pytest.raises(ValueError):
        parser.parse_file(str(empty))


def test_parse_unsupported_extension_still_raises_plain_value_error(tmp_path):
    weird = tmp_path / "x.xyz"
    weird.write_text("anything")
    parser = SRSDocumentParser()
    with pytest.raises(ValueError, match="Unsupported file type") as exc_info:
        parser.parse_file(str(weird))
    assert not isinstance(exc_info.value, EmptySRSDocumentError)


def test_parse_non_empty_txt_returns_text_normally(tmp_path):
    good = tmp_path / "ok.txt"
    good.write_text("Hello world.\n")
    parser = SRSDocumentParser()
    assert parser.parse_file(str(good)) == "Hello world."


def test_parse_references_only_txt_raises_empty_srs_document_error(tmp_path):
    """Post-truncation-empty (WP-CORE-2 cross-integration) is caught."""
    only_refs = tmp_path / "only_refs.txt"
    only_refs.write_text("References\n")
    parser = SRSDocumentParser()
    with pytest.raises(EmptySRSDocumentError):
        parser.parse_file(str(only_refs))


def test_empty_srs_document_error_message_contains_file_path(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    parser = SRSDocumentParser()
    with pytest.raises(EmptySRSDocumentError, match=re.escape(str(empty))):
        parser.parse_file(str(empty))


def test_parse_nonexistent_file_still_raises_file_not_found_post_contract():
    parser = SRSDocumentParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_file("/nonexistent/file.txt")


@pytest.mark.parametrize(
    "ext, patch_target",
    [
        (".pdf", "core.document_parser.read_pdf"),
        (".docx", "core.document_parser.read_docx"),
    ],
)
def test_each_binary_reader_empty_raises_empty_srs_document_error(
    tmp_path, monkeypatch, ext, patch_target
):
    placeholder = tmp_path / f"x{ext}"
    placeholder.write_bytes(b"placeholder bytes")
    monkeypatch.setattr(patch_target, lambda p: "")
    parser = SRSDocumentParser()
    with pytest.raises(EmptySRSDocumentError):
        parser.parse_file(str(placeholder))
```

- [ ] **Step 3: Create `extension/backend/tests/test_main_wiring.py`**

```python
"""Wiring tests for WP-CORE-3 empty-input contract.

Verifies the `_parse_srs_batch` helper and `initialize_rag` SOFT path
without touching real FastAPI, real LLMs, or real ChromaDB.
"""

import pytest

from core.document_parser import EmptySRSDocumentError
from main import _parse_srs_batch, initialize_rag


class _StubParser:
    """Minimal SRSDocumentParser-shaped stub for testing batch helper."""

    def __init__(self, behaviors):
        # behaviors: dict[path -> str or Exception]
        self.behaviors = behaviors
        self.calls = []

    def parse_file(self, path):
        self.calls.append(path)
        behavior = self.behaviors[path]
        if isinstance(behavior, Exception):
            raise behavior
        return behavior


def test_parse_srs_batch_skips_empty_file_and_continues_with_non_empty(tmp_path):
    parser = _StubParser(
        {
            "empty.txt": EmptySRSDocumentError("empty.txt parsed to empty content"),
            "good.txt": "good content here",
        }
    )
    combined, docs, err = _parse_srs_batch(parser, ["empty.txt", "good.txt"])
    assert err is None
    assert len(docs) == 1
    assert docs[0]["path"] == "good.txt"
    assert docs[0]["content"] == "good content here"
    assert "good content here" in combined
    # Empty file was attempted, then skipped
    assert parser.calls == ["empty.txt", "good.txt"]


def test_parse_srs_batch_returns_aggregate_error_when_all_files_empty():
    parser = _StubParser(
        {
            "a.txt": EmptySRSDocumentError("a.txt empty"),
            "b.txt": EmptySRSDocumentError("b.txt empty"),
        }
    )
    combined, docs, err = _parse_srs_batch(parser, ["a.txt", "b.txt"])
    assert err is not None
    assert err["success"] is False
    assert "All documents were empty" in err["error"]
    assert docs == []
    assert combined == ""


def test_parse_srs_batch_returns_per_file_error_for_non_empty_parse_failure():
    parser = _StubParser({"missing.txt": FileNotFoundError("missing.txt")})
    combined, docs, err = _parse_srs_batch(parser, ["missing.txt"])
    assert err is not None
    assert err["success"] is False
    assert "Failed to parse missing.txt" in err["error"]
    assert docs == []


def test_initialize_rag_silently_returns_empty_rag_on_empty_srs(monkeypatch):
    import main

    def fake_parse_file(self, path):
        raise EmptySRSDocumentError(path)

    class FakeRAG:
        def __init__(self):
            self.indexed_calls = []

        def index_document(self, **kwargs):
            self.indexed_calls.append(kwargs)
            return 0

    monkeypatch.setattr(
        "core.document_parser.SRSDocumentParser.parse_file", fake_parse_file
    )
    monkeypatch.setattr("main.RAGPipeline", FakeRAG)

    rag = main.initialize_rag(["/tmp/empty.txt"])
    assert isinstance(rag, FakeRAG)
    assert rag.indexed_calls == []
```

- [ ] **Step 4: Run pytest to confirm RED**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend"
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=line -q 2>&1 | tail -25
```

Expected: collection errors / ImportError on `EmptySRSDocumentError` and `_parse_srs_batch`. The 305 prior tests should still appear ERROR'd (collection failure) OR may pass individually depending on import-failure scope. **Acceptable red signal: any pytest exit code != 0 from missing `EmptySRSDocumentError` / `_parse_srs_batch` symbols.**

If collection-level errors cascade to all tests, the temporary acceptable state is `<N> errors, 305 passed`. The GREEN commit must turn this back to `320 passed, 31 deselected`.

- [ ] **Step 5: Commit RED phase**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/backend/tests/test_document_parser.py extension/backend/tests/test_main_wiring.py
git commit -m "$(cat <<'EOF'
test(parser, main): WP-CORE-3 red-phase tests for empty-input contract

Adds 11 parser-level T-EMPTY-* tests for the new
EmptySRSDocumentError + 4 wiring-level T-WIRE-* tests for the
_parse_srs_batch helper and initialize_rag SOFT-path.

These tests fail at collection until the GREEN commit lands the
exception class, the parse_file raise, the _parse_srs_batch helper,
and the six main.py call-site migrations.

WP-CORE-3 (F-3) spec v2:
docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md

Codex xhigh review handled 2 CRITICAL + 5 WARN inline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: GREEN — Implement exception + parser raise + helper + six call-site migrations (atomic)

**Goal:** Land all production changes in a single commit so the codebase is never in a state where the parser raises but call sites are stale. Tests must reach 320 passed / 31 deselected after this commit.

**Files:**
- Modify: `extension/backend/core/document_parser.py`
- Modify: `extension/backend/main.py`

- [ ] **Step 1: Add `EmptySRSDocumentError` to `core/document_parser.py`**

Insert immediately after `from core.document_parser_readers import read_docx, read_pdf, read_txt` (currently line 5), before `class SRSDocumentParser:`:

```python


class EmptySRSDocumentError(ValueError):
    """Raised by SRSDocumentParser.parse_file when the post-processed text is empty.

    Subclasses ValueError so coarse ``except ValueError`` callers
    keep the same outcome; the dedicated type lets new callers be
    explicit when they want to distinguish empty content from other
    value errors (FileNotFoundError, "Unsupported file type", etc.).
    """
```

- [ ] **Step 2: Add the raise at the end of `parse_file`**

Replace the final `return self._post_process(raw)` (currently `:51`) with:

```python
        processed = self._post_process(raw)
        if not processed:
            raise EmptySRSDocumentError(
                f"Document parsed to empty content (post-processing): {file_path}"
            )
        return processed
```

- [ ] **Step 3: Add `_parse_srs_batch` helper to `main.py`**

Locate the import block at the top of `main.py`. Confirm/update:

```python
from typing import Any, Dict, List, Optional, Tuple
from core.document_parser import EmptySRSDocumentError, SRSDocumentParser
```

Find a stable insertion point near other module-scope helpers (after `find_srs_files` or before the FastAPI app definition — pick whichever keeps related code grouped). Insert:

```python
def _parse_srs_batch(
    parser: SRSDocumentParser,
    file_paths: List[str],
) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Parse all SRS files for a batch endpoint.

    Returns (combined_text, srs_docs, error).
    - error is None on success (at least one non-empty file parsed).
    - error is a {"success": False, "error": ...} dict on:
        * per-file non-empty exception (FileNotFoundError, unsupported, etc.)
        * all files parsed to empty content (post-loop aggregate check).
    Per-file EmptySRSDocumentError is logged and the file is skipped;
    partial batches succeed.
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

- [ ] **Step 4: HARD migration at `main.py:55-62` (`generate_domain_model`)**

Find:

```python
def generate_domain_model(srs_path: str) -> Dict[str, Any]:
    """Generate domain model from SRS document using AI pipeline."""
    doc_parser = SRSDocumentParser()
    raw_text = doc_parser.parse_file(srs_path)
    print(f"   -> Parsed document: {len(raw_text)} characters")

    if not raw_text.strip():
        raise ValueError("Document is empty or could not be parsed.")
```

Replace with:

```python
def generate_domain_model(srs_path: str) -> Dict[str, Any]:
    """Generate domain model from SRS document using AI pipeline."""
    doc_parser = SRSDocumentParser()
    raw_text = doc_parser.parse_file(srs_path)
    print(f"   -> Parsed document: {len(raw_text)} characters")
```

(The new `parse_file` raise replaces the deleted post-call guard.)

- [ ] **Step 5: SOFT migration at `main.py:96-109` (`initialize_rag`)**

Find:

```python
def initialize_rag(srs_files: List[str]) -> RAGPipeline:
    """Initialize RAG pipeline and index SRS documents."""
    rag = RAGPipeline()

    if srs_files:
        srs_path = srs_files[0]
        doc_parser = SRSDocumentParser()
        raw_text = doc_parser.parse_file(srs_path)

        if raw_text.strip():
            filename = Path(srs_path).name
            ext = Path(srs_path).suffix[1:]
            chunk_count = rag.index_document(
                raw_text=raw_text, doc_id="srs_main", doc_name=filename, doc_type=ext
            )
            print(f"[RAG] Indexed {chunk_count} chunks from {filename}")

    return rag
```

Replace with:

```python
def initialize_rag(srs_files: List[str]) -> RAGPipeline:
    """Initialize RAG pipeline and index SRS documents."""
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

- [ ] **Step 6: MIXED migration at `main.py:305-332` (`/generate-model` batch)**

Find the block starting at `try:` near `:305`:

```python
    try:
        # Parse and combine all documents
        doc_parser = SRSDocumentParser()
        combined_text = ""
        srs_docs: List[Dict[str, Any]] = []
        
        for file_path in request.file_paths:
            print(f"  📄 Parsing: {file_path}")
            try:
                raw_text = doc_parser.parse_file(file_path)
                combined_text += f"\n\n--- Document: {Path(file_path).name} ---\n\n"
                combined_text += raw_text
                srs_docs.append({"path": file_path, "content": raw_text})
                print(f"     -> {len(raw_text)} characters")
            except Exception as e:
                print(f"     ❌ Failed to parse: {e}")
                return {
                    "success": False,
                    "error": f"Failed to parse {Path(file_path).name}: {str(e)}",
                }
        
        if not combined_text.strip():
            return {
                "success": False,
                "error": "All documents are empty or could not be parsed",
            }
        
        print(f"  📊 Total combined text: {len(combined_text)} characters")
```

Replace with:

```python
    try:
        doc_parser = SRSDocumentParser()
        combined_text, srs_docs, error = _parse_srs_batch(
            doc_parser, request.file_paths
        )
        if error is not None:
            return error

        print(f"  📊 Total combined text: {len(combined_text)} characters")
```

- [ ] **Step 7: SOFT migration at `main.py:363-378` (RAG re-index in `/generate-model`)**

Find:

```python
            rag = RAGPipeline()
            for file_path in request.file_paths:
                raw_text = doc_parser.parse_file(file_path)
                if raw_text.strip():
                    filename = Path(file_path).name
                    ext = Path(file_path).suffix[1:]
                    chunk_count = rag.index_document(
                        raw_text=raw_text,
                        doc_id=f"srs_{filename}",
                        doc_name=filename,
                        doc_type=ext,
                    )
                    print(f"     -> Indexed {chunk_count} chunks from {filename}")
```

Replace with:

```python
            rag = RAGPipeline()
            for file_path in request.file_paths:
                try:
                    raw_text = doc_parser.parse_file(file_path)
                except EmptySRSDocumentError as exc:
                    print(f"     [RAG] skip empty SRS: {exc}")
                    continue
                filename = Path(file_path).name
                ext = Path(file_path).suffix[1:]
                chunk_count = rag.index_document(
                    raw_text=raw_text,
                    doc_id=f"srs_{filename}",
                    doc_name=filename,
                    doc_type=ext,
                )
                print(f"     -> Indexed {chunk_count} chunks from {filename}")
```

- [ ] **Step 8: MIXED migration at `main.py:434-451` (`/generate-model-stream` batch, threaded)**

Find:

```python
            # Parse and combine all documents
            doc_parser = SRSDocumentParser()
            combined_text = ""
            srs_docs: List[Dict[str, Any]] = []
            
            for file_path in request.file_paths:
                try:
                    raw_text = doc_parser.parse_file(file_path)
                    combined_text += f"\n\n--- Document: {Path(file_path).name} ---\n\n"
                    combined_text += raw_text
                    srs_docs.append({"path": file_path, "content": raw_text})
                except Exception as e:
                    result_holder["error"] = f"Failed to parse {Path(file_path).name}: {str(e)}"
                    return
            
            if not combined_text.strip():
                result_holder["error"] = "All documents are empty or could not be parsed"
                return
```

Replace with:

```python
            doc_parser = SRSDocumentParser()
            combined_text, srs_docs, error = _parse_srs_batch(
                doc_parser, request.file_paths
            )
            if error is not None:
                result_holder["error"] = error["error"]
                return
```

- [ ] **Step 9: SOFT migration at `main.py:476-487` (RAG re-index in `/generate-model-stream`)**

Find:

```python
                rag = RAGPipeline()
                for file_path in request.file_paths:
                    raw_text = doc_parser.parse_file(file_path)
                    if raw_text.strip():
                        filename = Path(file_path).name
                        ext = Path(file_path).suffix[1:]
                        rag.index_document(...)
```

Replace with:

```python
                rag = RAGPipeline()
                for file_path in request.file_paths:
                    try:
                        raw_text = doc_parser.parse_file(file_path)
                    except EmptySRSDocumentError as exc:
                        print(f"     [RAG] skip empty SRS: {exc}")
                        continue
                    filename = Path(file_path).name
                    ext = Path(file_path).suffix[1:]
                    rag.index_document(
                        raw_text=raw_text,
                        doc_id=f"srs_{filename}",
                        doc_name=filename,
                        doc_type=ext,
                    )
```

Preserve the original full `rag.index_document(...)` argument list — the snippet above abbreviates only for readability.

- [ ] **Step 10: Run pytest — must hit 320**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer/extension/backend"
/Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest -m "not integration" --tb=short -q 2>&1 | tail -10
```

Expected: `320 passed, 31 deselected in <N>s`. If any test fails, debug and re-run until green. Do not commit until 320.

- [ ] **Step 11: Run pyright on `core/` (no new typing errors)**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
pyright extension/backend/core/document_parser.py extension/backend/main.py 2>&1 | tail -10
```

If pre-WP pyright on these files had errors, the count must not increase. If pyright is unavailable locally, document the skip in the commit message; CI will surface any regression.

- [ ] **Step 12: Commit GREEN phase**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add extension/backend/core/document_parser.py extension/backend/main.py
git commit -m "$(cat <<'EOF'
fix(parser, main): WP-CORE-3 EmptySRSDocumentError — explicit empty-input contract

Add EmptySRSDocumentError(ValueError); raise it from
SRSDocumentParser.parse_file when post-processed content is empty.
Migrate all six main.py call sites (1 sync HARD, 1 SOFT init,
2 MIXED batch loops via new _parse_srs_batch helper, 2 SOFT RAG
re-index loops) to explicit per-path policy.

Latent bug folded in: the previous post-loop combined_text.strip()
guards at /generate-model and /generate-model-stream were dead code
(separator headers made combined_text.strip() always non-empty);
the new helper uses srs_docs emptiness as the aggregate check,
which actually fires.

Behavior change (intentional, documented): mixed batches with one
empty + one non-empty file now succeed (empty skipped + logged);
previously the empty file silently degraded the combined input
with separator-only content. Read failures still kill batches.

Test baseline: 305 → 320 passed, 31 deselected.

WP-CORE-3 (F-3) spec v2:
docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md

Codex xhigh review: 2 CRITICAL + 5 WARN handled inline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: DOC — Persistent dev memory + audit state update

**Goal:** Record the WP in `development_docs/` so paper revision and future audits can reference rationale without re-deriving from git log. Flip the audit backlog finding to SHIPPED.

**Files:**
- Create: `development_docs/WP-CORE-3-empty-input-contract.md`
- Modify: `development_docs/INDEX.md`
- Modify: `.planning/pipeline_audit/improvements_backlog.md`
- Modify: `.planning/pipeline_audit/CURRENT.md`
- Modify: `.planning/pipeline_audit/decision_log.md`

- [ ] **Step 1: Create `development_docs/WP-CORE-3-empty-input-contract.md`**

Follow the convention in `development_docs/INDEX.md` (sections in order): status / branch / commit SHAs / spec + plan paths / one-paragraph TL;DR / motivation / architectural decisions (numbered, with rationale) / file-level changes table / methodology applied / empirical results / limitations + follow-ups / cross-references. Link to `[[WP-CORE-2-reference-truncate-fix]]`.

- [ ] **Step 2: Update `development_docs/INDEX.md`**

Add new row to ACTIVE table with WP-CORE-3 doc pointer.

- [ ] **Step 3: Update `.planning/pipeline_audit/improvements_backlog.md`**

Change F-3 status: `OPEN` → `SHIPPED (<green-commit-sha>)`.

- [ ] **Step 4: Update `.planning/pipeline_audit/CURRENT.md`**

Update last-update timestamp, last-action ("Iteration 2 closed — WP-CORE-3 shipped"), next ("Iteration 3 — core/architect.py close-lookup OR continue with F-1/F-2/F-4 on document_parser layer"), baseline (320 passed, 31 deselected).

- [ ] **Step 5: Append two entries to `.planning/pipeline_audit/decision_log.md`**

- `D-CODEX-REVIEW-WP-CORE-3` — capture the Codex xhigh review verbatim + the disposition (2 CRITICAL handled, 5 WARN handled).
- `D-EMPTY-INPUT-CONTRACT-2026-05-21` — capture: chose Alt C (parser raises uniformly, per-path policy at callers) over Alt B (two entry points) and Alt E (helper-only). Rationale: AGENTS.md "Stable entrypoints" + leak elimination.

- [ ] **Step 6: Commit DOC phase**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add development_docs/WP-CORE-3-empty-input-contract.md development_docs/INDEX.md .planning/pipeline_audit/improvements_backlog.md .planning/pipeline_audit/CURRENT.md .planning/pipeline_audit/decision_log.md
git commit -m "$(cat <<'EOF'
chore(artifacts): WP-CORE-3 dev_doc + audit state update

- development_docs/WP-CORE-3-empty-input-contract.md created
- development_docs/INDEX.md ACTIVE row added
- .planning/pipeline_audit/improvements_backlog.md F-3 → SHIPPED
- .planning/pipeline_audit/CURRENT.md pointer + baseline → 320
- .planning/pipeline_audit/decision_log.md +D-CODEX-REVIEW-WP-CORE-3 +D-EMPTY-INPUT-CONTRACT

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: PLANNING — Spec + plan into git history

- [ ] **Step 1: Stage and commit the planning artifacts**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add docs/superpowers/specs/2026-05-21-wp-core-3-empty-input-contract-design.md docs/superpowers/plans/2026-05-21-wp-core-3-empty-input-contract.md
git commit -m "$(cat <<'EOF'
chore(planning): WP-CORE-3 spec v2 + plan into git history

Spec v2 captures the Codex xhigh adversarial review and the
2 CRITICAL + 5 WARN inline dispositions.  Plan files the
RED → GREEN-atomic → DOC → PLANNING → HANDOFF sequence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: HANDOFF — Iteration 2 close + iteration 3 prep

- [ ] **Step 1: Write handoff doc**

`.planning/pipeline_audit/handoff-2026-05-21-<HHMM>.md`. Follow the structure of `handoff-2026-05-21-0220.md`: state of loop, baseline (320), HEAD after WP, backlog snapshot (F-3 → SHIPPED), next iteration candidates with rationale.

Next-iteration candidates:
- **Priority 2 close-lookup** — `core/architect.py` (752 LOC) close audit. Largest single surface in pipeline; previously deferred from iteration 2 in favor of F-3 cohesion. **Recommended for iteration 3.**
- **Continue ingestion-layer** — F-1 (`read_pdf` defensive handling) and/or F-2 (`read_txt` cp1254 binary garbage). Smaller scope but layer-completionist; less new ground for the audit map.
- **Continue heuristic-layer** — F-4 (TOC heuristic 120-line + cluster<2). Same file, but uncertain severity per F-4 marker; less obviously a fix.

- [ ] **Step 2: Update `.planning/pipeline_audit/CURRENT.md` next-pointer**

Reference the new handoff doc.

- [ ] **Step 3: Commit handoff**

```bash
cd "/Users/barandincoguz/Desktop/AI & NLP/DDD-Enforcer"
git add .planning/pipeline_audit/handoff-2026-05-21-*.md .planning/pipeline_audit/CURRENT.md
git commit -m "$(cat <<'EOF'
chore(planning): WP-CORE-3 iteration 2 handoff + CURRENT pointer

Captures: baseline 320, HEAD post-WP, F-3 SHIPPED, next-iteration
candidates (priority 2 architect.py close-lookup recommended).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Goal-backward verification (does the plan deliver on the spec?)

| Spec acceptance criterion | Plan task / step |
|---|---|
| `pytest -m "not integration"` 305 → 320 | Task 1 Step 4 (red) + Task 2 Step 10 (green) |
| Zero regression in existing 305 tests | Task 2 Step 10 (full suite re-run) |
| T-WIRE-1..4 lock the per-path behavior | Task 1 Step 3 (test file create) |
| `pyright` clean | Task 2 Step 11 |
| `git grep "raw_text.strip()" main.py` returns 0 | Task 2 Steps 4–9 (all 6 sites migrated) |
| `git grep "All documents are empty or could not be parsed" main.py` returns 0 | Task 2 Steps 6, 8 (post-loop guards replaced) |
| `git grep "except EmptySRSDocumentError" main.py` ≥ 4 | Task 2 Steps 3, 5, 7, 9 (helper + 3 SOFT sites) |
| Atomic Conventional Commits + trailer | Task 1 Step 5, Task 2 Step 12, Task 3 Step 6, Task 4 Step 1, Task 5 Step 3 |
| No git push | (omitted from plan — loop rule) |

All acceptance criteria covered by at least one plan step. **Plan is complete.**

---

## Out of scope (carried forward to backlog / future WPs)

- F-1 — `read_pdf` defensive handling (next-iteration candidate)
- F-2 — `read_txt` cp1254 binary garbage (next-iteration candidate)
- F-4 — TOC heuristic (next-iteration candidate, uncertain severity)
- F-7 — DOCX zero try/except (deferred)
- F-9 — Introduce `logging` module (deferred; this WP uses `print` per current convention)
- F-10 — Duplicate parse at `main.py:366, 480` (deferred; helper extraction does not solve the duplicate I/O issue)
- Batch-tolerant atomicity for read failures (today: one read failure kills batch; this WP intentionally preserves that; if changed, separate feature WP)
- Lifespan error-message refinement to mention "empty SRS" specifically (deferred to F-9 / observability)
- `core/architect.py` close-lookup (Priority 2 — recommended for iteration 3)
