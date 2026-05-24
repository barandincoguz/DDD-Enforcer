# F-8 XXE Hardening — Assessment & Close-out

| Field | Value |
|-------|-------|
| **Status** | ✅ CLOSED (2026-05-24, iteration 45) |
| **Branch / commit** | `main` / next commit closing iter 45 |
| **Spec / planning** | `.planning/pipeline_audit/NEXT-WORK-PLAN.md` §3 Rank 1; original finding observation 3395 (2026-05-19 codebase audit) |
| **Risk verdict** | **No Critical/High findings.** Both python-docx (lxml) and pypdf already harden against CWE-611 at construction time. |
| **Action taken** | Defense-in-depth only: dep floor bump, runtime assertion of upstream flags, input-size cap. No XML-parser swap, no `defusedxml` adoption. |

## TL;DR

The DDD-Enforcer SRS ingestion pipeline (`core/document_parser.py`)
reads three document types: PDF (`pypdf`), DOCX (`python-docx` → `lxml`),
and TXT (stdlib). XML parsers are only invoked transitively through
the third-party libraries; the project never imports `xml`, `lxml`,
or `defusedxml` directly (verified via `grep -rn "import (xml|lxml|defusedxml)"`).

Both upstream libraries already implement the textbook XXE
mitigations:

- **python-docx 1.2.0 → lxml 6.0.2**: both
  `docx.opc.oxml.oxml_parser` and `docx.oxml.parser.oxml_parser`
  are declared `etree.XMLParser(remove_blank_text=True, resolve_entities=False)`.
  Source: `docx/opc/oxml.py:21`, `docx/oxml/parser.py:19`.
- **pypdf 6.10.2**: XMP metadata is parsed via a custom
  `_XmpBuilder(ExpatBuilderNS)` whose
  `custom_entity_declaration_handler` raises on EVERY entity
  declaration. Source: `pypdf/xmp.py:170-206`. Additionally, the
  project never calls `reader.xmp_metadata`, so the path is latent.

No project code path was found that could expose `/etc/passwd`,
trigger SSRF, or fall to a billion-laughs DoS through these
libraries' default configurations.

## Audit details

### Attack-surface map

| # | Project entry point | Library | XML method | Hardening | Risk |
|---|--------------------|---------|------------|-----------|------|
| A | `core/document_parser_readers.py:290` `docx.Document(file_path)` | python-docx → lxml | `etree.XMLParser(resolve_entities=False)` | Library default | **Low** |
| B | `core/document_parser_readers.py:244` `PdfReader(file_path)` then `.pages` / `.extract_text()` | pypdf | Standard PDF object parsing — no XML parser on this path | N/A | **Low** |
| C | *(latent)* `PdfReader(...).xmp_metadata` | pypdf | `_XmpBuilder(ExpatBuilderNS)` denies entity decls; caps element count + input length | Library default | **Low** (project never invokes it) |
| D | `core/document_parser_readers.py:354` `read_txt` | stdlib `pathlib.read_bytes` + decode | No XML parser | N/A | **None** |

### Why no XML-parser swap

`defusedxml` is the standard remediation when project code constructs
its own XML parsers. We don't. Replacing the lxml-backed parser
inside vendored `python-docx` would break the library (it relies on
`lxml.etree.ElementNamespaceClassLookup` for OOXML namespaces) and
provides no marginal safety beyond what `resolve_entities=False`
already gives.

## Mitigations shipped (defense-in-depth)

### 1. pypdf floor bump

`extension/backend/requirements.txt:20`: pypdf `>=4.0.0` → `>=6.0,<7`.
The `_XmpBuilder` hardening that denies entity declarations was added
in pypdf 6.x. The floor bump guarantees the safe XMP path is always
present if a future contributor adds `reader.xmp_metadata` to the
ingestion pipeline.

### 2. Startup assertion (`core/security/xxe_safety.py`)

`assert_xxe_safe_parsers()` runs a **behavior probe** against both
`python-docx` parser instances at app startup. A malicious payload
declaring a custom entity (`<!ENTITY xxe "INJECTED_VALUE">`) is
parsed through each parser; the assertion verifies the sentinel
string `"INJECTED_VALUE"` did NOT leak into the root element's
text. If it did, entity expansion is enabled and `XXESafetyError`
is raised.

The behavior-probe approach replaced an earlier attribute-probe
design that read `parser.resolve_entities` directly. lxml's
`XMLParser` is C-extension-backed and its `resolve_entities` flag
is constructor-only — there is no readable Python attribute on the
live instance. Tests at iter 45 surfaced this: the attribute probe
falsely reported "attribute missing" on the shipped, hardened
parsers. The behavior probe is the only reliable signal.

Raises `XXESafetyError` (subclass of `RuntimeError`) when:

- The probe payload's entity expands to `"INJECTED_VALUE"` on
  either parser (a future python-docx that flips its default), OR
- The `python-docx` import itself fails, OR
- The probe raises any unexpected exception.

Wired into the FastAPI `lifespan` handler in `main.py:322-326`.
On regression, uvicorn refuses to boot, surfacing the issue at the
deployment gate instead of in production.

Brittle by design. False positives during contributor upgrades are
the feature — they force re-audit before pinning a new version.

### 3. Input size cap (`core/document_parser.py:84-99`)

`SRSDocumentParser.parse_file` checks `os.path.getsize(file_path)`
before invoking any third-party parser. Default cap `50 MB`
(`DEFAULT_MAX_SRS_BYTES` constant); overridable via
`DDD_MAX_SRS_BYTES` env var (int bytes). Oversized files raise
`OversizedSRSDocumentError` (subclass of `ValueError`).

Mitigates: billion-laughs / decompression-bomb / quadratic-blowup
attacks that target the parser even with entity loading disabled.

## Tests shipped

`tests/test_xxe_safety.py` — 4 tests:

1. `test_shipped_python_docx_parsers_pass_safety_assertion` — real
   library check; fails if upstream regresses.
2. `test_unsafe_parser_raises_with_xxe_message` — drives a real
   `lxml.etree.XMLParser(resolve_entities=True)` through the
   probe and verifies it raises `XXESafetyError` with the
   "resolves XML entities" message.
3. `test_docx_import_failure_is_wrapped_as_xxesafetyerror` —
   import failures don't leak as bare `ImportError`.
4. `test_unexpected_probe_failure_is_wrapped` — any probe-side
   exception (e.g., an lxml internal raise) is wrapped as
   `XXESafetyError`.

`tests/test_oversized_srs.py` — 4 tests:

1. `test_default_cap_constant_is_50_mb` — constant pinned.
2. `test_env_override_triggers_oversized_error` — env var path works.
3. `test_malformed_env_falls_back_to_default` — non-int env doesn't
   crash; falls back to default.
4. `test_at_cap_does_not_raise` — small file passes.

## Out of scope (deliberate non-actions)

- **`defusedxml` dependency** — no project XML construction site to
  apply it to.
- **Direct XML-parser flag wrapping for python-docx** — library
  already does it.
- **Touching `xmp_metadata`** — never called; adding a parser
  hardening for an unused path is YAGNI.
- **DOCX-XXE fixture acceptance test** — would require hand-crafting
  a malformed DOCX ZIP (python-docx itself sanitizes input). The
  startup assertion covers the same regression signal at lower
  fixture-maintenance cost. Subagent recommendation noted but not
  shipped this iteration.

## Follow-ups

- If a future contributor adds `reader.xmp_metadata` reads in
  `core/document_parser_readers.py`, expand
  `assert_xxe_safe_parsers` to also verify pypdf's `_XmpBuilder`
  is still the registered handler (and add a parallel
  `tests/test_xxe_safety.py` case).
- If python-docx is ever replaced with a different DOCX reader,
  delete `core/security/xxe_safety.py:_get_docx_parsers` and
  re-author for the new library; do NOT silently drop the assertion.

## References

- `extension/backend/core/document_parser_readers.py:244,290` —
  parser entry points.
- `extension/backend/core/document_parser.py:84-99` — size cap.
- `extension/backend/core/security/xxe_safety.py` — assertion.
- `extension/backend/main.py:322-326` — lifespan wiring.
- `extension/backend/requirements.txt:20` — pypdf floor.
- `extension/backend/requirements.lock:622,1589,1605` —
  `lxml==6.1.0`, `pypdf==6.10.2`, `python-docx==1.2.0` at audit time.
- Upstream: `python-docx 1.2.0/docx/opc/oxml.py:21` and
  `docx/oxml/parser.py:19` (resolve_entities=False default).
- Upstream: `pypdf 6.10.2/pypdf/xmp.py:170-206` (`_XmpBuilder`
  custom entity-decl rejection).
- CWE: [CWE-611](https://cwe.mitre.org/data/definitions/611.html).
