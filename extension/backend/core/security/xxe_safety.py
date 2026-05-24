"""XXE defense-in-depth: assert third-party XML parsers stay hardened.

CWE-611 mitigation. The DDD-Enforcer SRS ingestion pipeline reads
DOCX (python-docx → lxml) and PDF (pypdf). Both libraries ship with
external-entity loading disabled at their parser-construction sites
(see development_docs/F-8-xxe-assessment.md for the full audit).

This module asserts those defaults at startup via a BEHAVIOR PROBE
(not an attribute read): a malicious custom-entity payload is parsed
through each library's actual parser and we verify the entity does
NOT expand. Behavior probing is the only reliable approach because
``lxml.etree.XMLParser`` is C-extension-backed and its
``resolve_entities`` flag is constructor-only — not exposed as a
readable Python attribute on the instance.

An upstream regression (e.g., a python-docx release that drops
``resolve_entities=False`` from its default ``XMLParser`` config)
will cause the probe to detect entity expansion and raise
``XXESafetyError`` at boot, so uvicorn refuses to start. Brittle by
design — false positives during contributor upgrades are the
feature: they force a re-audit before pinning a new version.
"""

from __future__ import annotations

from typing import List, Tuple

# Behavior-probe payload. The custom entity "&xxe;" must NOT expand
# to its declared value "INJECTED_VALUE" when parsed through a safe
# parser. Hardcoded sentinel + DOCTYPE + ENTITY + reference triple.
_XXE_PROBE_PAYLOAD = (
    b'<?xml version="1.0"?>\n'
    b'<!DOCTYPE root [<!ENTITY xxe "INJECTED_VALUE">]>\n'
    b"<root>&xxe;</root>"
)
_XXE_PROBE_SENTINEL = "INJECTED_VALUE"


class XXESafetyError(RuntimeError):
    """Raised when a third-party XML parser is detected with unsafe defaults."""


def _get_docx_parsers() -> List[Tuple[str, object]]:
    """Return the two python-docx oxml parser instances, labelled.

    Split out so tests can monkey-patch a fake parser list without
    touching the real (immutable) lxml-parser attributes.
    """
    from docx.opc.oxml import oxml_parser as opc_parser
    from docx.oxml.parser import oxml_parser as doc_parser

    return [("opc", opc_parser), ("doc", doc_parser)]


def _entity_resolution_disabled(parser: object) -> bool:
    """Behavior probe: ``True`` if XML entities do NOT expand through parser.

    Parses :data:`_XXE_PROBE_PAYLOAD` and checks that the
    :data:`_XXE_PROBE_SENTINEL` string did not leak into the root
    element's text content. A parser that refuses to read the
    DOCTYPE at all (``XMLSyntaxError``) is treated as safe — the
    payload never reached anywhere it could exfiltrate from.
    """
    # lxml.etree is a Cython-generated submodule with no public type
    # stubs; pyright cannot resolve the symbol but the runtime import
    # is correct (see lxml docs).
    from lxml import etree  # type: ignore[attr-defined]

    try:
        tree = etree.fromstring(_XXE_PROBE_PAYLOAD, parser=parser)
    except etree.XMLSyntaxError:
        return True
    text = (tree.text or "") if tree is not None else ""
    return _XXE_PROBE_SENTINEL not in text


def assert_xxe_safe_parsers() -> None:
    """Verify python-docx ships its hardened lxml parsers.

    Probes both ``docx.opc.oxml.oxml_parser`` and
    ``docx.oxml.parser.oxml_parser`` with a custom-entity payload.
    Raises if entity expansion is detected on either parser, the
    import fails, or the probe itself raises an unexpected error.
    """
    try:
        parsers = _get_docx_parsers()
    except Exception as exc:
        raise XXESafetyError(
            f"python-docx import failed during XXE safety check: {exc}. "
            "Re-audit before shipping "
            "(see development_docs/F-8-xxe-assessment.md)."
        ) from exc

    for label, parser in parsers:
        try:
            disabled = _entity_resolution_disabled(parser)
        except Exception as exc:
            raise XXESafetyError(
                f"{label} parser probe raised {type(exc).__name__}: {exc}. "
                "Re-audit before shipping "
                "(see development_docs/F-8-xxe-assessment.md)."
            ) from exc
        if not disabled:
            raise XXESafetyError(
                f"{label} parser resolves XML entities — XXE risk (CWE-611). "
                "Re-audit before shipping "
                "(see development_docs/F-8-xxe-assessment.md)."
            )
