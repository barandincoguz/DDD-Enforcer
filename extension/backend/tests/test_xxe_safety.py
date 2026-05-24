"""F-8 (iter 45): XXE defense-in-depth tests.

Cover four cases:
  1. Shipped python-docx parsers satisfy ``assert_xxe_safe_parsers``
     (regression guard if the library ever flips its default).
  2. A parser with ``resolve_entities=True`` raises ``XXESafetyError``
     with the expected "resolves XML entities" message.
  3. ``python-docx`` import failure propagates as ``XXESafetyError``,
     not a bare ``ImportError``.
  4. The behavior probe itself raising an unexpected error is
     wrapped as ``XXESafetyError``.

Uses a real ``lxml.etree.XMLParser(resolve_entities=True)`` to
simulate an unsafe upstream — the only way to drive the behavior
probe through a positive-XXE case (attribute-patching the real
parsers won't work; their ``resolve_entities`` flag is C-level and
constructor-only).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from lxml import etree

from core.security.xxe_safety import (
    XXESafetyError,
    assert_xxe_safe_parsers,
)


def test_shipped_python_docx_parsers_pass_safety_assertion():
    # If this fails after a python-docx upgrade, the upstream library
    # has dropped the resolve_entities=False default — STOP and audit
    # before pinning the new version. See
    # development_docs/F-8-xxe-assessment.md.
    assert_xxe_safe_parsers()


def test_unsafe_parser_raises_with_xxe_message():
    unsafe_parser = etree.XMLParser(resolve_entities=True)
    with patch(
        "core.security.xxe_safety._get_docx_parsers",
        return_value=[("test_parser", unsafe_parser)],
    ):
        with pytest.raises(XXESafetyError, match="resolves XML entities"):
            assert_xxe_safe_parsers()


def test_docx_import_failure_is_wrapped_as_xxesafetyerror():
    def _boom():
        raise ImportError("docx is gone")

    with patch(
        "core.security.xxe_safety._get_docx_parsers",
        side_effect=_boom,
    ):
        with pytest.raises(XXESafetyError, match="docx is gone"):
            assert_xxe_safe_parsers()


def test_unexpected_probe_failure_is_wrapped():
    def _broken_probe(_parser):
        raise RuntimeError("simulated lxml crash")

    with patch(
        "core.security.xxe_safety._entity_resolution_disabled",
        side_effect=_broken_probe,
    ):
        with pytest.raises(XXESafetyError, match="simulated lxml crash"):
            assert_xxe_safe_parsers()
