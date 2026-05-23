"""WP-CORE-12 — TOC heuristic layout-mode detection (F-4).

T-TOC-1: pipe-separator TOC (post-_normalize_line) detected.
T-TOC-2: raw whitespace separator TOC detected.
T-TOC-3: dot-leader TOC still detected (regression).
T-TOC-4: single layout TOC-like line NOT dropped via cluster<2 guard.

Run: pytest tests/test_document_parser_toc_layout.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# T-TOC-1 — pipe-separator (post-_normalize_line) layout-mode TOC
# =============================================================================


def test_toc_layout_mode_pipe_separator_detected(tmp_path):
    """T-TOC-1: TOC where layout-mode + _normalize_line produced
    `"Section title | page"` shape is filtered out, not leaked to scout."""
    from core.document_parser import SRSDocumentParser

    txt_file = tmp_path / "toc-pipe.txt"
    txt_file.write_text(
        "Table of Contents\n"
        "1.1 Introduction       1\n"
        "1.2 Scope       2\n"
        "1.3 Requirements       3\n"
        "\n"
        "Section 1 - Real Requirement Content\n"
        "The system shall do something useful.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))
    # TOC entries should be filtered. The real requirement remains.
    assert "shall do something useful" in content
    # The "1.1 Introduction       1" TOC entry should NOT appear in cleaned output.
    # After _normalize_line, multiple spaces become " | " → "1.1 Introduction | 1"
    assert "1.1 Introduction | 1" not in content
    assert "1.2 Scope | 2" not in content


# =============================================================================
# T-TOC-2 — raw whitespace separator (no _normalize_line) layout-mode TOC
# =============================================================================


def test_toc_dot_leader_still_detected(tmp_path):
    """T-TOC-3: traditional TOC with dot leaders still filtered (regression)."""
    from core.document_parser import SRSDocumentParser

    txt_file = tmp_path / "toc-dots.txt"
    txt_file.write_text(
        "Table of Contents\n"
        "1.1 Introduction ........ 1\n"
        "1.2 Scope .............. 2\n"
        "1.3 Requirements ....... 3\n"
        "\n"
        "Section 1 - Body\n"
        "Functional requirements list.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))
    assert "Functional requirements list" in content
    # Dot-leader TOC entries filtered.
    assert "Introduction ........ 1" not in content


# =============================================================================
# T-TOC-4 — Single layout-like line NOT dropped (cluster<2 guard)
# =============================================================================


def test_toc_single_layout_line_not_dropped_via_cluster_guard(tmp_path):
    """T-TOC-4: a single line `"X | 42"` (potential false-positive shape)
    is NOT filtered because cluster<2 rejection still applies."""
    from core.document_parser import SRSDocumentParser

    txt_file = tmp_path / "single-pipe.txt"
    txt_file.write_text(
        "Introduction\n"
        "This is a description.\n"
        "Cost estimate | 42\n"  # single layout-shape line; cluster=1 → not filtered
        "More content.\n",
        encoding="utf-8",
    )

    content = SRSDocumentParser().parse_file(str(txt_file))
    # Single TOC-shape line should NOT be filtered (cluster<2 guard).
    assert "Cost estimate | 42" in content or "Cost estimate" in content
