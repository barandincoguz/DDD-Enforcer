"""
Document Parser

Parses SRS documents in various formats (PDF, DOCX, TXT) and cleans the text
for domain analysis. Removes headers, footers, table of contents, and
reference sections.
"""

import os
import re

import docx
from pypdf import PdfReader


class SRSDocumentParser:
    """Parser for SRS (Software Requirements Specification) documents."""

    def __init__(self):
        # Pattern to match table of contents entries (dotted lines with page numbers)
        self.toc_pattern = re.compile(r"\.{4,}\s*\d+")
        # Pattern to match page headers/footers (standalone page numbers)
        self.header_footer_pattern = re.compile(
            r"^\s*\d+\s*$|^\s*page\s*\d+\s*$", re.IGNORECASE
        )

    def parse_file(self, file_path: str) -> str:
        """
        Parse document and return cleaned text.

        Automatically detects file type and applies appropriate parser.
        Truncates at references section and removes noise.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            raw = self._read_pdf(file_path)
        elif ext == ".docx":
            raw = self._read_docx(file_path)
        elif ext == ".txt":
            raw = self._read_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        raw = self._truncate_at_references(raw)
        return self._clean_text(raw)

    def _read_pdf(self, path: str) -> str:
        """Extract text from PDF file."""
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_docx(self, path: str) -> str:
        """Extract text from DOCX file."""
        doc = docx.Document(path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    def _read_txt(self, path: str) -> str:
        """Read plain text file."""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _truncate_at_references(self, text: str) -> str:
        """Remove references/bibliography section from end of document."""
        stop_words = [
            "references",
            "bibliography",
            "kaynakca",
            "referanslar",
            "literatur",
        ]

        lines = text.split("\n")
        for i, line in enumerate(lines):
            if any(line.strip().lower().startswith(sw) for sw in stop_words):
                return "\n".join(lines[:i])
        return text

    def _clean_text(self, text: str) -> str:
        """Remove noise like headers, footers, and TOC entries."""
        result = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if self.header_footer_pattern.match(line):
                continue
            if self.toc_pattern.search(line):
                continue
            result.append(line)
        return "\n".join(result)
