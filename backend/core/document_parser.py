# document_parser.py

import os
import re

import docx
from pypdf import PdfReader


class SRSDocumentParser:
    def __init__(self):
        self.toc_pattern = re.compile(r"\.{4,}\s*\d+")
        self.header_footer_pattern = re.compile(
            r"^\s*\d+\s*$|^\s*page\s*\d+\s*$", re.IGNORECASE
        )

    def parse_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            raw = self._read_pdf(file_path)
        elif ext == ".docx":
            raw = self._read_docx(file_path)
        elif ext == ".txt":
            raw = self._read_txt(file_path)
        else:
            raise Exception(f"Unsupported file type: {ext}")

        raw = self._truncate_at_references(raw)
        return self._clean_lines(raw)

    # -------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------
    def _truncate_at_references(self, text: str) -> str:
        stop_words = ["references", "bibliography", "kaynakça"]

        lines = text.split("\n")
        for i, line in enumerate(lines):
            l = line.strip().lower()
            if any(l.startswith(sw) for sw in stop_words):
                return "\n".join(lines[:i])
        return text

    def _clean_lines(self, text: str) -> str:
        result = []
        for ln in text.split("\n"):
            ln = ln.strip()
            if not ln:
                continue
            if self.header_footer_pattern.match(ln):
                continue
            if self.toc_pattern.search(ln):
                continue
            result.append(ln)
        return "\n".join(result)

    def _read_pdf(self, fp):
        reader = PdfReader(fp)
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    def _read_docx(self, fp):
        d = docx.Document(fp)
        return "\n".join([p.text for p in d.paragraphs])

    def _read_txt(self, fp):
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
