import os
import re
from typing import List

from core.document_parser_readers import read_docx, read_pdf, read_txt

class SRSDocumentParser:
    def __init__(self):
        self.reference_heading_pattern = re.compile(
            r"^(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography|kaynakça)\s*$",
            re.IGNORECASE,
        )
        self.contents_heading_pattern = re.compile(
            r"^(?:table of contents|contents|i[cç]indekiler)\s*$",
            re.IGNORECASE,
        )
        self.toc_line_pattern = re.compile(
            r"^(?:#{1,6}\s*)?(?:\d+(?:\.\d+)*\.?\s+)?[A-Za-z0-9ÇĞİÖŞÜçğıöşü].*\.{4,}\s*\d+\s*$",
            re.IGNORECASE,
        )
        self.header_footer_pattern = re.compile(
            r"^\s*(?:page\s*)?(?:\d+|[ivxlcdm]+)(?:\s*(?:/|of)\s*\d+)?\s*$",
            re.IGNORECASE,
        )
        self.heading_pattern = re.compile(
            r"^(?:#{1,6}\s+.+|\d+(?:\.\d+)*\.?\s+\S.+)$"
        )
        self.list_item_pattern = re.compile(
            r"^(?:[-*•]\s+|\d+[.)]\s+|[A-Za-z][.)]\s+)"
        )

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
        return self._post_process(raw)

    def _post_process(self, text: str) -> str:
        cleaned = self._normalize_text(text)
        cleaned = self._truncate_at_references(cleaned)
        cleaned = self._clean_lines(cleaned)
        cleaned = self._merge_wrapped_lines(cleaned)
        return self._normalize_text(cleaned).strip()

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("\u00a0", " ").replace("\u200b", "")
        normalized = re.sub(r"(?<=\w)-\n(?=\w)", "", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized

    def _truncate_at_references(self, text: str) -> str:
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if self.reference_heading_pattern.match(line.strip()):
                return "\n".join(lines[:index])
        return text

    def _clean_lines(self, text: str) -> str:
        raw_lines = text.split("\n")
        toc_lines = self._find_toc_line_indexes(raw_lines)
        result: List[str] = []
        for index, raw_line in enumerate(raw_lines):
            line = self._normalize_line(raw_line)
            if index in toc_lines or self.header_footer_pattern.match(line):
                continue
            if not line:
                result.append("")
                continue
            result.append(line)
        return "\n".join(self._collapse_blank_lines(result))

    def _find_toc_line_indexes(self, raw_lines: List[str]) -> set[int]:
        window = min(len(raw_lines), 120)
        toc_indexes: set[int] = set()
        cluster: List[int] = []

        for index in range(window):
            line = self._normalize_line(raw_lines[index])
            is_toc_line = bool(self.toc_line_pattern.match(line))

            if is_toc_line:
                cluster.append(index)
                continue

            if not line and cluster:
                continue

            self._flush_toc_cluster(raw_lines, cluster, toc_indexes)
            cluster = []

        self._flush_toc_cluster(raw_lines, cluster, toc_indexes)
        return toc_indexes

    def _flush_toc_cluster(
        self,
        raw_lines: List[str],
        cluster: List[int],
        toc_indexes: set[int],
    ) -> None:
        if len(cluster) < 2:
            return

        toc_indexes.update(cluster)
        previous_index = cluster[0] - 1
        if previous_index >= 0:
            previous_line = self._normalize_line(raw_lines[previous_index])
            if self.contents_heading_pattern.match(previous_line):
                toc_indexes.add(previous_index)

    def _normalize_line(self, line: str) -> str:
        stripped = line.strip().replace("\x00", "")
        stripped = re.sub(r"(?<=\S)(?: {3,}|\t+)(?=\S)", " | ", stripped)
        stripped = re.sub(r"[ \t]+", " ", stripped)
        return stripped

    def _collapse_blank_lines(self, lines: List[str]) -> List[str]:
        collapsed: List[str] = []
        for line in lines:
            if line:
                collapsed.append(line)
                continue
            if collapsed and collapsed[-1] != "":
                collapsed.append("")
        if collapsed and collapsed[-1] == "":
            collapsed.pop()
        return collapsed

    def _merge_wrapped_lines(self, text: str) -> str:
        merged: List[str] = []
        for line in text.split("\n"):
            if not line:
                if merged and merged[-1] != "":
                    merged.append("")
                continue
            if not merged or merged[-1] == "":
                merged.append(line)
                continue
            previous = merged[-1]
            if self._should_merge(previous, line):
                merged[-1] = self._join_lines(previous, line)
            else:
                merged.append(line)
        return "\n".join(self._collapse_blank_lines(merged))

    def _should_merge(self, previous: str, current: str) -> bool:
        if previous.endswith("-"):
            return True
        if self._looks_like_heading(previous) or self._looks_like_heading(current):
            return False
        if self._looks_like_list_item(current):
            return False
        if self._looks_like_table_row(previous) or self._looks_like_table_row(current):
            return False
        if re.search(r"[.!?;:]$", previous):
            return False
        return True

    def _join_lines(self, previous: str, current: str) -> str:
        if previous.endswith("-"):
            return previous[:-1].rstrip() + current.lstrip()
        return f"{previous.rstrip()} {current.lstrip()}"

    def _looks_like_heading(self, line: str) -> bool:
        if self.heading_pattern.match(line) and len(line) <= 150:
            return True
        return len(line) <= 80 and line.isupper()

    def _looks_like_list_item(self, line: str) -> bool:
        return bool(self.list_item_pattern.match(line))

    def _looks_like_table_row(self, line: str) -> bool:
        return " | " in line
