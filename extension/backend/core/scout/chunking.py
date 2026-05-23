"""Section-aware SRS chunker (OQ1).

Replaces architect.py's character-based _split_text_into_chunks with a
parser that respects SRS section structure. Each chunk corresponds to
one numbered section (e.g. "2.1 Order Management"). Oversize sections
are subdivided into sub-chunks while preserving the section_id prefix.

Token budget is approximated at 4 chars/token (rough heuristic for
prose). Override per-model via the caller.
"""

import re
from typing import Dict, List


SECTION_HEADER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+?)$", re.MULTILINE)
_SUBCHUNK_SUFFIX = ".chunk_"


def count_truncated_sections(chunks: List[Dict]) -> int:
    """Return the number of distinct top-level sections that had to be
    subdivided into more than one sub-chunk by `section_aware_chunks`.

    A section is "truncated" (in the audit sense) when its content
    exceeded the token budget and was split into `<id>.chunk_1`,
    `<id>.chunk_2`, ...  This helper collapses those sub-chunks back to
    their base section id and counts the unique bases.  A section that
    fit in a single chunk contributes 0; a section that produced 5
    sub-chunks contributes 1.

    Closes the "ChunkMetadata.truncated_chunks always-zero" audit gap:
    `scout_fn` in `core/architect.py` plumbs this count into the
    `ScoutOutput.chunk_metadata.truncated_chunks` field so downstream
    paper-data consumers see the real subdivision count rather than the
    hardcoded default of 0.
    """
    truncated_bases: set[str] = set()
    for chunk in chunks:
        section_id = chunk.get("section_id", "")
        if not isinstance(section_id, str):
            continue
        if _SUBCHUNK_SUFFIX in section_id:
            truncated_bases.add(section_id.split(_SUBCHUNK_SUFFIX, 1)[0])
    return len(truncated_bases)


def section_aware_chunks(text: str, token_budget: int = 10000) -> List[Dict]:
    """Split SRS text into chunks at section boundaries.

    Returns a list of {"section_id", "section_title", "text"} dicts. If a
    single section's text exceeds the token_budget (approximated as
    token_budget * 4 chars), it is subdivided into sub-chunks suffixed
    with ".chunk_N" on the section_id.
    """
    char_budget = max(80, token_budget * 4)  # heuristic
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        # No section headers — fall back to a single chunk
        return [{"section_id": "0", "section_title": "Document", "text": text}]

    chunks: List[Dict] = []
    for i, m in enumerate(matches):
        section_id = m.group(1)
        section_title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        full = f"{section_id} {section_title}\n{body}"
        if len(full) <= char_budget:
            chunks.append({
                "section_id": section_id,
                "section_title": section_title,
                "text": full,
            })
        else:
            # Subdivide oversize section
            sub_count = 0
            cursor = 0
            while cursor < len(full):
                sub_end = min(cursor + char_budget, len(full))
                sub_count += 1
                chunks.append({
                    "section_id": f"{section_id}.chunk_{sub_count}",
                    "section_title": f"{section_title} (part {sub_count})",
                    "text": full[cursor:sub_end],
                })
                cursor = sub_end
    return chunks
