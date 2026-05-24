"""Atomic file-write helpers (``tmp + fsync + os.replace`` pattern).

Single source of truth for the atomic-write recipe used by
``core.run_manifest``, ``core.aggregate``, and ``core.latex_tables``.
Each of those modules previously copied the same ~8-line pattern;
this helper centralises it so a fix or audit applies in one place.

Behaviour:
  1. ``target.parent`` is created (``mkdir(parents=True, exist_ok=True)``).
  2. Payload is written to ``target.with_suffix(target.suffix + ".tmp")``.
  3. The file handle is flushed and ``os.fsync``-ed.
  4. ``os.replace(tmp, target)`` atomically swaps the .tmp into place.

If any step before the ``os.replace`` raises, the target file is
untouched (callers see no corruption); only the ``.tmp`` sibling
remains and can be discarded.

Test patches that simulate crashes should target
``core.io_atomic.os.replace`` (not the calling module's namespace —
the call now happens here).
"""

from __future__ import annotations

import os
from pathlib import Path


def write_text_atomic(
    target: Path,
    content: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write *content* to *target* atomically.

    See module docstring for the full recipe. Returns *target* for
    caller convenience.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding=encoding) as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return target
