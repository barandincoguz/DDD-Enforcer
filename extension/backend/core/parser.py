"""Public facade for Python source parsing used by the validation pipeline."""

from typing import Any, Dict

from core.code_parser.service import parse_source_code


class CodeParser:
    """Stable parser entrypoint used by validation and tests."""

    def parse_code(self, source_code: str, filename: str = "") -> Dict[str, Any]:
        return parse_source_code(source_code=source_code, filename=filename)
