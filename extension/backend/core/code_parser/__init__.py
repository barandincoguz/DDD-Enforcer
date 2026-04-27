"""Internal parser package used by the public `core.parser` facade."""

from core.code_parser.advanced_signals import (
    build_advanced_validation_payload,
    has_advanced_validation_signals,
)
from core.code_parser.service import parse_source_code

__all__ = [
    "build_advanced_validation_payload",
    "has_advanced_validation_signals",
    "parse_source_code",
]
