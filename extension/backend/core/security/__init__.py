"""Security primitives (F-8 XXE defense-in-depth + future entries)."""

from core.security.xxe_safety import (
    XXESafetyError,
    assert_xxe_safe_parsers,
)

__all__ = [
    "XXESafetyError",
    "assert_xxe_safe_parsers",
]
