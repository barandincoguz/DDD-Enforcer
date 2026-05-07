"""Pytest configuration for DDD-Enforcer backend tests.

Sets a placeholder GEMINI_API_KEY before tests collect so that modules
which read it at import time do not blow up in CI environments without a
real key. Tests asserting actual LLM behavior must mock the genai client
directly (see tests/test_architect_helpers.py for the pattern).
"""

import os

os.environ.setdefault(
    "GEMINI_API_KEY",
    "test-placeholder-not-a-real-key",
)
