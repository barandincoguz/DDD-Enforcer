from __future__ import annotations

import ast
from typing import Optional


def safe_unparse(node: ast.AST | None) -> Optional[str]:
    """Return a compact source-like representation for an AST node."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - ast.unparse is stable on supported versions
        return None


def dotted_name_from_expr(node: ast.AST | None) -> Optional[str]:
    """Resolve dotted names such as `pkg.module.Name` from simple expressions."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name_from_expr(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return dotted_name_from_expr(node.func)
    return safe_unparse(node)


def annotation_to_string(node: ast.AST | None) -> Optional[str]:
    """Serialize a type annotation to a stable string form."""
    return safe_unparse(node)


def extract_direct_attribute(
    node: ast.AST | None,
    owner_names: tuple[str, ...] = ("self", "cls"),
) -> Optional[str]:
    """Extract direct owner attributes like `self.amount` -> `amount`."""
    if not isinstance(node, ast.Attribute):
        return None
    if isinstance(node.value, ast.Name) and node.value.id in owner_names:
        return node.attr
    return None
