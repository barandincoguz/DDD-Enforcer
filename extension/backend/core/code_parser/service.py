from __future__ import annotations

import ast
from typing import Any, Dict

from core.code_parser.visitor import CodeStructureVisitor


def parse_source_code(source_code: str, filename: str = "") -> Dict[str, Any]:
    """Parse Python source into a normalized, backward-compatible AST summary."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as error:
        return {"error": f"Syntax Error: {error}"}

    visitor = CodeStructureVisitor()
    visitor.visit(tree)
    return visitor.build_result(filename).to_public_dict()
