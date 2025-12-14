"""
Code Parser

Parses Python source code using AST to extract structural information
needed for DDD violation detection. Does not execute the code.
"""

import ast
from typing import Any, Dict, List


class CodeParser:
    """
    Parses Python source code using AST (Abstract Syntax Tree).

    Extracts classes, methods, functions, and imports for DDD validation.
    """

    def parse_code(self, source_code: str) -> Dict[str, Any]:
        """
        Parse source code and return structural metadata.

        Returns dict with classes, imports, and functions, or error if
        the code has syntax errors.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return {"error": f"Syntax Error: {e}"}

        visitor = DomainVisitor()
        visitor.visit(tree)

        return {
            "classes": visitor.classes,
            "imports": visitor.imports,
            "functions": visitor.functions
        }


class DomainVisitor(ast.NodeVisitor):
    """AST visitor that extracts domain-relevant code elements."""

    def __init__(self):
        self.classes: List[Dict] = []
        self.imports: List[str] = []
        self.functions: List[str] = []

    def visit_Import(self, node: ast.Import):
        """Record import statements."""
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Record from-import statements."""
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class information including name, bases, and methods."""
        class_info = {
            "name": node.name,
            "bases": [
                base.id for base in node.bases
                if isinstance(base, ast.Name)
            ],
            "methods": [
                item.name for item in node.body
                if isinstance(item, ast.FunctionDef)
            ]
        }
        self.classes.append(class_info)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Record top-level function names (not class methods)."""
        self.functions.append(node.name)
        self.generic_visit(node)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    sample_code = """
from sales import Order
import datetime

class ClientManager:
    def create_client(self, name):
        pass

def calculate_total():
    pass
"""
    parser = CodeParser()
    print(parser.parse_code(sample_code))
