"""
Code Parser

Parses Python source code using AST to extract structural information
needed for DDD violation detection. Does not execute the code.
"""

import ast
from typing import Any, Dict, List, Optional


class CodeParser:
    """
    Parses Python source code using AST (Abstract Syntax Tree).

    Extracts classes, methods, functions, and imports for DDD validation.
    """

    def parse_code(self, source_code: str, filename: str = "") -> Dict[str, Any]:
        """
        Parse source code and return structural metadata.

        Args:
            source_code: Python source code to parse
            filename: Name of the file being parsed (for file-level violations)

        Returns dict with classes, imports, functions, and code patterns,
        or error if the code has syntax errors.
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            return {"error": f"Syntax Error: {e}"}

        visitor = DomainVisitor()
        visitor.visit(tree)

        return {
            "filename": filename,
            "classes": visitor.classes,
            "imports": visitor.imports,
            "functions": visitor.functions,
            "assignments": visitor.assignments,
            "function_calls": visitor.function_calls,
        }


class DomainVisitor(ast.NodeVisitor):
    """AST visitor that extracts domain-relevant code elements."""

    def __init__(self):
        self.classes: List[Dict[str, Any]] = []
        self.imports: List[Dict[str, Any]] = []  # Changed to dict for detailed info
        self.functions: List[Dict[str, Any]] = []  # Changed to dict for parameters
        self.assignments: List[Dict[str, Any]] = []
        self.function_calls: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None

    def visit_Import(self, node: ast.Import):
        """Record import statements with full details."""
        for alias in node.names:
            self.imports.append(
                {"module": alias.name, "type": "import", "line": node.lineno}
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Record from-import statements with module path."""
        if node.module:
            imported_names = [alias.name for alias in node.names]
            self.imports.append(
                {
                    "module": node.module,
                    "names": imported_names,
                    "type": "from",
                    "line": node.lineno,
                }
            )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class information including name, bases, and methods."""
        prev_class = self.current_class
        self.current_class = node.name

        class_info = {
            "name": node.name,
            "line": node.lineno,
            "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
            "methods": [
                item.name for item in node.body if isinstance(item, ast.FunctionDef)
            ],
        }
        self.classes.append(class_info)
        self.generic_visit(node)

        self.current_class = prev_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function/method details including parameters."""
        params = []
        for arg in node.args.args:
            param_info = {"name": arg.arg}
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    param_info["type"] = arg.annotation.id
                elif isinstance(arg.annotation, ast.Constant):
                    param_info["type"] = str(arg.annotation.value)
            params.append(param_info)

        func_info = {
            "name": node.name,
            "line": node.lineno,
            "parameters": params,
            "in_class": self.current_class,
        }
        self.functions.append(func_info)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        """Track assignments for Value Object violation detection."""
        for target in node.targets:
            assignment_info = {"line": node.lineno}

            # Capture target (e.g., order.status)
            if isinstance(target, ast.Attribute):
                if isinstance(target.value, ast.Name):
                    assignment_info["target"] = f"{target.value.id}.{target.attr}"
            elif isinstance(target, ast.Name):
                assignment_info["target"] = target.id

            # Capture value type
            if isinstance(node.value, ast.Constant):
                assignment_info["value_type"] = type(node.value.value).__name__
                assignment_info["value"] = str(node.value.value)
            elif isinstance(node.value, ast.Name):
                assignment_info["value_type"] = "variable"

            self.assignments.append(assignment_info)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Track function calls for domain event violations."""
        call_info = {"line": node.lineno}

        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                call_info["function"] = f"{node.func.value.id}.{node.func.attr}"
            else:
                call_info["function"] = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_info["function"] = node.func.id

        # Capture arguments (especially string literals for event names)
        call_info["args"] = []
        for arg in node.args:
            if isinstance(arg, ast.Constant):
                call_info["args"].append(str(arg.value))

        self.function_calls.append(call_info)
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
