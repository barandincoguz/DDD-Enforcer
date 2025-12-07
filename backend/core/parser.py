import ast
from typing import List, Dict, Any

class CodeParser:
    """
    Parses Python source code using the AST (Abstract Syntax Tree) module.
    Extracts high-level structure needed for DDD validation without executing the code.
    """

    def parse_code(self, source_code: str) -> Dict[str, Any]:
        """
        Main entry point. Takes source code string, returns structured metadata.
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
    def __init__(self):
        self.classes = []
        self.imports = []
        self.functions = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        class_info = {
            "name": node.name,
            "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
            "methods": []
        }
        
        # Sınıf içindeki metodları gez
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_info["methods"].append(item.name)
                
        self.classes.append(class_info)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Sınıf dışındaki global fonksiyonlar
        self.functions.append(node.name)
        self.generic_visit(node)

# --- Test etmek için küçük bir blok (Main içinde çalışmayacak) ---
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