from __future__ import annotations

import ast
from typing import List, Optional

from core.ast_helpers import (
    annotation_to_string,
    dotted_name_from_expr,
    extract_direct_attribute,
    safe_unparse,
)
from core.code_parser.helpers import (
    append_unique,
    build_parameter_records,
    call_kind,
    literal_keyword_args,
    literal_text,
    receiver_name,
    target_kind,
    target_string,
    value_metadata,
)
from core.code_parser.models import (
    AssignmentRecord,
    ClassRecord,
    FunctionCallRecord,
    FunctionRecord,
    ImportRecord,
    ParseResult,
)


COLLECTION_MUTATORS = {
    "append",
    "extend",
    "insert",
    "add",
    "remove",
    "pop",
    "clear",
    "discard",
    "update",
}


class CodeStructureVisitor(ast.NodeVisitor):
    """Collect a normalized, backward-compatible view of Python structure."""

    def __init__(self) -> None:
        self.classes: List[ClassRecord] = []
        self.imports: List[ImportRecord] = []
        self.functions: List[FunctionRecord] = []
        self.assignments: List[AssignmentRecord] = []
        self.function_calls: List[FunctionCallRecord] = []
        self._class_stack: List[ClassRecord] = []
        self._function_stack: List[FunctionRecord] = []

    def build_result(self, filename: str) -> ParseResult:
        return ParseResult(
            filename=filename,
            classes=self.classes,
            imports=self.imports,
            functions=self.functions,
            assignments=self.assignments,
            function_calls=self.function_calls,
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.append(
                ImportRecord(
                    module=alias.name,
                    line=node.lineno,
                    import_type="import",
                    asname=alias.asname,
                    imported_modules=[alias.name],
                )
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        members = [{"name": alias.name, "asname": alias.asname} for alias in node.names]
        imported_modules = [
            ".".join(part for part in [module, alias.name] if part) for alias in node.names
        ]
        self.imports.append(
            ImportRecord(
                module=module,
                line=node.lineno,
                import_type="from",
                names=[alias.name for alias in node.names],
                level=node.level,
                members=members,
                imported_modules=imported_modules,
            )
        )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        record = ClassRecord(
            name=node.name,
            line=node.lineno,
            bases=[
                base_name
                for base in node.bases
                if (base_name := dotted_name_from_expr(base))
            ],
            decorators=[
                decorator_name
                for decorator in node.decorator_list
                if (decorator_name := safe_unparse(decorator))
            ],
        )
        record.base_names = [base.rsplit(".", 1)[-1] for base in record.bases]
        record.is_dataclass = any(self._is_dataclass_decorator(name) for name in record.decorators)
        record.is_frozen = any(
            self._is_dataclass_decorator(name) and "frozen=True" in name.replace(" ", "")
            for name in record.decorators
        )
        self.classes.append(record)
        self._class_stack.append(record)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node, is_async=True)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_assignment(target=target, value=node.value, line=node.lineno, operation="=")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_assignment(
            target=node.target,
            value=node.value,
            line=node.lineno,
            operation="=",
            annotation=annotation_to_string(node.annotation),
        )
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record_assignment(
            target=node.target,
            value=node.value,
            line=node.lineno,
            operation=self._operation_symbol(node.op),
        )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        record = FunctionCallRecord(
            line=node.lineno,
            function=dotted_name_from_expr(node.func),
            args=[value for arg in node.args if (value := literal_text(arg)) is not None],
            kwargs=literal_keyword_args(node.keywords),
            receiver=receiver_name(node.func),
            call_kind=call_kind(node.func),
            in_class=self._current_class_name(),
            in_function=self._current_function_name(),
            nesting_level=self._current_nesting_level(),
        )
        self.function_calls.append(record)
        self._track_mutating_call(node)
        self.generic_visit(node)

    def _visit_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> None:
        record = FunctionRecord(
            name=node.name,
            line=node.lineno,
            parameters=build_parameter_records(node.args),
            in_class=self._current_class_name(),
            in_function=self._current_function_name(),
            nesting_level=len(self._function_stack),
            return_annotation=annotation_to_string(node.returns),
            decorators=[
                decorator_name
                for decorator in node.decorator_list
                if (decorator_name := safe_unparse(decorator))
            ],
            is_async=is_async,
        )
        self.functions.append(record)

        if self._class_stack and not self._function_stack:
            append_unique(self._class_stack[-1].methods, node.name)

        self._function_stack.append(record)
        self.generic_visit(node)
        self._function_stack.pop()

    def _record_assignment(
        self,
        target: ast.AST,
        value: Optional[ast.AST],
        line: int,
        operation: str,
        annotation: Optional[str] = None,
    ) -> None:
        record = AssignmentRecord(
            line=line,
            target=target_string(target),
            target_kind=target_kind(target),
            in_class=self._current_class_name(),
            in_function=self._current_function_name(),
            nesting_level=self._current_nesting_level(),
            annotation=annotation,
            operation=operation,
            **value_metadata(value),
        )
        self.assignments.append(record)
        self._track_class_attribute(target)
        self._track_instance_attribute(target)

    def _track_class_attribute(self, target: ast.AST) -> None:
        if not self._class_stack or self._function_stack:
            return
        if isinstance(target, ast.Name):
            append_unique(self._class_stack[-1].class_attributes, target.id)

    def _track_instance_attribute(self, target: ast.AST) -> None:
        if not self._class_stack or not self._function_stack:
            return
        attribute = extract_direct_attribute(target)
        if attribute is None:
            return
        append_unique(self._class_stack[-1].instance_attributes, attribute)
        self._mark_current_method_as_mutating()

    def _track_mutating_call(self, node: ast.Call) -> None:
        if not self._class_stack or not isinstance(node.func, ast.Attribute):
            return
        if node.func.attr not in COLLECTION_MUTATORS:
            return
        attribute = extract_direct_attribute(node.func.value)
        if attribute is None:
            return
        append_unique(self._class_stack[-1].instance_attributes, attribute)
        self._mark_current_method_as_mutating()

    def _mark_current_method_as_mutating(self) -> None:
        if not self._class_stack:
            return
        method_name = self._current_class_method_name()
        if method_name:
            append_unique(self._class_stack[-1].mutation_methods, method_name)

    def _current_class_method_name(self) -> Optional[str]:
        for record in self._function_stack:
            if record.in_class == self._current_class_name() and record.in_function is None:
                return record.name
        return None

    def _current_class_name(self) -> Optional[str]:
        if not self._class_stack:
            return None
        return self._class_stack[-1].name

    def _current_function_name(self) -> Optional[str]:
        if not self._function_stack:
            return None
        return self._function_stack[-1].name

    def _current_nesting_level(self) -> int:
        if not self._function_stack:
            return 0
        return len(self._function_stack) - 1

    def _operation_symbol(self, node: ast.operator) -> str:
        names = {
            ast.Add: "+=",
            ast.Sub: "-=",
            ast.Mult: "*=",
            ast.Div: "/=",
            ast.Mod: "%=",
            ast.Pow: "**=",
            ast.FloorDiv: "//=",
            ast.BitAnd: "&=",
            ast.BitOr: "|=",
            ast.BitXor: "^=",
            ast.LShift: "<<=",
            ast.RShift: ">>=",
        }
        return names.get(type(node), f"{type(node).__name__}=")

    def _is_dataclass_decorator(self, value: str) -> bool:
        return value.split("(", 1)[0].rsplit(".", 1)[-1] == "dataclass"
