from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

from core.ast_helpers import annotation_to_string, dotted_name_from_expr, safe_unparse
from core.code_parser.models import ParameterRecord


def append_unique(items: List[str], value: Optional[str]) -> None:
    if value and value not in items:
        items.append(value)


def build_parameter_records(arguments: ast.arguments) -> List[ParameterRecord]:
    records: List[ParameterRecord] = []
    positional = [("positional_only", arg) for arg in arguments.posonlyargs]
    positional.extend(("positional_or_keyword", arg) for arg in arguments.args)
    defaults = [None] * (len(positional) - len(arguments.defaults)) + list(arguments.defaults)

    for (kind, arg), default in zip(positional, defaults):
        records.append(
            ParameterRecord(
                name=arg.arg,
                annotation=annotation_to_string(arg.annotation),
                kind=kind,
                has_default=default is not None,
                default=safe_unparse(default),
            )
        )

    if arguments.vararg:
        records.append(
            ParameterRecord(
                name=arguments.vararg.arg,
                annotation=annotation_to_string(arguments.vararg.annotation),
                kind="var_positional",
            )
        )

    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        records.append(
            ParameterRecord(
                name=arg.arg,
                annotation=annotation_to_string(arg.annotation),
                kind="keyword_only",
                has_default=default is not None,
                default=safe_unparse(default),
            )
        )

    if arguments.kwarg:
        records.append(
            ParameterRecord(
                name=arguments.kwarg.arg,
                annotation=annotation_to_string(arguments.kwarg.annotation),
                kind="var_keyword",
            )
        )

    return records


def target_kind(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.Attribute):
        return "attribute"
    if isinstance(node, ast.Subscript):
        return "subscript"
    if isinstance(node, (ast.Tuple, ast.List)):
        return "unpack"
    return type(node).__name__.lower()


def target_string(node: ast.AST) -> Optional[str]:
    return safe_unparse(node)


def literal_text(node: ast.AST | None) -> Optional[str]:
    if isinstance(node, ast.Constant):
        return str(node.value)
    return None


def literal_keyword_args(keywords: List[ast.keyword]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for keyword in keywords:
        if keyword.arg is None:
            continue
        value = literal_text(keyword.value)
        if value is not None:
            values[keyword.arg] = value
    return values


def receiver_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Attribute):
        return dotted_name_from_expr(func.value)
    return None


def call_kind(func: ast.AST) -> str:
    if isinstance(func, ast.Attribute):
        return "attribute"
    if isinstance(func, ast.Name):
        return "name"
    return type(func).__name__.lower()


def value_metadata(node: ast.AST | None) -> Dict[str, Optional[str]]:
    if node is None:
        return {"value_type": None, "value_shape": None, "value": None}

    if isinstance(node, ast.Constant):
        value_type = type(node.value).__name__
        return {
            "value_type": value_type,
            "value_shape": value_type,
            "value": str(node.value),
        }

    if isinstance(node, ast.Name):
        return {
            "value_type": "variable",
            "value_shape": f"name:{node.id}",
            "value": node.id,
        }

    if isinstance(node, ast.Attribute):
        name = dotted_name_from_expr(node)
        return {
            "value_type": "attribute",
            "value_shape": f"attribute:{name}" if name else "attribute",
            "value": name,
        }

    if isinstance(node, ast.Call):
        call_name = dotted_name_from_expr(node.func)
        return {
            "value_type": "call",
            "value_shape": f"call:{call_name}" if call_name else "call",
            "value": call_name,
        }

    if isinstance(node, ast.List):
        return {"value_type": "list", "value_shape": "list", "value": None}
    if isinstance(node, ast.Dict):
        return {"value_type": "dict", "value_shape": "dict", "value": None}
    if isinstance(node, ast.Set):
        return {"value_type": "set", "value_shape": "set", "value": None}
    if isinstance(node, ast.Tuple):
        return {"value_type": "tuple", "value_shape": "tuple", "value": None}

    source = safe_unparse(node)
    node_type = type(node).__name__.lower()
    return {"value_type": node_type, "value_shape": node_type, "value": source}
