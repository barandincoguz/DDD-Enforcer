from __future__ import annotations

from typing import Any, Dict, List


EVENT_CALL_NAMES = {"publish", "emit", "dispatch", "notify", "raise_event"}
LOW_SIGNAL_IMPORT_MODULES = {
    "__future__",
    "abc",
    "collections",
    "dataclasses",
    "datetime",
    "decimal",
    "enum",
    "functools",
    "itertools",
    "json",
    "pathlib",
    "typing",
    "uuid",
}
SIGNAL_LIMIT = 20


def has_advanced_validation_signals(ast_data: Dict[str, Any]) -> bool:
    """Prefer filtered rich parser signals, but keep a legacy truthy fallback."""
    payload = build_advanced_validation_payload(ast_data)
    if any(payload.values()):
        return True

    if not _has_rich_parser_metadata(ast_data):
        return bool(
            ast_data.get("imports")
            or ast_data.get("assignments")
            or ast_data.get("function_calls")
        )
    return False


def build_advanced_validation_payload(ast_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Return compact parser evidence that is useful for advanced LLM checks."""
    return {
        "imports": _relevant_imports(ast_data.get("imports", [])),
        "assignments": _relevant_assignments(ast_data.get("assignments", [])),
        "function_calls": _relevant_function_calls(ast_data.get("function_calls", [])),
    }


def _has_rich_parser_metadata(ast_data: Dict[str, Any]) -> bool:
    checks = (
        (ast_data.get("imports", []), {"imported_modules", "level"}),
        (ast_data.get("assignments", []), {"target_kind", "value_shape"}),
        (ast_data.get("function_calls", []), {"kwargs", "receiver", "call_kind"}),
    )
    for records, keys in checks:
        if any(keys & set(record.keys()) for record in records if isinstance(record, dict)):
            return True
    return False


def _relevant_imports(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for record in records:
        module = str(record.get("normalized_module") or record.get("module") or "")
        first_segment = module.split(".", 1)[0]
        is_relative = bool(record.get("is_relative")) or int(record.get("level", 0)) > 0
        if not is_relative and first_segment in LOW_SIGNAL_IMPORT_MODULES:
            continue
        if not module and not record.get("imported_modules"):
            continue
        selected.append(
            {
                "module": record.get("module"),
                "type": record.get("type"),
                "line": record.get("line"),
                "level": record.get("level", 0),
                "is_relative": is_relative,
                "names": record.get("names", []),
                "imported_modules": record.get("imported_modules", []),
            }
        )
        if len(selected) >= SIGNAL_LIMIT:
            break
    return selected


def _relevant_assignments(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for record in records:
        target_kind = record.get("target_kind")
        annotation = record.get("annotation")
        value_type = record.get("value_type")
        value_shape = str(record.get("value_shape") or "")
        is_typed = bool(annotation)
        is_attribute = target_kind in {"attribute", "subscript"}
        uses_primitive_like_value = value_type in {"str", "int", "float", "bool", "dict", "list", "set", "tuple"}
        is_constructor_call = value_shape.startswith("call:")
        if not (is_typed or is_attribute or uses_primitive_like_value or is_constructor_call):
            continue
        selected.append(
            {
                "target": record.get("target"),
                "target_kind": target_kind,
                "annotation": annotation,
                "value_type": value_type,
                "value_shape": record.get("value_shape"),
                "value": record.get("value"),
                "line": record.get("line"),
                "in_class": record.get("in_class"),
                "in_function": (
                    record.get("in_function")
                    if int(record.get("nesting_level", 0) or 0) == 0
                    else None
                ),
            }
        )
        if len(selected) >= SIGNAL_LIMIT:
            break
    return selected


def _relevant_function_calls(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for record in records:
        function = str(record.get("function") or "")
        if not function:
            continue
        receiver = str(record.get("receiver") or "")
        last_segment = function.rsplit(".", 1)[-1]
        event_like = last_segment in EVENT_CALL_NAMES
        externally_visible = bool(receiver) and receiver not in {"self", "cls"}
        if not (event_like or externally_visible):
            continue
        selected.append(
            {
                "function": record.get("function"),
                "receiver": record.get("receiver"),
                "args": record.get("args", []),
                "kwargs": record.get("kwargs", {}),
                "line": record.get("line"),
                "in_class": record.get("in_class"),
                "in_function": record.get("in_function"),
            }
        )
        if len(selected) >= SIGNAL_LIMIT:
            break
    return selected
