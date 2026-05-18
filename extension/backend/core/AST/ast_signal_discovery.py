from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import List, Optional

from core.ast_helpers import dotted_name_from_expr, extract_direct_attribute
from core.AST.ast_signal_types import ClassFacts
from core.AST.ast_signal_utils import collect_terms, normalized_terms


DEFAULT_SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "site-packages",
    "migrations",
    "tests",
    "test",
}

DEPENDENCY_SUFFIXES = (
    "repo",
    "repository",
    "service",
    "client",
    "adapter",
    "gateway",
    "bus",
    "publisher",
    "store",
    "factory",
)
COLLECTION_METHODS = {"append", "extend", "add", "remove", "pop", "clear", "update"}
MUTATION_PREFIXES = (
    "add_",
    "remove_",
    "update_",
    "change_",
    "set_",
    "mark_",
    "activate",
    "deactivate",
    "apply_",
    "complete",
    "cancel",
)
EVENT_CALL_NAMES = {"publish", "emit", "dispatch", "notify", "raise_event"}


def find_python_files(
    workspace_path: str,
    skip_dirs: Optional[set[str]] = None,
    ignore_paths: Optional[List[str]] = None,
) -> List[str]:
    root = Path(workspace_path)
    if not root.exists() or not root.is_dir():
        return []

    ignored = ignore_paths or []
    discovered: List[str] = []
    for current_root, dirs, filenames in os.walk(root):
        dirs[:] = [name for name in dirs if name not in (skip_dirs or DEFAULT_SKIP_DIRS)]
        rel_path = os.path.relpath(current_root, root).replace("\\", "/")
        if any(part in rel_path for part in ignored):
            continue
        for name in filenames:
            if name.endswith(".py"):
                discovered.append(str(Path(current_root) / name))
    return discovered


def extract_class_facts(file_path: str) -> List[ClassFacts]:
    try:
        source = _read_source(file_path)
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError, OSError) as exc:
        # G01: raise instead of silently returning an empty fact list.
        # The caller (_collect_signals) decides whether to count this
        # toward run-failure metrics or surface to the user.
        raise ValueError(f"AST parse failed for {file_path}: {exc}") from exc

    path_parts = list(Path(file_path).with_suffix("").parts[-4:])
    module_tokens = collect_terms(path_parts)
    return [
        _build_class_facts(node, file_path, module_tokens)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    ]


def _read_source(file_path: str) -> str:
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            return Path(file_path).read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return Path(file_path).read_text(encoding="utf-8", errors="ignore")


def _build_class_facts(
    node: ast.ClassDef,
    file_path: str,
    module_tokens: set[str],
) -> ClassFacts:
    facts = ClassFacts(
        name=node.name,
        file_path=file_path,
        line=node.lineno,
        bases={
            dotted_name_from_expr(base)
            for base in node.bases
            if dotted_name_from_expr(base)
        },
        module_tokens=set(module_tokens),
        name_tokens=normalized_terms(node.name),
    )
    facts.method_tokens = set()

    for decorator in node.decorator_list:
        name = dotted_name_from_expr(decorator)
        if name:
            facts.decorators.add(name)
            if name == "dataclass":
                facts.is_dataclass = True
        if isinstance(decorator, ast.Call):
            call_name = dotted_name_from_expr(decorator.func)
            if call_name == "dataclass":
                facts.is_dataclass = True
                if any(
                    keyword.arg == "frozen"
                    and isinstance(keyword.value, ast.Constant)
                    and bool(keyword.value.value)
                    for keyword in decorator.keywords
                ):
                    facts.is_frozen = True

    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            _collect_method_facts(facts, item)
            continue
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            facts.class_attributes.add(item.target.id)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    facts.class_attributes.add(target.id)

    return facts


def _collect_method_facts(facts: ClassFacts, method: ast.FunctionDef) -> None:
    facts.methods.add(method.name)
    if not method.name.startswith("__"):
        facts.public_methods.add(method.name)
        facts.method_tokens.update(normalized_terms(method.name))
    if method.name == "__init__":
        facts.constructor_params.update(arg.arg for arg in method.args.args[1:])

    if method.name.startswith(MUTATION_PREFIXES):
        facts.mutation_methods.add(method.name)

    for inner in ast.walk(method):
        if isinstance(inner, ast.Call):
            _collect_call_facts(facts, inner, method.name)
        elif isinstance(inner, ast.Assign):
            for target in inner.targets:
                _collect_assignment_facts(facts, method.name, target, inner.value)
        elif isinstance(inner, ast.AnnAssign):
            _collect_assignment_facts(facts, method.name, inner.target, inner.value)


def _collect_assignment_facts(
    facts: ClassFacts,
    method_name: str,
    target: ast.AST,
    value: Optional[ast.AST],
) -> None:
    attr = extract_direct_attribute(target)
    if not attr:
        return

    facts.instance_attributes.add(attr)
    attr_lower = attr.lower()
    if attr_lower.endswith(DEPENDENCY_SUFFIXES):
        facts.dependency_attributes.add(attr)
    if _looks_like_collection_name(attr) or _is_collection_value(value):
        facts.collection_attributes.add(attr)
    if method_name != "__init__":
        facts.mutation_methods.add(method_name)


def _collect_call_facts(facts: ClassFacts, call: ast.Call, method_name: str) -> None:
    call_name = dotted_name_from_expr(call.func)
    if call_name and call_name.split(".")[-1] in EVENT_CALL_NAMES:
        for arg in call.args:
            event_name = _event_name_from_arg(arg)
            if event_name:
                facts.event_names.add(event_name)
    if isinstance(call.func, ast.Attribute) and call.func.attr in COLLECTION_METHODS:
        owner = extract_direct_attribute(call.func.value)
        if owner:
            facts.collection_attributes.add(owner)
            facts.mutation_methods.add(method_name)


def _is_collection_value(value: Optional[ast.AST]) -> bool:
    if isinstance(value, (ast.List, ast.Set, ast.Dict, ast.Tuple)):
        return True
    if isinstance(value, ast.Call):
        return _name_from_expr(value.func) in {"list", "set", "dict", "tuple"}
    return False


def _looks_like_collection_name(attr: str) -> bool:
    lowered = attr.lower()
    return lowered in {"items", "entries", "lines", "orders", "products", "events"} or (
        lowered.endswith("s") and lowered not in {"status", "address", "class"}
    )


def _event_name_from_arg(arg: ast.AST) -> Optional[str]:
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Attribute):
        return arg.attr
    return None
