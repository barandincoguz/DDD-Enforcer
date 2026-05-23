"""WP-CORE-31 — Cross-context import topology.

Three pure helpers that extract a directed dependency graph between
bounded contexts from the workspace Python source — independent of
whatever `allowed_dependencies` the LLM happens to emit.

Pipeline integration (auto-populate `BoundedContext.allowed_dependencies`
+ cross-check the derived graph against the LLM-declared graph) is a
follow-up (WP-CORE-31b). v1 ships the standalone helpers + tests so
downstream WPs can compose them without touching the pipeline hot path.

Helpers:
- `build_module_import_graph(python_files)` → module-name → set of imported module names.
- `map_modules_to_contexts(modules, context_names)` → module-name → context-name (or None).
- `derive_context_dependencies(import_graph, module_to_context)` → context-name → set of dependency context names.

Naming convention:
- `module_name` is the dotted Python module path (e.g., `billing.service`).
- `context_name` is whatever the LLM emitted as `BoundedContext.context_name`
  (e.g., `"Billing"` — typically PascalCase).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set


def _normalize(token: str) -> str:
    """Case-fold + strip underscores/hyphens for matching."""
    return token.lower().replace("_", "").replace("-", "")


def _file_to_module_name(file_path: str, workspace_root: Optional[str] = None) -> str:
    """Derive a dotted module path from a file path.

    With workspace_root: relative to root, with `.py` stripped.
    Without: best-effort using the path's parts (last 3 directories +
    file stem, joined by '.').
    """
    p = Path(file_path).with_suffix("")
    if workspace_root:
        try:
            rel = p.resolve().relative_to(Path(workspace_root).resolve())
            parts = rel.parts
        except ValueError:
            parts = p.parts[-4:]
    else:
        parts = p.parts[-4:]
    return ".".join(parts)


def _extract_imports(file_path: str) -> Set[str]:
    """Return the set of imported module names from a single .py file.

    Both `import x.y` and `from x.y import z` forms produce `x.y`.
    Relative imports (`from . import foo`) produce `.foo` and are
    intentionally retained as-is — context-mapping resolves them as
    same-package by token match.

    Parse errors propagate (callers decide whether to skip; mirrors
    `extract_class_facts` behavior in ast_signal_discovery.py).
    """
    source = Path(file_path).read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(file_path))
    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and not module:
                # `from . import foo` — record the relative root only.
                imports.add("." * node.level)
            elif module:
                imports.add(("." * node.level) + module if node.level else module)
    return imports


def build_module_import_graph(
    python_files: List[str],
    workspace_root: Optional[str] = None,
) -> Dict[str, Set[str]]:
    """For each input .py file, return {module_name: set_of_imported_modules}.

    Files that fail to parse are silently skipped — callers that want
    parse-failure surfacing should pre-validate.
    """
    graph: Dict[str, Set[str]] = {}
    for path in python_files:
        try:
            imports = _extract_imports(path)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        module_name = _file_to_module_name(path, workspace_root=workspace_root)
        graph[module_name] = imports
    return graph


def map_modules_to_contexts(
    modules: Iterable[str],
    context_names: List[str],
) -> Dict[str, Optional[str]]:
    """Map each module name to a bounded-context name via path-segment match.

    Heuristic: a module belongs to a context if any dotted segment of
    its name matches (case-insensitive, normalised) a context name.
    First match wins (closest-to-root segment is usually the most
    informative).

    Modules with no matching segment map to None — out-of-context modules
    (e.g., shared utilities) are intentionally retained so callers can
    decide whether to treat them as "unmapped" or "shared".
    """
    norm_ctx = {_normalize(n): n for n in context_names}
    mapping: Dict[str, Optional[str]] = {}
    for module in modules:
        ctx: Optional[str] = None
        for segment in module.split("."):
            np = _normalize(segment)
            if np in norm_ctx:
                ctx = norm_ctx[np]
                break
        mapping[module] = ctx
    return mapping


def derive_context_dependencies(
    import_graph: Mapping[str, Set[str]],
    module_to_context: Mapping[str, Optional[str]],
) -> Dict[str, Set[str]]:
    """Reduce a module-level import graph to a context-level dependency graph.

    For each import edge `module_A → module_B`:
    - Resolve both endpoints to their bounded context (via `module_to_context`).
    - If either endpoint is None → skip (unmapped / shared utility).
    - If both map to the same context → skip (intra-context import).
    - Else: record `ctx_A → ctx_B` as an allowed-dependency edge.

    Returns: `{context_name: set_of_dependency_context_names}`.
    Contexts that have NO outgoing edges are omitted from the output
    (callers should treat absent → empty set).
    """
    deps: Dict[str, Set[str]] = defaultdict(set)
    for module, imports in import_graph.items():
        src_ctx = module_to_context.get(module)
        if not src_ctx:
            continue
        for imported_module in imports:
            tgt_ctx = module_to_context.get(imported_module)
            if not tgt_ctx:
                # Imported module isn't mapped to any context — skip.
                # Common for stdlib (`os`, `json`) and third-party
                # (`pydantic`, `fastapi`) imports.
                continue
            if tgt_ctx == src_ctx:
                continue
            deps[src_ctx].add(tgt_ctx)
    return dict(deps)
