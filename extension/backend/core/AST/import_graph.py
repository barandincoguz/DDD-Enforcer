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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


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


def apply_import_topology_to_model(
    model_data: Dict[str, Any],
    python_files: List[str],
    workspace_root: Optional[str] = None,
) -> Dict[str, Any]:
    """WP-CORE-31b: pipeline-integration helper.

    Compose `build_module_import_graph` -> `map_modules_to_contexts` ->
    `derive_context_dependencies` on the workspace's Python files and
    push the resulting context-level dependency graph into the supplied
    `model_data` dict (mutating in place):

    * If a `BoundedContext.allowed_dependencies` is `None` or `[]` AND
      the AST-derived graph found cross-context import edges originating
      at that context, set `allowed_dependencies` to the sorted derived
      list and record the context name under `auto_populated`.
      EXCEPTION: when `model_data["context_map"]` is present with no
      `error` (A authoritatively produced the map), an empty
      `allowed_dependencies` is left untouched — the derived imports are
      recorded under `cross_check_diff` (diagnostics-only) so a missed
      relationship stays visible without overriding the strategic map.
    * If the LLM already declared `allowed_dependencies` AND the derived
      graph is non-empty for the same context, do NOT overwrite — instead
      record the sym-diff under `cross_check_diff` so reviewers see drift.
    * Unmapped contexts (those whose name appears nowhere in any module
      path) are skipped — auto-populating an unmapped context with `[]`
      would be misleading.

    Returns a JSON-safe diagnostics dict carrying:
        {
            "derived":           {ctx -> sorted([dep, ...])},
            "auto_populated":    [ctx, ...]  (sorted),
            "cross_check_diff":  {ctx -> {"extra_in_llm": [...], "extra_in_derived": [...]}},
        }
    """
    contexts = model_data.get("bounded_contexts", []) or []
    if not contexts:
        return {"derived": {}, "auto_populated": [], "cross_check_diff": {}}

    # When A (the context-mapper) authoritatively produced the map, an
    # intentionally-empty `allowed_dependencies` (e.g. a SEPARATE_WAYS context)
    # MUST NOT be repopulated from code imports — doing so would let V4 allow an
    # import the strategic map forbids. Switch to diagnostics-only in that case.
    # If `context_map` is absent or carries an error (A failed/disabled), keep
    # the legacy auto-fill behavior (no regression).
    cm = model_data.get("context_map")
    map_authoritative = bool(cm) and cm.get("error") is None

    context_names = [str(c.get("context_name", "")) for c in contexts]
    graph = build_module_import_graph(python_files, workspace_root=workspace_root)
    # Collect every module that participates in the graph (sources + targets)
    # so unmapped imports (stdlib, third-party) don't accidentally bleed into
    # a context bucket via a partial mapping.
    all_modules: Set[str] = set(graph.keys())
    for imports in graph.values():
        all_modules |= imports
    mapping = map_modules_to_contexts(all_modules, context_names)
    derived = derive_context_dependencies(graph, mapping)

    auto_populated: List[str] = []
    cross_check_diff: Dict[str, Dict[str, List[str]]] = {}
    for context in contexts:
        ctx_name = str(context.get("context_name", ""))
        derived_set: Set[str] = set(derived.get(ctx_name, set()))
        if not derived_set:
            continue
        existing = context.get("allowed_dependencies")
        if not existing:
            if map_authoritative:
                # A authoritatively declared deps (possibly empty, e.g. Separate
                # Ways). Do NOT repopulate — only record what code imports show.
                cross_check_diff[ctx_name] = {
                    "extra_in_llm": [],
                    "extra_in_derived": sorted(derived_set),
                }
                continue
            context["allowed_dependencies"] = sorted(derived_set)
            auto_populated.append(ctx_name)
            continue
        existing_set = {str(item) for item in existing if item is not None}
        extra_llm = sorted(existing_set - derived_set)
        extra_derived = sorted(derived_set - existing_set)
        if extra_llm or extra_derived:
            cross_check_diff[ctx_name] = {
                "extra_in_llm": extra_llm,
                "extra_in_derived": extra_derived,
            }

    return {
        "derived": {ctx: sorted(deps) for ctx, deps in derived.items()},
        "auto_populated": sorted(auto_populated),
        "cross_check_diff": cross_check_diff,
    }


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
