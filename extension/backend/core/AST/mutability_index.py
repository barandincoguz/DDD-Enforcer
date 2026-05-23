"""WP-CORE-27a — AST-derived mutability index.

A pure helper that walks every Python class in the supplied file set
and returns whether each class exposes mutation methods (the AST signal
that turns the D9 verifier's `is_mutable_in_code` field non-vacuous).

Used by `ASTModelSignalExtractor.enrich_domain_model` to cross-reference
LLM-claimed Value Objects against the actual code shape: a class that
the LLM labelled "VO" but whose AST shows mutation methods is a D9
violation (semantic contradiction between declared invariant and
implemented behaviour).

Side-effect-free.  Parse failures are swallowed silently (mirrors the
documented behaviour of `core.AST.import_graph.build_module_import_graph`
and the discovery helpers it composes with).
"""
from __future__ import annotations

from typing import Dict, List

from core.AST.ast_signal_discovery import extract_class_facts


def build_mutability_index(python_files: List[str]) -> Dict[str, bool]:
    """For each Python class in `python_files`, return
    `{class_name_lower: is_mutable}` where `is_mutable` is True iff the
    AST shows at least one mutation method on the class.

    Conflict resolution: if two source files define a class with the
    same name, the index entry collapses to `True` whenever ANY
    definition is mutable.  This is intentional — D9's risk model is
    asymmetric: a single mutable definition is enough to make the LLM's
    VO claim suspect.

    Parse / IO failures: silently skipped.  No exception is propagated
    from this function; callers that need parse-failure surfacing should
    pre-validate the file set.
    """
    index: Dict[str, bool] = {}
    for path in python_files:
        try:
            for facts in extract_class_facts(path):
                key = facts.name.lower()
                is_mutable = bool(facts.mutation_methods)
                if is_mutable:
                    index[key] = True
                else:
                    index.setdefault(key, False)
        except (SyntaxError, UnicodeDecodeError, OSError, ValueError):
            continue
    return index
