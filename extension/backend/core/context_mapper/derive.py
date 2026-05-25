"""Pure derivation of flat allowed_dependencies from a typed ContextMap.

Directional relationships add a downstream->upstream edge; mutual ones add both
directions; non-integration ones add nothing. Relationships referencing unknown
contexts are dropped. Non-mutual cycles are reported (warning) but edges are
kept — production does not hard-gate on D11 (it is unwired), so the correct
treatment is loud-but-non-fatal."""
from typing import Dict, List, Set, Tuple
from core.schemas import ContextMap

_DIRECTIONAL = {"CUSTOMER_SUPPLIER", "CONFORMIST", "ANTI_CORRUPTION_LAYER",
                "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE"}
_MUTUAL = {"PARTNERSHIP", "SHARED_KERNEL"}


def derive_allowed_dependencies(
    cmap: ContextMap, valid_names: Set[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    warnings: List[str] = []
    deps: Dict[str, Set[str]] = {}
    mutual_edges: Set[Tuple[str, str]] = set()

    for r in cmap.relationships:
        if r.context_a not in valid_names or r.context_b not in valid_names:
            warnings.append(
                f"dropped relationship {r.context_a}/{r.context_b} "
                f"({r.relationship_type}): unknown context name")
            continue
        if r.relationship_type in _DIRECTIONAL:
            # Schema guarantees a directional relationship's `upstream` is one of
            # context_a/context_b (validator raises otherwise), so exactly one of
            # these branches holds and both sides are non-None `str`.
            if r.upstream == r.context_a:
                upstream, downstream = r.context_a, r.context_b
            else:
                upstream, downstream = r.context_b, r.context_a
            deps.setdefault(downstream, set()).add(upstream)
        elif r.relationship_type in _MUTUAL:
            deps.setdefault(r.context_a, set()).add(r.context_b)
            deps.setdefault(r.context_b, set()).add(r.context_a)
            mutual_edges.add((r.context_a, r.context_b))
            mutual_edges.add((r.context_b, r.context_a))
        # SEPARATE_WAYS / BIG_BALL_OF_MUD -> no edges

    cycle = _find_cycle(deps, mutual_edges)
    if cycle:
        warnings.append(f"non-mutual dependency cycle: {' -> '.join(cycle)}")

    return {k: sorted(v) for k, v in deps.items()}, warnings


def _find_cycle(
    graph: Dict[str, Set[str]], mutual_edges: Set[Tuple[str, str]],
) -> List[str]:
    """DFS (WHITE/GRAY/BLACK) cycle detection, EXCLUDING mutual edges (which are
    intentional 2-cycles). Returns the first cycle path found, or []."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in graph}
    parent: Dict[str, str] = {}

    def neighbors(n: str) -> List[str]:
        return [m for m in graph.get(n, ()) if (n, m) not in mutual_edges]

    def visit(n: str) -> List[str]:
        color[n] = GRAY
        for m in neighbors(n):
            if color.get(m, WHITE) == GRAY:
                path = [m, n]
                cur = parent.get(n)
                while cur is not None and cur != m:
                    path.append(cur)
                    cur = parent.get(cur)
                path.append(m)
                return list(reversed(path))
            if color.get(m, WHITE) == WHITE:
                parent[m] = n
                found = visit(m)
                if found:
                    return found
        color[n] = BLACK
        return []

    for node in list(graph):
        if color[node] == WHITE:
            found = visit(node)
            if found:
                return found
    return []
