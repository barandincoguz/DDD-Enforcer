from core.schemas import ContextRelationship, ContextMap
from core.context_mapper.derive import derive_allowed_dependencies

NAMES = {"Ordering", "Inventory", "Billing", "Shipping"}

def _cm(*rels):
    return ContextMap(model_id="m", relationships=list(rels))

def _r(a, b, t, up=None):
    return ContextRelationship(context_a=a, context_b=b, relationship_type=t,
                               upstream=up, rationale="x")

def test_directional_downstream_depends_on_upstream():
    deps, warns = derive_allowed_dependencies(
        _cm(_r("Ordering", "Inventory", "CUSTOMER_SUPPLIER", up="Inventory")), NAMES)
    assert deps["Ordering"] == ["Inventory"]
    assert deps.get("Inventory", []) == []
    assert warns == []

def test_acl_conformist_ohs_pl_are_directional():
    for t in ("CONFORMIST", "ANTI_CORRUPTION_LAYER", "OPEN_HOST_SERVICE", "PUBLISHED_LANGUAGE"):
        deps, _ = derive_allowed_dependencies(
            _cm(_r("Ordering", "Inventory", t, up="Inventory")), NAMES)
        assert deps["Ordering"] == ["Inventory"], t

def test_mutual_both_directions():
    for t in ("PARTNERSHIP", "SHARED_KERNEL"):
        deps, warns = derive_allowed_dependencies(
            _cm(_r("Ordering", "Billing", t)), NAMES)
        assert deps["Ordering"] == ["Billing"] and deps["Billing"] == ["Ordering"], t
        assert warns == []

def test_separate_ways_and_bbom_no_edges():
    for t in ("SEPARATE_WAYS", "BIG_BALL_OF_MUD"):
        deps, warns = derive_allowed_dependencies(
            _cm(_r("Ordering", "Shipping", t)), NAMES)
        assert deps.get("Ordering", []) == [] and deps.get("Shipping", []) == []

def test_unknown_context_dropped_with_warning():
    deps, warns = derive_allowed_dependencies(
        _cm(_r("Ordering", "Ghost", "CUSTOMER_SUPPLIER", up="Ghost")), NAMES)
    assert deps.get("Ordering", []) == []
    assert any("Ghost" in w for w in warns)

def test_directional_cycle_warns_but_keeps_edges():
    deps, warns = derive_allowed_dependencies(_cm(
        _r("Ordering", "Inventory", "CUSTOMER_SUPPLIER", up="Inventory"),
        _r("Inventory", "Billing", "CUSTOMER_SUPPLIER", up="Billing"),
        _r("Billing", "Ordering", "CUSTOMER_SUPPLIER", up="Ordering"),
    ), NAMES)
    assert deps["Ordering"] == ["Inventory"]
    assert deps["Inventory"] == ["Billing"]
    assert deps["Billing"] == ["Ordering"]
    assert any("cycle" in w.lower() for w in warns)

def test_mutual_pair_excluded_from_cycle_detection():
    deps, warns = derive_allowed_dependencies(
        _cm(_r("Ordering", "Billing", "PARTNERSHIP")), NAMES)
    assert not any("cycle" in w.lower() for w in warns)
