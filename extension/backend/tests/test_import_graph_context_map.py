from core.AST.import_graph import apply_import_topology_to_model
import textwrap


def _write(tmp_path, rel, body):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_authoritative_map_blocks_autofill(tmp_path):
    # Ordering imports Inventory in code, but A declared SEPARATE_WAYS (empty deps).
    _write(tmp_path, "ordering/svc.py", "import inventory.models\n")
    _write(tmp_path, "inventory/models.py", "X = 1\n")
    model_data = {
        "context_map": {"model_id": "m", "error": None, "warnings": [],
                        "relationships": [{"context_a": "ordering", "context_b": "inventory",
                                           "relationship_type": "SEPARATE_WAYS", "upstream": None,
                                           "rationale": "r", "evidence_sentence_indices": []}]},
        "bounded_contexts": [
            {"context_name": "ordering", "allowed_dependencies": None},
            {"context_name": "inventory", "allowed_dependencies": None}],
    }
    diags = apply_import_topology_to_model(model_data,
        python_files=[str(tmp_path / "ordering/svc.py"), str(tmp_path / "inventory/models.py")],
        workspace_root=str(tmp_path))
    assert model_data["bounded_contexts"][0]["allowed_dependencies"] is None  # NOT auto-filled
    assert diags["auto_populated"] == []
    assert "ordering" in diags["cross_check_diff"]  # drift recorded for review


def test_failed_map_keeps_legacy_autofill(tmp_path):
    _write(tmp_path, "ordering/svc.py", "import inventory.models\n")
    _write(tmp_path, "inventory/models.py", "X = 1\n")
    model_data = {
        "context_map": {"model_id": "unknown", "error": "json_failed", "warnings": [], "relationships": []},
        "bounded_contexts": [
            {"context_name": "ordering", "allowed_dependencies": None},
            {"context_name": "inventory", "allowed_dependencies": None}],
    }
    diags = apply_import_topology_to_model(model_data,
        python_files=[str(tmp_path / "ordering/svc.py"), str(tmp_path / "inventory/models.py")],
        workspace_root=str(tmp_path))
    assert model_data["bounded_contexts"][0]["allowed_dependencies"] == ["inventory"]  # legacy fill
    assert "ordering" in diags["auto_populated"]


def test_no_context_map_keeps_legacy_autofill(tmp_path):
    # No context_map at all → legacy behavior (regression guard).
    _write(tmp_path, "ordering/svc.py", "import inventory.models\n")
    _write(tmp_path, "inventory/models.py", "X = 1\n")
    model_data = {
        "bounded_contexts": [
            {"context_name": "ordering", "allowed_dependencies": None},
            {"context_name": "inventory", "allowed_dependencies": None}],
    }
    diags = apply_import_topology_to_model(model_data,
        python_files=[str(tmp_path / "ordering/svc.py"), str(tmp_path / "inventory/models.py")],
        workspace_root=str(tmp_path))
    assert model_data["bounded_contexts"][0]["allowed_dependencies"] == ["inventory"]
    assert "ordering" in diags["auto_populated"]
