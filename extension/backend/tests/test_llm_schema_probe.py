"""Smoke tests for core.llm.schema_probe."""

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from core.llm.schema_probe import (
    BasicViolation,
    ComplexViolation,
    MediumViolation,
    PROMPTS,
    SCHEMAS,
    _parse_args,
    probe_cell,
    run_probe,
)
from core.llm.base import LLMResponse, TokenUsage
from core.llm.registry import MODELS


def test_three_schemas_registered():
    assert set(SCHEMAS.keys()) == {"basic", "medium", "complex"}
    assert SCHEMAS["basic"] is BasicViolation
    assert SCHEMAS["medium"] is MediumViolation
    assert SCHEMAS["complex"] is ComplexViolation


def test_each_schema_has_a_prompt():
    for name in SCHEMAS:
        assert name in PROMPTS
        assert isinstance(PROMPTS[name], str)
        assert PROMPTS[name].strip()


def test_basic_violation_validates_minimal_payload():
    inst = BasicViolation(name="Customer", description="A buyer")
    assert inst.name == "Customer"


def test_complex_violation_requires_entities():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ComplexViolation(
            context_name="X",
            description="missing entities field",
        )  # type: ignore[call-arg]


def _ok_response(model_id: str) -> LLMResponse:
    return LLMResponse(
        content='{"name": "Customer", "description": "A buyer"}',
        parsed=BasicViolation(name="Customer", description="A buyer"),
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model_id=model_id,
        provider="gemini",
        json_failed=False,
        latency_ms=42.0,
    )


def _failed_response(model_id: str) -> LLMResponse:
    return LLMResponse(
        content="garbage",
        parsed=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model_id=model_id,
        provider="gemini",
        json_failed=True,
        json_fail_reason="invalid_json: garbage",
        latency_ms=30.0,
    )


def test_probe_cell_aggregates_success_count():
    fake_client = MagicMock()
    fake_client.structured_output.side_effect = [
        _ok_response("gemini-3.1-pro-preview"),
        _ok_response("gemini-3.1-pro-preview"),
        _failed_response("gemini-3.1-pro-preview"),
    ]
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        result = probe_cell("gemini-3.1-pro-preview", "basic", trials=3)
    assert result.trials == 3
    assert result.success == 2
    assert result.json_failed == 1
    assert result.provider == "gemini"
    assert result.mean_latency_ms > 0
    assert result.total_tokens == 15 + 15 + 12
    assert any("invalid_json" in err for err in result.errors)


def test_run_probe_iterates_all_six_models_by_default():
    # Return a response whose model_id echoes the requested model (guard requires this)
    def echo_model(messages, schema, model, **kwargs):
        return _ok_response(model)

    fake_client = MagicMock()
    fake_client.structured_output.side_effect = echo_model
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        report = run_probe(trials=1)
    # 6 models × 3 schemas = 18 cells
    assert report["trials_per_cell"] == 1
    assert len(report["results"]) == 18
    model_ids_in_report = {r["model_id"] for r in report["results"]}
    assert model_ids_in_report == set(MODELS.keys())


def test_run_probe_honors_models_filter():
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("gemini-3.1-pro-preview")
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        report = run_probe(models=["gemini-3.1-pro-preview"], schemas=["basic"], trials=1)
    assert len(report["results"]) == 1
    assert report["results"][0]["model_id"] == "gemini-3.1-pro-preview"
    assert report["results"][0]["schema"] == "basic"


def test_parse_args_default_out_is_none_so_main_can_timestamp():
    """argparse default --out is None so main() can timestamp; models
    and schemas default to None so main() can substitute the full set."""
    args = _parse_args([])
    assert args.out is None
    assert args.trials == 1
    assert args.models is None
    assert args.schemas is None


def test_parse_args_overrides():
    args = _parse_args([
        "--out", "/tmp/probe.json",
        "--trials", "3",
        "--models", "gemini-3.1-pro-preview",
        "--schemas", "complex",
    ])
    assert args.out == "/tmp/probe.json"
    assert args.trials == 3
    assert args.models == ["gemini-3.1-pro-preview"]
    assert args.schemas == ["complex"]


def test_report_serializes_to_json():
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("gemini-3.1-pro-preview")
    with patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client):
        report = run_probe(models=["gemini-3.1-pro-preview"], schemas=["basic"], trials=1)
    # Should round-trip through json without error
    serialized = json.dumps(report)
    parsed = json.loads(serialized)
    assert parsed["results"][0]["success"] == 1


# =============================================================================
# Task 1: CellResult new fields
# =============================================================================

def test_cell_result_has_transport_error_and_raw_failures_fields():
    from core.llm.schema_probe import CellResult
    result = CellResult(
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        schema="basic",
        trials=5,
    )
    assert result.transport_error == 0
    assert result.raw_failures == []
    assert result.seeds_used == []


# =============================================================================
# Task 2: transport_error separated from json_failed
# =============================================================================

def test_transport_error_separated_from_json_failed():
    """Provider exceptions (auth, rate-limit, 5xx) increment transport_error,
    not json_failed. json_failed is reserved for resp.json_failed=True."""
    from core.llm.errors import RateLimitError

    fake_client = MagicMock()
    fake_client.structured_output.side_effect = [
        _ok_response("gemini-3.1-pro-preview"),
        RateLimitError(provider="gemini", status_code=429),
        _failed_response("gemini-3.1-pro-preview"),
    ]
    with patch(
        "core.llm.schema_probe.get_client_for_model",
        return_value=fake_client,
    ):
        result = probe_cell("gemini-3.1-pro-preview", "basic", trials=3)

    assert result.success == 1
    assert result.json_failed == 1, "schema-level fail only"
    assert result.transport_error == 1, "provider exception goes here"
    assert any("RateLimitError" in e for e in result.errors)


# =============================================================================
# Task 3: client construction wrapped in try/except
# =============================================================================

def test_client_construction_failure_yields_transport_error_cell():
    """If get_client_for_model raises, the cell records transport_error
    equal to trials and the probe continues."""
    def boom(_mid):
        raise RuntimeError("no API key")

    with patch(
        "core.llm.schema_probe.get_client_for_model",
        side_effect=boom,
    ):
        result = probe_cell("gemini-3.1-pro-preview", "basic", trials=5)

    assert result.trials == 5
    assert result.transport_error == 5
    assert result.success == 0
    assert result.json_failed == 0
    assert any("ClientConstructorError" in e for e in result.errors)


# =============================================================================
# Task 4: per-trial varying seed
# =============================================================================

def test_seed_varies_per_trial():
    """probe_cell should pass seed=42, 43, 44 ... per trial, and
    record the list in result.seeds_used."""
    seen_seeds = []

    def capture_seed(messages, schema, model, **kwargs):
        seen_seeds.append(kwargs.get("seed"))
        return _ok_response(model)

    fake_client = MagicMock()
    fake_client.structured_output.side_effect = capture_seed

    with patch(
        "core.llm.schema_probe.get_client_for_model",
        return_value=fake_client,
    ):
        result = probe_cell("gemini-3.1-pro-preview", "basic", trials=3)

    assert seen_seeds == [42, 43, 44]
    assert result.seeds_used == [42, 43, 44]


# =============================================================================
# Task 5: runtime fallback hard-fail guard
# =============================================================================

def test_runtime_fallback_raises_loudly():
    """If resp.model_id != requested model_id, probe_cell raises
    RuntimeError to prevent mislabeled artifacts under D1 lock."""
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("gemini-2.5-flash")
    with patch(
        "core.llm.schema_probe.get_client_for_model",
        return_value=fake_client,
    ):
        with pytest.raises(RuntimeError, match="Runtime fallback fired"):
            probe_cell("gemini-3.1-flash-lite", "basic", trials=1)


# =============================================================================
# Task 6: raw output captured on json_failed
# =============================================================================

def test_raw_output_captured_on_json_failed():
    """When a trial returns json_failed=True, the first 500 chars of
    resp.content are appended to result.raw_failures for auditability."""
    long_garbage = "x" * 800
    bad_resp = LLMResponse(
        content=long_garbage,
        parsed=None,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model_id="gemini-3.1-pro-preview",
        provider="gemini",
        json_failed=True,
        json_fail_reason="schema_mismatch",
        latency_ms=30.0,
    )

    fake_client = MagicMock()
    fake_client.structured_output.return_value = bad_resp
    with patch(
        "core.llm.schema_probe.get_client_for_model",
        return_value=fake_client,
    ):
        result = probe_cell("gemini-3.1-pro-preview", "basic", trials=1)

    assert result.json_failed == 1
    assert len(result.raw_failures) == 1
    assert len(result.raw_failures[0]) == 500
    assert result.raw_failures[0] == "x" * 500


# =============================================================================
# Task 7: load_dotenv() at main() entry
# =============================================================================

def test_main_loads_dotenv(tmp_path):
    """main() must call load_dotenv() before resolving any client,
    otherwise env-only deployments break."""
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("gemini-3.1-pro-preview")
    out_file = tmp_path / "probe.json"
    with patch("core.llm.schema_probe.load_dotenv") as mock_load, \
         patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client), \
         patch("core.llm.schema_probe._git_head_or_raise", return_value="deadbeef"), \
         patch("core.llm.schema_probe._git_dirty_status", return_value=(False, [])):
        from core.llm.schema_probe import main
        rc = main([
            "--out", str(out_file),
            "--models", "gemini-3.1-pro-preview",
            "--schemas", "basic",
            "--trials", "1",
        ])
    assert rc == 0
    mock_load.assert_called_once()


# =============================================================================
# Task 8: timestamped default --out
# =============================================================================

def test_main_writes_timestamped_default_out(tmp_path, monkeypatch):
    """When --out is omitted, main() writes runs/probe-YYYYMMDD-HHMMSS.json
    inside the current working directory."""
    monkeypatch.chdir(tmp_path)
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("gemini-3.1-pro-preview")
    with patch("core.llm.schema_probe.load_dotenv"), \
         patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client), \
         patch("core.llm.schema_probe._git_head_or_raise", return_value="deadbeef"), \
         patch("core.llm.schema_probe._git_dirty_status", return_value=(False, [])):
        from core.llm.schema_probe import main
        rc = main([
            "--models", "gemini-3.1-pro-preview",
            "--schemas", "basic",
            "--trials", "1",
        ])
    assert rc == 0
    matches = list((tmp_path / "runs").glob("probe-*.json"))
    matches = [m for m in matches if m.name.endswith(".json") and "manifest" not in m.name]
    assert len(matches) == 1, f"expected exactly one runs/probe-*.json, got {matches}"
    assert re.match(r"^probe-\d{8}-\d{6}\.json$", matches[0].name)


# =============================================================================
# Task 9: manifest sidecar + git helpers
# =============================================================================

def test_manifest_sidecar_written_with_all_keys(tmp_path, monkeypatch):
    """Manifest carries all 16 documented keys; aborted is None on a clean run."""
    monkeypatch.chdir(tmp_path)
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("gemini-3.1-pro-preview")
    with patch("core.llm.schema_probe.load_dotenv"), \
         patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client), \
         patch("core.llm.schema_probe._git_head_or_raise", return_value="abc123"), \
         patch("core.llm.schema_probe._git_dirty_status", return_value=(False, [])), \
         patch("core.llm.schema_probe._pkg_version", return_value="9.9.9"):
        from core.llm.schema_probe import main
        rc = main([
            "--out", str(tmp_path / "p.json"),
            "--models", "gemini-3.1-pro-preview",
            "--schemas", "basic",
            "--trials", "1",
        ])
    assert rc == 0
    manifest_path = tmp_path / "p.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())

    expected_keys = {
        "timestamp_start", "timestamp_end",
        "git_commit", "git_dirty", "git_dirty_files",
        "python_version", "platform", "package_versions",
        "models", "schemas", "trials_per_cell",
        "seed_strategy", "temperature_default",
        "prompts", "schema_fidelity_note",
        "aborted",
    }
    assert set(manifest.keys()) == expected_keys
    assert manifest["git_commit"] == "abc123"
    assert manifest["git_dirty"] is False
    assert manifest["git_dirty_files"] == []
    assert manifest["package_versions"] == {
        "google-genai": "9.9.9",
        "openai": "9.9.9",
        "pydantic": "9.9.9",
    }
    assert manifest["seed_strategy"] == "per_trial_42_plus_i"
    assert manifest["temperature_default"] == 0.05
    assert manifest["schema_fidelity_note"].startswith("ComplexViolation approximates")
    assert manifest["aborted"] is None


def test_git_head_failure_aborts_run(tmp_path, monkeypatch):
    """If git rev-parse fails, the probe aborts loudly; no manifest
    is written with an 'unknown' commit."""
    monkeypatch.chdir(tmp_path)
    import subprocess
    fake_client = MagicMock()
    fake_client.structured_output.return_value = _ok_response("gemini-3.1-pro-preview")
    with patch("core.llm.schema_probe.load_dotenv"), \
         patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client), \
         patch(
             "core.llm.schema_probe.subprocess.run",
             side_effect=subprocess.CalledProcessError(128, ["git"]),
         ):
        from core.llm.schema_probe import main
        with pytest.raises(subprocess.CalledProcessError):
            main([
                "--out", str(tmp_path / "p.json"),
                "--models", "gemini-3.1-pro-preview",
                "--schemas", "basic",
                "--trials", "1",
            ])


# =============================================================================
# Task 10: transport-error warning loop
# =============================================================================

def test_transport_warning_printed_for_high_failure_cells(tmp_path, monkeypatch, capsys):
    """If a cell has transport_error >= trials/2, main() prints a
    WARNING line to stdout. Threshold for N=3 is >= 1.5, so any cell
    with >= 2 transport errors triggers (but 3/3 definitely does)."""
    monkeypatch.chdir(tmp_path)
    from core.llm.errors import RateLimitError
    fake_client = MagicMock()
    fake_client.structured_output.side_effect = [
        RateLimitError(provider="gemini"),
        RateLimitError(provider="gemini"),
        RateLimitError(provider="gemini"),
    ]
    with patch("core.llm.schema_probe.load_dotenv"), \
         patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client), \
         patch("core.llm.schema_probe._git_head_or_raise", return_value="abc"), \
         patch("core.llm.schema_probe._git_dirty_status", return_value=(False, [])), \
         patch("core.llm.schema_probe._pkg_version", return_value="1.0"):
        from core.llm.schema_probe import main
        rc = main([
            "--out", str(tmp_path / "p.json"),
            "--models", "gemini-3.1-pro-preview",
            "--schemas", "basic",
            "--trials", "3",
        ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "transport errors" in captured.out


# =============================================================================
# Pre-flight fallback check (IMPORTANT 4)
# =============================================================================

def test_preflight_aborts_when_requested_model_has_runtime_fallback(
    tmp_path, monkeypatch, capsys,
):
    """If --models includes a model that GeminiClient runtime-falls-back
    from, main() must abort BEFORE running the probe and exit non-zero,
    leaving no output files behind. The real fallback dict is currently
    empty (paper-integrity for D1), so we inject a synthetic entry to
    exercise the pre-flight branch."""
    monkeypatch.chdir(tmp_path)
    from core.llm.schema_probe import main
    fake_fallbacks = {"gemini-3.1-pro-preview": "gemini-2.5-pro"}
    with patch("core.llm.schema_probe.load_dotenv"), \
         patch.dict(
             "core.llm.schema_probe._GEMINI_FALLBACKS",
             fake_fallbacks,
             clear=True,
         ):
        rc = main([
            "--out", str(tmp_path / "p.json"),
            "--models", "gemini-3.1-pro-preview",
            "--schemas", "basic",
            "--trials", "1",
        ])
    assert rc == 1
    captured = capsys.readouterr()
    assert "refusing to probe" in captured.err
    assert "gemini-3.1-pro-preview" in captured.err
    assert "gemini-2.5-pro" in captured.err
    # no output written
    assert not (tmp_path / "p.json").exists()


# =============================================================================
# Partial output on abort (IMPORTANT 5)
# =============================================================================

def test_runtime_fallback_in_loop_writes_partial_report_then_reraises(
    tmp_path, monkeypatch,
):
    """If an unexpected resolved-model mismatch happens mid-loop (i.e.,
    something not in _GEMINI_FALLBACKS), the probe still writes the
    partial report + manifest with aborted=<message>, then re-raises."""
    monkeypatch.chdir(tmp_path)
    fake_client = MagicMock()
    # Returns response from a model NOT in _GEMINI_FALLBACKS, so pre-flight
    # passes but in-loop guard fires.
    fake_client.structured_output.return_value = _ok_response("unexpected-model")
    with patch("core.llm.schema_probe.load_dotenv"), \
         patch("core.llm.schema_probe.get_client_for_model", return_value=fake_client), \
         patch("core.llm.schema_probe._git_head_or_raise", return_value="abc"), \
         patch("core.llm.schema_probe._git_dirty_status", return_value=(False, [])), \
         patch("core.llm.schema_probe._pkg_version", return_value="1.0"):
        from core.llm.schema_probe import main
        with pytest.raises(RuntimeError, match="Runtime fallback fired"):
            main([
                "--out", str(tmp_path / "p.json"),
                "--models", "gpt-oss:120b-cloud",
                "--schemas", "basic",
                "--trials", "1",
            ])
    # Both artifacts written despite re-raise
    report = json.loads((tmp_path / "p.json").read_text())
    manifest = json.loads((tmp_path / "p.manifest.json").read_text())
    assert "aborted" in report
    assert "Runtime fallback fired" in report["aborted"]
    assert "aborted" in manifest
    assert manifest["aborted"] is not None
