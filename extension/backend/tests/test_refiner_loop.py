"""Phase C4: bounded retry loop. Mock LLM provider, mock verifier."""

import pytest
from unittest.mock import MagicMock
from core.verifier.types import VerifierIssue, IssueSeverity, VerifierResult
from core.orchestration.errors import RefinementExhaustedError
from core.refiner.loop import refine_until_clean


def _ok_result():
    return VerifierResult(ok=True, issues=[])


def _err_result(stage="specialist"):
    return VerifierResult(
        ok=False,
        issues=[VerifierIssue(
            stage=stage,
            location=f"{stage}:x.entities[0]",
            issue_type="missing_evidence",
            severity=IssueSeverity.ERROR,
            message="missing",
        )],
    )


def test_refiner_returns_clean_when_verifier_passes_first_try():
    stage_runner = MagicMock(return_value={"ok": True})
    verifier = MagicMock(return_value=_ok_result())
    out, cycles = refine_until_clean(
        stage_name="specialist",
        initial_output={"ok": True},
        stage_runner=stage_runner,
        verifier=verifier,
        max_cycles=2,
    )
    assert cycles == 0
    assert verifier.call_count == 1
    stage_runner.assert_not_called()


def test_refiner_runs_one_cycle_to_fix_issues():
    stage_runner = MagicMock(side_effect=[{"fixed": True}])
    verifier = MagicMock(side_effect=[_err_result(), _ok_result()])
    out, cycles = refine_until_clean(
        stage_name="specialist",
        initial_output={"buggy": True},
        stage_runner=stage_runner,
        verifier=verifier,
        max_cycles=2,
    )
    assert cycles == 1
    assert out == {"fixed": True}
    stage_runner.assert_called_once()


def test_refiner_raises_after_max_cycles():
    stage_runner = MagicMock(side_effect=[{"still_buggy": 1}, {"still_buggy": 2}])
    verifier = MagicMock(side_effect=[_err_result(), _err_result(), _err_result()])
    with pytest.raises(RefinementExhaustedError) as excinfo:
        refine_until_clean(
            stage_name="specialist",
            initial_output={"buggy": True},
            stage_runner=stage_runner,
            verifier=verifier,
            max_cycles=2,
        )
    assert len(excinfo.value.issues) == 1


def test_refiner_determinism_same_input_same_cycle_count():
    """Two identical setups must produce identical cycle counts."""
    def setup():
        sr = MagicMock(side_effect=[{"fixed": True}])
        ver = MagicMock(side_effect=[_err_result(), _ok_result()])
        return sr, ver
    sr1, v1 = setup()
    _, c1 = refine_until_clean(stage_name="x", initial_output={}, stage_runner=sr1, verifier=v1, max_cycles=2)
    sr2, v2 = setup()
    _, c2 = refine_until_clean(stage_name="x", initial_output={}, stage_runner=sr2, verifier=v2, max_cycles=2)
    assert c1 == c2 == 1
