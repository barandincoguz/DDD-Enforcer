from core.critic.errors import CriticError


def test_critic_error_carries_reason_and_cycle():
    err = CriticError(reason="json_failed: schema_mismatch", cycle=2)
    assert err.cycle == 2
    assert "json_failed" in str(err)
