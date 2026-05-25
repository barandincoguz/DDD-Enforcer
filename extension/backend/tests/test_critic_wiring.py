import pytest
from core.architect import DomainArchitect


@pytest.fixture(autouse=True)
def _gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_build_critic_fn_present_by_default(monkeypatch):
    # Loop is ON by default: an unset flag yields a callable.
    monkeypatch.delenv("DDD_CRITIC_LOOP", raising=False)
    arch = DomainArchitect()
    assert callable(arch._build_critic_fn())


def test_build_critic_fn_none_when_explicitly_disabled(monkeypatch):
    for value in ("0", "false", "off"):
        monkeypatch.setenv("DDD_CRITIC_LOOP", value)
        arch = DomainArchitect()
        assert arch._build_critic_fn() is None, f"{value!r} should disable the loop"


def test_build_critic_fn_present_when_flag_on(monkeypatch):
    monkeypatch.setenv("DDD_CRITIC_LOOP", "1")
    arch = DomainArchitect()
    fn = arch._build_critic_fn()
    assert callable(fn)
