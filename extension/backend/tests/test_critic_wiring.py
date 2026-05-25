import pytest
from core.architect import DomainArchitect


@pytest.fixture(autouse=True)
def _gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_build_critic_fn_none_when_flag_off(monkeypatch):
    monkeypatch.delenv("DDD_CRITIC_LOOP", raising=False)
    arch = DomainArchitect()
    assert arch._build_critic_fn() is None


def test_build_critic_fn_present_when_flag_on(monkeypatch):
    monkeypatch.setenv("DDD_CRITIC_LOOP", "1")
    arch = DomainArchitect()
    fn = arch._build_critic_fn()
    assert callable(fn)
