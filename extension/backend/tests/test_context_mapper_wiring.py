import os
from core.architect import DomainArchitect


def test_context_mapper_fn_default_on(monkeypatch):
    monkeypatch.delenv("DDD_CONTEXT_MAP", raising=False)
    da = DomainArchitect()
    assert da._build_context_mapper_fn() is not None


def test_context_mapper_fn_opt_out(monkeypatch):
    for v in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DDD_CONTEXT_MAP", v)
        assert DomainArchitect()._build_context_mapper_fn() is None
