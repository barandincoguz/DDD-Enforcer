def test_facade_exports():
    from core.context_mapper import run_context_mapper, derive_allowed_dependencies, ContextMapperError
    assert callable(run_context_mapper) and callable(derive_allowed_dependencies)

def test_stage_registered():
    from configs.models import STAGE_TO_GROUP, stage_config
    assert STAGE_TO_GROUP["ContextMapper"] == "domain_extraction"
    cfg = stage_config("ContextMapper")
    assert cfg.model_id
