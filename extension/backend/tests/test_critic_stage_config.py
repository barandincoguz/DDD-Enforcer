from configs.models import stage_config


def test_critic_stage_uses_generation_group():
    arch = stage_config("Architect")
    critic = stage_config("Critic")
    assert critic.model_id == arch.model_id
    assert critic.temperature == arch.temperature
    assert critic.seed == arch.seed
