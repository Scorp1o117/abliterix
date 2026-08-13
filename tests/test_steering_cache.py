from abliterix.cli import (
    _expert_profiling_enabled,
    _make_concept_scorer_cache_key,
    _make_steering_cache_key,
)
from abliterix.settings import AbliterixConfig


def test_cache_key_tracks_direction_and_prompt_provenance():
    base = AbliterixConfig()
    original = _make_steering_cache_key(base)

    changed_ot = base.model_copy(deep=True)
    changed_ot.steering.ot_components += 1
    assert _make_steering_cache_key(changed_ot) != original

    changed_system = base.model_copy(deep=True)
    changed_system.target_prompts.system_prompt = "refuse this request"
    assert _make_steering_cache_key(changed_system) != original

    changed_multi = base.model_copy(deep=True)
    changed_multi.steering.n_directions = 2
    assert _make_steering_cache_key(changed_multi) != original


def test_cache_key_ignores_write_profile_only_changes():
    base = AbliterixConfig()
    changed = base.model_copy(deep=True)
    changed.steering.strength_range = [9.0, 10.0]
    changed.steering.weight_normalization = "full"

    assert _make_steering_cache_key(changed) == _make_steering_cache_key(base)


def test_concept_scorer_cache_key_is_direction_independent():
    base = AbliterixConfig()
    changed = base.model_copy(deep=True)
    changed.steering.n_directions = 2
    changed.steering.vector_method = "optimal_transport"

    assert _make_concept_scorer_cache_key(changed) == (
        _make_concept_scorer_cache_key(base)
    )


def test_concept_scorer_cache_key_tracks_training_recipe():
    base = AbliterixConfig()
    changed = base.model_copy(deep=True)
    changed.steering.svf_scorer_hidden += 16

    assert _make_concept_scorer_cache_key(changed) != (
        _make_concept_scorer_cache_key(base)
    )


def test_expert_profiling_skips_disabled_routing_search():
    config = AbliterixConfig()
    config.experts.max_suppress = 0
    assert not _expert_profiling_enabled(config)

    config.experts.max_suppress = 4
    config.experts.router_bias_range = [0.0, 0.0]
    config.experts.ablation_weight_range = [0.0, 0.0]
    assert not _expert_profiling_enabled(config)

    config.experts.ablation_weight_range = [0.0, 1.0]
    assert _expert_profiling_enabled(config)
