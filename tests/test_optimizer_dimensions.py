"""Tests for the new Optuna search dimensions wired into optimizer.py.

These tests cover the *settings* layer plus the building blocks the
optimiser uses to sample direct_transform / steering_variant. The full
TPE loop with a real model is not exercised here — it's integration-
tested on the GPU pod separately. See docs/benchmarks/.
"""

import pytest
import torch

from abliterix.core.optimizer_support import vector_scope_choices
from abliterix.settings import AbliterixConfig
from abliterix.types import DirectTransform


# ---------------------------------------------------------------------------
# SteeringConfig — the new search flags
# ---------------------------------------------------------------------------


def test_search_direct_transform_default_off():
    from abliterix.settings import SteeringConfig

    cfg = SteeringConfig()
    assert cfg.search_direct_transform is False
    assert cfg.search_direct_transform_choices == ["standard", "orba", "biprojected"]


def test_concept_gate_overrotation_defaults_to_legacy_clamp():
    from abliterix.settings import SteeringConfig

    cfg = SteeringConfig()
    assert cfg.concept_gate_angular_overrotation is False
    assert cfg.search_concept_gate_angular_overrotation is False
    assert cfg.concept_gate_angular_overrotation_phase == "all"
    assert cfg.search_concept_gate_angular_overrotation_phase is False
    assert cfg.concept_gate_intervention_geometry == "angular"
    assert cfg.search_concept_gate_intervention_geometry is False


def test_search_direct_transform_choices_overridable():
    from abliterix.settings import SteeringConfig

    cfg = SteeringConfig(
        search_direct_transform=True,
        search_direct_transform_choices=["orba", "biprojected"],
    )
    assert cfg.search_direct_transform is True
    assert cfg.search_direct_transform_choices == ["orba", "biprojected"]


def test_search_harmfulness_direction_default_off():
    from abliterix.settings import SteeringConfig

    cfg = SteeringConfig()
    assert cfg.search_harmfulness_direction is False


def test_search_harmfulness_direction_compatible_with_default_method():
    from abliterix.settings import SteeringConfig

    # search_harmfulness_direction must be settable independently of the
    # existing single-flag (ablate_harmfulness_direction) so users opt
    # into the *search* without committing to always using the pair.
    cfg = SteeringConfig(search_harmfulness_direction=True)
    assert cfg.search_harmfulness_direction is True
    assert cfg.ablate_harmfulness_direction is False


# ---------------------------------------------------------------------------
# DirectTransform enum round-trip — what optimizer.py uses
# ---------------------------------------------------------------------------


def test_direct_transform_enum_round_trip():
    for v in ("standard", "orba", "biprojected", "householder"):
        assert DirectTransform(v).value == v


# ---------------------------------------------------------------------------
# Optimizer signature accepts steering_vector_variants
# ---------------------------------------------------------------------------


def test_optimizer_signature_includes_variants_kwarg():
    """Smoke check that run_search exposes the new keyword arg.

    Reads the source file directly so the test passes on platforms where
    optional steering deps (bitsandbytes) don't install — the optimiser
    transitively pulls bnb in via core.steering.
    """
    from pathlib import Path

    src = Path("src/abliterix/optimizer.py").read_text()
    assert "steering_vector_variants" in src
    # Verify it's keyword-only (declared after `*,`).
    assert "*,\n    steering_vector_variants" in src
    # Verify default is None.
    assert "steering_vector_variants: dict[str," in src


# ---------------------------------------------------------------------------
# Mid-trial direct_transform override is restored
# ---------------------------------------------------------------------------


def test_direct_transform_swap_pattern_is_reversible():
    """Tests the save-swap-restore pattern the optimiser uses.

    Mirrors the logic in optimizer._objective so we know that mutating
    config.steering.direct_transform mid-trial is safe.
    """
    from abliterix.settings import SteeringConfig

    cfg = SteeringConfig(direct_transform=DirectTransform.STANDARD)

    # Simulate the optimiser's swap.
    saved = cfg.direct_transform
    cfg.direct_transform = DirectTransform.ORBA
    assert cfg.direct_transform == DirectTransform.ORBA

    # Simulate the finally-block restore.
    cfg.direct_transform = saved
    assert cfg.direct_transform == DirectTransform.STANDARD


def test_harmfulness_pair_tensor_shape_for_variant_swap():
    """Confirms the pair-variant produces a (2, layers+1, hidden) tensor
    that the multi-direction code path in steering.py already handles —
    the same shape as n_directions=2."""
    from abliterix.harmfulness import extract_harm_refusal_pair

    torch.manual_seed(0)
    benign = torch.randn(8, 5, 32)
    target = benign + torch.randn(1, 5, 32) * 0.5
    pair = extract_harm_refusal_pair(benign, target)
    assert pair.shape == (2, 5, 32)


def test_variants_dict_keys_are_categorical_friendly():
    """The keys we pass to Optuna's categorical sampler must be JSON-safe
    so they survive checkpoint serialisation."""
    keys = ["single", "harmfulness_pair"]
    for k in keys:
        assert isinstance(k, str)
        assert k.replace("_", "").isalnum()


def test_multi_direction_search_forces_per_layer_scope():
    """A global layer interpolation cannot silently ignore rank-k vectors."""
    config = AbliterixConfig(model={"model_id": "dummy/model"})
    single = torch.zeros(4, 8)
    pair = torch.zeros(2, 4, 8)

    assert vector_scope_choices(config, [single, pair]) == ["per layer"]


def test_single_direction_search_keeps_global_and_per_layer_choices():
    config = AbliterixConfig(model={"model_id": "dummy/model"})

    assert vector_scope_choices(config, [torch.zeros(4, 8)]) == [
        "global",
        "per layer",
    ]


@pytest.mark.parametrize("mode", ["adaptive_angular", "spherical", "vector_field"])
def test_runtime_hook_modes_reject_rank_k_recipes(mode):
    with pytest.raises(ValueError, match="runtime hook"):
        AbliterixConfig(
            model={"model_id": "test/model"},
            steering={"steering_mode": mode, "n_directions": 2},
        )


def test_rank_k_angular_is_available_for_hf():
    config = AbliterixConfig(
        model={"model_id": "test/model"},
        steering={"steering_mode": "angular", "n_directions": 2},
    )

    assert config.steering.n_directions == 2


def test_rank_k_concept_gate_requires_sign_agnostic_removal():
    with pytest.raises(ValueError, match="positive_alignment_only must be false"):
        AbliterixConfig(
            model={"model_id": "test/model"},
            steering={"steering_mode": "concept_gated_angular", "n_directions": 2},
        )

    config = AbliterixConfig(
        model={"model_id": "test/model"},
        steering={
            "steering_mode": "concept_gated_angular",
            "n_directions": 2,
            "concept_gate_positive_alignment_only": False,
        },
    )
    assert config.steering.n_directions == 2


def test_direction_router_requires_rank_k_global_concept_gate():
    with pytest.raises(ValueError, match="n_directions >= 2"):
        AbliterixConfig(
            model={"model_id": "test/model"},
            steering={
                "steering_mode": "concept_gated_angular",
                "concept_gate_scope": "global_prompt",
                "concept_gate_positive_alignment_only": False,
                "concept_gate_direction_router": True,
            },
        )

    config = AbliterixConfig(
        model={"model_id": "test/model"},
        steering={
            "steering_mode": "concept_gated_angular",
            "concept_gate_scope": "global_prompt",
            "concept_gate_positive_alignment_only": False,
            "concept_gate_direction_router": True,
            "n_directions": 2,
        },
    )
    assert config.steering.concept_gate_direction_router


def test_fixed_direction_search_requires_rank_k_and_excludes_router():
    with pytest.raises(ValueError, match="n_directions >= 2"):
        AbliterixConfig(
            model={"model_id": "test/model"},
            steering={"search_concept_gate_fixed_direction": True},
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        AbliterixConfig(
            model={"model_id": "test/model"},
            steering={
                "steering_mode": "concept_gated_angular",
                "concept_gate_scope": "global_prompt",
                "concept_gate_positive_alignment_only": False,
                "n_directions": 2,
                "concept_gate_direction_router": True,
                "search_concept_gate_fixed_direction": True,
            },
        )


def test_multi_direction_direct_rejects_transform_that_would_be_ignored():
    with pytest.raises(ValueError, match="direct_transform"):
        AbliterixConfig(
            model={"model_id": "test/model"},
            steering={
                "steering_mode": "direct",
                "n_directions": 2,
                "direct_transform": "orba",
            },
        )
