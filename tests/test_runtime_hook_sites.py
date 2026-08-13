from types import SimpleNamespace

import pytest
import torch
from torch import nn

from abliterix.core.steering import (
    _make_angular_hook,
    _make_concept_gated_angular_hook,
    _make_linear_projection_hook,
    apply_steering,
)
from abliterix.svf import ConceptScorer
from abliterix.types import SteeringMode, SteeringProfile


class _TupleModule(nn.Module):
    def forward(self, x):
        return x, None


class _MlpModule(nn.Module):
    def __init__(self, *, shared: bool):
        super().__init__()
        if shared:
            self.shared_experts = nn.Identity()

    def forward(self, x):
        return x, None


def _engine(config, layer_types, *, shared=True):
    layers = []
    for layer_type in layer_types:
        layers.append(
            SimpleNamespace(
                attention_layer_type=layer_type,
                attention=_TupleModule(),
                post_attention_layernorm=nn.Identity(),
                mlp=_MlpModule(shared=shared),
            )
        )
    return SimpleNamespace(
        config=config,
        transformer_layers=layers,
        has_expert_routing=lambda: False,
    )


def _apply(engine, config):
    vectors = torch.zeros(3, 4)
    vectors[:, 0] = 1
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={
            "attn.o_proj": SteeringProfile(
                max_weight=1.0,
                max_weight_position=0.0,
                min_weight=1.0,
                min_weight_distance=2.0,
            )
        },
        config=config,
    )


def test_angular_overrotation_is_opt_in_and_crosses_the_tangent():
    direction = torch.tensor([1.0, 0.0])
    hidden = torch.tensor([[1.0, 1.0]])

    clamped = _make_angular_hook(direction, 135.0)(None, (), hidden)
    overrotated = _make_angular_hook(
        direction, 135.0, allow_overrotation=True
    )(None, (), hidden)

    assert abs(clamped[0, 0].item()) < 1e-6
    assert overrotated[0, 0].item() < 0
    torch.testing.assert_close(overrotated.norm(), hidden.norm())


def test_linear_projection_removes_direction_without_restoring_norm():
    direction = torch.tensor([1.0, 0.0])
    hidden = torch.tensor([[1.0, 1.0]])

    projected = _make_linear_projection_hook(direction, 1.0)(None, (), hidden)
    reflected = _make_linear_projection_hook(direction, 1.5)(None, (), hidden)

    torch.testing.assert_close(projected, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(reflected, torch.tensor([[-0.5, 1.0]]))
    assert projected.norm() < hidden.norm()


class _AlwaysOnScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, hidden):
        return torch.ones(*hidden.shape[:-1], 1, device=hidden.device)


@pytest.mark.parametrize(
    ("phase", "prefill_crosses", "decode_crosses"),
    [("all", True, True), ("prefill", True, False), ("decode", False, True)],
)
def test_concept_gate_overrotation_phase_selects_forward_stage(
    phase, prefill_crosses, decode_crosses
):
    hook = _make_concept_gated_angular_hook(
        _AlwaysOnScorer(),
        torch.tensor([1.0, 0.0]),
        135.0,
        threshold=0.5,
        adaptive=False,
        pre_hook=False,
        allow_overrotation=True,
        overrotation_phase=phase,
    )
    prefill = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    decode = torch.tensor([[[1.0, 1.0]]])

    prefill_out = hook(None, (), prefill)
    decode_out = hook(None, (), decode)

    assert bool(torch.all(prefill_out[..., 0] < -1e-6)) is prefill_crosses
    assert bool(torch.all(decode_out[..., 0] < -1e-6)) is decode_crosses


@pytest.mark.parametrize(
    ("site", "expected_module"),
    [
        ("kda_output", "kda"),
        ("mla_output", "mla"),
        ("post_attention_residual", "post_attention"),
        ("mlp_output", "both"),
        ("shared_expert_output", "both"),
    ],
)
def test_runtime_hook_site_selects_architecture_module(
    abliterix_config, site, expected_module
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.ADAPTIVE_ANGULAR
    config.steering.runtime_hook_site = site
    engine = _engine(config, ["linear_attention", "attention"])

    _apply(engine, config)

    expected = 2 if expected_module in {"both", "post_attention"} else 1
    assert len(engine._angular_hooks) == expected
    assert sum(
        len(layer.attention._forward_hooks) for layer in engine.transformer_layers
    ) == (1 if expected_module in {"kda", "mla"} else 0)
    assert sum(
        len(layer.post_attention_layernorm._forward_pre_hooks)
        for layer in engine.transformer_layers
    ) == (2 if expected_module == "post_attention" else 0)


def test_runtime_hook_site_fails_when_architecture_has_no_shared_expert(
    abliterix_config,
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.ADAPTIVE_ANGULAR
    config.steering.runtime_hook_site = "shared_expert_output"
    engine = _engine(config, ["linear_attention", "attention"], shared=False)

    with pytest.raises(ValueError, match="unavailable"):
        _apply(engine, config)


def test_concept_gate_preserves_low_score_tokens_and_steers_high_score_tokens(
    abliterix_config,
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_threshold = 0.5

    layer = _TupleModule()
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[layer],
        has_expert_routing=lambda: False,
    )
    scorer = ConceptScorer(input_dim=4, hidden_dim=4)
    scorer.forward = lambda h: (h[..., :1] > 0).to(h.dtype)
    engine._concept_scorers = {0: scorer}

    vectors = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={
            "attn.o_proj": SteeringProfile(
                max_weight=1.0,
                max_weight_position=0.0,
                min_weight=1.0,
                min_weight_distance=1.0,
            )
        },
        config=config,
    )

    x = torch.tensor([[[-1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]])
    out, _ = layer(x)
    assert torch.equal(out[:, 0], x[:, 0])
    assert torch.allclose(out[:, 1, 0], torch.zeros_like(out[:, 1, 0]), atol=1e-6)
    # Angular removal preserves the full activation norm while rotating the
    # selected token onto the hyperplane orthogonal to the direction.
    assert torch.allclose(out[:, 1].norm(), x[:, 1].norm(), atol=1e-6)


def test_concept_gate_can_remove_negative_alignment_when_explicitly_enabled(
    abliterix_config,
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_threshold = 0.5
    config.steering.concept_gate_positive_alignment_only = False

    layer = _TupleModule()
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[layer],
        has_expert_routing=lambda: False,
    )
    scorer = ConceptScorer(input_dim=4, hidden_dim=4)
    scorer.forward = lambda h: torch.ones_like(h[..., :1])
    engine._concept_scorers = {0: scorer}

    vectors = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={
            "attn.o_proj": SteeringProfile(
                max_weight=1.0,
                max_weight_position=0.0,
                min_weight=1.0,
                min_weight_distance=1.0,
            )
        },
        config=config,
    )

    x = torch.tensor([[[-1.0, 2.0, 0.0, 0.0]]])
    out, _ = layer(x)
    assert torch.allclose(out[..., 0], torch.zeros(1, 1), atol=1e-6)
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-6)


def test_rank_two_concept_gate_installs_and_removes_joint_subspace(
    abliterix_config,
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_positive_alignment_only = False

    layer = _TupleModule()
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[layer],
        has_expert_routing=lambda: False,
    )
    scorer = ConceptScorer(input_dim=4, hidden_dim=4)
    scorer.forward = lambda h: torch.ones_like(h[..., :1])
    engine._concept_scorers = {0: scorer}
    vectors = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        ]
    )

    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={
            "attn.o_proj": SteeringProfile(1.0, 0.0, 1.0, 1.0),
        },
        config=config,
    )

    x = torch.tensor([[[3.0, 4.0, 12.0, 0.0]]])
    out, _ = layer(x)
    torch.testing.assert_close(
        out[..., :2], torch.zeros(1, 1, 2), atol=1e-5, rtol=0
    )
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1))


def test_prompt_concept_gate_latches_prefill_decision_through_decode(
    abliterix_config,
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_scope = "prompt"
    config.steering.concept_gate_threshold = 0.5

    layer = _TupleModule()
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[layer],
        has_expert_routing=lambda: False,
    )
    scorer = ConceptScorer(input_dim=4, hidden_dim=4)
    scorer.forward = lambda h: (h[..., 1:2] > 0).to(h.dtype)
    engine._concept_scorers = {0: scorer}
    vectors = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={
            "attn.o_proj": SteeringProfile(1.0, 0.0, 1.0, 1.0),
        },
        config=config,
    )

    # The final prefill token is harmful, so one decision gates the full
    # sequence and remains latched even when the decode token scores low.
    prefill = torch.tensor([[[-1.0, 2.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0]]])
    prefill_out, _ = layer(prefill)
    # Prompt gate is on for the sample, while the secondary adaptive-alignment
    # gate still preserves the negatively aligned first token.
    assert torch.equal(prefill_out[:, 0], prefill[:, 0])
    assert torch.allclose(prefill_out[:, 1, 0], torch.zeros(1), atol=1e-6)
    decode = torch.tensor([[[1.0, -2.0, 0.0, 0.0]]])
    decode_out, _ = layer(decode)
    assert torch.allclose(decode_out[..., 0], torch.zeros(1, 1), atol=1e-6)


def test_global_prompt_gate_broadcasts_first_layer_decision(abliterix_config):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_scope = "global_prompt"
    config.steering.concept_gate_global_decision_layer = 0
    config.steering.concept_gate_threshold = 0.5

    layers = [_TupleModule(), _TupleModule()]
    engine = SimpleNamespace(
        config=config,
        transformer_layers=layers,
        has_expert_routing=lambda: False,
    )
    scorers = {}
    for layer_idx in range(2):
        scorer = ConceptScorer(input_dim=4, hidden_dim=4)
        scorer.forward = lambda h: (h[..., 1:2] > 0).to(h.dtype)
        scorers[layer_idx] = scorer
    engine._concept_scorers = scorers
    vectors = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        # Layer 0 is outside the actual steering profile and must still get a
        # decision-only observer hook. Layer 1 receives the broadcast gate.
        profiles={"attn.o_proj": SteeringProfile(1.0, 1.0, 1.0, 0.1)},
        config=config,
    )

    prefill = torch.tensor([[[1.0, 2.0, 0.0, 0.0]]]).expand(1, 2, 4).clone()
    first_out, _ = layers[0](prefill)
    second_out, _ = layers[1](prefill)
    assert torch.equal(first_out, prefill)
    assert torch.allclose(second_out[..., 0], torch.zeros(1, 2), atol=1e-6)
    state = engine._global_concept_gate_state
    torch.testing.assert_close(state["gate_score"], torch.ones(1, 1))
    expected_signed = torch.tensor([[1.0 / (5.0**0.5)]])
    torch.testing.assert_close(state["signed_strength_feature"], expected_signed)
    torch.testing.assert_close(state["strength_feature"], expected_signed.abs())

    decode = torch.tensor([[[1.0, -2.0, 0.0, 0.0]]])
    decode_out, _ = layers[1](decode)
    assert torch.allclose(decode_out[..., 0], torch.zeros(1, 1), atol=1e-6)


def test_global_prompt_preset_gate_overrides_batch_scorer(abliterix_config):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_scope = "global_prompt"
    config.steering.concept_gate_global_decision_layer = 0

    layer = _TupleModule()
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[layer],
        has_expert_routing=lambda: False,
    )
    scorer = ConceptScorer(input_dim=4, hidden_dim=4)
    scorer.forward = lambda h: torch.zeros(*h.shape[:-1], 1, dtype=h.dtype)
    engine._concept_scorers = {0: scorer}
    vectors = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={"attn.o_proj": SteeringProfile(1.0, 0.0, 1.0, 1.0)},
        config=config,
    )

    # The scorer is always off, but a canonical prepass may explicitly preset
    # different decisions for rows in the real generation batch.
    engine._global_concept_gate_state["preset_gate"] = torch.tensor(
        [[[1.0]], [[0.0]]]
    )
    h = torch.tensor([[[1.0, 2.0, 0.0, 0.0]], [[1.0, 2.0, 0.0, 0.0]]])
    out, _ = layer(h)
    assert torch.allclose(out[0, ..., 0], torch.zeros(1), atol=1e-6)
    assert torch.equal(out[1], h[1])


def test_global_prompt_direction_router_applies_one_route_per_sample(
    abliterix_config,
):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_scope = "global_prompt"
    config.steering.concept_gate_global_decision_layer = 0
    config.steering.concept_gate_positive_alignment_only = False
    config.steering.concept_gate_direction_router = True

    layer = _TupleModule()
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[layer],
        has_expert_routing=lambda: False,
    )
    scorer = ConceptScorer(input_dim=4, hidden_dim=4)
    scorer.forward = lambda h: torch.ones(*h.shape[:-1], 1, dtype=h.dtype)
    engine._concept_scorers = {0: scorer}
    vectors = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        ]
    )
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={"attn.o_proj": SteeringProfile(1.0, 0.0, 1.0, 1.0)},
        config=config,
    )
    state = engine._global_concept_gate_state
    state["preset_gate"] = torch.ones(2, 1, 1)
    state["preset_route"] = torch.tensor([[[0]], [[1]]])

    h = torch.tensor([[[3.0, 4.0, 12.0, 0.0]], [[3.0, 4.0, 12.0, 0.0]]])
    out, _ = layer(h)

    torch.testing.assert_close(out[0, ..., 0], torch.zeros(1), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1, ..., 1], torch.zeros(1), atol=1e-5, rtol=0)
    assert out[0, 0, 1].abs() > 0
    assert out[1, 0, 0].abs() > 0


def test_global_prompt_state_initializes_decision_cache(abliterix_config):
    config = abliterix_config.model_copy(deep=True)
    config.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    config.steering.runtime_hook_site = "decoder_block"
    config.steering.concept_gate_scope = "global_prompt"
    config.steering.concept_gate_global_decision_layer = 0
    layer = _TupleModule()
    engine = SimpleNamespace(
        config=config,
        transformer_layers=[layer],
        has_expert_routing=lambda: False,
    )
    scorer = ConceptScorer(input_dim=4, hidden_dim=4)
    engine._concept_scorers = {0: scorer}
    vectors = torch.zeros(2, 4)
    apply_steering(
        engine,
        vectors,
        vector_index=None,
        profiles={"attn.o_proj": SteeringProfile(1.0, 0.0, 1.0, 1.0)},
        config=config,
    )
    assert engine._global_concept_gate_state["decision_cache"] == {}
    assert engine._global_concept_gate_state["strength_feature_cache"] == {}
    assert engine._global_concept_gate_state["signed_strength_feature_cache"] == {}
    assert engine._global_concept_gate_state["gate_score_cache"] == {}
