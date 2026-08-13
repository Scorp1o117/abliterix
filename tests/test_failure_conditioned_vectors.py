import pytest
import torch

from abliterix.settings import AbliterixConfig
from abliterix.vectors import (
    build_failure_conditioned_variants,
    build_renormalized_primary_variants,
)


def test_failure_conditioned_variants_preserve_primary_and_are_unit_norm():
    primary = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    states = torch.zeros(4, 2, 3)
    states[0:2, :, 2] = 2.0
    variants = build_failure_conditioned_variants(
        primary, states, [0, 1], [2, 3], [0.5, 1.0]
    )

    assert list(variants) == ["primary", "failure_alpha_0p5", "failure_alpha_1"]
    torch.testing.assert_close(variants["primary"], primary)
    for candidate in variants.values():
        torch.testing.assert_close(
            torch.linalg.vector_norm(candidate, dim=-1), torch.ones(2)
        )
    assert torch.all(variants["failure_alpha_1"][:, 2] > 0)


def test_failure_conditioned_primary_control_is_bitwise_exact():
    primary = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float16)
    states = torch.randn(4, 1, 3)
    variants = build_failure_conditioned_variants(
        primary, states, [0, 1], [2, 3], [0.5]
    )
    assert torch.equal(variants["primary"], primary)


def test_renormalized_primary_repeats_are_identical_unit_vectors():
    primary = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32)
    variants = build_renormalized_primary_variants(primary, 2)
    assert list(variants) == ["renormalized_primary_0", "renormalized_primary_1"]
    assert torch.equal(variants["renormalized_primary_0"], variants["renormalized_primary_1"])
    torch.testing.assert_close(
        torch.linalg.vector_norm(variants["renormalized_primary_0"], dim=-1),
        torch.ones(1),
    )


def test_failure_conditioned_variants_reject_overlapping_labels():
    with pytest.raises(ValueError, match="disjoint"):
        build_failure_conditioned_variants(
            torch.randn(2, 3), torch.randn(4, 2, 3), [0, 1], [1, 2], [0.5]
        )


def test_failure_conditioned_config_requires_complete_disjoint_inputs():
    with pytest.raises(ValueError, match="require refusal indices"):
        AbliterixConfig(
            steering={
                "calibration_failure_refusal_indices": [1],
                "calibration_failure_alphas": [0.5],
            }
        )


def test_align_decoder_residuals_pads_embed_and_mtp_slots():
    from abliterix.vectors import align_decoder_residuals_to_primary

    primary = torch.randn(6, 4)
    decoder = torch.arange(32, dtype=torch.float32).reshape(2, 4, 4)
    aligned = align_decoder_residuals_to_primary(primary, decoder)
    assert aligned.shape == (2, 6, 4)
    assert torch.equal(aligned[:, 0], torch.zeros(2, 4))
    assert torch.equal(aligned[:, -1], torch.zeros(2, 4))
    assert torch.equal(aligned[:, 1:5], decoder)


def test_compose_steered_peel_keeps_primary_and_excludes_unhooked_layers():
    from abliterix.vectors import compose_exact_primary_with_steered_peel

    primary = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    decoder = torch.zeros(4, 1, 3)
    decoder[0, 0, 2] = 4.0
    decoder[1, 0, 2] = 4.0
    decoder[2, 0, 1] = 1.0
    decoder[3, 0, 1] = 1.0
    composed = compose_exact_primary_with_steered_peel(
        primary,
        decoder,
        source_indices=[0, 1, 2, 3],
        refusal_indices=[0, 1],
        compliance_indices=[2, 3],
    )
    assert composed.shape == (2, 3, 3)
    assert torch.equal(composed[0], primary)
    # Only the padded decoder slot (index 1) has a leftover; embed/MTP stay 0.
    assert composed[1, 0].abs().sum() == 0
    assert composed[1, 2].abs().sum() == 0
    torch.testing.assert_close(composed[1, 1], torch.tensor([0.0, 0.0, 1.0]))


def test_failure_calibration_split_is_independent_of_target_eval_split():
    config = AbliterixConfig(
        target_eval_prompts={
            "dataset": "dummy",
            "split": "train[900:]",
            "column": "prompt",
        },
        steering={
            "calibration_failure_refusal_indices": [1],
            "calibration_failure_compliance_indices": [2],
            "calibration_failure_alphas": [-0.125],
            "calibration_failure_prompt_split": "train[800:900]",
            "calibration_failure_variant_repeats": 2,
        },
    )
    assert config.target_eval_prompts.split == "train[900:]"
    assert config.steering.calibration_failure_prompt_split == "train[800:900]"
    assert config.steering.calibration_failure_variant_repeats == 2
    with pytest.raises(ValueError, match="must be disjoint"):
        AbliterixConfig(
            steering={
                "calibration_failure_refusal_indices": [1],
                "calibration_failure_compliance_indices": [1, 2],
                "calibration_failure_alphas": [0.5],
            }
        )
