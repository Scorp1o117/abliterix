import pytest
from pydantic import ValidationError

from abliterix.settings import AbliterixConfig, SteeringConfig
from abliterix.types import PromptSource


def _source(**overrides):
    values = {"dataset": "paired", "split": "train[:8]", "column": "prompt"}
    values.update(overrides)
    return PromptSource(**values)


def test_response_pair_accepts_identical_prompt_sources():
    config = AbliterixConfig(
        steering=SteeringConfig(response_pair_enabled=True),
        benign_prompts=_source(),
        target_prompts=_source(),
    )

    assert config.steering.response_pair_enabled


def test_response_pair_rejects_prompt_or_system_mismatch():
    with pytest.raises(ValidationError, match="mismatched fields: split"):
        AbliterixConfig(
            steering=SteeringConfig(response_pair_enabled=True),
            benign_prompts=_source(),
            target_prompts=_source(split="train[8:16]"),
        )

    with pytest.raises(ValidationError, match="system_prompt"):
        AbliterixConfig(
            steering=SteeringConfig(response_pair_enabled=True),
            benign_prompts=_source(system_prompt="a"),
            target_prompts=_source(system_prompt="b"),
        )


def test_response_pair_rejects_empty_or_identical_continuations():
    with pytest.raises(ValidationError, match="must not be empty"):
        SteeringConfig(
            response_pair_enabled=True,
            response_pair_compliance_text=" ",
        )

    with pytest.raises(ValidationError, match="must differ"):
        SteeringConfig(
            response_pair_enabled=True,
            response_pair_compliance_text="same",
            response_pair_refusal_text="same",
        )


def test_response_trajectory_gate_keeps_direction_mode_separate():
    config = SteeringConfig(
        steering_mode="concept_gated_angular",
        concept_gate_training_source="response_trajectory",
        response_pair_enabled=False,
    )

    assert config.concept_gate_training_source == "response_trajectory"
    assert not config.response_pair_enabled


def test_response_trajectory_gate_requires_concept_gated_mode():
    with pytest.raises(ValidationError, match="concept_gated_angular"):
        SteeringConfig(concept_gate_training_source="response_trajectory")


def test_generated_prompt_trajectory_requires_valid_token_window():
    with pytest.raises(ValidationError, match="must not exceed"):
        SteeringConfig(
            steering_mode="concept_gated_angular",
            concept_gate_training_source="generated_prompt_trajectory",
            concept_gate_generated_max_new_tokens=4,
            concept_gate_generated_tokens_per_prompt=8,
        )
