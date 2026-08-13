from types import SimpleNamespace

import torch

from abliterix.eval.metrics import ComplianceResult
from abliterix.eval.stage_evaluator import StageEvaluator


class _Trial:
    def __init__(self):
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def test_prescreen_records_source_indices_for_each_outcome():
    optimization = SimpleNamespace(
        refusal_prescreen_seed=117,
        refusal_prescreen_size=3,
        refusal_prescreen_prune_min=4,
        refusal_prescreen_pass_max=3,
        prescreen_reverse_order_replay=False,
        validation_kl_enabled=False,
    )
    config = SimpleNamespace(optimization=optimization)
    result = ComplianceResult(
        labels=(True, False, True),
        evaluator="keyword",
        protocol_version="test",
    )
    detector = SimpleNamespace(
        evaluate_compliance_result=lambda *_args: result,
        _last_onset=(
            {
                "bucket": "zh_cannot",
                "prefix_refusal": True,
                "late_refusal": False,
                "full_refusal": True,
            },
            {
                "bucket": "non_refusal_prefix",
                "prefix_refusal": False,
                "late_refusal": False,
                "full_refusal": False,
            },
            {
                "bucket": "zh_cannot",
                "prefix_refusal": True,
                "late_refusal": False,
                "full_refusal": True,
            },
        ),
    )
    scorer = SimpleNamespace(
        target_msgs=[object(), object(), object()],
        benign_msgs=[],
        detector=detector,
    )
    messages = scorer.target_msgs
    cache = {
        f"prompt-{id(messages[0])}": torch.tensor([[1.0]]),
        f"prompt-{id(messages[1])}": torch.tensor([[0.0]]),
        f"prompt-{id(messages[2])}": torch.tensor([[1.0]]),
    }
    route_cache = {
        f"prompt-{id(messages[0])}": torch.tensor([[0]]),
        f"prompt-{id(messages[1])}": torch.tensor([[1]]),
        f"prompt-{id(messages[2])}": torch.tensor([[0]]),
    }
    strength_feature_cache = {
        f"prompt-{id(message)}": torch.tensor([[0.1 + 0.1 * index]])
        for index, message in enumerate(messages)
    }
    signed_strength_feature_cache = {
        f"prompt-{id(message)}": torch.tensor([[-0.1 + 0.1 * index]])
        for index, message in enumerate(messages)
    }
    gate_score_cache = {
        f"prompt-{id(message)}": torch.tensor([[0.7 + 0.1 * index]])
        for index, message in enumerate(messages)
    }
    engine = SimpleNamespace(
        _global_concept_gate_state={
            "decision_cache": cache,
            "route_cache": route_cache,
            "strength_feature_cache": strength_feature_cache,
            "signed_strength_feature_cache": signed_strength_feature_cache,
            "gate_score_cache": gate_score_cache,
        },
        _render_messages=lambda msgs: [f"prompt-{id(msg)}" for msg in msgs],
    )
    evaluator = StageEvaluator(config, engine, scorer)
    trial = _Trial()

    evaluator._run_prescreen(trial)

    ordered = evaluator._prescreen_indices
    assert trial.user_attrs["prescreen_refusal_indices"] == [ordered[0], ordered[2]]
    assert trial.user_attrs["prescreen_compliance_indices"] == [ordered[1]]
    assert trial.user_attrs["prescreen_prefix_refusal_indices"] == [
        ordered[0],
        ordered[2],
    ]
    assert trial.user_attrs["prescreen_late_refusal_indices"] == []
    assert trial.user_attrs["prescreen_prefix_classes"][str(ordered[0])] == "zh_cannot"
    assert trial.user_attrs["prescreen_gate_on_indices"] == [
        idx for idx in ordered if idx in {0, 2}
    ]
    assert trial.user_attrs["prescreen_gate_off_indices"] == [
        idx for idx in ordered if idx == 1
    ]
    assert trial.user_attrs["prescreen_direction_route_indices"] == {
        "0": [0, 2],
        "1": [1],
    }
    assert set(trial.user_attrs["prescreen_strength_features"]) == {
        str(index) for index in ordered
    }
    assert set(trial.user_attrs["prescreen_signed_strength_features"]) == {
        str(index) for index in ordered
    }
    assert set(trial.user_attrs["prescreen_gate_scores"]) == {
        str(index) for index in ordered
    }


def test_prescreen_dumps_steered_prefill_residuals(tmp_path):
    optimization = SimpleNamespace(
        refusal_prescreen_seed=117,
        refusal_prescreen_size=3,
        refusal_prescreen_prune_min=4,
        refusal_prescreen_pass_max=3,
        prescreen_reverse_order_replay=False,
        validation_kl_enabled=False,
        checkpoint_dir=str(tmp_path),
    )
    config = SimpleNamespace(
        optimization=optimization,
        steering=SimpleNamespace(dump_steered_prefill_residuals=True),
    )
    result = ComplianceResult(
        labels=(True, False, True),
        evaluator="keyword",
        protocol_version="test",
    )
    detector = SimpleNamespace(
        evaluate_compliance_result=lambda *_args: result,
    )
    scorer = SimpleNamespace(
        target_msgs=[object(), object(), object()],
        benign_msgs=[],
        detector=detector,
    )
    messages = scorer.target_msgs
    keys = [f"prompt-{id(message)}" for message in messages]
    residual_cache = {
        keys[0]: torch.zeros(4, 8),
        keys[1]: torch.ones(4, 8),
        keys[2]: torch.full((4, 8), 2.0),
    }
    engine = SimpleNamespace(
        _global_concept_gate_state={
            "decision_cache": {key: torch.tensor([[1.0]]) for key in keys},
            "prefill_residual_cache": residual_cache,
        },
        _render_messages=lambda msgs: [f"prompt-{id(msg)}" for msg in msgs],
    )
    evaluator = StageEvaluator(config, engine, scorer)
    trial = _Trial()

    evaluator._run_prescreen(trial)

    dump_path = tmp_path / "steered_prefill_residuals.pt"
    assert trial.user_attrs["steered_prefill_residual_dump"] == str(dump_path)
    assert trial.user_attrs["steered_prefill_residual_count"] == 3
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    assert payload["source_indices"] == evaluator._prescreen_indices
    assert payload["residuals"].shape == (3, 4, 8)
    assert payload["refusal_indices"] == [
        evaluator._prescreen_indices[0],
        evaluator._prescreen_indices[2],
    ]
    assert payload["compliance_indices"] == [evaluator._prescreen_indices[1]]
