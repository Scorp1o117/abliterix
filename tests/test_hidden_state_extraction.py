from types import SimpleNamespace

import torch
from torch import nn

from abliterix.core.engine import SteeringEngine


class _LogitsLimitedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits_to_keep: int | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        assert output_hidden_states
        self.logits_to_keep = logits_to_keep
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)
        return SimpleNamespace(hidden_states=(hidden, hidden + 1))


class _GenericForwardWrapper(nn.Module):
    """Mimic PEFT: generic public forward, explicit underlying base forward."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.wrapped = base

    def get_base_model(self) -> nn.Module:
        return self.wrapped

    def forward(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)


class _LegacyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        assert output_hidden_states
        self.called = True
        hidden = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)
        return SimpleNamespace(hidden_states=(hidden, hidden + 1))


def _engine(model: nn.Module) -> SteeringEngine:
    engine = object.__new__(SteeringEngine)
    engine.model = model
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(outlier_quantile=1.0),
    )
    engine._tokenize = lambda _messages: {"input_ids": torch.tensor([[1, 2]])}
    return engine


def test_hidden_state_extraction_limits_logits_when_base_forward_supports_it():
    base = _LogitsLimitedModel()
    engine = _engine(_GenericForwardWrapper(base))

    residuals = engine.extract_hidden_states([])

    assert base.logits_to_keep == 1
    assert residuals.shape == (1, 2, 3)


def test_hidden_state_extraction_omits_logits_limit_for_legacy_forward():
    model = _LegacyModel()
    engine = _engine(model)

    residuals = engine.extract_hidden_states([])

    assert model.called
    assert residuals.shape == (1, 2, 3)


def test_continuation_hidden_states_mean_pool_only_response_tokens():
    model = _LogitsLimitedModel()
    engine = _engine(model)
    engine._tokenize_with_continuations = lambda _messages, _texts: (
        {"input_ids": torch.tensor([[0, 1, 2, 4], [0, 0, 3, 9]])},
        torch.tensor([2, 1]),
    )

    residuals = engine.extract_continuation_hidden_states([], "answer")

    assert residuals.shape == (2, 2, 3)
    assert residuals[:, 0, 0].tolist() == [3.0, 9.0]
    assert residuals[:, 1, 0].tolist() == [4.0, 10.0]


def test_continuation_hidden_states_reject_bad_pooling():
    engine = _engine(_LegacyModel())

    try:
        engine.extract_continuation_hidden_states([], "answer", pooling="median")
    except ValueError as error:
        assert "pooling" in str(error)
    else:
        raise AssertionError("invalid pooling must fail")


def test_continuation_token_hidden_states_uniformly_sample_response_tokens():
    model = _LogitsLimitedModel()
    engine = _engine(model)
    engine._tokenize_with_continuations = lambda _messages, _texts: (
        {"input_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 8, 9]])},
        torch.tensor([4, 2]),
    )

    residuals = engine.extract_continuation_token_hidden_states(
        [], "answer", max_tokens_per_prompt=3
    )

    assert residuals.shape == (5, 2, 3)
    assert residuals[:, 0, 0].tolist() == [1.0, 3.0, 4.0, 8.0, 9.0]
    assert residuals[:, 1, 0].tolist() == [2.0, 4.0, 5.0, 9.0, 10.0]


def test_varying_continuation_token_hidden_states_select_first_tokens():
    model = _LogitsLimitedModel()
    engine = _engine(model)
    engine._tokenize_with_continuations = lambda _messages, _texts: (
        {"input_ids": torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 8, 9]])},
        torch.tensor([4, 2]),
    )

    residuals = engine.extract_continuation_token_hidden_states_varying(
        [], [], max_tokens_per_prompt=2, selection="first"
    )

    assert residuals.shape == (4, 2, 3)
    assert residuals[:, 0, 0].tolist() == [1.0, 2.0, 8.0, 9.0]
