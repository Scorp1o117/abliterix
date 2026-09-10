"""Qwen4Exp / Flash-Next residual width, PLE placement, and export gates."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from abliterix.core.engine import SteeringEngine
from abliterix.core.qwen4exp_runtime import (
    assemble_writeback_hidden_sequence,
    device_map_keeping_ngram_on_host,
    estimate_flash_next_resident_gib,
    full_precision_merge_refusal,
    is_ngram_parameter_name,
    iter_bnb_linear_targets,
    move_parameters_skipping_ngram,
    ngram_module_paths_from_index,
    ngram_parameter_names,
    override_device_map_ngram_cpu,
    pin_ngram_parameters_to_host,
    preflight_qwen4exp_load,
    stack_residual_rows,
    transformers_has_qwen4_exp,
    uses_writeback_residual_capture,
    writeback_hidden_size,
)
from abliterix.types import QuantMode, SteeringMode


class _Attn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _Mlp(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.down_proj = nn.Linear(hidden, hidden, bias=False)


class _GatedLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = _Attn(hidden)
        self.mlp = _Mlp(hidden)


class _GatedResidualStandin(nn.Module):
    """Decoder layer outs 10240; projection write-backs 2560."""

    def __init__(self, hidden: int = 2560, n_layers: int = 2, hc_count: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen4_exp",
            hidden_size=hidden,
            hc_count=hc_count,
            architectures=["Qwen4ExpForConditionalGeneration"],
            text_config=SimpleNamespace(
                model_type="qwen4_exp_text",
                hidden_size=hidden,
                hc_count=hc_count,
                num_hidden_layers=n_layers,
            ),
        )
        self._layers = nn.ModuleList([_GatedLayer(hidden) for _ in range(n_layers)])
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(layers=self._layers)
        )
        self.hidden = hidden
        self.hc_count = hc_count

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        **_kwargs,
    ) -> SimpleNamespace:
        assert output_hidden_states
        batch, seq = input_ids.shape
        embed = (
            input_ids.to(torch.float32)
            .unsqueeze(-1)
            .expand(-1, -1, self.hidden)
            .contiguous()
        )
        for layer in self._layers:
            _ = layer.self_attn.o_proj(embed)
            _ = layer.mlp.down_proj(embed)
        wide = embed.repeat(1, 1, self.hc_count)
        mixer = embed  # HC mixer collapses 10240 → 2560
        # Real Qwen4Exp: decoder-layer outs are 10240; last state is mixer 2560.
        # hidden_states[0] is NOT a 2560 embedding.
        layer_outs = tuple(wide.clone() for _ in self._layers)
        return SimpleNamespace(hidden_states=(*layer_outs, mixer))


class _MixedWidthNoHooks(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=2560)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
        **_kwargs,
    ) -> SimpleNamespace:
        assert output_hidden_states
        embed = input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 2560)
        wide = embed.repeat(1, 1, 4)
        return SimpleNamespace(hidden_states=(embed, wide))


class _PLEEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ngram_embedding = nn.Embedding(6, 2)


class _PLE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ple_embedding = _PLEEmb()


class _HostSkipModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)
        self.ple = _PLE()


def _engine(model: nn.Module) -> SteeringEngine:
    engine = object.__new__(SteeringEngine)
    engine.model = model
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(outlier_quantile=1.0, steering_mode=SteeringMode.LORA),
        model=SimpleNamespace(model_id="test/qwen4exp", quant_method=None),
    )
    engine._tokenize = lambda _messages: {"input_ids": torch.tensor([[1, 2, 3]])}
    engine._logits_to_keep_support = (id(model), False)
    return engine


def test_writeback_capture_returns_2560_not_10240():
    model = _GatedResidualStandin()
    engine = _engine(model)

    assert int(model(torch.tensor([[1, 2, 3]]), output_hidden_states=True).hidden_states[0].shape[-1]) == 10240
    assert int(model(torch.tensor([[1, 2, 3]]), output_hidden_states=True).hidden_states[-1].shape[-1]) == 2560

    residuals = engine.extract_hidden_states([])

    assert uses_writeback_residual_capture(model)
    assert writeback_hidden_size(model.config) == 2560
    # mixer (index 0) + 2 layer write-backs
    assert residuals.shape == (1, 3, 2560)
    assert residuals.shape[-1] != 10240


def test_assemble_skips_10240_hidden0_keeps_mixer_and_writebacks():
    writeback = torch.ones(1, 2, 2560)
    wide = torch.zeros(1, 2, 10240)
    mixer = torch.full((1, 2, 2560), 3.0)
    outputs = SimpleNamespace(hidden_states=(wide, wide, mixer))
    seq = assemble_writeback_hidden_sequence(
        [writeback, writeback], outputs, writeback_dim=2560
    )
    assert len(seq) == 3
    assert all(int(t.shape[-1]) == 2560 for t in seq)
    assert torch.equal(seq[0], mixer)


def test_mixed_10240_2560_stack_fails_closed_without_writebacks():
    model = _MixedWidthNoHooks()
    engine = _engine(model)

    with pytest.raises(RuntimeError, match="mixed last-dims"):
        engine.extract_hidden_states([])


def test_stack_residual_rows_rejects_10240_when_writeback_is_2560():
    wide = torch.zeros(1, 2, 10240)
    with pytest.raises(RuntimeError, match="write-back width 2560"):
        stack_residual_rows([wide], -1, writeback_dim=2560)


def test_device_map_override_pins_ngram_modules_from_index(tmp_path, monkeypatch):
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": "a.safetensors",
            "model.language_model.layers.0.mlp.experts.down_proj": "b.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen4_exp"}', encoding="utf-8"
    )
    names = ngram_module_paths_from_index(str(tmp_path))
    assert names == [
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
    ]

    inferred = {
        names[0]: 0,
        "model.language_model.layers.0.mlp": 0,
    }
    monkeypatch.setattr(
        "abliterix.core.qwen4exp_runtime._infer_auto_device_map",
        lambda *_args, **_kwargs: inferred,
    )
    mapped = device_map_keeping_ngram_on_host(
        str(tmp_path), "auto", {0: "64GB", "cpu": "24GB"}
    )
    assert mapped[names[0]] == "cpu"
    assert mapped["model.language_model.layers.0.mlp"] == 0

    raw = {names[0]: 0, "other": 0}
    assert override_device_map_ngram_cpu(raw, names)[names[0]] == "cpu"


def test_bnb_recipe_load_path_pins_ngram(monkeypatch):
    """Recipe is bnb_4bit + max_memory: the non-fast from_pretrained path."""
    from abliterix.core import engine as engine_module

    created = _HostSkipModel()
    captured: dict = {}

    class _Loader:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            captured.update(kwargs)
            return created

    monkeypatch.setattr(engine_module, "resolve_model_class", lambda *_a, **_k: _Loader)
    monkeypatch.setattr(engine_module, "model_type_from_id", lambda *_a, **_k: "qwen4_exp")
    monkeypatch.setattr(
        engine_module,
        "device_map_keeping_ngram_on_host",
        lambda model_id, device_map, max_memory: {"ple.ple_embedding.ngram_embedding": "cpu"},
    )
    pinned = {"n": 0}

    def _pin(model):
        pinned["n"] += 1
        return 1

    monkeypatch.setattr(engine_module, "pin_ngram_parameters_to_host", _pin)

    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        model=SimpleNamespace(
            device_map="auto",
            quant_method=QuantMode.BNB_4BIT,
            trust_remote_code=None,
            revision=None,
            text_only=False,
        )
    )
    engine.max_memory = {0: "64GB", "cpu": "24GB"}
    engine.trusted_models = {"m": True}
    engine._is_native_fp8 = False

    out = engine._load_model_fast("m", torch.float16, {"quantization_config": object()})
    assert out is created
    assert captured["device_map"] == {"ple.ple_embedding.ngram_embedding": "cpu"}
    assert pinned["n"] == 1


def test_pin_ngram_leaves_host_embeddings():
    model = _HostSkipModel()
    assert pin_ngram_parameters_to_host(model) == 0
    for name, param in model.named_parameters():
        if is_ngram_parameter_name(name):
            assert param.device.type == "cpu"


def test_ngram_parameters_skipped_from_device_move_and_bnb():
    model = _HostSkipModel()
    names = ngram_parameter_names(model)
    assert names
    assert all(is_ngram_parameter_name(n) for n in names)

    linear_names = {name for name, _ in iter_bnb_linear_targets(model)}
    assert "proj" in linear_names
    assert not any("ngram_embedding" in n for n in linear_names)

    ngram_before = {
        n: p.data_ptr() for n, p in model.named_parameters() if is_ngram_parameter_name(n)
    }
    moved, skipped = move_parameters_skipping_ngram(model, device="cpu")
    assert skipped >= 1
    ngram_after = {
        n: p.data_ptr() for n, p in model.named_parameters() if is_ngram_parameter_name(n)
    }
    assert ngram_before == ngram_after
    # Linear is eligible to move; n-gram is not a Linear target.
    assert moved >= 0


def test_export_adapter_succeeds_for_lora(monkeypatch, tmp_path):
    from abliterix.core import engine as engine_module

    class FakePeftModel:
        def __init__(self):
            self.saved_to = None
            self.config = SimpleNamespace(model_type="qwen4_exp")

        def named_parameters(self):
            # Upstream's empty-adapter guard requires a live lora_ parameter.
            yield "base_model.model.layers.0.o_proj.lora_A.default.weight", object()

        def save_pretrained(self, path):
            self.saved_to = path

    monkeypatch.setattr(engine_module, "PeftModel", FakePeftModel)
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.LORA),
        model=SimpleNamespace(model_id="test/qwen4exp"),
    )
    engine.model = FakePeftModel()
    engine._router_originals = []
    engine._expert_deltas = []
    engine.needs_reload = False

    engine.export_adapter(tmp_path)
    assert engine.model.saved_to == tmp_path


def test_export_merged_refuses_qwen4exp_full_precision_reload():
    engine = SteeringEngine.__new__(SteeringEngine)
    engine.config = SimpleNamespace(
        steering=SimpleNamespace(steering_mode=SteeringMode.LORA),
        model=SimpleNamespace(model_id="test/qwen4exp"),
    )
    engine.model = SimpleNamespace(
        config=SimpleNamespace(
            model_type="qwen4_exp",
            architectures=["Qwen4ExpForConditionalGeneration"],
        )
    )
    with pytest.raises(RuntimeError, match="export_adapter"):
        engine.export_merged()


def test_full_precision_merge_refusal_from_local_flash_next_dir():
    path = "/run/media/s117/KIOXIA 1TB/Models/Qwen3.8-Flash-Next"
    msg = full_precision_merge_refusal(path, None)
    assert msg is not None
    assert "336G" in msg or "export_adapter" in msg


def test_preflight_refuses_when_uma_cannot_hold_fused_experts(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"model_type": "qwen4_exp"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="fused 3-D experts"):
        preflight_qwen4exp_load(
            str(tmp_path),
            mem_available_bytes=30 * 1024**3,
            skip_ngram=True,
            quantize_fused_experts=False,
        )


def test_resident_estimate_skips_ngram():
    skipped = estimate_flash_next_resident_gib(skip_ngram=True, quantize_fused_experts=False)
    with_ngram = estimate_flash_next_resident_gib(
        skip_ngram=False, quantize_fused_experts=False
    )
    assert with_ngram - skipped > 80.0
    assert skipped > 200.0  # fused experts still BF16


def test_transformers_qwen4_exp_available_after_pin():
    assert transformers_has_qwen4_exp()


def test_flash_next_index_exposes_ngram_module_paths():
    path = "/run/media/s117/KIOXIA 1TB/Models/Qwen3.8-Flash-Next"
    names = ngram_module_paths_from_index(path)
    assert names
    assert all("ngram_embedding" in n for n in names)
    pinned = override_device_map_ngram_cpu({n: 0 for n in names}, names)
    assert all(v == "cpu" for v in pinned.values())
