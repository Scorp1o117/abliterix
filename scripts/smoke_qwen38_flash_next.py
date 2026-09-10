#!/usr/bin/env python3
"""OOM-safe Flash-Next smoke: uma_guard, qwen4_exp import, preflight, no 336G load."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

MODEL = "/run/media/s117/KIOXIA 1TB/Models/Qwen3.8-Flash-Next"


def main() -> int:
    from abliterix.uma_guard import snapshot, start_uma_guard
    from abliterix.core.qwen4exp_runtime import (
        estimate_flash_next_resident_gib,
        model_type_from_id,
        preflight_qwen4exp_load,
        transformers_has_qwen4_exp,
    )

    start_uma_guard()
    print(f"[smoke] uma_guard snapshot | {snapshot()}")

    import transformers

    print(f"[smoke] transformers {transformers.__version__}")
    has_arch = transformers_has_qwen4_exp()
    print(f"[smoke] qwen4_exp importable={has_arch}")
    if has_arch:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(MODEL)
        print(
            f"[smoke] AutoConfig model_type={cfg.model_type} "
            f"hidden={cfg.text_config.hidden_size} hc={cfg.text_config.hc_count}"
        )

    print(f"[smoke] local model_type={model_type_from_id(MODEL)}")
    print(
        "[smoke] resident estimate GiB (ngram skipped, fused BF16)="
        f"{estimate_flash_next_resident_gib(MODEL, skip_ngram=True):.1f}"
    )

    try:
        preflight_qwen4exp_load(MODEL, skip_ngram=True, quantize_fused_experts=False)
        print("[smoke] preflight allowed load (override or huge host)")
        print("[smoke] SKIP full from_pretrained: would materialize fused experts")
        return 0
    except RuntimeError as exc:
        print(f"[smoke] preflight refused load (expected on 128G UMA): {exc}")
        print("[smoke] no suicide; exit 0 after gated refuse")
        print("[smoke] HIT target remains refusal ~10/100, kl.target=0.05")
        print("[smoke] mergeable export is export_adapter (LoRA); not export_merged")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
