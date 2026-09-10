# Measure Qwen3.8-27B generate throughput vs batch size on this machine.
# Uses the same Abliterix engine / encoder / 4-bit load as the search.
# Fixed min=max new tokens so early-EOS padding cannot fake a winner.

from __future__ import annotations

import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AX_CONFIG", "configs/qwen38_27b_rocm.toml")
sys.argv = ["abliterix", "--config", "configs/qwen38_27b_rocm.toml", "--seed", "117"]

import torch
from abliterix.core.engine import SteeringEngine
from abliterix.settings import AbliterixConfig

SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NEW_TOKENS = 32
WARMUP = 1
REPEATS = 1


def _msgs(n: int):
    from abliterix.types import ChatMessage

    return [
        ChatMessage(
            system="You are a helpful assistant.",
            user=f"Say the number {i} and count to three.",
        )
        for i in range(n)
    ]


def main() -> None:
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 1
    cfg.inference.min_gen_tokens = NEW_TOKENS
    cfg.inference.max_gen_tokens = NEW_TOKENS
    print(f"loading {cfg.model.model_id} quant={cfg.model.quant_method}")
    engine = SteeringEngine(cfg)
    print(
        f"loaded  alloc={torch.cuda.memory_allocated()/1e9:.2f}G  "
        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G"
    )

    results: list[tuple[int, float, float]] = []
    for bs in SIZES:
        batch = _msgs(bs)
        try:
            for _ in range(WARMUP):
                engine.generate_text(
                    batch, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPEATS):
                out = engine.generate_text(
                    batch, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS
                )
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / REPEATS
            toks = bs * NEW_TOKENS
            tps = toks / dt
            print(
                f"bs={bs:<3}  {dt:6.2f}s  {tps:6.1f} tok/s  "
                f"lat={dt/bs*1000:.0f} ms/seq  n={len(out)}"
            )
            results.append((bs, dt, tps))
        except Exception as e:
            print(f"bs={bs:<3}  FAIL  {type(e).__name__}: {e}")
            break

    if results:
        best = max(results, key=lambda r: r[2])
        print(f"WINNER bs={best[0]}  {best[2]:.1f} tok/s")


if __name__ == "__main__":
    main()
