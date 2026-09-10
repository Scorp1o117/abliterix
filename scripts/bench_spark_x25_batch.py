# Measure Spark-X2.5-4B generate + residual-extract throughput vs batch size.
# Fixed min=max new tokens so early-EOS cannot fake a winner.

from __future__ import annotations

import os
import sys
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TRANSFORMERS_SKIP_ALLOCATOR_WARMUP", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("ABLITERIX_MAX_SEQ", "4096")
os.environ.setdefault("AX_CONFIG", "configs/spark_x25_4b_rocm_smoke.toml")
sys.argv = [
    "abliterix",
    "--config",
    "configs/spark_x25_4b_rocm_smoke.toml",
    "--seed",
    "117",
]

import torch
from abliterix.core.engine import SteeringEngine
from abliterix.settings import AbliterixConfig
from abliterix.types import ChatMessage

SIZES = [8, 16, 32, 64, 96, 128, 192, 256]
NEW_TOKENS = 32


def _msgs(n: int) -> list[ChatMessage]:
    return [
        ChatMessage(
            system="You are a helpful assistant.",
            user=f"Say the number {i} and count to three.",
        )
        for i in range(n)
    ]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 8
    cfg.inference.min_gen_tokens = NEW_TOKENS
    cfg.inference.max_gen_tokens = NEW_TOKENS
    print(f"loading {cfg.model.model_id}")
    engine = SteeringEngine(cfg)
    print(
        f"loaded  alloc={torch.cuda.memory_allocated()/1e9:.2f}G  "
        f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G"
    )

    gen_rows: list[tuple[int, float, float]] = []
    hs_rows: list[tuple[int, float, float]] = []

    print("\n=== generate (decode, 32 new tokens) ===")
    for bs in SIZES:
        batch = _msgs(bs)
        try:
            engine.generate_text(
                batch, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS
            )
            _sync()
            t0 = time.perf_counter()
            out = engine.generate_text(
                batch, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS
            )
            _sync()
            dt = time.perf_counter() - t0
            tps = (bs * NEW_TOKENS) / dt
            rss = torch.cuda.memory_allocated() / 1e9
            print(
                f"bs={bs:<3}  {dt:6.2f}s  {tps:7.1f} tok/s  "
                f"lat={dt/bs*1000:6.0f} ms/seq  alloc={rss:.1f}G  n={len(out)}"
            )
            gen_rows.append((bs, dt, tps))
        except Exception as e:
            print(f"bs={bs:<3}  GEN FAIL  {type(e).__name__}: {e}")
            break

    print("\n=== extract_hidden_states (prefill residuals) ===")
    for bs in SIZES:
        batch = _msgs(bs)
        try:
            engine.extract_hidden_states(batch)
            _sync()
            t0 = time.perf_counter()
            hs = engine.extract_hidden_states(batch)
            _sync()
            dt = time.perf_counter() - t0
            pps = bs / dt
            rss = torch.cuda.memory_allocated() / 1e9
            print(
                f"bs={bs:<3}  {dt:6.2f}s  {pps:7.1f} prompt/s  "
                f"shape={tuple(hs.shape)}  alloc={rss:.1f}G"
            )
            hs_rows.append((bs, dt, pps))
        except Exception as e:
            print(f"bs={bs:<3}  HS FAIL  {type(e).__name__}: {e}")
            break

    if gen_rows:
        best_g = max(gen_rows, key=lambda r: r[2])
        print(f"\nGEN WINNER  bs={best_g[0]}  {best_g[2]:.1f} tok/s")
    if hs_rows:
        best_h = max(hs_rows, key=lambda r: r[2])
        print(f"HS  WINNER  bs={best_h[0]}  {best_h[2]:.1f} prompt/s")


if __name__ == "__main__":
    main()
