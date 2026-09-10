#!/usr/bin/env python3
"""Retain-regularized CGA distill: short completions, extra benign CE, low LR.

v1 full-scale SFT collapsed (94/100 @ KL 3.25). This run only supervises the
first 32 continuation tokens, repeats benign rows, and scores the adapter
in-memory before writing a 52G merge.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_qwen38_vs_original import POCKET_BASELINE, _inject_original_baseline  # noqa: E402
from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.distill import continuation_labels, load_teacher_jsonl, sft_loss  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import ChatMessage, SteeringMode  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
TEACHER = ROOT / "artifacts" / "qwen38_cga_distill" / "teacher.jsonl"
ADAPTER = Path("/run/media/s117/OS/Models/Qwen3.8-27B-cga-distill-v2-lora")
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-cga-distill-v2")
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_cga_retain_eval.json")

MAX_CONT = 32
MAX_SEQ = 512
LR = 5e-5
ACCUM = 8
HARM_N = 200
BENIGN_REPEAT = 2
LORA_R = 8
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["bake_cga_retain", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    return cfg


def _pairs(engine) -> list[tuple[ChatMessage, str]]:
    rows = load_teacher_jsonl(TEACHER)
    harm = [r for r in rows if r["split"] == "harmful"][:HARM_N]
    beni = [r for r in rows if r["split"] == "benign"]
    ordered = harm + beni * BENIGN_REPEAT
    out = []
    for r in ordered:
        msg = ChatMessage(
            system=r.get("system") or "You are a helpful assistant.",
            user=r["user"],
        )
        raw = r["completion"]
        if not raw.startswith(" "):
            raw = " " + raw
        # Truncate to MAX_CONT tokens of the continuation.
        ids = engine.tokenizer(raw, add_special_tokens=False)["input_ids"][:MAX_CONT]
        text = engine.tokenizer.decode(ids, skip_special_tokens=False)
        if not text.startswith(" "):
            text = " " + text
        out.append((msg, text))
    return out


def _score(engine, cfg, tag: str) -> dict:
    engine.model.eval()
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    with torch.no_grad():
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    row = {
        "tag": tag,
        "keyword_refusals": int(refusals),
        "n": int(n),
        "full_distribution_kl_3token_vs_original": kl,
        "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
    }
    print(
        f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}",
        flush=True,
    )
    return row


def main() -> None:
    cfg = _cfg()
    print("loading original student...", flush=True)
    engine = SteeringEngine(cfg)
    pairs = _pairs(engine)
    print(f"sft pairs {len(pairs)} (harm<={HARM_N}, benign x{BENIGN_REPEAT})", flush=True)
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    engine.model = get_peft_model(engine.model, lora)
    engine.model.print_trainable_parameters()
    engine.model.train()
    engine.model.enable_input_require_grads()
    if hasattr(engine.model, "gradient_checkpointing_enable"):
        engine.model.gradient_checkpointing_enable()
    trainable = [p for p in engine.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR)
    step = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    for i, (msg, cont) in enumerate(pairs):
        inputs, cont_len = engine._tokenize_with_continuations([msg], [cont])
        ids = inputs["input_ids"]
        if ids.shape[1] > MAX_SEQ:
            ids = ids[:, -MAX_SEQ:]
            mask = inputs["attention_mask"][:, -MAX_SEQ:]
            cap = min(int(cont_len[0]), MAX_SEQ)
            cont_len = cont_len.new_tensor([cap])
        else:
            mask = inputs["attention_mask"]
        labels = continuation_labels(ids, mask, cont_len)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = engine.model(input_ids=ids, attention_mask=mask)
            loss = sft_loss(out.logits, labels) / ACCUM
        loss.backward()
        running += float(loss.detach()) * ACCUM
        if (i + 1) % ACCUM == 0 or (i + 1) == len(pairs):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 5 == 0 or (i + 1) == len(pairs):
                print(
                    f"item {i+1}/{len(pairs)} loss={running / max(step, 1):.4f}",
                    flush=True,
                )
                running = 0.0

    ADAPTER.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(ADAPTER))
    engine.tokenizer.save_pretrained(str(ADAPTER))
    print(f"wrote adapter {ADAPTER}", flush=True)

    row = _score(engine, cfg, "cga_retain_v2")
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "adapter": str(ADAPTER),
        "merged": str(MERGED) if row["in_budget"] else None,
        **row,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_cga_retain_eval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    if not row["in_budget"]:
        print("retain adapter not in budget; skip 52G merge", flush=True)
        return

    print("in budget — merging LoRA into BF16...", flush=True)
    merged = engine.model.merge_and_unload()
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
    ):
        src = Path(ORIGINAL) / extra
        if src.is_file() and not (MERGED / extra).exists():
            shutil.copy2(src, MERGED / extra)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
