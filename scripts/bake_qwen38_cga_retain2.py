#!/usr/bin/env python3
"""Bake CGA teacher into mergeable LoRA without Abliterix TPE.

Prior SFT collapsed (KL 3–4) because it trained long completions at high LR.
This run:
  * only the first 16 tokens of harmful CGA answers (leave "I cannot")
  * 3× benign CGA replies (≈ original on harmless)
  * LoRA r=4 on late layers only
  * lr=1e-5, eval vs original every 16 opt steps, stop on hit or KL>0.12
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
ADAPTER = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored-lora")
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_cga_retain2.json")

MAX_HARM = 16
MAX_BENIGN = 48
MAX_SEQ = 384
LR = 1e-5
ACCUM = 8
EVAL_EVERY = 16
MAX_STEPS = 48
HARM_N = 200
BENIGN_REPEAT = 3
LORA_R = 4
LATE_LAYERS = list(range(40, 64))


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["retain2", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    return cfg


def _truncate(engine, text: str, n_tok: int) -> str:
    if not text.startswith(" "):
        text = " " + text
    ids = engine.tokenizer(text, add_special_tokens=False)["input_ids"][:n_tok]
    out = engine.tokenizer.decode(ids, skip_special_tokens=False)
    return out if out.startswith(" ") else " " + out


def _pairs(engine) -> list[tuple[ChatMessage, str]]:
    rows = load_teacher_jsonl(TEACHER)
    harm = [r for r in rows if r["split"] == "harmful"][:HARM_N]
    beni = [r for r in rows if r["split"] == "benign"]
    ordered: list[dict] = []
    # Interleave so retain and harmful updates mix, not one epoch of only harm.
    bcycle = (beni * BENIGN_REPEAT) or beni
    bi = 0
    for h in harm:
        ordered.append(h)
        for _ in range(BENIGN_REPEAT):
            if bcycle:
                ordered.append(bcycle[bi % len(bcycle)])
                bi += 1
    out = []
    for r in ordered:
        msg = ChatMessage(
            system=r.get("system") or "You are a helpful assistant.",
            user=r["user"],
        )
        cap = MAX_HARM if r["split"] == "harmful" else MAX_BENIGN
        out.append((msg, _truncate(engine, r["completion"], cap)))
    return out


def _score(engine, cfg, tag: str) -> dict:
    was_training = engine.model.training
    engine.model.eval()
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    with torch.no_grad():
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
    if was_training:
        engine.model.train()
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


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_cga_retain2.json").write_text(text, encoding="utf-8")


def main() -> None:
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": "short CGA harmful prefix + benign retain, late LoRA r=4",
        "points": [],
    }
    print("loading original (angular, no Abliterix LoRA)...", flush=True)
    engine = SteeringEngine(cfg)
    pairs = _pairs(engine)
    print(f"sft pairs {len(pairs)}", flush=True)
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["o_proj", "out_proj", "down_proj"],
        layers_to_transform=LATE_LAYERS,
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
    hit = None
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = engine.model(input_ids=ids, attention_mask=mask)
            loss = sft_loss(out.logits, labels) / ACCUM
        loss.backward()
        running += float(loss.detach()) * ACCUM
        if (i + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            print(
                f"step {step} item {i+1}/{len(pairs)} loss={running / ACCUM:.4f}",
                flush=True,
            )
            running = 0.0
            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                row = _score(engine, cfg, f"retain2_step{step}")
                payload["points"].append(row)
                _dump(payload)
                if row["in_budget"]:
                    hit = row
                    break
                if row["full_distribution_kl_3token_vs_original"] > 0.12:
                    print("KL>0.12, stop before collapse", flush=True)
                    break
            if step >= MAX_STEPS:
                break

    ADAPTER.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(ADAPTER))
    engine.tokenizer.save_pretrained(str(ADAPTER))
    print(f"wrote adapter {ADAPTER}", flush=True)

    if hit is None and payload["points"]:
        last = payload["points"][-1]
        if last.get("in_budget"):
            hit = last
    if hit is None:
        payload["named_candidate"] = None
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("not in budget; skip 52G merge", flush=True)
        return

    print("in budget — merging...", flush=True)
    merged = engine.model.merge_and_unload()
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    for extra in ("chat_template.jinja", "generation_config.json"):
        src = Path(ORIGINAL) / extra
        if src.is_file():
            shutil.copy2(src, MERGED / extra)
    payload["named_candidate"] = hit["tag"]
    payload["merged"] = str(MERGED)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
