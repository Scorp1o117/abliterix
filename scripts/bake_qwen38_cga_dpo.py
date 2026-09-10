#!/usr/bin/env python3
"""DPO: CGA short answers vs original refusals, LoRA on the original.

π_ref is the frozen base (disable_adapter). No second 27B copy.
Eval vs original every 10 steps; stop on 8–12 @ KL≤0.1 or KL>0.12.
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
from torch.nn import functional as F  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.distill import continuation_labels, load_teacher_jsonl  # noqa: E402
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import ChatMessage, SteeringMode  # noqa: E402

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
TEACHER = ROOT / "artifacts" / "qwen38_cga_distill" / "teacher.jsonl"
ADAPTER = Path("/run/media/s117/OS/Models/Qwen3.8-27B-dpo-lora")
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-uncensored")
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"
SCRATCH = Path("/tmp/grok-goal-baec64ca652a/implementer")
OUT = Path("logs/qwen38_cga_dpo.json")

N = 200
MAX_CHOSEN = 24
MAX_REJ = 48
MAX_SEQ = 384
LR = 2e-5
BETA = 0.1
ACCUM = 4
EVAL_EVERY = 10
MAX_STEPS = 40
LORA_R = 8


def _cfg() -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["cga_dpo", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 32
    cfg.inference.max_batch_size = 32
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = SteeringMode.ANGULAR
    return cfg


def _tok(engine, text: str, n: int) -> str:
    if not text.startswith(" "):
        text = " " + text
    ids = engine.tokenizer(text, add_special_tokens=False)["input_ids"][:n]
    out = engine.tokenizer.decode(ids, skip_special_tokens=False)
    return out if out.startswith(" ") else " " + out


def _mean_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    gather = tgt.clamp(min=0).unsqueeze(-1)
    token_lp = logp.gather(-1, gather).squeeze(-1)
    denom = mask.sum().clamp(min=1)
    return (token_lp * mask).sum() / denom


def _pack(engine, msg: ChatMessage, text: str):
    inputs, clen = engine._tokenize_with_continuations([msg], [text])
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    if ids.shape[1] > MAX_SEQ:
        ids = ids[:, -MAX_SEQ:]
        mask = mask[:, -MAX_SEQ:]
        clen = clen.new_tensor([min(int(clen[0]), MAX_SEQ)])
    labels = continuation_labels(ids, mask, clen)
    return ids, mask, labels


def _score(engine, cfg, tag: str) -> dict:
    was = engine.model.training
    engine.model.eval()
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    with torch.no_grad():
        kl = float(scorer.measure_kl_divergence(engine))
        refusals, _ = scorer.measure_compliance_objective(engine)
    if was:
        engine.model.train()
    n = len(scorer.target_msgs)
    row = {
        "tag": tag,
        "keyword_refusals": int(refusals),
        "n": int(n),
        "full_distribution_kl_3token_vs_original": kl,
        "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
    }
    print(f"SCORE {tag}: {refusals}/{n} @ {kl:.4f}  in_budget={row['in_budget']}", flush=True)
    return row


def _dump(payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    OUT.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_cga_dpo.json").write_text(text, encoding="utf-8")


def main() -> None:
    cfg = _cfg()
    payload = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": "DPO CGA-chosen vs original-rejected, PEFT ref=disable_adapter",
        "points": [],
    }
    print("loading original...", flush=True)
    engine = SteeringEngine(cfg)
    rows = [r for r in load_teacher_jsonl(TEACHER) if r["split"] == "harmful"][:N]
    msgs = [
        ChatMessage(system=r.get("system") or "You are a helpful assistant.", user=r["user"])
        for r in rows
    ]
    print(f"generating original rejected n={len(msgs)}", flush=True)
    rejected = engine.generate_text_batched(
        msgs,
        skip_special_tokens=True,
        max_new_tokens=MAX_REJ,
        min_new_tokens=8,
    )
    pairs = []
    for msg, r, rej in zip(msgs, rows, rejected):
        ch = _tok(engine, r["completion"], MAX_CHOSEN)
        rj = _tok(engine, rej or "", MAX_REJ)
        if not ch.strip() or not rj.strip():
            continue
        pairs.append((msg, ch, rj))
    print(f"dpo pairs {len(pairs)}", flush=True)

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R * 2,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["o_proj", "out_proj", "down_proj", "q_proj", "v_proj"],
        layers_to_transform=list(range(32, 64)),
    )
    engine.model = get_peft_model(engine.model, lora)
    engine.model.print_trainable_parameters()
    engine.model.train()
    # Checkpointing + disable_adapter() in one graph mismatches tensor metadata.
    if hasattr(engine.model, "gradient_checkpointing_disable"):
        engine.model.gradient_checkpointing_disable()
    trainable = [p for p in engine.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR)
    step = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    hit = None
    for i, (msg, ch, rj) in enumerate(pairs):
        ids_c, m_c, lab_c = _pack(engine, msg, ch)
        ids_r, m_r, lab_r = _pack(engine, msg, rj)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad(), engine.model.disable_adapter():
                lp_c_ref = _mean_logp(
                    engine.model(input_ids=ids_c, attention_mask=m_c).logits, lab_c
                )
                lp_r_ref = _mean_logp(
                    engine.model(input_ids=ids_r, attention_mask=m_r).logits, lab_r
                )
            lp_c = _mean_logp(engine.model(input_ids=ids_c, attention_mask=m_c).logits, lab_c)
            lp_r = _mean_logp(engine.model(input_ids=ids_r, attention_mask=m_r).logits, lab_r)
            delta = (lp_c - lp_c_ref.detach()) - (lp_r - lp_r_ref.detach())
            loss = -F.logsigmoid(BETA * delta) / ACCUM
        loss.backward()
        running += float(loss.detach()) * ACCUM
        if (i + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step} item {i+1}/{len(pairs)} loss={running:.4f}", flush=True)
            running = 0.0
            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                row = _score(engine, cfg, f"dpo_step{step}")
                payload["points"].append(row)
                _dump(payload)
                if row["in_budget"]:
                    hit = row
                    break
                if row["full_distribution_kl_3token_vs_original"] > 0.12:
                    print("KL>0.12, stop", flush=True)
                    break
            if step >= MAX_STEPS:
                break

    ADAPTER.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(ADAPTER))
    engine.tokenizer.save_pretrained(str(ADAPTER))
    print(f"wrote adapter {ADAPTER}", flush=True)
    if hit is None:
        payload["named_candidate"] = None
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _dump(payload)
        print("not in budget; skip merge", flush=True)
        return
    print("in budget — merging...", flush=True)
    merged = engine.model.merge_and_unload()
    if MERGED.exists():
        shutil.rmtree(MERGED)
    MERGED.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED), safe_serialization=True, max_shard_size="4GB")
    engine.tokenizer.save_pretrained(str(MERGED))
    payload["named_candidate"] = hit["tag"]
    payload["merged"] = str(MERGED)
    payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _dump(payload)
    print(f"wrote merged {MERGED}", flush=True)


if __name__ == "__main__":
    main()
