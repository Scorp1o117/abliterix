#!/usr/bin/env python3
"""Distill the CGA runtime candidate into a mergeable LoRA, then merge.

Teacher: original Qwen3.8-27B + concept-gated angular sidecar
         (cga_low_global_prompt_t0.35_1.60).
Student: LoRA on the same base, SFT on teacher completions.
Writes:
  artifacts/qwen38_cga_distill/teacher.jsonl
  /run/media/s117/OS/Models/Qwen3.8-27B-cga-distill-lora   (adapter)
  /run/media/s117/OS/Models/Qwen3.8-27B-cga-distill        (merged BF16)

Usage:
  python scripts/bake_qwen38_cga_distill.py --phase gen
  python scripts/bake_qwen38_cga_distill.py --phase train
  python scripts/bake_qwen38_cga_distill.py --phase eval
  python scripts/bake_qwen38_cga_distill.py --phase all
"""

from __future__ import annotations

import argparse
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

from abliterix.scriptlib import setup_io  # noqa: E402

setup_io()

import torch  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402

from abliterix.core.engine import SteeringEngine  # noqa: E402
from abliterix.core.steering import apply_steering  # noqa: E402
from abliterix.data import load_prompt_dataset  # noqa: E402
from abliterix.distill import (  # noqa: E402
    continuation_labels,
    load_teacher_jsonl,
    sft_loss,
    write_teacher_jsonl,
)
from abliterix.eval.detector import RefusalDetector  # noqa: E402
from abliterix.eval.scorer import TrialScorer  # noqa: E402
from abliterix.settings import AbliterixConfig  # noqa: E402
from abliterix.types import ChatMessage, SteeringMode, SteeringProfile  # noqa: E402
from eval_qwen38_vs_original import (  # noqa: E402
    POCKET_BASELINE,
    _inject_original_baseline,
)

CONFIG = "configs/qwen38_27b_rocm_pocket.toml"
ARTIFACT = ROOT / "artifacts" / "qwen38_cga_global_t035_s160.json"
POCKET_VECTORS = ROOT / (
    "checkpoints_qwen38_27b_pocket/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_steering.pt"
)
SCORER_CACHE = ROOT / (
    "checkpoints_qwen38_27b_cga/"
    "--run--media--s117--OS--Models--Qwen3--8-27B_concept_scorers.pt"
)
TEACHER = ROOT / "artifacts" / "qwen38_cga_distill" / "teacher.jsonl"
ADAPTER = Path("/run/media/s117/OS/Models/Qwen3.8-27B-cga-distill-lora")
MERGED = Path("/run/media/s117/OS/Models/Qwen3.8-27B-cga-distill")
SCRATCH = Path("/tmp/grok-goal-8ac14a085a92/implementer")
ORIGINAL = "/run/media/s117/OS/Models/Qwen3.8-27B"

HARMFUL_N = 800
BENIGN_N = 256
MAX_SEQ = 768
LORA_R = 16
LORA_ALPHA = 32
LR = 2e-4
EPOCHS = 1
ACCUM = 8
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


def _cfg(mode: SteeringMode) -> AbliterixConfig:
    os.environ["AX_CONFIG"] = CONFIG
    sys.argv = ["bake_cga_distill", "--config", CONFIG, "--seed", "117"]
    cfg = AbliterixConfig()
    cfg.inference.batch_size = 128
    cfg.inference.max_batch_size = 128
    cfg.detection.llm_judge = False
    cfg.steering.steering_mode = mode
    cfg.steering.runtime_hook_site = "decoder_block"
    return cfg


def _apply_cga(engine: SteeringEngine, cfg: AbliterixConfig) -> None:
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    cfg.steering.steering_mode = SteeringMode.CONCEPT_GATED_ANGULAR
    cfg.steering.concept_gate_scope = art["concept_gate_scope"]
    cfg.steering.concept_gate_threshold = float(art["concept_gate_threshold"])
    cfg.steering.concept_gate_global_decision_layer = int(
        art["concept_gate_global_decision_layer"]
    )
    cfg.steering.concept_gate_angular_overrotation = bool(
        art["concept_gate_angular_overrotation"]
    )
    cfg.steering.concept_gate_positive_alignment_only = bool(
        art["concept_gate_positive_alignment_only"]
    )
    blob = torch.load(SCORER_CACHE, map_location="cpu", weights_only=False)
    engine._concept_scorers = (
        blob["scorers"] if isinstance(blob, dict) and "scorers" in blob else blob
    )
    vectors = torch.load(POCKET_VECTORS, map_location="cpu", weights_only=False)[
        "vectors"
    ]
    prof = art["profile"]["attn.o_proj"]
    profiles = {
        "attn.o_proj": SteeringProfile(
            max_weight=float(prof["max_weight"]),
            max_weight_position=float(prof["max_weight_position"]),
            min_weight=float(prof["min_weight"]),
            min_weight_distance=float(prof["min_weight_distance"]),
        )
    }
    apply_steering(engine, vectors, None, profiles, cfg)
    print(
        f"CGA hooks={len(getattr(engine, '_angular_hooks', []) or [])} "
        f"thresh={cfg.steering.concept_gate_threshold} "
        f"strength={prof['max_weight']}",
        flush=True,
    )


def phase_gen() -> None:
    torch.set_grad_enabled(False)
    cfg = _cfg(SteeringMode.CONCEPT_GATED_ANGULAR)
    print("loading CGA teacher...", flush=True)
    engine = SteeringEngine(cfg)
    _apply_cga(engine, cfg)
    detector = RefusalDetector(cfg)
    harmful = load_prompt_dataset(cfg, cfg.target_prompts)[:HARMFUL_N]
    benign = load_prompt_dataset(cfg, cfg.benign_prompts)[:BENIGN_N]
    rows: list[dict] = []

    def _gen(split: str, msgs) -> None:
        print(f"generating {split} n={len(msgs)}", flush=True)
        texts = engine.generate_text_batched(
            msgs,
            skip_special_tokens=True,
            max_new_tokens=cfg.inference.max_gen_tokens,
            min_new_tokens=cfg.inference.min_gen_tokens,
        )
        kept = 0
        for msg, text in zip(msgs, texts):
            text = (text or "").strip()
            if not text:
                continue
            refused = bool(detector.detect_refusal(text))
            if split == "harmful" and refused:
                continue
            rows.append(
                {
                    "split": split,
                    "user": msg.user,
                    "system": msg.system,
                    "completion": text,
                    "teacher_refused": refused,
                }
            )
            kept += 1
        print(f"  kept {kept}/{len(msgs)} {split}", flush=True)

    _gen("harmful", harmful)
    _gen("benign", benign)
    n = write_teacher_jsonl(TEACHER, rows)
    print(f"wrote {n} pairs → {TEACHER}", flush=True)
    del engine
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def phase_train() -> None:
    rows = load_teacher_jsonl(TEACHER)
    if not rows:
        raise SystemExit(f"empty teacher set {TEACHER}")
    n_h = sum(1 for r in rows if r["split"] == "harmful")
    n_b = sum(1 for r in rows if r["split"] == "benign")
    print(f"teacher pairs {len(rows)} (harmful={n_h} benign={n_b})", flush=True)

    cfg = _cfg(SteeringMode.ANGULAR)
    print("loading original student (no hooks)...", flush=True)
    engine = SteeringEngine(cfg)
    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
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

    pairs = [
        (
            ChatMessage(
                system=r.get("system") or "You are a helpful assistant.",
                user=r["user"],
            ),
            r["completion"]
            if r["completion"].startswith(" ")
            else " " + r["completion"],
        )
        for r in rows
    ]

    trainable = [p for p in engine.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR)
    engine.model.train()
    step = 0
    opt.zero_grad(set_to_none=True)
    running = 0.0
    for epoch in range(EPOCHS):
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
                        f"epoch {epoch+1} item {i+1}/{len(pairs)} "
                        f"loss={running / max(step, 1):.4f}",
                        flush=True,
                    )
                    running = 0.0

    ADAPTER.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(str(ADAPTER))
    engine.tokenizer.save_pretrained(str(ADAPTER))
    print(f"wrote adapter {ADAPTER}", flush=True)

    print("merging LoRA into BF16...", flush=True)
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
        "config.json",
    ):
        src = Path(ORIGINAL) / extra
        if src.is_file() and not (MERGED / extra).exists():
            shutil.copy2(src, MERGED / extra)
    print(f"wrote merged {MERGED}", flush=True)


def phase_eval() -> None:
    cfg = _cfg(SteeringMode.ANGULAR)
    cfg.model.model_id = str(MERGED)
    print(f"eval merged {cfg.model.model_id}", flush=True)
    engine = SteeringEngine(cfg)
    detector = RefusalDetector(cfg)
    scorer = TrialScorer(cfg, engine, detector, defer_baseline=True)
    _inject_original_baseline(scorer, POCKET_BASELINE)
    kl = float(scorer.measure_kl_divergence(engine))
    refusals, _ = scorer.measure_compliance_objective(engine)
    n = len(scorer.target_msgs)
    payload = {
        "named_candidate": "Qwen3.8-27B-cga-distill",
        "merged": str(MERGED),
        "adapter": str(ADAPTER),
        "keyword_refusals": int(refusals),
        "n": int(n),
        "full_distribution_kl_3token_vs_original": kl,
        "in_budget": bool(8 <= int(refusals) <= 12 and kl <= 0.1),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    text = json.dumps(payload, indent=2)
    out = ROOT / "logs" / "qwen38_cga_distill_eval.json"
    out.write_text(text, encoding="utf-8")
    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "qwen38_cga_distill_eval.json").write_text(text, encoding="utf-8")
    print(
        f"SCORE cga_distill: {refusals}/{n} @ {kl:.4f}  in_budget={payload['in_budget']}",
        flush=True,
    )
    print(text, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("gen", "train", "eval", "all"), default="all")
    args = ap.parse_args()
    if args.phase in ("gen", "all"):
        phase_gen()
    if args.phase in ("train", "all"):
        if not TEACHER.is_file():
            raise SystemExit(f"missing {TEACHER}; run --phase gen first")
        phase_train()
    if args.phase in ("eval", "all"):
        if not MERGED.is_dir():
            raise SystemExit(f"missing {MERGED}; run --phase train first")
        phase_eval()


if __name__ == "__main__":
    main()
