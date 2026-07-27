#!/usr/bin/env python3
"""Diagnose baseline response length=0 — exact abliterix config."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "/run/media/s117/OS/Models/Laguna-S-2.1-bnb"
SYSTEM_PROMPT = "You are a helpful assistant."

print("=== Loading tokenizer ===")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# --- Exact abliterix bnb config (engine.py:811-817) ---
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

print("=== Loading model (bnb 4bit, exact abliterix config) ===")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    quantization_config=bnb,
    device_map="auto",
    max_memory={0: "120GB"},
    trust_remote_code=True,
)
model.eval()

vram = torch.cuda.memory_allocated() / 1e9
print(f"VRAM: {vram:.2f} GB")
print(f"is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', 'N/A')}")

# Count 4bit params
n4 = sum(1 for p in model.parameters() if p.__class__.__name__ == "Params4bit")
print(f"Params4bit count: {n4}")

# --- Build chat exactly like abliterix _tokenize ---
test_prompts = [
    "What is 2+2?",
    "Write a haiku about the ocean.",
    "Explain quantum computing in one sentence.",
]
chats = []
for p in test_prompts:
    chats.append([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}])

texts = tok.apply_chat_template(chats, add_generation_prompt=True, tokenize=False, enable_thinking=False)
print(f"\n=== Rendered prompt[0] (len={len(texts[0])}) ===")
print(repr(texts[0][:300]))

# --- Generate exactly like abliterix _generate ---
print("\n=== Generating (max_new_tokens=100, do_sample=False) ===")
inputs = tok(texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(model.device)
print(f"Input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )

gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
print(f"Generated tokens per sample: {[g.shape[0] for g in gen_ids]}")

for i, g in enumerate(gen_ids):
    decoded_skip = tok.decode(g, skip_special_tokens=True)
    decoded_keep = tok.decode(g, skip_special_tokens=False)
    print(f"\n--- Sample {i} ---")
    print(f"  Gen token count: {g.shape[0]}")
    print(f"  Token IDs (first 15): {g[:15].tolist()}")
    print(f"  Decoded (skip_special): {repr(decoded_skip[:300])}")
    print(f"  Decoded (keep_special): {repr(decoded_keep[:300])}")
    print(f"  Word count (skip): {len(decoded_skip.split())}")

print("\n=== DONE ===")
