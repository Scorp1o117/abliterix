#!/usr/bin/env python3
"""Diagnose why baseline response length is 0."""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "/run/media/s117/OS/Models/Laguna-S-2.1-bnb"

print("=== Loading tokenizer ===")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

# 1) Check what apply_chat_template produces
chats = [[{"role": "user", "content": "What is 2+2?"}]]
print("\n=== apply_chat_template (enable_thinking=False) ===")
try:
    rendered = tok.apply_chat_template(chats, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    print(f"Length: {len(rendered)}")
    print(repr(rendered[:500]))
except Exception as e:
    print(f"ERROR: {e}")

print("\n=== apply_chat_template (no enable_thinking kwarg) ===")
try:
    rendered2 = tok.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
    print(f"Length: {len(rendered2)}")
    print(repr(rendered2[:500]))
except Exception as e:
    print(f"ERROR: {e}")

# 2) Load model and generate
print("\n=== Loading model (bnb 4bit) ===")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True
)
model.eval()

print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# 3) Generate with the rendered prompt
print("\n=== Generating ===")
inputs = tok(rendered, return_tensors="pt").to(model.device)
print(f"Input tokens: {inputs['input_ids'].shape[1]}")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tok.pad_token_id)

gen_tokens = out[:, inputs["input_ids"].shape[1]:]
print(f"Generated tokens: {gen_tokens.shape[1]}")
print(f"Token ids: {gen_tokens[0].tolist()[:20]}")

decoded = tok.decode(gen_tokens[0], skip_special_tokens=False)
print(f"Decoded (keep special): {repr(decoded[:300])}")

decoded_skip = tok.decode(gen_tokens[0], skip_special_tokens=True)
print(f"Decoded (skip special): {repr(decoded_skip[:300])}")
print(f"Word count (skip): {len(decoded_skip.split())}")
