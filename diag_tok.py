#!/usr/bin/env python3
"""Quick check: what does apply_chat_template actually produce?"""
from transformers import AutoTokenizer

MODEL = "/run/media/s117/OS/Models/Laguna-S-2.1-bnb"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

chats = [[{"role": "user", "content": "What is 2+2?"}]]

print("=== With enable_thinking=False ===")
try:
    r1 = tok.apply_chat_template(chats, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    print(f"len={len(r1)}")
    print(repr(r1))
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n=== Without enable_thinking kwarg ===")
try:
    r2 = tok.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
    print(f"len={len(r2)}")
    print(repr(r2))
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

# Check what chat_template string is actually being used
print("\n=== tokenizer.chat_template (first 200 chars) ===")
ct = tok.chat_template
print(repr(ct[:200]) if ct else "None")
