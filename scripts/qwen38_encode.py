# Abliterix custom encoder for Qwen3.8-27B.
#
# The bundled Jinja defaults enable_thinking=true / reasoning_effort=xhigh,
# which opens <think> and eats the eval token budget. Force thinking off
# so refusals land in the visible answer (empty <think></think> then text).

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer

MODEL_ID = "/run/media/s117/OS/Models/Qwen3.8-27B"


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def encode_messages(messages, add_generation_prompt=False, enable_thinking=False, **_kw):
    return _tokenizer().apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
