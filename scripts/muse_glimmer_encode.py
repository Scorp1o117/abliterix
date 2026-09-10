# Abliterix custom encoder for Muse Glimmer-30B.
#
# The bundled Jinja template keys off ``reasoning_strength`` (default "high")
# and ignores Abliterix's ``enable_thinking=False`` kwarg. High reasoning
# emits an ``assistant to=self`` chain-of-thought that can eat the eval
# token budget (same failure mode as Ling). Force low reasoning for text
# abliteration.

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer

MODEL_ID = "/run/media/s117/OS/Models/Muse-Glimmer-30B"


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def encode_messages(messages, add_generation_prompt=False, reasoning_strength="low", **_kw):
    return _tokenizer().apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        reasoning_strength=reasoning_strength,
    )
