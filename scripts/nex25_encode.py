# Abliterix custom encoder for Nex-N2.5-mini (Nex-AGI, Qwen3.5-35B-A3B base).
#
# The bundled Jinja template keys off ``reasoning_effort``: when the kwarg is
# absent (which is what Abliterix passes — it only sends
# ``enable_thinking=False``, ignored by this template) the template opens a
# bare ``<think>`` block, so the model spends the eval token budget thinking
# before it answers. With ``reasoning_effort="none"`` the template instead
# pre-closes an empty ``<think>\n\n</think>`` block and the model answers
# directly — the same failure mode the Muse-Glimmer encoder fixes.
#
# Default is "none" for abliteration; override with
# ``custom_encoder_kwargs = {reasoning_effort = "high"}`` to keep thinking.

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer

MODEL_ID = "/run/media/s117/OS/Models/Nex-N2.5-mini"


@lru_cache(maxsize=1)
def _tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def encode_messages(
    messages,
    add_generation_prompt=False,
    reasoning_effort="none",
    **_kw,
):
    return _tokenizer().apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        reasoning_effort=reasoning_effort,
    )
