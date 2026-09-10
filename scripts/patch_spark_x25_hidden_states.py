#!/usr/bin/env python3
"""Patch Spark-X2.5 modeling_spark.py so Abliterix can read hidden states.

Upstream Spark2_5Model.forward never collects per-layer residuals, and
Spark2_5ForCausalLM.forward does not forward output_hidden_states into the
inner model. Heretic/Abliterix need outputs.hidden_states of length
num_hidden_layers + 1 (embeddings + post-norm last layer), matching Llama.

Idempotent: re-running leaves an already-patched file unchanged.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = "ABLITERIX_HIDDEN_STATES_PATCH"

TIED_OLD = '_tied_weights_keys = ["lm_head.weight"]  # noqa: RUF012'
TIED_NEW = (
    '_tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}  # '
    + MARKER
)

MASK_OLD = '''            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }'''

MASK_NEW = '''            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,  # ''' + MARKER + '''
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }'''

INNER_FORWARD_OLD = '''        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )'''

INNER_FORWARD_NEW = '''        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,  # ''' + MARKER + '''
        )'''

LAYER_LOOP_OLD = '''        for decoder_layer in self.layers:
            layer_type = decoder_layer.layer_type
            position_embeddings = rope_cache.get(layer_type, rope_cache.get("full_attention"))
            layer_attention_mask = causal_mask_mapping.get(layer_type, causal_mask_mapping.get("full_attention"))

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    layer_attention_mask,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_ids=position_ids,
                )

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.to(dtype)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )'''

LAYER_LOOP_NEW = '''        output_hidden_states = kwargs.get("output_hidden_states", False)
        all_hidden_states = () if output_hidden_states else None  # ''' + MARKER + '''

        for decoder_layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (hidden_states,)
            layer_type = decoder_layer.layer_type
            position_embeddings = rope_cache.get(layer_type, rope_cache.get("full_attention"))
            layer_attention_mask = causal_mask_mapping.get(layer_type, causal_mask_mapping.get("full_attention"))

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    layer_attention_mask,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_ids=position_ids,
                )

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.to(dtype)
        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )'''


def patch_file(path: Path, *, dry_run: bool = False) -> str:
    text = path.read_text(encoding="utf-8")
    already_hs = MARKER in text and "hidden_states=all_hidden_states" in text
    already_tied = (
        TIED_NEW in text or '"lm_head.weight": "model.embedding.weight"' in text
    )
    already_mask = '"inputs_embeds": inputs_embeds' in text and (
        '"input_embeds": inputs_embeds' not in text
    )
    if already_hs and already_tied and already_mask:
        return "already_patched"

    new = text
    if not already_hs:
        if INNER_FORWARD_OLD not in new:
            raise SystemExit(f"Could not find CausalLM inner-model call in {path}")
        if LAYER_LOOP_OLD not in new:
            raise SystemExit(f"Could not find Spark2_5Model layer loop in {path}")
        new = new.replace(INNER_FORWARD_OLD, INNER_FORWARD_NEW, 1)
        new = new.replace(LAYER_LOOP_OLD, LAYER_LOOP_NEW, 1)
    if not already_tied:
        if TIED_OLD not in new:
            raise SystemExit(f"Could not find list-form _tied_weights_keys in {path}")
        new = new.replace(TIED_OLD, TIED_NEW, 1)
    if not already_mask:
        if MASK_OLD not in new:
            raise SystemExit(f"Could not find transformers-4.57 mask_kwargs in {path}")
        new = new.replace(MASK_OLD, MASK_NEW, 1)
    if dry_run:
        return "would_patch"
    backup = path.with_suffix(path.suffix + ".pre-abliterix")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(new, encoding="utf-8")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/run/media/s117/OS/Models/Spark-X2.5-4B",
        help="Local Spark-X2.5-4B directory",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = Path(args.model) / "modeling_spark.py"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    status = patch_file(path, dry_run=args.dry_run)
    print(f"{status}: {path}")


if __name__ == "__main__":
    main()
