# Abliterix
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bake runtime steering into a mergeable LoRA via teacher distillation.

Runtime modes (angular / concept-gated angular) cannot be represented by a
static weight edit. This module imitates the teacher's completions with a
PEFT LoRA on the original base, then merges the adapter so the result loads
as an ordinary HF checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = [
    "continuation_labels",
    "load_teacher_jsonl",
    "write_teacher_jsonl",
    "sft_loss",
]


def continuation_labels(
    input_ids: Tensor,
    attention_mask: Tensor,
    continuation_lengths: Tensor,
) -> Tensor:
    """Build causal-LM labels that supervise only the teacher continuation.

    ``input_ids`` are left-padded; the continuation occupies the final
    ``continuation_lengths[i]`` positions of row ``i``. Padding and the
    prompt prefix are ``-100``.
    """
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    seq = labels.shape[1]
    for i, n in enumerate(continuation_lengths.tolist()):
        n = int(n)
        if n <= 0 or n > seq:
            labels[i] = -100
            continue
        labels[i, : seq - n] = -100
    return labels


def sft_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Next-token cross-entropy with ``-100`` ignored, matching HF CausalLM."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def write_teacher_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_teacher_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
