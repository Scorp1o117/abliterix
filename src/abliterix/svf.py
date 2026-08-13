# Abliterix
# Copyright (C) 2026  Wangzhang Wu <wangzhangwu1216@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Steering Vector Fields (SVF) — learned context-dependent steering.

Implements the core idea from arxiv:2602.01654: instead of a static steering
vector, SVF learns a differentiable concept scoring function ``f(h)`` whose
gradient ``∇_h f`` defines a context-dependent steering direction at each
activation ``h``.  This makes the steering intervention adapt to the current
hidden state, enabling more precise and reliable control.

The ConceptScorer is a small MLP trained per layer to distinguish harmful
from harmless activations.  During inference, its gradient provides the
locally optimal steering direction at each token position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .util import print


class ConceptScorer(nn.Module):
    """Small MLP that scores activations on a harmful/harmless spectrum.

    Architecture: Linear → GELU → Linear → GELU → Linear → Sigmoid

    The gradient of the output with respect to the input provides the
    context-dependent steering direction.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Score activations.  Returns values in [0, 1]."""
        return self.net(x)


def train_concept_scorers(
    benign_states: Tensor,
    target_states: Tensor,
    hidden_dim: int,
    n_epochs: int = 50,
    lr: float = 1e-3,
    hidden_dim_scorer: int = 256,
    device: torch.device | str | None = None,
    validation_benign_states: Tensor | None = None,
    validation_target_states: Tensor | None = None,
    seed: int = 0,
) -> dict[int, ConceptScorer]:
    """Train one ConceptScorer per transformer layer.

    Benign activations are labelled 0 (low concept score); target (harmful)
    activations are labelled 1 (high concept score).  After training, the
    gradient of the scorer w.r.t. the input provides the direction that
    maximally increases the "harmful" score — which is exactly the
    context-dependent refusal direction to steer away from.

    Parameters
    ----------
    benign_states, target_states : Tensor
        Shape ``(n, layers+1, hidden_dim)``.
    hidden_dim : int
        Input dimension of each scorer.
    n_epochs : int
        Training epochs per layer.
    lr : float
        Learning rate.
    hidden_dim_scorer : int
        Hidden dimension for the scorer MLP.
    device : torch.device | str | None
        Device used for scorer training.  Defaults to the hidden-state device;
        callers loading cached CPU residuals can explicitly select a GPU.

    Returns
    -------
    dict[int, ConceptScorer]
        Mapping from transformer layer index (0-based) to trained scorer.
        Index 0 corresponds to the embedding layer and is excluded.
    """
    # Make scorer weights and shuffles independent of any RNG consumed while
    # constructing the steering basis. This is required for direction-only
    # paired experiments.
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    n_layers = benign_states.shape[1]
    train_device = torch.device(device) if device is not None else benign_states.device
    scorers: dict[int, ConceptScorer] = {}

    for layer_idx in range(1, n_layers):  # Skip embedding layer (index 0).
        b = benign_states[:, layer_idx, :].to(train_device, dtype=torch.float32)
        t = target_states[:, layer_idx, :].to(train_device, dtype=torch.float32)

        # Build dataset: benign = 0, target = 1.
        X = torch.cat([b, t], dim=0)
        y = torch.cat(
            [
                torch.zeros(b.shape[0], 1, device=train_device),
                torch.ones(t.shape[0], 1, device=train_device),
            ]
        )

        scorer = ConceptScorer(hidden_dim, hidden_dim_scorer).to(train_device)
        optimizer = torch.optim.Adam(scorer.parameters(), lr=lr)

        scorer.train()
        with torch.enable_grad():
            for _epoch in range(n_epochs):
                # Shuffle.
                perm = torch.randperm(X.shape[0], device=train_device)
                X_shuf, y_shuf = X[perm], y[perm]

                pred = scorer(X_shuf)
                loss = F.binary_cross_entropy(pred, y_shuf)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        scorer.eval()

        # Select on held-out prompt groups when supplied. This is important for
        # trajectory data, where several tokens from one prompt are correlated.
        with torch.no_grad():
            pred_labels = (scorer(X) > 0.5).float()
            train_acc = (pred_labels == y).float().mean().item()
            if validation_benign_states is not None:
                if validation_target_states is None:
                    raise ValueError(
                        "validation_target_states is required with validation_benign_states"
                    )
                vb = validation_benign_states[:, layer_idx, :].to(
                    train_device, dtype=torch.float32
                )
                vt = validation_target_states[:, layer_idx, :].to(
                    train_device, dtype=torch.float32
                )
                vX = torch.cat([vb, vt], dim=0)
                vy = torch.cat(
                    [
                        torch.zeros(vb.shape[0], 1, device=train_device),
                        torch.ones(vt.shape[0], 1, device=train_device),
                    ]
                )
                val_acc = ((scorer(vX) > 0.5).float() == vy).float().mean().item()
            else:
                val_acc = train_acc

        scorer.training_metrics = {
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
        }

        if val_acc > 0.6:  # Only keep scorers that generalise usefully.
            scorers[layer_idx - 1] = scorer  # Map to 0-based layer index.

    print(
        f"* {len(scorers)}/{n_layers - 1} layers with effective scorers "
        f"(validation accuracy > 60%)"
    )
    return scorers


def concept_scorer_cache_payload(
    scorers: dict[int, ConceptScorer],
    *,
    cache_key: str,
    input_dim: int,
    hidden_dim_scorer: int,
) -> dict:
    """Build a CPU-only, reconstructable concept-scorer cache payload."""
    return {
        "cache_key": cache_key,
        "input_dim": int(input_dim),
        "hidden_dim_scorer": int(hidden_dim_scorer),
        "scorers": {
            int(layer_idx): {
                "state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in scorer.state_dict().items()
                },
                "training_metrics": dict(
                    getattr(scorer, "training_metrics", {})
                ),
            }
            for layer_idx, scorer in scorers.items()
        },
    }


def load_concept_scorer_cache(
    payload: dict,
    *,
    expected_cache_key: str,
    device: torch.device | str,
) -> dict[int, ConceptScorer] | None:
    """Reconstruct cached scorers, returning ``None`` on provenance mismatch."""
    if payload.get("cache_key") != expected_cache_key:
        return None
    input_dim = int(payload["input_dim"])
    hidden_dim_scorer = int(payload["hidden_dim_scorer"])
    scorers: dict[int, ConceptScorer] = {}
    for raw_layer_idx, entry in payload["scorers"].items():
        scorer = ConceptScorer(input_dim, hidden_dim_scorer).to(device)
        scorer.load_state_dict(entry["state_dict"], strict=True)
        scorer.training_metrics = dict(entry.get("training_metrics", {}))
        scorer.eval()
        scorers[int(raw_layer_idx)] = scorer
    return scorers


def evaluate_concept_scorers(
    scorers: dict[int, ConceptScorer],
    benign_states: Tensor,
    target_states: Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Summarise held-out class separation across retained layer scorers."""
    if not scorers:
        return {
            "layers": 0.0,
            "accuracy_mean": 0.0,
            "benign_active_mean": 0.0,
            "target_active_mean": 0.0,
            "score_margin_mean": 0.0,
        }
    accuracies = []
    benign_rates = []
    target_rates = []
    margins = []
    with torch.no_grad():
        for layer_idx, scorer in scorers.items():
            scorer_device = next(scorer.parameters()).device
            b = benign_states[:, layer_idx + 1, :].to(
                scorer_device, dtype=torch.float32
            )
            t = target_states[:, layer_idx + 1, :].to(
                scorer_device, dtype=torch.float32
            )
            b_scores = scorer(b).flatten()
            t_scores = scorer(t).flatten()
            b_rate = (b_scores >= threshold).float().mean().item()
            t_rate = (t_scores >= threshold).float().mean().item()
            accuracy = 0.5 * ((1.0 - b_rate) + t_rate)
            accuracies.append(accuracy)
            benign_rates.append(b_rate)
            target_rates.append(t_rate)
            margins.append(t_scores.mean().item() - b_scores.mean().item())
    return {
        "layers": float(len(scorers)),
        "accuracy_mean": sum(accuracies) / len(accuracies),
        "benign_active_mean": sum(benign_rates) / len(benign_rates),
        "target_active_mean": sum(target_rates) / len(target_rates),
        "score_margin_mean": sum(margins) / len(margins),
    }
