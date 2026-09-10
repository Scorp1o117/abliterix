from torch import tensor

from abliterix.distill import continuation_labels, sft_loss


def test_continuation_labels_mask_prompt_and_left_pad():
    # row0: pad pad prompt cont cont  (cont len 2)
    # row1: pad prompt cont cont cont (cont len 3)
    input_ids = tensor(
        [
            [0, 0, 11, 21, 22],
            [0, 12, 31, 32, 33],
        ]
    )
    attn = tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ]
    )
    cont = tensor([2, 3])
    labels = continuation_labels(input_ids, attn, cont)
    assert labels[0].tolist() == [-100, -100, -100, 21, 22]
    assert labels[1].tolist() == [-100, -100, 31, 32, 33]


def test_sft_loss_ignores_masked_positions():
    # Causal shift: logits[t] predicts labels[t+1]. Only the last label
    # is supervised, so logits[1] must peak on class 3.
    logits = tensor(
        [[[0.0, 10.0, 0.0, 0.0], [0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    labels = tensor([[-100, -100, 3]])
    loss = sft_loss(logits, labels)
    assert loss.item() < 0.05
