import torch

from abliterix.core.engine import _BannedPrefixProcessor


def test_banned_prefix_processor_masks_only_listed_steps():
    scores = torch.zeros(2, 8)
    processor = _BannedPrefixProcessor([[3, 5], None])
    first = processor(torch.zeros(2, 4, dtype=torch.long), scores.clone())
    assert first[0, 3] == float("-inf")
    assert first[0, 5] == 0
    assert torch.equal(first[1], torch.zeros(8))

    second = processor(torch.zeros(2, 5, dtype=torch.long), scores.clone())
    assert second[0, 5] == float("-inf")
    assert second[0, 3] == 0


def test_banned_prefix_processor_ignores_finished_rows():
    processor = _BannedPrefixProcessor([[1], None])
    processor.start_len = 4
    scores = torch.zeros(2, 6)
    out = processor(torch.zeros(2, 6, dtype=torch.long), scores)
    assert torch.equal(out, scores)
