import torch

from abliterix.svf import (
    ConceptScorer,
    concept_scorer_cache_payload,
    load_concept_scorer_cache,
)


def test_concept_scorer_cache_round_trip_preserves_outputs_and_metrics():
    torch.manual_seed(3)
    scorer = ConceptScorer(8, 12)
    scorer.training_metrics = {"validation_accuracy": 0.875}
    x = torch.randn(4, 8)
    expected = scorer(x)
    payload = concept_scorer_cache_payload(
        {7: scorer},
        cache_key="gate-key",
        input_dim=8,
        hidden_dim_scorer=12,
    )

    loaded = load_concept_scorer_cache(
        payload,
        expected_cache_key="gate-key",
        device="cpu",
    )

    assert loaded is not None
    torch.testing.assert_close(loaded[7](x), expected)
    assert loaded[7].training_metrics == scorer.training_metrics


def test_concept_scorer_cache_rejects_wrong_provenance():
    payload = concept_scorer_cache_payload(
        {0: ConceptScorer(4, 8)},
        cache_key="old",
        input_dim=4,
        hidden_dim_scorer=8,
    )

    assert load_concept_scorer_cache(
        payload,
        expected_cache_key="new",
        device="cpu",
    ) is None
