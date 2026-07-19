import torch

from utility_label import rank_utility_labels, rank_utility_values


def test_rank_utility_matches_target_rank_gain():
    text = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
    ids = torch.tensor([[0.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    labels = torch.tensor([1, 1, 0])

    rank_gain = rank_utility_values(text, ids, labels, alpha0=1.0, temperature=1.0)
    labels_rank = rank_utility_labels(rank_gain)

    assert rank_gain[0] > 0
    assert rank_gain[1] == 0
    assert rank_gain[2] < 0
    assert torch.equal(labels_rank['utility_rank_label'], torch.tensor([1, 0, 0]))
    assert torch.equal(labels_rank['utility_rank_tie'], torch.tensor([0, 1, 0]))
    assert torch.equal(labels_rank['utility_rank_harm'], torch.tensor([0, 0, 1]))
