import inspect

import torch

from candidate_features import (
    CANDIDATE_FEATURE_NAMES,
    candidate_feature_chunk,
    prepare_candidate_context,
    rank_positions,
)
from candidate_fusion import candidate_fusion
from candidate_gate import CandidateFeatureNormalizer, CandidateGate, candidate_gate_checkpoint, load_candidate_gate_checkpoint
from candidate_policies import policy_gate
from candidate_gate_common import split_gate_users
from train_candidate_gate import select_architecture_config
from utility_label import fixed_text_fusion


def test_candidate_gate_zero_is_exact_text_only():
    text, ids = torch.randn(3, 11), torch.randn(3, 11)
    output = candidate_fusion(text, ids, torch.zeros_like(text), max_alpha=0.5)
    assert torch.equal(output, text)


def test_candidate_gate_one_matches_fixed_fusion():
    text, ids = torch.randn(3, 11), torch.randn(3, 11)
    output = candidate_fusion(text, ids, torch.ones_like(text), fusion_temperature=1.2, max_alpha=0.5)
    expected = fixed_text_fusion(text, ids, alpha=0.5, temperature=1.2)
    assert torch.equal(output, expected)


def test_candidate_features_have_no_label_argument_and_are_finite():
    assert 'labels' not in inspect.signature(prepare_candidate_context).parameters
    text, ids = torch.randn(2, 13), torch.randn(2, 13)
    context = prepare_candidate_context(text, ids, torch.randn(2, 10), torch.arange(13).float())
    features = candidate_feature_chunk(context)
    assert features.shape == (2, 13, len(CANDIDATE_FEATURE_NAMES))
    assert torch.isfinite(features).all()


def test_rank_percentile_and_candidate_chunks_match_full_features():
    text = torch.tensor([[4.0, 1.0, 3.0, 2.0]])
    ids = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
    context = prepare_candidate_context(text, ids, torch.zeros(1, 10), torch.ones(4))
    full = candidate_feature_chunk(context)
    chunked = torch.cat([candidate_feature_chunk(context, 0, 2), candidate_feature_chunk(context, 2, 4)], dim=1)
    assert torch.equal(full, chunked)
    assert torch.allclose(full[0, :, 8], torch.tensor([0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0]))


def test_topk_intersection_and_union_gates():
    text_rank = rank_positions(torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]]))
    id_rank = rank_positions(torch.tensor([[1.0, 4.0, 5.0, 2.0, 3.0]]))
    top2 = policy_gate({'policy': 'P2_id_topk', 'k': 2}, text_rank, id_rank)
    intersection = policy_gate({'policy': 'P3_intersection', 'text_k': 2, 'id_k': 2}, text_rank, id_rank)
    union = policy_gate({'policy': 'P4_union', 'k': 2}, text_rank, id_rank)
    assert top2.sum().item() == 2
    assert torch.equal(intersection, torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(union, torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]))


def test_candidate_gate_range_initialization_and_chunk_equivalence():
    torch.manual_seed(9)
    text, ids = torch.randn(2, 9), torch.randn(2, 9)
    context = prepare_candidate_context(text, ids, torch.randn(2, 10), torch.ones(9))
    model = CandidateGate(len(CANDIDATE_FEATURE_NAMES), initial_probability=0.01)
    full_features = candidate_feature_chunk(context)
    normalizer = CandidateFeatureNormalizer(torch.zeros(len(CANDIDATE_FEATURE_NAMES)), torch.ones(len(CANDIDATE_FEATURE_NAMES)))
    full_gate = model(normalizer.transform(full_features))
    chunk_gate = torch.cat([
        model(normalizer.transform(candidate_feature_chunk(context, 0, 4))),
        model(normalizer.transform(candidate_feature_chunk(context, 4, 9))),
    ], dim=1)
    assert torch.allclose(full_gate, torch.full_like(full_gate, 0.01), atol=1e-6)
    assert torch.equal(full_gate, chunk_gate)
    assert ((full_gate >= 0) & (full_gate <= 1)).all()


def test_candidate_gate_checkpoint_roundtrip(tmp_path):
    model = CandidateGate(len(CANDIDATE_FEATURE_NAMES), architecture='linear')
    normalizer = CandidateFeatureNormalizer(torch.zeros(len(CANDIDATE_FEATURE_NAMES)), torch.ones(len(CANDIDATE_FEATURE_NAMES)))
    payload = candidate_gate_checkpoint(model, normalizer, {
        'feature_schema': 'schema', 'feature_names': CANDIDATE_FEATURE_NAMES,
        'dataset': 'Industrial_and_Scientific', 'seed': 42, 'max_alpha': 0.5,
    })
    path = tmp_path / 'candidate.pt'
    torch.save(payload, path)
    restored, restored_normalizer, metadata = load_candidate_gate_checkpoint(path)
    assert restored.architecture == 'linear'
    assert metadata['feature_names'] == CANDIDATE_FEATURE_NAMES
    assert torch.equal(restored_normalizer.mean, normalizer.mean)


def test_gate_user_split_and_selection_interface_exclude_independent_valid():
    cache = {
        'user_ids': torch.arange(20),
    }
    train_indices, dev_indices = split_gate_users(cache, seed=42)
    train_users = set(cache['user_ids'][train_indices].tolist())
    dev_users = set(cache['user_ids'][dev_indices].tolist())
    assert not train_users.intersection(dev_users)
    assert 'valid_cache' not in inspect.signature(select_architecture_config).parameters


def test_only_candidate_gate_parameters_receive_gradients():
    frozen_text = torch.randn(2, 7, requires_grad=False)
    frozen_residual = torch.rand(2, 7, requires_grad=False)
    features = torch.randn(2, 7, len(CANDIDATE_FEATURE_NAMES))
    model = CandidateGate(len(CANDIDATE_FEATURE_NAMES))
    final = frozen_text + model(features) * frozen_residual
    torch.nn.functional.cross_entropy(final, torch.tensor([1, 3])).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert frozen_text.grad is None and frozen_residual.grad is None
