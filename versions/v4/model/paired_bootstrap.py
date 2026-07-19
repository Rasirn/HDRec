import torch


def bootstrap_evidence(result):
    """Classify positive evidence without overstating a CI that crosses zero."""
    if result['ci95_low'] > 0:
        return 'strong'
    if result['mean_difference'] > 0 and result['probability_improvement_gt_zero'] >= 0.90:
        return 'weak_positive'
    return 'none'


def paired_bootstrap(differences, num_samples=1000, seed=42):
    differences = differences.float().cpu()
    generator = torch.Generator().manual_seed(seed)
    means = torch.empty(num_samples)
    n = differences.numel()
    for index in range(num_samples):
        sample_indices = torch.randint(0, n, (n,), generator=generator)
        means[index] = differences[sample_indices].mean()
    interval = torch.quantile(means, torch.tensor([0.025, 0.975]))
    result = {
        'mean_difference': differences.mean().item(),
        'ci95_low': interval[0].item(),
        'ci95_high': interval[1].item(),
        'probability_improvement_gt_zero': (means > 0).float().mean().item(),
        'num_bootstrap_samples': int(num_samples),
        'seed': int(seed),
    }
    result['evidence'] = bootstrap_evidence(result)
    return result
