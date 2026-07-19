import torch


def paired_bootstrap(differences, num_samples=1000, seed=42):
    differences = differences.float().cpu()
    generator = torch.Generator().manual_seed(seed)
    means = torch.empty(num_samples)
    n = differences.numel()
    for index in range(num_samples):
        sample_indices = torch.randint(0, n, (n,), generator=generator)
        means[index] = differences[sample_indices].mean()
    interval = torch.quantile(means, torch.tensor([0.025, 0.975]))
    return {
        'mean_difference': differences.mean().item(),
        'ci95_low': interval[0].item(),
        'ci95_high': interval[1].item(),
        'probability_improvement_gt_zero': (means > 0).float().mean().item(),
        'num_bootstrap_samples': int(num_samples),
        'seed': int(seed),
    }
