import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_parent_module(model: nn.Module, module_name: str):
    parent = model
    parts = module_name.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class FlyLoRALinear(nn.Module):
    """FlyLoRA wrapped linear layer with task-aware implicit routing bias."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        k: int = 4,
        alpha: float = 32.0,
        sparsity_ratio: float = 0.25,
        bias_lr: float = 1e-3,
        num_tasks: int = 2,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.k = min(k, r)
        self.alpha = alpha
        self.sparsity_ratio = sparsity_ratio
        self.bias_lr = bias_lr
        self.num_tasks = num_tasks

        # frozen sparse random projection A in R^{r x n}
        A = torch.zeros(r, self.in_features)
        p = max(1, int(self.in_features * self.sparsity_ratio))
        for i in range(r):
            indices = torch.randperm(self.in_features)[:p]
            A[i, indices] = torch.randn(p) * (1.0 / max(1, r))
        self.register_buffer('flylora_A', A)

        # trainable projection B in R^{m x r}
        self.flylora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.zeros_(self.flylora_B)

        # task-wise routing bias d in R^{T x r}; updated by assignment statistics
        self.flylora_d = nn.Parameter(torch.zeros(num_tasks, r), requires_grad=False)
        self.active_task_id = 0

    def set_task_id(self, task_id: int):
        self.active_task_id = int(task_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        # y: (..., r)
        y = F.linear(x, self.flylora_A)
        d = self.flylora_d[self.active_task_id].to(y.dtype)
        y_biased = y + d

        _, selected_experts = torch.topk(y_biased.abs(), self.k, dim=-1)
        mask = torch.zeros_like(y_biased)
        mask.scatter_(-1, selected_experts, 1.0)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(selected_experts.reshape(-1), minlength=self.r).float()
                delta_bias = (counts.mean() - counts).sign() * self.bias_lr
                self.flylora_d[self.active_task_id] += delta_bias.to(self.flylora_d.device)

        activated_y = y * mask
        delta = F.linear(activated_y.to(self.flylora_B.dtype), self.flylora_B)
        delta = delta.to(base_out.dtype) * (self.alpha / max(1, self.r))
        return base_out + delta


def replace_with_flylora(
    model: nn.Module,
    target_module_names,
    r: int,
    k: int,
    alpha: float,
    sparsity_ratio: float,
    bias_lr: float,
):
    replaced = 0
    target_set = set(target_module_names)
    for module_name, module in list(model.named_modules()):
        if module_name not in target_set:
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent, child_name = _get_parent_module(model, module_name)
        setattr(
            parent,
            child_name,
            FlyLoRALinear(
                module,
                r=r,
                k=k,
                alpha=alpha,
                sparsity_ratio=sparsity_ratio,
                bias_lr=bias_lr,
            ),
        )
        replaced += 1
    return replaced


def set_flylora_task(model: nn.Module, task: str):
    task_id = 0 if task == 'text' else 1
    for module in model.modules():
        if isinstance(module, FlyLoRALinear):
            module.set_task_id(task_id)
