import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_parent_module(model: nn.Module, module_name: str):
    parent = model
    parts = module_name.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class DualFlyLoRALinear(nn.Module):
    """FlyLoRA layer with task-specific frozen sparse projections and trainable B matrices."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        k: int = 4,
        alpha: float = 32.0,
        sparsity_ratio: float = 0.25,
        bias_lr: float = 1e-3,
        output_mix: float = 0.5,
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
        self.output_mix = float(output_mix)

        self.register_buffer('flylora_A_text', self._build_sparse_random_A())
        self.register_buffer('flylora_A_id', self._build_sparse_random_A())

        self.flylora_B_text = nn.Parameter(torch.zeros(self.out_features, r))
        self.flylora_B_id = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.zeros_(self.flylora_B_text)
        nn.init.zeros_(self.flylora_B_id)

        self.flylora_d_text = nn.Parameter(torch.zeros(r), requires_grad=False)
        self.flylora_d_id = nn.Parameter(torch.zeros(r), requires_grad=False)
        self.active_task = 'fused'

    def _build_sparse_random_A(self):
        A = torch.zeros(self.r, self.in_features)
        p = max(1, int(self.in_features * self.sparsity_ratio))
        for i in range(self.r):
            indices = torch.randperm(self.in_features)[:p]
            A[i, indices] = torch.randn(p) * (1.0 / max(1, self.r))
        return A

    def set_task(self, task: str):
        if task not in ('text', 'id', 'fused'):
            raise ValueError(f'Unsupported task: {task}')
        self.active_task = task

    def set_output_mix(self, output_mix: float):
        self.output_mix = float(output_mix)

    def _compute_delta(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, d: torch.Tensor, branch: str):
        y = F.linear(x, A)
        y_biased = y + d.to(y.dtype)

        _, selected = torch.topk(y_biased.abs(), self.k, dim=-1)
        mask = torch.zeros_like(y_biased)
        mask.scatter_(-1, selected, 1.0)

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(selected.reshape(-1), minlength=self.r).float()
                delta_bias = (counts.mean() - counts).sign() * self.bias_lr
                if branch == 'text':
                    self.flylora_d_text += delta_bias.to(self.flylora_d_text.device)
                else:
                    self.flylora_d_id += delta_bias.to(self.flylora_d_id.device)

        delta = F.linear((y * mask).to(B.dtype), B)
        delta = delta.to(x.dtype) * (self.alpha / max(1, self.r))
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.active_task == 'text':
            delta = self._compute_delta(x, self.flylora_A_text, self.flylora_B_text, self.flylora_d_text, 'text')
            return base_out + delta.to(base_out.dtype)

        if self.active_task == 'id':
            delta = self._compute_delta(x, self.flylora_A_id, self.flylora_B_id, self.flylora_d_id, 'id')
            return base_out + delta.to(base_out.dtype)

        # fused mode: combine two LoRA branches at layer output end
        delta_text = self._compute_delta(x, self.flylora_A_text, self.flylora_B_text, self.flylora_d_text, 'text')
        delta_id = self._compute_delta(x, self.flylora_A_id, self.flylora_B_id, self.flylora_d_id, 'id')
        mix = max(0.0, min(1.0, self.output_mix))
        delta = mix * delta_text + (1.0 - mix) * delta_id
        return base_out + delta.to(base_out.dtype)


def replace_with_dual_flylora(
    model: nn.Module,
    target_module_names,
    r: int,
    k: int,
    alpha: float,
    sparsity_ratio: float,
    bias_lr: float,
    output_mix: float,
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
            DualFlyLoRALinear(
                module,
                r=r,
                k=k,
                alpha=alpha,
                sparsity_ratio=sparsity_ratio,
                bias_lr=bias_lr,
                output_mix=output_mix,
            ),
        )
        replaced += 1
    return replaced


def set_dual_flylora_task(model: nn.Module, task: str):
    for module in model.modules():
        if isinstance(module, DualFlyLoRALinear):
            module.set_task(task)


def set_dual_flylora_output_mix(model: nn.Module, output_mix: float):
    for module in model.modules():
        if isinstance(module, DualFlyLoRALinear):
            module.set_output_mix(output_mix)
