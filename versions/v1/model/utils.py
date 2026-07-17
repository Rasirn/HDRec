import torch
import logging
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from accelerate.logging import MultiProcessAdapter
from collections import defaultdict

MAX_VAL = 1e4

def get_logger(log_file, name=None, filemode='w', level=logging.INFO, to_console=True):
    """Initialize and return a logger that automatically adds a timestamp to each log message.

    Args:
        log_file (str): Path to the log file.
        name (str, optional): Logger name. Defaults to None.
        filemode (str, optional): Mode to open the log file. Defaults to 'w'.
        level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
        to_console (bool, optional): Whether to add a console handler. Defaults to True.

    Returns:
        logging.Logger: Configured logger instance.
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file, mode=filemode)
    file_handler.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return MultiProcessAdapter(logger, {})

def create_optimizer_and_scheduler(model: nn.Module, num_train_optimization_steps, args):
    """Create an optimizer and a learning rate scheduler for the model, with unified learning rate."""
    
    all_param_optimizer = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            all_param_optimizer.append(param)
    optimizer_grouped_parameters = [
        {'params': all_param_optimizer, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
    ]

    optimizer = AdamW(optimizer_grouped_parameters)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps)

    return optimizer, scheduler

class AverageMeter:
    """Track and compute average, sum, and count statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all stored values to their initial state."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the meter with a new value and count, and recalculate the average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

    def __repr__(self):
        """Provide a string representation of the current value and average."""
        return f"Value: {self.val}, Average: {self.avg}"

class AverageMeterSet:
    """A collection of AverageMeters to track multiple metrics."""
    
    def __init__(self):
        # Use defaultdict to automatically handle missing meters
        self.meters = defaultdict(AverageMeter)

    def __getitem__(self, key):
        """Retrieve the meter for a specific key."""
        return self.meters[key]

    def update(self, name, value, n=1):
        """Update a specific meter by its name."""
        self.meters[name].update(value, n)

    def reset(self):
        """Reset all meters in the set."""
        for meter in self.meters.values():
            meter.reset()

    def values(self):
        """Return current values for all meters."""
        return {name: meter.val for name, meter in self.meters.items()}

    def averages(self):
        """Return average values for all meters."""
        return {name: meter.avg for name, meter in self.meters.items()}

    def sums(self):
        """Return sum values for all meters."""
        return {name: meter.sum for name, meter in self.meters.items()}

    def counts(self):
        """Return count values for all meters."""
        return {name: meter.count for name, meter in self.meters.items()}

class Ranker(nn.Module):
    """A neural ranking model that computes various ranking metrics."""

    def __init__(self, metrics_ks):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()

    def forward(self, scores, labels, userID):
        """Compute the loss and various ranking metrics."""
        try:
            loss = self.ce(scores, labels).item()
        except Exception as e:
            print(f"Error computing loss: {e}")
            loss = 0.0
        
        res_users = []
        _, topk_indices = torch.topk(scores, k=10, dim=-1)  # [batch_size, 10]
        for i in range(scores.size(0)):
            # check if the true label is in the top-k predictions
            if labels[i].item() in topk_indices[i]:
                res_users.append(userID[i].item())  # record user IDs that meet the criteria

        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1)
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()

        results = []
        for k in self.ks:
            indicator = (rank < k).float()
            results.append(((1 / torch.log2(rank + 2)) * indicator).mean().item())  # ndcg@k
            results.append(indicator.mean().item())  # hr@k
        results.append((1 / (rank + 1)).mean().item())  # MRR
        # results.append((1 - (rank / valid_length)).mean().item())  # AUC

        return results + [loss], res_users
