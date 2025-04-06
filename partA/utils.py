### ✅ `utils.py`

import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean().item()
