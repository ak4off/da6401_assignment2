# partB/the_utils.py

import torch
import torch.nn as nn
import random
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Forces deterministic algorithms (may reduce speed)
    torch.backends.cudnn.benchmark = False     # Disables auto-tuner that could vary conv ops
def get_loss_function(loss_type="cross_entropy"):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss {loss_type} not supported")

def get_optimizer(model, lr, weight_decay):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return 100 * correct / labels.size(0)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📦 Total parameters: {total_params:,}")
    print(f"🧠 Trainable parameters: {trainable_params:,}")
