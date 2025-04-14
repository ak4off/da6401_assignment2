# partB/models/resnet_finetune.py

import torch.nn as nn
from torchvision import models


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_last_block(model):
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def load_resnet50(num_classes, dropout=0.2, dense_size=512, freeze_option=1):
    """
    Load a ResNet50 model with custom classifier and optional freezing.

    Args:
        num_classes (int): Number of target classes.
        dropout (float): Dropout probability for final FC.
        dense_size (int): Size of intermediate FC layer.
        freeze_option (int): 0 = freeze all except FC,
                             1 = freeze all except FC + last block (layer4),
                             2 = no freezing (full finetune)

    Returns:
        model (nn.Module): Fine-tunable ResNet50
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Apply freeze strategy
    if freeze_option == 0:
        freeze_all(model)
    elif freeze_option == 1:
        unfreeze_last_block(model)
    elif freeze_option == 2:
        unfreeze_all(model)
    else:
        raise ValueError("Invalid freeze_option. Choose 0, 1, or 2.")

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, dense_size),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(dense_size, num_classes)
    )

    return model
