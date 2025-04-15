import torch.nn as nn
from torchvision import models


def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_last_block(model):
    """
    For GoogLeNet, we'll define "last block" as the final Inception layer (inception5)
    and the classifier (fc). This is a bit coarser than ResNet's layer4.
    """
    for name, param in model.named_parameters():
        if "inception5" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def load_googlenet(num_classes, dropout=0.2, dense_size=512, freeze_option=1):
    """
    Load a GoogLeNet model with custom classifier and optional freezing.

    Args:
        num_classes (int): Number of target classes.
        dropout (float): Dropout probability for final FC.
        dense_size (int): Size of intermediate FC layer.
        freeze_option (int): 0 = freeze all except FC,
                             1 = freeze all except FC + last block (inception5),
                             2 = no freezing (full finetune)

    Returns:
        model (nn.Module): Fine-tunable GoogLeNet
    """
    model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=True)

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
