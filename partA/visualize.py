### visualize.py

import torch
import matplotlib.pyplot as plt
from model_cnn import CNN
from torchvision.utils import make_grid
import numpy as np


def visualize_filters(model_path):
    model = CNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    first_conv_layer = model.conv[0]  # First conv layer
    filters = first_conv_layer.weight.data.clone()
    
    # Normalize for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min())
    grid = make_grid(filters, nrow=8, normalize=True, padding=1)
    plt.figure(figsize=(10, 10))
    plt.title("First Conv Layer Filters")
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
