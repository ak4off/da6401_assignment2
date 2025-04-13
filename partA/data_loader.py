# # data_loader.py

import os
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch

def get_data_loaders(data_dir, batch_size, val_split=0.2, use_data_augmentation=False):
    if use_data_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    test_val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_val_transform)

    targets = np.array(full_train_dataset.targets)
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    for train_idx, val_idx in strat_split.split(np.zeros(len(targets)), targets):
        train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_train_dataset, val_idx)

    # Apply test/val transforms to val set as well
    val_dataset.dataset.transform = test_val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader