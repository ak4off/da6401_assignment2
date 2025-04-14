# resnet friendly
# partB/the_data_loader.py

import os
import random
from typing import Tuple, Dict
from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_transforms(image_size: int = 224, augment: bool = True) -> Dict[str, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]) if augment else transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return {
        'train': train_transform,
        'val': val_test_transform,
        'test': val_test_transform
    }


def stratified_split(dataset: datasets.ImageFolder, val_ratio: float = 0.2, seed: int = 42) -> Tuple[Subset, Subset]:
    targets = dataset.targets
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(dataset.samples, targets))
    return Subset(dataset, train_idx), Subset(dataset, val_idx)

def get_dataloaders(data_dir: str,
                    image_size: int = 224,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    augment: bool = True,
                    seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    torch.manual_seed(seed)
    random.seed(seed)

    # Get transforms
    transform_dict = get_transforms(image_size=image_size, augment=augment)

    # Load full train set
    # full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_dict['train'])
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_dict['train'])

    train_dataset, val_dataset = stratified_split(full_train_dataset, val_ratio=0.2, seed=seed)

    # Override transforms for val
    val_dataset.dataset.transform = transform_dict['val']

    # Load test set from val/ as per convention
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_dict['test'])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # return train_loader, val_loader, test_loader
    num_classes = len(full_train_dataset.classes)  # Get the number of classes

    return train_loader, val_loader, test_loader, num_classes