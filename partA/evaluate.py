### evaluate.py

import torch
from model_cnn import CNN
from data_loader import get_data_loaders
import matplotlib.pyplot as plt
import numpy as np


def evaluate_best_model(model_path, data_dir, batch_size=64, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_data_loaders(data_dir, batch_size)

    model = CNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return all_preds, all_labels

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    evaluate_best_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_classes=args.num_classes
    )

