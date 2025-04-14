# the_evaluator.py

import torch
import torch.nn.functional as F
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, test_loader, criterion, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating on Test Set"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * total_correct / total_samples
        return avg_loss, accuracy
