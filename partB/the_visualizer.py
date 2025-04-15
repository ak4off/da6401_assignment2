# partB/the_visualizer.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid
import os
import wandb

# def plot_confusion_matrix(y_true, y_pred, the_classes, normalize=True, save_path="confusion_matrix.png"):
# zstart debug
def plot_confusion_matrix(model, test_loader, the_classes, device, normalize=True, save_path="confusion_matrix.png", use_wandb=False):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
# ends
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=the_classes, yticklabels=the_classes,
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if use_wandb and wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.Image(plt)})
    else:
        plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')


def plot_sample_predictions(model, dataloader, the_classes, device, num_images=16,save_path="sample_predictions.png", use_wandb=False):
    model.eval()
    images_shown, true_labels, pred_labels = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            images_shown.extend(images.cpu())
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

            if len(images_shown) >= num_images:
                break

    images_shown = images_shown[:num_images]
    true_labels = true_labels[:num_images]
    pred_labels = pred_labels[:num_images]

    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        img = images_shown[i].permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        plt.subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)), i + 1)
        plt.imshow(img)
        plt.title(f"T: {the_classes[true_labels[i]]}\nP: {the_classes[pred_labels[i]]}", fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    if use_wandb and wandb.run is not None:
        wandb.log({"sample_predictions": wandb.Image(plt)})
    else:
        plt.show()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')    

'''
### ✅ Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        loss = output[0, class_idx]
        loss.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])  # [C]
        activations = self.activations[0]  # [C, H, W]

        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= heatmap.max()

        return heatmap.numpy()


def generate_gradcam(model, input_tensor, target_layer, class_idx=None, save_path="gradcam.png"):
# def generate_gradcam(model, input_tensor, target_layer, class_idx=None, save_path=None):
    model.eval()
    gradcam = GradCAM(model, target_layer)
    heatmap = gradcam.generate(input_tensor, class_idx)
    gradcam.remove_hooks()

    # Prepare input image for overlay
    img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Grad-CAM")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()



# ✅ This goes outside the class — define it once globally
def replace_relu_with_non_inplace(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU) and module.inplace:
            setattr(model, name, torch.nn.ReLU(inplace=False))
        else:
            replace_relu_with_non_inplace(module)

### ✅ Guided Backprop
class GuidedBackprop:
    def __init__(self, model):
        self.model = model.eval()
        replace_relu_with_non_inplace(self.model)  # 💡 Safely replaces all inplace ReLUs
        self.hook_handles = []
        self._register_hooks()
        self.model.to(next(self.model.parameters()).device)

    def _register_hooks(self):
        def relu_hook(module, grad_input, grad_output):
            return (torch.clamp(grad_input[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                self.hook_handles.append(module.register_full_backward_hook(relu_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        loss = output[0, class_idx]
        loss.backward()

        guided_gradients = input_tensor.grad[0].cpu().numpy()
        return guided_gradients

# ✅ Visualization wrapper
def generate_guided_backprop(model, input_tensor, class_idx=None, save_path="guided_backprop.png"):
    model.eval()
    guided_bp = GuidedBackprop(model)
    guided_grads = guided_bp.generate(input_tensor, class_idx)
    guided_bp.remove_hooks()

    # Normalize gradients to [0, 1]
    grad_img = guided_grads.transpose(1, 2, 0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min() + 1e-8)

    plt.imshow(grad_img)
    plt.title("Guided Backpropagation")
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

'''