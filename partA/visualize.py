import os
import matplotlib.pyplot as plt 
import wandb
from torchvision.utils import make_grid
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import datetime
import numpy as np


from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import wandb
import random

'''
def plot_test_predictions_grid(model, test_loader, the_classes, device, save_path="sastest_predictions_grid.png", use_wandb=False):
    model.eval()
    max_per_class = 2  # number of images per class
    class_to_samples = defaultdict(list)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                img_cpu = images[i].cpu()
                true_label = labels[i].item()
                pred_label = preds[i].item()

                if len(class_to_samples[true_label]) < max_per_class:
                    sample = (img_cpu, true_label, pred_label)
                    class_to_samples[true_label].append(sample)

            if all(len(v) >= max_per_class for v in class_to_samples.values()):
                break

    # Flatten collected samples
    final_samples = []
    for class_id in sorted(class_to_samples.keys()):
        final_samples.extend(class_to_samples[class_id])

    # Shuffle to mix correct & incorrect
    random.shuffle(final_samples)

    # Plotting
    n_cols = 5
    n_rows = (len(final_samples) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten()

    for idx, (img_tensor, true_label, pred_label) in enumerate(final_samples):
        ax = axes[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis("off")

        correct = (true_label == pred_label)
        color = "green" if correct else "red"
        icon = "✓" if correct else "✗"
        ax.set_title(f"{icon} Pred: {the_classes[pred_label]}\nTrue: {the_classes[true_label]}", fontsize=9, color=color)

    # Hide extra axes
    for i in range(len(final_samples), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if use_wandb and wandb.run is not None:
        wandb.log({"test_predictions_grid": wandb.Image(fig)})
    else:
        plt.savefig(save_path)
        print(f"Test prediction grid saved to: {save_path}")
    plt.close(fig)

'''
def visualize_filters(model, save_path="conv_filters.png", use_wandb=False):
    # Extract filters from the first conv layer
    first_conv_layer = model.conv[0]
    filters = first_conv_layer.weight.data.clone().cpu()

    # Normalize filters to [0, 1] for visualization
    filters = (filters - filters.min()) / (filters.max() - filters.min() + 1e-5)

    # Create a grid of filter images
    grid = make_grid(filters, nrow=8, padding=1)
    plt.figure(figsize=(10, 10))
    plt.title("First Conv Layer Filters")
    plt.imshow(grid.permute(1, 2, 0))  # C, H, W -> H, W, C
    plt.axis('off')
    fig, ax = plt.subplots(figsize=(10, 8))
    full_path = os.path.join(os.getcwd(), save_path)
    # plt.savefig(full_path)
    # print(f"Filter visualization saved to: {full_path}")
    # if use_wandb:
    if use_wandb and wandb.run is not None:
        #wandb.log({"conv_filters": wandb.Image(plt.gcf())})
	wand.log({"conv_filters": wandb.Image(fig)})
    else:
        plt.savefig(save_path)

    plt.close()


# extraaa
def plot_misclassified_grid(model, test_loader, the_classes, device, save_path="misclassified_grid.png", use_wandb=False):

    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((images[i].cpu(), preds[i].cpu(), labels[i].cpu()))
                if len(misclassified) >= 30:  # Limit to 10x3 grid
                    break
            if len(misclassified) >= 30:
                break

    fig, axes = plt.subplots(10, 3, figsize=(12, 28))
    axes = axes.flatten()

    for i, (img, pred, true) in enumerate(misclassified):
        ax = axes[i]
        img = img.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Pred: {the_classes[pred]}\nTrue: {the_classes[true]}", fontsize=9)

    plt.tight_layout()
    # plt.savefig(save_path)
    # print(f"Misclassified prediction grid saved to: {save_path}")
    # if use_wandb:
    if use_wandb and wandb.run is not None:
        wandb.log({"misclassified_grid": wandb.Image(fig)})
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred, the_classes, use_wandb=False):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=the_classes, yticklabels=the_classes, ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    # Save the figure with a timestamp
    img_name = f"confusion_matrix_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_path = os.path.join(os.getcwd(), img_name)
    # plt.savefig(save_path)
    # print(f"Confusion matrix saved to: {save_path}")

    # if use_wandb:
    if use_wandb and wandb.run is not None:
        wandb.log({"confusion_matrix": wandb.Image(fig)})
    else:
        plt.show()
    plt.close(fig)


def plot_test_predictions_grid(model, test_loader, the_classes, device, save_path="classwise_test_grid.png", use_wandb=False):
    import matplotlib.pyplot as plt
    import torch
    import wandb
    import random
    from collections import defaultdict

    model.eval()
    samples_per_class = defaultdict(list)
    max_per_class = 3
    num_classes_to_show = 10  # limit to first 10 classes

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                img_cpu = images[i].cpu()
                true_label = labels[i].item()
                pred_label = preds[i].item()

                if len(samples_per_class[true_label]) < max_per_class:
                    samples_per_class[true_label].append((img_cpu, true_label, pred_label))

            # Early exit if we already have 3 samples for 10 classes
            if len([c for c in samples_per_class if len(samples_per_class[c]) == max_per_class]) >= num_classes_to_show:
                break

    # Sort and select 10 classes with available samples
    selected_classes = sorted(samples_per_class.keys())[:num_classes_to_show]
    final_samples = []

    for cls in selected_classes:
        class_samples = samples_per_class[cls][:max_per_class]
        # Fill up missing if less than 3 samples
        while len(class_samples) < 3:
            class_samples.append((class_samples[0][0], cls, class_samples[0][2]))  # duplicate if needed
        final_samples.extend(class_samples)

    fig, axes = plt.subplots(10, 3, figsize=(13, 30))
    axes = axes.flatten()

    for idx, (img_tensor, true_label, pred_label) in enumerate(final_samples):
        ax = axes[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img)
        ax.axis("off")

        correct = (true_label == pred_label)
        color = "green" if correct else "red"
        status_icon = "✓" if correct else "✗"
        ax.set_title(f"{status_icon} Pred: {the_classes[pred_label]}\nTrue: {the_classes[true_label]}", fontsize=9, color=color)

    plt.tight_layout()
    if use_wandb and wandb.run is not None:
        wandb.log({"test_predictions_grid": wandb.Image(fig)})
    else:
        plt.savefig(save_path)
        print(f"Test prediction grid saved to: {save_path}")
    plt.close(fig)

'''
def plot_test_predictions_grid(model, test_loader, the_classes, device, save_path="sastest_predictions_grid.png", use_wandb=False):
    import matplotlib.pyplot as plt
    import torch
    import wandb

    model.eval()
    max_images = 30
    correct_samples = []
    incorrect_samples = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                img_cpu = images[i].cpu()
                true_label = labels[i].item()
                pred_label = preds[i].item()
                sample = (img_cpu, true_label, pred_label)

                if true_label == pred_label:
                    correct_samples.append(sample)
                else:
                    incorrect_samples.append(sample)

                if len(correct_samples) + len(incorrect_samples) >= 3 * max_images:
                    break
            if len(correct_samples) + len(incorrect_samples) >= 3 * max_images:
                break

    # Balance: 20 correct, 10 incorrect (if available)
    num_correct = min(20, len(correct_samples))
    num_incorrect = max_images - num_correct

    selected_correct = correct_samples[:num_correct]
    selected_incorrect = incorrect_samples[:num_incorrect]

    final_samples = selected_correct + selected_incorrect
    # Optional: shuffle so wrong predictions aren't all at the end
    import random
    random.shuffle(final_samples)

    fig, axes = plt.subplots(10, 3, figsize=(13, 30))
    axes = axes.flatten()

    for idx, (img_tensor, true_label, pred_label) in enumerate(final_samples):
        ax = axes[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())

        ax.imshow(img)
        ax.axis("off")

        correct = (true_label == pred_label)
        color = "green" if correct else "red"
        status_icon = "✓" if correct else "✗"
        ax.set_title(f"{status_icon} Pred: {the_classes[pred_label]}\nTrue: {the_classes[true_label]}", fontsize=9, color=color)

    for j in range(len(final_samples), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if use_wandb and wandb.run is not None:
        wandb.log({"test_predictions_grid": wandb.Image(fig)})
    else:
        plt.savefig(save_path)
        print(f"Test prediction grid saved to: {save_path}")
    plt.close(fig)
'''
'''
def plot_test_predictions_grid(model, test_loader, the_classes, device, save_path="test_predictions_grid.png", use_wandb=False):
    model.eval()
    images_shown = 0
    max_images = 30  # 10 rows × 3 cols

    fig, axes = plt.subplots(10, 3, figsize=(12, 28))
    axes = axes.flatten()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                if images_shown >= max_images:
                    break
                ax = axes[images_shown]
                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())  # normalize

                ax.imshow(img)
                ax.axis('off')
                true_label = the_classes[labels[i]]
                pred_label = the_classes[preds[i]]
                ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", fontsize=9)
                images_shown += 1

            if images_shown >= max_images:
                break

    plt.tight_layout()
    # plt.savefig(save_path)
    # print(f"Test prediction grid saved to: {save_path}")
    # if use_wandb:
    if use_wandb and wandb.run is not None:
        wandb.log({"test_predictions_grid": wandb.Image(fig)})
    plt.close(fig)'''
