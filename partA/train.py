import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from model_cnn import CNN
from data_loader import get_data_loaders

from visualize import plot_misclassified_grid, plot_confusion_matrix, plot_test_predictions_grid, visualize_filters
from evaluate import evaluate

import time
import datetime


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        config = wandb.config
        args.lr = config.lr
        args.batch_size = config.batch_size
        args.dropout = config.dropout
        args.activation = config.activation
        args.optimizer = config.optimizer
        args.num_filters = config.num_filters
        args.dense_neurons = config.dense_neurons
        args.batch_norm = config.batch_norm
        wandb.watch_called = False
    else:
        config = args

    # train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    # optional data augmentation
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size, use_data_augmentation=args.use_data_augmentation)

    num_classes = len(train_loader.dataset.dataset.classes)

    model = CNN(
        num_classes=num_classes,
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        activation=args.activation,
        dense_neurons=args.dense_neurons,
        dropout_rate=args.dropout,
        batch_norm=args.batch_norm
    ).to(device)

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # added scheduler
    lr_scheduler = None
    if args.use_scheduler:
        # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
        #                                                 patience=args.scheduler_patience, verbose=True)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5,
            patience=args.scheduler_patience, verbose=True)
    # added early stopping
    early_stop_counter = 0

    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=100)
        # wandb.watch(model)

    total_trainable_params = model.compute_parameters()
    estimated_total_flops = model.compute_flops()
    # print("Total trainable parameters:", model.compute_parameters())
    # print("Estimated total FLOPs:", model.compute_flops())

    if args.use_wandb:
        wandb.log({"total_trainable_params": total_trainable_params, "estimated_total_FLOPs": estimated_total_flops})
        # wandb.finish()

    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            # print(f"Saved best model at epoch {epoch+1} with val acc: {val_acc:.4f}")
            # if args.use_wandb:
            if args.use_wandb and wandb.run is not None:
                wandb.save("best_model.pth")
            early_stop_counter = 0  # reset counter
        else:
            early_stop_counter += 1
            if args.early_stopping_patience and early_stop_counter >= args.early_stopping_patience:
                # print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # if lr_scheduler:
        #     lr_scheduler.step(val_loss)
        if args.use_scheduler:
            lr_scheduler.step(val_acc)

            
        # scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # print(f"[{epoch+1}/{args.epochs}] "
        #       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
        #       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr
            })

    elapsed_time = time.time() - start_time
    # print(f"\nTotal training time: {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec")
    if args.use_wandb:
        wandb.log({
            "total_training_time": elapsed_time,
        })
    # Evaluate normally
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    # print(f"Test Accuracy: {test_acc:.4f}")

    # Compute predictions for confusion matrix
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Plot confusion matrix
    the_classes = test_loader.dataset.classes  # or set manually if needed
    plot_confusion_matrix(y_true, y_pred, the_classes, use_wandb=args.use_wandb)
    plot_test_predictions_grid(model, test_loader, the_classes, device, save_path="test_predictions_grid.png", use_wandb=args.use_wandb)
    plot_misclassified_grid(model, test_loader, the_classes, device, save_path="misclassified_grid.png", use_wandb=args.use_wandb)

    # Log to wandb
    if args.use_wandb:
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        wandb.finish()
    # if args.visualize_filters:
    # print("Visualizing first convolutional filters...")
    # visualize_filters(model, save_path="conv_filters.png")
    img_name = f"filters_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    visualize_filters(model, save_path=img_name, use_wandb=args.use_wandb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh", "leaky_relu","mish", "gelu", "silu"], default="relu")
    parser.add_argument("--num_filters", nargs='+', type=int, default=[32, 64, 128, 128, 256])
    parser.add_argument("--filter_sizes", nargs='+', type=int, default=[3, 5, 5, 7, 7])
    parser.add_argument("--dense_neurons", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cnn_from_scratch_project")
    
    # parser.add_argument("--visualize_filters", action="store_true", help="Visualize first conv layer filters after training")

    parser.add_argument("--use_data_augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable LR lr_scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="lr_scheduler patience")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")

    args = parser.parse_args()
    main(args)
