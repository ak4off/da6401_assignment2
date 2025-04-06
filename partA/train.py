import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from model_cnn import CNN
from data_loader import get_data_loaders
import time 

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

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb early and override args if needed
    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        config = wandb.config
        # Overwrite args with wandb config values
        args.lr = config.lr
        args.batch_size = config.batch_size
        args.dropout = config.dropout
        args.activation = config.activation
        args.optimizer = config.optimizer
        args.num_filters = config.num_filters
        wandb.watch_called = False  # Avoid multiple watch calls
    else:
        config = args  # fallback to argparse values

    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)

    # Model
    model = CNN(
        num_classes=args.num_classes,
        num_filters=args.num_filters,
        filter_sizes=args.filter_sizes,
        activation=args.activation,
        dense_neurons=args.dense_neurons,
        dropout_rate=args.dropout,
        batch_norm=args.batch_norm
    ).to(device)

    # Optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError("Unsupported optimizer")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        wandb.watch(model)

    print("Total trainable parameters:", model.compute_parameters())
    print("Estimated total FLOPs:", model.compute_flops())

    start_time = time.time()
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Saved new best model at epoch {epoch+1} with val acc: {val_acc:.4f}")
            if args.use_wandb:
                wandb.save("best_model.pth")

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | LR: {current_lr:.6f}")

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
    print(f"\n�� Total training time: {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    if args.use_wandb:
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_filters", nargs='+', type=int, default=[32, 64, 128, 128, 256])
    parser.add_argument("--filter_sizes", nargs='+', type=int, default=[3, 3, 3, 3, 3])
    parser.add_argument("--dense_neurons", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_norm", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="cnn_from_scratch_project")

    args = parser.parse_args()
    main(args)
