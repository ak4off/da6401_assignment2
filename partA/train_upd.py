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
    train_loader, val_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)

    model = CNN(
        num_classes=args.num_classes,
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
        wandb.watch(model)

    print("Total trainable parameters:", model.compute_parameters())
    print("Estimated total FLOPs:", model.compute_flops())
    start_time = time.time()        # to track time taken
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # print(f"[{epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        scheduler.step(val_loss)  # Step after val loss

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
    end_time = time.time()
    elapsed_time = end_time - start_time

    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    print(f"\n🎯 Total training time: {mins} min {secs} sec")

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
