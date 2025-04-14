# partB/the_trainer.py

import argparse
import os
import torch
import wandb
import datetime
import time

from the_data_loader import get_dataloaders
from models.resnet_finetune import load_resnet50
from the_evaluator import Evaluator
from the_utils import (
    set_seed, 
    get_optimizer, 
    get_scheduler, 
    get_loss_function, 
    save_model,
    print_model_summary
)
from the_visualize import plot_confusion_matrix, generate_gradcam, plot_sample_predictions, generate_guided_backprop

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 on iNaturalist subset")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/iNaturalist", help="Dataset directory")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")

    # Model
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50","googlenet"],)
    parser.add_argument("--freeze_option", type=int, default=1, choices=[0, 1, 2],
                        help="0: Only FC, 1: FC + last block, 2: full fine-tune")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--dense_size", type=int, default=512)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable LR scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=3, help="LR scheduler patience")
    # Logging & saving
    parser.add_argument("--save_model_path", type=str, default="./checkpoints/best_model.pt")
    parser.add_argument("--seed", type=int, default=42)

    # WandB
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="finetune-inat")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="finetune_run")

    return parser.parse_args()


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.use_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run_name, config=vars(args))

    # Load Data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augment=args.data_aug
    )

    # Load Model
    model = load_resnet50(
        num_classes=num_classes,
        dropout=args.dropout,
        dense_size=args.dense_size,
        freeze_option=args.freeze_option
    )
    model.to(device)
    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=100)

    print_model_summary(model)
    # added early stopping
    early_stop_counter = 0
    # Loss, Optimizer, Scheduler
    criterion = get_loss_function("cross_entropy")
    optimizer = get_optimizer(model, args.lr, args.weight_decay)
    scheduler = get_scheduler(optimizer)

    best_val_acc = 0.0
    start_time = time.time()
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if i % args.log_interval == 0:
                print(f"[Epoch {epoch+1} | Batch {i}] Loss: {loss.item():.4f}")

        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        evaluator = Evaluator(model, val_loader, criterion, device)
        val_loss, val_acc = evaluator.evaluate()

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # scheduler.step(val_loss)
        if args.use_scheduler:
            scheduler.step(val_acc)

            
        # scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "learning_rate": current_lr
            })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, args.save_model_path)
            print(f"Saved new best model with val acc: {val_acc:.2f}%")
            if args.use_wandb and wandb.run is not None:
                wandb.save("best_model.pth")
            early_stop_counter = 0  # reset counter
        else:
            early_stop_counter += 1
            if args.early_stopping_patience and early_stop_counter >= args.early_stopping_patience:
                # print(f"Early stopping triggered at epoch {epoch+1}")
                break

    elapsed_time = time.time() - start_time
    # print(f"\nTotal training time: {int(elapsed_time // 60)} min {int(elapsed_time % 60)} sec")
    if args.use_wandb:
        wandb.log({
            "total_training_time": elapsed_time,
        })
    # Final Test Evaluation
    evaluator = Evaluator(model, test_loader, criterion, device)
    test_loss, test_acc = evaluator.evaluate()
    print(f"Test Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")

    # After final test evaluation
    # Visualize confusion matrix
    class_names = test_loader.dataset.classes  # if dataset is an ImageFolder
    img_name = f"confusio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(model, test_loader, class_names, device,save_path=img_name, use_wandb=args.use_wandb)
    # plot_confusion_matrix(model, test_loader, class_names, device)
    img_name = f"pred_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_confusion_matrix(model, test_loader, class_names, device,save_path=img_name, use_wandb=args.use_wandb)
    # plot_sample_predictions(model, test_loader, class_names, device)

    if args.use_wandb:
        wandb.log({"test_acc": test_acc, "test_loss": test_loss})
        wandb.finish()



if __name__ == "__main__":
    args = parse_args()
    train(args)
