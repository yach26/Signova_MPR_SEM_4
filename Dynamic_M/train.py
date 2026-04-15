"""
train.py - Full training pipeline for sign language LSTM model.

Usage:
    python train.py [--dataset ../dataset] [--epochs 100] [--batch 32]

Features:
    - Training + validation loops
    - Early stopping with patience
    - Best model checkpoint saving
    - Confusion matrix + per-class accuracy
    - Training curve plots
"""

import os
import sys
import json
import argparse
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import get_model
from dataset_loader import get_dataloaders


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Sign Language LSTM")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output", type=str, default="models", help="Directory to save model/plots")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker count")
    return parser.parse_args()


# ─── Training Utilities ───────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Args:
        patience: Epochs to wait before stopping after last improvement.
        min_delta: Minimum change to qualify as improvement.
        path: File path to save the best checkpoint.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4, path: str = "best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop. Saves model if improved.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            print(f"    [OK] New best val_loss={val_loss:.4f} -> Saved to {self.path}")
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n[EarlyStopping] No improvement for {self.patience} epochs. "
                      f"Best epoch: {self.best_epoch}")
                return True
            return False


# ─── Epoch Functions ──────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Run one validation epoch. Returns (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str,
):
    """Save training loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0].plot(epochs, train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_losses, label="Val Loss", linewidth=2)
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in train_accs], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, [a * 100 for a in val_accs], label="Val Acc", linewidth=2)
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved to {save_path}")


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    classes: List[str],
    save_path: str,
):
    """Save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=classes, yticklabels=classes)
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=classes, yticklabels=classes)
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Confusion matrix saved to {save_path}")


# ─── Main Training Loop ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Resolve paths
    dataset_path = os.path.abspath(args.dataset)
    output_path = os.path.abspath(args.output)
    os.makedirs(output_path, exist_ok=True)

    model_path = os.path.join(output_path, "best_model.pth")
    curves_path = os.path.join(output_path, "training_curves.png")
    cm_path = os.path.join(output_path, "confusion_matrix.png")
    config_path = os.path.join(output_path, "training_config.json")

    print("\n" + "=" * 60)
    print("  Sign Language LSTM - Training")
    print("=" * 60)
    print(f"  Dataset : {dataset_path}")
    print(f"  Output  : {output_path}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  LR      : {args.lr}")
    print(f"  Patience: {args.patience}")
    print("=" * 60 + "\n")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, classes = get_dataloaders(
        dataset_path=dataset_path,
        batch_size=args.batch,
        val_split=args.val_split,
        num_workers=args.workers,
        seed=args.seed,
    )
    num_classes = len(classes)

    # ── Model ───────────────────────────────────────────────────────────────
    print()
    model = get_model(num_classes=num_classes, device=device)

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    early_stopper = EarlyStopping(patience=args.patience, path=model_path)

    # ── Save config ─────────────────────────────────────────────────────────
    config = {
        "classes": classes,
        "num_classes": num_classes,
        "input_size": 63,
        "hidden_size": 128,
        "num_layers": 2,
        "sequence_length": 30,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "device": str(device),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n[Train] Starting training for up to {args.epochs} epochs...\n")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        t_loss, t_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Validate
        v_loss, v_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)

        # LR scheduling
        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc

        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"Train: loss={t_loss:.4f} acc={t_acc*100:.1f}% | "
            f"Val: loss={v_loss:.4f} acc={v_acc*100:.1f}% | "
            f"LR={current_lr:.2e}"
        )

        # Early stopping check (also saves checkpoint)
        if early_stopper(v_loss, model, epoch):
            break

    total_time = time.time() - t0
    print(f"\n[Done] Training complete in {total_time:.1f}s | Best val acc: {best_val_acc*100:.2f}%")

    # ── Final evaluation with best model ──────────────────────────────────
    print("\n[Eval] Loading best model for final evaluation...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device)
    print(f"[Eval] Final validation accuracy: {final_acc*100:.2f}%")

    # Classification report
    print("\n[Report] Per-class metrics:")
    print(classification_report(final_labels, final_preds, target_names=classes, digits=3))

        # -- Save plots --------------------------------------------------------
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
    plot_confusion_matrix(final_labels, final_preds, classes, cm_path)

    print(f"\n[Output] All files saved to: {output_path}")
    print(f"  Model     : {model_path}")
    print(f"  Config    : {config_path}")
    print(f"  Curves    : {curves_path}")
    print(f"  Conf. mat : {cm_path}")


if __name__ == "__main__":
    main()
