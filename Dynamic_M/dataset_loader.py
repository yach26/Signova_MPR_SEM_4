"""
dataset_loader.py - Dataset loading and augmentation for sign language sequences.

Loads .npy files from the dataset/ folder, applies optional augmentation,
and returns PyTorch DataLoaders for training and validation.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional


# ─── Label Management ─────────────────────────────────────────────────────────

def load_labels(dataset_path: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Discover class labels from dataset folder structure and export to JSON.

    Args:
        dataset_path: Path to the dataset root directory.

    Returns:
        classes: Sorted list of class names.
        label_map: Dictionary mapping class name → integer index.
    """
    classes = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    label_map = {cls: idx for idx, cls in enumerate(classes)}

    # Export labels to JSON for use in inference
    labels_file = os.path.join(dataset_path, "..", "labels.json")
    with open(labels_file, "w") as f:
        json.dump({"classes": classes, "label_map": label_map}, f, indent=2)
    print(f"[Labels] Saved to {labels_file}: {label_map}")

    return classes, label_map


# ─── Data Augmentation ────────────────────────────────────────────────────────

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply random augmentations to a keypoint sequence for better generalization.

    Augmentations:
        - Gaussian noise: Simulates sensor noise.
        - Random time warp: Slight speed variation.
        - Random scaling: Hand size variation.
        - Random horizontal flip: Mirror view.

    Args:
        sequence: np.ndarray of shape (seq_len, 63).

    Returns:
        Augmented sequence of shape (seq_len, 63).
    """
    seq = sequence.copy()

    # 1. Add small Gaussian noise
    if np.random.random() < 0.5:
        noise = np.random.normal(0, 0.01, seq.shape).astype(np.float32)
        seq += noise

    # 2. Random scaling (±10%)
    if np.random.random() < 0.5:
        scale = np.random.uniform(0.9, 1.1)
        seq *= scale

    # 3. Random horizontal flip (negate x-coordinates: indices 0, 3, 6, ...)
    if np.random.random() < 0.3:
        seq = seq.reshape(seq.shape[0], 21, 3)
        seq[:, :, 0] = -seq[:, :, 0]  # Flip x
        seq = seq.reshape(seq.shape[0], 63)

    # 4. Random temporal shift (circular roll ±2 frames)
    if np.random.random() < 0.3:
        shift = np.random.randint(-2, 3)
        seq = np.roll(seq, shift, axis=0)

    # Clip to valid range
    seq = np.clip(seq, -2.0, 2.0)
    return seq


# ─── Dataset Class ────────────────────────────────────────────────────────────

class SignLanguageDataset(Dataset):
    """
    PyTorch Dataset for sign language gesture sequences.

    Args:
        sequences: List of np.ndarray sequences, each of shape (seq_len, 63).
        labels: List of integer labels.
        augment: Whether to apply data augmentation.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        augment: bool = False,
    ):
        self.sequences = sequences
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()

        # Apply augmentation during training
        if self.augment:
            seq = augment_sequence(seq)

        # Convert to tensors
        x = torch.tensor(seq, dtype=torch.float32)       # (seq_len, 63)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_dataset(
    dataset_path: str,
    sequence_length: int = 30,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Load all .npy sequences from the dataset directory.

    Args:
        dataset_path: Path to dataset root.
        sequence_length: Expected sequence length (for validation).

    Returns:
        sequences: List of arrays, each (sequence_length, 63).
        labels: List of integer class labels.
        classes: List of class names.
    """
    classes, label_map = load_labels(dataset_path)
    sequences, labels = [], []
    skipped = 0

    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        npy_files = sorted([f for f in os.listdir(cls_path) if f.endswith(".npy")])

        for npy_file in npy_files:
            file_path = os.path.join(cls_path, npy_file)
            try:
                seq = np.load(file_path).astype(np.float32)

                # Validate shape
                if seq.shape == (sequence_length, 63):
                    sequences.append(seq)
                    labels.append(label_map[cls])
                elif seq.shape[1] == 63 and seq.shape[0] != sequence_length:
                    # Pad or truncate to sequence_length
                    seq = _pad_or_truncate(seq, sequence_length)
                    sequences.append(seq)
                    labels.append(label_map[cls])
                else:
                    print(f"  [SKIP] Unexpected shape {seq.shape} in {file_path}")
                    skipped += 1

            except Exception as e:
                print(f"  [ERROR] Loading {file_path}: {e}")
                skipped += 1

        print(f"  [Loaded] {cls}: {len(npy_files) - skipped} sequences")

    print(f"\n[Dataset] Total: {len(sequences)} sequences | Skipped: {skipped}")
    print(f"[Dataset] Classes: {classes}")
    return sequences, labels, classes


def _pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or truncate sequence to target length."""
    current_len = seq.shape[0]
    if current_len >= target_len:
        return seq[:target_len]
    else:
        pad = np.zeros((target_len - current_len, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])


# ─── DataLoader Factory ───────────────────────────────────────────────────────

def get_dataloaders(
    dataset_path: str,
    sequence_length: int = 30,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train and validation DataLoaders.

    Args:
        dataset_path: Path to dataset root directory.
        sequence_length: Frames per sequence.
        batch_size: Mini-batch size.
        val_split: Fraction of data for validation.
        num_workers: DataLoader worker processes.
        seed: Random seed for reproducibility.

    Returns:
        train_loader, val_loader, classes
    """
    sequences, labels, classes = load_dataset(dataset_path, sequence_length)

    # Stratified split to preserve class balance
    X_train, X_val, y_train, y_val = train_test_split(
        sequences,
        labels,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    train_dataset = SignLanguageDataset(X_train, y_train, augment=True)
    val_dataset = SignLanguageDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\n[DataLoader] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"[DataLoader] Batch size: {batch_size} | Workers: {num_workers}")

    return train_loader, val_loader, classes


if __name__ == "__main__":
    # Quick test
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "../dataset"
    train_loader, val_loader, classes = get_dataloaders(dataset_path, batch_size=16)
    x, y = next(iter(train_loader))
    print(f"\n[Test] Batch x: {x.shape} | y: {y.shape}")
